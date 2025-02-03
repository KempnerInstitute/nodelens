from warnings import warn
from typing import Optional
from contextlib import contextmanager
import torch
from torch import nn
from tqdm import tqdm

from alignment.utils import (check_iterable,
                             get_maximum_strides,
                             weighted_average,
                             get_device,
                             remove_by_idx,
                             set_net_mode,
                             get_unfold_params,
                             smart_pca,
                             alignment)
from alignment.models.layers import (LAYER_REGISTRY, 
                                     REGISTRY_REQUIREMENTS, 
                                     check_metaparameters)


class AttributeReference:
    """
    Simple class designed to be a reference to the parent class as an attribute.

    This is required for compatibility with using DDP for training pytorch modules,
    since we'll sometimes train DDP models and sometimes not, we want the code to
    work the same way. However, if you instantiate a DPP model from a network:

    net = AlignmentNetwork()
    ddp_net = DDP(net)

    Then the AlignmentNetwork methods will only be accessible in ddp_net.module.__. Therefore,
    if we have a system whereby the AlignmentNetwork methods can also be accessed in
    net.module.__, then the code can be the same regardless of whether we're using DDP or not.

    Usage
    -----
        net = AlignmentNetwork() (or any object instantiation)
        net.module = AttributeReference(net)
        ---or---
        class class_name:
            def __init__(self, *args, **kwargs):
                self.module = AttributeReference(self)
    """

    def __init__(self, parent):
        self.parent = parent

    def __getattr__(self, name):
        if hasattr(self.parent, name):
            return getattr(self.parent, name)
        else:
            raise AttributeError(f"parent object (instance of {type(self.parent)}) has no attribute '{name}'")


class AlignmentNetwork(nn.Module):
    """
    This is the base class for a neural network used for alignment-related experiments.

    The point of all the wrangling of standard torch workflows in this class is to make
    it easy to perform all the alignment-related computations for networks with different
    architectures without having to rewrite similar code over and over again. In this way,
    the user only needs to pass in a base model, and specify the layers to participate in 
    alignment computation along with their disired input layer. No need for registration 
    the model and layers manually.

    The forward method of **AlignmentNetwork** passes the input (*x*) through the base model
    forward method. If hidden activations are requested, then the output of each layer participated
    in the alignment computation is saved through setting up forward hooks. The alignment methods 
    are applied to the hidden activation at the output of corresponding input layer and the weight
    of layer participate in the experiment.

    Note: some shape wrangling (like that which happens between a convolutional layer and a
    linear layer are often treated as a nn.Module layer), but these don't require alignment-
    related processing.

    Note: to maintain the flexiblity of the code, user is resposible to make sure that the requested
    input layer is feasible for the requested alignment layer and has a compatible output shape for
    alignment computation on the target layer.

    An target alignment layer should have the following properties:
    1. Be a child of the nn.Module class with a forward method
    2. Have processing stage with weights for measuring alignment
    """

    def __init__(self, base_model: nn.Module, alignment_layer_names: Optional[dict] = None, **kwargs):
        super().__init__()  # register it as a nn.Module
        self.base_model = base_model
        self.alignment_layers = nn.ModuleList()  # a list of all the layers for the alignment computation
        self.alignment_names = []
        self.hidden = {}  # a parallel list to maintain the output/activation of the input layers aka inputs to the alignment layers
        # Maybe we can put this in the forward_hooks context manager so it's only created and used when needed? 
        # Basically I'm wondering if it's used at any point other than when calling forward with store_hidden=True?
        self.hooks = {} # a dictionary list to maintain the handle of the forward hooks on input layers
        self._initialize_layers(alignment_layer_names, **kwargs)  # initialize the architecture using child class method
        
        #if reference:
            # create reference to self in "model" attribute for compatibility with DDP
        #    self.module = AttributeReference(self)

    def _initialize_layers(self, alignment_layer_names, **kwargs):
        """
        This method initialize the layers participating the alignment computation and if alignment_layer_names is not None, their 
        corresponding input layers. after this method:
        self.alignment_layers will hold the list of layers participating in alignment computation
        self.alignment_names will hold the names of the self.alignment_layers with the same order
        and if alignment_layer_names is not None, self.layer_to_input_names will hold a copy of this map from
        alignment layer names to their input layer names.
        """
        if alignment_layer_names is None:
            self.layer_to_input_names = None
            for name, layer in self.base_model.named_modules():
                if not hasattr(layer, 'weight'): continue
                self.alignment_layers.append(layer)
                self.alignment_names.append(name)
        else:
            self.layer_to_input_names = {}
            for name, layer in self.base_model.named_modules():
                if name in alignment_layer_names.keys():
                    if not hasattr(layer, 'weight'):
                        warn_message = f"Skipping the selected layer {name} ({layer.__class__.__name__}) becuase it does not have a weight attribute"
                        warn(warn_message, RuntimeWarning, stacklevel=1)
                        continue 
                    self.alignment_layers.append(layer)
                    self.alignment_names.append(name)
                    # making the order the same as alignment_layers
                    self.layer_to_input_names[name] = alignment_layer_names[name]

    def is_classification_layer_included(self):
        """
        convenience method to check if the last layer is included in the alignment layers?
        This check is useful since we don't dropout classification layer
        # it gets the name of the last layer in network and check if it is in alignment layer names
        """
        classification_layer_name = [name for name, layer in self.base_model.named_modules() if hasattr(layer, 'weight')][-1]
        return classification_layer_name in self.alignment_names

    def num_layers(self, all=False):
        """
        convenience method for getting the number of layers in network
        if all=False (default), will get the number of alignment layers
        if all=True, will get total number of layers in network that has
        weight attribute and can be used for alignment computation
        """
        if all:
            return sum(1 for m in self.base_model.modules() if hasattr(m, 'weight'))
        return len(self.alignment_layers)
    
    @contextmanager
    def forward_hooks(self):
        try:
            self.setup_forward_hooks()
            yield
        finally:
            self.remove_forward_hooks()

    def setup_forward_hooks(self):
        def get_activation(name):
            def activation_hook(module, input, output):
                self.hidden[name] = output
            return activation_hook
        
        def get_input(name):
            def input_hook(module, input, output):
                self.hidden[name] = input[0]
            return input_hook
        
        if self.layer_to_input_names is None:
            for name, alignment_layer in zip(self.alignment_names, self.alignment_layers):
                self.hooks[name] = alignment_layer.register_forward_hook(get_input(name))
        else:
            for name, input_layer in self.base_model.named_modules():
                if name in self.layer_to_input_names.values():
                    self.hooks[name] = input_layer.register_forward_hook(get_activation(name))
                if name in self.layer_to_input_names.keys() and self.layer_to_input_names[name] is None:
                    self.hooks[name] = input_layer.register_forward_hook(get_input(name))

    def remove_forward_hooks(self):
        for _, hook in self.hooks.items():
            hook.remove()
    
    def forward(self, x, store_hidden=False):
        """
        standard forward pass of the base_model with option of storing 
        the activation/output of the corresponding input layers to the 
        alignment layers using forward hook
        """
        if store_hidden:
            # Use a context manager to ensure hooks are removed after forward pass
            with self.forward_hooks():
                self.hidden = {} # reset the stored activation
                out = self.base_model(x)
        else:
            out = self.base_model(x)
        return out

    def get_dropout(self):
        """
        Return list of dropout probability for any dropout layers in network
        """
        p = []
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                p.append(module.p)
        return p

    def set_dropout(self, p):
        """
        Set dropout of all layers in a network
        Note that this will overwrite whatever was previously used
        """
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.p = p

    def set_dropout_by_layer(self, p):
        """
        Set dropout of each layer in a network independently

        p must be an iterable indicating the probability of dropout for each layer
        """
        # get dropout layers (in order!)
        dropout_layers = []
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                dropout_layers.append(module)

        assert len(dropout_layers) == len(p), "p must contain the same number of elements as the number of dropout layers in the network"

        # assign each p to the dropout layer
        for layer, drop_prob in zip(dropout_layers, p):
            layer.p = drop_prob

    @torch.no_grad()
    def get_layer_inputs(self, x, precomputed=False):
        """method for getting list of layer inputs throughout the network"""
        if not precomputed:
            # do a forward pass and store hidden activations if not precomputed
            _ = self.forward(x, store_hidden=True)

        # extract activations by inputs to alignment layers out of slef.hidden
        layer_inputs = []
        for name in self.alignment_names:
            name_idx = name
            if self.layer_to_input_names is not None and self.layer_to_input_names[name] is not None:
                name_idx = self.layer_to_input_names[name]
            layer_inputs.append(self.hidden[name_idx])

        return layer_inputs

    @torch.no_grad()
    def get_alignment_layers(self):
        """convenience method for retrieving registered layers for alignment measurements throughout the network"""
        return self.alignment_layers

    @torch.no_grad()
    def get_alignment_weights(self, flatten=False):
        """
        convenience method for retrieving registered weights for alignment measurements throughout the network

        if flatten=True, will flatten weights so they have shape (nodes/channels, numel_per_weight)
        """
        # go through each layer and retrieve weight as desired
        weights = []
        for layer in self.alignment_layers:
            # get weight data for this layer
            weight = layer.weight.data.clone()

            # if requesting flat weights, flatten them
            if flatten:
                weight = weight.flatten(start_dim=1)

            # add weights to list
            weights.append(weight)

        
        return weights

    def _preprocess_inputs(self, inputs_to_layers, compress_convolutional=True):
        """
        helper method for processing inputs to layers as needed for certain alignment operations

        Operations by layer type
        ------------------------
        linear layer:
            will leave inputs to layers unchanged if input to a feedforward layer
        convolutional layer:
            if compress_convolutional=True, will unfold inputs to (batch * num_strides, conv_weight_dim)
            otherwise, will unfold inputs to (batch, conv_weight_dim, num_strides)
        """
        # initialize new list of inputs to layers
        preprocessed = []

        # do requested processing and add to output
        for input, layer in zip(inputs_to_layers, self.alignment_layers):
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                # if convolutional layer, unfold layer to (batch / conv_dim / num_strides)
                layer_prms = get_unfold_params(layer)
                unfolded_input = torch.nn.functional.unfold(input, layer.kernel_size, **layer_prms)
                if compress_convolutional:
                    unfolded_input = unfolded_input.transpose(1, 2).contiguous().view(-1, unfolded_input.size(1))
                preprocessed.append(unfolded_input)
            else:
                # if linear layer, no preprocessing should ever be required
                preprocessed.append(input)

        # return processed input data
        return preprocessed

    @torch.no_grad()
    def compare_weights(self, weights, norm=False):
        """
        compare network weights with **weights** (usually to measure change in weights)
        if norm=True, will measure norm of weight changes rather than structure
        """
        current_weights = self.get_alignment_weights()
        delta_weights = []
        for iw, cw in zip(weights, current_weights):
            if norm:
                delta_weights.append(torch.norm(cw.flatten(1) - iw.flatten(1), dim=1))
            else:
                delta_weights.append(cw - iw)
        return delta_weights

    @torch.no_grad()
    def measure_alignment(self, x, precomputed=False, method="alignment", relative=True):
        """
        measure alignment of the networks weights with the inputs to each layer from batch **x**
        """
        # Pre-layer activations start with input (x) and ignore output
        inputs_to_layers = self.get_layer_inputs(x, precomputed=precomputed)
        preprocessed = self._preprocess_inputs(inputs_to_layers, compress_convolutional=True)
        weights = self.get_alignment_weights(flatten=True)
        return [alignment(input, weight, method=method, relative=relative) for input, weight in zip(preprocessed, weights)]

    @torch.no_grad()
    def measure_alignment_weights(self, x, weights, precomputed=False, method="alignment", relative=True):
        """
        alternative for using predefined weights (usually for alignment of weight updates)

        standard usage
        --------------
        net = AlignmentNetwork() # of some sort, like MLP(), for example
        init_weights = net.get_alignment_weights()
        ... do some training ...
        delta_weights = net.compare_weights(init_weights)
        delta_alignment = net.measure_alignment_on_weights(x, delta_weights) # where x is a potentially precomputed input
        """
        inputs_to_layers = self.get_layer_inputs(x, precomputed=precomputed)
        preprocessed = self._preprocess_inputs(inputs_to_layers, compress_convolutional=True)
        weights = [w.flatten(start_dim=1) for w in weights]
        return [alignment(input, weight, method=method, relative=relative) for input, weight in zip(preprocessed, weights)]

    @torch.no_grad()
    def forward_targeted_dropout(self, x, idxs, layers):
        """
        perform forward pass with targeted dropout on output of hidden layers

        **idxs** and **layers** are matched length tuples describing the layer to dropout
        in and the idxs in that layer to dropout. The dropout happens in the activations
        of the layer (so layer=(0) corresponds to the output of the first layer).

        returns the output accounting for targeted dropout and also the full list of hidden
        activations after targeted dropout
        """
        assert check_iterable(idxs) and check_iterable(layers), "idxs and layers need to be iterables with the same length"
        assert len(idxs) == len(layers), "idxs and layers need to be iterables with the same length"
        assert len(layers) == len(set(layers)), "layers must not have any repeated elements"

        hidden_outputs_dict = {}
        
        def dropout(name, dropout_idx):
            def dropout_hook(module, input, output):
                fraction_dropout = len(dropout_idx) / output.shape[1]
                output[:, dropout_idx] = 0
                output = output * (1 - fraction_dropout)
                hidden_outputs_dict[name] = output
                return output
            return dropout_hook
        
        def get_output(name):
            def output_hook(module, input, output):
                hidden_outputs_dict[name] = output
            return output_hook
        
        @contextmanager
        def dropout_context():
            hooks = []
            try:
                for idx_layer, (name, layer) in enumerate(zip(self.alignment_names, self.alignment_layers)):
                    if idx_layer in layers:
                        dropout_idx = idxs[{val: idx for idx, val in enumerate(layers)}[idx_layer]]
                        hooks.append(layer.register_forward_hook(dropout(name , dropout_idx)))
                    else:
                        hooks.append(layer.register_forward_hook(get_output(name)))
                
                yield
            finally:
                for hook in hooks:
                    hook.remove()
        
        with dropout_context():
            x = self.base_model(x)

        assert self.num_layers() == len(hidden_outputs_dict), "number of outputs and the number of alignment layers need to be the same"
        hidden_outputs = [hidden_outputs_dict[name] for name in self.alignment_names]

        # return output of network and outputs of each alignment layer
        return x, hidden_outputs

    @torch.no_grad()
    def forward_eigenvector_dropout(self, x, eigenvalues, eigenvectors, idxs, layers):
        """
        perform forward pass with targeted dropout of loadings on eigenvectors on input to hidden layers

        **eigenvalues**, **eigenvectors**, **idxs** and **layers** are matched length tuples describing:
        eigenvalues: the eigenvalues of each eigenvector for the input to each layer
        eigenvectors: the eigenvectors of activity corresponding to the input to each layer
        idxs: which eigenvectors to dropout from the activity as it propagates through the network
        layers: which layers to do dropouts in (an index)

        for convolutional layers, dropout is done on each stride independently (with the same subspace)

        returns the output accounting for targeted dropout and also the full list of hidden
        activations after targeted dropout. will correct the norm based on the fraction of
        variance contained in the eigenvalues
        """
        assert check_iterable(idxs) and check_iterable(layers), "idxs and layers need to be iterables with the same length"
        assert len(idxs) == len(layers), "idxs and layers need to be iterables with the same length"
        assert len(layers) == len(set(layers)), "layers must not have any repeated elements"
        assert len(layers) == len(eigenvalues), "list of eigenvalues must have same length as list of layers"
        assert len(layers) == len(eigenvectors), "list of eigenvectors must have same length as list of layers"
        device = get_device(x)
        
        hidden_inputs_dict = {}
        org_forward_methods = {}
        
        @contextmanager
        def eigenvector_dropout_context():
            hooks = []

            def get_input(name):
                def input_hook(module, input, output):
                    hidden_inputs_dict[name] = input
                return input_hook

            try:
                for idx_layer, (name, layer) in enumerate(zip(self.alignment_names, self.alignment_layers)):
                    if idx_layer in layers:
                        # we need to get the target subspace after dropping out eigenvectors

                        # get index to target layer
                        idx_to_layer = {val: idx for idx, val in enumerate(layers)}[idx_layer]

                        # get dropout indices of which eigenvectors to remove
                        dropout_idx = idxs[idx_to_layer]

                        # retrieve only the requested eigenvectors & eigenvalues
                        dropout_evec = remove_by_idx(eigenvectors[idx_to_layer].to(device), dropout_idx, 1)
                        dropout_eval = remove_by_idx(eigenvalues[idx_to_layer].to(device), dropout_idx, 0)

                        # correction is defined as the square root as the ratio of variance preserved in the subspace
                        # this will roughly preserve the average norm of the data for each sample
                        dropout_correction = torch.sqrt(torch.sum(eigenvalues[idx_to_layer]) / torch.sum(dropout_eval))

                        # do forward pass through this layer
                        kwargs = dict(subspace=dropout_evec, correction=dropout_correction)
                        self._forward_subspace(name, layer, hidden_inputs_dict, hooks, org_forward_methods, **kwargs)
                    else:
                        hooks.append(layer.register_backward_hook(get_input(name)))

                yield

            finally:
                for hook in hooks:
                    hook.remove()

        with eigenvector_dropout_context():
            x = self.base_model(x)
        
        for name, layer in zip(self.alignment_names, self.alignment_layers):
            if name in org_forward_methods.keys():
                layer.forward = org_forward_methods[name]

        assert self.num_layers() == len(hidden_inputs_dict), f"number of inputs {len(hidden_inputs_dict)} and the number of alignment layers {self.num_layers()} need to be the same"
        hidden_inputs = [hidden_inputs_dict[name] for name in self.alignment_names]

        # return output of network and inputs to each alignment layer
        return x, hidden_inputs

    def _forward_subspace(self, name, layer, hidden_inputs_dict, hooks, org_forward_methods, subspace=None, correction=None):
        """helper for sending to forward function of desired type"""
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            self._forward_subspace_convolutional(name, layer, hidden_inputs_dict, org_forward_methods, subspace=subspace, correction=correction)
        else:
            self._forward_subspace_linear(name, layer, hidden_inputs_dict, hooks, subspace=subspace, correction=correction)

    def _forward_subspace_linear(self, name, layer, hidden_inputs_dict, hooks, subspace=None, correction=None):
        """
        implement forward pass for linear layer with optional subspace projection of input to layer
        """
        def subspace_linear(name, hidden_inputs_dict, subspace, correction):
            def modify_input_hook(module, input):
                if subspace is not None:
                    input = torch.matmul(torch.matmul(input[0], subspace), subspace.T)
                    if correction is not None:
                        input = input * correction
                hidden_inputs_dict[name] = input
                return input
            return modify_input_hook

        hooks.append(layer.register_forward_pre_hook(subspace_linear(name, hidden_inputs_dict, subspace, correction)))

    def _forward_subspace_convolutional(self, name, layer, hidden_inputs_dict, org_forward_methods, subspace=None, correction=None):
        """
        implement forward pass for convolutional layer with optional subspace projection of input

        if subspace provided, will project x onto the subspace then onto it's transpose to keep
        only some dimensions of the activity while keeping x in the same basis.
        (e.g. new_x = subspace @ subspace.T @ x)

        then will pass through the layer.

        projects onto the subspace within each stride of the convolution
        """

        def _conv_with_subspace(self, x, name=name, hidden_inputs_dict=hidden_inputs_dict, subspace=subspace, correction=correction):
            """internal helper for convolving in a subspace"""
            # start by getting size of input to conv layer and layer parameters
            h_max, w_max = get_maximum_strides(x.size(2), x.size(3), self)
            layer_prms = get_unfold_params(self)

            # perform convolution in unfolded space
            weight = self.weight.data
            weight = weight.view(weight.size(0), -1)

            # this is the layer we want to reimplement with a subspace projection
            x = torch.nn.functional.unfold(x, self.kernel_size, **layer_prms)

            # project out subspace
            x = torch.matmul(subspace, torch.matmul(subspace.T, x))

            # apply multiplicative gain correction if provided
            if correction is not None:
                x = x * correction

            # save input to target conv layer
            input_to_conv = x.clone()
            hidden_inputs_dict[name] = input_to_conv

            # convolve
            x = torch.matmul(weight, x).view(x.size(0), weight.size(0), h_max, w_max)

            # add bias
            x = x + self.bias.view(-1, 1, 1)

            return x

        if subspace is not None:
            org_forward_methods[name] = layer.forward
            # if not packaged in sequential, can do this directly
            layer.forward = _conv_with_subspace.__get__(layer, nn.Module)

    @torch.no_grad()
    def measure_eigenfeatures(self, inputs, with_updates=True, centered=True):
        """
        measure the eigenvalues and eigenvectors of the input to each layer
        and also measure how much each weight array uses each eigenvector

        computing eigenfeatures is intensive for big matrices so it's not a
        good idea to this on unfolded data in convolutional layers. It may be
        a good idea to do it sometimes -- I think sklearn's IncrementalPCA
        algorithm is best for this. But it still takes a while so shouldn't be
        done frequently, only after training for important networks.

        if centered=True, will measure eigenfeatures of true covariance matrix.
        if centered=False, will measure eigenfeatures of uncentered X.T @ X where
        x is the input to each alignment layer.

        for convolutional layers, will unfold and measure eigenfeatures for each
        stride (and take the average across strides weighted by input variance)
        """
        # retrieve weights, reshape, and flatten inputs as required
        weights = self.get_alignment_weights(flatten=True)
        inputs = self._preprocess_inputs(inputs, compress_convolutional=True)

        # measure eigenfeatures
        return self._measure_layer_eigenfeatures(inputs, weights, centered=centered, with_updates=with_updates)

    def measure_class_eigenfeatures(self, inputs, labels, eigenvectors, rms=False, with_updates=True):
        """
        propagate an entire dataset through the network and measure the contribution
        of each eigenvector to each element of the class

        to keep things in a useful tensor format, will match samples across classes and
        therefore may ignore "extra" samples if the dataloader doesn't have equal
        representation across classes. Keep in mind it will use the first N samples per
        class where N=min_samples_per_class, so using a random dataloader is a good idea.

        returns list of beta by class, where the list has len()==num_layers
        and each element is a tensor with size (num_classes, num_dimensions, num_samples_per_class)

        if rms=True, will convert beta_by_class to an average with the RMS method
        """
        # get stacked indices to the elements of each class
        classes = torch.unique(labels)
        num_classes = len(classes)
        idx_to_class = [torch.where(labels == ii)[0] for ii in classes]
        num_per_class = [len(idx) for idx in idx_to_class]
        min_per_class = min(num_per_class)
        if any([npc > min_per_class for npc in num_per_class]):
            max_per_class = max(num_per_class)
            if (max_per_class / min_per_class) > 2:
                warn_message = f"Number of elements to each class is unequal (min={min_per_class}, max={max_per_class}). Clipping examples."
                warn(warn_message, RuntimeWarning, stacklevel=1)
            idx_to_class = [idx[:min_per_class] for idx in idx_to_class]

        # use single tensor for fast indexing
        idx_to_class = torch.stack(idx_to_class).unsqueeze(1)

        # measure the contribution of each eigenvector on the representation of each input
        beta_activity = []
        inputs = self._preprocess_inputs(inputs, compress_convolutional=False)
        zipped = zip(inputs, eigenvectors, self.get_alignment_layers())
        for input, evec, layer in zipped:
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                print("measure_class_eigenfeatures has not integrated new convolutional approach")
                stride_var = torch.var(input, dim=1, keepdim=True)
                projection = torch.matmul(evec.T, input)
                projection = weighted_average(projection, stride_var, dim=2)
                beta_activity.append(projection.T.unsqueeze(0))
            else:
                beta_activity.append((input @ evec).T.unsqueeze(0))

        # organize activity by class in extra dimension
        beta_by_class = [torch.gather(betas.expand(num_classes, -1, -1), 2, idx_to_class.expand(-1, betas.size(1), -1)) for betas in beta_activity]

        # get average by class with RMS method (root-mean-square) if requested
        if rms:
            beta_by_class = [torch.sqrt(torch.mean(beta**2, dim=2)) for beta in beta_by_class]

        # return beta by class in requested format (average or not based on rms value)
        return beta_by_class

    def _measure_layer_eigenfeatures(self, inputs, weights, centered=True, with_updates=True):
        """
        helper method for measuring eigenfeatures of each layer

        input should be preprocessed weights (see _preprocess_inputs()) using compress_convolutional=True
        weights should be preprocessed weights (in the case of convolutional layers, see get_alignment_weights())
        """
        beta, eigenvalues, eigenvectors = [], [], []

        # go through each layers inputs, weights, and metaparameters
        zipped = enumerate(zip(inputs, weights))
        iterate = tqdm(zipped) if with_updates else zipped
        for ii, (input, weight) in iterate:
            """
            #ATL 240227: this used to be divided into linear vs. convolutional
            but now _preprocess_inputs will fold stride dimension in with batch dimension
            so it will behave the same way as a linear layer
            """
            # measure evals and evecs across input
            w, v = smart_pca(input.T, centered=centered)

            # Measure abs value of dot product of weights on eigenvectors for each layer
            weight = weight / torch.norm(weight, dim=1, keepdim=True)
            beta.append(weight.cpu() @ v)

            # Append eigenvalues and eigenvectors to output
            eigenvalues.append(w)
            eigenvectors.append(v)

            """
            #ATL 240227 - obsolete code now that stride dimension is being folded into batch dimension
            # if a convolutional layer, then:
            if metaprm['unfold']:
                # measure variance across dimensions (the actual variance within each stride)
                # then take average across batch
                bvar = torch.mean(torch.var(input, dim=1), dim=0) 
                
                # get eigenvalues and eigenvectors for each stride (treat stride as batch dimension here)
                w, v = smart_pca(input.permute((2, 1, 0)), centered=centered)
                
                # Measure abs value of dot product of weights on eigenvectors for each layer
                num_strides = v.size(0)
                weight = weight / torch.norm(weight, dim=1, keepdim=True)
                b = torch.bmm(weight.cpu().unsqueeze(0).expand(num_strides, -1, -1), v)

                # Contract across strides by weighted average of average variance per stride
                b_weighted_by_var = weighted_average(b, bvar.view(-1, 1, 1), 0)
                w_weighted_by_var = weighted_average(w, bvar.view(-1, 1), 0)
                v_weighted_by_var = weighted_average(v, bvar.view(-1, 1, 1), 0)

                # Append to output
                beta.append(b_weighted_by_var)
                eigenvalues.append(w_weighted_by_var)
                eigenvectors.append(v_weighted_by_var)
            """

        return beta, eigenvalues, eigenvectors

    def _process_collect_activity(self, dataset, train_set=True, with_updates=True, use_training_mode=False):
        """
        helper for processing and collecting activity of network in response to all inputs of dataloader

        automatically places all data on cpu

        returns inputs to each alignment layer, concatenated across entire dataloader as a per layer list
        returns labels of entire dataset

        with_updates turns on or off the progress bar (using tqdm)
        if use_training_mode=False, will put net into evaluation mode (and return to original mode)
        if use_training_mode=True, will put net into training mode (and return to original mode)
        """
        # get device of network
        device = get_device(self)

        # put network in evaluation mode
        training_mode = set_net_mode(self, training=use_training_mode)

        # store input and measure activations for every element in dataloader
        allinputs = []
        alllabels = []
        dataloader = dataset.train_loader if train_set else dataset.test_loader
        dataloop = tqdm(dataloader) if with_updates else dataloader
        for batch in dataloop:
            input, labels = dataset.unwrap_batch(batch, device=device)
            layer_inputs = [input.cpu() for input in self.get_layer_inputs(input, precomputed=False)]
            allinputs.append(layer_inputs)
            alllabels.append(labels.cpu())

        # return network to original training/eval mode, whatever it was
        set_net_mode(self, training=training_mode)

        # create large list of tensors containing input to each layer
        inputs = [torch.cat([input[layer] for input in allinputs], dim=0) for layer in range(self.num_layers())]
        labels = torch.cat(alllabels, dim=0)

        # return outputs
        return inputs, labels

    @torch.no_grad()
    def shape_eigenfeatures(self, idx_layers, eigenvalues, eigenvectors, eval_transform):
        """
        method for shaping the eigenfeatures of a network

        use eval_transform to shape a network by changing the scale of each
        eigenvector's contribution to the weights based on the associated
        eigenvalue for a specific set of layers.

        idx_layers is a list indicating which layers to shape (where the indices
        should correspond to order of the layers in self.get_alignment_layers())

        eigenvalues and eigenvectors should be a list with length=len(idx_layers)
        and each should correspond to the eigenvalues & eigenvectors of the input
        to each layer in idx_layers

        eval_transform is a callable function that takes a set of eigenvalues and
        returns the desired scale of eigenvectors associated with each eigenvalue
        for example, if eigenvalues[0]=[1, 0.5, 0.25, 0.125]*37.9991, eval_transform
        might return [1, 1, 1, 0] which simply "kills" the last eigenvector
        alternatively, it could return [1, 0.25, 0.25**2, 0.125**2]*37.9991**2/sum
        where it shapes each eigenvector by the square of the eigenvalues
        """
        # do some input checks
        assert all([idx in range(self.num_layers()) for idx in idx_layers]), (
            "idx_layers includes some indices not in alignment layers",
            f"(provided: {idx_layers}, alignment layer indecies: {list(range(self.num_layers()))})",
        )
        assert len(idx_layers) == len(eigenvalues), "length of idx_layers and eigenvalues doesn't match"
        assert len(idx_layers) == len(eigenvectors), "length of idx_layers and eigenvectors doesn't match"

        # make sure eigenvalues and eigenvalues are on same device as network
        device = get_device(self)
        eigenvalues = [evals.to(device) for evals in eigenvalues]
        eigenvectors = [evecs.to(device) for evecs in eigenvectors]

        # get weights and original shapes of requested alignment layers
        weight_shape = [self.get_alignment_weights()[idx].shape for idx in idx_layers]
        weights = [self.get_alignment_weights(flatten=True)[idx] for idx in idx_layers]

        # measure original norm of weights
        norm_of_weights = [torch.norm(weight, dim=1, keepdim=True) for weight in weights]

        # normalize weight vector
        weights = [weight / torch.norm(weight, dim=1, keepdim=True) for weight in weights]

        # for each layer, process the eigenvalues, shape the weights, and update the network
        zipped = zip(idx_layers, eigenvalues, eigenvectors, weights, norm_of_weights, weight_shape)
        for idx, evals, evecs, weight, norm_weight, shape in zipped:
            # transform eigenvalues
            eval_keep_fraction = eval_transform(evals)
            assert (
                type(eval_keep_fraction) == type(evals) and eval_keep_fraction.shape == evals.shape
            ), "eval_transform returned new evals with the wrong type or shape"
            # define a projection matrix that scales the contribution of each eigenvalue by eval_keep_fraction
            proj_matrix = evecs @ torch.diag(eval_keep_fraction) @ evecs.T
            # shape the weights
            shaped_weights = weight @ proj_matrix
            # renormalize them to their original norm
            shaped_weights = shaped_weights / torch.norm(shaped_weights, dim=1, keepdim=True)  # normalize
            shaped_weights = shaped_weights * norm_weight
            # reshape to original shape
            shaped_weights = torch.reshape(shaped_weights, shape)
            # update the network
            self.get_alignment_layers()[idx].weight.data = shaped_weights
