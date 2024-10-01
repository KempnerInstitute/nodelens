These are notes compiled by Ella Batty on a potential change to the structure of the code base. This change was discussed with Andrew and Houman, and seemed likely to be helpful as long as Pytorch forward hooks are not computationally expensive or slow. It is similar to the outline in code.md with some tweaks.

**Proposed Changes**:

- Instead of having the AlignmentNetwork class inherited by specific model classes (such as AlexNet), we can have the AlignmentNetwork class inherit the specific model classes (which we should not have to alter from their standard implementation based on the following changes). 
- We can then get the internal activations of a model using the [Pytorch forward hook](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook) method instead of having to register specific layers. This change means that we do not have to rewrite the model class for our purposes and we can extend to networks that are not sequential in the forward pass (like ResNet and llama). 
- From AlignmentNetwork class, we can drop the register layer method and associated helper methods such as _include_layer, . Instead, we can enable a method which allows you to specify layers you want to compute alignment for, and the relevant input layer for each. The method can default to the previous layer as the input unless otherwise specified. This allows us to compute alignment for a given layer from previous layer inputs skipping dropout or not (currently the codebase is constrained to skipping dropout). It also means we can compute alignment for non-sequential models. For each layer, we can automatically figure out the transform parameters based on whether the input is conv or linear so we don't have to specify this. 
- This change also allows us to modify the internal activations based on alignment metrics **before** passing them to the next layer (so altering the forward pass). 

**Needs to be investigated before implementing changes**:
- Ensure Pytorch forward hooks are not computationally expensive or slow
- We will need to investigate how this could work if we want to train a model based on a forward pass altered using hooks- I'm not sure how gradients would work. There are backward hooks but I'm less familiar with these.
