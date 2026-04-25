Extending the Framework
=======================

This guide explains how to extend LossLens with custom components
using the registry system.

Overview
--------

The framework uses a **registry system** where components register themselves
using decorators. When you import a module with ``@register_*`` decorators,
the components automatically become available.

This enables:

1. **Plugin-based architecture** - Add new components without modifying core code
2. **Configuration-driven instantiation** - Create components from config files
3. **Auto-discovery** - Automatically find and register components from packages
4. **Metadata tracking** - Store component capabilities, requirements, and documentation

Available Registries
--------------------

.. list-table::
   :header-rows: 1

   * - Registry
     - Decorator
     - Description
   * - ``METRIC_REGISTRY``
     - ``@register_metric``
     - Per-neuron metrics (RQ, MI, etc.)
   * - ``ANALYZER_REGISTRY``
     - ``@register_analyzer``
     - Analysis pipelines (clustering, halo)
   * - ``PRUNER_REGISTRY``
     - ``@register_pruner``
     - Pruning strategies
   * - ``VISUALIZER_REGISTRY``
     - ``@register_visualizer``
     - Visualization components
   * - ``EVALUATOR_REGISTRY``
     - ``@register_evaluator``
     - Model evaluation
   * - ``EXPERIMENT_REGISTRY``
     - ``@register_experiment``
     - Full experiment pipelines

Creating a Custom Metric
------------------------

Here's how to create and register a custom alignment metric:

.. code-block:: python

   from alignment.core.registry import register_metric
   from alignment.core.protocols import BaseMetric
   import torch

   @register_metric(
       "activation_kurtosis",
       category="statistical",
       description="Measures kurtosis of activation distributions per neuron",
       tags=["statistics", "distribution", "outlier"],
       aliases=["kurtosis", "act_kurt"],
   )
   class ActivationKurtosis(BaseMetric):
       """
       Compute excess kurtosis of activations for each neuron.

       High kurtosis indicates heavy tails (potential outlier neurons).
       """

       name = "activation_kurtosis"
       requires_inputs = False
       requires_weights = False
       requires_outputs = True

       def __init__(self, fisher: bool = True):
           """
           Args:
               fisher: If True, compute excess kurtosis (subtract 3).
           """
           self.fisher = fisher

       def compute(
           self,
           inputs=None,
           weights=None,
           outputs=None,
           **kwargs
       ) -> torch.Tensor:
           """
           Compute kurtosis for each neuron/channel.

           Args:
               outputs: Activations [batch_size, num_neurons]

           Returns:
               Kurtosis values [num_neurons]
           """
           if outputs is None:
               raise ValueError("ActivationKurtosis requires outputs")

           # Handle different tensor shapes
           if outputs.dim() == 4:
               # Conv layer: [batch, channels, h, w] -> [batch, channels]
               outputs = outputs.mean(dim=(2, 3))

           # Compute per-neuron statistics
           mean = outputs.mean(dim=0)
           std = outputs.std(dim=0) + 1e-8
           z = (outputs - mean) / std
           m4 = (z ** 4).mean(dim=0)

           if self.fisher:
               return m4 - 3.0
           return m4

Creating a Custom Analyzer
--------------------------

Analyzers perform higher-level analysis on metrics:

.. code-block:: python

   from alignment.core.registry import register_analyzer
   from alignment.core.protocols import BaseAnalyzer
   import numpy as np

   @register_analyzer(
       "layer_similarity",
       category="comparison",
       description="Analyze similarity between layers using CKA",
       tags=["cka", "similarity", "cross-layer"],
   )
   class LayerSimilarityAnalyzer(BaseAnalyzer):
       """Analyze representational similarity between layers using CKA."""

       name = "layer_similarity"
       requires = ["activations"]
       provides = ["similarity_matrix", "layer_clusters"]

       def __init__(self, method: str = "linear_cka"):
           self.method = method

       def analyze(self, metrics, model=None, activations=None, **kwargs):
           """Compute layer-to-layer similarity matrix."""
           if activations is None:
               raise ValueError("LayerSimilarityAnalyzer requires activations")

           # Your analysis logic here
           layer_names = list(activations.keys())
           n_layers = len(layer_names)
           similarity_matrix = np.zeros((n_layers, n_layers))

           # ... compute CKA similarity ...

           return {
               "similarity_matrix": similarity_matrix.tolist(),
               "layer_names": layer_names,
               "method": self.method,
           }

       def visualize(self, results, output_dir=None, **kwargs):
           """Generate similarity heatmap."""
           # Your visualization logic here
           return []  # List of saved figure paths

Creating a Custom Pruner
------------------------

Pruning strategies define how to select neurons for removal:

.. code-block:: python

   from alignment.core.registry import register_pruner
   from alignment.core.protocols import BasePruner
   import torch

   @register_pruner(
       "entropy_based",
       category="information",
       description="Prune neurons with low activation entropy",
       tags=["entropy", "information", "diversity"],
   )
   class EntropyBasedPruner(BasePruner):
       """Prune neurons based on activation entropy."""

       name = "entropy_based"
       structured = True

       def __init__(self, n_bins: int = 50):
           self.n_bins = n_bins

       def compute_importance(self, model, layer_name, activations=None, **kwargs):
           """Compute entropy-based importance scores."""
           if activations is None:
               raise ValueError("Requires activations")

           # Compute entropy per neuron
           n_neurons = activations.size(-1)
           entropies = torch.zeros(n_neurons)

           for i in range(n_neurons):
               hist = torch.histc(activations[..., i], bins=self.n_bins)
               probs = hist / hist.sum()
               probs = probs[probs > 0]
               entropies[i] = -(probs * torch.log2(probs)).sum()

           return entropies

Using Custom Components
-----------------------

Once registered, custom components can be used by name:

.. code-block:: python

   from alignment.core.registry import get_metric, initialize_registries

   # Initialize (discovers built-in + custom components)
   initialize_registries()

   # Use by name
   metric = get_metric("activation_kurtosis", fisher=True)
   scores = metric.compute(outputs=activations)

   # Use alias
   metric = get_metric("kurtosis")  # Same as "activation_kurtosis"

   # Search for metrics
   from alignment.core import METRIC_REGISTRY
   statistical_metrics = METRIC_REGISTRY.search(tags=["statistics"])

Plugin Discovery
----------------

Place your custom components in these locations for auto-discovery:

- ``./plugins/`` (project-local)
- ``~/.alignment/plugins/`` (user-global)

They will be automatically loaded when the framework initializes.

Or manually load from a custom location:

.. code-block:: python

   from alignment.core.registry import discover_plugins

   discover_plugins(["./my_custom_plugins/"])

Using in Configuration Files
----------------------------

Custom components can be referenced in YAML configs by name:

.. code-block:: yaml

   pruning:
     algorithms:
       - "activation_kurtosis"    # Your custom metric!
       - "entropy_based"          # Your custom pruner!
       - "magnitude"              # Built-in

Best Practices
--------------

1. **Use meaningful names**: Choose descriptive, unique names
2. **Add metadata**: Tags and descriptions help discoverability
3. **Follow protocols**: Implement the required interface methods
4. **Document**: Add docstrings explaining what your component does
5. **Test**: Include tests for your custom components
6. **Handle edge cases**: Check for None inputs, empty tensors, etc.
