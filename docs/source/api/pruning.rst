Pruning API Reference
=====================

This section provides detailed documentation for all pruning strategies available in the alignment framework.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The pruning module provides a unified interface for various neural network pruning strategies:

- **Magnitude-based**: Prune based on weight magnitudes
- **Gradient-based**: Use gradient information for importance
- **Random**: Baseline random pruning strategies
- **Structured**: Hardware-efficient structured pruning (planned)

Base Classes
------------

.. automodule:: alignment.pruning.base
   :members:
   :undoc-members:
   :show-inheritance:

BasePruningStrategy
~~~~~~~~~~~~~~~~~~~

.. autoclass:: alignment.pruning.base.BasePruningStrategy
   :members:
   :undoc-members:
   
   **Abstract Methods:**
   
   - :meth:`compute_importance`: Calculate importance scores
   - :meth:`create_mask`: Generate pruning mask from scores
   
   **Common Parameters:**
   
   - **sparsity** (*float*): Target sparsity level (0.0 to 1.0)
   - **structured** (*bool*): Whether to use structured pruning
   - **global_pruning** (*bool*): Prune globally vs per-layer

PruningConfig
~~~~~~~~~~~~~

.. autoclass:: alignment.pruning.base.PruningConfig
   :members:
   :undoc-members:
   
   **Attributes:**
   
   .. attribute:: sparsity
      :type: float
      
      Target sparsity level (fraction of weights to prune)
      
   .. attribute:: structured
      :type: bool
      
      Use structured pruning (channels/filters) (default: False)
      
   .. attribute:: global_pruning
      :type: bool
      
      Apply global magnitude threshold (default: False)
      
   .. attribute:: iterative
      :type: bool
      
      Use iterative pruning with recovery (default: False)

Magnitude-based Pruning
-----------------------

.. automodule:: alignment.pruning.strategies.magnitude
   :members:
   :undoc-members:
   :show-inheritance:

MagnitudePruning
~~~~~~~~~~~~~~~~

.. autoclass:: alignment.pruning.strategies.magnitude.MagnitudePruning
   :members:
   :undoc-members:
   
   **Description:**
   
   Prunes weights with smallest absolute values. This is the most common 
   baseline pruning method.
   
   **Algorithm:**
   
   1. Compute absolute values of all weights
   2. Determine threshold based on sparsity target
   3. Prune weights below threshold
   
   **Example:**
   
   .. code-block:: python
      
      from alignment.pruning.strategies import MagnitudePruning
      from alignment.pruning import PruningConfig
      
      config = PruningConfig(sparsity=0.9, global_pruning=True)
      pruner = MagnitudePruning(config)
      
      # Compute importance scores
      scores = pruner.compute_importance(model)
      
      # Create pruning mask
      mask = pruner.create_mask(scores)
      
      # Apply pruning
      pruned_model = pruner.apply(model)

IterativeMagnitudePruning
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: alignment.pruning.strategies.magnitude.IterativeMagnitudePruning
   :members:
   :undoc-members:
   
   **Description:**
   
   Implements iterative magnitude pruning with recovery periods between 
   pruning steps. This often achieves better accuracy than one-shot pruning.
   
   **Parameters:**
   
   - **n_iterations** (*int*, default=10): Number of pruning iterations
   - **recovery_epochs** (*int*, default=5): Training epochs between iterations
   - **sparsity_schedule** (*str*, default="linear"): How to increase sparsity
     
     - ``"linear"``: Linear increase
     - ``"exponential"``: Exponential increase
     - ``"polynomial"``: Polynomial schedule
   
   **Example:**
   
   .. code-block:: python
      
      config = PruningConfig(sparsity=0.9)
      pruner = IterativeMagnitudePruning(
          config,
          n_iterations=10,
          recovery_epochs=5
      )
      
      # Run iterative pruning
      for iteration in range(pruner.n_iterations):
          mask = pruner.prune_iteration(model, iteration)
          # Fine-tune model for recovery_epochs
          train_model(model, epochs=pruner.recovery_epochs)

GlobalMagnitudePruning
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: alignment.pruning.strategies.magnitude.GlobalMagnitudePruning
   :members:
   :undoc-members:
   
   **Description:**
   
   Applies a global magnitude threshold across all layers rather than 
   per-layer thresholds. This can lead to more efficient pruning by 
   automatically allocating sparsity.
   
   **Advantages:**
   
   - Automatic sparsity allocation
   - No need to tune per-layer ratios
   - Often better accuracy at high sparsity

Gradient-based Pruning
----------------------

.. automodule:: alignment.pruning.strategies.gradient
   :members:
   :undoc-members:
   :show-inheritance:

GradientPruning
~~~~~~~~~~~~~~~

.. autoclass:: alignment.pruning.strategies.gradient.GradientPruning
   :members:
   :undoc-members:
   
   **Description:**
   
   Uses gradient magnitudes as importance scores. Weights with small 
   gradients are considered less important for the loss.
   
   **Parameters:**
   
   - **accumulate_gradients** (*bool*, default=True): Accumulate over batches
   - **normalize** (*bool*, default=True): Normalize scores per layer
   
   **Example:**
   
   .. code-block:: python
      
      config = PruningConfig(sparsity=0.8)
      pruner = GradientPruning(config, accumulate_gradients=True)
      
      # Accumulate gradients over dataset
      for batch in dataloader:
          loss = model(batch)
          loss.backward()
          pruner.accumulate_gradients(model)
      
      # Prune based on accumulated gradients
      mask = pruner.create_mask_from_gradients()

FisherPruning
~~~~~~~~~~~~~

.. autoclass:: alignment.pruning.strategies.gradient.FisherPruning
   :members:
   :undoc-members:
   
   **Description:**
   
   Uses Fisher information (squared gradients) as importance scores. 
   This approximates the effect on the loss function.
   
   **Mathematical Basis:**
   
   .. math::
      
      F_i = \mathbb{E}[(\nabla_{\theta_i} \mathcal{L})^2]
   
   **Parameters:**
   
   - **n_samples** (*int*, default=1000): Samples for Fisher estimation
   - **damping** (*float*, default=1e-5): Damping factor for stability

MomentumPruning
~~~~~~~~~~~~~~~

.. autoclass:: alignment.pruning.strategies.gradient.MomentumPruning
   :members:
   :undoc-members:
   
   **Description:**
   
   Combines gradient magnitude with momentum information from training. 
   Weights with both small gradients and small momentum are pruned.
   
   **Parameters:**
   
   - **momentum_weight** (*float*, default=0.5): Weight for momentum term
   - **use_adam_stats** (*bool*, default=False): Use Adam optimizer stats

Random Pruning
--------------

.. automodule:: alignment.pruning.strategies.random
   :members:
   :undoc-members:
   :show-inheritance:

RandomPruning
~~~~~~~~~~~~~

.. autoclass:: alignment.pruning.strategies.random.RandomPruning
   :members:
   :undoc-members:
   
   **Description:**
   
   Randomly prunes weights regardless of their values. Useful as a 
   baseline for comparing other pruning methods.
   
   **Example:**
   
   .. code-block:: python
      
      config = PruningConfig(sparsity=0.9)
      pruner = RandomPruning(config, seed=42)
      
      mask = pruner.create_mask(model)

LayerwiseRandomPruning
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: alignment.pruning.strategies.random.LayerwiseRandomPruning
   :members:
   :undoc-members:
   
   **Description:**
   
   Applies random pruning with the same sparsity ratio per layer, 
   rather than globally.

BernoulliPruning
~~~~~~~~~~~~~~~~

.. autoclass:: alignment.pruning.strategies.random.BernoulliPruning
   :members:
   :undoc-members:
   
   **Description:**
   
   Each weight is independently pruned with probability p (sparsity level). 
   This can result in variable actual sparsity.

Using Pruning in Experiments
----------------------------

Integration with Experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All pruning strategies integrate seamlessly with the experiment framework:

.. code-block:: python
   
   from alignment.experiments import ProgressiveDropoutExperiment
   from alignment.pruning import get_pruning_strategy
   
   # Configure experiment with pruning
   config = ExperimentConfig(
       name="magnitude_pruning_experiment",
       model_name="resnet18",
       dataset_name="cifar10",
       
       # Pruning configuration
       pruning_strategy="magnitude",  # or "gradient", "fisher", etc.
       pruning_config={
           "sparsity": 0.9,
           "global_pruning": True,
           "iterative": True,
           "n_iterations": 10
       }
   )
   
   experiment = ProgressiveDropoutExperiment(config)
   results = experiment.run()

Custom Pruning Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~

Create custom pruning strategies by inheriting from `BasePruningStrategy`:

.. code-block:: python
   
   from alignment.pruning.base import BasePruningStrategy
   from alignment.pruning import register_pruning_strategy
   
   @register_pruning_strategy("my_custom_pruning")
   class MyCustomPruning(BasePruningStrategy):
       """Custom pruning based on my importance metric."""
       
       def compute_importance(self, model):
           """Compute custom importance scores."""
           importance_scores = {}
           
           for name, param in model.named_parameters():
               if 'weight' in name:
                   # Your custom importance computation
                   scores = custom_importance_function(param)
                   importance_scores[name] = scores
           
           return importance_scores
       
       def create_mask(self, importance_scores):
           """Create pruning mask from importance scores."""
           masks = {}
           
           for name, scores in importance_scores.items():
               # Determine threshold
               threshold = torch.quantile(scores, self.config.sparsity)
               
               # Create binary mask
               masks[name] = scores > threshold
           
           return masks

Best Practices
--------------

**Choosing a Pruning Strategy:**

1. **Magnitude Pruning**: Good default choice, simple and effective
2. **Gradient-based**: When you have task-specific importance
3. **Fisher Pruning**: For minimal loss degradation
4. **Random**: Only as a baseline

**Sparsity Levels:**

- Start with moderate sparsity (50-80%)
- Use iterative pruning for high sparsity (>90%)
- Monitor accuracy degradation
- Consider structured pruning for deployment

**Fine-tuning:**

- Always fine-tune after pruning
- Use lower learning rates
- Longer training for higher sparsity
- Consider knowledge distillation

See Also
--------

- :doc:`/user_guide/pruning_strategies` - User guide for pruning strategies
- :doc:`/api/experiments` - Using pruning in experiments 