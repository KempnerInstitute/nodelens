Pruning API Reference
=====================

.. currentmodule:: alignment.pruning

Main Interface
--------------

.. autofunction:: get_pruning_strategy
.. autofunction:: list_pruning_strategies

Configuration
-------------

.. autoclass:: PruningConfig
   :members:
   :undoc-members:
   :show-inheritance:

Base Classes
------------

.. autoclass:: alignment.pruning.base.BasePruningStrategy
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: alignment.pruning.base.IterativePruningStrategy
   :members:
   :undoc-members:
   :show-inheritance:

Magnitude-based Strategies
--------------------------

.. autoclass:: alignment.pruning.strategies.MagnitudePruning
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: alignment.pruning.strategies.IterativeMagnitudePruning
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: alignment.pruning.strategies.GlobalMagnitudePruning
   :members:
   :undoc-members:
   :show-inheritance:

Gradient-based Strategies
-------------------------

.. autoclass:: alignment.pruning.strategies.GradientPruning
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: alignment.pruning.strategies.FisherPruning
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: alignment.pruning.strategies.MomentumPruning
   :members:
   :undoc-members:
   :show-inheritance:

Random Strategies
-----------------

.. autoclass:: alignment.pruning.strategies.RandomPruning
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: alignment.pruning.strategies.LayerwiseRandomPruning
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: alignment.pruning.strategies.BernoulliPruning
   :members:
   :undoc-members:
   :show-inheritance:

Parallel Strategies
-------------------

.. autoclass:: alignment.pruning.strategies.ParallelModePruning
   :members:
   :undoc-members:
   :show-inheritance:
   
   .. automethod:: prune_parallel
   .. automethod:: combine_masks

.. autoclass:: alignment.pruning.strategies.TensorizedPruning
   :members:
   :undoc-members:
   :show-inheritance:
   
   .. automethod:: compute_pruning_tensor
   .. automethod:: analyze_pruning_patterns

.. autoclass:: alignment.pruning.strategies.AsyncParallelPruning
   :members:
   :undoc-members:
   :show-inheritance:
   
   .. automethod:: prune_modules_parallel

.. autoclass:: alignment.pruning.strategies.ParallelPruningResult
   :members:
   :undoc-members:
   :show-inheritance:

Pruning Experiments
-------------------

.. autoclass:: alignment.pruning.experiments.ProgressiveDropoutExperiment
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: alignment.pruning.experiments.CascadingLayerPruningExperiment
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: alignment.pruning.experiments.LayerIsolatedPruningExperiment
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: alignment.pruning.experiments.EigenvectorDropoutExperiment
   :members:
   :undoc-members:
   :show-inheritance: 