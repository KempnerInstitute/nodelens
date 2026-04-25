Pruning API Reference
=====================

.. currentmodule:: nodelens.pruning

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

.. autoclass:: nodelens.pruning.base.BasePruningStrategy
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: nodelens.pruning.base.IterativePruningStrategy
   :members:
   :undoc-members:
   :show-inheritance:

Magnitude-based Strategies
--------------------------

.. autoclass:: nodelens.pruning.strategies.MagnitudePruning
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: nodelens.pruning.strategies.IterativeMagnitudePruning
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: nodelens.pruning.strategies.GlobalMagnitudePruning
   :members:
   :undoc-members:
   :show-inheritance:

Gradient-based Strategies
-------------------------

.. autoclass:: nodelens.pruning.strategies.GradientPruning
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: nodelens.pruning.strategies.FisherPruning
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: nodelens.pruning.strategies.MomentumPruning
   :members:
   :undoc-members:
   :show-inheritance:

Random Strategies
-----------------

.. autoclass:: nodelens.pruning.strategies.RandomPruning
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: nodelens.pruning.strategies.LayerwiseRandomPruning
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: nodelens.pruning.strategies.BernoulliPruning
   :members:
   :undoc-members:
   :show-inheritance:

Parallel Strategies
-------------------

.. autoclass:: nodelens.pruning.strategies.ParallelModePruning
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: prune_parallel
   .. automethod:: combine_masks

.. autoclass:: nodelens.pruning.strategies.TensorizedPruning
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: compute_pruning_tensor
   .. automethod:: analyze_pruning_patterns

.. autoclass:: nodelens.pruning.strategies.AsyncParallelPruning
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: prune_modules_parallel

.. autoclass:: nodelens.pruning.strategies.ParallelPruningResult
   :members:
   :undoc-members:
   :show-inheritance:

Pruning Experiments
-------------------

.. autoclass:: nodelens.pruning.experiments.ProgressiveDropoutExperiment
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: nodelens.pruning.experiments.CascadingLayerPruningExperiment
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: nodelens.pruning.experiments.LayerIsolatedPruningExperiment
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: nodelens.pruning.experiments.EigenvectorDropoutExperiment
   :members:
   :undoc-members:
   :show-inheritance:
