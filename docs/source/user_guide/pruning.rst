Pruning Guide
=============

This guide covers the pruning capabilities in the alignment framework, including different strategies and experiment types.

Overview
--------

The framework provides comprehensive pruning capabilities:

- **Multiple Pruning Strategies**: Magnitude, gradient, random, and alignment-based
- **Structured and Unstructured Pruning**: Support for both approaches
- **Various Experiment Types**: Global, layer-wise, cascading, and eigenvector-based

Pruning Strategies
------------------

The framework includes several pruning strategies in ``alignment.pruning.strategies``:

Magnitude-based Pruning
^^^^^^^^^^^^^^^^^^^^^^^

Prunes weights or neurons based on their magnitude:

.. code-block:: python

    from alignment.pruning.strategies import MagnitudePruning

    strategy = MagnitudePruning()
    masks = strategy.compute_masks(model, pruning_ratio=0.5)

Gradient-based Pruning
^^^^^^^^^^^^^^^^^^^^^^

Uses gradient information to determine importance:

.. code-block:: python

    from alignment.pruning.strategies import GradientPruning

    strategy = GradientPruning()
    masks = strategy.compute_masks(model, dataloader, pruning_ratio=0.5)

Random Pruning
^^^^^^^^^^^^^^

Baseline strategy that randomly prunes connections:

.. code-block:: python

    from alignment.pruning.strategies import RandomPruning

    strategy = RandomPruning(seed=42)
    masks = strategy.compute_masks(model, pruning_ratio=0.5)

Alignment-based Pruning
^^^^^^^^^^^^^^^^^^^^^^^

Uses alignment metrics to guide pruning decisions:

.. code-block:: python

    from alignment.pruning.strategies import AlignmentPruning

    strategy = AlignmentPruning(metric="rayleigh_quotient")
    masks = strategy.compute_masks(model, dataloader, pruning_ratio=0.5)

Pruning Experiments
-------------------

Global Pruning
^^^^^^^^^^^^^^

Applies the same pruning rate across all layers:

.. code-block:: python

    from alignment.pruning.experiments import GlobalDropoutExperiment, GlobalDropoutConfig

    config = GlobalDropoutConfig(
        experiment_name="global_pruning_mnist",
        dataset_name="mnist",
        model_name="mlp",
        hidden_sizes=[128, 64],
        dropout_rates=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
        dropout_structure="magnitude"
    )

    experiment = GlobalDropoutExperiment(config)
    results = experiment.run()

Layer-wise Pruning
^^^^^^^^^^^^^^^^^^

Analyzes the effect of pruning individual layers:

.. code-block:: python

    from alignment.pruning.experiments import LayerIsolatedPruningExperiment, LayerIsolatedConfig

    config = LayerIsolatedConfig(
        experiment_name="layer_analysis",
        dataset_name="mnist",
        model_name="mlp",
        pruning_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
        pruning_strategy="magnitude",
        layers_to_prune=["fc1", "fc2"]  # Specific layers
    )

    experiment = LayerIsolatedPruningExperiment(config)
    results = experiment.run()

Cascading Pruning
^^^^^^^^^^^^^^^^^

Progressive pruning that cascades through network layers:

.. code-block:: python

    from alignment.pruning.experiments import CascadingLayerPruningExperiment, CascadingConfig

    config = CascadingConfig(
        experiment_name="cascading_analysis",
        dataset_name="mnist",
        model_name="mlp",
        cascade_direction="forward",
        base_pruning_ratio=0.2,
        cascade_factor=1.5  # Increase pruning by 50% each layer
    )

    experiment = CascadingLayerPruningExperiment(config)
    results = experiment.run()

Eigenvector-based Pruning
^^^^^^^^^^^^^^^^^^^^^^^^^

Uses spectral analysis for pruning:

.. code-block:: python

    from alignment.pruning.experiments import EigenvectorDropoutExperiment, EigenvectorConfig

    config = EigenvectorConfig(
        experiment_name="eigenvector_pruning",
        dataset_name="mnist",
        model_name="mlp",
        num_components=10,
        component_selection="top",  # or "bottom"
        pruning_ratios=[0.1, 0.3, 0.5]
    )

    experiment = EigenvectorDropoutExperiment(config)
    results = experiment.run()

Structured vs Unstructured Pruning
-----------------------------------

Unstructured Pruning
^^^^^^^^^^^^^^^^^^^^

Removes individual weights/connections:

.. code-block:: python

    config = GlobalDropoutConfig(
        structured_pruning=False,  # Default
        pruning_strategy="magnitude"
    )

- **Pros**: Fine-grained control, potentially higher accuracy retention
- **Cons**: Requires sparse matrix support for speedup

Structured Pruning
^^^^^^^^^^^^^^^^^^

Removes entire neurons/channels/filters:

.. code-block:: python

    config = GlobalDropoutConfig(
        structured_pruning=True,
        pruning_strategy="magnitude",
        structure_type="neuron"  # or "channel", "filter"
    )

- **Pros**: Direct speedup, hardware-friendly
- **Cons**: Coarser granularity, potentially more accuracy loss

Analyzing Pruning Results
-------------------------

The experiments return comprehensive results:

.. code-block:: python

    results = experiment.run()

    # Pruning performance
    for ratio in results['pruning_ratios']:
        metrics = results['pruning_results'][ratio]
        print(f"Pruning {ratio*100}%:")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Remaining params: {metrics['remaining_params']}")

    # Layer-wise analysis (for layer-wise experiments)
    if 'layer_results' in results:
        for layer, data in results['layer_results'].items():
            print(f"\nLayer {layer}:")
            print(f"  Sensitivity: {data['sensitivity']}")
            print(f"  Optimal pruning: {data['optimal_ratio']}")

Visualization
-------------

The framework automatically generates pruning analysis plots:

- Accuracy vs pruning ratio curves
- Layer sensitivity heatmaps
- Parameter reduction charts
- Alignment metric evolution

Custom Pruning Strategies
-------------------------

Implement custom strategies by extending the base class:

.. code-block:: python

    from alignment.pruning.strategies import BasePruningStrategy

    class MyCustomPruning(BasePruningStrategy):
        def compute_importance_scores(self, model, dataloader=None):
            """Compute importance scores for each parameter."""
            scores = {}
            for name, param in model.named_parameters():
                if 'weight' in name:
                    # Your custom importance calculation
                    scores[name] = custom_importance(param)
            return scores

        def create_masks(self, scores, pruning_ratio):
            """Create binary masks from scores."""
            masks = {}
            for name, score in scores.items():
                threshold = torch.quantile(score.flatten(), pruning_ratio)
                masks[name] = score > threshold
            return masks

Best Practices
--------------

1. **Start Conservative**: Begin with small pruning ratios (10-30%)
2. **Fine-tune After Pruning**: Allow the model to adapt after pruning
3. **Compare Strategies**: Test multiple strategies on your specific task
4. **Monitor Multiple Metrics**: Don't just track accuracy
5. **Consider Hardware**: Choose structured/unstructured based on deployment

Common Pitfalls
---------------

1. **Pruning Too Aggressively**: Gradual pruning often works better
2. **Ignoring Layer Sensitivity**: Some layers are more critical
3. **Not Fine-tuning**: Models often recover performance with fine-tuning
4. **Wrong Granularity**: Match pruning type to hardware constraints

Advanced Topics
---------------

Iterative Pruning
^^^^^^^^^^^^^^^^^

Prune in multiple rounds:

.. code-block:: python

    config = GlobalDropoutConfig(
        iterative_pruning=True,
        pruning_schedule=[0.2, 0.4, 0.6],  # Cumulative
        fine_tune_epochs=5  # Between rounds
    )

Dynamic Pruning
^^^^^^^^^^^^^^^

Adjust pruning during training:

.. code-block:: python

    config = GlobalDropoutConfig(
        dynamic_pruning=True,
        initial_sparsity=0.0,
        final_sparsity=0.9,
        pruning_frequency=100  # Steps
    )

See Also
--------

- :doc:`experiments` - Overview of experiment types
- :doc:`metrics` - Alignment metrics for pruning
- :doc:`configuration` - Detailed configuration options
