Pruning Guide
=============

The alignment framework provides comprehensive pruning capabilities for neural networks,
including magnitude-based, gradient-based, random, and parallel pruning strategies.

Basic Usage
-----------

Simple pruning example:

.. code-block:: python

    from alignment.pruning import get_pruning_strategy, PruningConfig
    
    # Basic magnitude pruning
    strategy = get_pruning_strategy('magnitude')
    mask = strategy.prune(model.fc1, amount=0.5)
    
    # With configuration
    config = PruningConfig(amount=0.7, structured=True)
    strategy = get_pruning_strategy('magnitude', config=config)

Pruning Modes
-------------

The framework now supports different pruning modes:

- **low**: Prune weights with the lowest importance scores (default)
- **high**: Prune weights with the highest importance scores
- **random**: Prune weights randomly

.. code-block:: python

    from alignment.pruning import PruningConfig
    from alignment.pruning.strategies import MagnitudePruning
    
    # Prune low-magnitude weights (default)
    config_low = PruningConfig(amount=0.5, pruning_mode='low')
    strategy_low = MagnitudePruning(config_low)
    
    # Prune high-magnitude weights
    config_high = PruningConfig(amount=0.5, pruning_mode='high')
    strategy_high = MagnitudePruning(config_high)

Available Strategies
--------------------

Magnitude-based Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **MagnitudePruning**: Basic magnitude-based pruning
- **IterativeMagnitudePruning**: Gradual pruning with fine-tuning
- **GlobalMagnitudePruning**: Global pruning across all layers

.. code-block:: python

    # Iterative pruning with fine-tuning
    from alignment.pruning.strategies import IterativeMagnitudePruning
    
    config = PruningConfig(
        amount=0.9,
        iterations=10,
        fine_tune_epochs=5
    )
    strategy = IterativeMagnitudePruning(config)
    results = strategy.iterative_prune(model, dataloader, optimizer, criterion)

Gradient-based Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~

- **GradientPruning**: Prune based on gradient magnitudes
- **FisherPruning**: Use Fisher information for importance
- **MomentumPruning**: Consider gradient momentum

.. code-block:: python

    from alignment.pruning.strategies import FisherPruning
    
    strategy = FisherPruning()
    # Requires inputs to compute gradients
    mask = strategy.prune(model.fc1, inputs=sample_batch)

Random Strategies
~~~~~~~~~~~~~~~~~

- **RandomPruning**: Uniform random pruning
- **LayerwiseRandomPruning**: Layer-specific random pruning
- **BernoulliPruning**: Probabilistic pruning

Parallel Pruning Strategies
---------------------------

The framework provides several strategies for applying multiple pruning modes simultaneously:

ParallelModePruning
~~~~~~~~~~~~~~~~~~~

Apply multiple pruning modes in parallel and analyze their effects:

.. code-block:: python

    from alignment.pruning.strategies import ParallelModePruning
    
    # Apply low, high, and random pruning simultaneously
    strategy = ParallelModePruning(
        modes=['low', 'high', 'random'],
        base_strategy='magnitude'
    )
    
    result = strategy.prune_parallel(layer, amount=0.5)
    
    # Access individual masks
    low_mask = result.masks['low']
    high_mask = result.masks['high']
    random_mask = result.masks['random']
    
    # Analyze sparsity
    for mode, sparsity in result.sparsities.items():
        print(f"{mode}: {sparsity:.2%} sparse")
    
    # Combine masks
    combined_union = strategy.combine_masks(result.masks, method='union')
    combined_intersection = strategy.combine_masks(result.masks, method='intersection')
    combined_majority = strategy.combine_masks(result.masks, method='majority')

TensorizedPruning
~~~~~~~~~~~~~~~~~

Efficient GPU-optimized computation of multiple pruning configurations:

.. code-block:: python

    from alignment.pruning.strategies import TensorizedPruning
    
    strategy = TensorizedPruning()
    
    # Compute pruning tensor: [num_modes, num_amounts, *weight_shape]
    pruning_tensor = strategy.compute_pruning_tensor(
        layer,
        modes=['low', 'high', 'random'],
        amounts=[0.1, 0.3, 0.5, 0.7, 0.9]
    )
    
    # Analyze pruning patterns
    analysis = strategy.analyze_pruning_patterns(pruning_tensor)
    print("Sparsity progression:", analysis['sparsity_progression'])
    print("Mode overlap:", analysis['mode_overlap'])

AsyncParallelPruning
~~~~~~~~~~~~~~~~~~~~

Prune multiple modules in parallel using CPU cores:

.. code-block:: python

    from alignment.pruning.strategies import AsyncParallelPruning
    
    strategy = AsyncParallelPruning()
    
    # Prune multiple layers with different amounts
    modules = [model.layer1, model.layer2, model.layer3]
    results = strategy.prune_modules_parallel(
        modules,
        amounts=[0.5, 0.6, 0.7],
        modes=['low', 'high'],
        max_workers=4
    )
    
    # Each result contains masks for all modes
    for i, layer_results in enumerate(results):
        print(f"Layer {i}:")
        for mode, mask in layer_results.items():
            sparsity = (mask == 0).float().mean()
            print(f"  {mode}: {sparsity:.2%} sparse")

Advanced Usage
--------------

Structured Pruning
~~~~~~~~~~~~~~~~~~

Remove entire channels or filters:

.. code-block:: python

    config = PruningConfig(
        amount=0.5,
        structured=True
    )
    strategy = get_pruning_strategy('magnitude', config=config)
    mask = strategy.prune(conv_layer)

Custom Pruning Strategy
~~~~~~~~~~~~~~~~~~~~~~~

Create your own pruning strategy:

.. code-block:: python

    from alignment.pruning.base import BasePruningStrategy
    
    class MyCustomPruning(BasePruningStrategy):
        def compute_importance_scores(self, module, inputs=None, **kwargs):
            # Your custom importance computation
            return module.weight.abs() * custom_metric

Pruning Experiments
-------------------

The framework includes pre-built experiments:

.. code-block:: python

    from alignment.pruning.experiments import (
        ProgressiveDropoutExperiment,
        LayerIsolatedPruningExperiment
    )
    
    # Progressive dropout experiment
    experiment = ProgressiveDropoutExperiment(
        model=model,
        dataset='cifar10',
        pruning_rates=[0.1, 0.3, 0.5, 0.7, 0.9]
    )
    results = experiment.run()

Visualization
-------------

Visualize pruning patterns:

.. code-block:: python

    import matplotlib.pyplot as plt
    from alignment.analysis.visualization import PruningVisualizer
    
    visualizer = PruningVisualizer()
    
    # Compare different pruning modes
    fig = visualizer.compare_pruning_modes(
        model,
        modes=['low', 'high', 'random'],
        amount=0.5
    )
    
    # Show sparsity heatmap
    fig = visualizer.sparsity_heatmap(pruned_model)

Best Practices
--------------

1. **Start with small pruning amounts**: Begin with 10-30% pruning and increase gradually
2. **Use iterative pruning**: For high sparsity (>70%), use iterative pruning with fine-tuning
3. **Monitor performance**: Track accuracy degradation during pruning
4. **Compare strategies**: Use parallel pruning to compare different approaches
5. **Consider structured pruning**: For deployment, structured pruning may be more efficient

Example: Complete Pruning Pipeline
----------------------------------

.. code-block:: python

    import torch
    from alignment.pruning import PruningConfig
    from alignment.pruning.strategies import ParallelModePruning
    from alignment.training import Trainer
    
    # 1. Setup
    model = load_model()
    config = PruningConfig(amount=0.8, iterations=5)
    
    # 2. Compare pruning modes
    strategy = ParallelModePruning(config=config)
    result = strategy.prune_parallel(model.backbone)
    
    # 3. Select best mode based on validation
    best_mode = evaluate_modes(result.masks, val_loader)
    
    # 4. Apply selected pruning
    model.backbone.weight.data *= result.masks[best_mode]
    
    # 5. Fine-tune
    trainer = Trainer(model, train_loader, val_loader)
    trainer.fine_tune(epochs=10)
    
    # 6. Make pruning permanent
    strategy.remove_pruning(model.backbone)

See Also
--------

- :doc:`/api/pruning` - Complete API reference
- :doc:`experiments` - Using pruning in experiments
- ``examples/pruning_parallel_demo.py`` - Parallel pruning demonstration 