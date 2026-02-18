Pruning Strategies Guide
========================

This guide documents all pruning strategies available in the alignment framework and their use cases.

Overview
--------

Pruning is a technique for reducing neural network size by removing parameters while maintaining performance. The alignment framework provides several pruning strategies to analyze how network sparsity affects alignment metrics.

Available Pruning Strategies
----------------------------

1. Magnitude-Based Pruning
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Module**: :mod:`alignment.pruning.strategies.magnitude`

**Classes**:

- :class:`MagnitudePruning`: Basic magnitude pruning
- :class:`GlobalMagnitudePruning`: Global magnitude pruning across all layers
- :class:`IterativeMagnitudePruning`: Gradual pruning with fine-tuning

**Description**: Removes weights with the smallest absolute values.

**Theory**: Small magnitude weights contribute less to the network's output and can be removed with minimal impact.

**Usage**:

.. code-block:: python

   from alignment.pruning import get_pruning_strategy

   # Basic magnitude pruning
   strategy = get_pruning_strategy("magnitude")
   mask = strategy.compute_mask(layer.weight, amount=0.5)
   strategy.apply_mask(layer, mask)

   # Global magnitude pruning
   strategy = get_pruning_strategy("global_magnitude")
   masks = strategy.compute_masks_for_model(model, amount=0.5)

**Parameters**:

- ``amount``: Fraction of weights to prune (0-1)
- ``structured``: If True, prunes entire channels/filters
- ``dim``: Dimension for structured pruning (0=output, 1=input)

2. Random Pruning
~~~~~~~~~~~~~~~~~

**Module**: :mod:`alignment.pruning.strategies.random`

**Classes**:

- :class:`RandomPruning`: Uniform random pruning
- :class:`LayerwiseRandomPruning`: Random pruning with per-layer control
- :class:`BernoulliPruning`: Probabilistic pruning with Bernoulli sampling

**Description**: Randomly removes weights regardless of their values.

**Theory**: Used as a baseline to compare against informed pruning strategies.

**Usage**:

.. code-block:: python

   strategy = get_pruning_strategy("random")
   mask = strategy.compute_mask(layer.weight, amount=0.5)

3. Gradient-Based Pruning
~~~~~~~~~~~~~~~~~~~~~~~~~

**Module**: :mod:`alignment.pruning.strategies.gradient`

**Classes**:

- :class:`GradientPruning`: Basic gradient magnitude pruning
- :class:`FisherPruning`: Fisher information-based pruning
- :class:`MomentumPruning`: Momentum-aware gradient pruning

**Description**: Prunes weights based on gradient information.

**Theory**: Weights with small gradients have less impact on the loss function.

**Usage**:

.. code-block:: python

   strategy = get_pruning_strategy("gradient")

   # Requires gradient computation
   loss.backward()
   mask = strategy.compute_mask(
       layer.weight,
       amount=0.5,
       gradient=layer.weight.grad
   )

**Requirements**: Requires gradient computation via backpropagation.

4. Structured Pruning
~~~~~~~~~~~~~~~~~~~~~

All strategies support structured pruning by setting ``structured=True``:

.. code-block:: python

   from alignment.pruning import PruningConfig

   config = PruningConfig(
       strategy="magnitude",
       amount=0.3,
       structured=True,
       dim=0  # Prune output channels
   )

   strategy = get_pruning_strategy(config.strategy)
   mask = strategy.compute_mask(layer.weight, **config.to_dict())

5. Iterative Pruning
~~~~~~~~~~~~~~~~~~~~

The framework provides iterative pruning through dedicated strategies:

.. code-block:: python

   from alignment.pruning.strategies.magnitude import IterativeMagnitudePruning

   strategy = IterativeMagnitudePruning(
       iterations=10,
       final_sparsity=0.9
   )

   for step in range(strategy.iterations):
       mask = strategy.compute_mask_for_iteration(
           layer.weight,
           iteration=step
       )
       strategy.apply_mask(layer, mask)

       # Fine-tune between iterations
       fine_tune(model, epochs=5)

Pruning Schedules
-----------------

Create pruning schedules for gradual sparsification:

.. code-block:: python

   from alignment.pruning.schedules import PolynomialSchedule, LinearSchedule

   # Polynomial schedule (recommended)
   schedule = PolynomialSchedule(
       initial_sparsity=0.0,
       final_sparsity=0.9,
       begin_step=1000,
       end_step=10000,
       power=3
   )

   # Get sparsity for current step
   current_sparsity = schedule(step=5000)

**Schedule Types**:

- ``LinearSchedule``: Linear interpolation
- ``PolynomialSchedule``: Smooth polynomial interpolation
- ``ExponentialSchedule``: Exponential decay
- ``CosineSchedule``: Cosine annealing

Best Practices
--------------

1. Choosing a Strategy
~~~~~~~~~~~~~~~~~~~~~~

- **Magnitude pruning**: Good default choice, simple and effective
- **Gradient-based**: When task-specific importance is crucial
- **Fisher pruning**: For second-order importance estimation
- **Structured**: When hardware efficiency is important

2. Pruning Amount
~~~~~~~~~~~~~~~~~

- Start with small amounts (10-30%) for initial experiments
- Most networks can handle 50-70% sparsity with minimal accuracy loss
- 90%+ sparsity is possible but requires careful tuning

3. Iterative vs One-Shot
~~~~~~~~~~~~~~~~~~~~~~~~

- **One-shot**: Fast, good for analysis
- **Iterative**: Better performance, allows adaptation

4. Fine-Tuning
~~~~~~~~~~~~~~

Always fine-tune after pruning for best results:

.. code-block:: python

   # Prune
   strategy = get_pruning_strategy("magnitude")
   mask = strategy.compute_mask(layer.weight, amount=0.5)
   strategy.apply_mask(layer, mask)

   # Fine-tune
   for epoch in range(fine_tune_epochs):
       train(model, train_loader, optimizer, criterion)

Utility Functions
-----------------

Check Sparsity
~~~~~~~~~~~~~~~

.. code-block:: python

   from alignment.pruning.utils import get_sparsity, get_model_sparsity

   # Layer sparsity
   sparsity = get_sparsity(layer)

   # Model sparsity
   model_sparsity = get_model_sparsity(model)

Remove Pruning
~~~~~~~~~~~~~~

.. code-block:: python

   from alignment.pruning.utils import remove_pruning

   # Makes pruning permanent and removes masks
   remove_pruning(layer)

Integration with Alignment Metrics
----------------------------------

Pruning affects alignment metrics in various ways:

1. **Rayleigh Quotient**: May increase as unimportant directions are removed
2. **Mutual Information**: Can decrease if information pathways are disrupted
3. **Spectral Properties**: Eigenvalue distribution changes with sparsity

Example analysis:

.. code-block:: python

   from alignment.experiments import GeneralAlignmentExperiment

   # Track how metrics change with pruning
   config = {
       "model_name": "resnet18",
       "dataset": "cifar10",
       "metrics": ["rayleigh_quotient", "mutual_information_gaussian", "spectral_gap"],
       "pruning_amounts": [0.0, 0.3, 0.5, 0.7, 0.9],
       "pruning_strategy": "magnitude"
   }

   experiment = GeneralAlignmentExperiment(config)
   results = experiment.run()

   # Visualize metric changes vs sparsity
   experiment.visualize_results()

Common Issues and Solutions
---------------------------

Issue: Performance Degrades Significantly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution**: Use iterative pruning with fine-tuning between steps

Issue: Structured Pruning Removes Important Channels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution**: Use custom importance scores based on your task

Issue: Pruning Masks Not Persisting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution**: Ensure hooks are properly registered, or make pruning permanent

Issue: Memory Not Reduced After Pruning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution**: Use structured pruning or sparse tensor formats

See Also
--------

- :doc:`/api/pruning` - Complete API reference
- :doc:`experiments` - Pruning experiments guide
- :doc:`/examples/pruning_experiments` - Example code
