Experiments Guide
=================

This guide covers how to run and configure experiments in the alignment framework.

Overview
--------

The framework provides several built-in experiments for studying neural network pruning and alignment:

- **Progressive Dropout**: Gradually increases dropout during training
- **Eigenvector-based Pruning**: Prunes based on eigenvector analysis
- **Layer-wise Pruning**: Isolates pruning effects to specific layers
- **Cascading Pruning**: Sequentially prunes layers

Progressive Dropout Experiment
------------------------------

Progressive dropout gradually increases the dropout rate during training, allowing the network to adapt to increasing sparsity.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from alignment.experiments.progressive_dropout import ProgressiveDropoutExperiment
   from alignment.experiments.base import ExperimentConfig

   # Configure experiment
   config = ExperimentConfig(
       experiment_name="progressive_dropout_test",
       model_name="resnet18",
       dataset="cifar10",
       
       # Progressive dropout specific
       initial_dropout=0.0,
       final_dropout=0.9,
       warmup_epochs=10,
       increase_epochs=40,
       
       # Training parameters
       epochs=100,
       batch_size=128,
       learning_rate=0.1
   )

   # Create and run experiment
   experiment = ProgressiveDropoutExperiment(config)
   results = experiment.run()

Results Structure
~~~~~~~~~~~~~~~~~

.. code-block:: python

   {
       'dropout_fractions': [0.0, 0.1, 0.2, ...],
       'accuracies': {
           'strategy_name': [acc1, acc2, ...],
           ...
       },
       'metrics': {
           'layer_name': {
               'metric_name': scores,
               ...
           }
       }
   }

Pruning Strategies
~~~~~~~~~~~~~~~~~~

The experiment supports multiple pruning strategies:

- **magnitude**: Prune neurons with smallest weight magnitudes
- **gradient**: Prune based on gradient information
- **random**: Random pruning (baseline)
- **metric-based**: Use alignment metrics (RQ, MI) for pruning

Eigenvector-based Dropout
-------------------------

This experiment uses eigenvector analysis to determine which neurons to drop based on their alignment with principal components.

Configuration
~~~~~~~~~~~~~

.. code-block:: python

   from alignment.experiments.eigenvector import EigenvectorDropoutExperiment

   config = ExperimentConfig(
       experiment_name="eigenvector_pruning",
       
       # Eigenvector specific
       drop_percentage=0.5,
       eigenvector_threshold=0.1,
       use_magnitude_weighting=True,
       
       # Metrics to track
       metrics=["rayleigh_quotient", "mutual_information"],
       save_frequency=10
   )

   experiment = EigenvectorDropoutExperiment(config)
   results = experiment.run()

Key Parameters
~~~~~~~~~~~~~~

- ``drop_percentage``: Percentage of neurons to drop
- ``eigenvector_threshold``: Threshold for eigenvector analysis
- ``use_magnitude_weighting``: Whether to use magnitude weighting

Layer-isolated Pruning
----------------------

This experiment applies pruning to specific layers while keeping others intact, useful for studying layer-specific effects.

Usage
~~~~~

.. code-block:: python

   from alignment.experiments.layer_isolated import LayerIsolatedPruningExperiment

   config = ExperimentConfig(
       experiment_name="layer_isolation_study",
       
       # Layer isolation settings
       target_layers=["layer1.0.conv1", "layer2.0.conv1"],
       pruning_rates=[0.3, 0.5, 0.7],
       isolation_mode="sequential",  # or "parallel"
       
       # Pruning strategy
       pruning_method="magnitude",
       fine_tune_epochs=10
   )

   experiment = LayerIsolatedPruningExperiment(config)
   results = experiment.run()

Results Analysis
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze layer sensitivity
   layer_results = results['layer_results']
   for layer_name, layer_data in layer_results.items():
       print(f"\nLayer: {layer_name}")
       for pct, acc in zip(layer_data['pruning_percentages'], 
                          layer_data['accuracies']):
           print(f"  {pct*100}% pruned: {acc:.2f}% accuracy")

Cascading Layer Pruning
-----------------------

Cascading pruning sequentially prunes layers, studying how pruning propagates through the network.

Configuration
~~~~~~~~~~~~~

.. code-block:: python

   from alignment.experiments.cascading import CascadingLayerPruningExperiment

   config = ExperimentConfig(
       experiment_name="cascading_pruning",
       
       # Cascading specific
       layer_order=["conv1", "conv2", "fc1"],  # Order of pruning
       cascade_threshold=0.01,  # Min activation threshold
       pruning_per_layer=0.3,
       
       # Analysis options
       track_information_flow=True,
       save_intermediate_models=True
   )

   experiment = CascadingLayerPruningExperiment(config)
   results = experiment.run()

Running Multiple Experiments
----------------------------

Using ExperimentRunner
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from alignment.experiments.runner import ExperimentRunner
   from alignment.experiments.base import ExperimentConfig

   # Define multiple configurations
   configs = [
       ExperimentConfig(
           experiment_name="exp1",
           experiment_type="progressive_dropout",
           initial_dropout=0.0,
           final_dropout=0.5
       ),
       ExperimentConfig(
           experiment_name="exp2", 
           experiment_type="eigenvector",
           drop_percentage=0.3
       )
   ]

   # Run all experiments
   runner = ExperimentRunner(
       configs=configs,
       parallel=True,  # Run in parallel
       num_workers=4
   )

   all_results = runner.run_all()

Configuration Options
---------------------

Essential Parameters
~~~~~~~~~~~~~~~~~~~~

- ``experiment_name``: Experiment identifier
- ``experiment_type``: Type of experiment
- ``model_name``: Model architecture to use
- ``dataset``: Dataset for evaluation
- ``device``: Computing device (cuda/cpu)
- ``seed``: Random seed for reproducibility

Model Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model_config = {
       "hidden_dims": [300, 200, 100],  # For MLP
       "conv_channels": [32, 64],       # For CNN
       "dropout_rate": 0.5,
       "activation": "relu"
   }

Training Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   training_config = {
       "epochs": 10,
       "learning_rate": 0.001,
       "optimizer": "adam",
       "train_before_dropout": True
   }

Metric Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   metric_configs = {
       "rayleigh_quotient": {
           "scale_by_norm": False,
           "aggregation_op": "mean"
       },
       "mutual_information": {
           "estimation_method": "gaussian"
       }
   }

Advanced Features
-----------------

Custom Pruning Functions
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def custom_pruning_fn(weights, scores, pruning_fraction):
       """Custom pruning logic"""
       threshold = torch.quantile(scores, pruning_fraction)
       mask = scores > threshold
       return weights * mask.unsqueeze(1)

   config.custom_pruning_fn = custom_pruning_fn

Callbacks
~~~~~~~~~

.. code-block:: python

   def on_pruning_step(experiment, step, metrics):
       """Called after each pruning step"""
       print(f"Step {step}: {metrics}")

   config.callbacks = [on_pruning_step]

Checkpointing
~~~~~~~~~~~~~

.. code-block:: python

   config.checkpoint_interval = 1000  # Save every 1000 steps
   config.checkpoint_dir = "./checkpoints"
   config.save_best = True  # Save best performing model

Best Practices
--------------

1. **Start with Small Models**: Test configurations on small models first
2. **Use Appropriate Batch Sizes**: Balance memory usage and training stability
3. **Set Random Seeds**: Ensure reproducibility across runs
4. **Monitor Metrics**: Track multiple metrics for comprehensive analysis
5. **Save Intermediate Results**: Enable checkpointing for long experiments

Troubleshooting
---------------

Out of Memory Errors
~~~~~~~~~~~~~~~~~~~~

- Reduce batch size
- Use gradient accumulation
- Enable CPU offloading for large models

Slow Experiments
~~~~~~~~~~~~~~~~

- Use GPU acceleration
- Reduce number of pruning steps
- Enable parallel data loading

Inconsistent Results
~~~~~~~~~~~~~~~~~~~~

- Set fixed random seeds
- Disable non-deterministic operations
- Verify data loading consistency 