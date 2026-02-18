Experiments Guide
=================

This guide covers the different types of experiments available in the alignment framework.

Overview
--------

The framework provides several experiment types for analyzing neural network alignment and pruning:

1. **General Alignment Experiment** - Comprehensive alignment analysis with multi-network support
2. **Layer-wise Pruning Experiments** - Analyze pruning effects on individual layers
3. **Global Pruning Experiments** - Apply uniform pruning across all layers
4. **Cascading Pruning Experiments** - Progressive pruning through network layers
5. **Eigenvector-based Pruning** - Use spectral properties for pruning decisions

General Alignment Experiment
----------------------------

The main experiment class that supports:

- Training single or multiple networks
- Computing alignment metrics during and after training
- Applying various pruning strategies
- Comprehensive analysis and visualization

.. code-block:: python

    from alignment.experiments import GeneralAlignmentExperiment, GeneralAlignmentConfig

    config = GeneralAlignmentConfig(
        experiment_name="mnist_alignment",
        dataset_name="mnist",
        model_name="mlp",
        hidden_sizes=[128, 64],
        num_epochs=10,
        compute_alignment=True,
        alignment_metrics=["rayleigh_quotient", "mutual_information_gaussian"]
    )

    experiment = GeneralAlignmentExperiment(config)
    results = experiment.run()

Multi-Network Analysis
^^^^^^^^^^^^^^^^^^^^^^

Train and analyze multiple networks in parallel:

.. code-block:: python

    config = GeneralAlignmentConfig(
        experiment_name="multi_network_study",
        num_networks=5,  # Train 5 networks
        dataset_name="mnist",
        model_name="cnn",
        num_epochs=20,
        compute_alignment=True
    )

    experiment = GeneralAlignmentExperiment(config)
    results = experiment.run()

    # Results include statistics across all networks
    print(f"Mean accuracy: {results['mean_accuracy']}")
    print(f"Std accuracy: {results['std_accuracy']}")

Pruning Experiments
-------------------

Layer-wise Pruning
^^^^^^^^^^^^^^^^^^

Analyze the effect of pruning individual layers:

.. code-block:: python

    from alignment.pruning.experiments import LayerIsolatedPruningExperiment, LayerIsolatedConfig

    config = LayerIsolatedConfig(
        experiment_name="layer_analysis",
        dataset_name="mnist",
        model_name="mlp",
        hidden_sizes=[128, 64],
        pruning_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
        pruning_strategy="magnitude"
    )

    experiment = LayerIsolatedPruningExperiment(config)
    results = experiment.run()

Global Pruning
^^^^^^^^^^^^^^

Apply the same pruning rate across all layers:

.. code-block:: python

    from alignment.pruning.experiments import GlobalDropoutExperiment, GlobalDropoutConfig

    config = GlobalDropoutConfig(
        experiment_name="global_pruning",
        dataset_name="cifar10",
        model_name="resnet18",
        dropout_rates=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
        dropout_structure="magnitude"  # or "random", "gradient"
    )

    experiment = GlobalDropoutExperiment(config)
    results = experiment.run()

Cascading Layer Pruning
^^^^^^^^^^^^^^^^^^^^^^^

Progressive pruning that cascades through the network:

.. code-block:: python

    from alignment.pruning.experiments import CascadingLayerPruningExperiment, CascadingConfig

    config = CascadingConfig(
        experiment_name="cascading_analysis",
        dataset_name="mnist",
        model_name="mlp",
        cascade_direction="forward",  # or "backward"
        pruning_ratios=[0.1, 0.2, 0.3, 0.4, 0.5]
    )

    experiment = CascadingLayerPruningExperiment(config)
    results = experiment.run()

Eigenvector-based Pruning
^^^^^^^^^^^^^^^^^^^^^^^^^

Use eigendecomposition for pruning decisions:

.. code-block:: python

    from alignment.pruning.experiments import EigenvectorDropoutExperiment, EigenvectorConfig

    config = EigenvectorConfig(
        experiment_name="eigenvector_pruning",
        dataset_name="mnist",
        model_name="mlp",
        num_components=10,  # Number of eigenvectors to keep
        pruning_ratios=[0.1, 0.3, 0.5, 0.7]
    )

    experiment = EigenvectorDropoutExperiment(config)
    results = experiment.run()

Configuration Options
---------------------

Common configuration parameters across experiments:

**Model Configuration:**

- ``model_name``: "mlp", "cnn", "resnet18", etc.
- ``hidden_sizes``: List of hidden layer sizes (for MLP)
- ``activation``: Activation function ("relu", "tanh", etc.)

**Training Configuration:**

- ``num_epochs``: Number of training epochs
- ``batch_size``: Batch size for training
- ``learning_rate``: Learning rate
- ``optimizer``: Optimizer type ("adam", "sgd", etc.)

**Alignment Configuration:**

- ``compute_alignment``: Whether to compute alignment metrics
- ``alignment_metrics``: List of metrics to compute
- ``alignment_layers``: Which layers to analyze

**Pruning Configuration:**

- ``pruning_strategy``: "magnitude", "gradient", "random", "alignment"
- ``pruning_ratios``: List of pruning ratios to test
- ``structured_pruning``: Whether to use structured pruning

Running Experiments
-------------------

From Configuration Files
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python scripts/run_experiment.py --config configs/my_experiment.yaml

From Python
^^^^^^^^^^^

.. code-block:: python

    from alignment.experiments import create_experiment_from_config
    import yaml

    # Load configuration
    with open("configs/my_experiment.yaml", "r") as f:
        config_dict = yaml.safe_load(f)

    # Create and run experiment
    experiment = create_experiment_from_config(config_dict)
    results = experiment.run()

Analyzing Results
-----------------

All experiments return a results dictionary containing:

- Training metrics (loss, accuracy over time)
- Final model performance
- Alignment metrics (if computed)
- Pruning analysis (for pruning experiments)
- Visualizations and plots

.. code-block:: python

    # Access results
    results = experiment.run()

    # Training history
    train_loss = results['training_history']['train_loss']
    val_accuracy = results['training_history']['val_accuracy']

    # Alignment metrics
    if 'alignment_metrics' in results:
        rq_scores = results['alignment_metrics']['rayleigh_quotient']
        mi_scores = results['alignment_metrics']['mutual_information']

    # Pruning results
    if 'pruning_results' in results:
        for ratio, metrics in results['pruning_results'].items():
            print(f"Pruning {ratio}: Accuracy = {metrics['accuracy']}")

Visualization
-------------

The framework automatically generates visualizations:

- Training curves
- Alignment metric evolution
- Pruning performance plots
- Layer-wise analysis

Plots are saved to the experiment output directory and can be customized through configuration.

Best Practices
--------------

1. **Start Small**: Test with small models and datasets first
2. **Use Checkpointing**: Enable model checkpointing for long experiments
3. **Monitor Memory**: Some alignment metrics are memory-intensive
4. **Reproducibility**: Always set seeds for reproducible results
5. **Incremental Analysis**: Start with few pruning ratios, then refine

Advanced Features
-----------------

Custom Metrics
^^^^^^^^^^^^^^

Add custom alignment metrics:

.. code-block:: python

    from alignment.metrics import register_metric

    @register_metric("my_custom_metric")
    def my_metric(model, dataloader, device):
        # Implement your metric
        return metric_value

Custom Pruning Strategies
^^^^^^^^^^^^^^^^^^^^^^^^^

Implement custom pruning strategies:

.. code-block:: python

    from alignment.pruning.strategies import BasePruningStrategy

    class MyPruningStrategy(BasePruningStrategy):
        def compute_importance_scores(self, model, dataloader):
            # Implement importance scoring
            return scores

Parallel Execution
^^^^^^^^^^^^^^^^^^

For multi-network experiments, parallel execution is automatic when ``num_networks > 1``.

See Also
--------

- :doc:`configuration` - Detailed configuration options
- :doc:`metrics` - Available alignment metrics
- :doc:`pruning` - Pruning strategies and concepts
