Quickstart Guide
================

This guide will get you up and running with LossLens in minutes.

.. contents:: Table of Contents
   :local:
   :depth: 2

Installation
------------

Basic Installation
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/KempnerInstitute/alignment.git
   cd alignment

   # Install the package
   pip install -e .

Full Installation
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install with all optional dependencies
   pip install -e .[all]

   # Or install specific extras
   pip install -e .[train]    # Training and large-model utilities
   pip install -e .[all]      # Development and training extras
   pip install -e .[docs]     # Documentation building

Your First Experiment
---------------------

1. Basic Metric Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from alignment.metrics import RayleighQuotient

   # Create some sample data
   inputs = torch.randn(100, 512)    # 100 samples, 512 features
   weights = torch.randn(256, 512)   # 256 neurons, 512 input features

   # Compute Rayleigh Quotient
   rq = RayleighQuotient()
   scores = rq.compute(inputs=inputs, weights=weights)

   print(f"RQ scores shape: {scores.shape}")  # (256,)
   print(f"Mean RQ: {scores.mean():.4f}")

2. Using a Pre-trained Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from alignment.models import ModelWrapper
   from alignment.metrics import get_metric
   import torchvision.models as models

   # Load a pre-trained ResNet
   model = models.resnet18(pretrained=True)

   # Wrap it for metric computation
   wrapped_model = ModelWrapper(
       model,
       tracked_layers=["layer1.0.conv1", "layer2.0.conv1", "layer3.0.conv1"]
   )

   # Create sample input
   x = torch.randn(32, 3, 224, 224)

   # Forward pass and collect activations
   output, activations = wrapped_model.forward_with_activations(x)

   # Compute metrics on specific layer
   metric = get_metric("rayleigh_quotient")
   layer_name = "layer1.0.conv1"
   scores = metric.compute(
       inputs=activations[layer_name],
       weights=wrapped_model.get_layer_weights()[layer_name]
   )

3. Running a Pruning Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from alignment.experiments import ProgressiveDropoutExperiment
   from alignment.experiments.base import ExperimentConfig

   # Configure experiment
   config = ExperimentConfig(
       name="resnet_pruning_demo",
       model_name="resnet18",
       dataset_name="cifar10",

       # Metrics to track
       metrics=["rayleigh_quotient", "mutual_information"],

       # Pruning settings
       dropout_rates=[0.0, 0.3, 0.5, 0.7, 0.9],
       pruning_strategy="low",  # Prune low RQ neurons

       # Training settings
       epochs=10,
       batch_size=128,
       learning_rate=0.1
   )

   # Run experiment
   experiment = ProgressiveDropoutExperiment(config)
   results = experiment.run()

   # Analyze results
   print("Accuracy at different sparsity levels:")
   for rate, acc in results["accuracy"].items():
       print(f"  Dropout {rate}: {acc:.2%}")

Common Use Cases
----------------

Comparing Network Architectures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from alignment.experiments import ExperimentRunner
   from alignment.experiments.base import ExperimentConfig

   # Define experiments for different architectures
   architectures = ["resnet18", "vgg16", "efficientnet_b0"]
   configs = []

   for arch in architectures:
       configs.append(ExperimentConfig(
           name=f"compare_{arch}",
           model_name=arch,
           dataset_name="cifar10",
           metrics=["rayleigh_quotient", "cka", "spectral_analysis"],
           epochs=50
       ))

   # Run all experiments
   runner = ExperimentRunner(configs, parallel=True)
   results = runner.run()

   # Generate comparison report
   runner.generate_report("architecture_comparison.html")

Analyzing Layer Importance
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from alignment.experiments import LayerIsolatedPruningExperiment
   from alignment.analysis import LayerImportanceAnalyzer

   config = ExperimentConfig(
       name="layer_importance_analysis",
       model_name="resnet50",
       dataset_name="imagenet",
       metrics=["rayleigh_quotient"],
       dropout_rates=[0.0, 0.5, 0.9],
       isolation_mode="sequential"
   )

   # Run layer-isolated pruning
   experiment = LayerIsolatedPruningExperiment(config)
   results = experiment.run()

   # Analyze layer importance
   analyzer = LayerImportanceAnalyzer(results)
   importance_scores = analyzer.compute_importance()
   analyzer.plot_importance_heatmap()

Custom Metric Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from alignment.metrics.base import BaseMetric
   from alignment.core import register_metric
   import torch

   @register_metric("gradient_alignment")
   class GradientAlignment(BaseMetric):
       """Measures alignment between weights and gradients."""

       def __init__(self):
           super().__init__(name="gradient_alignment")

       @property
       def requires_inputs(self) -> bool:
           return False

       @property
       def requires_weights(self) -> bool:
           return True

       @property
       def requires_outputs(self) -> bool:
           return False

       def compute(self, weights=None, gradients=None, **kwargs):
           # Compute cosine similarity between weights and gradients
           w_flat = weights.view(weights.size(0), -1)
           g_flat = gradients.view(gradients.size(0), -1)

           # Normalize
           w_norm = torch.nn.functional.normalize(w_flat, dim=1)
           g_norm = torch.nn.functional.normalize(g_flat, dim=1)

           # Cosine similarity
           alignment = (w_norm * g_norm).sum(dim=1)

           return alignment

   # Use the custom metric
   metric = get_metric("gradient_alignment")
   scores = metric.compute(weights=weights, gradients=gradients)

Working with Configuration Files
--------------------------------

Creating a Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # experiment_config.yaml
   name: my_experiment
   description: Testing different pruning strategies

   # Model settings
   model_name: resnet18
   pretrained: true

   # Dataset settings
   dataset_name: cifar10
   batch_size: 128
   num_workers: 4

   # Training settings
   epochs: 100
   learning_rate: 0.1
   optimizer: sgd
   lr_schedule: cosine

   # Metrics to compute
   metrics:
     - rayleigh_quotient
     - mutual_information
     - cka

   metric_configs:
     rayleigh_quotient:
       scale_by_norm: true
     cka:
       kernel: rbf
       sigma: 1.0

   # Pruning settings
   dropout_rates: [0.0, 0.2, 0.4, 0.6, 0.8]
   pruning_strategy: magnitude
   pruning_mode: global_joint

Loading and Running
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from alignment.infrastructure.configuration import load_config
   from alignment.experiments import create_experiment

   # Load configuration
   config = load_config("experiment_config.yaml")

   # Create and run experiment
   experiment = create_experiment(config)
   results = experiment.run()

   # Save results
   experiment.save_results("results/my_experiment/")

Visualization and Analysis
--------------------------

Plotting Metrics
~~~~~~~~~~~~~~~~

.. code-block:: python

   from alignment.analysis import MetricVisualizer

   # Load results
   results = load_results("results/my_experiment/")

   # Create visualizer
   viz = MetricVisualizer(results)

   # Plot metric evolution
   viz.plot_metric_vs_sparsity(
       metric="rayleigh_quotient",
       layers=["layer1.0.conv1", "layer2.0.conv1"],
       save_path="rq_vs_sparsity.png"
   )

   # Plot layer-wise comparison
   viz.plot_layer_comparison(
       metrics=["rayleigh_quotient", "mutual_information"],
       sparsity=0.5,
       save_path="layer_comparison.png"
   )

Generating Reports
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from alignment.analysis import ReportGenerator

   # Generate comprehensive report
   generator = ReportGenerator(results)
   generator.generate_html_report(
       output_path="report.html",
       include_plots=True,
       include_tables=True,
       include_config=True
   )

Tips and Tricks
---------------

1. **Memory Management**

   .. code-block:: python

      # Use CPU for large matrix operations
      config = ExperimentConfig(
          force_cpu_for_large_metric_ops=True,
          cpu_threshold=1e7
      )

2. **Faster Experiments**

   .. code-block:: python

      # Reduce computation for quick tests
      config = ExperimentConfig(
          epochs=5,  # Fewer epochs
          eval_batches=10,  # Evaluate on subset
          metrics=["rayleigh_quotient"],  # Fewer metrics
          dropout_rates=[0.0, 0.5, 0.9]  # Fewer rates
      )

3. **Distributed Training**

   .. code-block:: python

      # Enable distributed training
      config = ExperimentConfig(
          distributed=True,
          backend="nccl",
          world_size=4
      )

4. **Debugging**

   .. code-block:: python

      # Enable debug logging
      import logging
      logging.basicConfig(level=logging.DEBUG)

      # Use smaller dataset
      config = ExperimentConfig(
          dataset_name="mnist",  # Smaller than CIFAR
          batch_size=32,
          debug_mode=True
      )

Next Steps
----------

- :doc:`/user_guide/experiments` - Detailed experiment guide
- :doc:`/user_guide/metrics` - All available metrics
- :doc:`/user_guide/configuration` - Configuration options
- Repository examples and configs - Advanced examples
- Top-level README - Current API entry points

Common Issues
-------------

**CUDA Out of Memory**

.. code-block:: python

   # Reduce batch size
   config.batch_size = 32

   # Force CPU for metrics
   config.force_cpu_for_large_metric_ops = True

   # Use gradient accumulation
   config.gradient_accumulation = 4

**Slow Metric Computation**

.. code-block:: python

   # Use faster metrics
   config.metrics = ["rayleigh_quotient"]  # Fast
   # Avoid: ["pid", "knn_mi"]  # Slow

   # Compute on subset
   config.metric_sample_size = 1000

**Import Errors**

.. code-block:: bash

   # Ensure you're in the right directory
   cd alignment

   # Reinstall in development mode
   pip install -e .

   # Check installation
   python -c "import alignment; print(alignment.__version__)"
