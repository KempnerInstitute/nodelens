Configuration Guide
===================

This guide explains all configuration options available in the alignment framework.

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
--------

The alignment framework uses a hierarchical configuration system that allows fine-grained control over experiments, metrics, training, and analysis.

Configuration can be specified via:

1. **Python dictionaries/dataclasses**
2. **YAML files**
3. **Command-line arguments**
4. **Environment variables**

Basic Configuration
-------------------

Using Python
~~~~~~~~~~~~

.. code-block:: python

   from alignment.experiments.base import ExperimentConfig

   config = ExperimentConfig(
       name="my_experiment",
       model_name="resnet18",
       dataset_name="cifar10",
       metrics=["rayleigh_quotient", "mutual_information"],
       device="cuda"
   )

Using YAML
~~~~~~~~~~

.. code-block:: yaml

   # config.yaml
   name: my_experiment
   model_name: resnet18
   dataset_name: cifar10
   metrics:
     - rayleigh_quotient
     - mutual_information
   device: cuda

Loading configuration:

.. code-block:: python

   from alignment.infrastructure.configuration import load_config

   config = load_config("config.yaml")

Core Configuration Options
--------------------------

Experiment Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   ExperimentConfig(
       # Basic Information
       name: str,                    # Unique experiment identifier
       description: str = "",        # Human-readable description
       tags: List[str] = [],        # Tags for organization

       # Model Configuration
       model_name: str,             # Model architecture name
       model_config: Dict = {},     # Model-specific parameters
       pretrained: bool = False,    # Use pretrained weights
       checkpoint_path: str = None, # Load from checkpoint

       # Dataset Configuration
       dataset_name: str,           # Dataset name
       data_path: str = "./data",   # Data directory
       batch_size: int = 128,       # Batch size
       num_workers: int = 4,        # DataLoader workers
       pin_memory: bool = True,     # Pin memory for GPU

       # Metrics Configuration
       metrics: List[str] = [],     # Metrics to compute
       metric_configs: Dict = {},   # Per-metric configuration

       # Device Configuration
       device: str = "cuda",        # Device to use
       mixed_precision: bool = False, # Use AMP

       # Reproducibility
       seed: int = 42,              # Random seed
       deterministic: bool = True,  # Deterministic operations
   )

Model Configuration
~~~~~~~~~~~~~~~~~~~

**Built-in Models:**

.. code-block:: python

   # ResNet variants
   model_name = "resnet18"  # Also: resnet34, resnet50, resnet101, resnet152

   # VGG variants
   model_name = "vgg16"     # Also: vgg11, vgg13, vgg19

   # EfficientNet
   model_name = "efficientnet_b0"  # Also: b1-b7

   # Vision Transformer
   model_name = "vit_b_16"  # Also: vit_b_32, vit_l_16

   # Custom models
   model_name = "mlp"       # Multi-layer perceptron
   model_name = "cnn2p2"    # 2-conv 2-pool CNN

**Model-specific Configuration:**

.. code-block:: python

   # MLP configuration
   model_config = {
       "input_dim": 784,
       "hidden_dims": [512, 256, 128],
       "output_dim": 10,
       "activation": "relu",
       "dropout": 0.5,
       "batch_norm": True
   }

   # CNN configuration
   model_config = {
       "input_channels": 3,
       "conv_channels": [32, 64, 128],
       "kernel_sizes": [3, 3, 3],
       "pool_sizes": [2, 2, 2],
       "fc_dims": [256, 128],
       "output_dim": 10
   }

Dataset Configuration
~~~~~~~~~~~~~~~~~~~~~

**Built-in Datasets:**

.. code-block:: python

   # Vision datasets
   dataset_name = "cifar10"     # 10 classes, 32x32 images
   dataset_name = "cifar100"    # 100 classes, 32x32 images
   dataset_name = "mnist"       # Handwritten digits
   dataset_name = "fashion_mnist" # Fashion items
   dataset_name = "imagenet"    # Large-scale image classification

   # Custom datasets
   dataset_name = "custom"      # Requires custom_dataset_path

**Dataset-specific Options:**

.. code-block:: python

   # Data augmentation
   data_config = {
       "augmentation": {
           "random_crop": True,
           "random_flip": True,
           "color_jitter": True,
           "normalize": True
       },
       "validation_split": 0.1,  # Validation set size
       "stratified": True        # Stratified sampling
   }

Training Configuration
----------------------

Basic Training Options
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   training_config = {
       # Training duration
       "epochs": 100,
       "steps_per_epoch": None,     # None = full epoch

       # Optimization
       "optimizer": "sgd",          # sgd, adam, adamw, rmsprop
       "learning_rate": 0.1,
       "momentum": 0.9,             # For SGD
       "weight_decay": 1e-4,
       "betas": (0.9, 0.999),      # For Adam/AdamW

       # Learning rate schedule
       "lr_schedule": "cosine",     # cosine, step, exponential, none
       "lr_milestones": [30, 60, 90], # For step schedule
       "lr_gamma": 0.1,             # LR decay factor
       "warmup_epochs": 5,          # Linear warmup

       # Regularization
       "dropout": 0.5,
       "label_smoothing": 0.1,
       "mixup_alpha": 0.2,
       "cutmix_alpha": 1.0
   }

Advanced Training Options
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   advanced_training = {
       # Gradient clipping
       "gradient_clip_norm": 1.0,
       "gradient_clip_value": None,

       # Early stopping
       "early_stopping": True,
       "patience": 10,
       "min_delta": 1e-4,

       # Checkpointing
       "checkpoint_interval": 10,   # Epochs between checkpoints
       "save_best": True,
       "best_metric": "accuracy",   # Metric to monitor

       # Memory optimization
       "gradient_accumulation": 1,  # Accumulate gradients
       "gradient_checkpointing": False, # Trade compute for memory

       # Distributed training
       "distributed": False,
       "backend": "nccl",          # nccl, gloo
       "world_size": 1,
       "find_unused_parameters": False
   }

Metric Configuration
--------------------

Global Metric Options
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   metric_config = {
       # Computation options
       "force_cpu_for_large_ops": True,
       "cpu_threshold": 1e7,        # Matrix size threshold
       "batch_size": 1000,          # Metric computation batch size

       # Metric-specific defaults
       "scale_by_norm": False,      # For RQ
       "normalize": True,           # For MI
       "kernel": "linear",          # For CKA

       # CNN-specific
       "cnn_aggregation": "mean",   # mean, max, sum
       "per_channel": False         # Compute per channel
   }

Per-Metric Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   metric_configs = {
       "rayleigh_quotient": {
           "scale_by_norm": True,
           "relative": False,
           "epsilon": 1e-8,
           "force_cpu": True
       },

       "mutual_information": {
           "estimation_method": "gaussian",  # gaussian, knn, binning
           "num_samples": 1000,
           "normalize": True,
           "num_bins": 30              # For binning method
       },

       "cka": {
           "kernel": "rbf",            # linear, rbf
           "sigma": 1.0,               # RBF kernel width
           "threshold": 0.01           # Eigenvalue threshold
       },

       "pid": {
           "method": "broja",          # broja, barrett, williams
           "max_variables": 100,
           "num_samples": 5000
       }
   }

Pruning Configuration
---------------------

Pruning Strategy Options
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   pruning_config = {
       # Strategy selection
       "pruning_strategy": "magnitude",  # magnitude, gradient, fisher, random

       # Basic options
       "sparsity": 0.9,                 # Target sparsity
       "structured": False,             # Structured pruning
       "global_pruning": True,          # Global vs layer-wise

       # Iterative pruning
       "iterative": True,
       "n_iterations": 10,
       "recovery_epochs": 5,
       "sparsity_schedule": "linear",   # linear, exponential, polynomial

       # Advanced options
       "exclude_layers": ["fc"],        # Layers to exclude
       "min_sparsity_per_layer": 0.5,   # Minimum per-layer sparsity
       "importance_aggregation": "l2"   # l1, l2, max
   }

Experiment-Specific Pruning
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Progressive dropout
   progressive_config = {
       "dropout_rates": [0.0, 0.2, 0.4, 0.6, 0.8],
       "dropout_mode": "scaled",        # scaled, unscaled
       "pruning_mode": "layer_wise",    # global_joint, layer_wise
       "pruning_metric": "rayleigh_quotient"
   }

   # Layer-isolated pruning
   isolated_config = {
       "target_layers": ["conv1", "conv2"],
       "isolation_mode": "sequential",  # sequential, parallel
       "restoration_mode": "full"       # full, partial, none
   }

   # Cascading pruning
   cascading_config = {
       "cascade_direction": "forward",  # forward, backward, middle_out
       "cascade_threshold": 0.01,
       "recompute_scores": True,
       "track_information_flow": True
   }

Logging and Monitoring
----------------------

Logging Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   logging_config = {
       # Console logging
       "log_level": "INFO",            # DEBUG, INFO, WARNING, ERROR
       "log_interval": 100,            # Steps between logs
       "log_metrics": True,
       "log_gradients": False,

       # File logging
       "log_dir": "./logs",
       "log_to_file": True,
       "separate_process_logs": True,  # For distributed

       # Tensorboard
       "use_tensorboard": True,
       "tensorboard_dir": "./runs",
       "log_images": True,
       "log_histograms": True
   }

Weights & Biases Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   wandb_config = {
       "use_wandb": True,
       "wandb_project": "alignment",
       "wandb_entity": "your-entity",
       "wandb_tags": ["experiment"],
       "wandb_notes": "Experiment notes",

       # What to log
       "log_code": True,
       "log_model": True,
       "log_gradients": True,
       "gradient_log_freq": 100,

       # Artifacts
       "save_artifacts": True,
       "artifact_type": "model"
   }

Analysis Configuration
----------------------

Result Analysis
~~~~~~~~~~~~~~~

.. code-block:: python

   analysis_config = {
       # Visualization
       "plot_metrics": True,
       "plot_format": "png",           # png, pdf, svg
       "plot_dpi": 300,
       "plot_style": "seaborn",

       # Statistical analysis
       "compute_statistics": True,
       "confidence_level": 0.95,
       "bootstrap_samples": 1000,

       # Report generation
       "generate_report": True,
       "report_format": "html",        # html, pdf, markdown
       "include_plots": True,
       "include_tables": True
   }

Comparison Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   comparison_config = {
       # Multiple experiments
       "experiments_to_compare": ["exp1", "exp2", "exp3"],
       "comparison_metrics": ["accuracy", "sparsity", "rq_mean"],

       # Statistical tests
       "statistical_test": "wilcoxon", # wilcoxon, t-test, mann-whitney
       "multiple_comparison_correction": "bonferroni",

       # Visualization
       "comparison_plots": ["bar", "line", "scatter"],
       "error_bars": "std",            # std, sem, ci95
   }

Environment Variables
---------------------

The framework respects several environment variables:

.. code-block:: bash

   # Data paths
   export ALIGNMENT_DATA_DIR="/path/to/data"
   export ALIGNMENT_CACHE_DIR="/path/to/cache"

   # Compute settings
   export ALIGNMENT_DEVICE="cuda:0"
   export ALIGNMENT_NUM_WORKERS="8"
   export ALIGNMENT_MIXED_PRECISION="1"

   # Logging
   export ALIGNMENT_LOG_LEVEL="DEBUG"
   export ALIGNMENT_LOG_DIR="/path/to/logs"

   # Distributed
   export ALIGNMENT_DISTRIBUTED="1"
   export ALIGNMENT_WORLD_SIZE="4"
   export ALIGNMENT_RANK="0"

Command-Line Interface
----------------------

Override configuration via command line:

.. code-block:: bash

   python run_experiment.py \
       --config config.yaml \
       --name "override_name" \
       --model_name resnet50 \
       --epochs 200 \
       --learning_rate 0.01 \
       --device cuda:1

Configuration Validation
------------------------

The framework validates configurations:

.. code-block:: python

   from alignment.infrastructure.configuration import validate_config

   # Validate configuration
   errors = validate_config(config)
   if errors:
       print("Configuration errors:", errors)

   # Auto-fix common issues
   from alignment.infrastructure.configuration import fix_config

   fixed_config = fix_config(config)

Best Practices
--------------

1. **Use YAML for complex configurations** - Easier to read and version control
2. **Override via command line** - For hyperparameter sweeps
3. **Set environment variables** - For machine-specific settings
4. **Validate configurations** - Catch errors early
5. **Use configuration templates** - Start from working examples
6. **Document custom options** - Add comments in YAML files

See Also
--------

- :doc:`/api/experiments` - Experiment API documentation
- :doc:`experiments` - Experiments user guide
