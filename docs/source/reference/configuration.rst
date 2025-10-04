Configuration Parameters Reference
====================================

Complete reference for all configuration parameters in YAML files.

Experiment Settings
-------------------

Basic Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   name: "experiment_name"              # Required: Experiment identifier
   description: "Experiment description"  # Optional: Description text
   tags: ["tag1", "tag2"]               # Optional: Tags for organization
   seed: 42                             # Random seed for reproducibility
   device: "cuda"                       # Device: "cuda", "cpu", "cuda:0"

Model Configuration
-------------------

Model Selection
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``model_name``
     - Model type: "torchvision_model", "timm_model", "mlp", "cnn2p2"
   * - ``pretrained``
     - Use pretrained weights (boolean)
   * - ``model_config``
     - Model-specific parameters (dict)

Model-Specific Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

**Torchvision Models:**

.. code-block:: yaml

   model_name: "torchvision_model"
   model_config:
     model_name: "resnet18"        # Model architecture
     pretrained: true             # Use ImageNet weights
     num_classes: 10              # Output classes

**TIMM Models:**

.. code-block:: yaml

   model_name: "timm_model"
   model_config:
     model_name: "vit_base_patch16_224"
     pretrained: true
     num_classes: 10
     img_size: 224

**Custom MLP:**

.. code-block:: yaml

   model_name: "mlp"
   model_config:
     input_dim: 784
     hidden_dims: [512, 256, 128]
     output_dim: 10
     activation: "relu"            # "relu", "tanh", "sigmoid", "gelu"
     dropout_rate: 0.5

Dataset Configuration
---------------------

Dataset Selection
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``dataset_name``
     - Dataset: "mnist", "cifar10", "cifar100", "imagenet"
   * - ``data_path``
     - Path to dataset files
   * - ``batch_size``
     - Batch size for training/evaluation
   * - ``num_workers``
     - Number of data loading workers

Dataset Parameters
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   dataset_name: "cifar10"
   data_path: "./data"
   batch_size: 128
   num_workers: 4
   dataset_config:
     download: true               # Download if not present
     normalize: true              # Apply normalization
     augmentation: true           # Data augmentation for training

Training Configuration
----------------------

Basic Training
~~~~~~~~~~~~~~

.. code-block:: yaml

   # Training control
   train_before_dropout: true     # Train before analysis
   training_epochs: 10            # Number of epochs
   learning_rate: 0.001           # Learning rate
   optimizer: "adam"              # Optimizer type
   
   # Advanced training options
   scheduler: "cosine"            # LR scheduler: null, "cosine", "step"
   weight_decay: 0.0001           # L2 regularization
   momentum: 0.9                  # For SGD optimizer

Optimizer Options
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Optimizer
     - Best For
   * - ``adam``
     - General purpose, most models
   * - ``adamw``
     - Transformers, weight decay
   * - ``sgd``
     - Traditional training, momentum

Metrics Configuration
---------------------

Metric Selection
~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Basic metrics
   metrics: ["rayleigh_quotient", "mutual_information_gaussian"]
   
   # Advanced configuration
   metric_configs:
     rayleigh_quotient:
       scale_by_norm: false       # Normalize by weight norms
       aggregation_op: "mean"     # Aggregation method
       force_cpu: true            # Use CPU for large operations
     mutual_information_gaussian:
       bins: 50                   # Histogram bins
       estimation_method: "gaussian"

Layer Tracking
~~~~~~~~~~~~~~

.. code-block:: yaml

   # Automatic layer discovery
   tracked_layers: null
   
   # Manual layer specification
   tracked_layers:
     - "conv1"
     - "layer1.0.conv1"
     - "fc"
   
   # CNN-specific settings
   exclude_classification_layer: true
   cnn_rq_aggregation_op: "mean"

Analysis Configuration
----------------------

Alignment Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Alignment measurement
   compute_alignment: true
   measure_alignment_during_training: true
   alignment_frequency: 1         # Measure every N epochs
   alignment_methods: ["rayleigh_quotient"]

Distribution Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   measure_expected_distribution: true
   distribution_bins: 50
   cnn_mode: "unfold"            # CNN processing: "unfold", "patchwise"

Pruning Configuration
---------------------

Pruning Control
~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Enable/disable pruning
   do_pruning_experiments: true
   
   # Pruning strategies
   pruning_strategies: ["magnitude", "alignment", "random"]
   pruning_amounts: [0.1, 0.3, 0.5, 0.7, 0.9]
   pruning_selection_mode: "low"  # "low", "high", "random"

Fine-tuning
~~~~~~~~~~~

.. code-block:: yaml

   fine_tune_after_pruning: true
   fine_tune_epochs: 5
   fine_tune_learning_rate: 0.0001  # Usually lower than training LR

Advanced Pruning
~~~~~~~~~~~~~~~~

.. code-block:: yaml

   pruning_scope: "layer"         # "global" or "layer"
   pruning_alignment_metric: "rayleigh_quotient"
   alignment_structured_pruning: false
   cascading_direction: "forward"

Visualization Configuration
---------------------------

Plot Generation
~~~~~~~~~~~~~~~

.. code-block:: yaml

   generate_plots: true
   plot_format: "png"            # "png", "pdf", "svg"
   plot_dpi: 300                 # Resolution for raster formats

Output Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Directory structure
   checkpoint_dir: "./checkpoints"
   log_dir: "./logs"
   plots_dir: "./plots"
   
   # Checkpointing
   save_best: true
   checkpoint_interval: 1000

Distributed Training
--------------------

.. code-block:: yaml

   # Multi-GPU training
   distributed: true
   world_size: 4                 # Number of GPUs
   
   # Multi-network training
   num_networks: 10              # Train multiple networks in parallel

Complete Configuration Template
-------------------------------

See ``configs/template_comprehensive.yaml`` for a complete example with all available parameters and detailed comments.

Quick Configuration Examples
----------------------------

**Simple MNIST MLP:**

.. code-block:: yaml

   name: "mnist_mlp"
   model_name: "mlp"
   dataset_name: "mnist"
   metrics: ["rayleigh_quotient"]

**ResNet-18 on CIFAR-10:**

.. code-block:: yaml

   name: "resnet18_cifar"
   model_name: "torchvision_model"
   model_config:
     model_name: "resnet18"
     pretrained: true
     num_classes: 10
   dataset_name: "cifar10"
   do_pruning_experiments: true

**Vision Transformer:**

.. code-block:: yaml

   name: "vit_experiment"
   model_name: "timm_model"
   model_config:
     model_name: "vit_base_patch16_224"
     pretrained: true
     num_classes: 10
   dataset_name: "cifar10"
   batch_size: 64
   learning_rate: 0.00001
