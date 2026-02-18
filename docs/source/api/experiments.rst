Experiments API Reference
=========================

This section provides detailed documentation for all experiment types available in the alignment framework.

.. contents:: Table of Contents
   :local:
   :depth: 2

Base Experiment Classes
-----------------------

.. automodule:: alignment.experiments.base
   :members:
   :undoc-members:
   :show-inheritance:

ExperimentConfig
~~~~~~~~~~~~~~~~

.. autoclass:: alignment.experiments.base.ExperimentConfig
   :members:
   :undoc-members:

   **Core Configuration Options:**

   .. attribute:: name
      :type: str

      Unique identifier for the experiment

   .. attribute:: model_name
      :type: str

      Name of the model architecture to use (e.g., "resnet18", "mlp", "cnn2p2")

   .. attribute:: dataset_name
      :type: str

      Dataset to use (e.g., "cifar10", "mnist", "imagenet")

   .. attribute:: metrics
      :type: List[str]

      List of metrics to compute. Available metrics:

      - ``"rayleigh_quotient"``: Neuron alignment with input variance
      - ``"mutual_information"``: Information shared between layers
      - ``"pid_shared"``: Shared information (PID)
      - ``"pid_unique"``: Unique information per neuron
      - ``"pid_synergy"``: Synergistic information
      - ``"cka"``: Centered Kernel Alignment
      - ``"cca"``: Canonical Correlation Analysis
      - ``"weight_cosine_similarity"``: Cosine similarity between weights
      - ``"node_redundancy"``: Redundancy between neurons

   .. attribute:: device
      :type: str
      :default: "cuda" if available else "cpu"

      Device to run experiments on

   .. attribute:: seed
      :type: int
      :default: 42

      Random seed for reproducibility

Progressive Dropout Experiment
------------------------------

.. automodule:: alignment.experiments.progressive_dropout
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: alignment.experiments.progressive_dropout.ProgressiveDropoutExperiment
   :members:
   :undoc-members:

   **Description:**

   This experiment gradually increases dropout rates during evaluation to study how networks
   degrade as neurons are progressively removed. It's useful for understanding network
   robustness and identifying critical neurons.

   **Key Configuration Options:**

   .. attribute:: dropout_rates
      :type: List[float]
      :default: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

      List of dropout rates to evaluate

   .. attribute:: dropout_mode
      :type: str
      :default: "scaled"

      How to apply dropout:

      - ``"scaled"``: Scale remaining activations by 1/(1-p)
      - ``"unscaled"``: No scaling (true dropout)

   .. attribute:: pruning_mode
      :type: str
      :default: "global_joint"

      Pruning strategy:

      - ``"global_joint"``: Prune globally across all layers
      - ``"layer_wise"``: Prune each layer independently
      - ``"structured"``: Remove entire channels/filters

   .. attribute:: pruning_strategy
      :type: str
      :default: "low"

      Which neurons to prune:

      - ``"low"``: Remove low-scoring neurons
      - ``"high"``: Remove high-scoring neurons
      - ``"random"``: Random pruning

   .. attribute:: pruning_metric
      :type: str
      :default: "rayleigh_quotient"

      Metric to use for importance scoring

   .. attribute:: exclude_classification_layer
      :type: bool
      :default: True

      Whether to exclude the final classification layer from pruning

   **Example Usage:**

   .. code-block:: python

      from alignment.experiments import ProgressiveDropoutExperiment
      from alignment.experiments.base import ExperimentConfig

      config = ExperimentConfig(
          name="progressive_dropout_resnet",
          model_name="resnet18",
          dataset_name="cifar10",
          metrics=["rayleigh_quotient", "mutual_information"],
          dropout_rates=[0.0, 0.2, 0.4, 0.6, 0.8],
          dropout_mode="scaled",
          pruning_mode="layer_wise",
          pruning_strategy="low",
          pruning_metric="rayleigh_quotient"
      )

      experiment = ProgressiveDropoutExperiment(config)
      results = experiment.run()

      # Results contain:
      # - Accuracy at each dropout rate
      # - Metric values for remaining neurons
      # - Layer-wise statistics

Experiment Runner
-----------------

.. automodule:: alignment.experiments.runner
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: alignment.experiments.runner.ExperimentRunner
   :members:
   :undoc-members:

   **Description:**

   The ExperimentRunner manages multiple experiments, handling parallel execution,
   result aggregation, and resource management.

   **Key Features:**

   - Parallel experiment execution
   - Automatic result saving and loading
   - Progress tracking and logging
   - Resource management (GPU allocation)
   - Experiment resumption on failure

   **Example Usage:**

   .. code-block:: python

      from alignment.experiments import ExperimentRunner
      from alignment.experiments.base import ExperimentConfig

      # Define multiple experiments
      configs = []

      # Progressive dropout with different strategies
      for strategy in ["low", "high", "random"]:
          configs.append(ExperimentConfig(
              name=f"progressive_{strategy}",
              model_name="resnet18",
              dataset_name="cifar10",
              metrics=["rayleigh_quotient"],
              pruning_strategy=strategy
          ))

      # Different dropout rates
      for rate in [0.3, 0.5, 0.7]:
          configs.append(ExperimentConfig(
              name=f"dropout_rate_{rate}",
              model_name="resnet18",
              dataset_name="cifar10",
              dropout_rates=[0.0, rate]
          ))

      # Run all experiments
      runner = ExperimentRunner(
          configs=configs,
          results_dir="./results",
          parallel=True,
          max_workers=4,
          gpu_per_worker=0.25  # Share GPUs
      )

      all_results = runner.run()

      # Analyze results
      runner.generate_report(output_path="./report.html")

Advanced Configuration Options
------------------------------

Training Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. attribute:: train_before_dropout
   :type: bool

   Whether to train the model before applying dropout (default: True)

.. attribute:: training_epochs
   :type: int

   Number of epochs to train (default: 100)

.. attribute:: learning_rate
   :type: float

   Initial learning rate (default: 0.1)

.. attribute:: optimizer
   :type: str

   Optimizer to use: "sgd", "adam", "adamw" (default: "sgd")

.. attribute:: lr_schedule
   :type: str

   Learning rate schedule: "cosine", "step", "exponential", "none" (default: "cosine")

Metric Computation Options
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. attribute:: metric_configs
   :type: Dict[str, Dict]

   Per-metric configuration options (default: {})

.. attribute:: scale_by_norm
   :type: bool

   Whether to scale metrics by weight norm (default: False)

.. attribute:: force_cpu_for_large_metric_ops
   :type: bool

   Move large operations to CPU to save GPU memory (default: False)

.. attribute:: metric_batch_size
   :type: int

   Batch size for metric computation (default: 1000)

Logging and Checkpointing
~~~~~~~~~~~~~~~~~~~~~~~~~

.. attribute:: checkpoint_dir
   :type: str

   Directory for saving checkpoints (default: "./checkpoints")

.. attribute:: checkpoint_interval
   :type: int

   Steps between checkpoints (default: 1000)

.. attribute:: save_best
   :type: bool

   Save best model based on validation accuracy (default: True)

.. attribute:: wandb_project
   :type: Optional[str]

   Weights & Biases project name (default: None)

.. attribute:: tensorboard_dir
   :type: Optional[str]

   TensorBoard logging directory (default: None)

Distributed Training
~~~~~~~~~~~~~~~~~~~~

.. attribute:: distributed
   :type: bool

   Enable distributed training (default: False)

.. attribute:: world_size
   :type: int

   Number of distributed processes (default: 1)

.. attribute:: backend
   :type: str

   Distributed backend: "nccl", "gloo" (default: "nccl")

Result Analysis
---------------

All experiments return a standardized results dictionary containing:

.. code-block:: python

   {
       "config": ExperimentConfig,  # Full configuration
       "metrics": {
           "metric_name": {
               "layer_name": {
                   "dropout_rate": values
               }
           }
       },
       "accuracy": {
           "dropout_rate": accuracy_value
       },
       "timing": {
           "total_time": seconds,
           "metric_computation_time": seconds
       },
       "metadata": {
           "hostname": str,
           "gpu_info": dict,
           "timestamp": str
       }
   }

See Also
--------

- :doc:`/user_guide/experiments` - User guide for experiments
- :doc:`/api/metrics` - Available metrics documentation
- :doc:`/api/pruning` - Pruning strategies documentation
