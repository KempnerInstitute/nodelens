# Configuration Reference

This document details all configurable parameters for experiments, primarily managed through YAML files and parsed by `src/alignment/config.py` using OmegaConf.

## Main Experiment Configuration (`ExperimentConfig`)

This is the top-level configuration object, typically defined in your main YAML file (e.g., `configs/config_alignment_experiment.yaml`).

| Parameter                  | Type                            | Default                           | Description                                                                                                | Options                                         |
|----------------------------|---------------------------------|-----------------------------------|------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| `experiment_name`          | `str`                           | `"default_experiment"`            | A descriptive name for the experiment.                                                                     |                                                 |
| `experiment_type`          | `str`                           | `"progressive_dropout"`           | Type of experiment to run (e.g., "progressive_dropout", "alignment_stats").                             |                                                 |
| `results_path`             | `str`                           | `"results"`                       | Base path where experiment results will be saved.                                                          |                                                 |
| `use_timestamp`            | `bool`                          | `True`                            | If true, a timestamped subdirectory is created within `results_path` for this experiment.                | `True`, `False`                                 |
| `device`                   | `Optional[str]`                 | `None`                            | Computation device (e.g., "cpu", "cuda", "cuda:0"). If `None`, attempts to auto-detect.                   |                                                 |
| `no_save`                  | `bool`                          | `False`                           | If true, disables saving of any results or checkpoints.                                                    | `True`, `False`                                 |
| `just_plot`                | `bool`                          | `False`                           | If true, attempts to load existing results and generate plots only, skipping computation.                  | `True`, `False`                                 |
| `save_networks`            | `bool`                          | `False`                           | If true, trained model checkpoints will be saved (also see `checkpointing` config).                         | `True`, `False`                                 |
| `show_all`                 | `bool`                          | `False`                           | (Purpose may vary by experiment type, generally for enabling more verbose output or plots).                | `True`, `False`                                 |
| `timestamp`                | `Optional[str]`                 | `None`                            | Specific timestamp to use for results directory if `use_timestamp` is true (overrides auto-generation).    |                                                 |
| `debug_mode`               | `bool`                          | `False`                           | Enables more verbose logging and debugging outputs.                                                        | `True`, `False`                                 |
| `seed`                     | `Optional[int]`                 | `None`                            | Random seed for reproducibility. If `None`, a random seed is used.                                         |                                                 |
| `use_ddp`                  | `bool`                          | `False`                           | Whether to use Distributed Data Parallel for training.                                                     | `True`, `False`                                 |
| `ddp_backend`              | `Optional[str]`                 | `"nccl"`                          | DDP backend to use.                                                                                        | `"nccl"`, `"gloo"`, `"mpi"`                       |
| `ddp_rank`                 | `int`                           | `0`                               | (Runtime set) DDP rank of the current process.                                                             |                                                 |
| `ddp_world_size`           | `int`                           | `1`                               | (Runtime set) Total number of DDP processes.                                                               |                                                 |
| `ddp_local_rank`           | `int`                           | `0`                               | (Runtime set) Local DDP rank on the current node.                                                          |                                                 |
| `dataset`                  | `DatasetConfig`                 | (see below)                       | Configuration for the dataset. See [Dataset Configuration](#dataset-configuration-datasetconfig).            |                                                 |
| `model`                    | `ModelConfig`                   | (see below)                       | Configuration for the neural network model. See [Model Configuration](#model-configuration-modelconfig).         |                                                 |
| `training`                 | `TrainingConfig`                | (see below)                       | Configuration for the training process. See [Training Configuration](#training-configuration-trainingconfig).    |                                                 |
| `alignment_settings`       | `AlignmentConfig`               | (see below)                       | Configuration for alignment metric calculations. See [Alignment Configuration](#alignment-configuration-alignmentconfig). |                                                 |
| `pruning_settings`         | `PruningConfig`                 | (see below)                       | Configuration for pruning experiments. See [Pruning Configuration](#pruning-configuration-pruningconfig).      |                                                 |
| `checkpointing`            | `CheckpointingConfig`           | (see below)                       | Configuration for model checkpointing. See [Checkpointing Configuration](#checkpointing-configuration-checkpointingconfig). |                                                 |
| `wandb`                    | `WandbConfig`                   | (see below)                       | Configuration for Weights & Biases logging. See [W&B Configuration](#wb-configuration-wandbconfig).          |                                                 |
| `extra`                    | `ExtraConfig`                   | (see below)                       | Extra/miscellaneous configuration parameters. See [Extra Configuration](#extra-configuration-extraconfig).     |                                                 |

---

## Dataset Configuration (`DatasetConfig`)
Parameters for specifying and loading the dataset, under the `dataset` key in the main YAML.

| Parameter        | Type            | Default   | Description                                                                       | Options                                     |
|------------------|-----------------|-----------|-----------------------------------------------------------------------------------|---------------------------------------------|
| `dataset_name`   | `str`           | `"MNIST"` | Name of the dataset to use.                                                       | `"MNIST"`, `"CIFAR10"`, `"CIFAR100"`, `"ImageNet"`, custom |
| `data_path`      | `Optional[str]` | `None`    | Path to the dataset. If `None`, often defaults to a standard location (e.g., `./data`). |                                             |
| `batch_size`     | `int`           | `128`     | Batch size for data loaders.                                                      | Positive integer                            |
| `num_workers`    | `int`           | `4`       | Number of worker processes for data loading.                                        | Non-negative integer                        |

---

## Model Configuration (`ModelConfig`)
Parameters for defining the neural network architecture, under the `model` key in the main YAML.

| Parameter                | Type                                  | Default        | Description                                                                                                                        | Options                                                                  |
|--------------------------|---------------------------------------|----------------|------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| `model_name`             | `str`                                 | `"MLP"`        | Main model type identifier. Determines which specific parameter block below is used (e.g., `mlp_params`).                              | `"MLP"`, `"CNN2P2"`, `"torchvision_<name>"`, `"hf_<name>"`, `"external"`   |
| `output_dim`             | `int`                                 | `10`           | Number of output classes/features for the final layer.                                                                               | Positive integer                                                         |
| `dropout_rate`           | `float`                               | `0.0`          | General dropout rate applied in some models (0.0 to 1.0).                                                                          | `0.0` - `1.0`                                                            |
| `cnn_mode`               | `Optional[str]`                       | `"unfold"`     | For `AlignmentNetwork` wrapper with CNNs: how to process feature maps for alignment. If `None`, defaults to `"unfold"`.            | `"unfold"`, `"patchwise"`, `"batch_patch_combined"`                      |
| `mlp_params`             | `Optional[MLPParamsConfig]`           | `None`         | Configuration for MLP models. Used if `model_name` is "MLP". See [MLP Parameters](#mlp-parameters-mlpparamsconfig).                    |                                                                          |
| `cnn2p2_params`          | `Optional[CNN2P2ParamsConfig]`        | `None`         | Configuration for CNN2P2 models. Used if `model_name` is "CNN2P2". See [CNN2P2 Parameters](#cnn2p2-parameters-cnn2p2paramsconfig).      |                                                                          |
| `external_params`        | `Optional[ExternalModelParamsConfig]` | `None`         | Configuration for external models. Used if `model_name` suggests an external model. See [External Model Parameters](#external-model-parameters-externalmodelparamsconfig). |                                                                          |
| `extra_model_params`     | `Dict[str, Any]`                      | `{}`           | Dictionary for ad-hoc parameters passed directly to internal model constructors.                                                   |                                                                          |

### MLP Parameters (`MLPParamsConfig`)
Nested under `model.mlp_params`. Used when `model.model_name` is `"MLP"`.

| Parameter      | Type            | Default        | Description                                               | Options                                       |
|----------------|-----------------|----------------|-----------------------------------------------------------|-----------------------------------------------|
| `input_dim`    | `Optional[int]` | `784`          | Input dimensionality (e.g., 784 for flattened MNIST).     | Positive integer                              |
| `hidden_dims`  | `List[int]`     | `[128, 64]`    | List of hidden layer dimensions.                          | List of positive integers                     |
| `activation`   | `str`           | `"relu"`       | Activation function for hidden layers.                    | `"relu"`, `"tanh"`, `"sigmoid"`, `"identity"` |

### CNN2P2 Parameters (`CNN2P2ParamsConfig`)
Nested under `model.cnn2p2_params`. Used when `model.model_name` is `"CNN2P2"`. (A 2-Convolutional-Layer, 2-Pooling-Layer CNN).

| Parameter          | Type          | Default        | Description                                                        |
|--------------------|---------------|----------------|--------------------------------------------------------------------|
| `in_channels`      | `int`         | `1`            | Number of input channels (e.g., 1 for MNIST, 3 for CIFAR).         |
| `conv_channels`    | `List[int]`   | `[32, 64]`     | List of output channels for the two convolutional layers.          |
| `kernel_sizes`     | `List[int]`   | `[5, 5]`       | List of kernel sizes for the two convolutional layers.             |
| `strides`          | `List[int]`   | `[1, 1]`       | List of strides for the two convolutional layers.                  |
| `paddings`         | `List[int]`   | `[0, 0]`       | List of paddings for the two convolutional layers.                 |
| `pool_kernel_size` | `int`         | `2`            | Kernel size for pooling layers.                                    |
| `pool_stride`      | `int`         | `2`            | Stride for pooling layers.                                         |
| `hidden_fc_dim`    | `int`         | `128`          | Dimension of the hidden fully connected layer before the output.   |
| `example_input_hw` | `List[int]`   | `[28, 28]`     | Example input height and width (e.g., `[28,28]` for MNIST). Used to calculate flattened features. |

### External Model Parameters (`ExternalModelParamsConfig`)
Nested under `model.external_params`. Used when `model.model_name` indicates an external model (e.g., `"torchvision_resnet18"`, `"hf_bert-base-uncased"`) or is set to `"external"`.

| Parameter                  | Type            | Default    | Description                                                              | Options                                         |
|----------------------------|-----------------|------------|--------------------------------------------------------------------------|-------------------------------------------------|
| `source`                   | `Optional[str]` | `None`     | Source of the external model.                                            | `"torchvision"`, `"huggingface_transformers"`   |
| `name_or_path`             | `Optional[str]` | `None`     | Name of the model (e.g., "resnet18") or path to model weights/config.    |                                                 |
| `pretrained`               | `bool`          | `True`     | Whether to load pretrained weights.                                      | `True`, `False`                                 |
| `freeze_feature_extractor` | `bool`          | `False`    | If true, freezes the weights of the loaded feature extractor part of the model. | `True`, `False`                                 |

---

## Training Configuration (`TrainingConfig`)
Parameters for the training loop, under the `training` key in the main YAML.

| Parameter             | Type    | Default           | Description                                                                    | Options                                    |
|-----------------------|---------|-------------------|--------------------------------------------------------------------------------|--------------------------------------------|
| `epochs`              | `int`   | `10`              | Number of training epochs.                                                     | Positive integer                           |
| `replicates`          | `int`   | `1`               | Number of times to replicate the training (e.g., with different seeds if varied). | Positive integer                           |
| `optimizer`           | `str`   | `"Adam"`          | Optimizer to use for training.                                                 | `"Adam"`, `"SGD"`, `"RMSprop"`, `"AdamW"`    |
| `learning_rate`       | `float` | `1e-3`            | Learning rate for the optimizer.                                               | Positive float                             |
| `weight_decay`        | `float` | `0.0`             | Weight decay (L2 penalty) for the optimizer.                                   | Non-negative float                         |
| `loss`                | `str`   | `"cross_entropy"` | Loss function to use.                                                          | e.g., `"cross_entropy"`                    |
| `momentum`            | `float` | `0.9`             | Momentum factor for optimizers like SGD.                                       | Non-negative float                         |
| `training_method`     | `str`   | `"auto"`          | Method for training (e.g., handling multiple networks).                        | `"auto"`, `"sequential"`, `"fully_tensorized"` |
| `train_before_dropout`| `bool`  | `True`            | Controls initial training phase before dropout or pruning experiments start.   | `True`, `False`                            |

---

## Alignment Configuration (`AlignmentConfig`)
Parameters for calculating alignment metrics, under the `alignment_settings` key in the main YAML.

| Parameter                        | Type                             | Default      | Description                                                                                                | Options                                                                                 |
|----------------------------------|----------------------------------|--------------|------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| `metric`                         | `str`                            | `"RQ"`       | Primary alignment metric to compute.                                                                       | `"RQ"`, `"NullSpace"`, `"MI"`, `"WeightSimilarity"`, `"NodeRedundancy"`, `"RankAlignment"` |
| `scale_by_norm`                  | `bool`                           | `False`      | For Rayleigh Quotient (RQ), whether to scale by the norm of the weight vector (relative RQ).               | `True`, `False`                                                                         |
| `cnn_mode`                       | `str`                            | `"unfold"`   | For CNNs, how feature maps are processed before metric calculation.                                        | `"unfold"`, `"patchwise"`, `"batch_patch_combined"`, `"filter_patch_summary"`, `"filter_specific_covariance_rq"` |
| `cnn_rq_aggregation_op`          | `str`                            | `"mean"`     | For RQ on CNNs, how scores from patches/filters are aggregated.                                            | `"mean"`, `"max"`, `"var"`, `"sum"`                                                       |
| `run_progressive`                | `bool`                           | `True`       | (Likely influences experimental flow, e.g., in progressive pruning, rather than a direct metric parameter). | `True`, `False`                                                                         |
| `run_eigenvector`                | `bool`                           | `False`      | (Likely related to RQ, possibly for computing alignment with principal eigenvectors).                        | `True`, `False`                                                                         |
| `callbacks`                      | `Optional[CallbackSettings]`     | `None`       | Configuration for metric tracking callbacks during training. See [Callback Settings](#callback-settings-callbacksettings). |                                                                                         |
| `force_cpu_for_large_metric_ops` | `bool`                           | `True`       | If true, offloads potentially large intermediate computations for metrics (like covariance matrices) to CPU. | `True`, `False`                                                                         |

### Callback Settings (`CallbackSettings`)
Nested under `alignment_settings.callbacks`. Configures metrics to be tracked during training epochs.

| Parameter           | Type                          | Default     | Description                                                                          |
|---------------------|-------------------------------|-------------|--------------------------------------------------------------------------------------|
| `alignment_metrics` | `List[MetricTrackerConfig]`   | `[]` (empty list) | List of metrics to track. See [Metric Tracker Configuration](#metric-tracker-configuration-metrictrackerconfig). |

### Metric Tracker Configuration (`MetricTrackerConfig`)
Used within `alignment_settings.callbacks.alignment_metrics`. Defines a specific metric to track.

| Parameter     | Type            | Default   | Description                                                                   |
|---------------|-----------------|-----------|-------------------------------------------------------------------------------|
| `name`        | `str`           | `"RQ"`    | Name of the alignment metric to track (must be a registered metric name).     |
| `num_batches` | `Optional[int]` | `5`       | Number of batches from the validation/test loader to use for computing the metric. If `None`, uses all batches. |

---

## Pruning Configuration (`PruningConfig`)
Parameters for pruning experiments, under the `pruning_settings` key in the main YAML. Also see `doc/pruning_modes.md`.

| Parameter                      | Type            | Default            | Description                                                                                          | Options                                                          |
|--------------------------------|-----------------|--------------------|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| `dropout_min`                  | `float`         | `0.0`              | Minimum dropout/pruning rate to test (0.0 to 1.0).                                                   | `0.0` - `1.0`                                                    |
| `dropout_max`                  | `float`         | `0.9`              | Maximum dropout/pruning rate to test (0.0 to 1.0).                                                   | `0.0` - `1.0`                                                    |
| `dropout_steps`                | `int`           | `40`               | Number of steps between `dropout_min` and `dropout_max`.                                             | Positive integer (or 0 if min == max)                            |
| `dropout_mode`                 | `str`           | `"scaled"`         | How dropout is applied. "scaled" usually refers to inverted dropout.                                 | `"scaled"`, `"unscaled"`                                         |
| `dropout_pruning_mode`         | `str`           | `"global_joint"`   | Strategy for pruning nodes. See `doc/pruning_modes.md` for details.                                  | `"global_joint"`, `"layer_wise"`, `"layer_isolated"`, `"cascading_layer"` |
| `exclude_classification_layer` | `bool`          | `True`             | Whether to exclude the final classification layer from pruning.                                        | `True`, `False`                                                  |
| `use_multi_strategy_dropout`   | `bool`          | `True`             | Whether to use an optimized multi-strategy dropout implementation.                                     | `True`, `False`                                                  |
| `num_batches_for_scores`       | `Optional[int]` | `5`                | Number of batches to use for calculating scores (e.g., RQ) that guide pruning. If `None`, uses all available data. | Positive integer or `None`                                       |

---

## Checkpointing Configuration (`CheckpointingConfig`)
Parameters for saving and loading model checkpoints, under the `checkpointing` key in the main YAML.

| Parameter            | Type    | Default   | Description                                                                         | Options         |
|----------------------|---------|-----------|-------------------------------------------------------------------------------------|-----------------|
| `save_checkpoints`   | `bool`  | `False`   | If true, enables saving of model checkpoints during training.                       | `True`, `False` |
| `checkpoint_frequency` | `int`   | `1`       | Frequency (in epochs) at which to save checkpoints if `save_checkpoints` is true.   | Positive integer|
| `load_checkpoint`    | `bool`  | `False`   | If true, attempts to load a model checkpoint at the start of the experiment.          | `True`, `False` |

---

## W&B Configuration (`WandbConfig`)
Parameters for Weights & Biases integration, under the `wandb` key in the main YAML.

| Parameter       | Type            | Default              | Description                                                                    |
|-----------------|-----------------|----------------------|--------------------------------------------------------------------------------|
| `use_wandb`     | `bool`          | `False`              | If true, enables logging to Weights & Biases.                                  |
| `wandb_project` | `Optional[str]` | `"neural_alignment"` | Name of the W&B project.                                                       |
| `wandb_entity`  | `Optional[str]` | `None`               | W&B entity (username or team name). If `None`, uses W&B default.               |

---

## Extra Configuration (`ExtraConfig`)
Miscellaneous parameters, under the `extra` key in the main YAML.

| Parameter           | Type            | Default   | Description                                                      |
|---------------------|-----------------|-----------|------------------------------------------------------------------|
| `log_frequency`     | `int`           | `1`       | Frequency for logging (e.g., per epoch or per N steps).          |
| `log_images`        | `bool`          | `True`    | Whether to log images (e.g., to W&B, if applicable).             |
| `detailed_logging`  | `bool`          | `True`    | Enables more detailed logging output during experiments.         |
| `dummy_extra_param` | `Optional[str]` | `None`    | A placeholder parameter if no other extra parameters are needed. |

---

This reference should help in understanding and modifying the YAML configuration files for your experiments. For metric-specific parameters not listed here (often passed directly to metric computation functions), please refer to the [Metrics Documentation](metrics/README.md). 