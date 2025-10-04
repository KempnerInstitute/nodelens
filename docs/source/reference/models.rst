Supported Models Reference
===========================

This reference lists all supported model architectures and how to configure them.

Vision Models (Torchvision)
----------------------------

Convolutional Networks
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Model Name
     - Configuration
     - Description
   * - ``alexnet``
     - ``model_name: "torchvision_model"``
     - Classic CNN architecture, good for CIFAR-10/ImageNet
   * - ``resnet18``
     - ``model_name: "torchvision_model"``
     - Lightweight ResNet, fast training
   * - ``resnet50``
     - ``model_name: "torchvision_model"``
     - Standard ResNet, good balance of accuracy/speed
   * - ``vgg16``
     - ``model_name: "torchvision_model"``
     - Deep convolutional network
   * - ``efficientnet_b0``
     - ``model_name: "torchvision_model"``
     - Modern efficient architecture

Example Configuration:

.. code-block:: yaml

   model_name: "torchvision_model"
   model_config:
     model_name: "resnet18"
     pretrained: true
     num_classes: 10  # For CIFAR-10

Vision Transformers (TIMM)
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model Name
     - Description
   * - ``vit_base_patch16_224``
     - Vision Transformer Base with 16x16 patches
   * - ``vit_small_patch16_224``
     - Smaller Vision Transformer
   * - ``deit_base_patch16_224``
     - Data-efficient Image Transformer
   * - ``swin_tiny_patch4_window7_224``
     - Swin Transformer architecture

Example Configuration:

.. code-block:: yaml

   model_name: "timm_model"
   model_config:
     model_name: "vit_base_patch16_224"
     pretrained: true
     num_classes: 10
     img_size: 224

Custom Models
-------------

Multi-Layer Perceptrons
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   model_name: "mlp"
   model_config:
     input_dim: 784
     hidden_dims: [512, 256, 128]
     output_dim: 10
     activation: "relu"
     dropout_rate: 0.5

Convolutional Networks
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   model_name: "cnn2p2"
   model_config:
     in_channels: 3
     conv_channels: [32, 64]
     kernel_sizes: [5, 5]
     hidden_fc_dim: 256
     output_dim: 10

Hugging Face Models
-------------------

Language Models
~~~~~~~~~~~~~~~

.. code-block:: yaml

   model_name: "hf_causal_lm"
   model_config:
     model_id: "meta-llama/Meta-Llama-3-8B"
     torch_dtype: "float16"
     device_map: "auto"

Vision Models
~~~~~~~~~~~~~

.. code-block:: yaml

   model_name: "hf_vision_model"
   model_config:
     model_id: "google/vit-base-patch16-224"
     trust_remote_code: false

Layer Tracking Configuration
----------------------------

Automatic Layer Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Auto-discover all layers with weights
   tracked_layers: null

Manual Layer Specification
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # ResNet-18 specific layers
   tracked_layers:
     - "conv1"
     - "layer1.0.conv1"
     - "layer2.0.conv1"
     - "layer3.0.conv1"
     - "layer4.0.conv1"
     - "fc"

   # Vision Transformer layers
   tracked_layers:
     - "patch_embed.proj"
     - "blocks.0.attn.qkv"
     - "blocks.6.attn.qkv"
     - "head"

Model-Specific Settings
-----------------------

CNN Processing Mode
~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # For convolutional models
   cnn_mode: "unfold"  # Options: "unfold", "patchwise", "batch_patch_combined"

Preprocessing Options
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   dataset_config:
     normalize: true
     augmentation: true
     resize_to: 224  # For Vision Transformers

Training Parameters
~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Model-specific training settings
   training_epochs: 10
   learning_rate: 0.001  # Higher for training from scratch
   # learning_rate: 0.0001  # Lower for fine-tuning pretrained models
   optimizer: "adam"  # or "adamw" for transformers

Complete Model Examples
-----------------------

See ``configs/examples/`` for complete working configurations:

- ``resnet18_analysis.yaml`` - ResNet-18 on CIFAR-10
- ``alexnet_analysis.yaml`` - AlexNet configuration  
- ``vit_b16_analysis.yaml`` - Vision Transformer setup
- ``vision_networks_master.yaml`` - All models with options
