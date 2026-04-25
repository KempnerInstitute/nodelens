Models API Reference
====================

This section documents the model components of NodeLens.

Model Architectures
-------------------

.. automodule:: nodelens.models.architectures
   :members:
   :undoc-members:
   :show-inheritance:

Pre-defined Models
~~~~~~~~~~~~~~~~~~

The framework includes several pre-defined model architectures:

.. autofunction:: nodelens.models.architectures.get_model

Available Models
^^^^^^^^^^^^^^^^

- ``mlp``: Multi-layer perceptron
- ``cnn2p2``: 2-conv 2-pool CNN
- ``resnet18``, ``resnet34``, ``resnet50``: ResNet variants
- ``vgg11``, ``vgg13``, ``vgg16``, ``vgg19``: VGG variants
- ``efficientnet_b0`` through ``efficientnet_b7``: EfficientNet variants
- ``vit_b_16``, ``vit_b_32``, ``vit_l_16``: Vision Transformer variants

Model Registry
--------------

.. automodule:: nodelens.models.registry
   :members:
   :undoc-members:
   :show-inheritance:

Model Utilities
---------------

.. automodule:: nodelens.models.utils
   :members:
   :undoc-members:
   :show-inheritance:
