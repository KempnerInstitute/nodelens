Training API Reference
======================

This section documents the training components of NodeLens.

Base Training
-------------

.. automodule:: nodelens.training.base
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: nodelens.training.base.BaseTrainer
   :members:
   :special-members: __init__
   :undoc-members:

Multi-Network Training
----------------------

.. automodule:: nodelens.training.multi_network
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: nodelens.training.multi_network.MultiNetworkTrainer
   :members:
   :undoc-members:

Training Utilities
------------------

.. automodule:: nodelens.training.utils
   :members:
   :undoc-members:
   :show-inheritance:

Optimization
~~~~~~~~~~~~

.. autofunction:: nodelens.training.utils.get_optimizer
.. autofunction:: nodelens.training.utils.get_scheduler

Training Loops
~~~~~~~~~~~~~~

.. autofunction:: nodelens.training.utils.train_epoch
.. autofunction:: nodelens.training.utils.evaluate
