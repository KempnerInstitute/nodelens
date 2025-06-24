Training API Reference
======================

This section documents the training components of the alignment framework.

Base Training
-------------

.. automodule:: alignment.training.base
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: alignment.training.base.BaseTrainer
   :members:
   :special-members: __init__
   :undoc-members:

Multi-Network Training
----------------------

.. automodule:: alignment.training.multi_network
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: alignment.training.multi_network.MultiNetworkTrainer
   :members:
   :undoc-members:

Training Utilities
------------------

.. automodule:: alignment.training.utils
   :members:
   :undoc-members:
   :show-inheritance:

Optimization
~~~~~~~~~~~~

.. autofunction:: alignment.training.utils.get_optimizer
.. autofunction:: alignment.training.utils.get_scheduler

Training Loops
~~~~~~~~~~~~~~

.. autofunction:: alignment.training.utils.train_epoch
.. autofunction:: alignment.training.utils.evaluate 