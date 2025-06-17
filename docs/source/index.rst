.. Neural Network Alignment documentation master file

Neural Network Alignment Framework
==================================

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg
   :target: https://pytorch.org/
   :alt: PyTorch Version

This framework provides a comprehensive suite of tools for studying neural network alignment properties, 
implementing various pruning strategies, and analyzing network behavior through information-theoretic metrics.

Key Features
------------

* **Modular Architecture**: Clean separation of concerns with dedicated modules for models, metrics, experiments, and utilities
* **Advanced Metrics**: Implementation of Rayleigh Quotient (RQ), Mutual Information (MI), Partial Information Decomposition (PID), CKA, CCA, and more
* **Pruning Strategies**: Progressive dropout, eigenvector-based pruning, layer-isolated pruning, and cascading methods
* **Tensorized Dropout**: Efficient structured pruning implementation
* **Model Wrapper**: Automatic activation tracking and layer weight extraction
* **Experiment Framework**: Reproducible experiment management with configuration support

Getting Started
---------------

.. code-block:: bash

   # Clone the repository
   git clone <repository-url>
   cd alignment/src/alignment_refactor

   # Install dependencies
   pip install -r requirements.txt

Quick Example
-------------

.. code-block:: python

   from alignment_refactor.models.architectures.standard_models import create_model
   from alignment_refactor.models import ModelWrapper
   from alignment_refactor.metrics import RayleighQuotient
   from alignment_refactor.experiments.progressive_dropout import ProgressiveDropoutExperiment
   from alignment_refactor.experiments.base import ExperimentConfig

   # Create a model
   model = create_model('mlp', 'mnist', hidden_dims=[300, 200])
   
   # Wrap it for tracking
   wrapped_model = ModelWrapper(model, tracked_layers=['network.0', 'network.3'])
   
   # Run an experiment
   config = ExperimentConfig(
       name="mnist_pruning",
       model_name="mlp",
       dataset_name="mnist",
       metrics=["rayleigh_quotient"]
   )
   experiment = ProgressiveDropoutExperiment(config)
   results = experiment.run()

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   user_guide/installation
   user_guide/quickstart
   user_guide/experiments
   user_guide/configuration
   user_guide/metrics
   user_guide/models

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/core
   api/models
   api/metrics
   api/experiments
   api/data
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Examples & Tutorials
   
   examples/basic_usage
   examples/pruning_experiments
   examples/custom_metrics
   examples/advanced_experiments

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources
   
   migration_guide
   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 