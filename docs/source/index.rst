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

* **36 Alignment Metrics**: Comprehensive suite including Rayleigh quotient, mutual information, spectral metrics, and more
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
   git clone https://github.com/KempnerInstitute/alignment.git
   cd alignment

   # Install dependencies
   pip install -e .[all]

Quick Example
-------------

.. code-block:: python

   from alignment.core import ModelWrapper
   from alignment.metrics import get_metric
   import torch

   # Create a model
   model = torch.nn.Sequential(
       torch.nn.Linear(784, 256),
       torch.nn.ReLU(),
       torch.nn.Linear(256, 10)
   )
   
   # Wrap it for tracking
   wrapped_model = ModelWrapper(model)
   
   # Compute alignment metrics
   metric = get_metric("rayleigh_quotient")()
   inputs = torch.randn(100, 784)
   activations = wrapped_model.extract_activations(inputs)
   scores = metric.compute(inputs=activations[0], weights=model[0].weight)

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   user_guide/installation
   user_guide/quickstart
   examples/basic_usage

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   user_guide/experiments
   user_guide/configuration
   user_guide/metrics
   user_guide/pruning_strategies

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/index
   api/core
   api/models
   api/metrics
   api/experiments
   api/pruning
   api/data
   api/training
   api/analysis
   api/utils
   api/external

.. toctree::
   :maxdepth: 2
   :caption: Examples & Tutorials
   
   examples/index
   examples/basic_usage

.. toctree::
   :maxdepth: 2
   :caption: Reference Documentation
   
   ALIGNMENT_MODULE_GUIDE
   METRICS_REFERENCE
   METRICS_IMPLEMENTATION_DETAILS
   ALL_METRICS_LIST

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide
   
   developer_guide/architecture
   developer_guide/internal/index
   contributing
   changelog

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources
   
   BUILD_DOCUMENTATION

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 

Alignment Framework Documentation
=================================

A comprehensive framework for neural network alignment analysis, providing tools for computing alignment metrics, running experiments, and analyzing results.

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   user_guide/getting_started
   user_guide/metrics
   user_guide/pruning
   user_guide/experiments
   user_guide/visualization

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/metrics
   api/models
   api/pruning
   api/experiments
   api/analysis

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   examples/basic_usage
   examples/pruning_demo
   examples/experiment_workflow
   examples/custom_metrics

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide:

   developer_guide/contributing
   developer_guide/architecture
   developer_guide/adding_metrics
   developer_guide/testing

Features
--------

* **Comprehensive Metrics**: Rayleigh quotient, mutual information, spectral analysis, and more
* **Advanced Pruning**: Multiple strategies with low/high/random modes and parallel execution
* **Experiment Framework**: Structured experiments with automatic tracking and analysis
* **Visualization Tools**: Publication-ready plots and interactive dashboards
* **GPU Optimization**: Efficient computation with automatic memory management
* **Extensible Design**: Easy to add custom metrics and experiments

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install -e .

Basic usage:

.. code-block:: python

   from alignment import ModelWrapper, get_metric
   from alignment.pruning import get_pruning_strategy
   
   # Wrap your model
   wrapped_model = ModelWrapper(model)
   
   # Compute metrics
   metric = get_metric('rayleigh_quotient')
   scores = metric.compute(inputs=data, weights=model.weight)
   
   # Apply pruning
   strategy = get_pruning_strategy('magnitude')
   mask = strategy.prune(model.layer, amount=0.5)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 