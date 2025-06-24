.. Neural Network Alignment documentation master file

Neural Network Alignment Framework
==================================

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg
   :target: https://pytorch.org/
   :alt: PyTorch Version

A comprehensive framework for studying neural network alignment properties, 
implementing various pruning strategies, and analyzing network behavior through information-theoretic metrics.

Key Features
------------

* **36+ Alignment Metrics**: Comprehensive suite including Rayleigh quotient, mutual information, spectral metrics, and more
* **Modular Architecture**: Clean separation of concerns with dedicated modules for models, metrics, experiments, and utilities
* **Advanced Pruning**: Multiple strategies with low/high/random modes, parallel execution, and tensorized operations
* **Comprehensive Experiments**: Fully configurable experiment system supporting all models, datasets, and metrics
* **Automatic Analysis**: Built-in visualization, reporting, and statistical analysis tools
* **GPU Optimized**: Efficient implementations with automatic memory management
* **Extensible Design**: Easy to add custom metrics, models, and experiments

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

   from alignment import ModelWrapper, get_metric
   import torch

   # Create and wrap a model
   model = torch.nn.Sequential(
       torch.nn.Linear(784, 256),
       torch.nn.ReLU(),
       torch.nn.Linear(256, 10)
   )
   wrapped_model = ModelWrapper(model)
   
   # Compute alignment metrics
   metric = get_metric("rayleigh_quotient")()
   inputs = torch.randn(100, 784)
   activations = wrapped_model.extract_activations(inputs)
   scores = metric.compute(inputs=activations[0], weights=model[0].weight)

Running Experiments
-------------------

The framework provides multiple ways to run experiments:

**Quick Demo** - Basic introduction:

.. code-block:: bash

   python examples/quick_demo.py

**Standard Experiment** - Complete workflow:

.. code-block:: bash

   python examples/standard_alignment_experiment.py

**Comprehensive Experiment** - Full framework capabilities:

.. code-block:: bash

   # With full configuration
   python examples/comprehensive_alignment_experiment.py \
       --config configs/comprehensive_alignment_config.yaml

   # Quick test
   python examples/comprehensive_alignment_experiment.py \
       --config configs/quick_test_config.yaml

   # Override parameters
   python examples/comprehensive_alignment_experiment.py \
       --config configs/quick_test_config.yaml \
       --model_name resnet50 --dataset_name cifar10

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   user_guide/installation
   user_guide/quickstart
   user_guide/getting_started
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
   api/infrastructure

.. toctree::
   :maxdepth: 2
   :caption: Examples & Tutorials
   
   examples/index
   examples/basic_usage
   examples/comprehensive_experiment
   examples/pruning_demo
   examples/visualization_guide

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

Module Overview
===============

The framework is organized into several key modules:

Core Modules
------------

* **core**: Foundational abstractions, protocols, and registry system
* **models**: Model wrappers and architectures
* **metrics**: 36+ alignment metrics organized by type
* **pruning**: Comprehensive pruning strategies and experiments
* **experiments**: Experiment framework and runners
* **data**: Dataset handling and processing
* **training**: Training utilities and callbacks

Supporting Modules
------------------

* **infrastructure**: Runtime support (distributed computing, storage, configuration)
* **analysis**: Post-experiment analysis, aggregation, reporting, and visualization

Examples Available
------------------

1. **quick_demo.py**: Minimal example showing basic workflow
2. **standard_alignment_experiment.py**: Complete experiment template
3. **pruning_strategies_demo.py**: All pruning features demonstration
4. **pruning_visualization_demo.py**: Visualization capabilities
5. **comprehensive_alignment_experiment.py**: Full framework demonstration with YAML configuration

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 