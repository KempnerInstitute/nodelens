Alignment Analysis Framework Documentation
==========================================

A framework for analyzing neural network alignment, pruning, and information-theoretic properties.

Overview
--------

The Alignment Analysis Framework provides tools for:

- Computing alignment metrics between neural representations and task structure
- Implementing and testing pruning strategies on neural networks
- Training and analyzing multiple networks with parallel execution
- Evaluating information-theoretic properties of learned representations

Key Features
------------

- 30+ alignment metrics including Rayleigh quotient, mutual information, and spectral methods
- Multiple pruning strategies: magnitude-based, gradient-based, and alignment-based
- Support for vision models (ResNet, VGG, EfficientNet, ViT) and language models
- Flexible experiment framework with YAML configuration
- GPU-optimized implementations

Quick Start
-----------

.. code-block:: python

    from alignment.experiments import GeneralAlignmentExperiment
    from alignment.configs.config_loader import load_config

    config = load_config('configs/examples/resnet18_analysis.yaml')
    experiment = GeneralAlignmentExperiment(config)
    results = experiment.run()

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/experiments
   user_guide/metrics
   user_guide/pruning
   user_guide/configuration

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/index
   reference/metrics
   reference/models
   reference/configuration

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   api/index
   api/experiments
   api/metrics
   api/pruning
   api/models
   api/data

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 1
   :caption: Contributing

   contributing

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
