NodeLens Documentation
======================

NodeLens is a research codebase for node- and channel-level metrics,
interpretability analysis, and structured interventions. The Python package is
imported as ``nodelens``.

Overview
--------

The codebase provides tools for:

- Computing alignment metrics between neural representations and task structure
- Implementing and testing pruning strategies on neural networks
- Estimating channel-level loss sensitivity in LLM feed-forward layers
- Evaluating information-theoretic properties of learned representations
- Packaging paper artifacts for public release

Key Features
------------

- Alignment metrics including Rayleigh quotient, mutual information, and spectral methods
- Multiple pruning strategies: magnitude-based, gradient-based, and alignment-based
- Support for vision models (ResNet, VGG, EfficientNet, ViT) and language models
- Flexible experiment framework with YAML configuration
- Paper-specific release folders under ``projects/``

Quick Start
-----------

.. code-block:: python

    from nodelens.experiments import GeneralAlignmentExperiment
    from nodelens.configs.config_loader import load_config

    config = load_config('configs/examples/mnist_basic.yaml')
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
   user_guide/pruning_strategies
   user_guide/configuration

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/index
   reference/metrics
   reference/models
   reference/configuration

.. toctree::
   :maxdepth: 1
   :caption: Contributing

   contributing

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
