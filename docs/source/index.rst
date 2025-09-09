.. Neural Network Alignment documentation master file

Alignment Analysis Framework Documentation
==========================================

A comprehensive framework for analyzing neural network alignment, pruning, and information-theoretic properties.

Overview
--------

The Alignment Analysis Framework provides tools for:

- Alignment Analysis: Measure how neural representations align with data and task structure
- Pruning Experiments: Test various pruning strategies and their effects on model performance
- Multi-Network Analysis: Train and analyze multiple networks in parallel
- Information Theory Metrics: Compute mutual information, Rayleigh quotients, and other metrics
- Comprehensive Visualization: Generate plots and reports for analysis

Key Features
------------

- 30+ Alignment Metrics: Including Rayleigh quotient, mutual information, spectral metrics
- Multiple Pruning Strategies: Magnitude, gradient, random, and alignment-based pruning
- Flexible Experiments: Support for various experiment types and configurations
- GPU Optimized: Efficient implementations with automatic device management
- Extensible Design: Easy to add custom metrics and strategies

Quick Start
-----------

.. code-block:: python

    from alignment.experiments import GeneralAlignmentExperiment, GeneralAlignmentConfig

    # Configure experiment
    config = GeneralAlignmentConfig(
        experiment_name="mnist_alignment",
        dataset_name="mnist",
        model_name="mlp",
        hidden_sizes=[128, 64],
        num_epochs=10,
        compute_alignment=True,
        alignment_metrics=["rayleigh_quotient", "mutual_information_gaussian"]
    )

    # Run experiment
    experiment = GeneralAlignmentExperiment(config)
    results = experiment.run()

    # Analyze results
    print(f"Final accuracy: {results['final_metrics']['accuracy']}")
    print(f"Alignment scores: {results['alignment_metrics']}")

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
   :caption: API Reference

   api/experiments
   api/metrics
   api/pruning
   api/models
   api/data
   api/analysis

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer_guide/architecture
   developer_guide/contributing
   developer_guide/testing
   developer_guide/internal/index

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/basic_alignment
   examples/pruning_analysis
   examples/multi_network
   examples/custom_metrics

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   changelog
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 