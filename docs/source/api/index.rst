API Reference
=============

This section contains the complete API reference for NodeLens.

.. toctree::
   :maxdepth: 2
   :caption: Core APIs

   core
   models
   metrics
   experiments
   pruning

.. toctree::
   :maxdepth: 2
   :caption: Supporting APIs

   data
   training
   analysis
   infrastructure

.. toctree::
   :maxdepth: 1
   :caption: Utilities

   utils

Module Overview
---------------

Core Modules
~~~~~~~~~~~~

**nodelens.core**
   Base classes, registries, and core functionality

**nodelens.models**
   Model wrappers and architecture definitions

**nodelens.metrics**
   36+ metrics for neural network analysis

**nodelens.experiments**
   Experiment runners and configurations

**nodelens.pruning**
   Pruning strategies and utilities

Supporting Modules
~~~~~~~~~~~~~~~~~~

**nodelens.data**
   Dataset wrappers and data processing

**nodelens.training**
   Training loops and optimization

**nodelens.analysis**
   Result analysis and visualization

**nodelens.infrastructure**
   Configuration, logging, distributed computing

Quick Links
-----------

Most Common Classes
~~~~~~~~~~~~~~~~~~~

- :class:`nodelens.experiments.base.ExperimentConfig` - Configure experiments
- :class:`nodelens.metrics.RayleighQuotient` - Primary alignment metric
- :class:`nodelens.models.ModelWrapper` - Wrap models for analysis
- :class:`nodelens.experiments.ProgressiveDropoutExperiment` - Main pruning experiment
- :class:`nodelens.pruning.strategies.MagnitudePruning` - Standard pruning method

Key Functions
~~~~~~~~~~~~~

- :func:`nodelens.core.get_metric` - Get metric by name
- :func:`nodelens.core.get_experiment` - Get experiment by type
- :func:`nodelens.core.list_metrics` - List available metrics
- :func:`nodelens.infrastructure.configuration.load_config` - Load YAML config
- :func:`nodelens.analysis.load_results` - Load experiment results
