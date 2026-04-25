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
- :class:`nodelens.experiments.GeneralAlignmentExperiment` - General metric experiment
- :class:`nodelens.experiments.ClusterAnalysisExperiment` - Vision clustering and pruning experiment
- :class:`nodelens.pruning.strategies.MagnitudePruning` - Standard pruning method

Key Functions
~~~~~~~~~~~~~

- :func:`nodelens.metrics.get_metric` - Get metric by name
- :func:`nodelens.metrics.list_metrics` - List available metrics
- :func:`nodelens.configs.config_loader.load_config` - Load YAML config
- :func:`nodelens.pruning.get_pruning_strategy` - Get pruning strategy by name
