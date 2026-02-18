API Reference
=============

This section contains the complete API reference for the alignment framework.

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

**alignment.core**
   Base classes, registries, and core functionality

**alignment.models**
   Model wrappers and architecture definitions

**alignment.metrics**
   36+ metrics for neural network analysis

**alignment.experiments**
   Experiment runners and configurations

**alignment.pruning**
   Pruning strategies and utilities

Supporting Modules
~~~~~~~~~~~~~~~~~~

**alignment.data**
   Dataset wrappers and data processing

**alignment.training**
   Training loops and optimization

**alignment.analysis**
   Result analysis and visualization

**alignment.infrastructure**
   Configuration, logging, distributed computing

Quick Links
-----------

Most Common Classes
~~~~~~~~~~~~~~~~~~~

- :class:`alignment.experiments.base.ExperimentConfig` - Configure experiments
- :class:`alignment.metrics.RayleighQuotient` - Primary alignment metric
- :class:`alignment.models.ModelWrapper` - Wrap models for analysis
- :class:`alignment.experiments.ProgressiveDropoutExperiment` - Main pruning experiment
- :class:`alignment.pruning.strategies.MagnitudePruning` - Standard pruning method

Key Functions
~~~~~~~~~~~~~

- :func:`alignment.core.get_metric` - Get metric by name
- :func:`alignment.core.get_experiment` - Get experiment by type
- :func:`alignment.core.list_metrics` - List available metrics
- :func:`alignment.infrastructure.configuration.load_config` - Load YAML config
- :func:`alignment.analysis.load_results` - Load experiment results 