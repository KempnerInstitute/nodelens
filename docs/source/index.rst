NodeLens Documentation
======================

NodeLens is a research codebase for node- and channel-level metrics,
interpretability analysis, and structured interventions. The Python package is
imported as ``nodelens``.

Overview
--------

The codebase provides tools for:

- Computing alignment, information, redundancy, activation, and loss-sensitive metrics
- Capturing activations and gradients from vision models, transformers, and LLMs
- Testing metric-defined channels with ablation, pruning, and sensitivity probes
- Running reproducible experiments from YAML configuration files
- Generating plots, tables, JSON summaries, and manifest files

Key Features
------------

- Metrics including Rayleigh quotient, mutual information, redundancy, synergy,
  activation statistics, gradient scores, curvature scores, and SCAR loss proxies
- Structured pruning strategies for channel-level model analysis
- Support for vision models and Hugging Face causal language models
- Project workflows under ``projects/`` that show complete applied analyses
- Config-driven entry points for both small smoke tests and large LLM studies

Quick Start
-----------

.. code-block:: bash

    python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml
    python scripts/run_experiment.py --config configs/vision_prune/resnet18_cifar10_full.yaml
    python scripts/run_experiment.py --config configs/prune_llm/llama3_8b_unified.yaml

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
