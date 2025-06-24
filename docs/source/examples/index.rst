Examples and Tutorials
======================

This section contains examples and tutorials for using the alignment framework.

Quick Start Examples
--------------------

.. toctree::
   :maxdepth: 1
   
   basic_usage
   comprehensive_experiment

Available Example Scripts
-------------------------

The ``examples/`` directory contains several demonstration scripts:

1. **quick_demo.py** - Minimal Introduction
   
   - Basic model wrapping and metric computation
   - Simple pruning demonstration
   - No configuration needed
   - Runtime: ~1 minute

   .. code-block:: bash
   
      python examples/quick_demo.py

2. **standard_alignment_experiment.py** - Complete Workflow
   
   - Train model on MNIST
   - Compute alignment metrics
   - Compare pruning strategies
   - Generate visualizations
   - Runtime: ~5-10 minutes

   .. code-block:: bash
   
      python examples/standard_alignment_experiment.py

3. **pruning_strategies_demo.py** - Advanced Pruning
   
   - All pruning modes (low/high/random)
   - Parallel pruning execution
   - Tensorized GPU operations
   - Performance comparisons
   - Runtime: ~2-3 minutes

   .. code-block:: bash
   
      python examples/pruning_strategies_demo.py

4. **pruning_visualization_demo.py** - Visualization Features
   
   - Performance plots
   - Multi-seed analysis
   - Comprehensive comparison grids
   - Real pruning demonstrations
   - Runtime: ~2 minutes

   .. code-block:: bash
   
      python examples/pruning_visualization_demo.py

5. **comprehensive_alignment_experiment.py** - Full Framework Demo
   
   - YAML configuration system
   - All models and datasets
   - 36+ alignment metrics
   - Advanced training options
   - Automatic reporting
   - Runtime: Varies by configuration

   .. code-block:: bash
   
      # Quick test
      python examples/comprehensive_alignment_experiment.py \
          --config configs/quick_test_config.yaml
      
      # Full experiment
      python examples/comprehensive_alignment_experiment.py \
          --config configs/comprehensive_alignment_config.yaml

Example Notebooks
-----------------

Interactive Jupyter notebooks are coming soon:

- **Getting Started Tutorial** - Step-by-step introduction
- **Metrics Deep Dive** - Exploring all available metrics
- **Custom Experiments** - Building your own experiments
- **Analysis Workshop** - Using the analysis tools

Configuration Examples
----------------------

The ``configs/`` directory contains example configurations:

- **comprehensive_alignment_config.yaml** - Full configuration with all options documented
- **quick_test_config.yaml** - Minimal configuration for testing

Common Patterns
---------------

Loading and Running Experiments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from alignment.experiments import GeneralAlignmentExperiment
   
   # From configuration file
   experiment = GeneralAlignmentExperiment.from_yaml("config.yaml")
   results = experiment.run()

Computing Metrics on a Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from alignment import ModelWrapper, get_metric
   
   wrapped_model = ModelWrapper(model)
   metric = get_metric("rayleigh_quotient")()
   
   # Forward pass
   outputs, activations = wrapped_model.forward_with_activations(inputs)
   
   # Compute metric
   scores = metric.compute(
       inputs=activations["layer_name_input"],
       weights=model.layer.weight
   )

Batch Processing Multiple Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from alignment.data.processing import BatchMetricProcessor
   
   processor = BatchMetricProcessor(
       metrics=["rayleigh_quotient", "mutual_information_gaussian"],
       device="cuda"
   )
   
   results = processor.process_dataset(dataloader, model)

Next Steps
----------

1. Start with ``quick_demo.py`` to understand the basics
2. Run ``standard_alignment_experiment.py`` for a complete workflow
3. Explore advanced features with the other demos
4. Create your own experiments using ``comprehensive_alignment_experiment.py``
5. Refer to the :doc:`../user_guide/index` for detailed documentation 