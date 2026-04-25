Experiments API Reference
=========================

NodeLens experiments are configuration-driven. The public API centers on the
base experiment classes and the three main experiment families used by the
runner.

Base Experiment Classes
-----------------------

.. automodule:: nodelens.experiments.base
   :members:
   :undoc-members:
   :show-inheritance:

General Alignment Experiments
-----------------------------

.. automodule:: nodelens.experiments.general_alignment
   :members:
   :undoc-members:
   :show-inheritance:

Cluster Analysis Experiments
----------------------------

.. automodule:: nodelens.experiments.cluster_experiments
   :members:
   :undoc-members:
   :show-inheritance:

LLM Experiments
---------------

.. automodule:: nodelens.experiments.llm_experiments
   :members:
   :undoc-members:
   :show-inheritance:

Runner
------

The standard command-line entry point is:

.. code-block:: bash

   python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml

For Python workflows, load a config and instantiate the matching experiment
class from ``nodelens.experiments``.
