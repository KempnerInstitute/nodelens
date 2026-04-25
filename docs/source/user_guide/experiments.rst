Experiments Guide
=================

NodeLens experiments are usually launched from YAML configs. The runner chooses
the experiment class from the config, loads the model and dataset, computes the
requested metrics, and writes a structured output directory.

Experiment Types
----------------

``alignment_analysis``
   General metric analysis for smaller models and vision workflows.

``cluster_analysis``
   Metric-space clustering, pruning, and halo-style redundancy analysis for
   vision models.

``llm_alignment``
   LLM activation/gradient capture, channel metrics, ablation probes, and
   structured FFN pruning.

Run From The Command Line
-------------------------

.. code-block:: bash

   python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml

Use ``--base-output-dir`` to choose where job directories are written:

.. code-block:: bash

   python scripts/run_experiment.py \
     --config configs/vision_prune/resnet18_cifar10_full.yaml \
     --base-output-dir outputs/resnet18_cifar10

Run From Python
---------------

For custom scripts, load a config and instantiate the matching experiment
class directly.

.. code-block:: python

   from nodelens.configs.config_loader import load_config
   from nodelens.experiments import (
       ClusterAnalysisExperiment,
       GeneralAlignmentExperiment,
       LLMAlignmentExperiment,
   )

   config = load_config("configs/examples/mnist_basic.yaml")

   if config.experiment_type == "llm_alignment":
       experiment = LLMAlignmentExperiment(config)
   elif config.experiment_type == "cluster_analysis":
       experiment = ClusterAnalysisExperiment(config)
   else:
       experiment = GeneralAlignmentExperiment(config)

   results = experiment.run()

Output Structure
----------------

A typical experiment directory contains:

.. code-block:: text

   experiment_config.yaml
   logs/
   results/
   figures/
   analysis/

The exact files depend on the experiment type. LLM runs usually include
per-layer metric scores, pruning summaries, calibration metadata, and
evaluation outputs. Vision workflows often include pruning curves, clustering
diagnostics, and metric visualizations.

Choosing A Config
-----------------

Start from the closest existing config:

.. list-table::
   :header-rows: 1

   * - Use case
     - Configs
   * - Small smoke tests
     - ``configs/examples/*.yaml``
   * - Vision pruning and clustering
     - ``configs/vision_prune/*.yaml``
   * - LLM channel metrics and SCAR runs
     - ``configs/prune_llm/*.yaml``

When creating a new experiment, change one axis at a time: model, dataset,
tracked layers, metrics, pruning strategy, or evaluation settings. This makes
result differences easier to interpret.

Result Analysis
---------------

Use ``scripts/run_analysis.py`` for post-hoc analysis when an experiment has
already written results:

.. code-block:: bash

   python scripts/run_analysis.py \
     --results-dir outputs/my_run \
     --output-dir outputs/my_run/analysis_extra

Project-specific aggregation scripts live under ``projects/`` or in the
project's public artifact bundle when a study needs extra figure and table
generation logic.
