Quickstart Guide
================

This guide shows the shortest path from installation to a working NodeLens
experiment. Most users should start with a YAML config and the shared runner,
then move to direct Python APIs only when they need a custom workflow.

.. contents:: Table of Contents
   :local:
   :depth: 2

Installation
------------

Basic installation:

.. code-block:: bash

   git clone https://github.com/KempnerInstitute/NodeLens.git
   cd NodeLens
   pip install -e .

Full installation with optional dependencies:

.. code-block:: bash

   pip install -e .[all]

Run A Config
------------

The main entry point is ``scripts/run_experiment.py``. It loads a YAML config,
creates the requested model and dataset, computes metrics, and writes results to
a timestamped output directory.

.. code-block:: bash

   python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml

For a larger vision pruning workflow:

.. code-block:: bash

   python scripts/run_experiment.py \
     --config configs/vision_prune/resnet18_cifar10_full.yaml

For an LLM channel-analysis workflow:

.. code-block:: bash

   python scripts/run_experiment.py \
     --config configs/prune_llm/llama3_8b_unified.yaml

Output Layout
-------------

Experiment outputs are written under the configured output directory or the
``--base-output-dir`` argument. A typical job directory contains:

.. code-block:: text

   experiment_config.yaml
   logs/
   results/
   figures/
   analysis/

Use ``results/`` for numeric outputs, ``figures/`` for generated plots, and
``experiment_config.yaml`` to confirm the exact settings used by the run.

Use Metrics Directly
--------------------

If you already have layer inputs and weights, call metrics directly:

.. code-block:: python

   from nodelens.metrics import get_metric, list_metrics

   print(list_metrics())

   metric = get_metric("rayleigh_quotient")
   scores = metric.compute(inputs=layer_inputs, weights=layer_weights)

Activation statistics, information metrics, redundancy metrics, gradient-based
scores, and SCAR loss-proxy metrics are all available through the same
``get_metric`` registry.

Wrap A Model
------------

``ModelWrapper`` can track activations for selected layers of a PyTorch model.

.. code-block:: python

   import torch
   import torchvision.models as models
   from nodelens.models import ModelWrapper

   model = models.resnet18(weights=None)
   wrapper = ModelWrapper(model, tracked_layers=["layer1.0.conv1"])

   x = torch.randn(4, 3, 224, 224)
   output, activations = wrapper.forward_with_activations(x)
   weights = wrapper.get_layer_weights(layers=["layer1.0.conv1"])

   print(activations["layer1.0.conv1"].shape)
   print(weights["layer1.0.conv1"].shape)

Choose A Starting Config
------------------------

.. list-table::
   :header-rows: 1

   * - Goal
     - Starting point
   * - Fast smoke test
     - ``configs/examples/mnist_basic.yaml``
   * - Small vision pruning run
     - ``configs/examples/resnet_pruning.yaml``
   * - Full ResNet/CIFAR-10 pruning workflow
     - ``configs/vision_prune/resnet18_cifar10_full.yaml``
   * - Fast LLM smoke test
     - ``configs/examples/gpt2_fast_test.yaml``
   * - LLM FFN channel analysis
     - ``configs/prune_llm/llama3_8b_unified.yaml``

Common Adjustments
------------------

- Use ``--base-output-dir outputs/my_run`` to keep results in a predictable
  location.
- Reduce batch size or the number of evaluation batches when debugging memory
  issues.
- Start from an existing config and change one part at a time: model, dataset,
  tracked layers, metrics, or pruning settings.
- For LLM runs, confirm model access and cache location before launching long
  jobs.

Next Steps
----------

- :doc:`/user_guide/experiments` - Experiment workflow details
- :doc:`/user_guide/metrics` - Available metrics and inputs
- :doc:`/user_guide/pruning` - Pruning strategies and mask behavior
- :doc:`/user_guide/configuration` - YAML configuration options
