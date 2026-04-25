Examples and Tutorials
======================

NodeLens examples are primarily configuration-driven. The same entry point,
``scripts/run_experiment.py``, can run small smoke tests, vision pruning jobs,
and LLM channel analyses.

Runnable Configs
----------------

Small examples:

.. code-block:: bash

   python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml
   python scripts/run_experiment.py --config configs/examples/resnet_pruning.yaml
   python scripts/run_experiment.py --config configs/examples/gpt2_fast_test.yaml

Vision pruning and clustering:

.. code-block:: bash

   python scripts/run_experiment.py --config configs/vision_prune/resnet18_cifar10_full.yaml
   python scripts/run_experiment.py --config configs/vision_prune/vgg16_cifar10_full.yaml
   python scripts/run_experiment.py --config configs/vision_prune/mobilenetv2_cifar10_full.yaml

LLM channel analysis and structured FFN pruning:

.. code-block:: bash

   python scripts/run_experiment.py --config configs/prune_llm/llama3_8b_unified.yaml
   python scripts/run_experiment.py --config configs/prune_llm/mistral_7b_unified.yaml
   python scripts/run_experiment.py --config configs/prune_llm/qwen2_7b_unified.yaml

Common Pattern
--------------

Most workflows follow this structure:

.. code-block:: text

   choose a YAML config
       -> run scripts/run_experiment.py
       -> inspect the timestamped output directory
       -> run optional aggregation or plotting scripts

Direct Metric Use
-----------------

Metrics can also be used directly when a script already has layer inputs,
weights, outputs, or gradients.

.. code-block:: python

   from nodelens.metrics import get_metric, list_metrics

   print(list_metrics())

   metric = get_metric("rayleigh_quotient")
   scores = metric.compute(inputs=layer_inputs, weights=layer_weights)

Batch Processing
----------------

For workflows that need several metrics over the same captured tensors, use the
batch processor from ``nodelens.dataops.processing``.

.. code-block:: python

   from nodelens.dataops.processing import BatchMetricProcessor

   processor = BatchMetricProcessor(
       metrics=["rayleigh_quotient", "mutual_information_gaussian"],
       device="cuda",
   )

   results = processor.process_dataset(dataloader, model)

Project Workflows
-----------------

The ``projects/`` directory contains applied workflows that combine configs,
helper scripts, artifact descriptions, and reproduction notes. These folders
are useful when a paper or larger analysis needs more context than a single
YAML file can provide.

Current project:

- ``projects/supernodes_scar/``: loss-sensitive FFN channel analysis and
  structured pruning for LLMs.

Next Steps
----------

- Read the top-level ``README.md`` for the repository overview.
- Read ``docs/usage.md`` for the config-driven workflow.
- Browse ``configs/`` to find the closest starting point for a new experiment.
- Use ``projects/`` when reproducing a specific applied study.
