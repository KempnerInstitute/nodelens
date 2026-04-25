Pruning Strategies Guide
========================

This page summarizes the pruning strategy registry. For full experiments, use a
YAML config and ``scripts/run_experiment.py``. Use the direct Python API when a
custom script already owns the model, layer selection, and evaluation loop.

Registry
--------

.. code-block:: python

   from nodelens.pruning import get_pruning_strategy, list_pruning_strategies

   print(list_pruning_strategies())
   strategy = get_pruning_strategy("magnitude")

Main Strategy Families
----------------------

Magnitude-based
   ``magnitude``, ``global_magnitude``, ``iterative_magnitude``. These remove
   low-magnitude weights or channels and are useful default baselines.

Gradient-based
   ``gradient``, ``fisher``, ``momentum``. These use gradients or
   gradient-derived saliency and require a backward pass or stored gradients.

Alignment-based
   ``alignment``, ``global_alignment``, ``cascading_alignment``. These use
   NodeLens metric scores, such as Rayleigh quotient or activation statistics,
   as pruning signals.

Random baselines
   ``random`` and ``bernoulli``. These are useful controls for separating
   metric value from sparsity effects.

LLM baselines
   ``wanda``, ``sparsegpt``, ``owl``, ``llm_pruner``, ``flap``, ``ria``, and
   ``slimllm``. These are used by LLM configs when comparing channel or weight
   pruning methods.

Parallel and adaptive strategies
   ``parallel_mode``, ``tensorized``, ``async_parallel``,
   ``adaptive_movement``, and ``adaptive_sensitivity``. These support larger
   sweeps or adaptive pruning behavior.

Structured Pruning
------------------

Structured pruning removes complete channels or filters. It is the right
choice when the question is about channel-level importance, architecture-level
compression, or hardware-friendly intervention.

Unstructured pruning removes individual weights. It can preserve quality at
higher sparsity, but it answers a different question and usually needs sparse
runtime support for speedups.

Example Configs
---------------

.. code-block:: bash

   python scripts/run_experiment.py --config configs/examples/resnet_pruning.yaml
   python scripts/run_experiment.py --config configs/vision_prune/resnet18_cifar10_full.yaml
   python scripts/run_experiment.py --config configs/prune_llm/llama3_8b_unified.yaml

Best Practices
--------------

- Compare strategies at the same pruning granularity.
- Include random and magnitude controls.
- Keep calibration and evaluation data explicit in the config.
- Treat pruning as an intervention when the goal is interpretability, not only
  as a compression benchmark.
- Store configs with results so runs can be audited later.
