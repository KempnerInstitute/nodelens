Pruning Guide
=============

NodeLens uses pruning as both an intervention tool and a compression baseline.
The same metric scores used for interpretability can be turned into masks, then
the pruned model can be evaluated to test whether those scores identify
functionally important channels or weights.

Available Strategies
--------------------

Registered strategies include:

- ``magnitude`` and ``global_magnitude``
- ``gradient``, ``fisher``, and ``momentum``
- ``alignment`` and ``global_alignment``
- ``eigenvector``
- ``movement`` and ``adaptive_movement``
- ``random`` and ``bernoulli``
- LLM baselines such as ``wanda`` and ``sparsegpt``

List strategies from Python:

.. code-block:: python

   from nodelens.pruning import list_pruning_strategies

   print(list_pruning_strategies())

Use A Strategy Directly
-----------------------

For low-level scripts, create a strategy from the registry:

.. code-block:: python

   from nodelens.pruning import PruningConfig, get_pruning_strategy

   config = PruningConfig(amount=0.5, pruning_mode="low")
   strategy = get_pruning_strategy("magnitude", config=config)
   mask = strategy.prune(layer, amount=0.5)

The exact method signature depends on the strategy. Config-driven experiments
are the safer entry point for full-model pruning because they handle layer
selection, dependency constraints, evaluation, and result logging.

Run A Pruning Config
--------------------

Vision example:

.. code-block:: bash

   python scripts/run_experiment.py \
     --config configs/vision_prune/resnet18_cifar10_full.yaml

LLM example:

.. code-block:: bash

   python scripts/run_experiment.py \
     --config configs/prune_llm/llama3_8b_unified.yaml

Structured And Unstructured Masks
---------------------------------

Unstructured pruning removes individual weights. It is useful for sparsity
experiments, but it may need sparse kernels to produce wall-clock speedups.

Structured pruning removes complete channels, filters, or FFN units. It is
coarser, but it is easier to connect to architecture-level interventions and
hardware-friendly compression.

NodeLens supports both settings through strategy-specific options and YAML
configs. For LLM FFN studies, structured channel pruning is often the relevant
setting because the goal is to ask which whole channels are functionally
important.

Interpreting Results
--------------------

Pruning experiments typically report:

- the requested sparsity or pruning fraction
- the layer or channel groups that were masked
- model performance after applying the mask
- metric summaries for protected, pruned, or retained channels
- optional figures and JSON summaries for downstream analysis

When comparing pruning methods, keep the pruning granularity fixed. A
structured channel-pruning result should not be treated as directly comparable
to an unstructured weight-pruning result unless the goal is explicitly to
compare different deployment regimes.

Best Practices
--------------

- Start with a small config and verify the output layout before launching a
  large run.
- Compare each informed strategy against random and magnitude baselines.
- Record the calibration dataset, evaluation dataset, sparsity, and mask
  granularity.
- For LLM runs, keep structured and unstructured baselines clearly labeled.
- Use ablation probes when the goal is scientific interpretation rather than
  only compression quality.
