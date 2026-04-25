Codebase Notes
==============

This page summarizes the parts of NodeLens that contributors usually need to
understand before adding metrics, experiment types, or pruning strategies.

Core extension points:

- ``src/nodelens/core/registry.py`` registers metrics, models, and experiments.
- ``src/nodelens/metrics/`` contains metric implementations.
- ``src/nodelens/models/`` wraps PyTorch and Hugging Face models for activation capture.
- ``src/nodelens/pruning/`` contains masks, pruning configs, and strategies.
- ``src/nodelens/experiments/`` connects configs, data, models, metrics, and evaluation.
- ``configs/`` contains runnable YAML examples.

Keep reusable code in ``src/nodelens`` and keep project-specific workflows under
``projects/``.
