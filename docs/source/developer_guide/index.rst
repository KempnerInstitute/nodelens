Developer Guide
===============

This section contains documentation for developers who want to extend or contribute
to NodeLens.

.. toctree::
   :maxdepth: 2

   extensibility
   internal/index

Overview
--------

NodeLens is designed to be highly extensible. You can add:

- **Custom Metrics**: Define new per-neuron alignment metrics
- **Custom Analyzers**: Create new analysis pipelines (clustering, halo, etc.)
- **Custom Pruners**: Implement new pruning strategies
- **Custom Visualizers**: Add new plot types
- **Custom Evaluators**: Define new evaluation methods

See :doc:`extensibility` for detailed instructions and examples.

Internal Documentation
----------------------

The :doc:`internal/index` section contains documentation for maintainers about
codebase organization and documentation structure.
