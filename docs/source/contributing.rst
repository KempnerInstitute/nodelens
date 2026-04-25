Contributing Guide
==================

We welcome contributions to NodeLens. This guide will help you get started.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bugfix
4. Make your changes
5. Submit a pull request
6. Check the project documentation for repository-specific conventions.

Development Setup
-----------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/KempnerInstitute/nodelens.git
   cd nodelens

   # Install in development mode with all extras
   pip install -e ".[all]"

   # Run tests
   pytest tests/

Code Style
----------

We use the following tools to maintain code quality:

- **ruff**: Linting
- **black**: Code formatting
- **isort**: Import sorting
- **mypy**: Type checking

Run all checks:

.. code-block:: bash

   # Format code
   black src/ tests/

   # Sort imports
   isort src/ tests/

   # Run linting
   ruff check src tests

   # Type checking
   mypy src/

Testing
-------

All new features should include tests:

.. code-block:: python

   # tests/test_my_feature.py
   import pytest
   from nodelens.my_module import my_function

   def test_my_function():
       result = my_function(input_data)
       assert result == expected_output

Run tests:

.. code-block:: bash

   # Run all tests
   pytest

   # Run specific test file
   pytest tests/test_metrics.py

   # Run with coverage
   pytest --cov=nodelens

Documentation
-------------

Update documentation for new features:

1. Add docstrings to all functions and classes
2. Update relevant RST files in ``docs/source/``
3. Add examples if applicable
4. Build docs locally to check:

.. code-block:: bash

   cd docs
   make html

Pull Request Guidelines
-----------------------

1. **Title**: Use a clear, descriptive title
2. **Description**: Explain what changes you made and why
3. **Tests**: Ensure all tests pass
4. **Documentation**: Update docs if needed
5. **Release notes**: For paper-facing changes, update the relevant file under ``projects/``

Example PR description:

.. code-block:: text

   ## Summary
   Added new Fisher pruning strategy to the pruning module.

   ## Changes
   - Implemented FisherPruning class in pruning/strategies/gradient.py
   - Added tests in tests/test_pruning_strategies.py
   - Updated documentation in docs/source/user_guide/pruning_strategies.rst

   ## Testing
   - All existing tests pass
   - Added 5 new tests for Fisher pruning
   - Tested on CIFAR-10 with ResNet18

Areas for Contribution
----------------------

We especially welcome contributions in these areas:

- **New Metrics**: Information-theoretic or alignment metrics
- **Pruning Strategies**: Novel pruning algorithms
- **Experiments**: New experiment types for analysis
- **Documentation**: Tutorials, examples, and guides
- **Performance**: Optimizations for large-scale experiments
- **Visualization**: New plotting functions

Questions?
----------

- Open an issue on GitHub for bugs or feature requests
- Start a discussion for general questions
- Check existing issues before creating new ones

Thank you for contributing to NodeLens.
