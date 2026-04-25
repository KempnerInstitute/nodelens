Installation Guide
==================

Requirements
------------

- Python 3.8+
- PyTorch 1.9+
- CUDA toolkit (recommended for GPU support)

Installation
------------

Using Conda (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create and activate the conda environment:

.. code-block:: bash

   git clone <repository-url>
   cd nodelens

   conda env create -f environment.yml
   conda activate nodelens

   pip install -e .

Using Pip
~~~~~~~~~

Install directly from source:

.. code-block:: bash

   git clone <repository-url>
   cd nodelens
   pip install -e .

Verification
------------

Test the installation:

.. code-block:: python

   import nodelens
   from nodelens.metrics import METRIC_REGISTRY

   # List available metrics
   print(METRIC_REGISTRY.list())

GPU Support
-----------

To use GPU acceleration, ensure PyTorch is installed with CUDA support:

.. code-block:: bash

   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Verify CUDA availability:

.. code-block:: python

   import torch
   print(torch.cuda.is_available())

Next Steps
----------

- See :doc:`quickstart` for basic usage
- Check the repository ``examples/`` folder for runnable examples
- Read the top-level README for current API entry points
