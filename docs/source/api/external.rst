External Components
===================

This section documents external components integrated into the alignment framework.

BROJA 2PID
----------

The framework includes the BROJA 2PID implementation for Partial Information Decomposition.

.. automodule:: alignment.external.BROJA_2PID
   :members:
   :undoc-members:
   :show-inheritance:

This implementation is based on the paper:
"Quantifying Unique Information" by Bertschinger, Rauh, Olbrich, Jost, and Ay (2014).

Usage Example
~~~~~~~~~~~~~

The BROJA 2PID implementation is used internally by the PID metrics. You typically won't need to use it directly, but here's how it works:

.. code-block:: python

   from alignment.metrics.information import PartialInformationDecomposition
   
   pid = PartialInformationDecomposition(method="broja")
   results = pid.compute(inputs=X, outputs=Y) 