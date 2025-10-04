Available Metrics Reference
============================

This reference lists all available alignment and analysis metrics in the framework.

Alignment Metrics
-----------------

Rayleigh Quotient Family
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric Name
     - Description
   * - ``rayleigh_quotient``
     - Standard Rayleigh quotient measuring alignment between weight and data covariance
   * - ``delta_alignment``
     - Change in alignment during training
   * - ``rq_alternative``
     - Alternative Rayleigh quotient formulation

Information Theory Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric Name
     - Description
   * - ``mutual_information``
     - Mutual information between layer activations
   * - ``mutual_information_gaussian``
     - Gaussian approximation of mutual information
   * - ``conditional_mutual_information``
     - Conditional mutual information analysis
   * - ``pairwise_redundancy_gaussian``
     - Pairwise redundancy using Gaussian approximation
   * - ``gaussian_pid_synergy_mmi``
     - Partial Information Decomposition synergy
   * - ``higher_order_information``
     - Higher-order information measures

Similarity Metrics
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric Name
     - Description
   * - ``cosine_similarity``
     - Cosine similarity between weight vectors
   * - ``weight_similarity``
     - Weight-based similarity measures
   * - ``node_correlation``
     - Correlation between node activations
   * - ``node_redundancy``
     - Redundancy analysis between nodes

Spectral Metrics
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric Name
     - Description
   * - ``spectral_gap``
     - Gap between largest eigenvalues
   * - ``spectral_alignment``
     - Spectral analysis of alignment
   * - ``eigenvalue_distribution``
     - Distribution of eigenvalues

Task-Specific Metrics
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric Name
     - Description
   * - ``classification_alignment``
     - Alignment specific to classification tasks
   * - ``vision_alignment``
     - Computer vision specific alignment measures
   * - ``general_task_alignment``
     - General task-agnostic alignment

Usage Example
-------------

.. code-block:: python

   from alignment.metrics import get_metric
   
   # Single metric
   metric = get_metric("rayleigh_quotient")
   score = metric.compute(layer_weights, layer_inputs)
   
   # Multiple metrics in config
   metrics = ["rayleigh_quotient", "mutual_information_gaussian"]

Configuration in YAML
---------------------

.. code-block:: yaml

   # Basic metrics configuration
   metrics: ["rayleigh_quotient", "mutual_information_gaussian"]
   
   # Advanced metrics with parameters
   metric_configs:
     rayleigh_quotient:
       scale_by_norm: false
       aggregation_op: "mean"
     mutual_information_gaussian:
       bins: 50
       estimation_method: "gaussian"

Metric Parameters
-----------------

Common Parameters
~~~~~~~~~~~~~~~~~

- ``scale_by_norm``: Whether to normalize by weight norms
- ``aggregation_op``: How to aggregate scores ("mean", "max", "sum", "var")
- ``force_cpu``: Force CPU computation for large operations
- ``bins``: Number of bins for histogram-based methods

Information Theory Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``estimation_method``: "gaussian", "kraskov", "histogram"
- ``n_neighbors``: Number of neighbors for k-NN estimators
- ``regularization``: Regularization parameter for stability

Spectral Parameters
~~~~~~~~~~~~~~~~~~~

- ``n_eigenvalues``: Number of eigenvalues to compute
- ``method``: Computation method ("power", "lanczos", "full")
- ``tolerance``: Convergence tolerance for iterative methods
