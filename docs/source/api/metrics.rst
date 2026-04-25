Metrics API Reference
=====================

This section provides detailed documentation for all metrics available in LossLens.

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
--------

LossLens provides 36+ metrics for analyzing neural network behavior, organized into several categories:

- **Alignment Metrics**: Measure how well neurons align with input statistics
- **Information Theory Metrics**: Quantify information flow and dependencies
- **Similarity Metrics**: Compare representations across layers or networks
- **Spectral Metrics**: Analyze eigenvalue/eigenvector properties
- **Task-Specific Metrics**: Domain-specific measures

Base Metric Classes
-------------------

.. automodule:: alignment.metrics.base
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: alignment.metrics.base.BaseMetric
   :members:
   :undoc-members:

   All metrics inherit from this base class and must implement:

   - :meth:`compute`: Main computation method
   - :attr:`requires_inputs`: Whether metric needs input activations
   - :attr:`requires_weights`: Whether metric needs weight matrices
   - :attr:`requires_outputs`: Whether metric needs output activations

Alignment Metrics
-----------------

Rayleigh Quotient (RQ)
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: alignment.metrics.rayleigh.RayleighQuotient
   :members:
   :undoc-members:

   **Mathematical Definition:**

   .. math::

      RQ(w) = \frac{w^T C w}{w^T w}

   where :math:`w` is the weight vector and :math:`C` is the input covariance matrix.

   **Parameters:**

   - **scale_by_norm** (*bool*, default=False): If True, divides by weight norm
   - **relative** (*bool*, default=False): If True, normalizes by trace of covariance
   - **epsilon** (*float*, default=1e-8): Small value for numerical stability
   - **force_cpu** (*bool*, default=False): Force computation on CPU for large matrices

   **Example:**

   .. code-block:: python

      from alignment.metrics import RayleighQuotient

      rq = RayleighQuotient(scale_by_norm=True)
      scores = rq.compute(
          inputs=layer_inputs,    # Shape: (batch, input_dim)
          weights=layer_weights   # Shape: (output_dim, input_dim)
      )
      # Returns: tensor of shape (output_dim,)

   **Interpretation:**

   - High RQ: Neuron aligns with high-variance input directions
   - Low RQ: Neuron aligns with low-variance directions
   - Used for importance scoring in pruning

Generalized Rayleigh Quotient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: alignment.metrics.rayleigh.GeneralizedRayleighQuotient
   :members:
   :undoc-members:

   **Mathematical Definition:**

   .. math::

      GRQ(w) = \frac{w^T A w}{w^T B w}

   where :math:`A` and :math:`B` are positive definite matrices.

   **Use Cases:**

   - Comparing alignment with different covariance structures
   - Multi-task learning scenarios
   - Transfer learning analysis

Information Theory Metrics
--------------------------

Mutual Information (MI)
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: alignment.metrics.information.MutualInformationGaussian
   :members:
   :undoc-members:

   **Mathematical Definition:**

   .. math::

      I(X;Y) = H(X) + H(Y) - H(X,Y)

   **Parameters:**

   - **estimation_method** (*str*, default="gaussian"): Method for MI estimation

     - ``"gaussian"``: Assume Gaussian distributions
     - ``"knn"``: k-nearest neighbors estimator
     - ``"binning"``: Histogram-based estimation

   - **normalize** (*bool*, default=False): Normalize to [0,1] using joint entropy
   - **num_samples** (*int*, default=1000): Samples for estimation

   **Example:**

   .. code-block:: python

      from alignment.metrics import MutualInformationGaussian

      mi = MutualInformationGaussian(estimation_method="knn")
      scores = mi.compute(
          inputs=layer_inputs,
          outputs=layer_outputs
      )

Conditional Mutual Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: alignment.metrics.information.ConditionalMutualInformation
   :members:
   :undoc-members:

   **Mathematical Definition:**

   .. math::

      I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)

   **Use Cases:**

   - Analyzing information flow through layers
   - Understanding conditional dependencies
   - Causal analysis in networks

Partial Information Decomposition (PID)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: alignment.metrics.information.PartialInformationDecomposition
   :members:
   :undoc-members:

   **Components:**

   - **Unique Information**: Information that only one variable provides
   - **Redundant Information**: Information shared by multiple variables
   - **Synergistic Information**: Information only available from combination

   **Parameters:**

   - **method** (*str*, default="broja"): PID estimation method

     - ``"broja"``: BROJA estimator (recommended)
     - ``"barrett"``: Barrett's Gaussian PID
     - ``"williams"``: Williams & Beer framework

   - **max_variables** (*int*, default=100): Maximum variables to consider

   **Example:**

   .. code-block:: python

      from alignment.metrics import PartialInformationDecomposition

      pid = PartialInformationDecomposition(method="broja")
      results = pid.compute(
          inputs=layer_inputs,
          outputs=layer_outputs
      )

      # Access components
      unique = results["unique_information"]
      redundant = results["redundant_information"]
      synergy = results["synergistic_information"]

Similarity Metrics
------------------

Centered Kernel Alignment (CKA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: alignment.metrics.similarity.CKA
   :members:
   :undoc-members:

   **Mathematical Definition:**

   .. math::

      CKA(X,Y) = \frac{||Y^T X||_F^2}{||X^T X||_F ||Y^T Y||_F}

   **Parameters:**

   - **kernel** (*str*, default="linear"): Kernel type

     - ``"linear"``: Linear kernel (fast)
     - ``"rbf"``: RBF/Gaussian kernel

   - **threshold** (*float*, default=0.01): Eigenvalue threshold
   - **sigma** (*float*, optional): RBF kernel bandwidth

   **Example:**

   .. code-block:: python

      from alignment.metrics import CKA

      cka = CKA(kernel="rbf", sigma=1.0)
      similarity = cka.compute(
          X=representations1,  # Shape: (n_samples, n_features1)
          Y=representations2   # Shape: (n_samples, n_features2)
      )
      # Returns: scalar in [0, 1]

Canonical Correlation Analysis (CCA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: alignment.metrics.similarity.CCA
   :members:
   :undoc-members:

   **Parameters:**

   - **n_components** (*int*, default=50): Number of canonical components
   - **reg** (*float*, default=1e-3): Regularization parameter
   - **use_pytorch** (*bool*, default=True): Use PyTorch implementation

   **Variants:**

   - **SVCCA**: Singular Vector CCA (with PCA preprocessing)
   - **PWCCA**: Projection Weighted CCA

Procrustes Distance
~~~~~~~~~~~~~~~~~~~

.. autoclass:: alignment.metrics.similarity.ProcrustesDistance
   :members:
   :undoc-members:

   **Description:**

   Measures the distance between two sets of points after optimal rotation,
   translation, and scaling.

   **Use Cases:**

   - Comparing learned representations
   - Analyzing representation drift
   - Cross-model alignment

Spectral Metrics
----------------

Spectral Analysis
~~~~~~~~~~~~~~~~~

.. autoclass:: alignment.metrics.spectral.SpectralAnalysis
   :members:
   :undoc-members:

   **Computed Properties:**

   - Eigenvalue distribution
   - Spectral gap
   - Effective rank
   - Participation ratio

   **Example:**

   .. code-block:: python

      from alignment.metrics import SpectralAnalysis

      spectral = SpectralAnalysis()
      results = spectral.compute(weights=layer_weights)

      eigenvalues = results["eigenvalues"]
      spectral_gap = results["spectral_gap"]
      effective_rank = results["effective_rank"]

Weight Spectral Norm
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: alignment.metrics.spectral.WeightSpectralNorm
   :members:
   :undoc-members:

   **Description:**

   Computes the largest singular value of weight matrices, important for:

   - Lipschitz continuity
   - Generalization bounds
   - Training stability

Task-Specific Metrics
---------------------

Classification Metrics
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: alignment.metrics.task_specific.ClassificationAlignment
   :members:
   :undoc-members:

   **Measures:**

   - Class separation in representation space
   - Within-class vs between-class variance
   - Linear separability

Regression Metrics
~~~~~~~~~~~~~~~~~~

.. autoclass:: alignment.metrics.task_specific.RegressionAlignment
   :members:
   :undoc-members:

   **Measures:**

   - Target-representation correlation
   - Prediction smoothness
   - Feature importance for regression

Advanced Metric Features
------------------------

Metric Collections
~~~~~~~~~~~~~~~~~~

.. autoclass:: alignment.metrics.MetricCollection
   :members:
   :undoc-members:

   **Example:**

   .. code-block:: python

      from alignment.metrics import MetricCollection

      metrics = MetricCollection([
          RayleighQuotient(scale_by_norm=True),
          MutualInformationGaussian(),
          CKA(kernel="linear")
      ])

      results = metrics.compute_all(
          inputs=inputs,
          weights=weights,
          outputs=outputs
      )
      # Returns: dict with metric names as keys

Distributed Computation
~~~~~~~~~~~~~~~~~~~~~~~

All metrics support distributed computation:

.. code-block:: python

   # Automatic distributed reduction
   metric = RayleighQuotient()
   scores = metric.compute_distributed(
       inputs=local_inputs,
       weights=weights,
       world_size=4,
       rank=rank
   )

Memory-Efficient Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large-scale analysis:

.. code-block:: python

   # Automatic CPU offloading
   metric = RayleighQuotient(
       force_cpu_for_large_ops=True,
       cpu_threshold=1e7
   )

   # Batch processing
   metric = MutualInformationGaussian(
       batch_size=100,
       accumulate=True
   )

Custom Metrics
--------------

Creating Custom Metrics
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from alignment.metrics.base import BaseMetric
   from alignment.core import register_metric

   @register_metric("my_custom_metric")
   class MyCustomMetric(BaseMetric):
       """Custom metric implementation."""

       def __init__(self, param1=1.0):
           super().__init__(name="my_custom_metric")
           self.param1 = param1

       @property
       def requires_inputs(self) -> bool:
           return True

       @property
       def requires_weights(self) -> bool:
           return True

       @property
       def requires_outputs(self) -> bool:
           return False

       def compute(self, inputs=None, weights=None, outputs=None, **kwargs):
           # Your computation here
           return scores

Metric Registry
~~~~~~~~~~~~~~~

.. autofunction:: alignment.core.registry.register_metric
.. autofunction:: alignment.core.registry.get_metric
.. autofunction:: alignment.core.registry.list_metrics

Performance Considerations
--------------------------

**Computation Time:**

============== ========= ============
Metric         Speed     Memory Usage
============== ========= ============
RQ             Fast      Low
Linear CKA     Fast      Medium
Gaussian MI    Medium    Medium
KNN MI         Slow      Low
PID            Slow      High
RBF CKA        Slow      High
============== ========= ============

**Optimization Tips:**

1. Use ``force_cpu=True`` for large matrices
2. Enable batch processing for memory constraints
3. Use linear kernels when possible
4. Cache covariance matrices across metrics

See Also
--------

- :doc:`/user_guide/metrics` - User guide for metrics
- :doc:`/api/experiments` - Using metrics in experiments
