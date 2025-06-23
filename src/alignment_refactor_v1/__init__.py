"""
Alignment analysis library for neural networks.

This library provides tools for analyzing alignment in neural networks,
particularly focusing on the alignment between neural network weights and
activations, which can provide insights into the learning dynamics of networks.

Main components:
- metrics: Alignment metrics like RQ and MI
- models: Neural network architectures
- datasets: Dataset loading and preprocessing 
- dropout: Dropout implementation and analysis
- experiments: Experiment runners
- utils: Utility functions

Basic usage:
```python
from alignment import MLP, DataSet, RQMetric, progressive_dropout

# Load data and create model
dataset = DataSet.from_torchvision("mnist")
model = MLP(784, [128, 64], 10)
model.train(dataset)

# Apply progressive dropout
results = progressive_dropout([model], dataset, [0.1, 0.2, 0.3], RQMetric())
```
"""

# Models
from alignment_refac1.models import (
    MLP,
    CNN2P2,
    AlignmentNetwork,
)

# Rename CNN2P2 to CNN for backward compatibility
CNN = CNN2P2

# Datasets
from alignment_refac1.datasets import (
    DataSet,
    load_dataset,
)

# Metrics
from alignment_refac1.metrics import (
    AlignmentMetric,
    get_metric,
)

# Dropout
from alignment_refac1.dropout import (
    progressive_dropout,
    eigenvector_dropout,
)

# Config
from alignment_refac1.config import (
    Config,
    load_config,
)

# Utilities for backward compatibility
from alignment_refac1.utils.core import (
    setup_logging,
    timer,
    debug,
    to_numpy,
    to_tensor,
    check_iterable,
    ensure_device,
    timed
)

from alignment_refac1.utils.math import (
    orthogonalize,
    compute_correlation_matrix,
    matrix_angles,
    project_to_subspace,
)

# Training
from alignment_refac1.training import (
    train_network,
    test_network,
)

# Version
__version__ = "0.5.0"

__all__ = [
    # Models
    "MLP",
    "CNN",
    "CNN2P2",
    "AlignmentNetwork",
    
    # Datasets
    "DataSet",
    "load_dataset",
    
    # Metrics
    "AlignmentMetric",
    "get_metric",
    
    # Dropout
    "progressive_dropout",
    "eigenvector_dropout",
    
    # Config
    "Config",
    "load_config",
    
    # Utilities
    "setup_logging",
    "timer",
    "debug",
    "to_numpy",
    "to_tensor",
    "check_iterable",
    "ensure_device",
    "timed",
    "orthogonalize",
    "compute_correlation_matrix",
    "matrix_angles",
    "project_to_subspace",
    
    # Training
    "train_network",
    "test_network",
    
    # Version
    "__version__",
] 