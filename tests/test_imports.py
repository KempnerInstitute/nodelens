#!/usr/bin/env python3
"""
Test script to verify the imports are working correctly.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Testing imports...")

# Test importing from metrics_utils
try:
    from alignment.utils.metrics_utils import (
        AlignmentMetricBase, 
        RQMetric, 
        AlignmentMetricsFactory
    )
    print("✓ Successfully imported from metrics_utils")
except ImportError as e:
    print(f"✗ Failed to import from metrics_utils: {e}")

# Test importing utils.plotting
try:
    from alignment.utils.plotting import (
        plot_dropout_results,
        plot_experiment_summary
    )
    print("✓ Successfully imported from utils.plotting")
except ImportError as e:
    print(f"✗ Failed to import from utils.plotting: {e}")

# Test importing from metrics
try:
    from alignment.metrics import (
        AlignmentMetric,
        get_metric,
        register_metric
    )
    print("✓ Successfully imported from metrics")
except ImportError as e:
    print(f"✗ Failed to import from metrics: {e}")

# Test importing from models.base
try:
    from alignment.models.base import AlignmentNetwork
    print("✓ Successfully imported AlignmentNetwork from models.base")
except ImportError as e:
    print(f"✗ Failed to import from models.base: {e}")

# Test importing dropout functions
try:
    from alignment.dropout import progressive_dropout, eigenvector_dropout
    print("✓ Successfully imported progressive_dropout from dropout")
except ImportError as e:
    print(f"✗ Failed to import from dropout: {e}")

# Test importing training functions
try:
    from alignment.training import train_network, test_network
    print("✓ Successfully imported training functions")
except ImportError as e:
    print(f"✗ Failed to import training functions: {e}")

# Test importing config
try:
    from alignment.config import Config, load_config
    print("✓ Successfully imported Config")
except ImportError as e:
    print(f"✗ Failed to import Config: {e}")

print("Import tests completed.") 