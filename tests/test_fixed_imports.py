#!/usr/bin/env python3
"""
Test script to verify imports are working correctly after cleanup.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("Testing imports after cleanup...")

# Test importing from metrics_utils directly (should be mostly deprecated)
try:
    from alignment.utils.metrics_utils import AlignmentMetricBase # NodeRedundancyMetric might be here if not removed
    # RQMetric, AlignmentMetricsFactory are removed/ported
    print("✓ Successfully imported remaining components (if any) from metrics_utils")
except ImportError as e:
    print(f"INFO: Could not import from metrics_utils (expected if fully deprecated): {e}")

# Test importing from base.py with the updated imports
try:
    from alignment.models.base import AlignmentNetwork
    print("✓ Successfully imported AlignmentNetwork from models.base")
except ImportError as e:
    print(f"✗ Failed to import from models.base: {e}")

# Verify that alignment_metrics.py no longer exists
try:
    import alignment.alignment_metrics
    print("✗ alignment_metrics module still exists - this should fail!")
except ImportError:
    print("✓ alignment_metrics module correctly removed")

print("Import tests completed.") 