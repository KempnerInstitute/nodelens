"""
Test suite for the Neural Network Alignment framework.

This test suite includes:
- Unit tests for individual components
- Integration tests for full experiments
- Performance benchmarks
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path)) 