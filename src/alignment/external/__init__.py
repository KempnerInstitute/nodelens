"""
External dependencies for alignment metrics.
"""

# Import BROJA_2PID if available
try:
    from .BROJA_2PID import BROJA_2PID

    __all__ = ["BROJA_2PID"]
except ImportError:
    __all__ = []
