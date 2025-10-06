#!/usr/bin/env python3
"""
Build documentation for the Neural Network Alignment framework.
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    """Build the documentation."""
    # Change to docs directory
    docs_dir = Path(__file__).parent / "docs"
    if not docs_dir.exists():
        print(f"Error: Documentation directory not found at {docs_dir}")
        sys.exit(1)

    os.chdir(docs_dir)

    # Check if sphinx is installed
    try:
        import sphinx
    except ImportError:
        print("Sphinx not installed. Installing documentation requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-docs.txt"])

    # Build HTML documentation
    print("Building HTML documentation...")
    result = subprocess.run(["make", "html"], capture_output=True, text=True)

    if result.returncode == 0:
        print("Documentation built successfully!")
        print(f"View documentation at: {docs_dir}/build/html/index.html")
    else:
        print("Error building documentation:")
        print(result.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
