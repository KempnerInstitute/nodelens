#!/usr/bin/env python3
"""
Run all tests for the alignment metrics package.

This script runs all the tests in sequence and reports
the results in a unified manner.
"""

import os
import sys
import time
import subprocess
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Define the tests to run
TESTS = [
    {"name": "Standalone Metrics Test", "module": "tests.test_metrics_standalone"},
    {"name": "Basic Standalone Test", "module": "tests.test_standalone"},
    {"name": "Benchmark Test", "module": "tests.benchmark_ml", "skip_normal": True}
]

def run_test(test_info):
    """Run a single test and return the result."""
    print(f"\n{'='*80}")
    print(f"Running {test_info['name']}...")
    print(f"{'='*80}")
    
    cmd = [sys.executable, "-m", test_info["module"]]
    start_time = time.time()
    
    try:
        # Run the test
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        # Print the output
        if result.stdout:
            print(result.stdout)
            
        # Print any errors
        if result.stderr:
            print("ERRORS:")
            print(result.stderr)
            
        success = result.returncode == 0
        status = "SUCCESS" if success else "FAILED"
        print(f"\n{status} in {elapsed:.2f} seconds")
        return success
        
    except Exception as e:
        print(f"Error running test: {e}")
        return False

def main():
    """Run all tests."""
    print(f"Running all tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Track results
    results = []
    
    # Parse arguments
    run_all = "--all" in sys.argv
    
    # Run each test
    for test_info in TESTS:
        if test_info.get("skip_normal", False) and not run_all:
            print(f"\nSkipping {test_info['name']} (use --all to run)")
            continue
            
        success = run_test(test_info)
        results.append({"name": test_info["name"], "success": success})
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for result in results:
        status = "PASSED" if result["success"] else "FAILED"
        all_passed = all_passed and result["success"]
        print(f"{result['name']}: {status}")
    
    print("\nOverall result:", "PASSED" if all_passed else "FAILED")
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 