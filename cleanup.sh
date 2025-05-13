#!/bin/bash
# cleanup.sh - Script to clean up the root directory after reorganization

echo "Cleaning up root directory..."

# Files that have been moved to benchmarks/
echo "Removing files that were moved to benchmarks/"
rm -f benchmark_dropout_strategies.py
rm -f benchmark_network_training.py

# Files that have been moved to tests/
echo "Removing files that were moved to tests/"
rm -f test_dropout_scaling.py
rm -f test_cascading_pruning.py

# Files that have been moved to scripts/
echo "Removing files that were moved to scripts/"
rm -f direct_pruning_test.py
rm -f run_multi_strategy_experiment.py
rm -f run_fixed_experiment.py
rm -f run_cascading_with_plots.py
rm -f run_cascading_test.py
rm -f run_benchmark.sh
rm -f run_cascading_test.sh

# Files that have been moved to _archive/
echo "Removing files that were moved to _archive/"
rm -f debug_pruning_strategies.py

# Log and debug output files
echo "Removing log and debug output files"
rm -f pruning_test_output.log
rm -f direct_pruning_test.log
rm -f debug_output.log
rm -f experiment_output.log
rm -f pruning_debug_results_fixed.txt
rm -f pruning_debug_results.txt

echo "Cleanup complete. All files have been organized into appropriate directories." 