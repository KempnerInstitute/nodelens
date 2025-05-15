# Alignment Tests

This directory contains test files for the alignment metrics package.

## Test Files

- `test_benchmark.py`: Tests the alignment metrics using the full package imports
- `test_metrics_standalone.py`: Tests the RQ metric implementation in a standalone manner
- `benchmark_ml.py`: Benchmarks different processing approaches (sequential, batched, tensorized)

## Running Tests

To run the tests, you can use the following commands from the project root:

```bash
# Run the basic alignment metric test
python -m tests.test_benchmark

# Run the standalone metric test (no package imports)
python -m tests.test_metrics_standalone

# Run the benchmark to compare different processing approaches
python -m tests.benchmark_ml
```

## Benchmarking Results

The `benchmark_ml.py` script compares three different approaches:

1. **Sequential**: Process networks one at a time
2. **Batched**: Process networks in small batches
3. **Tensorized**: Process all networks simultaneously using tensor operations

In our tests, we observed modest speedups for the batched (1.05x) and tensorized (1.06x) approaches. With larger networks and datasets, these differences would likely become more pronounced. 