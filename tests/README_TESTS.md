# Alignment Library Tests

This directory contains tests for the alignment library, covering various metrics, data structures, and algorithms.

## Test Organization

The tests are organized into modules that correspond to the library's structure:

- `test_rq_alt_denom.py`: Tests for the alternative Rayleigh Quotient implementation
- `test_node_redundancy.py`: Tests for the node redundancy metric
- `test_mi_projection.py`: Tests for the MI Projected vs Mean Input metric
- `test_weight_similarity.py`: Tests for weight similarity metrics (cosine, dot product, euclidean)
- `test_metrics_standalone.py`: Standalone tests for basic metrics functionality
- `test_benchmark.py`: Integration tests for multiple metrics

## Running Tests

You can run all tests with:

```bash
cd /path/to/alignment
python -m unittest discover tests
```

Or run a specific test:

```bash
python -m tests.test_weight_similarity
```

## Test Coverage

The current test suite covers all metrics that were recently ported from the old metrics_utils.py:

1. **Rayleigh Quotient Metrics**:
   - Standard RQ computation (`rq`)
   - Alternative RQ with different denominator (`rq_alt_denom`)

2. **MI Metrics**:
   - Gaussian approximation MI (`mi_gaussian`)
   - Direct binning MI (`mi_direct`) 
   - MI between projected input and mean input (`mi_proj_vs_mean_input`)

3. **Redundancy Metrics**:
   - Gaussian redundancy between neurons (`redundancy_gaussian`)
   - Node redundancy based on input correlations (`node_redundancy`)

4. **Weight Similarity Metrics**:
   - Cosine similarity (`weight_cosine_similarity`)
   - Dot product similarity (`weight_dot_similarity`)
   - Euclidean distance (`weight_euclidean_distance`)

## Status of Benchmark Files

The following benchmark files outside the src directory have been analyzed:

- `direct_pruning_test.py`: Used for directly testing pruning functionality. It can be used as reference for testing but does not need to be included in the formal test directory.

- `test_dropout_scaling.py`: Tests the scaling factor calculation for high dropout ratios. A good candidate for porting to a formal unit test.

- `benchmark_dropout_strategies.py`: Benchmark script comparing sequential vs. multi-strategy dropout approaches. This is more of a performance benchmark than a test.

- `benchmark_network_training.py`: Benchmarks network training with different configurations.

Many of these files are useful as reference implementations and for performance testing but should not be considered formal unit tests.

## Recommendations for Next Steps

1. **Add tests for the Partial Information Decomposition (PID) metrics**
   - Create test file similar to others for PID components

2. **Add tests for high-level functions**
   - `compute_metrics_for_layers()`
   - `compute_all_node_scores()`
   - `compute_pairwise_metric()`

3. **Consider merging/removing duplicate test scripts**
   - `test_metrics_standalone.py` function could be incorporated into the metric-specific tests

4. **Add integration tests**
   - Test the metrics in the context of actual neural networks
   - Test metrics with real-world datasets

5. **Add benchmarks as a separate category**
   - Move performance-related scripts to a dedicated benchmarks directory 