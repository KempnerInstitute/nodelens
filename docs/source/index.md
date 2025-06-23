# Alignment Framework Documentation

Welcome to the comprehensive documentation for the Neural Network Alignment Analysis Framework.

## Documentation Structure

### Getting Started
- [Installation Guide](user_guide/installation.md)
- [Quick Start Tutorial](user_guide/quickstart.md)
- [Basic Usage Examples](examples/basic_usage.md)

### User Guides
- [Complete Module Guide](ALIGNMENT_MODULE_GUIDE.md)
- [Metrics Reference](METRICS_REFERENCE.md) - Mathematical descriptions of all 36 metrics
- [All Metrics List](ALL_METRICS_LIST.md) - Quick reference of available metrics
- [Pruning Strategies Guide](user_guide/pruning_strategies.md) - All pruning methods explained

### Module Documentation
- [Analysis Module](../../src/alignment/analysis/README.md) - Result aggregation and reporting
- [Data Module](../../src/alignment/data/README.md) - Dataset handling
- [Experiments Module](../../src/alignment/experiments/README.md) - Experiment framework
- [Infrastructure Module](../../src/alignment/infrastructure/README.md) - System utilities
- [Pruning Module](../../src/alignment/pruning/README.md) - Pruning framework
- [Training Module](../../src/alignment/training/README.md) - Training utilities

### Developer Guides
- [Architecture Guide](developer_guide/architecture.md) - Framework design and principles
- [Codebase Organization](../CODEBASE_ORGANIZATION.md) - Directory structure and conventions
- [Contributing Guide](CONTRIBUTING.md) - How to contribute

### API Reference
- [Core API](api/core.md)
- [Metrics API](api/metrics.md)
- [Models API](api/models.md)
- [Experiments API](api/experiments.md)

### Examples and Tutorials
- [Basic Usage](examples/basic_usage.md)
- [Running Experiments](examples/running_experiments.md)
- [Custom Metrics](examples/custom_metrics.md)
- [Pruning Analysis](examples/pruning_analysis.md)

### Advanced Topics
- [Gaussian MI with Edgeworth Expansions](../../archive/refactoring_docs/GAUSSIAN_MI_SUMMARY.md)
- [Task-Specific Metrics](../../archive/refactoring_docs/TASK_SPECIFIC_REORG_SUMMARY.md)
- [Distributed Computing](user_guide/distributed.md)

## Quick Links

### Most Common Tasks
1. [How to compute alignment metrics](ALIGNMENT_MODULE_GUIDE.md#basic-usage)
2. [How to run a general alignment experiment](examples/running_experiments.md)
3. [How to apply pruning](user_guide/pruning_strategies.md#quick-start)
4. [How to analyze results](../../src/alignment/analysis/README.md)

### Key Concepts
- **Alignment Metrics**: Quantitative measures of how neural network representations align with various properties
- **Model Wrapper**: Interface for extracting activations and weights from any PyTorch model
- **Pruning**: Techniques for removing network parameters while maintaining performance
- **Experiments**: Structured approach to running and tracking alignment analyses

## Framework Overview

The alignment framework provides:

1. **36+ Alignment Metrics** across 6 categories:
   - Information-theoretic (MI, PID, etc.)
   - Rayleigh quotient variants
   - Similarity measures
   - Spectral analysis
   - Task-specific metrics
   - Higher-order interactions

2. **Comprehensive Pruning Suite**:
   - Multiple pruning strategies
   - Structured and unstructured pruning
   - Pruning experiments and analysis

3. **Flexible Experiment Framework**:
   - General alignment experiments
   - Specialized analysis experiments
   - Batch execution and grid search

4. **Analysis and Visualization**:
   - Result aggregation
   - Report generation (HTML, Markdown, JSON)
   - Interactive visualizations

## Getting Help

- **Issues**: Report bugs or request features on GitHub
- **Discussions**: Ask questions in GitHub Discussions
- **Examples**: See the `examples/` directory for working code
- **API Docs**: Comprehensive API documentation in the `api/` section

## Recent Updates

- Added General Alignment Experiment for complete analysis pipelines
- Reorganized pruning module with dedicated experiments
- Improved documentation structure and accessibility
- Added comprehensive analysis and reporting tools

---

*This documentation is continuously updated. For the latest information, check the [GitHub repository](https://github.com/KempnerInstitute/alignment).* 