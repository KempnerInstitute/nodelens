# Network Alignment Analysis

Welcome to the Network Alignment Analysis documentation. This project provides tools and methodologies for understanding the structure of neural networks through alignment analysis.

## Overview

The Network Alignment Analysis toolkit is designed to help researchers and practitioners analyze neural networks by measuring alignment between weight matrices and input activations. This approach provides insights into network structure, training dynamics, and pruning strategies.

Key capabilities include:

- Measuring various alignment metrics between neural networks
- Conducting pruning experiments with different strategies
- Training multiple networks in parallel with optimized implementations
- Analyzing network properties during training and evaluation

## Getting Started

```bash
# Clone the repository
git clone https://github.com/your-username/alignment
cd alignment

# Create and activate the environment
mamba env create -f environment.yml
mamba activate networkAlignmentAnalysis

# Install the package
pip install -e .[all]
```

## Quick Links

- [Documentation Overview](documentation.md)
- [Metrics System](metrics/README.md)
- [Experiment Framework](experiment/README.md)
- [Usage Guide](usage.md)
- [API Reference](api/README.md)

## Citation

If you use this codebase in your research, please cite it as:

```bibtex
@software{alignment,
  author = {Your Team},
  title = {Network Alignment Analysis},
  url = {https://github.com/your-username/alignment},
  year = {2023},
} 