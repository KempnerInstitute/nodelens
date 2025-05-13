# Network Alignment Analysis

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://your-username.github.io/alignment/)

This repository is for a project to understand the structure of neural
networks with a method called "alignment". It contains modules which make
doing alignment-related experiments easy and the scripts that run the 
experiments. The repo is equipped to train pytorch models with DDP on an HPC
cluster. You'll find a few brief instructions about how to use the repository
here in the README, but for more information please feel free to reach out!

## Setup
The code requires a basic ML python environment. Setup can be done with a
standard python environment manager like conda (or mamba!). To get started,
clone the repository from GitHub, then navigate to the cloned folder. 

```
mamba env create -f environment.yml
mamba activate networkAlignmentAnalysis
```

## Installation

After creating and activating the environment, you can install the package.

```
pip install -e .[all]
```

There's no unit test, but to check if the install was successful, run the 
following script while in the environment and in the top directory:

```
python src/alignment/experiments/alignment_experiments.py --config configs/config_alignment_experiment.yaml
```


## Documentatio

The codebase is fully documented with comprehensive guides and API references:

- [Main Documentation](doc/DOCUMENTATION.md): Overview of the entire codebase
- [Metrics System](doc/metrics/README.md): Documentation for the metrics system
- [Experiment Framework](doc/experiment/README.md): Guide to running experiments
- [Performance Optimizations](doc/performance/README.md): Tensorized training and multi-strategy dropout
- [API Reference](doc/api/README.md): Comprehensive API reference

### Online Documentation

For a more user-friendly documentation experience, visit our [GitHub Pages site](https://your-username.github.io/alignment/).

### Guides and Tutorials

- [Usage Guide](doc/usage.md): How to use the codebase
- [Pruning Modes](doc/pruning_modes.md): Documentation for different pruning strategies

## Codebase Structure

The codebase is organized into the following directories:

- `src/alignment/`: Core source code implementing alignment metrics and algorithms
- `tests/`: Unit and integration tests
- `scripts/`: Utility scripts for running experiments and analysis
- `benchmarks/`: Performance evaluation scripts
- `configs/`: Configuration files for experiments
- `results/`: Output directory for experiment results
- `doc/`: Documentation

## Contributing
Feel free to contribute to this project by opening issues or submitting pull
requests. It's already a collaborative project, so more minds are great if you
have ideas or anything to contribute!

## License
This project is licensed under the MIT License. If you use anything from this
repository for more than learning about code and/or pytorch, please cite us. 
There's no paper associated with the code at the moment, but you can cite our
GitHub repository URL or email us for any updates about this issue.




