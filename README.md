# Network Alignment Analysis

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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

Create your environment with the following command. Mamba is much faster than
conda, but replace `mamba` with `conda` if you haven't installed `mamba`.

```
mamba env create -f environment.yml
mamba activate networkAlignmentAnalysis
```

If this doesn't work, it's probably because of the pytorch packages. Try 
commenting out the pytorch packages from the environment.yml file, then using
the above line to create and activate your environment. Type `nvidia-smi` in
your terminal to figure out the right CUDA version to use, and install them
using the instructions on the pytorch 
[website](https://pytorch.org/get-started/locally/).

## Installation

After creating and activating the environment, you can install the package.

```
pip install -e .[all]
```

There's no unit test, but to check if the install was successful, run the 
following script while in the environment and in the top directory:

```
python experiment.py configs/config_alignment_stats.yaml
```

Note that you need to at least fill in the two place holders in the config file for dataset path and results path.

For checking HPC cluster related code, there's a MWE 
[here](cluster/ddp-example/), note that details will differ regarding the HPC
system etc.

## Usage and Tutorial
Detailed usage and tutorials can be found in a dedicated markdown file 
[here](docs/usage.md).

## Contributing
Feel free to contribute to this project by opening issues or submitting pull
requests. It's already a collaborative project, so more minds are great if you
have ideas or anything to contribute!

## License
This project is licensed under the MIT License. If you use anything from this
repository for more than learning about code and/or pytorch, please cite us. 
There's no paper associated with the code at the moment, but you can cite our
GitHub repository URL or email us for any updates about this issue.




