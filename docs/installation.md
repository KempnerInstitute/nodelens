# Installation Guide

## Requirements

- Python 3.8 or higher
- PyTorch 1.12 or higher
- CUDA-capable GPU (recommended)

---

## Installation

### Using Conda (Recommended)

```bash
git clone https://github.com/KempnerInstitute/alignment.git
cd alignment
conda env create -f environment.yml
conda activate alignment
pip install -e .
```

### Using Pip

```bash
git clone https://github.com/KempnerInstitute/alignment.git
cd alignment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## Dependencies

### Core

- torch >= 1.12.0
- torchvision >= 0.13.0
- numpy >= 1.21.0
- pyyaml >= 5.4

### Optional

- transformers (for HuggingFace models)
- datasets (for text datasets)
- matplotlib (for visualization)
- seaborn (for enhanced plots)

Install optional dependencies:

```bash
pip install transformers datasets matplotlib seaborn
```

---

## Verification

Test installation:

```bash
python -c "import alignment; print(f'Version: {alignment.__version__}')"
python examples/07_mnist_intelligent_pruning.py
```

---

## Troubleshooting

### ModuleNotFoundError

Ensure package is installed:

```bash
pip install -e .
pip show alignment
```

### CUDA Out of Memory

Reduce batch size in configuration or use CPU:

```bash
export CUDA_VISIBLE_DEVICES=""
```

### Missing Transformers

```bash
pip install transformers
```

---

## Development Installation

For development with testing:

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## Updating

```bash
git pull
conda env update -f environment.yml
# or
pip install -r requirements.txt --upgrade
```
