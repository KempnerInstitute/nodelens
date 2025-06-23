# Codebase Reorganization Plan

## Current Issues

### 1. Unclear Separation of Concerns
- **Core vs Utils**: The distinction is unclear. Both contain fundamental functionality.
- **Pruning in Utils**: Pruning is a major feature that deserves its own module, not buried in utils.
- **Model Wrappers**: Currently in models/wrappers.py, but wrappers are a core abstraction.

### 2. Missing Documentation
- Experiments module lacks comprehensive documentation
- Pruning strategies are undocumented
- No clear module-level documentation explaining the purpose of each directory

### 3. Poor Naming Conventions
- Pruning algorithms don't have descriptive names
- File names don't always match their content (e.g., wrappers.py vs wrapper functionality)

### 4. Scattered Functionality
- Pruning code is split between utils/pruning.py and experiments/cascading.py
- Model-related code is split between core and models directories

## Proposed Reorganization

### New Structure
```
src/alignment/
├── core/                   # Core abstractions and interfaces
│   ├── __init__.py
│   ├── base.py            # BaseMetric, BaseModel, etc.
│   ├── registry.py        # Central registry system
│   ├── protocols.py       # Type protocols and interfaces
│   └── wrappers.py        # ModelWrapper (moved from models/)
│
├── models/                 # Model architectures and components
│   ├── __init__.py
│   ├── architectures/     # Specific model implementations
│   │   ├── mlp.py
│   │   ├── cnn.py
│   │   └── transformer.py
│   └── layers/            # Custom layers and modules
│
├── metrics/                # All metric implementations
│   ├── __init__.py
│   ├── information/       # Information-theoretic metrics
│   ├── similarity/        # Similarity metrics
│   ├── spectral/         # Spectral metrics
│   ├── rayleigh/         # Rayleigh quotient metrics
│   └── task_specific/    # Task-specific metrics
│
├── pruning/               # Dedicated pruning module (NEW)
│   ├── __init__.py
│   ├── base.py           # BasePruningStrategy
│   ├── strategies/       # Pruning algorithms
│   │   ├── magnitude.py  # Magnitude-based pruning
│   │   ├── gradient.py   # Gradient-based pruning
│   │   ├── fisher.py     # Fisher information pruning
│   │   ├── lottery.py    # Lottery ticket hypothesis
│   │   └── cascading.py  # Cascading pruning
│   ├── utilities.py      # Pruning utilities
│   └── analysis.py       # Pruning analysis tools
│
├── experiments/           # Experiment runners and configs
│   ├── __init__.py
│   ├── base.py           # BaseExperiment
│   ├── configs/          # Experiment configurations
│   ├── runners/          # Experiment runners
│   │   ├── alignment.py
│   │   ├── pruning.py
│   │   └── distributed.py
│   └── analysis/         # Experiment analysis tools
│
├── data/                  # Data loading and processing
│   ├── __init__.py
│   ├── datasets/         # Dataset implementations
│   ├── loaders.py        # DataLoader utilities
│   └── transforms.py     # Data transformations
│
├── training/              # Training loops and optimizers
│   ├── __init__.py
│   ├── trainers.py       # Training loop implementations
│   ├── optimizers.py     # Custom optimizers
│   └── schedulers.py     # Learning rate schedulers
│
├── analysis/              # Analysis and visualization
│   ├── __init__.py
│   ├── visualizers/      # Visualization tools
│   ├── statistics.py     # Statistical analysis
│   └── reporting.py      # Report generation
│
├── infrastructure/        # Infrastructure and utilities (RENAMED from utils)
│   ├── __init__.py
│   ├── distributed.py    # Distributed computing utilities
│   ├── logging.py        # Logging configuration
│   ├── checkpoint.py     # Checkpointing utilities
│   ├── config.py         # Configuration management
│   ├── gpu_binning.py    # GPU-specific utilities
│   └── tracking.py       # Experiment tracking (renamed)
│
└── examples/              # Example scripts and notebooks
    ├── __init__.py
    ├── tutorials/        # Tutorial notebooks
    └── scripts/          # Example scripts
```

## Key Changes

### 1. New Pruning Module
Create a dedicated `pruning/` module with:
- Clear strategy names (magnitude.py, gradient.py, etc.)
- Base pruning class for consistency
- Separated utilities and analysis tools

### 2. Rename Utils to Infrastructure
- Makes the purpose clearer
- Contains only true infrastructure/utility code
- Pruning moved to its own module

### 3. Move ModelWrapper to Core
- Wrappers are a core abstraction
- Better fits with other core components

### 4. Better Organization of Experiments
- Clear separation between runners, configs, and analysis
- Each experiment type has its own runner

### 5. Clearer Naming Conventions
- Files named after their primary class/functionality
- Consistent naming patterns across modules

## Migration Plan

### Phase 1: Create New Structure
1. Create new directories
2. Move files to new locations
3. Update imports

### Phase 2: Refactor Pruning
1. Extract pruning code from utils and experiments
2. Create clear strategy classes
3. Document each strategy

### Phase 3: Documentation
1. Add README.md to each module
2. Document the purpose of each directory
3. Create module-level docstrings

### Phase 4: Update Examples
1. Update all example scripts
2. Create new pruning examples
3. Update tutorials

## Documentation Structure

Each module should have:
```
module/
├── README.md          # Module overview and usage
├── __init__.py        # Module exports and docstring
└── ...                # Implementation files
```

## Benefits

1. **Clearer Organization**: Each module has a single, clear purpose
2. **Better Discoverability**: Pruning strategies are easy to find and understand
3. **Consistent Structure**: Similar patterns across all modules
4. **Easier Maintenance**: Related code is grouped together
5. **Better Documentation**: Clear module boundaries make documentation easier 