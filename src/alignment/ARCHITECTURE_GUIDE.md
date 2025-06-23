# Alignment Framework Architecture Guide

This guide explains the architectural design and module organization of the alignment framework.

## Module Overview

### Core Module (`core/`)

**Purpose**: Contains the fundamental abstractions and interfaces that define the framework's architecture.

**Contents**:
- `base.py`: Abstract base classes (BaseMetric, BaseModel, etc.)
- `registry.py`: Central registration system for metrics and models
- `protocols.py`: Type protocols and interfaces for type checking

**When to use Core**:
- Defining new abstract interfaces
- Implementing framework-wide patterns
- Managing the registration system
- Type definitions and protocols

**Example**:
```python
from alignment.core.base import BaseMetric
from alignment.core.registry import register_metric

@register_metric("my_metric")
class MyMetric(BaseMetric):
    # Implementation
```

### Utils Module (`utils/`)

**Purpose**: Contains utility functions and helper classes that support the main functionality.

**Current Issues**:
- Name doesn't clearly indicate purpose
- Contains major features (like pruning) that deserve their own modules
- Mix of infrastructure and feature-specific utilities

**Proposed Rename**: `infrastructure/`

**What belongs in utils/infrastructure**:
- Distributed computing helpers
- Logging configuration
- Checkpointing utilities
- Configuration management
- General-purpose helpers

**What doesn't belong**:
- Pruning (should be its own module)
- Metric-specific utilities (should be in metrics/)
- Model-specific utilities (should be in models/)

## Current Architecture Problems

### 1. Unclear Module Boundaries

**Problem**: The distinction between modules is not always clear.

**Example**: 
- `ModelWrapper` is in `models/wrappers.py` but is a core abstraction
- Pruning is in `utils/` but is a major feature

**Solution**: 
- Move core abstractions to `core/`
- Create dedicated modules for major features

### 2. Scattered Functionality

**Problem**: Related functionality is spread across multiple modules.

**Example**:
- Pruning code in both `utils/pruning.py` and `experiments/cascading.py`
- Model-related code in both `core/` and `models/`

**Solution**: 
- Consolidate related functionality
- Create clear module boundaries

### 3. Poor Discoverability

**Problem**: It's hard to find specific functionality.

**Example**:
- Pruning strategies have generic names
- No clear documentation of what's in each module

**Solution**:
- Use descriptive file names
- Add README.md to each module
- Improve documentation

## Proposed Architecture

### Clear Module Responsibilities

1. **core/**: Framework abstractions and interfaces
   - Base classes
   - Registration system
   - Type protocols
   - Core wrappers (ModelWrapper)

2. **models/**: Neural network architectures and components
   - Architecture implementations
   - Custom layers
   - Model utilities

3. **metrics/**: All metric implementations
   - Organized by category
   - Consistent interface
   - Clear documentation

4. **pruning/**: Dedicated pruning module
   - All pruning strategies
   - Pruning utilities
   - Pruning analysis tools

5. **experiments/**: Experiment runners and analysis
   - Experiment implementations
   - Configuration management
   - Result analysis

6. **infrastructure/**: Support utilities (renamed from utils/)
   - Distributed computing
   - Logging
   - Checkpointing
   - Configuration

7. **data/**: Data loading and processing
   - Dataset implementations
   - Data transformations
   - DataLoader utilities

8. **training/**: Training loops and optimization
   - Training implementations
   - Custom optimizers
   - Learning rate schedulers

9. **analysis/**: Analysis and visualization
   - Visualization tools
   - Statistical analysis
   - Report generation

## Design Principles

### 1. Single Responsibility
Each module should have one clear purpose.

### 2. High Cohesion
Related functionality should be in the same module.

### 3. Low Coupling
Modules should depend on abstractions, not implementations.

### 4. Clear Interfaces
Public APIs should be well-defined and documented.

### 5. Discoverability
It should be easy to find functionality.

## Migration Strategy

### Phase 1: Documentation
- Add README.md to each module
- Document current structure
- Identify issues

### Phase 2: Refactoring
- Move misplaced code
- Create new modules
- Update imports

### Phase 3: Cleanup
- Remove deprecated code
- Consolidate duplicates
- Improve naming

### Phase 4: Testing
- Update tests
- Ensure compatibility
- Fix issues

## Example: Pruning Module Reorganization

### Current Structure
```
utils/
  pruning.py  # All pruning utilities mixed together
experiments/
  cascading.py  # Contains pruning-specific code
```

### Proposed Structure
```
pruning/
  __init__.py
  base.py  # BasePruningStrategy
  strategies/
    magnitude.py  # MagnitudePruning
    gradient.py   # GradientPruning
    sensitivity.py # SensitivityPruning
    random.py     # RandomPruning
  structured/
    channel.py    # ChannelPruning
    filter.py     # FilterPruning
  schedules.py    # Pruning schedules
  utilities.py    # Helper functions
  analysis.py     # Analysis tools
  README.md       # Module documentation
```

## Benefits of Reorganization

1. **Better Organization**: Clear module purposes
2. **Easier Navigation**: Find code quickly
3. **Improved Maintainability**: Related code together
4. **Better Testing**: Clear module boundaries
5. **Enhanced Documentation**: Module-specific docs
6. **Cleaner APIs**: Well-defined interfaces

## Conclusion

The proposed reorganization addresses current architectural issues and provides a cleaner, more maintainable structure. The key improvements are:

- Clear separation of concerns
- Better module naming
- Dedicated modules for major features
- Improved documentation
- Consistent organization patterns

This will make the codebase easier to understand, maintain, and extend. 