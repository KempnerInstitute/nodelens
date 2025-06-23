# Codebase Reorganization - Phase 2

This document describes the second phase of codebase reorganization focused on better logical grouping.

## Major Changes

### 1. Pruning Module Consolidation

**Problem:** Pruning-related experiments were scattered in the experiments folder with unclear names.

**Solution:** Created a unified pruning module structure:

```
pruning/
├── base.py                    # Base pruning classes
├── strategies/                 # Pruning algorithms
│   ├── magnitude.py
│   ├── gradient.py
│   └── random.py
├── structured/                 # Structured pruning
│   ├── channel.py
│   ├── filter.py
│   └── ...
└── experiments/               # NEW: Pruning experiments
    ├── progressive.py         # (was progressive_dropout.py)
    ├── cascading_layer.py     # (was cascading.py)
    ├── layer_wise.py          # (was layer_isolated.py)
    └── eigenvector_based.py   # (was eigenvector.py)
```

**Benefits:**
- All pruning-related code in one place
- Clear hierarchy: strategies → experiments
- Better discoverability

### 2. Improved Naming Conventions

**Files Renamed:**
- `progressive_dropout.py` → `progressive.py` (in pruning context, dropout is implied)
- `cascading.py` → `cascading_layer.py` (clearer about what cascades)
- `layer_isolated.py` → `layer_wise.py` (simpler, clearer)
- `eigenvector.py` → `eigenvector_based.py` (indicates it's a method/approach)

### 3. Module Organization Principles Applied

1. **Domain-Driven Structure**: Code organized by domain (pruning, training, metrics)
2. **Clear Hierarchies**: Base classes → Implementations → Experiments
3. **Self-Contained Modules**: Each module has everything needed for its domain

## Import Changes

The public API is maintained for backward compatibility:

```python
# Old way (still works via re-exports)
from alignment.experiments import ProgressiveDropoutExperiment

# New way (preferred)
from alignment.pruning.experiments import ProgressiveDropoutExperiment
```

## Remaining Issues to Address

### 1. Model Base Class
The `apply_structured_dropout` method in `models/base.py` should either:
- Be moved to pruning module as a utility
- Be renamed to better reflect its purpose
- Have a clearer integration with pruning strategies

### 2. Large Experiment Files
Some experiment files are very large (400+ lines). Consider:
- Breaking into smaller components
- Extracting common utilities
- Creating experiment mixins for shared functionality

### 3. Metrics Organization
Consider creating subcategories in metrics:
```
metrics/
├── alignment/       # Alignment-specific metrics
├── information/     # Information theory metrics
├── similarity/      # Similarity metrics
├── performance/     # Performance metrics (accuracy, etc.)
└── pruning/        # Pruning-specific metrics
```

## Next Steps

1. Update documentation to reflect new structure
2. Add cross-references between related modules
3. Create migration guide for users
4. Consider creating a "recipes" or "examples" folder showing common workflows 