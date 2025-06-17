# Alignment Metrics Framework - Refactoring Plan

## Overview

This document outlines the comprehensive refactoring plan for the alignment metrics codebase, transforming it into a modern, scalable, and maintainable framework suitable for HPC environments and multi-GPU computing.

## Current State Analysis

### Strengths of Current Codebase
1. **Comprehensive Metrics**: Rich set of alignment metrics including RQ, MI, PID, and weight-based metrics
2. **Experiment Variety**: Multiple experiment types (progressive dropout, layer-isolated, cascading)
3. **Flexible Configuration**: YAML-based configuration system with dataclasses
4. **DDP Support**: Basic distributed training support already exists

### Areas for Improvement
1. **Code Organization**: Large monolithic files (metrics.py has 1400+ lines)
2. **Separation of Concerns**: Mixed responsibilities in single modules
3. **Memory Management**: Inconsistent handling of CPU/GPU memory for large operations
4. **Extensibility**: Adding new metrics/experiments requires modifying core files
5. **Type Safety**: Incomplete type annotations in many places
6. **Testing**: Limited test coverage and no clear testing structure

## Refactoring Architecture

### Core Design Principles

1. **Protocol-Based Design**: Use Python protocols for all major interfaces
2. **Registry Pattern**: Central registration for metrics, models, experiments
3. **Composition Over Inheritance**: Favor composition for flexibility
4. **Distributed-First**: Built-in support for multi-GPU/distributed computing
5. **Memory-Aware**: Automatic CPU offloading for large operations
6. **Type-Safe**: Full type annotations throughout

### Module Structure

```
src/alignment_refactor/
├── core/                    # Core abstractions
│   ├── protocols.py        # Interface definitions
│   ├── registry.py         # Component registration
│   └── base.py            # Base implementations
├── metrics/                # Metric implementations
│   ├── rayleigh/         # Rayleigh Quotient-based metrics
│   ├── information/       # Information-theoretic metrics
│   └── similarity/       # Similarity-based metrics
├── models/                # Model wrappers
│   ├── wrappers.py       # Model wrapping functionality
│   └── architectures/    # Specific architectures
├── experiments/          # Experiment runners
│   └── runners/         # Specific experiments
├── data/                # Data handling
│   └── datasets/       # Dataset implementations
├── analysis/           # Analysis tools
│   └── visualizers/   # Visualization modules
└── utils/             # Utilities
```

## Implementation Phases

### Phase 1: Core Infrastructure ✅
- [x] Create protocol definitions
- [x] Implement registry system
- [x] Create base classes
- [x] Set up module structure

### Phase 2: Metrics Migration ✅
- [x] Create metrics base module
- [x] Implement Rayleigh Quotient metric
- [x] Implement remaining alignment metrics (DeltaAlignment)
- [x] Implement information-theoretic metrics (MI, Redundancy)
- [x] Implement similarity-based metrics (Cosine, Weight similarity)

### Phase 3: Model Wrappers ✅
- [x] Create ModelWrapper base class
- [x] Implement activation tracking
- [x] Add dropout mask support
- [x] Create AlignmentNetwork compatibility wrapper

### Phase 4: Data Module
- [ ] Create dataset wrapper protocol
- [ ] Implement MNIST, CIFAR, ImageNet wrappers
- [ ] Add data preprocessing utilities
- [ ] Implement distributed data loading

### Phase 5: Experiments
- [ ] Create experiment base class
- [ ] Migrate progressive dropout
- [ ] Migrate layer-isolated pruning
- [ ] Migrate cascading pruning
- [ ] Add new experiment types

### Phase 6: Analysis & Visualization
- [ ] Create result aggregators
- [ ] Implement plotting utilities
- [ ] Add result reporting
- [ ] Create interactive dashboards

### Phase 7: Utilities & Polish
- [ ] Distributed computing utilities
- [ ] Checkpoint management
- [ ] Logging configuration
- [ ] Documentation
- [ ] Comprehensive tests

## Key Improvements

### 1. Modular Metric System
```python
# Old way
rq_scores = AlignmentMetrics.RQ(inputs, weights)

# New way
metric = RayleighQuotient(relative=True)
rq_scores = metric.compute(inputs=inputs, weights=weights)
```

### 2. Flexible Model Wrapping
```python
# Automatic activation tracking
wrapper = ModelWrapper(model, tracked_layers=['layer1', 'layer2'])
outputs, activations = wrapper.forward_with_activations(inputs)
```

### 3. Distributed Computing
```python
# Automatic distributed reduction
metric = RayleighQuotient()
scores = metric.compute_distributed(
    inputs, weights, 
    world_size=4, rank=0
)
```

### 4. Memory Management
```python
# Automatic CPU offloading for large operations
metric = RayleighQuotient(force_cpu_for_large_ops=True)
# Automatically uses CPU if tensor size > threshold
```

### 5. Easy Extension
```python
@register_metric("my_metric")
class MyMetric(BaseMetric):
    def compute(self, inputs, weights, outputs):
        # Implementation
        pass
```

## Migration Guide

### For Users
1. Update imports to use new module structure
2. Replace static method calls with instance methods
3. Use configuration objects instead of dictionaries
4. Leverage new distributed computing features

### For Developers
1. Implement new metrics by inheriting BaseMetric
2. Register components using decorators
3. Follow protocol interfaces
4. Add comprehensive type hints
5. Include unit tests for new components

## Testing Strategy

1. **Unit Tests**: Test each metric/component in isolation
2. **Integration Tests**: Test complete pipelines
3. **Performance Tests**: Ensure no regression in speed
4. **Distributed Tests**: Test multi-GPU functionality
5. **Memory Tests**: Verify CPU offloading works correctly

## Next Steps

1. Complete metric implementations
2. Create model wrapper system
3. Implement experiment runners
4. Add comprehensive documentation
5. Create migration scripts for existing experiments
6. Develop test suite
7. Performance benchmarking
8. Create example notebooks

## Benefits

1. **Scalability**: Easy to run on HPC clusters with multiple GPUs
2. **Maintainability**: Clear separation of concerns
3. **Extensibility**: Easy to add new metrics/experiments
4. **Performance**: Optimized memory usage and computation
5. **Usability**: Clean API with good documentation
6. **Reliability**: Type safety and comprehensive testing 