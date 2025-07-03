# Training Consolidation Plan

## Overview
This document outlines the plan to consolidate duplicate training implementations across experiments to use the new `ExperimentTrainer` class.

## Current State
Multiple experiments have their own `_train_model()` implementations:
- `general_alignment._train_model()` - Supports multi-network training
- `layer_wise._train_model()` - Basic single network training
- `eigenvector_based._train_model()` - Basic single network training  
- `cascading_layer._train_model()` - Basic single network training
- `standard_alignment.train_model()` - Standalone training function

## Migration Strategy

### Step 1: Update GeneralAlignmentExperiment
1. Replace `_train_model()` with `ExperimentTrainer`
2. Convert training config to `ExperimentTrainingConfig`
3. Preserve multi-network support
4. Update metrics collection

### Step 2: Update Pruning Experiments
For each pruning experiment (layer_wise, eigenvector_based, cascading_layer):
1. Remove `_train_model()` method
2. Add `_create_trainer()` method that returns configured `ExperimentTrainer`
3. Update `run()` method to use trainer
4. Preserve experiment-specific metrics

### Step 3: Update Configuration Classes
1. Add training config fields to base `ExperimentConfig` if not present
2. Ensure all experiments can configure training parameters
3. Maintain backward compatibility

### Step 4: Testing
1. Create test script to verify training behavior is preserved
2. Test multi-network training
3. Test single network training
4. Verify metrics and checkpointing

## Benefits
- Eliminates ~400 lines of duplicate code
- Consistent training behavior across experiments
- Easier to add new training features (mixed precision, distributed, etc.)
- Better testing coverage
- Unified checkpoint format

## Implementation Order
1. GeneralAlignmentExperiment (most complex, proves the approach)
2. Layer-wise pruning (simplest pruning experiment)
3. Other pruning experiments
4. Documentation updates 