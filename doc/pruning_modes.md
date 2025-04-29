# Pruning Modes in Alignment Analysis

This document explains the three different pruning modes available in the alignment analysis framework.

## Overview

The alignment framework provides three different methods for pruning nodes based on their Rayleigh Quotient (RQ) values:

1. **Global Pruning** (`global`): Concatenates all nodes from all layers and prunes the lowest X% across all nodes. This was the original behavior in v1 and can result in uneven pruning across layers.

2. **Per-Layer Combined Pruning** (`per_layer_combined`): Prunes exactly X% from each layer separately but applies all pruning simultaneously. This ensures balanced pruning across layers (v2-like behavior).

3. **Per-Layer Independent Pruning** (`per_layer_independent`): Prunes X% from only one layer at a time, generating separate results for each layer. This helps analyze the impact of pruning individual layers.

## Configuration

To specify a pruning mode, set the following parameter in your config file:

```yaml
extra:
  dropout_pruning_mode: "global" | "per_layer_combined" | "per_layer_independent"
  num_drops: 9  # Number of pruning points to test
```

## Pruning Modes Explained

### 1. Global Pruning (`global`)

- **How it works**: All nodes from all layers are concatenated into a single array, sorted by RQ value, and the lowest X% are pruned
- **Effect**: Different layers may have different pruning percentages based on their relative RQ values
- **Use when**: You want to compare the relative importance of nodes across the entire network
- **Config**:
  ```yaml
  dropout_pruning_mode: "global"
  ```

### 2. Per-Layer Combined Pruning (`per_layer_combined`)

- **How it works**: Each layer's nodes are sorted by RQ independently, and exactly X% from each layer are pruned simultaneously
- **Effect**: All layers lose exactly X% of their nodes in a balanced way
- **Use when**: You want balanced pruning across layers while measuring the combined effect
- **Config**:
  ```yaml
  dropout_pruning_mode: "per_layer_combined"
  ```

### 3. Per-Layer Independent Pruning (`per_layer_independent`)

- **How it works**: One layer at a time is pruned by X%, with separate results for each layer
- **Effect**: Creates N different experiments, where N is the number of layers
- **Use when**: You want to analyze the impact of pruning individual layers in isolation
- **Config**:
  ```yaml
  dropout_pruning_mode: "per_layer_independent"
  ```

## Important Notes

1. When using the `per_layer_independent` mode, the results will contain data for each layer separately in the returned metrics.

2. The original v1 codebase defaulted to `global` mode, while v2 implemented functionality closer to `per_layer_combined`.

## Visual Comparison

```
+----------------+-----------------------------+-----------------------------+
|                |          X% Global          |      X% Each Layer          |
|                | (Different % per layer)     | (Exact X% per layer)        |
+----------------+-----------------------------+-----------------------------+
| All Layers     | "global"                    | "per_layer_combined"        |
| Simultaneously | (original v1)               | (v2-like)                   |
+----------------+-----------------------------+-----------------------------+
| One Layer      | N/A                         | "per_layer_independent"     |
| at a Time      |                             | (new option)                |
+----------------+-----------------------------+-----------------------------+
``` 