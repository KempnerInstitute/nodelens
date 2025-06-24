# Config Cleanup and Documentation Summary

## Config Files Cleanup

### Kept (Active Configs)
1. **`configs/comprehensive_alignment_config.yaml`** - Full configuration with ALL options documented
2. **`configs/quick_test_config.yaml`** - Minimal config for quick testing

### Archived (Moved to `_archive/configs/`)
All old config files that were only referenced in archived scripts have been moved:
- config_alignment_experiment.yaml
- training_example.yaml
- config_with_eigenvector.yaml
- config_refactored_test.yaml
- config_pruning_modes.yaml
- config_alignment_stats_v2.yaml
- config_alignment_stats_old.yaml
- config_alignment_stats.yaml

## Documentation Added to Examples

Added comprehensive run instructions to the top of each example file:

### 1. `quick_demo.py`
- Added usage instructions
- Listed requirements
- Described expected output
- Explained that no configuration is needed

### 2. `standard_alignment_experiment.py`
- Added detailed usage instructions
- Listed what the script does automatically
- Specified output files and their locations
- Added requirements section

### 3. `pruning_strategies_demo.py`
- Added usage instructions
- Listed all demonstrated features
- Specified requirements (including optional CUDA)
- Described console output

### 4. `pruning_visualization_demo.py`
- Added usage instructions
- Listed visualization types created
- Specified output file locations
- Described both simulated and real pruning demos

### 5. `comprehensive_alignment_experiment.py`
- Already had detailed instructions
- Enhanced with configuration section
- Added output structure details
- Referenced available config files

## Key Improvements

1. **Cleaner Structure**: Only active configs remain in `configs/`
2. **Clear Documentation**: Every example now has clear run instructions at the top
3. **Self-Contained**: Each script explains its requirements and outputs
4. **No Confusion**: Old configs that don't work with current code are archived

## Usage Pattern

Users can now:
1. Read the header of any example to understand how to run it
2. Find only working configs in the `configs/` directory
3. Understand requirements before running
4. Know exactly what output to expect 