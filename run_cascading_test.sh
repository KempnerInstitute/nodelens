#!/bin/bash

# Load necessary modules (if needed)
# module purge
# module load python/3.12.5-fasrc01
# module load cuda/12.4.1-fasrc01
# module load cudnn/8.9.2.26_cuda12-fasrc01

# Define the experiment parameters
CONFIG_FILE="configs/config_alignment_experiment.yaml"
EXPERIMENT_TYPE="progressive_dropout"
PRUNING_MODE="cascading_layer"
RUN_NAME="cascading_test"

echo "Starting alignment experiment with cascading layer pruning at $(date)"
start_time=$(date +%s)

# Add the alignment package to the Python path
export PYTHONPATH=/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/alignment

# Execute the alignment experiment
cd /n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/alignment
python -c "
import logging
import sys
from alignment.config import ExperimentConfig
from alignment.experiments.alignment_experiments import AlignmentExperiment, set_logging_level

# Set up logging
set_logging_level(logging.INFO)

# Load configuration
config_path = '${CONFIG_FILE}'
config = ExperimentConfig.load(config_path)

# Set experiment parameters
config.experiment_type = '${EXPERIMENT_TYPE}'
config.extra.dropout_pruning_mode = '${PRUNING_MODE}'
config.run.name = '${RUN_NAME}'

# Create and run experiment
experiment = AlignmentExperiment(config)
results, networks = experiment.run()

print('Experiment completed successfully')
"

end_time=$(date +%s)
echo "Alignment experiment finished at $(date)"
echo "Total duration: $((end_time - start_time)) seconds." 