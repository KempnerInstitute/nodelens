#!/bin/bash
#SBATCH --job-name=alignment-ddp     # Job name
#SBATCH --partition=kempner          # Partition with GPUs
#SBATCH --nodes=2                    # Number of nodes
#SBATCH --ntasks-per-node=4          # Tasks per node (match GPUs)
#SBATCH --cpus-per-task=16           # CPUs per task
#SBATCH --mem=512G                   # Memory per node
#SBATCH --gres=gpu:4                 # GPUs per node
#SBATCH --time=24:00:00              # Time limit
#SBATCH --mail-type=begin,end        # Email notifications
#SBATCH --account=kempner_lab        # Account to charge

# Set up job directory
JOB_NAME="alignment-experiment"
source cluster/slurm_settings.txt
JOB_DIR=${JOB_FOLDER}/${JOB_NAME}-${SLURM_JOB_ID}
mkdir -p $JOB_DIR

# Set up logging
logfile="${JOB_DIR}/slurm-${SLURM_JOB_ID}.out"
echo "Job started at $(date)" > $logfile
echo "Running on nodes: $SLURM_JOB_NODELIST" >> $logfile

# Set up DDP environment variables
export MASTER_PORT=12355
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

echo "WORLD_SIZE=$WORLD_SIZE" >> $logfile
echo "MASTER_ADDR=$MASTER_ADDR" >> $logfile
echo "MASTER_PORT=$MASTER_PORT" >> $logfile

# Load modules and activate environment
module purge
module load python/3.9.12
module load cuda/11.7

# Activate conda environment
conda activate alignment_env

# Record start time
start_time=$(date +%s)

# Create Python script for the experiment
cat > ${JOB_DIR}/run_experiment.py << 'EOF'
import os
import torch
import torch.distributed as dist
from pathlib import Path

# Import from refactored codebase
from alignment_refactor import ModelWrapper, DatasetWrapper
from alignment_refactor.experiments import ProgressiveDropoutExperiment
from alignment_refactor.experiments.base import ExperimentConfig
from alignment_refactor.utils.distributed import setup_distributed, cleanup_distributed
from alignment_refactor.utils.logging import setup_logging

def main():
    # Setup distributed environment
    setup_distributed()
    
    # Get rank and world size
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    # Setup logging (only on main process)
    if rank == 0:
        setup_logging(Path(os.environ.get('JOB_DIR', '.')))
    
    # Configuration
    config = ExperimentConfig(
        name=f"resnet50_imagenet_ddp",
        description="Progressive dropout analysis on ResNet50",
        
        # Model settings
        model_name="resnet50",
        pretrained=True,
        
        # Dataset settings
        dataset_name="imagenet",
        data_path="/n/holylabs/LABS/kempner_shared/datasets/imagenet",
        batch_size=256 // world_size,  # Scale batch size per GPU
        num_workers=8,
        
        # Experiment settings
        metrics=["rayleigh_quotient", "mi_gaussian", "pid_shared"],
        tracked_layers=[
            "layer1.0.conv1", "layer1.1.conv1", "layer1.2.conv1",
            "layer2.0.conv1", "layer2.1.conv1", "layer2.2.conv1",
            "layer3.0.conv1", "layer3.1.conv1", "layer3.2.conv1",
            "layer4.0.conv1", "layer4.1.conv1", "layer4.2.conv1",
        ],
        
        # Training settings
        train_before_dropout=True,
        training_epochs=90,
        learning_rate=0.1 * (world_size * 256 / 256),  # Linear LR scaling
        optimizer="sgd",
        
        # Distributed settings
        distributed=True,
        world_size=world_size,
        rank=rank,
        
        # Other settings
        checkpoint_dir=f"{os.environ.get('JOB_DIR', '.')}/checkpoints",
        log_dir=f"{os.environ.get('JOB_DIR', '.')}/logs",
        exclude_classification_layer=True,
        scale_by_norm=False,
        force_cpu_for_large_metric_ops=True,
    )
    
    # Create experiment
    experiment = ProgressiveDropoutExperiment(
        config=config,
        dropout_range=(0.0, 0.9, 10),  # 10 dropout levels from 0% to 90%
        dropout_mode="magnitude",      # Drop based on metric magnitude
        compute_metrics_during_training=True,
        metric_computation_interval=10,  # Compute metrics every 10 epochs
    )
    
    # Run experiment
    print(f"Rank {rank}: Starting experiment...")
    results = experiment.run()
    
    # Save results (only on main process)
    if rank == 0:
        experiment.save_results()
        print("Experiment completed successfully!")
    
    # Cleanup
    cleanup_distributed()

if __name__ == "__main__":
    main()
EOF

# Run the experiment using srun
# Each process will be launched on the appropriate node/GPU
echo "Starting distributed training..." >> $logfile
srun python ${JOB_DIR}/run_experiment.py >> $logfile 2>&1

# Record end time
end_time=$(date +%s)
total_time=$((end_time - start_time))

echo "Job completed at $(date)" >> $logfile
echo "Total runtime: $total_time seconds" >> $logfile

# Copy results to a permanent location (only from rank 0 node)
if [[ $SLURM_PROCID -eq 0 ]]; then
    RESULTS_DIR="/n/holylabs/LABS/kempner_lab/Users/$USER/alignment_results/${SLURM_JOB_ID}"
    mkdir -p $RESULTS_DIR
    cp -r ${JOB_DIR}/* $RESULTS_DIR/
    echo "Results copied to: $RESULTS_DIR" >> $logfile
fi 