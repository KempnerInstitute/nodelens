#!/bin/bash

# Benchmark script for testing tensorized network training
# This script runs multiple benchmark configurations to compare
# training speeds for different numbers of networks

ROOT_DIR="/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/alignment"
cd $ROOT_DIR

# Set up common parameters
EPOCHS=1
DEVICE="cuda"
BATCH_SIZE=64
HIDDEN_SIZES="32,16"

echo "----------------------------------------------"
echo "Network Training Method Benchmark"
echo "----------------------------------------------"
echo "Date: $(date)"
echo "Device: $DEVICE"
echo "Hidden sizes: $HIDDEN_SIZES"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "----------------------------------------------"
echo ""

# Run benchmarks with different network counts
for NUM_NETWORKS in 1 3 5 10 20
do
    echo "=== Running benchmark with $NUM_NETWORKS networks ==="
    python benchmark_network_training.py \
        --num_networks $NUM_NETWORKS \
        --hidden_sizes $HIDDEN_SIZES \
        --epochs $EPOCHS \
        --device $DEVICE \
        --batch_size $BATCH_SIZE
    echo ""
done

echo "----------------------------------------------"
echo "Benchmark completed"
echo "----------------------------------------------" 