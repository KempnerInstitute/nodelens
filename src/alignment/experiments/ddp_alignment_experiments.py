#!/usr/bin/env python

import os
import sys
import random
import traceback
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# DDP import is not directly used, torch.nn.parallel.DistributedDataParallel is used by experiment if needed
# from torch.nn.parallel import DistributedDataParallel as DDP 

# Assuming your new codebase modules (the ones that contain your config and experiment):
from alignment.config import ExperimentConfig
# Import the factory function to get the correct experiment class
from alignment.experiments.alignment_experiments import get_experiment_class 

def ddp_worker(rank, world_size, cfg):
    """
    Each worker process:
      - init process group
      - set device
      - modify cfg to reflect rank's device
      - build and run the specified AlignmentExperiment
    """
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(cfg.ddp_port) # ddp_port is added to cfg in main
        
        # Use ddp_backend from config
        backend = getattr(cfg, 'ddp_backend', 'nccl') # Default to nccl if not in config
        dist.init_process_group(backend, rank=rank, world_size=world_size)

        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        cfg.device = str(device)  # Store as string for ExperimentConfig compatibility
        cfg.ddp_rank = rank # Store rank and world_size in config for experiment to use
        cfg.ddp_world_size = world_size
        cfg.ddp_local_rank = rank # Assuming one process per GPU

        print(f"\n[DDP_RANK={rank}] world_size={world_size}, port={cfg.ddp_port}, device={device}, backend={backend}")

        # Get the appropriate experiment class based on config
        ExperimentClass = get_experiment_class(cfg.experiment_type)
        experiment = ExperimentClass(cfg) # Pass the modified cfg
        
        # The experiment.run() itself should handle DDP awareness for its internal logic
        # such as data loading, model wrapping, and metric aggregation where necessary.
        # The Experiment base class in alignment_experiments.py now has more DDP logic.
        results, nets = experiment.execute_experiment() # Use the unified execution method

        # Result saving and other rank 0 tasks are typically handled within the experiment
        # or its _run_plotting_and_saving method, which should be DDP-aware.
        # This ddp_worker mainly sets up the DDP environment and launches the experiment.

    except Exception as e:
        print(f"[DDP_RANK={rank}] ERROR => {e}")
        traceback.print_exc()
        raise e
    finally:
        dist.destroy_process_group()
        print(f"[DDP_RANK={rank}] Exiting rank {rank}...")

def main():
    parser = argparse.ArgumentParser(description="DDP trainer for GeneralAlignmentExperiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)

    # e.g. read how many GPUs from cfg.ddp_world_size (or default 4)
    world_size = getattr(cfg, "ddp_world_size", 4)
    ddp_port = random.randint(20000, 30000)
    # store it in cfg for ddp_worker usage
    cfg.ddp_port = ddp_port

    print(f"Launching {world_size} processes with tcp://127.0.0.1:{ddp_port} for DDP.")
    torch.multiprocessing.set_start_method("spawn", force=True)

    mp.spawn(
        ddp_worker,
        nprocs=world_size,
        args=(world_size, cfg),
        join=True
    )
    print("All DDP processes done successfully.")

if __name__ == "__main__":
    main()