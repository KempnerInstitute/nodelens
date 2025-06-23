#!/usr/bin/env python

import os
import sys
import random
import traceback
import argparse
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# DDP import is not directly used, torch.nn.parallel.DistributedDataParallel is used by experiment if needed
# from torch.nn.parallel import DistributedDataParallel as DDP 

# Assuming your new codebase modules (the ones that contain your config and experiment):
from alignment_refac1.config import ExperimentConfig
# Import the factory function to get the correct experiment class
from alignment_refac1.experiments.alignment_experiments import get_experiment_class 

# Basic logger for the main DDP runner process (before workers spawn and set up their own)
# This will go to console.
main_runner_logger = logging.getLogger("DDP_RUNNER")

def ddp_worker(rank, world_size, cfg):
    """
    Each worker process:
      - init process group
      - set device
      - modify cfg to reflect rank's device
      - build and run the specified AlignmentExperiment
    """
    # Each worker can have its own logger if needed, or rely on Experiment's logger
    worker_logger = logging.getLogger(f"DDP_WORKER_RANK_{rank}")
    # The Experiment class will set up more detailed file/console logging via setup_logging

    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(cfg.ddp_port)
        
        backend = getattr(cfg, 'ddp_backend', 'nccl')
        dist.init_process_group(backend, rank=rank, world_size=world_size)

        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        cfg.device = str(device)
        cfg.ddp_rank = rank
        cfg.ddp_world_size = world_size
        cfg.ddp_local_rank = rank

        # Use worker_logger for messages specific to this worker's setup phase
        worker_logger.info(f"Worker initialized. World_size={world_size}, Port={cfg.ddp_port}, Device={device}, Backend={backend}")

        ExperimentClass = get_experiment_class(cfg.experiment_type)
        # The ExperimentClass __init__ will call setup_logging based on cfg
        experiment = ExperimentClass(cfg)
        
        results, nets = experiment.execute_experiment()

    except Exception as e:
        worker_logger.error(f"ERROR => {e}\n{traceback.format_exc()}")
        # No need to raise e again if mp.spawn join=True, as it will propagate
        # However, if we want to ensure the entire script exits non-zero, raising is good.
        raise # Re-raise the exception to ensure mp.spawn sees it and potentially aborts other processes
    finally:
        if dist.is_initialized(): # Check if initialized before trying to destroy
            dist.destroy_process_group()
        worker_logger.info(f"Exiting worker rank {rank}...")

def main():
    # Basic configuration for the main_runner_logger (console only)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser(description="DDP launcher for Alignment Experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)

    world_size = getattr(cfg, "ddp_world_size", 1) # Default to 1 if not specified (for non-DDP test)
    if world_size <= 0:
        main_runner_logger.warning(f"ddp_world_size in config is {world_size}, must be >= 1. Setting to 1.")
        world_size = 1
    
    cfg.ddp_world_size = world_size # Ensure cfg reflects the world_size being used

    if world_size > 1:
        ddp_port = getattr(cfg, "ddp_port", None) # Check if port is already in config
        if ddp_port is None:
            ddp_port = random.randint(20000, 30000)
            cfg.ddp_port = ddp_port # Store it in cfg for ddp_worker usage
        
        main_runner_logger.info(f"Launching {world_size} DDP processes with MASTER_PORT={ddp_port}.")
        # Ensure CUDA is available if world_size > 1 and backend is nccl (common case)
        if not torch.cuda.is_available() and getattr(cfg, 'ddp_backend', 'nccl') == 'nccl':
            main_runner_logger.error("CUDA not available, but ddp_world_size > 1 and backend is nccl. DDP will likely fail.")
            # Potentially exit or force CPU backend if appropriate, for now just warn.

        if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
            main_runner_logger.warning(
                f"Requested DDP world size {world_size} but only {torch.cuda.device_count()} GPUs available/detected. "
                f"Ensure this is intended (e.g., for multi-node CPU DDP or if GPUs are masked).")

        torch.multiprocessing.set_start_method("spawn", force=True)
        try:
            mp.spawn(
                ddp_worker,
                nprocs=world_size,
                args=(world_size, cfg),
                join=True
            )
            main_runner_logger.info("All DDP processes finished.")
        except Exception as e:
            main_runner_logger.error(f"DDP run failed with an exception in one of the workers: {e}")
            # No need to print traceback again if worker logged it.
    else:
        main_runner_logger.info("ddp_world_size is 1. Running in single-process mode (no DDP spawn).")
        # Directly call worker logic for rank 0 / single process
        # This makes the script usable for single GPU/CPU runs without DDP setup.
        cfg.device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        cfg.ddp_rank = 0
        cfg.ddp_world_size = 1
        cfg.ddp_local_rank = 0
        ExperimentClass = get_experiment_class(cfg.experiment_type)
        experiment = ExperimentClass(cfg)
        experiment.execute_experiment()
        main_runner_logger.info("Single-process run finished.")

if __name__ == "__main__":
    main()