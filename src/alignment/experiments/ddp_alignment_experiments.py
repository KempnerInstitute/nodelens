#!/usr/bin/env python

import os
import sys
import random
import traceback
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Assuming your new codebase modules (the ones that contain your config and experiment):
from alignment.config import ExperimentConfig
from alignment.experiments.alignment_experiments import GeneralAlignmentExperiment

def ddp_worker(rank, world_size, cfg):
    """
    Each worker process:
      - init process group
      - set device
      - modify cfg to reflect rank's device
      - build and run the GeneralAlignmentExperiment
    """
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(cfg.ddp_port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        cfg.device = device  # so the experiment uses the correct GPU

        print(f"\n[DDP_RANK={rank}] world_size={world_size}, port={cfg.ddp_port}, device={device}")

        # Build & run your GeneralAlignmentExperiment from the new codebase
        experiment = GeneralAlignmentExperiment(cfg)
        if cfg.just_plot:
            experiment.plot_from_existing()
        else:
            results, nets = experiment.run()
            # Only rank=0 might save results
            if rank == 0 and not cfg.no_save:
                experiment.save_results(results)

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