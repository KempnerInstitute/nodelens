import torch
from alignment_v2.models.registry import get_model
from alignment_v2.experiments import arglib
from alignment_v2.datasets import get_dataset
from alignment_v2 import processing
from alignment_v2 import plotting
from alignment_v2.experiments.experiment import Experiment
import os
import random
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from alignment_v2.train import progressive_dropout  

# python experiment.py alignment_info_stats --save-networks --network MLP --dataset MNIST --use_wandb --dropout_by_layer --epochs 50 --ddp  --num-drops 14 --batch-size 1000 --replicates 10
# python experiment.py alignment_info_stats --save-networks --network MLP --dataset MNIST --use_wandb --epochs 50 --ddp  --num-drops 14 --batch-size 1000 --replicates 10
# python experiment.py alignment_info_stats --save-networks --network AlexNet --dataset ImageNet --use_wandb --dropout_by_layer --epochs 5 --ddp  --num-drops 14 --batch-size 1000
# python experiment.py alignment_info_stats --save-networks --network AlexNet --dataset ImageNet --use_wandb --epochs 5 --ddp  --num-drops 14 --batch-size 1000

class AlignmentStatisticsInfo(Experiment):
    def get_basename(self):
        return "alignment_info_stats"

    def prepare_path(self):
        return [self.args.network, self.args.dataset, self.args.optimizer]

    def make_args(self, parser):
        parser = arglib.add_standard_training_parameters(parser)
        parser = arglib.add_checkpointing(parser)
        parser = arglib.add_dropout_experiment_details(parser)
        parser = arglib.add_network_metaparameters(parser)
        parser = arglib.add_alignment_analysis_parameters(parser)

        # new line: add a ddp argument
        parser.add_argument(
            "--ddp",
            default=False,
            action="store_true",
            help="If set, will run DDP code instead of normal single-GPU training"
        )

        parser.add_argument('--dropout_by_layer', default=False, action='store_true', help="Perform dropout experiment by layer")
        return parser

    def create_networks(self):
        if self.args.optimizer == "Adam":
            optim = torch.optim.Adam
        elif self.args.optimizer == "SGD":
            optim = torch.optim.SGD
        else:
            raise ValueError(f"optimizer ({self.args.optimizer}) not recognized")

        nets = [
            get_model(
                self.args.network,
                build=True,
                dataset=self.args.dataset,
                dropout=self.args.default_dropout,
                ignore_flag=self.args.ignore_flag,
            )
            for _ in range(self.args.replicates)
        ]
        nets = [net.to(self.device) for net in nets]
        optimizers = [optim(net.parameters(), lr=self.args.default_lr, weight_decay=self.args.default_wd) for net in nets]

        prms = {
            "vals": [self.args.network],
            "name": "network",
            "dataset": self.args.dataset,
            "dropout": self.args.default_dropout,
            "lr": self.args.default_lr,
            "weight_decay": self.args.default_wd,
        }
        return nets, optimizers, prms

    def ddp_worker(self, rank, world_size, port, epochs, batch_size, lr, dropout_rate):
        """
        Child process for DDP training. Only rank=0 will save final results to a file.
        """
        try:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(port)
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")

            print(f"[DDP_RANK={rank}] device={device} world_size={world_size}")

            model_name = self.args.network
            dataset_name = self.args.dataset

            net = get_model(
                model_name,
                build=True,
                dataset=dataset_name,
                dropout=dropout_rate,
                ignore_flag=self.args.ignore_flag
            )
            net.to(device)

            ddp_net = DDP(net, device_ids=[rank], output_device=rank)
            loader_params = dict(batch_size=batch_size, shuffle=False, num_workers=4)
            
            dataset = get_dataset(
                dataset_name,
                build=True,
                transform_parameters=ddp_net.module,
                loader_parameters=loader_params,
                device="cuda",
                distributed=True
            )
                        
            print(f"[DDP_RANK={rank}] dataset created")

            optimizer = torch.optim.Adam(ddp_net.parameters(), lr=lr, weight_decay=0)

            # We'll explicitly set run=None in child processes to avoid parent's wandb run
            train_params = dict(
                num_epochs=epochs,
                alignment=True,
                alignment_expansion=False,
                compare_expected=False,
                frequency=1,
                delta_alignment=False,
                run=None
            )

            nets = [ddp_net]
            optimizers = [optimizer]

            print(f"[DDP_RANK={rank}] Starting training..")
            results = processing.train_networks(self, nets, optimizers, dataset, **train_params)

            print(f"[DDP_RANK={rank}] Done training. Keys: {list(results[0].keys()) if isinstance(results, tuple) else list(results.keys())}")

            if rank == 0:
                # do progressive dropout, etc.
                if isinstance(results, tuple):
                    train_res, test_res = results
                    alignment = test_res.get("alignment", None)
                else:
                    alignment = results.get("alignment", None)

                if alignment is None:
                    print("[DDP_RANK=0] no alignment => skip progressive_dropout")
                    final_ddp_results = {}
                else:
                    drop_params = dict(
                        num_drops=self.args.num_drops,
                        by_layer=self.args.dropout_by_layer,
                        train_set=False
                    )                    
                    drop_res = progressive_dropout(nets, dataset, alignment=alignment, **drop_params)
                    if isinstance(drop_res, tuple):
                        print("[DDP_RANK=0] dropout keys:", list(drop_res[0].keys()))
                    else:
                        print("[DDP_RANK=0] dropout keys:", list(drop_res.keys()))

                    final_ddp_results = {
                        "train_results": results,   
                        "dropout_results": drop_res
                    }
                
                # Rank 0 saves the final results to disk
                torch.save(final_ddp_results, "ddp_results.pth")

        except Exception as e:
            print(f"[DDP_RANK={rank}] ERROR => {e}")
            raise e
        finally:
            dist.destroy_process_group()
            print(f"[DDP_RANK={rank}] Exiting rank {rank}...")

    def main(self):
        if self.args.ddp:
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            epochs = self.args.epochs
            batch_size = self.args.batch_size
            lr = self.args.default_lr
            dropout_rate = self.args.default_dropout

            port = random.randint(20000, 30000)
            print(f"Spawning {world_size} processes with port={port}")
            torch.multiprocessing.set_start_method("spawn", force=True)

            mp.spawn(
                self.ddp_worker,
                nprocs=world_size,
                args=(world_size, port, epochs, batch_size, lr, dropout_rate),
                join=True
            )
            print("DDP training finished!")

            ddp_results = {}
            if os.path.isfile("ddp_results.pth"):
                ddp_results = torch.load("ddp_results.pth")

            self.plot(ddp_results)

            return ddp_results, []

        else:
            # Normal single-process training
            nets, optimizers, prms = self.create_networks()
            dataset = self.prepare_dataset(nets[0])

            print("Training the full network (non-DDP mode)...")
            train_results, test_results = processing.train_networks(self, nets, optimizers, dataset)

            print("Performing full-layer dropout experiment...")
            dropout_results_pre, dropout_parameters_pre = processing.progressive_dropout_experiment(
                self, nets, dataset, alignment=test_results.get("alignment", None), train_set=False
            )

            results = dict(
                prms=prms,
                initial_train_results=train_results,
                initial_test_results=test_results,
                full_dropout_results_pre=dropout_results_pre,
                full_dropout_parameters_pre=dropout_parameters_pre,
            )

            # We can do our normal plotting
            self.plot(results)
            return results, nets

    def plot(self, results):
        """
        Main plotting loop
        """
        print("Plotting training and dropout results...")

        if not results:
            print("No results to plot.")
            return

        # 1) For normal training:
        if "initial_train_results" in results and "initial_test_results" in results and "prms" in results:
            plotting.plot_train_results(
                self,
                results["initial_train_results"],
                results["initial_test_results"],
                results["prms"]
            )

            if "full_dropout_results_pre" in results and "full_dropout_parameters_pre" in results:
                print("Plotting full-layer dropout results (pre- and post-retraining)...")
                plotting.plot_dropout_results(
                    self,
                    results["full_dropout_results_pre"],
                    results["full_dropout_parameters_pre"],
                    results["prms"],
                    dropout_type="nodes_pre_retrain_full"
                )

        # 2) For DDP final results
        if "train_results" in results and "dropout_results" in results:
            print("Plotting DDP results:")
            train_res = results["train_results"]
            drop_res = results["dropout_results"]

            if isinstance(train_res, dict) and "loss" in train_res:
                if "loss" in train_res and "accuracy" in train_res:
                    prms = {"vals": ["DDP-Net"], "name": "DDP-Net"}
                    # or you can store prms in ddp_results too
                    plotting.plot_train_results(self, train_res, None, prms)

            # Plot the dropout
            if isinstance(drop_res, dict) and "progdrop_loss_high" in drop_res:
                # minimal prms for plotting
                ddp_prms = {"vals": ["DDP-Net"], "name": "DDP-Net"}
                plotting.plot_dropout_results(
                    self,
                    drop_res,
                    {"by_layer": False, "num_drops": 3},  
                    ddp_prms,
                    dropout_type="ddp_nodes"
                )