import os
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from natsort import natsorted

import torch
import wandb
from matplotlib import pyplot as plt

class Experiment(ABC):
    def __init__(self, cfg) -> None:
        self.args = cfg
        self.basename = self.get_basename()
        self.basepath = Path(self.args.results_path) / self.basename

        self.meta_args = ["no_save", "just_plot", "save_networks", "show_params", "show_all", "device"]

        if self.args.device is None:
            self.args.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.args.use_timestamp and self.args.just_plot:
            assert self.args.timestamp is not None, "if use_timestamp=True and just_plot=True, need a timestamp"

        self.register_timestamp()
        self.wandb_run = self.configure_wandb()
        self.device = self.args.device

    def report(self, init=False, args=False, meta_args=False) -> None:
        if init:
            print("Experiment object details:")
            print(f"basename: {self.basename}")
            print(f"basepath: {self.basepath}")
            print(f"experiment folder: {self.get_exp_path()}")
            print("using device:", self.device)
            if self.args.save_networks and self.args.no_save:
                print("Warning: no_save=True conflicts with save_networks=True. Nothing will be saved.")

        if args:
            for key, val in vars(self.args).items():
                if key in self.meta_args:
                    continue
                print(f"{key}={val}")

        if meta_args:
            for key, val in vars(self.args).items():
                if key not in self.meta_args:
                    continue
                print(f"{key}={val}")

    def register_timestamp(self) -> None:
        if self.args.timestamp is not None:
            self.timestamp = self.args.timestamp
        else:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.args.use_timestamp:
                self.args.timestamp = self.timestamp

    def get_dir(self, create=True) -> Path:
        exp_path = self.basepath / self.get_exp_path()
        if create and not exp_path.exists():
            exp_path.mkdir(parents=True)
        return exp_path

    def get_exp_path(self) -> Path:
        exp_path = Path("/".join(self.prepare_path()))
        if self.args.use_timestamp:
            exp_path = exp_path / self.timestamp
        return exp_path

    def get_path(self, name, create=True) -> Path:
        return self.get_dir(create=create) / name

    def configure_wandb(self):
        if self.args.checkpointing.use_wandb:
            wandb.login()
            run = wandb.init(
                project=self.get_basename(),
                name="",
                config=self.args,
            )
            if str(self.basepath).startswith("/n/home"):
                os.environ["WANDB_MODE"] = "offline"
            return run
        return None

    @abstractmethod
    def get_basename(self) -> str:
        pass

    @abstractmethod
    def prepare_path(self) -> List[str]:
        pass

    def get_prms_path(self):
        return self.get_dir() / "prms.pth"

    def get_results_path(self):
        return self.get_dir() / "results.pth"

    def get_network_path(self, name):
        return self.get_dir() / f"{name}.pt"

    def get_checkpoint_path(self):
        return self.get_dir() / "checkpoint.tar"

    def _update_args(self, prms):
        if prms.keys() > vars(self.args).keys():
            diff = set(prms.keys()).difference(vars(self.args).keys())
            raise ValueError(f"Saved parameters contain unknown keys: {diff}")

        for ak in vars(self.args):
            if ak in self.meta_args:
                continue
            if ak in prms and prms[ak] != vars(self.args)[ak]:
                print(f"Updating arg {ak} from {vars(self.args)[ak]} to {prms[ak]}")
                setattr(self.args, ak, prms[ak])

    def save_experiment(self, results):
        torch.save(vars(self.args), self.get_prms_path())
        torch.save(results, self.get_results_path())

    def load_experiment(self, no_results=False):
        if not self.get_prms_path().exists():
            raise ValueError(f"No saved params at: {self.get_prms_path()}")
        if not self.get_results_path().exists():
            raise ValueError(f"No saved results at: {self.get_results_path()}")

        prms = torch.load(self.get_prms_path())
        self._update_args(prms)
        if no_results:
            return None
        return torch.load(self.get_results_path())

    def save_networks(self, nets, id=None):
        name = f"net_{id}_" if id else "net_"
        for idx, net in enumerate(nets):
            cname = name + f"{idx}"
            torch.save(net.state_dict(), self.get_network_path(cname))

    def load_networks(self, nets, id=None, check_number=True):
        name = f"net_{id}_" if id else "net_"
        pattern = self.get_network_path(name + "*").name
        matches = natsorted([match.stem for match in self.get_dir().rglob(pattern)])
        if check_number:
            msg = f"# networks in checkpoint {len(matches)} != needed {len(nets)}"
            assert len(matches) == len(nets), msg
        for idx, match in enumerate(matches):
            c_state = torch.load(self.get_network_path(match))
            nets[idx].load_state_dict(c_state)
        return nets

    @abstractmethod
    def main(self) -> Tuple[Dict, List[torch.nn.Module]]:
        pass

    @abstractmethod
    def plot(self, results: Dict) -> None:
        pass

    def plot_ready(self, name):
        if not self.args.no_save:
            plt.savefig(str(self.get_path(name)))
        if self.wandb_run is not None:
            self.wandb_run.log({name: wandb.Image(plt)})
        if not self.args.show_all:
            plt.show()

    def run(self):
        if self.args.just_plot:
            self.plot_from_existing()
            return

        results, nets = self.main()

        if not self.args.no_save:
            self.save_experiment(results)
            if self.args.save_networks:
                self.save_networks(nets)

        if not self.args.just_plot:
            self.plot(results)

        return results, nets

    def plot_from_existing(self):
        stored = self.load_experiment(no_results=False)
        print("Loaded existing results. Now plotting.")
        self.plot(stored)