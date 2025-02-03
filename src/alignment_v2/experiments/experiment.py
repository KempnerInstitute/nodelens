# alignment/experiments/experiment.py

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

from alignment.datasets import get_dataset
from alignment.utils import compress_directory


class Experiment(ABC):
    def __init__(self, cfg) -> None:
        """Experiment constructor"""
        self.args = cfg
        self.basename = self.get_basename()  # Register basename of experiment
        self.basepath = Path(self.args.results_path) / self.basename  # Register basepath of experiment
        
        # a list of meta arguments that shouldn't be updated when loading an old experiment
        self.meta_args = ["no_save", "just_plot", "save_networks", "show_params", "show_all", "device"]
        
        # manage device
        if self.args.device is None:
            self.args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # do checks
        if self.args.use_timestamp and self.args.just_plot:
            assert self.args.timestamp is not None, "if use_timestamp=True and plotting stored results, must provide a timestamp"
        
        self.register_timestamp()  # Register timestamp of experiment
        self.run = self.configure_wandb()  # Create a wandb run object (or None depending on args.use_wandb)
        self.device = self.args.device

    def report(self, init=False, args=False, meta_args=False) -> None:
        """Method for programmatically reporting details about experiment"""
        # Report general details about experiment
        if init:
            print(f"Experiment object details:")
            print(f"basename: {self.basename}")
            print(f"basepath: {self.basepath}")
            print(f"experiment folder: {self.get_exp_path()}")
            print("using device: ", self.device)

            # Report any other relevant details
            if self.args.save_networks and self.args.no_save:
                print("Note: setting no_save to True will overwrite save_networks. Nothing will be saved.")

        # Report experiment parameters
        if args:
            for key, val in vars(self.args).items():
                if key in self.meta_args:
                    continue
                print(f"{key}={val}")

        # Report experiment meta parameters
        if meta_args:
            for key, val in vars(self.args).items():
                if key not in self.meta_args:
                    continue
                print(f"{key}={val}")

    def register_timestamp(self) -> None:
        """
        Method for registering formatted timestamp.

        If timestamp not provided, then the current time is formatted and used to identify this particular experiment.
        If the timestamp is provided, then that time is used and should identify a previously run and saved experiment.
        """
        if self.args.timestamp is not None:
            self.timestamp = self.args.timestamp
        else:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.args.use_timestamp:
                self.args.timestamp = self.timestamp

    def get_dir(self, create=True) -> Path:
        """
        Method for return directory of target file using prepare_path.
        """
        # Make full path to experiment directory
        exp_path = self.basepath / self.get_exp_path()

        # Make experiment directory if it doesn't yet exist
        if create and not (exp_path.exists()):
            exp_path.mkdir(parents=True)

        return exp_path

    def get_exp_path(self) -> Path:
        """Method for returning child directories of this experiment"""
        # exp_path is the base path followed by whatever folders define this particular experiment
        # (usually things like ['network_name', 'dataset_name', 'test', 'etc'])
        exp_path = Path("/".join(self.prepare_path()))

        # if requested, will also use a timestamp to distinguish this run from others
        if self.args.use_timestamp:
            exp_path = exp_path / self.timestamp

        return exp_path

    def get_path(self, name, create=True) -> Path:
        """Method for returning path to file"""
        exp_path = self.get_dir(create=create)
        return exp_path / name

    def configure_wandb(self):
        """create a wandb run file and set environment parameters appropriately"""
        if self.args.checkpointing.use_wandb:
            wandb.login()
            run = wandb.init(
                project=self.get_basename(),
                name="",
                config=self.args,
            )

            if str(self.basepath).startswith("/n/home"):
                # ATL Note 240223: We can update the "startswith" list to be
                # a registry of path locations that require WANDB_MODE to be offline
                # in a smarter way, but I think that using /n/ is sufficient in general
                os.environ["WANDB_MODE"] = "offline"

            return run

        return None

    @abstractmethod
    def get_basename(self) -> str:
        """Required method for defining the base name of the Experiment"""
        pass

    @abstractmethod
    def prepare_path(self) -> List[str]:
        """
        Required method for defining a pathname for each experiment.

        Must return a list of strings that will be appended to the base path to make an experiment directory.
        See ``get_dir()`` for details.
        """
        pass

    def get_prms_path(self):
        """Method for loading path to experiment parameters file"""
        return self.get_dir() / "prms.pth"

    def get_results_path(self):
        """Method for loading path to experiment results files"""
        return self.get_dir() / "results.pth"

    def get_network_path(self, name):
        """Method for loading path to saved network file"""
        return self.get_dir() / f"{name}.pt"

    def get_checkpoint_path(self):
        """Method for loading path to network checkpoint file"""
        return self.get_dir() / "checkpoint.tar"

    def _update_args(self, prms):
        """Method for updating arguments from saved parameter dictionary"""
        if prms.keys() > vars(self.args).keys():
            raise ValueError(f"Saved parameters contain keys not found in ArgumentParser:  {set(prms.keys()).difference(vars(self.args).keys())}")

        for ak in vars(self.args):
            if ak in self.meta_args:
                continue  # don't update meta arguments
            if ak in prms and prms[ak] != vars(self.args)[ak]:
                print(f"Requested argument {ak}={vars(self.args)[ak]} differs from saved, which is: {ak}={prms[ak]}. Using saved...")
                setattr(self.args, ak, prms[ak])

    def save_repo(self):
        """Method for saving a copy of the code repo at the time this experiment was run"""
        compress_directory(self.get_dir() / "frozen_repo.zip")

    def save_experiment(self, results):
        """Method for saving experiment parameters and results to file"""
        torch.save(vars(self.args), self.get_prms_path())
        torch.save(results, self.get_results_path())

    def load_experiment(self, no_results=False):
        """Method for loading saved experiment parameters and results"""
        if not self.get_prms_path().exists():
            raise ValueError(f"saved parameters at: f{self.get_prms_path()} not found!")
        if not self.get_results_path().exists():
            raise ValueError(f"saved results at: f{self.get_results_path()} not found!")

        prms = torch.load(self.get_prms_path())
        self._update_args(prms)

        if no_results:
            return None

        return torch.load(self.get_results_path())

    def save_networks(self, nets, id=None):
        """
        Method for saving any networks that were trained
        """
        name = f"net_{id}_" if id is not None else "net_"
        for idx, net in enumerate(nets):
            cname = name + f"{idx}"
            torch.save(net.state_dict(), self.get_network_path(cname))

    def load_networks(self, nets, id=None, check_number=True):
        """
        Method for loading any networks that were trained
        """
        name = f"net_{id}_" if id is not None else "net_"
        pattern = self.get_network_path(name + "*").name
        matches = natsorted([match.stem for match in self.get_dir().rglob(pattern)])
        if check_number:
            msg = f"the number of detected networks with name signature {name}*.pt does not match the number of requested networks ({len(matches)}/{len(nets)})"
            assert len(matches) == len(nets), msg
        for idx, match in enumerate(matches):
            c_state_dict = torch.load(self.get_network_path(match))
            nets[idx].load_state_dict(c_state_dict)
        return nets

    @abstractmethod
    def main(self) -> Tuple[Dict, List[torch.nn.Module]]:
        """
        Required method for operating main experiment functions.
        """
        pass

    @abstractmethod
    def plot(self, results: Dict) -> None:
        """
        Required method for operating main plotting functions.
        """
        pass

    # -- support for main processing loop --
    def prepare_dataset(self, transform_parameters):
        """simple method for getting dataset"""
        return get_dataset(
            self.args.dataset.name,
            build=True,
            dataset_parameters=dict(download=self.args.dataset.download, root=self.args.dataset.path),
            transform_parameters=transform_parameters,
            loader_parameters={"batch_size": self.args.training.batch_size},
            device=self.args.device,
        )

    def plot_ready(self, name):
        """standard method for saving and showing plot when it's ready"""
        if not self.args.no_save:
            plt.savefig(str(self.get_path(name)))
        if self.run is not None:
            self.run.log({name: wandb.Image(plt)})
        if not self.args.show_all:
            plt.show()

    # -------------------------------------------------------------------------
    #                   Additional convenience methods
    # -------------------------------------------------------------------------
    def run(self):
        """
        Top-level method: runs the main experiment or just plots existing results,
        then optionally saves results, networks, and handles final reporting.
        """
        # If user requested just to show saved results, skip main experiment
        if self.args.just_plot:
            self.plot_from_existing()
            return

        # Otherwise, do the main experiment
        results, nets = self.main()

        # Save results to disk
        if not self.args.no_save:
            self.save_experiment(results)

        # If user wants to save trained networks
        if self.args.save_networks and not self.args.no_save:
            self.save_networks(nets)

        # If not only doing plotting, do the plotting
        if not self.args.just_plot:
            self.plot(results)

        return results, nets

    def plot_from_existing(self):
        """
        If 'just_plot' is set, load old results from disk and run the plot method.
        """
        stored = self.load_experiment(no_results=False)
        print("Loaded existing results from disk. Now plotting.")
        self.plot(stored)