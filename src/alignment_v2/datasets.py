import torch
import argparse
import sys

from alignment_v2.models.registry import (get_model, get_transform_parameters)
from alignment_v2 import processing
from alignment_v2.config import ExperimentConfig
from alignment_v2.experiments.experiment import Experiment

class GeneralAlignmentExperiment(Experiment):
    """
    This experiment can run training or inference on general networks (e.g. MLP, AlexNet),
    computing alignment using one or multiple alignment methods, either during training
    or inference, based on the new config fields.
    """
    def get_basename(self):
        return "general_alignment_experiment"

    def prepare_path(self):
        return [self.args.model.name, self.args.dataset.name, self.args.optimizer.name]

    def create_networks(self):
        if self.args.optimizer.name == "Adam":
            optim = torch.optim.Adam
        elif self.args.optimizer.name == "SGD":
            optim = torch.optim.SGD
        else:
            raise ValueError(f"optimizer ({self.args.optimizer.name}) not recognized")
        nets = [
            get_model(
                self.args.model.name,
                alignment_layer_names=self.args.model.alignment_layers,
                build=True,
                dataset=self.args.dataset.name,
                dropout=self.args.model.dropout,
            )
            for _ in range(self.args.training.replicates)
        ]
        nets = [net.to(self.device) for net in nets]
        optimizers = [optim(net.parameters(), lr=self.args.optimizer.lr, weight_decay=self.args.optimizer.weight_decay) for net in nets]
        prms = {
            "vals": [self.args.model.name],
            "name": "network",
            "dataset": self.args.dataset.name,
            "dropout": self.args.model.dropout,
            "lr": self.args.optimizer.lr,
            "weight_decay": self.args.optimizer.weight_decay,
        }
        return nets, optimizers, prms

    def prepare_dataset(self, transform_parameters):
        ds_params = dict(root=self.args.dataset.path)
        if self.args.dataset.name.lower() != "imagenet":
            ds_params["download"] = self.args.dataset.download
        return get_dataset(
            self.args.dataset.name,
            build=True,
            dataset_parameters=ds_params,
            transform_parameters=transform_parameters,
            loader_parameters={"batch_size": self.args.training.batch_size},
            device=self.args.device,
        )

    def main(self):
        nets, optimizers, prms = self.create_networks()
        dataset = self.prepare_dataset(get_transform_parameters(self.args.model.name, self.args.dataset.name))
        train_results, test_results = processing.train_networks(self, nets, optimizers, dataset)
        dropout_results, dropout_parameters = processing.progressive_dropout_experiment(
            self, nets, dataset, alignment=test_results.get("alignment", None), train_set=False
        )
        eigen_results = processing.measure_eigenfeatures(self, nets, dataset, train_set=False)
        evec_dropout_results, evec_dropout_parameters = processing.eigenvector_dropout(self, nets, dataset, eigen_results, train_set=False)
        results = dict(
            prms=prms,
            train_results=train_results,
            test_results=test_results,
            dropout_results=dropout_results,
            dropout_parameters=dropout_parameters,
            eigen_results=eigen_results,
            evec_dropout_results=evec_dropout_results,
            evec_dropout_parameters=evec_dropout_parameters,
        )
        return results, nets

    def plot(self, results):
        from alignment_v2.core import plotting
        plotting.plot_train_results(self, results["train_results"], results["test_results"], results["prms"])
        plotting.plot_dropout_results(
            self,
            results["dropout_results"],
            results["dropout_parameters"],
            results["prms"],
            dropout_type="nodes",
        )
        plotting.plot_eigenfeatures(self, results["eigen_results"], results["prms"])
        plotting.plot_dropout_results(
            self,
            results["evec_dropout_results"],
            results["evec_dropout_parameters"],
            results["prms"],
            dropout_type="eigenvectors",
        )

def cli_main():
    parser = argparse.ArgumentParser(description="Run a general alignment experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config")
    args = parser.parse_args(sys.argv[1:])
    cfg = ExperimentConfig.load(args.config)
    experiment = GeneralAlignmentExperiment(cfg)
    if cfg.just_plot:
        experiment.plot_from_existing()
    else:
        results, nets = experiment.run()
        if not cfg.no_save:
            experiment.save_results(results)

if __name__ == "__main__":
    cli_main()