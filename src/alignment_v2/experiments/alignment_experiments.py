import torch
import argparse
import sys

from alignment_v2.datasets import get_dataset
from alignment_v2.models.registry import get_model, get_transform_parameters
from alignment_v2 import processing
from alignment_v2.config import ExperimentConfig
from alignment_v2.experiments.experiment import Experiment
from alignment_v2 import plotting

class GeneralAlignmentExperiment(Experiment):
    """
    This experiment can run training or inference on general networks (e.g. MLP, AlexNet),
    computing alignment using one or multiple alignment methods, either during training
    or inference, based on the config.
    """

    def get_basename(self):
        return "general_alignment_experiment"

    def prepare_path(self):
        return [self.args.model.name, self.args.dataset.name, self.args.optimizer.name]

    def create_networks(self):
        if self.args.optimizer.name == "Adam":
            optim_cls = torch.optim.Adam
        elif self.args.optimizer.name == "SGD":
            optim_cls = torch.optim.SGD
        else:
            raise ValueError(f"optimizer ({self.args.optimizer.name}) not recognized")

        nets = []
        for _ in range(self.args.training.replicates):
            net = get_model(
                self.args.model.name,
                alignment_layer_names=self.args.model.alignment_layers,
                build=True,
                dataset=self.args.dataset.name,
                dropout=self.args.model.dropout,
            )
            net.to(self.device)
            nets.append(net)

        optimizers = [
            optim_cls(net.parameters(), lr=self.args.optimizer.lr, weight_decay=self.args.optimizer.weight_decay)
            for net in nets
        ]

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
        """
        Reverting to a simpler approach that doesn't use 'download'.
        """
        # If your config has no 'download' field, remove it entirely.
        ds_params = dict(root=self.args.dataset.path)
        return get_dataset(
            dataset_name=self.args.dataset.name,
            build=True,
            dataset_parameters=ds_params,
            transform_parameters=transform_parameters,
            loader_parameters={"batch_size": self.args.training.batch_size},
            device=self.args.device,
        )

    def main(self):
        nets, optimizers, prms = self.create_networks()

        dataset = self.prepare_dataset(get_transform_parameters(self.args.model.name, self.args.dataset.name))

        # If epochs == 0, effectively skip training (train_networks calls the loop, which sees 0 epochs).
        train_results, test_results = processing.train_networks(self, nets, optimizers, dataset)

        # optional progressive dropout experiment
        dropout_results, dropout_parameters = processing.progressive_dropout_experiment(
            self, nets, dataset, alignment=test_results.get("alignment", None), train_set=False
        )

        # measure eigenfeatures
        eigen_results = processing.measure_eigenfeatures(self, nets, dataset, train_set=False)

        # do targeted dropout with eigenfeatures
        evec_dropout_results, evec_dropout_parameters = processing.eigenvector_dropout(
            self, nets, dataset, eigen_results, train_set=False
        )

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

    def save_results(self, results):
        """
        Simple wrapper calling base Experiment's save_experiment() method,
        ensuring no error if code calls experiment.save_results(results).
        """
        self.save_experiment(results)


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