import torch

from alignment.models.registry import (get_model,
                                       get_transform_parameters)
from alignment.core import processing
from alignment.core import plotting
from alignment.experiments.experiment import Experiment

class AlignmentStatistics(Experiment):
    def get_basename(self):
        return "alignment_stats"

    def prepare_path(self):
        return [self.args.model.name, self.args.dataset.name, self.args.optimizer.name]

    def create_networks(self):
        """
        method for creating networks
        """
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