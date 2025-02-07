import torch
import argparse
import sys

from alignment.datasets import get_dataset
from alignment.models.registry import get_model, get_transform_parameters
from alignment import processing
from alignment.config import ExperimentConfig
from alignment.experiments.experiment import Experiment
from alignment import plotting

class GeneralAlignmentExperiment(Experiment):
    def get_basename(self):
        return "general_alignment_experiment"

    def prepare_path(self):
        return [self.args.model.name, self.args.dataset.name, self.args.training.name]

    def create_networks(self):
        if self.args.training.name == "Adam":
            optim_cls = torch.optim.Adam
        elif self.args.training.name == "SGD":
            optim_cls = torch.optim.SGD
        else:
            raise ValueError(f"Unknown optimizer {self.args.training.name}")

        nets = []
        for _ in range(self.args.training.replicates):
            net = get_model(
                self.args.model.name,
                alignment_layer_names=self.args.model.alignment_layers,
                build=True,
                dataset=self.args.dataset.name,
                dropout=self.args.model.dropout,
            ).to(self.device)
            nets.append(net)

        optimizers = [
            optim_cls(net.parameters(), lr=self.args.training.lr, weight_decay=self.args.training.weight_decay)
            for net in nets
        ]

        prms = dict(
            vals=[self.args.model.name],
            name="network",
            dataset=self.args.dataset.name,
            dropout=self.args.model.dropout,
            lr=self.args.training.lr,
            weight_decay=self.args.training.weight_decay,
        )
        return nets, optimizers, prms

    def prepare_dataset(self, transform_parameters):
        ds_params = dict(root=self.args.dataset.path)
        dataset = get_dataset(
            dataset_name=self.args.dataset.name,
            build=True,
            dataset_parameters=ds_params,
            transform_parameters=transform_parameters,
            loader_parameters={"batch_size": self.args.training.batch_size},
            device=self.args.device,
        )
        return dataset

    def main(self):
        # create
        nets, optimizers, prms = self.create_networks()

        if self.args.training.do_train:
            # do training
            dataset = self.prepare_dataset(get_transform_parameters(self.args.model.name, self.args.dataset.name))
            train_results, test_results = processing.train_networks(self, nets, optimizers, dataset)
        else:
            # skip training, possibly load pre-trained weights if we have them
            # user code to load:
            #   for net in nets: net.load_state_dict(torch.load(...))
            # Then do test if compute_during_inference is true
            train_results = {}
            dataset = self.prepare_dataset(get_transform_parameters(self.args.model.name, self.args.dataset.name))
            test_results = processing.test_networks(self, nets, dataset)  # A hypothetical function

        # do progressive dropout
        dropout_results, dropout_params = processing.progressive_dropout_experiment(
            self, nets, dataset, alignment=test_results.get("alignment", None), train_set=False
        )

        # measure eigenfeatures
        #eigen_results = processing.measure_eigenfeatures(self, nets, dataset, train_set=False)

        # do eigenvector dropout
        #evec_dropout_results, evec_dropout_params = processing.eigenvector_dropout(
        #    self, nets, dataset, eigen_results, train_set=False
        #)

        results = {
            "prms": prms,
            "train_results": train_results,
            "test_results": test_results,
            "dropout_results": dropout_results,
            "dropout_parameters": dropout_params,
        #    "eigen_results": eigen_results,
        #    "evec_dropout_results": evec_dropout_results,
        #    "evec_dropout_parameters": evec_dropout_params,
        }
        return results, nets

    def plot(self, results):

        # check plots config toggles
        if self.args.plots.show_loss:
            plotting.plot_train_results(self, results["train_results"], results["test_results"], results["prms"])
        if self.args.plots.show_dropout:
            plotting.plot_dropout_results(
                self,
                results["dropout_results"],
                results["dropout_parameters"],
                results["prms"],
                dropout_type="nodes",
            )
        # if self.args.plots.show_eigenfeatures:
        #     plotting.plot_eigenfeatures(self, results["eigen_results"], results["prms"])
        # if self.args.plots.show_eig_dropout:
        #     plotting.plot_dropout_results(
        #         self,
        #         results["evec_dropout_results"],
        #         results["evec_dropout_parameters"],
        #         results["prms"],
        #         dropout_type="eigenvectors",
        #     )

    def save_results(self, results):
        self.save_experiment(results)


def cli_main():
    parser = argparse.ArgumentParser(description="General alignment experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
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