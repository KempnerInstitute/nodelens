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
            net.cnn_mode = self.args.alignment.cnn_mode
            nets.append(net)
            for name, layer in net.base_model.named_modules():
                if hasattr(layer, "weight"):
                    print(name, layer.weight.shape)

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
            train_results = {}

            # Example: if user wants a pretrained AlexNet from torchvision:
            if self.args.model.name == "AlexNet":
                import torchvision.models as tvm
                from torchvision.models import AlexNet_Weights
                # use the official pretrained weight
                pretrained_torchvision_alexnet = tvm.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).to(self.device)
                # now you'd want to copy weights into your existing net or just reassign nets[0], etc.
                # This is just an example snippet:
                nets[0].base_model.load_state_dict(pretrained_torchvision_alexnet.state_dict())
                nets[0].eval()

            dataset = self.prepare_dataset(get_transform_parameters(self.args.model.name, self.args.dataset.name))
            # Now call the new function 'processing.test_networks(...)'
            test_results = processing.test_networks(self, nets, dataset)

        # do progressive dropout
        dropout_results, dropout_params = processing.progressive_dropout_experiment(
            self, nets, dataset, alignment=test_results.get("alignment", None), train_set=False
        )

        results = {
            "prms": prms,
            "train_results": train_results,
            "test_results": test_results,
            "dropout_results": dropout_results,
            "dropout_parameters": dropout_params,
        }
        return results, nets

    def plot(self, results):
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