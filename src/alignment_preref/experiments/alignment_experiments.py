# alignment_experiments.py

import torch
import argparse
import sys

from alignment_refac1.datasets import get_dataset
from alignment_refac1.models.registry import get_model, get_transform_parameters
from alignment_refac1 import processing
from alignment_refac1.config import ExperimentConfig
from alignment_refac1.experiments.experiment import Experiment
from Code.alignment.src.alignment_preref import plotting_rem
from alignment_refac1.processing import evaluate_pretrained_model

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
            for name, layer in net.base_model.named_modules():
                if hasattr(layer, "weight"):
                    print(name, layer.weight.shape)
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
        nets, optimizers, prms = self.create_networks()

        if self.args.training.do_train:
            dataset = self.prepare_dataset(get_transform_parameters(self.args.model.name, self.args.dataset.name))
            train_results, test_results = processing.train_networks(self, nets, optimizers, dataset)
        else:
            train_results = {}
            if self.args.model.name == "AlexNet":
                import torchvision.models as tvm
                from torchvision.models import AlexNet_Weights
                pretrained_torchvision_alexnet = tvm.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).to(self.device)
                nets[0].base_model.load_state_dict(pretrained_torchvision_alexnet.state_dict())
                nets[0].eval()

            dataset = self.prepare_dataset(get_transform_parameters(self.args.model.name, self.args.dataset.name))
            print("Evaluating downloaded pretrained weights for baseline accuracy...")
            acc = evaluate_pretrained_model(nets[0], dataset)
            print("accuracy:", acc)
            test_results = processing.test_networks(self, nets, dataset)

        # Combine top-level results dict
        results = {
            "prms": prms,
            "train_results": train_results,
            "test_results": test_results
        }

        # If train_results or test_results have alignment/grad_alignment_corr => merge them
        if "alignment" in train_results:
            results["alignment"] = train_results["alignment"]
        if "alignment_distribution" in train_results:
            results["alignment_distribution"] = train_results["alignment_distribution"]
        if "expected_distribution" in train_results:
            results["expected_distribution"] = train_results["expected_distribution"]
        if "grad_alignment_corr" in train_results:
            results["grad_alignment_corr"] = train_results["grad_alignment_corr"]

        if "alignment" in test_results:
            # If you want to store the test alignment as well, put them in a separate key or merge
            results["alignment_test"] = test_results["alignment"]
        if "grad_alignment_corr" in test_results:
            results["grad_alignment_corr_test"] = test_results["grad_alignment_corr"]

        # Now do progressive_dropout
        dropout_results, dropout_params = processing.progressive_dropout_experiment(
            self, nets, dataset, alignment=test_results.get("alignment", None), train_set=False
        )
        results["dropout_results"] = dropout_results
        results["dropout_parameters"] = dropout_params

        if len(nets) > 0:
            results["alignment_names"] = nets[0].alignment_names

        return results, nets

    def plot(self, results):
        if self.args.plots.show_loss:
            plotting_rem.plot_train_results(self, 
                                        results["train_results"], 
                                        results["test_results"], 
                                        results["prms"])
        if self.args.plots.show_dropout:
            plotting_rem.plot_dropout_results(
                self,
                results["dropout_results"],
                results["dropout_parameters"],
                results["prms"],
                dropout_type="nodes",
            )
        if "grad_alignment_corr" in results:
            # plotting.plot_grad_alignment_correlation(self, results["grad_alignment_corr"])
            pass
        else:
            print("No grad_alignment_corr found in results")

        if "alignment" not in results:
            print("No alignment key in results")
        else:
            # plotting or other usage
            pass

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