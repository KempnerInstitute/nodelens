
import sys
import torch
from alignment_v2.registry import get_model, get_transform_parameters
from alignment_v2 import processing
from alignment_v2 import plotting
from alignment_v2.datasets import get_dataset
from alignment_v2.config import ExperimentConfig

class AlignmentStatsExperiment:
    def __init__(self, config):
        self.args = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.run = None

    def get_checkpoint_path(self):
        return "ckpt.pt"

    def prepare_dataset(self):
        transform_params = get_transform_parameters(self.args.model.name, self.args.dataset.name)
        ds = get_dataset(self.args.dataset.name, build=True,
                         dataset_parameters=dict(download=self.args.dataset.download, root=self.args.dataset.path),
                         transform_parameters=transform_params,
                         loader_parameters=dict(batch_size=self.args.training.batch_size))
        return ds

    def create_networks(self):
        if self.args.optimizer.name.lower() == "adam":
            from torch.optim import Adam
            opt_cls = Adam
        elif self.args.optimizer.name.lower() == "sgd":
            from torch.optim import SGD
            opt_cls = SGD
        else:
            raise ValueError("Unknown optimizer")
        nets = []
        opts = []
        for _ in range(self.args.training.replicates):
            net = get_model(self.args.model.name,
                            alignment_layer_names=self.args.model.alignment_layers,
                            build=True,
                            dataset=self.args.dataset.name,
                            dropout=self.args.model.dropout)
            net.to(self.device)
            opt = opt_cls(net.parameters(), lr=self.args.optimizer.lr, weight_decay=self.args.optimizer.weight_decay)
            nets.append(net)
            opts.append(opt)
        prms = {"vals": [self.args.model.name],
                "name": "network",
                "dataset": self.args.dataset.name,
                "dropout": self.args.model.dropout,
                "lr": self.args.optimizer.lr,
                "weight_decay": self.args.optimizer.weight_decay}
        return nets, opts, prms

    def main(self):
        nets, opts, prms = self.create_networks()
        ds = self.prepare_dataset()
        train_res, test_res = processing.train_networks(self, nets, opts, ds)
        eig_res = processing.measure_eigenfeatures(self, nets, ds, train_set=False)
        results = {"prms": prms, "train_results": train_res, "test_results": test_res, "eigen_results": eig_res}
        return results, nets

    def plot(self, results):
        plotting.plot_train_results(self, results["train_results"], results["test_results"], results["prms"])
        plotting.plot_eigenfeatures(self, results["eigen_results"], results["prms"])

    
if __name__=="__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please provide the config file path as the first argument.")
    config_path = sys.argv[1]
    cfg = ExperimentConfig.load(config_path)
    exp = AlignmentStatsExperiment(cfg)
    results, nets = exp.main()
    exp.plot(results)