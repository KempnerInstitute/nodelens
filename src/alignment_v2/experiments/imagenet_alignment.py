import sys
from config import ExperimentConfig
import torch
from torch.optim import Adam
from alignment_v2.registry import get_model, get_transform_parameters
import alignment_v2.processing
import alignment_v2.plotting
from alignment_v2.datasets import get_dataset
from alignment_v2.config import ExperimentConfig

class ImagenetAlignmentExperiment:
    def __init__(self, config):
        self.args = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.run = None

    def get_checkpoint_path(self):
        return "imagenet_ckpt.pt"

    def prepare_dataset(self):
        transform_params = get_transform_parameters(self.args.model.name, self.args.dataset.name)
        ds = get_dataset(self.args.dataset.name, build=True,
                         dataset_parameters=dict(download=self.args.dataset.download, root=self.args.dataset.path),
                         transform_parameters=transform_params,
                         loader_parameters=dict(batch_size=self.args.training.batch_size))
        return ds

    def create_networks(self):
        # Using pretrained model from torchvision (e.g., alexnet)
        net = get_model(self.args.model.name,
                        alignment_layer_names=self.args.model.alignment_layers,
                        build=True,
                        dataset=self.args.dataset.name)
        net.to(self.device)
        opt = Adam(net.parameters(), lr=self.args.optimizer.lr, weight_decay=self.args.optimizer.weight_decay)
        prms = {"vals": [self.args.model.name],
                "name": "network",
                "dataset": self.args.dataset.name}
        return [net], [opt], prms

    def main(self):
        nets, opts, prms = self.create_networks()
        ds = self.prepare_dataset()
        test_res = processing.train_networks(self, nets, opts, ds, num_epochs=0)[1]
        eig_res = processing.measure_eigenfeatures(self, nets, ds, train_set=False)
        results = {"prms": prms, "test_results": test_res, "eigen_results": eig_res}
        return results, nets

    def plot(self, results):
        plotting.plot_eigenfeatures(self, results["eigen_results"], results["prms"])

if __name__=="__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please provide the config file path as the first argument.")
    config_path = sys.argv[1]
    cfg = ExperimentConfig.load(config_path)
    exp = ImagenetAlignmentExperiment(cfg)
    results, nets = exp.main()
    exp.plot(results)