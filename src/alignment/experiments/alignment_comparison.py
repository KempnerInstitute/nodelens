import torch

from alignment.experiments.experiment import Experiment
from alignment.models.registry import get_model, get_model_parameters
from alignment import processing
from alignment import plotting


class AlignmentComparison(Experiment):
    def get_basename(self):
        """
        define basename for the AlignmentComparison experiment
        """
        return "alignment_comparison"

    def prepare_path(self):
        """
        Define save location for each instance of this experiment type
        """
        return [self.args.extra.comparison, self.args.model.name, self.args.dataset.name, self.args.optimizer.name]

    # ----------------------------------------------
    # ------ methods for main experiment loop ------
    # ----------------------------------------------
    def create_networks(self):
        """
        method for creating networks

        depending on the experiment parameters (which comparison, which metaparams etc)
        this method will create multiple networks with requested parameters and return
        their optimizers and a params dictionary with the experiment parameters associated
        with each network
        """
        model_constructor = get_model(self.args.model.name)
        model_parameters = get_model_parameters(self.args.model.name, self.args.dataset.name)

        # get optimizer
        if self.args.optimizer.name == "Adam":
            optim = torch.optim.Adam
        elif self.args.optimizer.name == "SGD":
            optim = torch.optim.SGD
        else:
            raise ValueError(f"optimizer ({self.args.optimizer.name}) not recognized")

        # compare learning rates
        if self.args.extra.comparison == "lr":
            lrs = [lr for lr in self.args.extra.lrs for _ in range(self.args.training.replicates)]
            nets = [
                model_constructor(
                    dropout=self.args.model.dropout,
                    **model_parameters,
                    ignore_flag=self.args.alignment.ignore_flag,
                )
                for _ in lrs
            ]
            nets = [net.to(self.device) for net in nets]
            optimizers = [optim(net.parameters(), lr=lr, weight_decay=self.args.optimizer.weight_decay) for net, lr in zip(nets, lrs)]
            prms = {
                "lrs": lrs,  # the value of the independent variable for each network
                "name": "lr",  # the name of the parameter being varied
                "vals": self.args.extra.lrs,  # the list of unique values for the relevant parameter
            }
            return nets, optimizers, prms

        # compare training with different regularizers
        elif self.args.extra.comparison == "regularizer":
            dropout_values = [self.args.extra.compare_dropout * (reg == "dropout") for reg in self.args.extra.regularizers]
            weight_decay_values = [self.args.extra.compare_wd * (reg == "weight_decay") for reg in self.args.extra.regularizers]
            dropouts = [do for do in dropout_values for _ in range(self.args.training.replicates)]
            weight_decays = [wd for wd in weight_decay_values for _ in range(self.args.training.replicates)]
            nets = [model_constructor(dropout=do, **model_parameters, ignore_flag=self.args.alignment.ignore_flag) for do in dropouts]
            nets = [net.to(self.device) for net in nets]
            optimizers = [optim(net.parameters(), lr=self.args.optimizer.lr, weight_decay=wd) for net, wd in zip(nets, weight_decays)]
            prms = {
                "dropouts": dropouts,  # dropout values by network
                "weight_decays": weight_decays,  # weight decay values by network
                "name": "regularizer",  # name of experiment
                "vals": self.args.extra.regularizers,  # name of unique regularizers
            }
            return nets, optimizers, prms

        else:
            raise ValueError(f"Comparison={self.args.extra.comparison} is not recognized")

    def main(self):
        """
        main experiment loop

        create networks (this is where the specific experiment is determined)
        train and test networks
        do supplementary analyses
        """
        # create networks
        nets, optimizers, prms = self.create_networks()

        # load dataset
        dataset = self.prepare_dataset(nets[0])

        # train networks
        train_results, test_results = processing.train_networks(self, nets, optimizers, dataset)

        # do targeted dropout experiment
        dropout_results, dropout_parameters = processing.progressive_dropout_experiment(
            self, nets, dataset, alignment=test_results["alignment"], train_set=False
        )

        # measure eigenfeatures
        eigen_results = processing.measure_eigenfeatures(self, nets, dataset, train_set=False)

        # do targeted dropout experiment
        evec_dropout_results, evec_dropout_parameters = processing.eigenvector_dropout(self, nets, dataset, eigen_results, train_set=False)

        # make full results dictionary
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

        # return results and trained networks
        return results, nets

    def plot(self, results):
        """
        main plotting loop
        """
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
