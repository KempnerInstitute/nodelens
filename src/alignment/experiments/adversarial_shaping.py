import torch

from alignment.experiments.experiment import Experiment
from alignment.models.registry import (get_model,
                                       get_model_parameters,
                                       get_transform_parameters)
from alignment.models.base import AlignmentNetwork
from alignment import processing
from alignment import plotting
from alignment.utils import get_eval_transform_by_cutoff


class AdversarialShaping(Experiment):
    def get_basename(self):
        """
        define basename for the AdversarialShaping experiment
        """
        return "adversarial_shaping"

    def prepare_path(self):
        """
        Define save location for each instance of this experiment type
        """
        return [self.args.model.name, self.args.dataset.name, self.args.optimizer.name]

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
        base_model_constructor = get_model(self.args.model.name)
        model_parameters = get_model_parameters(self.args.model.name, self.args.dataset.name)

        # get optimizer
        if self.args.optimizer.name == "Adam":
            optim = torch.optim.Adam
        elif self.args.optimizer.name == "SGD":
            optim = torch.optim.SGD
        else:
            raise ValueError(f"optimizer ({self.args.optimizer.name}) not recognized")

        cutoffs = [co for co in self.args.extra.cutoffs for _ in range(self.args.training.replicates)]
        nets = [
            AlignmentNetwork(base_model_constructor(
                dropout=self.args.model.dropout,
                **model_parameters,
            ))
            for _ in cutoffs
        ]
        nets = [net.to(self.device) for net in nets]
        optimizers = [optim(net.parameters(), lr=self.args.optimizer.lr, weight_decay=self.args.optimizer.weight_decay) for net in nets]
        prms = {
            "cutoffs": cutoffs,  # the value of the independent variable for each network
            "name": "cutoff",  # the name of the parameter being varied
            "vals": self.args.extra.cutoffs,  # the list of unique values for the relevant parameter
        }
        return nets, optimizers, prms

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
        dataset = self.prepare_dataset(get_transform_parameters(self.args.model.name, self.args.dataset.name))

        # train networks
        special_parameters = dict(
            manual_shape=True,
            manual_frequency=self.args.extra.manual_frequency,
            manual_transforms=[get_eval_transform_by_cutoff(co) for co in prms["cutoffs"]],
            manual_layers=list(range(nets[0].num_layers())),
        )

        train_results, test_results = processing.train_networks(self, nets, optimizers, dataset, alignment=True, **special_parameters)

        # measure eigenfeatures
        eigen_results = processing.measure_eigenfeatures(self, nets, dataset, train_set=False)

        # do adversarial attack experiment
        adversarial_parameters = dict(
            epsilons=torch.linspace(0, 1, 11),
            use_sign=True,
            fgsm_transform=lambda x: x,
        )
        adversarial_results = processing.measure_adversarial_attacks(nets, dataset, self, eigen_results, train_set=False, **adversarial_parameters)

        # make full results dictionary
        results = dict(
            prms=prms,
            train_results=train_results,
            test_results=test_results,
            eigen_results=eigen_results,
            adversarial_results=adversarial_results,
        )

        # return results and trained networks
        return results, nets

    def plot(self, results):
        """
        main plotting loop
        """
        plotting.plot_train_results(self, results["train_results"], results["test_results"], results["prms"])
        plotting.plot_eigenfeatures(self, results["eigen_results"], results["prms"])
        plotting.plot_adversarial_results(self, results["eigen_results"], results["adversarial_results"], results["prms"])
