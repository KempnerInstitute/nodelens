import torch
from alignment.models.registry import get_model
from alignment.experiments import arglib
from alignment import processing
from alignment import plotting
from alignment.experiments.experiment import Experiment

#python experiment.py alignment_info_stats --save-networks --network MLP --dataset MNIST --use-wandb --dropout_by_layer --epochs 5

class AlignmentStatisticsInfo(Experiment):
    def get_basename(self):
        return "alignment_info_stats"

    def prepare_path(self):
        return [self.args.network, self.args.dataset, self.args.optimizer]

    def make_args(self, parser):
        """
        Method for adding experiment specific arguments to the argument parser
        """
        parser = arglib.add_standard_training_parameters(parser)
        parser = arglib.add_checkpointing(parser)
        parser = arglib.add_dropout_experiment_details(parser)
        parser = arglib.add_network_metaparameters(parser)
        parser = arglib.add_alignment_analysis_parameters(parser)
        
        # Add argument for controlling dropout by layer
        parser.add_argument('--dropout_by_layer', default=False, action='store_true', help="Perform dropout experiment by layer")
    
        return parser

    def create_networks(self):
        """
        method for creating networks

        depending on the experiment parameters (which comparison, which metaparams etc)
        this method will create multiple networks with requested parameters and return
        their optimizers and a params dictionary with the experiment parameters associated
        with each network
        """
        # get optimizer
        if self.args.optimizer == "Adam":
            optim = torch.optim.Adam
        elif self.args.optimizer == "SGD":
            optim = torch.optim.SGD
        else:
            raise ValueError(f"optimizer ({self.args.optimizer}) not recognized")

        nets = [
            get_model(
                self.args.network,
                build=True,
                dataset=self.args.dataset,
                dropout=self.args.default_dropout,
                ignore_flag=self.args.ignore_flag,
            )
            for _ in range(self.args.replicates)
        ]
        nets = [net.to(self.device) for net in nets]

        optimizers = [optim(net.parameters(), lr=self.args.default_lr, weight_decay=self.args.default_wd) for net in nets]

        prms = {
            "vals": [self.args.network],  # require iterable for identifying how many types of networks there are (just one type...)
            "name": "network",
            "dataset": self.args.dataset,
            "dropout": self.args.default_dropout,
            "lr": self.args.default_lr,
            "weight_decay": self.args.default_wd,
        }
        return nets, optimizers, prms
    
    def main(self):
        """
        Main experiment loop
        """
        # Step 1: Train the full network
        nets, optimizers, prms = self.create_networks()
        dataset = self.prepare_dataset(nets[0])
        
        print("Training the full network...")
        train_results, test_results = processing.train_networks(self, nets, optimizers, dataset)

        # Step 2: Dropout across all layers based on alignment and retrain
        print("Performing full-layer dropout experiment...")
        dropout_results_pre, dropout_parameters_pre = processing.progressive_dropout_experiment(
            self, nets, dataset, alignment=test_results.get("alignment", None), train_set=False
        )

        # Retrain the pruned network
        # print("Retraining the pruned network after full-layer dropout...")
        # retrain_results, retrain_test_results = processing.retrain_network_with_dropout_stats(
        #     self, nets, optimizers, dataset, dropout_results_pre
        # )

        # Step 3: Sequential dropout experiment (drop nodes layer by layer based on alignment)
        # print("Performing sequential dropout experiment...")
        # sequential_dropout_results_pre, sequential_dropout_parameters_pre = processing.sequential_dropout_experiment(
        #     self, nets, dataset
        # )
        
        # Retrain the network after sequential dropout
        # print("Retraining the pruned network after sequential dropout...")
        # sequential_retrain_results, sequential_retrain_test_results = processing.retrain_network_with_dropout_stats(
        #     self, nets, optimizers, dataset, sequential_dropout_results_pre
        # )
        
        # Step 4: Store all the results
        results = dict(
            prms=prms,
            initial_train_results=train_results,
            initial_test_results=test_results,
            
            # Full-layer dropout results (pre and post retrain)
            full_dropout_results_pre=dropout_results_pre,
            full_dropout_parameters_pre=dropout_parameters_pre,
            #full_dropout_retrain_results=retrain_results,
            #full_dropout_retrain_test_results=retrain_test_results,
            
            # Sequential-layer dropout results (pre and post retrain)
            #sequential_dropout_results_pre=sequential_dropout_results_pre,
            #sequential_dropout_parameters_pre=sequential_dropout_parameters_pre,
            #sequential_retrain_results=sequential_retrain_results,
            #sequential_retrain_test_results=sequential_retrain_test_results
        )

        return results, nets

    def plot(self, results):
        """
        Main plotting loop
        """
        print("Plotting training and dropout results...")

        # Plot initial training results
        plotting.plot_train_results(self, results["initial_train_results"], results["initial_test_results"], results["prms"])

        # Plot for full-layer dropout (pre- and post-retrain)
        print("Plotting full-layer dropout results (pre- and post-retraining)...")
        plotting.plot_dropout_results(
            self,
            results["full_dropout_results_pre"],
            results["full_dropout_parameters_pre"],
            results["prms"],
            dropout_type="nodes_pre_retrain_full"
        )
        # plotting.plot_dropout_results(
        #     self,
        #     results["full_dropout_retrain_results"],
        #     results["full_dropout_parameters_pre"],
        #     results["prms"],
        #     dropout_type="nodes_post_retrain_full"
        # )

        # Plot for sequential-layer dropout (pre- and post-retrain)
        # print("Plotting sequential-layer dropout results (pre- and post-retraining)...")
        # plotting.plot_dropout_results(
        #     self,
        #     results["sequential_dropout_results_pre"],
        #     results["sequential_dropout_parameters_pre"],
        #     results["prms"],
        #     dropout_type="nodes_pre_retrain_sequential"
        # )
        # plotting.plot_dropout_results(
        #     self,
        #     results["sequential_retrain_results"],
        #     results["sequential_dropout_parameters_pre"],
        #     results["prms"],
        #     dropout_type="nodes_post_retrain_sequential"
        # )   
    # def plot(self, results):
    #     """
    #     main plotting loop
    #     """
    #     plotting.plot_train_results(self, results["train_results"], results["test_results"], results["prms"])
    #     plotting.plot_dropout_results(
    #         self,
    #         results["dropout_results"],
    #         results["dropout_parameters"],
    #         results["prms"],
    #         dropout_type="nodes",
    #     )
        #plotting.plot_eigenfeatures(self, results["eigen_results"], results["prms"])
        #plotting.plot_dropout_results(
        #    self,
        #    results["evec_dropout_results"],
        #    results["evec_dropout_parameters"],
        #    results["prms"],
        #    dropout_type="eigenvectors",
        #)

