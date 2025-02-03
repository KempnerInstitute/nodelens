import torch
from registry import get_model, get_transform_parameters
import processing
import plotting

# assume you have some Experiment base or define your own
class AlignmentStatistics:
    def __init__(self,args,device="cuda"):
        self.args=args
        self.device=device
        self.run=None

    def get_basename(self):
        return "alignment_stats"

    def get_checkpoint_path(self):
        return "checkpoint.pt"

    def prepare_dataset(self,transform_params):
        # build dataset
        from datasets import get_dataset
        ds=get_dataset(self.args.dataset.name,build=True,
                       dataset_parameters=dict(download=self.args.dataset.download,root=self.args.dataset.path),
                       transform_parameters=transform_params,
                       loader_parameters=dict(batch_size=self.args.training.batch_size))
        return ds

    def create_networks(self):
        if self.args.optimizer.name=="Adam":
            from torch.optim import Adam
            opt_constr=Adam
        elif self.args.optimizer.name=="SGD":
            from torch.optim import SGD
            opt_constr=SGD
        else:
            raise ValueError("unknown optimizer")
        nets=[]
        opts=[]
        for _ in range(self.args.training.replicates):
            net=get_model(self.args.model.name,
                          alignment_layer_names=self.args.model.alignment_layers,
                          build=True,
                          dataset=self.args.dataset.name,
                          dropout=self.args.model.dropout)
            net.to(self.device)
            op=opt_constr(net.parameters(),lr=self.args.optimizer.lr,weight_decay=self.args.optimizer.weight_decay)
            nets.append(net)
            opts.append(op)
        prms=dict(vals=[self.args.model.name],
                  name="network",
                  dataset=self.args.dataset.name,
                  dropout=self.args.model.dropout,
                  lr=self.args.optimizer.lr,
                  weight_decay=self.args.optimizer.weight_decay)
        return nets,opts,prms

    def main(self):
        nets,opts,prms=self.create_networks()
        ds=self.prepare_dataset(get_transform_parameters(self.args.model.name,self.args.dataset.name))
        tr,te=processing.train_networks(self,nets,opts,ds)
        drop_res,drop_par=processing.progressive_dropout_experiment(self,nets,ds,alignment=te.get("alignment",None),train_set=False)
        eig_res=processing.measure_eigenfeatures(self,nets,ds,train_set=False)
        evec_drop,evec_par=processing.eigenvector_dropout(self,nets,ds,eig_res,train_set=False)
        results=dict(
            prms=prms,
            train_results=tr,
            test_results=te,
            dropout_results=drop_res,
            dropout_parameters=drop_par,
            eigen_results=eig_res,
            evec_dropout_results=evec_drop,
            evec_dropout_parameters=evec_par
        )
        return results,nets

    def plot(self,results):
        plotting.plot_train_results(self,results["train_results"],results["test_results"],results["prms"])
        plotting.plot_dropout_results(self,results["dropout_results"],results["dropout_parameters"],results["prms"],"nodes")
        plotting.plot_eigenfeatures(self,results["eigen_results"],results["prms"])
        plotting.plot_dropout_results(self,results["evec_dropout_results"],results["evec_dropout_parameters"],results["prms"],"eigenvectors")

    def plot_ready(self,name):
        pass  # hook if you want to save or show