from typing import Optional
import torchvision.models as tv
from models import MLP, CNN2P2
from base import AlignmentNetwork
from torch import nn

MODEL_REGISTRY={
    "MLP":MLP,
    "CNN2P2":CNN2P2,
    "AlexNet":tv.alexnet
}

DATASET_ARGUMENTS={
    "MLP":{
        "MNIST":dict(input_dim=784,output_dim=10),
        "CIFAR10":dict(input_dim=3072,output_dim=10),
        "CIFAR100":dict(input_dim=3072,output_dim=100)
    },
    "CNN2P2":{
        "MNIST":dict(in_channels=1,output_dim=10),
        "CIFAR10":dict(in_channels=3,output_dim=10),
        "CIFAR100":dict(in_channels=3,output_dim=100)
    },
    "AlexNet":{
        "MNIST":dict(num_classes=10),
        "CIFAR10":dict(num_classes=10),
        "CIFAR100":dict(num_classes=100)
    }
}

def gray_to_rgb(batch):
    batch[0]=batch[0].expand(-1,3,-1,-1)
    return batch

TRANSFORM_PARAMETERS={
    "MLP":{
        "MNIST":dict(flatten=True,resize=None),
        "CIFAR10":dict(flatten=True,resize=None),
        "CIFAR100":dict(flatten=True,resize=None)
    },
    "CNN2P2":{
        "MNIST":dict(flatten=False,resize=None),
        "CIFAR10":dict(flatten=False,resize=None),
        "CIFAR100":dict(flatten=False,resize=None)
    },
    "AlexNet":{
        "MNIST":dict(flatten=False,resize=(256,256),extra_transform=[gray_to_rgb]),
        "CIFAR10":dict(flatten=False,resize=(256,256)),
        "CIFAR100":dict(flatten=False,resize=(256,256))
    }
}

def get_transform_parameters(model_name,dataset):
    return TRANSFORM_PARAMETERS[model_name][dataset]

def get_model_parameters(model_name,dataset):
    if model_name not in DATASET_ARGUMENTS:
        raise ValueError(f"No model {model_name}")
    if dataset not in DATASET_ARGUMENTS[model_name]:
        raise ValueError(f"No dataset {dataset} in {model_name}")
    return DATASET_ARGUMENTS[model_name][dataset]

def get_model(model_name,alignment_layer_names=None,build=False,dataset=None,**kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_name}")
    base=MODEL_REGISTRY[model_name]
    if not build:
        return base
    if dataset:
        dsargs=get_model_parameters(model_name,dataset)
        for k,v in dsargs.items():
            if k not in kwargs:
                kwargs[k]=v
    net=base(**kwargs)
    if isinstance(net,nn.Module):
        return AlignmentNetwork(net,alignment_layer_names=alignment_layer_names)
    return net