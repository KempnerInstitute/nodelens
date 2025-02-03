from typing import Optional
import torchvision.models as tv
from models import MLP, CNN2P2
from base import AlignmentNetwork
from torch import nn

MODEL_REGISTRY = {
    "mlp": MLP,
    "cnn2p2": CNN2P2,
    "alexnet": tv.alexnet,
    # you can add other torchvision models here
}

DATASET_ARGUMENTS = {
    "mlp": {
        "mnist": dict(input_dim=784, output_dim=10),
        "cifar10": dict(input_dim=3072, output_dim=10),
        "cifar100": dict(input_dim=3072, output_dim=100)
    },
    "cnn2p2": {
        "mnist": dict(in_channels=1, output_dim=10),
        "cifar10": dict(in_channels=3, output_dim=10),
        "cifar100": dict(in_channels=3, output_dim=100)
    },
    "alexnet": {
        "imagenet": dict(num_classes=1000)
    }
}

def gray_to_rgb(batch):
    batch[0] = batch[0].expand(-1, 3, -1, -1)
    return batch

TRANSFORM_PARAMETERS = {
    "mlp": {
        "mnist": dict(flatten=True, resize=None),
        "cifar10": dict(flatten=True, resize=None),
        "cifar100": dict(flatten=True, resize=None)
    },
    "cnn2p2": {
        "mnist": dict(flatten=False, resize=None),
        "cifar10": dict(flatten=False, resize=None),
        "cifar100": dict(flatten=False, resize=None)
    },
    "alexnet": {
        "imagenet": dict(center_crop=224, flatten=False, resize=(256,256))
    }
}

def get_transform_parameters(model_name, dataset):
    return TRANSFORM_PARAMETERS[model_name.lower()][dataset.lower()]

def get_model_parameters(model_name, dataset):
    if model_name.lower() not in DATASET_ARGUMENTS:
        raise ValueError(f"Unknown model {model_name}")
    if dataset.lower() not in DATASET_ARGUMENTS[model_name.lower()]:
        raise ValueError(f"Unknown dataset {dataset} for model {model_name}")
    return DATASET_ARGUMENTS[model_name.lower()][dataset.lower()]

def get_model(model_name, alignment_layer_names: Optional[dict] = None, build=False, dataset=None, **kwargs):
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_name}")
    base_model = MODEL_REGISTRY[model_name]
    if not build:
        return base_model
    if dataset:
        dsargs = get_model_parameters(model_name, dataset)
        for k, v in dsargs.items():
            kwargs.setdefault(k, v)
    net = base_model(**kwargs)
    return AlignmentNetwork(net, alignment_layer_names=alignment_layer_names)