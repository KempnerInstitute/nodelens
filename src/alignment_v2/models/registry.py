from typing import Optional

from torchvision.models import alexnet

from alignment.models import models
from alignment.models.base import AlignmentNetwork

MODEL_REGISTRY = {
    "MLP": models.MLP,
    "CNN2P2": models.CNN2P2,
    "AlexNet": alexnet,
}

DATASET_ARGUMENTS = {
    "MLP": {
        "MNIST": dict(input_dim=784, output_dim=10),
        "CIFAR10": dict(input_dim=3072, output_dim=10),
        "CIFAR100": dict(input_dim=3072, output_dim=100),
    },
    "CNN2P2": {
        "MNIST": dict(in_channels=1, output_dim=10),
        "CIFAR10": dict(in_channels=3, num_hidden=[4096, 128], output_dim=10),
        "CIFAR100": dict(in_channels=3, num_hidden=[4096, 128], output_dim=100),
        "ImageNet": dict(in_channels=3, output_dim=1000),
    },
    "AlexNet": {
        "MNIST": dict(num_classes=10),
        "CIFAR10": dict(num_classes=10),
        "CIFAR100": dict(num_classes=100),
        "ImageNet": dict(num_classes=1000),
    },
}

def gray_to_rgb(batch):
    batch[0] = batch[0].expand(-1, 3, -1, -1)
    return batch

TRANSFORM_PARAMETERS = {
    "MLP": {
        "MNIST": dict(flatten=True, resize=None),
        "CIFAR10": dict(flatten=True, resize=None),
        "CIFAR100": dict(flatten=True, resize=None),
        "ImageNet": dict(flatten=True),
    },
    "CNN2P2": {
        "MNIST": dict(flatten=False, resize=None),
        "CIFAR10": dict(flatten=False, resize=None),
        "CIFAR100": dict(flatten=False, resize=None),
        "ImageNet": dict(flatten=False),
    },
    "AlexNet": {
        "MNIST": dict(flatten=False, resize=(256, 256), extra_transform=[gray_to_rgb,]),
        "CIFAR10": dict(flatten=False, resize=(256, 256)),
        "CIFAR100": dict(flatten=False, resize=(256, 256)),
        "ImageNet": dict(center_crop=224, flatten=False, resize=(256, 256)),
    },
}

def get_transform_parameters(model_name, dataset):
    return TRANSFORM_PARAMETERS[model_name][dataset]

def get_model_parameters(model_name, dataset):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model ({model_name}) is not in MODEL_REGISTRY")
    if dataset not in DATASET_ARGUMENTS[model_name]:
        raise ValueError(f"Dataset ({dataset}) is not in the DATASET_ARGUMENTS lookup for model ({model_name})")
    return DATASET_ARGUMENTS[model_name][dataset]

def get_model(model_name, alignment_layer_names: Optional[dict] = None, build=False, dataset=None, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model ({model_name}) is not in MODEL_REGISTRY")
    base_model = MODEL_REGISTRY[model_name]
    if not build: return base_model
    
    if dataset is not None:
        dataset_specific_arguments = get_model_parameters(model_name, dataset)
        for key, val in dataset_specific_arguments.items():
            if key not in kwargs:
                kwargs[key] = val

    base_model = base_model(**kwargs)
    return AlignmentNetwork(base_model=base_model, alignment_layer_names=alignment_layer_names)