import sys
from pathlib import Path
from warnings import warn
from abc import ABC, abstractmethod

import torch
import torchvision
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2 as transforms

REQUIRED_PROPERTIES = ["dataset_constructor", "loss_function"]

def default_loader_parameters(distributed, batch_size=1024, num_workers=2, shuffle=True, pin_memory=True, persistent_workers=True):
    return dict(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False if distributed else shuffle,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

class DataSet(ABC):
    def __init__(self, device=None, distributed=False, dataset_parameters={}, transform_parameters={}, loader_parameters={}):
        self.set_properties()
        self.check_properties()
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.distributed = distributed
        self.extra_transform = transform_parameters.pop("extra_transform", None)
        self.transform_parameters = transform_parameters
        self.make_transform(**transform_parameters)
        self.dataloader_parameters = default_loader_parameters(distributed, **loader_parameters)
        self.dataset_parameters = dataset_parameters
        self.load_dataset(**dataset_parameters)

    def check_properties(self):
        if not all([hasattr(self, prop) for prop in REQUIRED_PROPERTIES]):
            not_found = [prop for prop in REQUIRED_PROPERTIES if not hasattr(self, prop)]
            raise ValueError(f"The following required properties were not set: {not_found}")

    @abstractmethod
    def set_properties(self):
        pass

    @abstractmethod
    def dataset_kwargs(self, train=True, **kwargs):
        pass

    def load_dataset(self, **kwargs):
        self.train_dataset = self.dataset_constructor(**self.dataset_kwargs(train=True, **kwargs))
        self.test_dataset = self.dataset_constructor(**self.dataset_kwargs(train=False, **kwargs))
        self.train_sampler = DistributedSampler(self.train_dataset) if self.distributed else None
        self.test_sampler = DistributedSampler(self.test_dataset) if self.distributed else None
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, sampler=self.train_sampler, **self.dataloader_parameters)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, sampler=self.test_sampler, **self.dataloader_parameters)

    def unwrap_batch(self, batch, device=None):
        device = self.device if device is None else device
        if self.extra_transform:
            if isinstance(self.extra_transform, list):
                for et in self.extra_transform:
                    batch = et(batch)
            else:
                warn("extra_transform is not a list, this is deprecated!", DeprecationWarning, stacklevel=2)
                batch = self.extra_transform(batch)
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        return inputs, targets

    def make_transform(self, center_crop=None, resize=None, flatten=False, out_channels=None):
        chain = [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
        if center_crop:
            chain.append(transforms.CenterCrop(center_crop))
        chain.append(transforms.Normalize((self.dist_params["mean"]), (self.dist_params["std"])))
        if resize:
            chain.append(transforms.Resize(resize, antialias=True))
        if out_channels:
            chain.append(transforms.Grayscale(num_output_channels=out_channels))
        if flatten:
            chain.append(transforms.Lambda(torch.flatten))
        self.transform = transforms.Compose(chain)

    def measure_loss(self, outputs, targets, reduction=None):
        if reduction is None:
            return self.loss_function(outputs, targets)
        old_reduction = self.loss_function.reduction
        self.loss_function.reduction = reduction
        loss = self.loss_function(outputs, targets)
        self.loss_function.reduction = old_reduction
        return loss

    def measure_accuracy(self, outputs, targets, k=1, percentage=True):
        topk = outputs.topk(k, dim=1, sorted=True, largest=True)[1]
        correct = torch.sum(torch.any(topk == targets.view(-1, 1), dim=1))
        return 100 * correct / outputs.size(0) if percentage else correct

class MNIST(DataSet):
    def set_properties(self):
        self.dataset_constructor = torchvision.datasets.MNIST
        self.loss_function = nn.CrossEntropyLoss()
        self.dist_params = dict(mean=[0.1307], std=[0.3081])

    def dataset_kwargs(self, train=True, download=False, root=None, **kwargs):
        return dict(
            train=train,
            download=download,
            root=root,
            transform=self.transform,
        )

class CIFAR10(DataSet):
    def set_properties(self):
        self.dataset_constructor = torchvision.datasets.CIFAR10
        self.loss_function = nn.CrossEntropyLoss()
        self.dist_params = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def dataset_kwargs(self, train=True, download=False, root=None, **kwargs):
        return dict(
            train=train,
            download=download,
            root=root,
            transform=self.transform,
        )

class CIFAR100(CIFAR10):
    def set_properties(self):
        self.dataset_constructor = torchvision.datasets.CIFAR100
        self.loss_function = nn.CrossEntropyLoss()
        self.dist_params = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

class ImageNet2012(DataSet):
    def set_properties(self):
        self.dataset_constructor = torchvision.datasets.ImageNet
        self.loss_function = nn.CrossEntropyLoss()
        self.dist_params = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.center_crop = 224

    def dataset_kwargs(self, train=True, download=None, root=None, **kwargs):
        kwargs.pop('download', None)
        return dict(
            split="train" if train else "val",
            root=root,
            transform=self.transform,
            **kwargs,
        )

DATASET_REGISTRY = {
    "MNIST": MNIST,
    "CIFAR10": CIFAR10,
    "CIFAR100": CIFAR100,
    "ImageNet": ImageNet2012,
}

def get_dataset(dataset_name, build=False, dataset_parameters={}, transform_parameters={}, loader_parameters={}, **kwargs):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset ({dataset_name}) is not in DATASET_REGISTRY")
    dataset_cls = DATASET_REGISTRY[dataset_name]
    if build:
        if not isinstance(transform_parameters, dict):
            raise TypeError("transform_parameters must be a dictionary")
        return dataset_cls(
            dataset_parameters=dataset_parameters,
            transform_parameters=transform_parameters,
            loader_parameters=loader_parameters,
            **kwargs,
        )
    return dataset_cls

if __name__ == "__main__":
    from alignment.config import ExperimentConfig
    try:
        yaml_path, args_list = sys.argv[1]
    except IndexError:
        raise ValueError(f"Usage: {sys.argv[0]} [CONFIG_PATH]")
    cfg = ExperimentConfig.load(yaml_path)
    dataset = get_dataset(
        cfg.dataset.name,
        build=True,
        dataset_parameters=dict(root=Path(cfg.dataset.path)),
    )