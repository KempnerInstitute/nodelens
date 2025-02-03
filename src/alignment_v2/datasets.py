import sys
from pathlib import Path
from warnings import warn
from abc import ABC, abstractmethod
import torch
import torchvision
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2 as transforms
from config import ExperimentConfig

REQUIRED_PROPERTIES = ["dataset_constructor", "loss_function"]

def default_loader_parameters(distributed, batch_size=1024, num_workers=2,
                              shuffle=True, pin_memory=True, persistent_workers=True):
    return dict(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False if distributed else shuffle,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

class DataSet(ABC):
    def __init__(self, device=None, distributed=False, dataset_parameters={},
                 transform_parameters={}, loader_parameters={}):
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
        for prop in REQUIRED_PROPERTIES:
            if not hasattr(self, prop):
                raise ValueError(f"Missing required property: {prop}")

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
        from torch.utils.data import DataLoader
        self.train_loader = DataLoader(self.train_dataset, sampler=self.train_sampler, **self.dataloader_parameters)
        self.test_loader = DataLoader(self.test_dataset, sampler=self.test_sampler, **self.dataloader_parameters)

    def unwrap_batch(self, batch, device=None):
        device = device or self.device
        if self.extra_transform:
            if isinstance(self.extra_transform, list):
                for t in self.extra_transform:
                    batch = t(batch)
            else:
                warn("extra_transform should be a list")
                batch = self.extra_transform(batch)
        x, y = batch
        return x.to(device), y.to(device)

    def make_transform(self, center_crop=None, resize=None, flatten=False, out_channels=None):
        t = [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        if center_crop:
            t.append(transforms.CenterCrop(center_crop))
        t.append(transforms.Normalize(self.dist_params["mean"], self.dist_params["std"]))
        if resize:
            t.append(transforms.Resize(resize, antialias=True))
        if out_channels:
            t.append(transforms.Grayscale(num_output_channels=out_channels))
        if flatten:
            t.append(transforms.Lambda(torch.flatten))
        self.transform = transforms.Compose(t)

    def measure_loss(self, outputs, targets, reduction=None):
        if reduction is None:
            return self.loss_function(outputs, targets)
        old = self.loss_function.reduction
        self.loss_function.reduction = reduction
        l = self.loss_function(outputs, targets)
        self.loss_function.reduction = old
        return l

    def measure_accuracy(self, outputs, targets, k=1, percentage=True):
        topk = outputs.topk(k, dim=1)[1]
        correct = torch.sum(torch.any(topk == targets.view(-1, 1), dim=1))
        return 100 * correct / outputs.size(0) if percentage else correct

class MNIST(DataSet):
    def set_properties(self):
        self.dataset_constructor = torchvision.datasets.MNIST
        from torch import nn
        self.loss_function = nn.CrossEntropyLoss()
        self.dist_params = dict(mean=[0.1307], std=[0.3081])

    def dataset_kwargs(self, train=True, download=False, root=None):
        return dict(train=train, download=download, root=root, transform=self.transform)

class CIFAR10(DataSet):
    def set_properties(self):
        self.dataset_constructor = torchvision.datasets.CIFAR10
        from torch import nn
        self.loss_function = nn.CrossEntropyLoss()
        self.dist_params = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def dataset_kwargs(self, train=True, download=False, root=None):
        return dict(train=train, download=download, root=root, transform=self.transform)

class CIFAR100(CIFAR10):
    def set_properties(self):
        self.dataset_constructor = torchvision.datasets.CIFAR100
        from torch import nn
        self.loss_function = nn.CrossEntropyLoss()
        self.dist_params = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

class ImageNet2012(DataSet):
    def set_properties(self):
        self.dataset_constructor = torchvision.datasets.ImageNet
        from torch import nn
        self.loss_function = nn.CrossEntropyLoss()
        self.dist_params = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.center_crop = 224

    def dataset_kwargs(self, train=True, root=None):
        return dict(split="train" if train else "val", root=root, transform=self.transform)

DATASET_REGISTRY = {
    "mnist": MNIST,
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "imagenet": ImageNet2012,
}

def get_dataset(name, build=False, dataset_parameters={}, transform_parameters={}, loader_parameters={}, **kwargs):
    if name.lower() not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {name}")
    ds = DATASET_REGISTRY[name.lower()]
    if build:
        return ds(dataset_parameters=dataset_parameters,
                  transform_parameters=transform_parameters,
                  loader_parameters=loader_parameters, **kwargs)
    return ds