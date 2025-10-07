"""
Data loader utilities for the alignment framework.

This module provides utilities for creating data loaders with
proper configuration for distributed training and memory efficiency.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)


@dataclass
class DataLoaderConfig:
    """Configuration for data loaders."""

    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # Distributed settings
    distributed: bool = False
    rank: Optional[int] = None
    world_size: Optional[int] = None

    # Memory settings
    device: Optional[Union[str, torch.device]] = None
    non_blocking: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataLoader kwargs."""
        config = {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle if not self.distributed else False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
        }

        # Add optional parameters
        if self.num_workers > 0:
            config["persistent_workers"] = self.persistent_workers
            config["prefetch_factor"] = self.prefetch_factor

        return config


def create_data_loader(dataset: Any, config: Optional[DataLoaderConfig] = None, **kwargs: Any) -> DataLoader:
    """
    Create a data loader with proper configuration.

    Args:
        dataset: Dataset to load
        config: Data loader configuration
        **kwargs: Additional keyword arguments for DataLoader

    Returns:
        Configured DataLoader
    """
    if config is None:
        config = DataLoaderConfig()

    # Get base configuration
    loader_kwargs = config.to_dict()

    # Override with any provided kwargs
    loader_kwargs.update(kwargs)

    # Create sampler if needed
    if config.distributed:
        sampler = DistributedSampler(dataset, num_replicas=config.world_size, rank=config.rank, shuffle=config.shuffle, drop_last=config.drop_last)
        loader_kwargs["sampler"] = sampler
        loader_kwargs["shuffle"] = False  # Sampler handles shuffling
        loader_kwargs.pop("drop_last", None)  # Sampler handles this
    elif not loader_kwargs.get("shuffle", True):
        # Use sequential sampler for deterministic ordering
        loader_kwargs["sampler"] = SequentialSampler(dataset)
        loader_kwargs["shuffle"] = False

    return DataLoader(dataset, **loader_kwargs)


def create_distributed_loader(dataset: Any, batch_size: int, shuffle: bool = True, num_workers: int = 4, **kwargs: Any) -> DataLoader:
    """
    Create a data loader for distributed training.

    Automatically detects distributed environment and configures
    the loader appropriately.

    Args:
        dataset: Dataset to load
        batch_size: Batch size per GPU
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        **kwargs: Additional DataLoader arguments

    Returns:
        DataLoader configured for distributed training
    """
    # Check if distributed training is initialized
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        distributed = True
        logger.info(f"Creating distributed loader (rank {rank}/{world_size})")
    else:
        rank = 0
        world_size = 1
        distributed = False
        logger.info("Creating single-process loader")

    config = DataLoaderConfig(
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, distributed=distributed, rank=rank, world_size=world_size, **kwargs
    )

    return create_data_loader(dataset, config)


def collate_with_padding(batch: list, padding_value: float = 0.0, target_padding_value: int = -100):
    """
    Collate function that handles variable-sized inputs.

    Args:
        batch: List of (input, target) tuples
        padding_value: Value to use for input padding
        target_padding_value: Value to use for target padding

    Returns:
        Padded batch tensors
    """
    # Separate inputs and targets
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Find maximum dimensions
    max_dims = [max(inp.size(i) for inp in inputs) for i in range(inputs[0].dim())]

    # Pad inputs
    padded_inputs = []
    for inp in inputs:
        padding = []
        for i in range(inp.dim() - 1, -1, -1):
            padding.extend([0, max_dims[i] - inp.size(i)])
        if any(p > 0 for p in padding):
            padded_inp = torch.nn.functional.pad(inp, padding, value=padding_value)
        else:
            padded_inp = inp
        padded_inputs.append(padded_inp)

    # Stack inputs
    inputs_tensor = torch.stack(padded_inputs)

    # Handle targets
    if isinstance(targets[0], torch.Tensor):
        targets_tensor = torch.stack(targets)
    else:
        targets_tensor = torch.tensor(targets)

    return inputs_tensor, targets_tensor


class InfiniteDataLoader:
    """
    A data loader that loops infinitely over the dataset.

    Useful for training scenarios where you want to run for
    a fixed number of iterations rather than epochs.
    """

    def __init__(self, data_loader: DataLoader):
        """
        Initialize the infinite data loader.

        Args:
            data_loader: Base DataLoader to wrap
        """
        self.data_loader = data_loader
        self._iterator = None

    def __iter__(self):
        """Get iterator."""
        return self

    def __next__(self):
        """Get next batch, restarting if necessary."""
        if self._iterator is None:
            self._iterator = iter(self.data_loader)

        try:
            batch = next(self._iterator)
        except StopIteration:
            # Restart the iterator
            self._iterator = iter(self.data_loader)
            batch = next(self._iterator)

        return batch

    def __len__(self):
        """Length is undefined for infinite loader."""
        return float("inf")
