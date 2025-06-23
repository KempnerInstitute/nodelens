"""
Distributed computing utilities for multi-GPU training.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, List, Union, Dict, Any, Callable, Tuple
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def setup_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None
) -> bool:
    """
    Setup distributed training environment.
   e
    Args:
        backend: Backend to use ('nccl', 'gloo')
        init_method: URL specifying how to initialize the process group
        world_size: Number of processes
        rank: Rank of the current process
        
    Returns:
        True if distributed setup successful, False otherwise
    """
    if not torch.cuda.is_available() and backend == "nccl":
        logger.warning("CUDA not available, falling back to gloo backend")
        backend = "gloo"
    
    # Try to get from environment
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    
    if world_size > 1:
        try:
            if init_method is None:
                init_method = os.environ.get("INIT_METHOD", "env://")
            
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank
            )
            
            if torch.cuda.is_available():
                torch.cuda.set_device(rank % torch.cuda.device_count())
            
            logger.info(f"Initialized distributed training: rank {rank}/{world_size}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            return False
    
    return False


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Cleaned up distributed training")


def is_main_process() -> bool:
    """Check if this is the main process."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def reduce_tensor(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    dst: int = 0
) -> torch.Tensor:
    """
    Reduce tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation
        dst: Destination rank
        
    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor
    
    tensor = tensor.clone()
    dist.reduce(tensor, dst=dst, op=op)
    
    if op == dist.ReduceOp.SUM and is_main_process():
        tensor = tensor / dist.get_world_size()
    
    return tensor


def gather_tensor(
    tensor: torch.Tensor,
    dst: int = 0
) -> Optional[List[torch.Tensor]]:
    """
    Gather tensors from all processes.
    
    Args:
        tensor: Local tensor
        dst: Destination rank
        
    Returns:
        List of tensors if on destination rank, None otherwise
    """
    if not dist.is_initialized():
        return [tensor]
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if rank == dst:
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.gather(tensor, gathered, dst=dst)
        return gathered
    else:
        dist.gather(tensor, dst=dst)
        return None


def all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM
) -> torch.Tensor:
    """
    All-reduce tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation
        
    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor
    
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=op)
    
    if op == dist.ReduceOp.SUM:
        tensor = tensor / dist.get_world_size()
    
    return tensor


def broadcast(
    tensor: torch.Tensor,
    src: int = 0
) -> torch.Tensor:
    """
    Broadcast tensor from source to all processes.
    
    Args:
        tensor: Tensor to broadcast
        src: Source rank
        
    Returns:
        Broadcasted tensor
    """
    if not dist.is_initialized():
        return tensor
    
    tensor = tensor.clone()
    dist.broadcast(tensor, src=src)
    return tensor


class DistributedMetricComputer:
    """
    Compute metrics in a distributed manner across multiple GPUs/nodes.
    
    This class provides high-level functionality for distributed metric computation,
    building on the basic distributed utilities.
    """
    
    def __init__(self, backend: str = 'nccl'):
        """
        Initialize distributed computing.
        
        Args:
            backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)
        """
        self.backend = backend
        self.initialized = False
        self.rank = 0
        self.world_size = 1
    
    def setup(self, rank: Optional[int] = None, world_size: Optional[int] = None):
        """
        Setup distributed computing environment.
        
        Args:
            rank: Process rank (auto-detected if None)
            world_size: Total number of processes (auto-detected if None)
        """
        if self.initialized:
            return
        
        # Auto-detect from environment
        if rank is None:
            rank = int(os.environ.get('RANK', 0))
        if world_size is None:
            world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        self.rank = rank
        self.world_size = world_size
        
        if world_size > 1:
            # Initialize process group
            dist.init_process_group(
                backend=self.backend,
                rank=rank,
                world_size=world_size
            )
            self.initialized = True
    
    def cleanup(self):
        """Cleanup distributed environment."""
        if self.initialized and dist.is_initialized():
            dist.destroy_process_group()
            self.initialized = False
    
    @contextmanager
    def distributed_context(self):
        """Context manager for distributed operations."""
        try:
            yield
        finally:
            if self.initialized:
                dist.barrier()
    
    def all_gather_metrics(self, local_metric: torch.Tensor) -> List[torch.Tensor]:
        """
        Gather metrics from all processes.
        
        Args:
            local_metric: Local metric value
            
        Returns:
            List of metrics from all processes
        """
        if not self.initialized or self.world_size == 1:
            return [local_metric]
        
        # Ensure tensor is on correct device
        if not local_metric.is_cuda and self.backend == 'nccl':
            local_metric = local_metric.cuda()
        
        # Gather from all processes
        gathered = [torch.zeros_like(local_metric) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_metric)
        
        return gathered
    
    def reduce_metrics(self, 
                      local_metric: torch.Tensor,
                      reduction: str = 'mean') -> torch.Tensor:
        """
        Reduce metrics across all processes.
        
        Args:
            local_metric: Local metric value
            reduction: Reduction operation ('mean', 'sum', 'max', 'min')
            
        Returns:
            Reduced metric
        """
        if not self.initialized or self.world_size == 1:
            return local_metric
        
        # Clone to avoid modifying original
        metric = local_metric.clone()
        
        if not metric.is_cuda and self.backend == 'nccl':
            metric = metric.cuda()
        
        # Reduce across processes
        if reduction == 'sum':
            dist.all_reduce(metric, op=dist.ReduceOp.SUM)
        elif reduction == 'mean':
            dist.all_reduce(metric, op=dist.ReduceOp.SUM)
            metric = metric / self.world_size
        elif reduction == 'max':
            dist.all_reduce(metric, op=dist.ReduceOp.MAX)
        elif reduction == 'min':
            dist.all_reduce(metric, op=dist.ReduceOp.MIN)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
        
        return metric
    
    def distributed_metric_computation(self,
                                     metric_fn: Callable,
                                     data_loader: torch.utils.data.DataLoader,
                                     **metric_kwargs) -> Dict[str, float]:
        """
        Compute metrics in distributed fashion.
        
        Args:
            metric_fn: Metric computation function
            data_loader: Distributed data loader
            **metric_kwargs: Additional arguments for metric function
            
        Returns:
            Dictionary of computed metrics
        """
        local_results = []
        
        with self.distributed_context():
            for batch in data_loader:
                # Compute metric on local batch
                result = metric_fn(batch, **metric_kwargs)
                local_results.append(result)
        
        # Aggregate local results
        if local_results:
            local_metric = torch.tensor(
                sum(local_results) / len(local_results),
                device='cuda' if self.backend == 'nccl' else 'cpu'
            )
        else:
            local_metric = torch.tensor(0.0)
        
        # Reduce across all processes
        global_metric = self.reduce_metrics(local_metric, reduction='mean')
        
        return {'metric': global_metric.item()}


class DistributedModelWrapper:
    """Wrapper for distributed model evaluation."""
    
    def __init__(self, model: torch.nn.Module, device_ids: Optional[List[int]] = None):
        """
        Initialize distributed model wrapper.
        
        Args:
            model: Model to wrap
            device_ids: GPU device IDs to use
        """
        self.model = model
        self.device_ids = device_ids
        self.ddp_model = None
    
    def setup_ddp(self, rank: int):
        """Setup DistributedDataParallel."""
        if self.device_ids:
            device = self.device_ids[rank % len(self.device_ids)]
        else:
            device = rank
        
        torch.cuda.set_device(device)
        self.model = self.model.cuda(device)
        
        self.ddp_model = DDP(
            self.model,
            device_ids=[device],
            output_device=device
        )
    
    def get_model(self) -> torch.nn.Module:
        """Get the wrapped model."""
        return self.ddp_model if self.ddp_model is not None else self.model


def distributed_metric_aggregation(
    metric_computer: Any,
    data_partitions: List[Tuple[torch.Tensor, ...]],
    metric_name: str,
    **compute_kwargs
) -> float:
    """
    Aggregate metric computation across data partitions.
    
    Args:
        metric_computer: Metric computation object
        data_partitions: List of data partitions (inputs, weights, outputs)
        metric_name: Name of metric to compute
        **compute_kwargs: Additional arguments for compute
        
    Returns:
        Aggregated metric value
    """
    dist_computer = DistributedMetricComputer()
    dist_computer.setup()
    
    try:
        # Compute local metrics
        local_scores = []
        for partition in data_partitions:
            inputs, weights, outputs = partition
            score = metric_computer.compute(
                inputs=inputs,
                weights=weights,
                outputs=outputs,
                **compute_kwargs
            )
            local_scores.append(score)
        
        # Average local scores
        if local_scores:
            local_avg = sum(local_scores) / len(local_scores)
            local_tensor = torch.tensor(local_avg, dtype=torch.float32)
        else:
            local_tensor = torch.tensor(0.0, dtype=torch.float32)
        
        # Reduce across processes
        global_avg = dist_computer.reduce_metrics(local_tensor, reduction='mean')
        
        return global_avg.item()
        
    finally:
        dist_computer.cleanup()


class DistributedBatchProcessor:
    """Process batches in distributed fashion with automatic load balancing."""
    
    def __init__(self, 
                 world_size: int,
                 rank: int,
                 device: Optional[torch.device] = None):
        """
        Initialize distributed batch processor.
        
        Args:
            world_size: Total number of processes
            rank: Current process rank
            device: Device to use (auto-detected if None)
        """
        self.world_size = world_size
        self.rank = rank
        self.device = device or torch.device(f'cuda:{rank}')
    
    def split_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Split batch across processes.
        
        Args:
            batch: Input batch
            
        Returns:
            Local portion of batch
        """
        batch_size = batch.size(0)
        chunk_size = (batch_size + self.world_size - 1) // self.world_size
        
        start_idx = self.rank * chunk_size
        end_idx = min(start_idx + chunk_size, batch_size)
        
        if start_idx < batch_size:
            return batch[start_idx:end_idx].to(self.device)
        else:
            # Return empty tensor if this rank has no data
            return torch.empty(0, *batch.shape[1:], device=self.device)
    
    def gather_results(self, local_result: torch.Tensor) -> torch.Tensor:
        """
        Gather results from all processes.
        
        Args:
            local_result: Local computation result
            
        Returns:
            Concatenated results from all processes
        """
        # Get sizes from all processes
        local_size = torch.tensor(local_result.size(0), device=self.device)
        sizes = [torch.zeros_like(local_size) for _ in range(self.world_size)]
        dist.all_gather(sizes, local_size)
        
        # Gather tensors with variable sizes
        max_size = max(s.item() for s in sizes)
        padded_result = torch.zeros(max_size, *local_result.shape[1:], device=self.device)
        if local_result.size(0) > 0:
            padded_result[:local_result.size(0)] = local_result
        
        gathered = [torch.zeros_like(padded_result) for _ in range(self.world_size)]
        dist.all_gather(gathered, padded_result)
        
        # Concatenate non-padded portions
        results = []
        for i, size in enumerate(sizes):
            if size > 0:
                results.append(gathered[i][:size])
        
        return torch.cat(results, dim=0) if results else torch.empty(0, *local_result.shape[1:]) 