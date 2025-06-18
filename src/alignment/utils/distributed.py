"""
Distributed computing utilities for multi-GPU training.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, List, Union, Dict, Any
import logging

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
    
    def __init__(
        self,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        backend: str = 'nccl'
    ):
        """
        Initialize distributed metric computer.
        
        Args:
            world_size: Total number of processes
            rank: Rank of current process
            backend: Backend to use ('nccl', 'gloo', 'mpi')
        """
        self.backend = backend
        
        if world_size is not None and rank is not None:
            # Manual initialization
            self.world_size = world_size
            self.rank = rank
        else:
            # Try to get from environment or current state
            if dist.is_initialized():
                self.world_size = dist.get_world_size()
                self.rank = dist.get_rank()
            else:
                # Single process mode
                self.world_size = 1
                self.rank = 0
                logger.info("Running in single process mode")
    
    def compute_metrics_distributed(
        self,
        model_wrapper,
        dataloader: torch.utils.data.DataLoader,
        metrics: Dict[str, Any],
        gather_results: bool = True
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute metrics in a distributed manner.
        
        Args:
            model_wrapper: Wrapped model
            dataloader: DataLoader (should use DistributedSampler)
            metrics: Dictionary of metrics to compute
            gather_results: Whether to gather results across all ranks
            
        Returns:
            Results dictionary
        """
        device = torch.device(f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu')
        model_wrapper.model.to(device)
        
        # Local computation
        local_results = {}
        local_counts = {}
        
        for batch_idx, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            
            # Get activations and weights
            outputs, activations = model_wrapper.forward_with_activations(inputs)
            weights = model_wrapper.get_layer_weights()
            
            # Process each layer
            for layer_name in model_wrapper.tracked_layers:
                if layer_name not in local_results:
                    local_results[layer_name] = {}
                    local_counts[layer_name] = {}
                
                layer_inputs = activations.get(f"{layer_name}_input")
                layer_weights = weights.get(layer_name)
                layer_outputs = activations.get(f"{layer_name}_output", outputs)
                
                # Compute metrics
                for metric_name, metric in metrics.items():
                    scores = metric.compute(
                        inputs=layer_inputs,
                        weights=layer_weights,
                        outputs=layer_outputs
                    )
                    
                    # Accumulate results
                    if metric_name not in local_results[layer_name]:
                        local_results[layer_name][metric_name] = scores
                        local_counts[layer_name][metric_name] = 1
                    else:
                        local_results[layer_name][metric_name] += scores
                        local_counts[layer_name][metric_name] += 1
        
        # Average local results
        for layer_name in local_results:
            for metric_name in local_results[layer_name]:
                count = local_counts[layer_name][metric_name]
                local_results[layer_name][metric_name] /= count
        
        # Gather results if requested
        if gather_results and self.world_size > 1:
            return self._gather_results(local_results)
        else:
            return local_results
    
    def _gather_results(
        self,
        local_results: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Gather results from all ranks.
        
        Args:
            local_results: Local results from this rank
            
        Returns:
            Aggregated results
        """
        global_results = {}
        
        for layer_name, layer_metrics in local_results.items():
            global_results[layer_name] = {}
            
            for metric_name, scores in layer_metrics.items():
                # Gather tensors from all ranks
                gathered = gather_tensor(scores, dst=0)
                
                if gathered is not None:  # We're on rank 0
                    # Average the gathered results
                    global_results[layer_name][metric_name] = torch.stack(gathered).mean(dim=0)
                else:
                    # For non-rank-0 processes, we can optionally broadcast the result
                    global_results[layer_name][metric_name] = scores
        
        # Optionally broadcast results to all ranks
        if self.rank == 0:
            for layer_name in global_results:
                for metric_name in global_results[layer_name]:
                    global_results[layer_name][metric_name] = broadcast(
                        global_results[layer_name][metric_name], src=0
                    )
        
        return global_results 