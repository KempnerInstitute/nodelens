"""
Distributed computing utilities for multi-GPU training.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, List, Union
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