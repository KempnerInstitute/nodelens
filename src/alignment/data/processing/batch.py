"""
Batch processing utilities for efficient metric computation on large datasets.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BatchMetricProcessor:
    """
    Efficiently process metrics over large datasets using batched computation.
    
    This class handles:
    - Memory-efficient batch processing
    - Progress tracking
    - Accumulation strategies
    - GPU memory management
    """
    
    def __init__(
        self,
        device: torch.device = None,
        max_memory_gb: float = 8.0,
        show_progress: bool = True
    ):
        """
        Initialize batch processor.
        
        Args:
            device: Device to use for computation
            max_memory_gb: Maximum GPU memory to use (in GB)
            show_progress: Whether to show progress bar
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_memory_gb = max_memory_gb
        self.show_progress = show_progress
        
    def process_dataset(
        self,
        model_wrapper,
        dataloader: DataLoader,
        metrics: Dict[str, Any],
        num_batches: Optional[int] = None,
        accumulation_strategy: str = 'concatenate'
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Process metrics over entire dataset.
        
        Args:
            model_wrapper: Wrapped model for activation extraction
            dataloader: DataLoader for the dataset
            metrics: Dictionary of metric_name -> metric_instance
            num_batches: Limit number of batches to process
            accumulation_strategy: How to combine results ('concatenate', 'average', 'running_mean')
            
        Returns:
            Dictionary of layer_name -> metric_name -> scores
        """
        results = {}
        accumulators = {}
        
        # Initialize progress bar
        total_batches = len(dataloader) if num_batches is None else min(num_batches, len(dataloader))
        pbar = tqdm(total=total_batches, disable=not self.show_progress)
        
        try:
            for batch_idx, (inputs, _) in enumerate(dataloader):
                if num_batches and batch_idx >= num_batches:
                    break
                
                # Move to device
                inputs = inputs.to(self.device)
                
                # Check memory usage
                if self.device.type == 'cuda':
                    self._check_gpu_memory()
                
                # Get activations and weights
                outputs, activations = model_wrapper.forward_with_activations(inputs)
                weights = model_wrapper.get_layer_weights()
                
                # Process each layer
                for layer_name in model_wrapper.tracked_layers:
                    if layer_name not in results:
                        results[layer_name] = {}
                        accumulators[layer_name] = {}
                    
                    # Get layer data
                    layer_inputs = activations.get(f"{layer_name}_input")
                    layer_weights = weights.get(layer_name)
                    layer_outputs = activations.get(f"{layer_name}_output", outputs)
                    
                    # Compute metrics
                    for metric_name, metric in metrics.items():
                        scores = self._compute_metric_safe(
                            metric, 
                            inputs=layer_inputs,
                            weights=layer_weights,
                            outputs=layer_outputs
                        )
                        
                        # Accumulate results
                        if metric_name not in accumulators[layer_name]:
                            accumulators[layer_name][metric_name] = []
                        
                        accumulators[layer_name][metric_name].append(scores.cpu())
                
                # Update progress
                pbar.update(1)
                pbar.set_description(f"Batch {batch_idx+1}/{total_batches}")
                
                # Clear GPU cache periodically
                if self.device.type == 'cuda' and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        
        finally:
            pbar.close()
        
        # Combine accumulated results
        for layer_name in accumulators:
            for metric_name in accumulators[layer_name]:
                scores_list = accumulators[layer_name][metric_name]
                
                if accumulation_strategy == 'concatenate':
                    # For metrics that need all data at once
                    results[layer_name][metric_name] = torch.cat(scores_list, dim=0)
                elif accumulation_strategy == 'average':
                    # Simple average across batches
                    results[layer_name][metric_name] = torch.stack(scores_list).mean(dim=0)
                elif accumulation_strategy == 'running_mean':
                    # Weighted running mean
                    result = scores_list[0]
                    for i, scores in enumerate(scores_list[1:], 1):
                        alpha = 1.0 / (i + 1)
                        result = (1 - alpha) * result + alpha * scores
                    results[layer_name][metric_name] = result
        
        return results
    
    def _compute_metric_safe(self, metric, **kwargs) -> torch.Tensor:
        """Safely compute metric with error handling."""
        try:
            return metric.compute(**kwargs)
        except Exception as e:
            logger.error(f"Error computing metric {metric.name}: {e}")
            # Return zeros as fallback
            if 'weights' in kwargs and kwargs['weights'] is not None:
                return torch.zeros(kwargs['weights'].shape[0])
            elif 'outputs' in kwargs and kwargs['outputs'] is not None:
                return torch.zeros(kwargs['outputs'].shape[1])
            else:
                return torch.zeros(1)
    
    def _check_gpu_memory(self):
        """Check GPU memory usage and warn if approaching limit."""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9  # Convert to GB
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            
            if memory_used > self.max_memory_gb * 0.9:
                logger.warning(
                    f"GPU memory usage high: {memory_used:.2f}GB used, "
                    f"{memory_reserved:.2f}GB reserved"
                )


class StreamingMetricComputer:
    """
    Compute metrics in a streaming fashion for extremely large datasets.
    
    This is useful when even batch processing would exceed memory limits.
    """
    
    def __init__(self, buffer_size: int = 1000):
        """
        Initialize streaming computer.
        
        Args:
            buffer_size: Size of internal buffer for accumulation
        """
        self.buffer_size = buffer_size
        self.reset()
    
    def reset(self):
        """Reset internal buffers."""
        self.buffers = {}
        self.counts = {}
    
    def update(self, layer_name: str, metric_name: str, values: torch.Tensor):
        """
        Update streaming statistics with new values.
        
        Args:
            layer_name: Name of the layer
            metric_name: Name of the metric
            values: New values to incorporate
        """
        key = (layer_name, metric_name)
        
        if key not in self.buffers:
            self.buffers[key] = []
            self.counts[key] = 0
        
        # Add to buffer
        self.buffers[key].append(values.cpu())
        self.counts[key] += values.shape[0] if values.ndim > 0 else 1
        
        # Process buffer if full
        if len(self.buffers[key]) >= self.buffer_size:
            self._process_buffer(key)
    
    def _process_buffer(self, key: Tuple[str, str]):
        """Process and clear buffer."""
        # This is a simplified version - in practice you might want
        # to implement more sophisticated streaming algorithms
        values = torch.cat(self.buffers[key])
        
        # Compute running statistics
        mean = values.mean()
        std = values.std()
        
        # Store summary statistics instead of all values
        self.buffers[key] = [torch.tensor([mean, std])]
    
    def get_results(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get final results."""
        results = {}
        
        for (layer_name, metric_name), buffer in self.buffers.items():
            if layer_name not in results:
                results[layer_name] = {}
            
            # Process any remaining buffer
            if len(buffer) > 1:
                self._process_buffer((layer_name, metric_name))
            
            results[layer_name][metric_name] = buffer[0] if buffer else torch.zeros(2)
        
        return results


def compute_metrics_parallel(
    model_wrapper,
    dataloader: DataLoader,
    metrics: Dict[str, Any],
    num_workers: int = 4,
    devices: Optional[List[torch.device]] = None
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute metrics in parallel across multiple GPUs.
    
    Args:
        model_wrapper: Wrapped model
        dataloader: Data loader
        metrics: Dictionary of metrics to compute
        num_workers: Number of parallel workers
        devices: List of devices to use (defaults to all available GPUs)
        
    Returns:
        Results dictionary
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import torch.multiprocessing as mp

    # Determine devices
    if devices is None:
        if torch.cuda.is_available():
            devices = [torch.device(f'cuda:{i}') for i in range(min(num_workers, torch.cuda.device_count()))]
        else:
            devices = [torch.device('cpu')] * num_workers
    
    # If only one device or no GPU, use regular processing
    if len(devices) <= 1:
        processor = BatchMetricProcessor(device=devices[0] if devices else None)
        return processor.process_dataset(model_wrapper, dataloader, metrics)
    
    # Split data across workers
    dataset = dataloader.dataset
    chunk_size = len(dataset) // num_workers
    chunks = []
    
    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_workers - 1 else len(dataset)
        subset = torch.utils.data.Subset(dataset, range(start_idx, end_idx))
        chunks.append(subset)
    
    # Process chunks in parallel
    def process_chunk(chunk_data, device_id):
        """Process a data chunk on a specific device."""
        device = devices[device_id % len(devices)]
        
        # Create dataloader for chunk
        chunk_loader = DataLoader(
            chunk_data,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=0  # Avoid nested multiprocessing
        )
        
        # Create processor for this device
        processor = BatchMetricProcessor(device=device, show_progress=False)
        
        # Move model to device
        model_wrapper.to(device)
        
        # Process chunk
        return processor.process_dataset(model_wrapper, chunk_loader, metrics)
    
    # Execute parallel processing
    all_results = []
    
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all chunks
            futures = {
                executor.submit(process_chunk, chunk, i): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
    
    except Exception as e:
        logger.error(f"Parallel processing failed: {e}")
        # Fallback to sequential processing
        processor = BatchMetricProcessor()
        return processor.process_dataset(model_wrapper, dataloader, metrics)
    
    # Merge results from all workers
    merged_results = {}
    
    for result in all_results:
        for layer_name, layer_metrics in result.items():
            if layer_name not in merged_results:
                merged_results[layer_name] = {}
            
            for metric_name, scores in layer_metrics.items():
                if metric_name not in merged_results[layer_name]:
                    merged_results[layer_name][metric_name] = []
                
                merged_results[layer_name][metric_name].append(scores)
    
    # Concatenate scores from all workers
    for layer_name in merged_results:
        for metric_name in merged_results[layer_name]:
            scores_list = merged_results[layer_name][metric_name]
            merged_results[layer_name][metric_name] = torch.cat(scores_list, dim=0)
    
    return merged_results


def batch_mutual_information(
    X: torch.Tensor,
    Y: torch.Tensor,
    bins: int = 10,
    batch_size: int = 1000,
    method: str = "histogram"
) -> torch.Tensor:
    """
    Compute mutual information for multiple variable pairs in batches.
    
    This function efficiently computes MI for many pairs of variables,
    processing them in batches to manage memory usage.
    
    Args:
        X: First variables (n_pairs, n_samples)
        Y: Second variables (n_pairs, n_samples)
        bins: Number of bins for histogram method
        batch_size: Process this many pairs at once
        method: MI estimation method ('histogram' or 'kraskov')
        
    Returns:
        MI values for each pair
    """
    # Import the optimized MI function
    from .optimized import gpu_mutual_information
    
    n_pairs = X.shape[0]
    mi_values = torch.zeros(n_pairs, device=X.device)
    
    for i in range(0, n_pairs, batch_size):
        end_idx = min(i + batch_size, n_pairs)
        batch_x = X[i:end_idx]
        batch_y = Y[i:end_idx]
        
        # Compute MI for each pair in batch
        for j in range(batch_x.shape[0]):
            mi_values[i + j] = gpu_mutual_information(
                batch_x[j], batch_y[j], bins=bins, method=method
            )
    
    return mi_values 