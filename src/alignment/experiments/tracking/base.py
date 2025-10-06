"""
Experiment tracking integration for alignment metrics.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Base class for experiment tracking.
    """
    
    def __init__(self, experiment_name: str, config: Dict[str, Any]):
        self.experiment_name = experiment_name
        self.config = config
        self.step = 0
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log scalar metrics."""
        # Base implementation: just log to console
        if step is None:
            step = self.step
        logger.info(f"[Step {step}] Metrics: {metrics}")
    
    def log_histogram(self, name: str, values: Union[torch.Tensor, np.ndarray], step: Optional[int] = None):
        """Log histogram of values."""
        # Base implementation: log summary statistics
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        if step is None:
            step = self.step
        logger.info(f"[Step {step}] {name} - mean: {np.mean(values):.4f}, std: {np.std(values):.4f}")
    
    def log_image(self, name: str, image: np.ndarray, step: Optional[int] = None):
        """Log an image."""
        # Base implementation: log image info
        if step is None:
            step = self.step
        logger.info(f"[Step {step}] Image '{name}' - shape: {image.shape}")
    
    def finish(self):
        """Finish tracking."""
        pass


class WandBTracker(ExperimentTracker):
    """
    Weights & Biases integration for experiment tracking.
    """
    
    def __init__(
        self, 
        experiment_name: str, 
        config: Dict[str, Any],
        project: str = "alignment-metrics",
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        super().__init__(experiment_name, config)
        
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=experiment_name,
                config=config,
                tags=tags or []
            )
            self.enabled = True
            logger.info(f"WandB tracking initialized: {self.run.url}")
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
            self.enabled = False
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log scalar metrics to WandB."""
        if not self.enabled:
            return
        
        if step is None:
            step = self.step
            self.step += 1
        
        self.wandb.log(metrics, step=step)
    
    def log_histogram(self, name: str, values: Union[torch.Tensor, np.ndarray], step: Optional[int] = None):
        """Log histogram to WandB."""
        if not self.enabled:
            return
        
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        
        if step is None:
            step = self.step
        
        self.wandb.log({name: self.wandb.Histogram(values)}, step=step)
    
    def log_image(self, name: str, image: np.ndarray, step: Optional[int] = None):
        """Log image to WandB."""
        if not self.enabled:
            return
        
        if step is None:
            step = self.step
        
        self.wandb.log({name: self.wandb.Image(image)}, step=step)
    
    def log_alignment_scores(
        self, 
        layer_scores: Dict[str, Dict[str, torch.Tensor]], 
        step: Optional[int] = None
    ):
        """Log alignment scores with proper organization."""
        if not self.enabled:
            return
        
        for layer_name, metrics in layer_scores.items():
            for metric_name, scores in metrics.items():
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                
                # Log summary statistics
                self.log_metrics({
                    f"{layer_name}/{metric_name}/mean": np.mean(scores),
                    f"{layer_name}/{metric_name}/std": np.std(scores),
                    f"{layer_name}/{metric_name}/min": np.min(scores),
                    f"{layer_name}/{metric_name}/max": np.max(scores),
                }, step=step)
                
                # Log histogram
                self.log_histogram(f"{layer_name}/{metric_name}/distribution", scores, step=step)
    
    def finish(self):
        """Finish WandB run."""
        if self.enabled:
            self.wandb.finish()


class TensorBoardTracker(ExperimentTracker):
    """
    TensorBoard integration for experiment tracking.
    """
    
    def __init__(
        self, 
        experiment_name: str, 
        config: Dict[str, Any],
        log_dir: str = "./runs"
    ):
        super().__init__(experiment_name, config)
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.log_dir = Path(log_dir) / experiment_name
            self.writer = SummaryWriter(self.log_dir)
            self.enabled = True
            
            # Log config as text
            config_str = json.dumps(config, indent=2)
            self.writer.add_text("config", f"```json\n{config_str}\n```", 0)
            
            logger.info(f"TensorBoard tracking initialized: {self.log_dir}")
        except ImportError:
            logger.warning("tensorboard not installed. Install with: pip install tensorboard")
            self.enabled = False
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log scalar metrics to TensorBoard."""
        if not self.enabled:
            return
        
        if step is None:
            step = self.step
            self.step += 1
        
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
    
    def log_histogram(self, name: str, values: Union[torch.Tensor, np.ndarray], step: Optional[int] = None):
        """Log histogram to TensorBoard."""
        if not self.enabled:
            return
        
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values)
        
        if step is None:
            step = self.step
        
        self.writer.add_histogram(name, values, step)
    
    def log_image(self, name: str, image: np.ndarray, step: Optional[int] = None):
        """Log image to TensorBoard."""
        if not self.enabled:
            return
        
        if step is None:
            step = self.step
        
        # TensorBoard expects CHW format
        if image.ndim == 3 and image.shape[-1] == 3:
            image = image.transpose(2, 0, 1)
        
        self.writer.add_image(name, image, step)
    
    def log_alignment_scores(
        self, 
        layer_scores: Dict[str, Dict[str, torch.Tensor]], 
        step: Optional[int] = None
    ):
        """Log alignment scores with proper organization."""
        if not self.enabled:
            return
        
        for layer_name, metrics in layer_scores.items():
            for metric_name, scores in metrics.items():
                if isinstance(scores, torch.Tensor):
                    scores_np = scores.cpu().numpy()
                else:
                    scores_np = scores
                
                # Log summary statistics
                self.log_metrics({
                    f"{layer_name}/{metric_name}/mean": np.mean(scores_np),
                    f"{layer_name}/{metric_name}/std": np.std(scores_np),
                    f"{layer_name}/{metric_name}/min": np.min(scores_np),
                    f"{layer_name}/{metric_name}/max": np.max(scores_np),
                }, step=step)
                
                # Log histogram
                self.log_histogram(f"{layer_name}/{metric_name}", scores, step=step)
    
    def finish(self):
        """Close TensorBoard writer."""
        if self.enabled:
            self.writer.close()


class MultiTracker(ExperimentTracker):
    """
    Use multiple trackers simultaneously.
    """
    
    def __init__(self, trackers: List[ExperimentTracker]):
        self.trackers = trackers
        self.step = 0
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log to all trackers."""
        for tracker in self.trackers:
            tracker.log_metrics(metrics, step)
    
    def log_histogram(self, name: str, values: Union[torch.Tensor, np.ndarray], step: Optional[int] = None):
        """Log to all trackers."""
        for tracker in self.trackers:
            tracker.log_histogram(name, values, step)
    
    def log_image(self, name: str, image: np.ndarray, step: Optional[int] = None):
        """Log to all trackers."""
        for tracker in self.trackers:
            tracker.log_image(name, image, step)
    
    def log_alignment_scores(
        self, 
        layer_scores: Dict[str, Dict[str, torch.Tensor]], 
        step: Optional[int] = None
    ):
        """Log to all trackers that support this method."""
        for tracker in self.trackers:
            if hasattr(tracker, 'log_alignment_scores'):
                tracker.log_alignment_scores(layer_scores, step)
    
    def finish(self):
        """Finish all trackers."""
        for tracker in self.trackers:
            tracker.finish()


def create_tracker(
    tracker_type: str,
    experiment_name: str,
    config: Dict[str, Any],
    **kwargs
) -> ExperimentTracker:
    """
    Factory function to create experiment trackers.
    
    Args:
        tracker_type: Type of tracker ('wandb', 'tensorboard', 'both')
        experiment_name: Name of the experiment
        config: Configuration dictionary
        **kwargs: Additional arguments for specific trackers
        
    Returns:
        ExperimentTracker instance
    """
    if tracker_type == 'wandb':
        return WandBTracker(experiment_name, config, **kwargs)
    elif tracker_type == 'tensorboard':
        return TensorBoardTracker(experiment_name, config, **kwargs)
    elif tracker_type == 'both':
        wandb_tracker = WandBTracker(experiment_name, config, **kwargs)
        tb_tracker = TensorBoardTracker(experiment_name, config, **kwargs)
        return MultiTracker([wandb_tracker, tb_tracker])
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")


class DummyTracker(ExperimentTracker):
    """
    Dummy tracker that does nothing (for when tracking is disabled).
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__("dummy", {})
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        pass
    
    def log_histogram(self, name: str, values: Union[torch.Tensor, np.ndarray], step: Optional[int] = None):
        pass
    
    def log_image(self, name: str, image: np.ndarray, step: Optional[int] = None):
        pass 