"""
Checkpoint management utilities.
"""

from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import torch
import json
import logging
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


def save_checkpoint(
    state: Dict[str, Any],
    filepath: Union[str, Path],
    is_best: bool = False,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save a checkpoint.
    
    Args:
        state: State dictionary to save
        filepath: Path to save checkpoint
        is_best: Whether this is the best checkpoint
        metadata: Additional metadata to save
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    checkpoint = {
        'state': state,
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
    }
    
    if metadata:
        checkpoint['metadata'] = metadata
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    logger.info(f"Saved checkpoint to {filepath}")
    
    # Copy to best if needed
    if is_best:
        best_path = filepath.parent / 'best_checkpoint.pt'
        shutil.copy2(filepath, best_path)
        logger.info(f"Saved best checkpoint to {best_path}")


def load_checkpoint(
    filepath: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None
) -> Dict[str, Any]:
    """
    Load a checkpoint.
    
    Args:
        filepath: Path to checkpoint
        map_location: Device mapping location
        
    Returns:
        Checkpoint dictionary
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=map_location)
    logger.info(f"Loaded checkpoint from {filepath}")
    
    # Handle old format checkpoints
    if 'state' not in checkpoint:
        # Assume entire checkpoint is the state
        checkpoint = {'state': checkpoint}
    
    return checkpoint


class CheckpointManager:
    """
    Manages checkpoints with automatic cleanup and best model tracking.
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        metric_name: Optional[str] = None,
        mode: str = 'min'
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            metric_name: Metric to track for best checkpoint
            mode: 'min' or 'max' for metric comparison
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.mode = mode
        
        self.checkpoints = []
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_checkpoint = None
        
        # Load existing checkpoints
        self._load_checkpoint_history()
    
    def _load_checkpoint_history(self):
        """Load checkpoint history from directory."""
        history_file = self.checkpoint_dir / 'checkpoint_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                data = json.load(f)
                self.checkpoints = data.get('checkpoints', [])
                self.best_metric = data.get('best_metric', self.best_metric)
                self.best_checkpoint = data.get('best_checkpoint')
    
    def _save_checkpoint_history(self):
        """Save checkpoint history."""
        history_file = self.checkpoint_dir / 'checkpoint_history.json'
        with open(history_file, 'w') as f:
            json.dump({
                'checkpoints': self.checkpoints,
                'best_metric': self.best_metric,
                'best_checkpoint': self.best_checkpoint
            }, f, indent=2)
    
    def save(
        self,
        state: Dict[str, Any],
        step: int,
        metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            state: State to save
            step: Current step/epoch
            metrics: Current metrics
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint name
        checkpoint_name = f'checkpoint_step_{step}.pt'
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Check if best
        is_best = False
        if self.metric_name and metrics and self.metric_name in metrics:
            metric_value = metrics[self.metric_name]
            if self.mode == 'min':
                is_best = metric_value < self.best_metric
            else:
                is_best = metric_value > self.best_metric
            
            if is_best:
                self.best_metric = metric_value
                self.best_checkpoint = checkpoint_name
        
        # Save checkpoint
        save_checkpoint(
            state,
            checkpoint_path,
            is_best=is_best,
            metadata={'step': step, 'metrics': metrics}
        )
        
        # Update history
        self.checkpoints.append({
            'filename': checkpoint_name,
            'step': step,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        # Save history
        self._save_checkpoint_history()
        
        return str(checkpoint_path)
    
    def load_latest(
        self,
        map_location: Optional[Union[str, torch.device]] = None
    ) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint."""
        if not self.checkpoints:
            return None
        
        latest = self.checkpoints[-1]
        checkpoint_path = self.checkpoint_dir / latest['filename']
        return load_checkpoint(checkpoint_path, map_location)
    
    def load_best(
        self,
        map_location: Optional[Union[str, torch.device]] = None
    ) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint."""
        if not self.best_checkpoint:
            return None
        
        checkpoint_path = self.checkpoint_dir / self.best_checkpoint
        return load_checkpoint(checkpoint_path, map_location)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by step
        self.checkpoints.sort(key=lambda x: x['step'])
        
        # Keep best checkpoint
        to_remove = []
        while len(self.checkpoints) - len(to_remove) > self.max_checkpoints:
            for i, ckpt in enumerate(self.checkpoints):
                if ckpt['filename'] != self.best_checkpoint:
                    to_remove.append(i)
                    break
            if len(self.checkpoints) - len(to_remove) <= self.max_checkpoints:
                break
        
        # Remove checkpoints
        for idx in reversed(to_remove):
            ckpt = self.checkpoints.pop(idx)
            checkpoint_path = self.checkpoint_dir / ckpt['filename']
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"Removed old checkpoint: {ckpt['filename']}") 