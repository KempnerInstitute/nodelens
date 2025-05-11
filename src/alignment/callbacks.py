"""
Callbacks for use during training.
"""
import logging
from typing import Dict, List, Any, Callable, Optional

import torch
from torch.utils.data import DataLoader

from alignment.metrics import get_metric, compute_all_node_scores
from alignment.models import AlignmentNetwork # Assuming AlignmentNetwork is in alignment.models
from alignment.config import BaseConfig # For accessing debug_mode if passed via config in ExperimentRunner

logger = logging.getLogger(__name__)

class AlignmentMetricTracker:
    """
    A callback to compute and store alignment metrics at the end of each epoch.
    """
    def __init__(self, 
                 metric_name: str, 
                 data_loader: DataLoader, 
                 device: torch.device,
                 num_batches: int = 5,
                 experiment_config: Optional[BaseConfig] = None): # Pass experiment config for settings like debug_mode
        """
        Args:
            metric_name: Name of the alignment metric to compute (e.g., "RQ", "MI").
            data_loader: DataLoader to use for generating activations for metric computation.
                         Typically the validation loader or a fixed subset of the training loader.
            device: The torch.device to run computations on.
            num_batches: Number of batches from data_loader to use for collecting activations.
            experiment_config: Optional experiment configuration object to access global settings like debug_mode.
        """
        self.metric_name = metric_name
        self.metric_instance = get_metric(self.metric_name)
        self.data_loader = data_loader
        self.device = device
        self.num_batches = num_batches
        self.experiment_config = experiment_config
        self.metrics_history: List[Dict[str, Any]] = [] # To store [(epoch, layer_scores_dict), ...]

    def __call__(self, epoch_context: Dict[str, Any]):
        """
        Called at the end of an epoch by the training loop.
        
        Args:
            epoch_context: Dictionary containing information about the current epoch,
                           including 'model', 'epoch', 'train_loss', 'train_accuracy', etc.
        """
        model = epoch_context.get("model")
        current_epoch = epoch_context.get("epoch")
        
        if model is None or current_epoch is None:
            logger.warning("AlignmentMetricTracker: Model or epoch not found in epoch_context. Skipping.")
            return

        if not isinstance(model, AlignmentNetwork):
            # If not an AlignmentNetwork, it might still be compatible if it has .alignment_layers and .alignment_names
            if not (hasattr(model, "alignment_layers") and hasattr(model, "alignment_names")):
                logger.warning(
                    f"Epoch {current_epoch}, Metric '{self.metric_name}': Model is not an AlignmentNetwork "
                    f"and lacks .alignment_layers/.alignment_names. Skipping alignment metric computation."
                )
                return
            else:
                logger.info(
                    f"Epoch {current_epoch}, Metric '{self.metric_name}': Model is not an AlignmentNetwork, "
                    f"but required attributes .alignment_layers/.alignment_names found. Proceeding."
                )


        debug_mode_callback = False
        if self.experiment_config and hasattr(self.experiment_config, 'debug_mode'):
            debug_mode_callback = self.experiment_config.debug_mode
        
        logger.info(f"Epoch {current_epoch}: Computing alignment metric '{self.metric_name}'...")
        
        original_training_state = model.training
        try:
            model.eval() # Set to eval mode for consistent activation collection
            
            # compute_all_node_scores is imported from alignment.metrics
            node_scores_per_layer = compute_all_node_scores(
                model=model,
                metric_instance=self.metric_instance,
                device=self.device,
                data_loader=self.data_loader,
                num_batches=self.num_batches,
                debug_mode=debug_mode_callback 
            )
            
            # Store detailed scores (dictionary of tensors)
            # Or one could aggregate them here (e.g., mean score per layer)
            # For now, storing the raw dictionary returned by compute_all_node_scores
            self.metrics_history.append({
                "epoch": current_epoch,
                "metric_name": self.metric_name,
                "scores_per_layer": node_scores_per_layer 
            })
            logger.info(f"Epoch {current_epoch}: Finished computing '{self.metric_name}'.")
            
        except Exception as e:
            logger.error(
                f"Epoch {current_epoch}: Error computing alignment metric '{self.metric_name}': {e}", 
                exc_info=debug_mode_callback
            )
        finally:
            if original_training_state:
                model.train() # Restore original training state 