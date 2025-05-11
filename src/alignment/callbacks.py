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
    A callback to compute and store specified alignment metrics at the end of each epoch.
    """
    def __init__(self, 
                 metric_configs: List[Dict[str, Any]], 
                 data_loader: DataLoader, 
                 device: torch.device,
                 num_batches: Optional[int] = 5,
                 experiment_config: Optional[BaseConfig] = None,
                 global_cnn_mode: Optional[str] = "unfold",
                 global_cnn_rq_aggregation_op: Optional[str] = "mean"):
        """
        Args:
            metric_configs: List of metric configurations (dicts). Each dict should have "name"
                            and optionally "scale_by_norm".
            data_loader: DataLoader to use for generating activations for metric computation.
                         Typically the validation loader or a fixed subset of the training loader.
            device: The torch.device to run computations on.
            num_batches: Number of batches from data_loader to use for collecting activations.
            experiment_config: Optional experiment configuration object to access global settings like debug_mode.
            global_cnn_mode: Global CNN mode for RQ metric if used in callbacks
            global_cnn_rq_aggregation_op: Global CNN RQ aggregation operation if used in callbacks
        """
        if not isinstance(metric_configs, list) or not all(isinstance(mc, dict) and "name" in mc for mc in metric_configs):
            raise ValueError("metric_configs must be a list of dictionaries, each with at least a 'name' key.")
        if not metric_configs:
            raise ValueError("metric_configs cannot be empty for AlignmentMetricTracker.")
            
        self.metric_configs = metric_configs
        self.data_loader = data_loader
        self.device = device
        self.num_batches = num_batches
        self.experiment_config = experiment_config
        self.metrics_evolution: List[Dict[str, Any]] = [] # Stores history: {epoch: X, all_scores_per_layer: {layer: {metric: scores}}} 
        self.global_cnn_mode = global_cnn_mode
        self.global_cnn_rq_aggregation_op = global_cnn_rq_aggregation_op

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
                    f"Epoch {current_epoch}, Metric '{self.metric_configs[0]['name']}': Model is not an AlignmentNetwork "
                    f"and lacks .alignment_layers/.alignment_names. Skipping alignment metric computation."
                )
                return
            else:
                logger.info(
                    f"Epoch {current_epoch}, Metric '{self.metric_configs[0]['name']}': Model is not an AlignmentNetwork, "
                    f"but required attributes .alignment_layers/.alignment_names found. Proceeding."
                )


        debug_mode_callback = False
        if self.experiment_config and hasattr(self.experiment_config, 'debug_mode'):
            debug_mode_callback = self.experiment_config.debug_mode
        
        metric_names_str = ", ".join([mc['name'] for mc in self.metric_configs])
        logger.info(f"Epoch {current_epoch}: Computing alignment metrics ({metric_names_str})...")
        
        original_training_state = model.training
        try:
            model.eval() # Set to eval mode for consistent activation collection
            
            all_scores_data = compute_all_node_scores(
                model=model,
                metric_configs=self.metric_configs, 
                device=self.device,
                data_loader=self.data_loader,
                num_batches=self.num_batches,
                debug_mode=debug_mode_callback,
                configured_cnn_mode=self.global_cnn_mode,
                configured_cnn_rq_op=self.global_cnn_rq_aggregation_op
            )
            
            self.metrics_evolution.append({
                "epoch": current_epoch,
                "all_scores_per_layer": all_scores_data 
            })
            logger.info(f"Epoch {current_epoch}: Finished computing metrics ({metric_names_str}).")
            
        except Exception as e:
            logger.error(
                f"Epoch {current_epoch}: Error computing alignment metrics ({metric_names_str}): {e}", 
                exc_info=debug_mode_callback
            )
        finally:
            if original_training_state:
                model.train() # Restore original training state 