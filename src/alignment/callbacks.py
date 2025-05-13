"""
Callbacks for use during training.
"""
import logging
from typing import Dict, List, Any, Optional, Union

import torch
from torch.utils.data import DataLoader

# Updated imports for refactored metrics and data collection
# from alignment.metrics import get_metric, compute_all_node_scores # OLD
from alignment.metrics import compute_metrics_for_layers # NEW
from alignment.utils.activation_utils import collect_layer_data # NEW
from alignment.models import AlignmentNetwork # Assuming AlignmentNetwork is in alignment.models
from alignment.config import BaseConfig # For accessing global config settings

logger = logging.getLogger(__name__)

class AlignmentMetricTracker:
    """
    A callback to compute and store specified alignment metrics at the end of each epoch
    using the refactored data collection and metric computation pipeline.
    """
    def __init__(self,
                 metric_names: List[str], # Now just takes a list of names
                 data_loader: DataLoader,
                 device: Union[str, torch.device],
                 num_batches: Optional[int] = 5,
                 target_layers: Optional[List[str]] = None, # Optional: specify layers, otherwise use model default
                 metric_kwargs: Optional[Dict[str, Dict[str, Any]]] = None, # Optional: kwargs for specific metrics
                 experiment_config: Optional[BaseConfig] = None):
        """
        Args:
            metric_names: List of metric names to compute (must match keys in metrics.ALIGNMENT_METRICS_REGISTRY).
            data_loader: DataLoader to use for generating activations for metric computation.
            device: The torch.device to run computations on.
            num_batches: Number of batches from data_loader to use for collecting activations.
            target_layers: Optional list of layer names to compute metrics for. If None, attempts
                           to get layers from model.alignment_names or default weight-having layers.
            metric_kwargs: Optional dictionary of keyword arguments for specific metrics.
                           Format: {"metric_name": {"kwarg1": value1, ...}}
            experiment_config: Optional experiment configuration object.
        """
        if not isinstance(metric_names, list) or not metric_names:
            raise ValueError("metric_names must be a non-empty list of strings.")

        self.metric_names = metric_names
        self.data_loader = data_loader
        self.device = device
        self.num_batches = num_batches if num_batches is not None and num_batches > 0 else 1 # Ensure at least 1 batch
        self.target_layers = target_layers # Store explicitly, can be None
        self.metric_kwargs = metric_kwargs or {}
        self.experiment_config = experiment_config
        # Stores history: {epoch: X, all_scores_per_layer: {layer: {metric: scores}}}
        self.metrics_evolution: List[Dict[str, Any]] = []

    def _get_effective_target_layers(self, model: torch.nn.Module) -> List[str]:
        """Determine the layers to target for metric computation."""
        if self.target_layers:
            # Validate provided target layers exist in the model
            model_layer_names = {name for name, _ in model.named_modules()}
            valid_targets = [name for name in self.target_layers if name in model_layer_names]
            invalid_targets = [name for name in self.target_layers if name not in model_layer_names]
            if invalid_targets:
                logger.warning(f"Target layers not found in model: {invalid_targets}. Using valid subset: {valid_targets}")
            if not valid_targets:
                 logger.warning("No valid target layers specified or found. Will use default layers with weights.")
                 # Fall through to default logic
            else:
                 return valid_targets

        # If self.target_layers is None or empty after validation, use defaults
        if isinstance(model, AlignmentNetwork) and hasattr(model, 'alignment_names') and model.alignment_names:
            logger.info(f"Using target layers from AlignmentNetwork.alignment_names: {model.alignment_names}")
            return model.alignment_names
        else:
            # Default: Use names of all layers that have a 'weight' attribute
            default_layers = [name for name, module in model.named_modules() if hasattr(module, 'weight') and module.weight is not None]
            logger.info(f"Target layers not specified. Using default layers with weights: {default_layers}")
            return default_layers

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

        # --- DDP Handling ---
        ddp_rank = 0
        ddp_world_size = 1
        if self.experiment_config and hasattr(self.experiment_config, 'use_ddp') and self.experiment_config.use_ddp:
            if hasattr(self.experiment_config, 'ddp_rank'):
                ddp_rank = self.experiment_config.ddp_rank
            if hasattr(self.experiment_config, 'ddp_world_size'):
                ddp_world_size = self.experiment_config.ddp_world_size
        
        is_main_process = (ddp_rank == 0)

        # If DDP is active, only proceed with metric computation on the main process (rank 0)
        if ddp_world_size > 1 and not is_main_process:
            return # Other ranks do nothing for this callback
        # --- End DDP Handling ---

        effective_target_layers = self._get_effective_target_layers(model)
        if not effective_target_layers:
             logger.warning(f"Epoch {current_epoch}: No target layers identified for metric computation. Skipping.")
             return

        # This is a simplification; a more robust way would check metric function signatures
        metric_names_lower = [m.lower() for m in self.metric_names]
        
        final_collect_inputs = any("rayleigh_quotient" in m or "rq" in m or "redundancy" in m or "pid_" in m for m in metric_names_lower)
        final_collect_outputs = any("mi_" in m or "pid_" in m for m in metric_names_lower)

        if not final_collect_inputs and not final_collect_outputs and self.metric_names:
             logger.warning(f"Epoch {current_epoch}: Metric names {self.metric_names} configured for tracker, but no input/output collection explicitly triggered by known patterns. This might be okay if metrics don't need activations.")
        
        # Use the final flags for the debug log AND the function call
        logger.debug(f"Epoch {current_epoch}: Calling collect_layer_data with effective flags: (inputs={final_collect_inputs}, outputs={final_collect_outputs})...")
        
        # Store original training state and set model to eval - this was missing from the snippet
        original_training_state = model.training
        try:
            model.eval() 

            # 1. Collect Data
            # logger.debug(f"Epoch {current_epoch}: Collecting layer data (inputs={collect_inputs_flag}, outputs={collect_outputs_flag})...") # Old log, replaced by the one above
            
            model_to_collect_from = model.module if (ddp_world_size > 1 and isinstance(model, torch.nn.parallel.DistributedDataParallel)) else model
            
            collected_data = collect_layer_data(
                model=model_to_collect_from, 
                dataloader=self.data_loader, 
                target_layers=effective_target_layers,
                num_batches=self.num_batches,
                device=self.device,
                collect_inputs=final_collect_inputs, # Use new flag
                collect_outputs=final_collect_outputs, # Use new flag
                flatten_spatial=True 
            )
            logger.debug(f"Epoch {current_epoch}: Data collection complete.")

            if not collected_data:
                 logger.warning(f"Epoch {current_epoch}: Data collection returned empty. Skipping metric computation.")
                 return

            # 2. Compute Metrics
            logger.debug(f"Epoch {current_epoch}: Computing metrics from collected data...")
            
            # Prepare metric_configs for compute_metrics_for_layers
            # This list of dicts tells compute_metrics_for_layers which metrics to run and with what specific args.
            metric_configs_for_computation: List[Dict[str, Any]] = []
            for name in self.metric_names:
                conf = {"name": name}
                
                # Add metric-specific kwargs passed during tracker initialization
                if name in self.metric_kwargs:
                    conf.update(self.metric_kwargs[name])
                
                # Add global/default settings from experiment_config if not already overridden by metric_kwargs
                # Example: scale_by_norm for RQ, cnn_mode, cnn_rq_aggregation_op, force_cpu_for_large_metric_ops
                if self.experiment_config and hasattr(self.experiment_config, 'alignment_settings'):
                    if hasattr(self.experiment_config.alignment_settings, 'scale_by_norm'):
                        if name.lower() == "rq" or "rayleigh_quotient" in name.lower():
                            logger.debug(f"Setting scale_by_norm={self.experiment_config.alignment_settings.scale_by_norm} for metric {name}")
                            conf.setdefault("scale_by_norm", self.experiment_config.alignment_settings.scale_by_norm)
                    
                    # Add additional parameters that might be needed for metrics computation
                    if hasattr(self.experiment_config.alignment_settings, 'force_cpu_for_large_metric_ops'):
                        conf.setdefault("force_cpu_for_large_metric_ops", 
                                       self.experiment_config.alignment_settings.force_cpu_for_large_metric_ops)
                    if hasattr(self.experiment_config.alignment_settings, 'cnn_mode'):
                        conf.setdefault("configured_cnn_mode", self.experiment_config.alignment_settings.cnn_mode)
                    if hasattr(self.experiment_config.alignment_settings, 'cnn_rq_aggregation_op'):
                        conf.setdefault("configured_cnn_rq_op", self.experiment_config.alignment_settings.cnn_rq_aggregation_op)
                
                # Make sure verbose is set for debug
                conf.setdefault("verbose", True if self.experiment_config and hasattr(self.experiment_config, 'debug_mode') and self.experiment_config.debug_mode else False)
                
                metric_configs_for_computation.append(conf)
            
            logger.debug(f"Epoch {current_epoch}: Prepared metric_configs_for_computation: {metric_configs_for_computation}")

            computed_metrics = compute_metrics_for_layers(
                model=model_to_collect_from, # Pass the same model instance used for collection 
                collected_data=collected_data,
                metric_configs=metric_configs_for_computation, # Pass the constructed list of dicts
                device=self.device # Device for model/weights and where metrics are computed
                # metric_kwargs is now handled by preparing metric_configs_for_computation
            )
            logger.debug(f"Epoch {current_epoch}: Metric computation complete.")

            # 3. Store Results
            self.metrics_evolution.append({
                "epoch": current_epoch,
                "all_scores_per_layer": computed_metrics # The structure matches the old format
            })
            logger.info(f"Epoch {current_epoch}: Finished computing and storing metrics ({', '.join(self.metric_names)})")

        except Exception as e:
            debug_mode = getattr(self.experiment_config, 'debug_mode', False) if self.experiment_config else False
            logger.error(
                f"Epoch {current_epoch}: Error computing alignment metrics ({', '.join(self.metric_names)}): {e}",
                exc_info=debug_mode
            )
        finally:
            if original_training_state:
                model.train() # Restore original training state 