"""
Hook management utilities for activation capture.

This module provides lifecycle-managed hooks that automatically clean up
to prevent memory leaks.
"""

import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class HookManager:
    """
    Manage forward hooks with automatic cleanup.

    This class ensures that all registered hooks are properly removed,
    preventing memory leaks and stale hook accumulation.

    Example:
        >>> hook_mgr = HookManager()
        >>> with hook_mgr.temporary_hooks(model, ['layer1', 'layer2']) as cache:
        ...     output = model(input)
        ...     activations = cache  # Dict[str, Tensor]
        # Hooks automatically removed after context
    """

    def __init__(self):
        """Initialize hook manager."""
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.cache: Dict[str, torch.Tensor] = {}

    def register_forward_hook(self, module: nn.Module, hook_fn: Callable, name: Optional[str] = None) -> torch.utils.hooks.RemovableHandle:
        """
        Register a forward hook and track it for cleanup.

        Args:
            module: Module to attach hook to
            hook_fn: Hook function with signature (module, input, output)
            name: Optional name for logging

        Returns:
            RemovableHandle for the registered hook
        """
        handle = module.register_forward_hook(hook_fn)
        self.hooks.append(handle)

        if name:
            logger.debug(f"Registered hook on {name}")

        return handle

    def cleanup(self):
        """Remove all registered hooks and clear cache."""
        num_hooks = len(self.hooks)
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.cache.clear()

        if num_hooks > 0:
            logger.debug(f"Cleaned up {num_hooks} hooks")

    @contextmanager
    def temporary_hooks(self, model: nn.Module, layer_names: List[str], track_inputs: bool = True, track_outputs: bool = True):
        """
        Context manager for temporary hooks that auto-cleanup.

        Args:
            model: PyTorch model
            layer_names: Names of layers to hook
            track_inputs: Whether to capture layer inputs
            track_outputs: Whether to capture layer outputs

        Yields:
            Dict mapping layer names to captured tensors

        Example:
            >>> with hook_mgr.temporary_hooks(model, ['conv1', 'fc1']) as cache:
            ...     output = model(input_batch)
            ...     conv1_acts = cache['conv1_output']
            ...     fc1_acts = cache['fc1_input']
        """
        try:
            # Register hooks for specified layers
            for name, module in model.named_modules():
                if name in layer_names:
                    # Create closure to capture name
                    def make_hook(layer_name):
                        def hook(mod, inp, out):
                            if track_inputs and inp is not None:
                                # Handle tuple inputs
                                input_tensor = inp[0] if isinstance(inp, tuple) else inp
                                self.cache[f"{layer_name}_input"] = input_tensor.detach()

                            if track_outputs and out is not None:
                                # Handle tuple outputs
                                output_tensor = out[0] if isinstance(out, tuple) else out
                                self.cache[f"{layer_name}_output"] = output_tensor.detach()

                        return hook

                    self.register_forward_hook(module, make_hook(name), name=name)

            yield self.cache

        finally:
            # Always cleanup, even if exception occurs
            self.cleanup()

    def __del__(self):
        """Ensure cleanup on object destruction."""
        if hasattr(self, "hooks") and len(self.hooks) > 0:
            logger.warning(
                f"HookManager destroyed with {len(self.hooks)} hooks still registered. "
                "Consider using cleanup() explicitly or temporary_hooks() context manager."
            )
            self.cleanup()

    def __enter__(self):
        """Support using HookManager itself as context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit."""
        self.cleanup()
        return False  # Don't suppress exceptions


class PersistentHookManager(HookManager):
    """
    Hook manager that maintains hooks across multiple forward passes.

    Unlike temporary hooks, these persist until explicitly cleaned up.
    Useful for experiment-long activation tracking.
    """

    def __init__(self, auto_clear_cache: bool = True):
        """
        Initialize persistent hook manager.

        Args:
            auto_clear_cache: Whether to clear cache after each forward pass
        """
        super().__init__()
        self.auto_clear_cache = auto_clear_cache
        self._forward_count = 0

    def register_persistent_hooks(self, model: nn.Module, layer_names: List[str], track_inputs: bool = True, track_outputs: bool = True):
        """
        Register hooks that persist across forward passes.

        Args:
            model: PyTorch model
            layer_names: Names of layers to hook
            track_inputs: Whether to capture layer inputs
            track_outputs: Whether to capture layer outputs
        """
        for name, module in model.named_modules():
            if name in layer_names:

                def make_hook(layer_name):
                    def hook(mod, inp, out):
                        # Clear cache on new forward pass if enabled
                        if self.auto_clear_cache and f"{layer_name}_count" not in self.cache:
                            self.cache.clear()

                        # Initialize count if not present
                        if f"{layer_name}_count" not in self.cache:
                            self.cache[f"{layer_name}_count"] = 0

                        if track_inputs and inp is not None:
                            input_tensor = inp[0] if isinstance(inp, tuple) else inp
                            self.cache[f"{layer_name}_input"] = input_tensor.detach()

                        if track_outputs and out is not None:
                            output_tensor = out[0] if isinstance(out, tuple) else out
                            self.cache[f"{layer_name}_output"] = output_tensor.detach()

                        self.cache[f"{layer_name}_count"] += 1

                    return hook

                self.register_forward_hook(module, make_hook(name), name=name)

        logger.info(f"Registered {len(self.hooks)} persistent hooks")

    def get_cached_activations(self, clear_after: bool = False) -> Dict[str, torch.Tensor]:
        """
        Get currently cached activations.

        Args:
            clear_after: Whether to clear cache after retrieving

        Returns:
            Dictionary of cached activations
        """
        # Filter out internal tracking keys
        activations = {k: v for k, v in self.cache.items() if not k.endswith("_count")}

        if clear_after:
            self.cache.clear()

        return activations
