"""
Central registry for managing all framework components.

This module provides a unified registration system for metrics, models,
datasets, and experiments, making them easily discoverable and instantiable.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Registry:
    """Generic registry for framework components."""

    def __init__(self, name: str):
        """
        Initialize a registry.

        Args:
            name: Name of the registry (e.g., "metrics", "models")
        """
        self.name = name
        self._registry: Dict[str, Type[Any]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, cls: Optional[Type[T]] = None, **metadata: Any) -> Union[Callable[[Type[T]], Type[T]], Type[T]]:
        """
        Register a class in the registry.

        Can be used as a decorator or called directly.

        Args:
            name: Name to register the class under
            cls: Class to register (if not using as decorator)
            **metadata: Additional metadata to store with the registration

        Returns:
            Registered class or decorator function
        """

        def decorator(cls_to_register: Type[T]) -> Type[T]:
            if name in self._registry:
                logger.warning(f"Overwriting existing registration '{name}' in {self.name} registry")
            self._registry[name] = cls_to_register
            self._metadata[name] = metadata

            # Add registry info to the class
            setattr(cls_to_register, "_registry_name", name)
            setattr(cls_to_register, "_registry", self.name)

            logger.debug(f"Registered '{name}' in {self.name} registry")
            return cls_to_register

        if cls is None:
            # Used as decorator
            return decorator
        else:
            # Direct registration
            return decorator(cls)

    def get(self, name: str) -> Type[Any]:
        """
        Get a registered class by name.

        Args:
            name: Name of the registered class

        Returns:
            The registered class

        Raises:
            KeyError: If name is not registered
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(f"'{name}' not found in {self.name} registry. " f"Available: {available}")
        return self._registry[name]

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a registered class."""
        return self._metadata.get(name, {})

    def list(self) -> List[str]:
        """List all registered names."""
        return list(self._registry.keys())

    def create(self, name: str, **kwargs: Any) -> Any:
        """
        Create an instance of a registered class.

        Args:
            name: Name of the registered class
            **kwargs: Arguments to pass to the class constructor

        Returns:
            Instance of the registered class
        """
        cls = self.get(name)
        return cls(**kwargs)

    def __contains__(self, name: str) -> bool:
        """Check if a name is registered."""
        return name in self._registry

    def __len__(self) -> int:
        """Get number of registered items."""
        return len(self._registry)


# Create global registries
METRIC_REGISTRY = Registry("metrics")
MODEL_REGISTRY = Registry("models")
DATASET_REGISTRY = Registry("datasets")
EXPERIMENT_REGISTRY = Registry("experiments")
AGGREGATOR_REGISTRY = Registry("aggregators")
REPORTER_REGISTRY = Registry("reporters")


# Decorator functions for registration
def register_metric(name: str, **metadata: Any) -> Callable:
    """Register a metric class."""
    return METRIC_REGISTRY.register(name, **metadata)


def register_model(name: str, **metadata: Any) -> Callable:
    """Register a model class."""
    return MODEL_REGISTRY.register(name, **metadata)


def register_dataset(name: str, **metadata: Any) -> Callable:
    """Register a dataset class."""
    return DATASET_REGISTRY.register(name, **metadata)


def register_experiment(name: str, **metadata: Any) -> Callable:
    """Register an experiment class."""
    return EXPERIMENT_REGISTRY.register(name, **metadata)


def register_aggregator(name: str, **metadata: Any) -> Callable:
    """Register an aggregator class."""
    return AGGREGATOR_REGISTRY.register(name, **metadata)


def register_reporter(name: str, **metadata: Any) -> Callable:
    """Register a reporter class."""
    return REPORTER_REGISTRY.register(name, **metadata)


# Getter functions
def get_metric(name: str, **kwargs: Any) -> Any:
    """Get a metric instance by name."""
    return METRIC_REGISTRY.create(name, **kwargs)


def get_model(name: str, **kwargs: Any) -> Any:
    """Get a model instance by name."""
    return MODEL_REGISTRY.create(name, **kwargs)


def get_dataset(name: str, **kwargs: Any) -> Any:
    """Get a dataset instance by name."""
    return DATASET_REGISTRY.create(name, **kwargs)


def get_experiment(name: str, **kwargs: Any) -> Any:
    """Get an experiment instance by name."""
    return EXPERIMENT_REGISTRY.create(name, **kwargs)


def get_aggregator(name: str, **kwargs: Any) -> Any:
    """Get an aggregator instance by name."""
    return AGGREGATOR_REGISTRY.create(name, **kwargs)


def get_reporter(name: str, **kwargs: Any) -> Any:
    """Get a reporter instance by name."""
    return REPORTER_REGISTRY.create(name, **kwargs)


# Auto-discovery function
def discover_and_register(module_path: str, registry_type: str = "all") -> None:
    """
    Auto-discover and register components from a module.

    Args:
        module_path: Python module path to scan
        registry_type: Type of components to register ("all", "metrics", etc.)
    """
    import importlib
    import pkgutil

    try:
        module = importlib.import_module(module_path)

        # Recursively walk through submodules
        for importer, modname, ispkg in pkgutil.walk_packages(path=module.__path__, prefix=module.__name__ + ".", onerror=lambda x: None):
            try:
                importlib.import_module(modname)
                logger.debug(f"Imported module: {modname}")
            except Exception as e:
                logger.warning(f"Failed to import {modname}: {e}")

    except Exception as e:
        logger.error(f"Failed to discover components from {module_path}: {e}")
