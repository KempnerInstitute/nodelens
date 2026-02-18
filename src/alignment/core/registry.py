"""
Central registry for managing all framework components.

This module provides a unified registration system for metrics, models,
datasets, experiments, analyzers, visualizers, and pruning strategies,
making them easily discoverable and instantiable.

The registry system enables:
1. Plugin-based architecture - add new components without modifying core code
2. Configuration-driven instantiation - create components from config files
3. Auto-discovery - automatically find and register components from packages
4. Metadata tracking - store component capabilities, requirements, and documentation

Usage:
    # Register a new metric
    @register_metric("my_custom_metric", category="information", requires_pairs=True)
    class MyCustomMetric(BaseMetric):
        ...

    # Use the metric
    metric = get_metric("my_custom_metric", param1=value1)

    # Or from config
    metric = create_from_config({"name": "my_custom_metric", "param1": value1})
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ComponentInfo:
    """Metadata about a registered component."""

    name: str
    cls: Type[Any]
    category: str = "default"
    description: str = ""
    requires: List[str] = field(default_factory=list)  # Dependencies
    provides: List[str] = field(default_factory=list)  # What it provides
    config_schema: Optional[Dict[str, Any]] = None  # Expected config params
    tags: Set[str] = field(default_factory=set)  # Searchable tags
    version: str = "1.0.0"
    author: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class Registry:
    """
    Generic registry for framework components with enhanced metadata and discovery.

    Features:
    - Category-based organization
    - Tag-based search
    - Dependency tracking
    - Config schema validation
    - Plugin discovery
    """

    def __init__(self, name: str, base_protocol: Optional[Type] = None):
        """
        Initialize a registry.

        Args:
            name: Name of the registry (e.g., "metrics", "models")
            base_protocol: Optional protocol/base class that all registered items should follow
        """
        self.name = name
        self.base_protocol = base_protocol
        self._registry: Dict[str, ComponentInfo] = {}
        self._by_category: Dict[str, List[str]] = {}
        self._by_tag: Dict[str, Set[str]] = {}
        self._aliases: Dict[str, str] = {}  # alias -> canonical name

    def register(
        self,
        name: str,
        cls: Optional[Type[T]] = None,
        category: str = "default",
        description: str = "",
        requires: Optional[List[str]] = None,
        provides: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
        config_schema: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> Union[Callable[[Type[T]], Type[T]], Type[T]]:
        """
        Register a class in the registry with comprehensive metadata.

        Can be used as a decorator or called directly.

        Args:
            name: Canonical name to register the class under
            cls: Class to register (if not using as decorator)
            category: Category for organization (e.g., "information", "gradient", "magnitude")
            description: Human-readable description
            requires: List of dependencies (other registered components)
            provides: List of capabilities this component provides
            tags: Searchable tags for discovery
            aliases: Alternative names that map to this component
            config_schema: Schema for expected configuration parameters
            **extra: Additional metadata

        Returns:
            Registered class or decorator function

        Example:
            @registry.register(
                "rayleigh_quotient",
                category="alignment",
                description="Measures alignment between weights and activation covariance",
                requires=["activations", "weights"],
                provides=["per_neuron_score"],
                tags=["alignment", "covariance", "rq"],
                aliases=["rq", "rayleigh", "alignment_score"]
            )
            class RayleighQuotient:
                ...
        """
        requires = requires or []
        provides = provides or []
        tags = set(tags) if tags else set()
        aliases = aliases or []

        def decorator(cls_to_register: Type[T]) -> Type[T]:
            if name in self._registry:
                logger.warning(f"Overwriting existing registration '{name}' in {self.name} registry")

            # Create component info
            info = ComponentInfo(
                name=name,
                cls=cls_to_register,
                category=category,
                description=description or cls_to_register.__doc__ or "",
                requires=requires,
                provides=provides,
                config_schema=config_schema,
                tags=tags,
                extra=extra,
            )

            self._registry[name] = info

            # Index by category
            if category not in self._by_category:
                self._by_category[category] = []
            if name not in self._by_category[category]:
                self._by_category[category].append(name)

            # Index by tags
            for tag in tags:
                if tag not in self._by_tag:
                    self._by_tag[tag] = set()
                self._by_tag[tag].add(name)

            # Register aliases
            for alias in aliases:
                self._aliases[alias] = name
                # Also add aliases to tag index
                if alias not in self._by_tag:
                    self._by_tag[alias] = set()
                self._by_tag[alias].add(name)

            # Add registry info to the class
            setattr(cls_to_register, "_registry_name", name)
            setattr(cls_to_register, "_registry", self.name)
            setattr(cls_to_register, "_registry_info", info)

            logger.debug(f"Registered '{name}' in {self.name} registry (category={category})")
            return cls_to_register

        if cls is None:
            return decorator
        else:
            return decorator(cls)

    def resolve_name(self, name: str) -> str:
        """Resolve an alias to its canonical name."""
        return self._aliases.get(name, name)

    def get(self, name: str) -> Type[Any]:
        """
        Get a registered class by name (supports aliases).

        Args:
            name: Name or alias of the registered class

        Returns:
            The registered class

        Raises:
            KeyError: If name is not registered
        """
        canonical = self.resolve_name(name)
        if canonical not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(f"'{name}' not found in {self.name} registry. Available: {available}")
        return self._registry[canonical].cls

    def get_info(self, name: str) -> ComponentInfo:
        """Get full component info for a registered class."""
        canonical = self.resolve_name(name)
        if canonical not in self._registry:
            raise KeyError(f"'{name}' not found in {self.name} registry")
        return self._registry[canonical]

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a registered class (legacy compatibility)."""
        info = self.get_info(name)
        return {
            "category": info.category,
            "description": info.description,
            "requires": info.requires,
            "provides": info.provides,
            "tags": list(info.tags),
            **info.extra,
        }

    def list(self, category: Optional[str] = None) -> List[str]:
        """
        List registered names, optionally filtered by category.

        Args:
            category: Optional category to filter by

        Returns:
            List of registered names
        """
        if category:
            return self._by_category.get(category, [])
        return list(self._registry.keys())

    def list_categories(self) -> List[str]:
        """List all categories."""
        return list(self._by_category.keys())

    def search(self, query: str = "", tags: Optional[List[str]] = None) -> List[str]:
        """
        Search for components by name substring or tags.

        Args:
            query: Substring to search in names and descriptions
            tags: Tags to filter by (AND logic)

        Returns:
            List of matching component names
        """
        results = set(self._registry.keys())

        # Filter by query
        if query:
            query_lower = query.lower()
            results = {
                name
                for name, info in self._registry.items()
                if query_lower in name.lower() or query_lower in info.description.lower() or any(query_lower in tag for tag in info.tags)
            }

        # Filter by tags (AND logic)
        if tags:
            for tag in tags:
                tag_matches = self._by_tag.get(tag, set())
                results = results & tag_matches

        return list(results)

    def create(self, name: str, **kwargs: Any) -> Any:
        """
        Create an instance of a registered class.

        Args:
            name: Name or alias of the registered class
            **kwargs: Arguments to pass to the class constructor

        Returns:
            Instance of the registered class
        """
        cls = self.get(name)
        return cls(**kwargs)

    def create_from_config(self, config: Dict[str, Any]) -> Any:
        """
        Create an instance from a configuration dictionary.

        The config should have a 'name' or 'type' key specifying the component,
        and other keys are passed as constructor arguments.

        Args:
            config: Configuration dictionary

        Returns:
            Instance of the registered class
        """
        config = dict(config)  # Copy to avoid mutation
        name = config.pop("name", None) or config.pop("type", None)
        if not name:
            raise ValueError("Config must have 'name' or 'type' key")
        return self.create(name, **config)

    def __contains__(self, name: str) -> bool:
        """Check if a name or alias is registered."""
        canonical = self.resolve_name(name)
        return canonical in self._registry

    def __len__(self) -> int:
        """Get number of registered items."""
        return len(self._registry)

    def __iter__(self):
        """Iterate over registered names."""
        return iter(self._registry.keys())

    def summary(self) -> str:
        """Get a human-readable summary of the registry."""
        lines = [f"\n=== {self.name.upper()} REGISTRY ==="]
        lines.append(f"Total: {len(self)} components in {len(self._by_category)} categories\n")

        for category, names in sorted(self._by_category.items()):
            lines.append(f"  {category}:")
            for name in sorted(names):
                info = self._registry[name]
                desc = info.description[:50] + "..." if len(info.description) > 50 else info.description
                lines.append(f"    - {name}: {desc}")

        return "\n".join(lines)


# =============================================================================
# GLOBAL REGISTRIES
# =============================================================================
# These registries are the central point for all framework components.
# Components register themselves using decorators, and are instantiated
# via configuration or direct API calls.

METRIC_REGISTRY = Registry("metrics")
MODEL_REGISTRY = Registry("models")
DATASET_REGISTRY = Registry("datasets")
EXPERIMENT_REGISTRY = Registry("experiments")
AGGREGATOR_REGISTRY = Registry("aggregators")
REPORTER_REGISTRY = Registry("reporters")

# New registries for enhanced modularity
ANALYZER_REGISTRY = Registry("analyzers")  # Analysis pipelines (clustering, halo, etc.)
VISUALIZER_REGISTRY = Registry("visualizers")  # Visualization components
PRUNER_REGISTRY = Registry("pruners")  # Pruning strategies
EVALUATOR_REGISTRY = Registry("evaluators")  # Evaluation methods (accuracy, perplexity, etc.)
PREPROCESSOR_REGISTRY = Registry("preprocessors")  # Data/activation preprocessors

# Collect all registries for easy iteration
ALL_REGISTRIES = {
    "metrics": METRIC_REGISTRY,
    "models": MODEL_REGISTRY,
    "datasets": DATASET_REGISTRY,
    "experiments": EXPERIMENT_REGISTRY,
    "aggregators": AGGREGATOR_REGISTRY,
    "reporters": REPORTER_REGISTRY,
    "analyzers": ANALYZER_REGISTRY,
    "visualizers": VISUALIZER_REGISTRY,
    "pruners": PRUNER_REGISTRY,
    "evaluators": EVALUATOR_REGISTRY,
    "preprocessors": PREPROCESSOR_REGISTRY,
}


# =============================================================================
# DECORATOR FUNCTIONS FOR REGISTRATION
# =============================================================================


def register_metric(name: str, **metadata: Any) -> Callable:
    """
    Register a metric class.

    Example:
        @register_metric("my_metric", category="information", tags=["mi", "entropy"])
        class MyMetric(BaseMetric):
            ...
    """
    return METRIC_REGISTRY.register(name, **metadata)


def register_model(name: str, **metadata: Any) -> Callable:
    """Register a model/architecture class."""
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


def register_analyzer(name: str, **metadata: Any) -> Callable:
    """
    Register an analyzer class (clustering, halo analysis, etc.).

    Example:
        @register_analyzer("kmeans_clustering", category="clustering")
        class KMeansClustering(BaseAnalyzer):
            ...
    """
    return ANALYZER_REGISTRY.register(name, **metadata)


def register_visualizer(name: str, **metadata: Any) -> Callable:
    """Register a visualizer class."""
    return VISUALIZER_REGISTRY.register(name, **metadata)


def register_pruner(name: str, **metadata: Any) -> Callable:
    """
    Register a pruning strategy.

    Example:
        @register_pruner("magnitude", category="baseline", tags=["simple", "weight-based"])
        class MagnitudePruning(BasePruner):
            ...
    """
    return PRUNER_REGISTRY.register(name, **metadata)


def register_evaluator(name: str, **metadata: Any) -> Callable:
    """Register an evaluator class."""
    return EVALUATOR_REGISTRY.register(name, **metadata)


def register_preprocessor(name: str, **metadata: Any) -> Callable:
    """Register a preprocessor class."""
    return PREPROCESSOR_REGISTRY.register(name, **metadata)


# =============================================================================
# GETTER FUNCTIONS
# =============================================================================


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


def get_analyzer(name: str, **kwargs: Any) -> Any:
    """Get an analyzer instance by name."""
    return ANALYZER_REGISTRY.create(name, **kwargs)


def get_visualizer(name: str, **kwargs: Any) -> Any:
    """Get a visualizer instance by name."""
    return VISUALIZER_REGISTRY.create(name, **kwargs)


def get_pruner(name: str, **kwargs: Any) -> Any:
    """Get a pruner instance by name."""
    return PRUNER_REGISTRY.create(name, **kwargs)


def get_evaluator(name: str, **kwargs: Any) -> Any:
    """Get an evaluator instance by name."""
    return EVALUATOR_REGISTRY.create(name, **kwargs)


def get_preprocessor(name: str, **kwargs: Any) -> Any:
    """Get a preprocessor instance by name."""
    return PREPROCESSOR_REGISTRY.create(name, **kwargs)


# =============================================================================
# UNIFIED COMPONENT FACTORY
# =============================================================================


def create_component(registry_name: str, component_name: str, **kwargs: Any) -> Any:
    """
    Create a component from any registry by name.

    Args:
        registry_name: Name of the registry ("metrics", "pruners", etc.)
        component_name: Name of the component within that registry
        **kwargs: Arguments to pass to the component constructor

    Returns:
        Instance of the component

    Example:
        metric = create_component("metrics", "rayleigh_quotient", relative=True)
        pruner = create_component("pruners", "magnitude", amount=0.5)
    """
    if registry_name not in ALL_REGISTRIES:
        raise KeyError(f"Unknown registry: {registry_name}. Available: {list(ALL_REGISTRIES.keys())}")
    return ALL_REGISTRIES[registry_name].create(component_name, **kwargs)


def create_from_config(config: Dict[str, Any], registry_name: Optional[str] = None) -> Any:
    """
    Create a component from a configuration dictionary.

    The config should have 'registry' and 'name' keys, or just 'name' if registry_name is provided.

    Args:
        config: Configuration dictionary with at least 'name' key
        registry_name: Optional registry name (if not in config)

    Returns:
        Instance of the component

    Example:
        config = {"registry": "metrics", "name": "rayleigh_quotient", "relative": True}
        metric = create_from_config(config)

        # Or with explicit registry
        config = {"name": "magnitude", "amount": 0.5}
        pruner = create_from_config(config, registry_name="pruners")
    """
    config = dict(config)
    reg_name = config.pop("registry", None) or registry_name
    if not reg_name:
        raise ValueError("Config must have 'registry' key or registry_name must be provided")

    return ALL_REGISTRIES[reg_name].create_from_config(config)


def list_all_components() -> Dict[str, List[str]]:
    """List all registered components across all registries."""
    return {name: reg.list() for name, reg in ALL_REGISTRIES.items()}


def print_registry_summary(registry_name: Optional[str] = None) -> None:
    """Print a summary of registered components."""
    if registry_name:
        if registry_name in ALL_REGISTRIES:
            print(ALL_REGISTRIES[registry_name].summary())
        else:
            print(f"Unknown registry: {registry_name}")
    else:
        for name, registry in ALL_REGISTRIES.items():
            if len(registry) > 0:
                print(registry.summary())


# =============================================================================
# AUTO-DISCOVERY
# =============================================================================


def discover_and_register(module_path: str, registry_type: str = "all") -> int:
    """
    Auto-discover and register components from a module.

    Components are discovered by importing modules, which triggers
    the @register_* decorators.

    Args:
        module_path: Python module path to scan (e.g., "alignment.metrics")
        registry_type: Type of components to register ("all", "metrics", etc.)

    Returns:
        Number of modules imported
    """
    import importlib
    import pkgutil

    count = 0
    try:
        module = importlib.import_module(module_path)

        # Recursively walk through submodules
        for importer, modname, ispkg in pkgutil.walk_packages(path=module.__path__, prefix=module.__name__ + ".", onerror=lambda x: None):
            try:
                importlib.import_module(modname)
                count += 1
                logger.debug(f"Imported module: {modname}")
            except Exception as e:
                logger.warning(f"Failed to import {modname}: {e}")

    except Exception as e:
        logger.error(f"Failed to discover components from {module_path}: {e}")

    return count


def discover_plugins(plugin_dirs: Optional[List[str]] = None) -> int:
    """
    Discover and load plugins from specified directories.

    Plugins are Python files or packages that register components using
    the @register_* decorators. This allows users to extend the framework
    without modifying core code.

    Args:
        plugin_dirs: List of directories to search for plugins.
                    Defaults to ["./plugins", "~/.alignment/plugins"]

    Returns:
        Number of plugins loaded
    """
    import importlib.util
    import sys

    if plugin_dirs is None:
        plugin_dirs = [
            "./plugins",
            os.path.expanduser("~/.alignment/plugins"),
        ]

    count = 0
    for plugin_dir in plugin_dirs:
        plugin_path = Path(plugin_dir)
        if not plugin_path.exists():
            continue

        logger.info(f"Scanning for plugins in: {plugin_path}")

        # Find all Python files
        for py_file in plugin_path.glob("**/*.py"):
            if py_file.name.startswith("_"):
                continue

            module_name = f"alignment_plugin_{py_file.stem}"

            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    count += 1
                    logger.info(f"Loaded plugin: {py_file}")
            except Exception as e:
                logger.warning(f"Failed to load plugin {py_file}: {e}")

    return count


# =============================================================================
# INITIALIZATION
# =============================================================================

_initialized = False


def initialize_registries(discover_builtin: bool = True, discover_plugins_flag: bool = True) -> None:
    """
    Initialize all registries by discovering and registering built-in components.

    This should be called once at application startup.

    Args:
        discover_builtin: Whether to discover built-in components
        discover_plugins_flag: Whether to discover user plugins
    """
    global _initialized
    if _initialized:
        return

    if discover_builtin:
        # Discover built-in components from alignment package
        builtin_modules = [
            "alignment.metrics",
            "alignment.pruning.strategies",
            "alignment.analysis",
            "alignment.models",
        ]
        for module in builtin_modules:
            try:
                discover_and_register(module)
            except Exception as e:
                logger.debug(f"Could not discover from {module}: {e}")

    if discover_plugins_flag:
        discover_plugins()

    _initialized = True
    logger.info(f"Registries initialized: {sum(len(r) for r in ALL_REGISTRIES.values())} components")
