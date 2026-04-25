"""
Experiment runner for managing and executing experiments.
"""

import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from nodelens.core.registry import get_experiment
from nodelens.experiments.base import BaseExperiment, ExperimentConfig

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Runner for managing multiple experiments.

    This class provides:
    - Sequential and parallel experiment execution
    - Result aggregation
    - Progress tracking
    - Error handling
    """

    def __init__(
        self,
        base_config: Optional[ExperimentConfig] = None,
        results_dir: str = "./results",
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize experiment runner.

        Args:
            base_config: Base configuration to use for all experiments
            results_dir: Directory to save results
            parallel: Whether to run experiments in parallel
            max_workers: Maximum number of parallel workers
        """
        self.base_config = base_config or ExperimentConfig(name="default")
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.parallel = parallel
        self.max_workers = max_workers or mp.cpu_count()

        self.experiments = []
        self.results = {}

    def add_experiment(
        self, experiment_class: Union[str, Type[BaseExperiment]], config_overrides: Optional[Dict[str, Any]] = None, name_suffix: Optional[str] = None
    ):
        """
        Add an experiment to run.

        Args:
            experiment_class: Experiment class or registered name
            config_overrides: Configuration overrides
            name_suffix: Suffix to add to experiment name
        """
        # Create config
        config = ExperimentConfig(**self.base_config.to_dict())
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(config, key, value)

        # Update name if suffix provided
        if name_suffix:
            config.name = f"{config.name}_{name_suffix}"

        # Get experiment class
        if isinstance(experiment_class, str):
            experiment_class = get_experiment(experiment_class)

        self.experiments.append((experiment_class, config))
        logger.info(f"Added experiment: {config.name}")

    def add_grid_search(self, experiment_class: Union[str, Type[BaseExperiment]], param_grid: Dict[str, List[Any]]):
        """
        Add experiments for grid search over parameters.

        Args:
            experiment_class: Experiment class or registered name
            param_grid: Dictionary mapping parameter names to lists of values
        """
        # Generate all combinations
        import itertools

        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]

        for i, values in enumerate(itertools.product(*param_values)):
            # Create config overrides
            overrides = dict(zip(param_names, values))

            # Create descriptive suffix
            suffix_parts = []
            for name, value in overrides.items():
                if isinstance(value, float):
                    suffix_parts.append(f"{name}_{value:.4f}")
                else:
                    suffix_parts.append(f"{name}_{value}")
            suffix = "_".join(suffix_parts)

            # Add experiment
            self.add_experiment(experiment_class, config_overrides=overrides, name_suffix=suffix)

        logger.info(f"Added {len(list(itertools.product(*param_values)))} grid search experiments")

    def run_experiment(self, experiment_class: Type[BaseExperiment], config: ExperimentConfig) -> Dict[str, Any]:
        """
        Run a single experiment.

        Args:
            experiment_class: Experiment class
            config: Experiment configuration

        Returns:
            Experiment results
        """
        logger.info(f"Starting experiment: {config.name}")
        start_time = datetime.now()

        try:
            # Create and run experiment
            experiment = experiment_class(config)
            results = experiment.run()

            # Add metadata
            results["_metadata"] = {"success": True, "duration": (datetime.now() - start_time).total_seconds(), "error": None}

            # Save individual results
            self._save_experiment_results(config.name, results)

            logger.info(f"Completed experiment: {config.name}")

        except Exception as e:
            logger.error(f"Failed experiment {config.name}: {e}")
            results = {"_metadata": {"success": False, "duration": (datetime.now() - start_time).total_seconds(), "error": str(e)}}

        return results

    def run_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all experiments.

        Returns:
            Dictionary mapping experiment names to results
        """
        logger.info(f"Running {len(self.experiments)} experiments")

        if self.parallel:
            # Run in parallel
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for exp_class, config in self.experiments:
                    future = executor.submit(self.run_experiment, exp_class, config)
                    futures.append((config.name, future))

                # Collect results
                for name, future in futures:
                    self.results[name] = future.result()
        else:
            # Run sequentially
            for exp_class, config in self.experiments:
                self.results[config.name] = self.run_experiment(exp_class, config)

        # Save aggregated results
        self._save_aggregated_results()

        # Print summary
        self._print_summary()

        return self.results

    def _save_experiment_results(self, name: str, results: Dict[str, Any]):
        """Save individual experiment results."""
        results_path = self.results_dir / f"{name}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    def _save_aggregated_results(self):
        """Save aggregated results from all experiments."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "num_experiments": len(self.experiments),
            "num_successful": sum(1 for r in self.results.values() if r.get("_metadata", {}).get("success", False)),
            "experiments": self.results,
        }

        summary_path = self.results_dir / "experiment_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Saved aggregated results to {summary_path}")

    def _print_summary(self):
        """Print summary of experiment results."""
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)

        successful = 0
        failed = 0
        total_time = 0

        for name, result in self.results.items():
            metadata = result.get("_metadata", {})
            success = metadata.get("success", False)
            duration = metadata.get("duration", 0)

            if success:
                successful += 1
                status = "[SUCCESS]"
            else:
                failed += 1
                status = "[FAILED]"

            total_time += duration

            print(f"{status} | {name} | {duration:.2f}s")

            if not success:
                error = metadata.get("error", "Unknown error")
                print(f"         Error: {error}")

        print("=" * 60)
        print(f"Total: {len(self.results)} experiments")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total time: {total_time:.2f}s")
        print("=" * 60 + "\n")

    def get_best_experiment(self, metric_name: str, layer_name: str, minimize: bool = False) -> Optional[str]:
        """
        Get the best experiment based on a metric.

        Args:
            metric_name: Name of the metric
            layer_name: Name of the layer
            minimize: Whether to minimize the metric

        Returns:
            Name of the best experiment
        """
        best_name = None
        best_value = float("inf") if minimize else float("-inf")

        for name, result in self.results.items():
            if not result.get("_metadata", {}).get("success", False):
                continue

            # Get final metrics
            metrics = result.get("metrics", {})
            if not metrics:
                continue

            # Get last step metrics
            last_step = max(metrics.keys()) if metrics else None
            if last_step is None:
                continue

            # Get metric value
            value = metrics[last_step].get(metric_name, {}).get(layer_name)
            if value is None:
                continue

            # Update best
            if minimize:
                if value < best_value:
                    best_value = value
                    best_name = name
            else:
                if value > best_value:
                    best_value = value
                    best_name = name

        if best_name:
            logger.info(f"Best experiment for {metric_name}/{layer_name}: {best_name} ({best_value:.4f})")

        return best_name
