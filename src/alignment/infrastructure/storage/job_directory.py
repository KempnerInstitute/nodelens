"""
Job directory management for organizing experiment outputs.

This module provides utilities for creating unique, timestamped directories
for each experiment job. All results, logs, checkpoints, and visualizations
are stored within a single job directory for cleaner organization.

Directory Structure:
    base_output_dir/
        {experiment_name}_{timestamp}_{job_id}/
            results/
                results.json
                pruning_results.json
            logs/
                experiment.log
            checkpoints/
            figures/
            analysis/
"""

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


def get_slurm_job_id() -> Optional[str]:
    """
    Get the SLURM job ID if running under SLURM.

    Returns:
        SLURM job ID as string, or None if not running under SLURM.
    """
    return os.environ.get("SLURM_JOB_ID")


def get_slurm_array_task_id() -> Optional[str]:
    """
    Get the SLURM array task ID if running as part of a job array.

    Returns:
        SLURM array task ID as string, or None if not a job array.
    """
    return os.environ.get("SLURM_ARRAY_TASK_ID")


def generate_unique_id() -> str:
    """
    Generate a short unique identifier.

    Returns:
        8-character unique ID.
    """
    return uuid.uuid4().hex[:8]


def create_job_directory(
    base_output_dir: Union[str, Path],
    experiment_name: str,
    timestamp: Optional[str] = None,
    job_id: Optional[str] = None,
    create_subdirs: bool = True,
) -> Path:
    """
    Create a unique job directory for experiment outputs.

    The directory name is formatted as:
        {experiment_name}_{timestamp}_{job_id}

    Where:
        - experiment_name: Name of the experiment from config
        - timestamp: ISO format timestamp (defaults to current time)
        - job_id: SLURM job ID if available, otherwise a unique ID

    Args:
        base_output_dir: Base directory for all experiment outputs.
        experiment_name: Name of the experiment.
        timestamp: Optional timestamp string. If None, uses current time.
        job_id: Optional job ID. If None, uses SLURM_JOB_ID or generates unique ID.
        create_subdirs: Whether to create standard subdirectories.

    Returns:
        Path to the created job directory.

    Example:
        >>> job_dir = create_job_directory(
        ...     "/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM",
        ...     "llama3_8b_pruning"
        ... )
        >>> print(job_dir)
        /n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM/llama3_8b_pruning_20241209_143052_12345678
    """
    base_output_dir = Path(base_output_dir)

    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get job ID
    if job_id is None:
        # Try SLURM job ID first
        slurm_job_id = get_slurm_job_id()
        slurm_array_id = get_slurm_array_task_id()

        if slurm_job_id:
            if slurm_array_id:
                job_id = f"{slurm_job_id}_{slurm_array_id}"
            else:
                job_id = slurm_job_id
        else:
            # Generate a unique ID for non-SLURM runs
            job_id = generate_unique_id()

    # Sanitize experiment name (remove special characters)
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in experiment_name)

    # Create directory name
    dir_name = f"{safe_name}_{timestamp}_{job_id}"
    job_dir = base_output_dir / dir_name

    # Create the directory
    job_dir.mkdir(parents=True, exist_ok=True)

    # Create standard subdirectories
    if create_subdirs:
        subdirs = ["results", "logs", "checkpoints", "figures", "analysis"]
        for subdir in subdirs:
            (job_dir / subdir).mkdir(exist_ok=True)

    logger.info(f"Created job directory: {job_dir}")

    return job_dir


def get_job_directory_paths(job_dir: Union[str, Path]) -> dict:
    """
    Get standard paths within a job directory.

    Args:
        job_dir: Path to the job directory.

    Returns:
        Dictionary with paths to standard subdirectories and files.
    """
    job_dir = Path(job_dir)

    return {
        "root": job_dir,
        "results": job_dir / "results",
        "logs": job_dir / "logs",
        "checkpoints": job_dir / "checkpoints",
        "figures": job_dir / "figures",
        "analysis": job_dir / "analysis",
        # Common file paths
        "experiment_log": job_dir / "logs" / "experiment.log",
        "config_file": job_dir / "experiment_config.yaml",
        "results_file": job_dir / "results" / "results.json",
    }


def setup_job_logging(
    job_dir: Union[str, Path],
    log_level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup logging to write to the job directory.

    Args:
        job_dir: Path to the job directory.
        log_level: Logging level.

    Returns:
        Configured root logger.
    """
    paths = get_job_directory_paths(job_dir)
    log_file = paths["experiment_log"]

    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    root_logger = logging.getLogger()
    root_logger.info(f"Logging to: {log_file}")

    return root_logger


class JobDirectory:
    """
    Context manager for job directories.

    Provides convenient access to job directory paths and handles
    cleanup on errors.

    Example:
        >>> with JobDirectory(base_dir, "my_experiment") as job:
        ...     # Access paths
        ...     print(job.results_dir)
        ...     print(job.figures_dir)
        ...
        ...     # Save files
        ...     job.save_config(config)
        ...     job.save_results(results)
    """

    def __init__(
        self,
        base_output_dir: Union[str, Path],
        experiment_name: str,
        timestamp: Optional[str] = None,
        job_id: Optional[str] = None,
        setup_logging: bool = True,
        log_level: int = logging.INFO,
    ):
        """
        Initialize job directory.

        Args:
            base_output_dir: Base directory for experiment outputs.
            experiment_name: Name of the experiment.
            timestamp: Optional timestamp string.
            job_id: Optional job ID.
            setup_logging: Whether to configure logging.
            log_level: Logging level.
        """
        self.base_output_dir = Path(base_output_dir)
        self.experiment_name = experiment_name
        self._timestamp = timestamp
        self._job_id = job_id
        self._setup_logging = setup_logging
        self._log_level = log_level

        self._job_dir: Optional[Path] = None
        self._paths: Optional[dict] = None

    @property
    def job_dir(self) -> Path:
        """Get the job directory path."""
        if self._job_dir is None:
            raise RuntimeError("JobDirectory not initialized. Use as context manager.")
        return self._job_dir

    @property
    def results_dir(self) -> Path:
        """Get the results subdirectory."""
        return self._paths["results"]

    @property
    def logs_dir(self) -> Path:
        """Get the logs subdirectory."""
        return self._paths["logs"]

    @property
    def checkpoints_dir(self) -> Path:
        """Get the checkpoints subdirectory."""
        return self._paths["checkpoints"]

    @property
    def figures_dir(self) -> Path:
        """Get the figures subdirectory."""
        return self._paths["figures"]

    @property
    def analysis_dir(self) -> Path:
        """Get the analysis subdirectory."""
        return self._paths["analysis"]

    def __enter__(self) -> "JobDirectory":
        """Create job directory and setup logging."""
        self._job_dir = create_job_directory(
            self.base_output_dir,
            self.experiment_name,
            timestamp=self._timestamp,
            job_id=self._job_id,
        )
        self._paths = get_job_directory_paths(self._job_dir)

        if self._setup_logging:
            setup_job_logging(self._job_dir, self._log_level)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Handle cleanup on exit."""
        if exc_type is not None:
            logger.error(f"Job failed with error: {exc_val}")
        return False  # Don't suppress exceptions

    def save_config(self, config, filename: str = "experiment_config.yaml"):
        """
        Save experiment configuration to the job directory.

        Args:
            config: Configuration object with save() method or dict.
            filename: Name of the config file.
        """
        import yaml

        config_path = self._job_dir / filename

        if hasattr(config, "save"):
            config.save(config_path)
        elif hasattr(config, "to_dict"):
            with open(config_path, "w") as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False)
        elif isinstance(config, dict):
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Cannot save config of type {type(config)}")

        logger.info(f"Saved config to: {config_path}")

    def save_results(self, results: dict, filename: str = "results.json"):
        """
        Save results to the results subdirectory.

        Args:
            results: Results dictionary.
            filename: Name of the results file.
        """
        import json

        results_path = self.results_dir / filename

        def convert_to_serializable(obj):
            if hasattr(obj, "tolist"):
                return obj.tolist()
            elif hasattr(obj, "item"):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(i) for i in obj]
            return obj

        serializable = convert_to_serializable(results)

        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Saved results to: {results_path}")
        return results_path
