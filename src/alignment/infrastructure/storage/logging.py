"""
Logging utilities for the alignment framework.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


def setup_logging(
    log_level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
    disable_existing: bool = True,
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_level: Logging level
        log_file: Optional file to log to
        format_string: Custom format string
        disable_existing: Whether to disable existing loggers

    Returns:
        Root logger
    """
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())

    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"

    # Configure logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": disable_existing,
        "formatters": {"default": {"format": format_string, "datefmt": "%Y-%m-%d %H:%M:%S"}},
        "handlers": {"console": {"class": "logging.StreamHandler", "level": log_level, "formatter": "default", "stream": sys.stdout}},
        "root": {"level": log_level, "handlers": ["console"]},
    }

    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logging_config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "level": log_level,
            "formatter": "default",
            "filename": str(log_file),
            "mode": "a",
        }
        logging_config["root"]["handlers"].append("file")

    logging.config.dictConfig(logging_config)

    logger = logging.getLogger()
    logger.info(f"Logging configured with level {logging.getLevelName(log_level)}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None, prefix: str = "", logger: Optional[logging.Logger] = None):
    """
    Log metrics in a structured format.

    Args:
        metrics: Dictionary of metrics
        step: Optional step/iteration number
        prefix: Prefix for metric names
        logger: Logger to use (defaults to root)
    """
    if logger is None:
        logger = logging.getLogger()

    # Format metrics
    formatted = []
    for key, value in metrics.items():
        metric_name = f"{prefix}{key}" if prefix else key
        if isinstance(value, float):
            formatted.append(f"{metric_name}: {value:.4f}")
        else:
            formatted.append(f"{metric_name}: {value}")

    # Create log message
    if step is not None:
        message = f"[Step {step}] " + " | ".join(formatted)
    else:
        message = " | ".join(formatted)

    logger.info(message)


class MetricLogger:
    """
    Logger specifically for tracking metrics over time.
    """

    def __init__(self, log_dir: Union[str, Path], experiment_name: str = "experiment"):
        """
        Initialize metric logger.

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        # Create log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics_{timestamp}.jsonl"
        self.summary_file = self.log_dir / f"{experiment_name}_summary_{timestamp}.json"

        self.metrics_history = []
        self.start_time = datetime.now()

    def log(self, metrics: Dict[str, Any], step: int):
        """
        Log metrics for a specific step.

        Args:
            metrics: Metrics to log
            step: Current step
        """
        entry = {"step": step, "timestamp": datetime.now().isoformat(), "metrics": metrics}

        # Append to history
        self.metrics_history.append(entry)

        # Write to file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def write_summary(self, final_metrics: Optional[Dict[str, Any]] = None):
        """
        Write a summary of the experiment.

        Args:
            final_metrics: Optional final metrics
        """
        summary = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "num_steps": len(self.metrics_history),
            "final_metrics": final_metrics or {},
        }

        # Compute metric statistics
        if self.metrics_history:
            metric_stats = {}
            all_metrics = set()

            # Collect all metric names
            for entry in self.metrics_history:
                all_metrics.update(entry["metrics"].keys())

            # Compute stats for each metric
            for metric_name in all_metrics:
                values = [entry["metrics"][metric_name] for entry in self.metrics_history if metric_name in entry["metrics"]]

                if values and all(isinstance(v, (int, float)) for v in values):
                    metric_stats[metric_name] = {"min": min(values), "max": max(values), "mean": sum(values) / len(values), "final": values[-1]}

            summary["metric_statistics"] = metric_stats

        # Write summary
        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        get_logger(__name__).info(f"Wrote experiment summary to {self.summary_file}")
