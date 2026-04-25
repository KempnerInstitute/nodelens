"""JSON report generation utilities."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import pandas as pd

logger = logging.getLogger(__name__)


class JSONReporter:
    """Generates JSON reports for alignment analysis results."""

    def __init__(self, title: str = "Alignment Analysis Report"):
        """
        Initialize JSON reporter.

        Args:
            title: Report title
        """
        self.title = title
        self.data = {"title": title, "timestamp": datetime.now().isoformat(), "sections": {}}

    def add_section(self, name: str, data: Any):
        """
        Add a section to the report.

        Args:
            name: Section name
            data: Section data (DataFrame will be converted to dict)
        """
        if isinstance(data, pd.DataFrame):
            data = data.to_dict("records")
        self.data["sections"][name] = data

    def generate(self, output_path: Union[str, Path]):
        """Generate the JSON report."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)

        logger.info(f"Generated JSON report: {output_path}")
