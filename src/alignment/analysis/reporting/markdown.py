"""Markdown report generation utilities."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)


class MarkdownReporter:
    """Generates Markdown reports for alignment analysis results."""

    def __init__(self, title: str = "Alignment Analysis Report"):
        """
        Initialize Markdown reporter.

        Args:
            title: Report title
        """
        self.title = title
        self.sections = []

    def add_section(self, name: str, content: str):
        """Add a section to the report."""
        self.sections.append((name, content))

    def add_table(self, name: str, df: pd.DataFrame):
        """Add a DataFrame as a Markdown table."""
        content = df.to_markdown(index=False)
        self.add_section(name, content)

    def generate(self, output_path: Union[str, Path]):
        """Generate the Markdown report."""
        content = f"# {self.title}\n\n"
        content += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"

        for name, section_content in self.sections:
            content += f"## {name}\n\n{section_content}\n\n"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)

        logger.info(f"Generated Markdown report: {output_path}")
