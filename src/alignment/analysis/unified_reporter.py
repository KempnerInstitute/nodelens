"""
Unified reporting module for alignment analysis.

This module consolidates reporting functionality from multiple reporter classes,
providing a single interface for generating reports in various formats.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class UnifiedReporter:
    """
    Unified reporter class that can generate reports in multiple formats:
    - HTML
    - Markdown
    - JSON
    - LaTeX (optional)

    This consolidates functionality from HTMLReporter, MarkdownReporter, and JSONReporter.
    """

    def __init__(self, title: str = "Alignment Analysis Report"):
        """
        Initialize unified reporter.

        Args:
            title: Report title
        """
        self.title = title
        self.sections = []
        self.data = {}
        self.figures = []
        self.tables = []
        self.metadata = {
            "title": title,
            "created": datetime.now().isoformat(),
            "version": "1.0"
        }

    def add_section(self, name: str, content: Any, section_type: str = "text"):
        """
        Add a section to the report.

        Args:
            name: Section name
            content: Section content (text, DataFrame, dict, etc.)
            section_type: Type of section ('text', 'data', 'table', 'code')
        """
        self.sections.append({
            "name": name,
            "content": content,
            "type": section_type
        })

    def add_figure(self, figure_path: Union[str, Path], caption: str = "", width: Optional[str] = None):
        """
        Add a figure to the report.

        Args:
            figure_path: Path to the figure
            caption: Figure caption
            width: Optional width specification (e.g., "80%", "600px")
        """
        self.figures.append({
            "path": str(figure_path),
            "caption": caption,
            "width": width
        })

    def add_table(self, name: str, data: Union[pd.DataFrame, Dict, List]):
        """
        Add a table to the report.

        Args:
            name: Table name
            data: Table data (DataFrame, dict, or list of dicts)
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data

        self.tables.append({
            "name": name,
            "data": df
        })

    def add_metadata(self, key: str, value: Any):
        """Add metadata to the report."""
        self.metadata[key] = value

    # ========== HTML Generation ==========

    def generate_html(self, output_path: Union[str, Path], include_toc: bool = True):
        """
        Generate HTML report.

        Args:
            output_path: Path to save the HTML file
            include_toc: Whether to include table of contents
        """
        html = self._generate_html_header(include_toc)

        # Add table of contents
        if include_toc:
            html += self._generate_html_toc()

        # Add sections
        for section in self.sections:
            html += self._generate_html_section(section)

        # Add tables
        if self.tables:
            html += '<div class="section"><h2>Tables</h2>'
            for table in self.tables:
                html += f'<h3>{table["name"]}</h3>'
                html += table["data"].to_html(classes='data-table', index=False)
            html += '</div>'

        # Add figures
        if self.figures:
            html += '<div class="section"><h2>Figures</h2>'
            for fig in self.figures:
                html += self._generate_html_figure(fig)
            html += '</div>'

        html += self._generate_html_footer()

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)

        logger.info(f"Generated HTML report: {output_path}")

    def _generate_html_header(self, include_toc: bool) -> str:
        """Generate HTML header with styles."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1, h2, h3 {{ color: #2c3e50; }}
        h1 {{ border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ border-bottom: 1px solid #ecf0f1; padding-bottom: 5px; margin-top: 30px; }}
        .metadata {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .section {{
            margin: 30px 0;
        }}
        .data-table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        .data-table th, .data-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .data-table th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        .data-table tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .figure {{
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .figure img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .figure-caption {{
            margin-top: 10px;
            font-style: italic;
            color: #666;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        .toc {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .toc ul {{
            list-style-type: none;
            padding-left: 20px;
        }}
        .toc > ul {{
            padding-left: 0;
        }}
        .toc a {{
            text-decoration: none;
            color: #3498db;
        }}
        .toc a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>
    <div class="metadata">
        <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
        <strong>Version:</strong> {self.metadata.get('version', '1.0')}"""

        # Add custom metadata
        for key, value in self.metadata.items():
            if key not in ['title', 'created', 'version']:
                html += f"<br><strong>{key.title()}:</strong> {value}"

        html += "</div>"
        return html

    def _generate_html_toc(self) -> str:
        """Generate table of contents."""
        toc = '<div class="toc"><h2>Table of Contents</h2><ul>'

        for i, section in enumerate(self.sections):
            section_id = f"section-{i}"
            toc += f'<li><a href="#{section_id}">{section["name"]}</a></li>'

        if self.tables:
            toc += '<li><a href="#tables">Tables</a></li>'

        if self.figures:
            toc += '<li><a href="#figures">Figures</a></li>'

        toc += '</ul></div>'
        return toc

    def _generate_html_section(self, section: Dict) -> str:
        """Generate HTML for a section."""
        section_id = f"section-{self.sections.index(section)}"
        html = f'<div class="section" id="{section_id}"><h2>{section["name"]}</h2>'

        content = section["content"]
        section_type = section["type"]

        if section_type == "text":
            html += f"<p>{content}</p>"
        elif section_type == "code":
            html += f"<pre><code>{content}</code></pre>"
        elif section_type == "data":
            if isinstance(content, pd.DataFrame):
                html += content.to_html(classes='data-table', index=False)
            elif isinstance(content, dict):
                html += "<pre>" + json.dumps(content, indent=2) + "</pre>"
            else:
                html += f"<p>{str(content)}</p>"
        elif section_type == "table":
            if isinstance(content, pd.DataFrame):
                html += content.to_html(classes='data-table', index=False)

        html += '</div>'
        return html

    def _generate_html_figure(self, fig: Dict) -> str:
        """Generate HTML for a figure."""
        html = '<div class="figure">'

        width_attr = f'style="width: {fig["width"]}"' if fig.get("width") else ""
        html += f'<img src="{fig["path"]}" alt="{fig["caption"]}" {width_attr}>'

        if fig["caption"]:
            html += f'<div class="figure-caption">{fig["caption"]}</div>'

        html += '</div>'
        return html

    def _generate_html_footer(self) -> str:
        """Generate HTML footer."""
        return """
</body>
</html>"""

    # ========== Markdown Generation ==========

    def generate_markdown(self, output_path: Union[str, Path], include_toc: bool = True):
        """
        Generate Markdown report.

        Args:
            output_path: Path to save the Markdown file
            include_toc: Whether to include table of contents
        """
        md = f"# {self.title}\n\n"
        md += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"

        # Add metadata
        if self.metadata:
            md += "## Metadata\n\n"
            for key, value in self.metadata.items():
                if key not in ['title', 'created']:
                    md += f"- **{key.title()}**: {value}\n"
            md += "\n"

        # Add table of contents
        if include_toc:
            md += "## Table of Contents\n\n"
            for section in self.sections:
                md += f"- [{section['name']}](#{section['name'].lower().replace(' ', '-')})\n"
            if self.tables:
                md += "- [Tables](#tables)\n"
            if self.figures:
                md += "- [Figures](#figures)\n"
            md += "\n"

        # Add sections
        for section in self.sections:
            md += f"## {section['name']}\n\n"

            content = section["content"]
            section_type = section["type"]

            if section_type == "text":
                md += f"{content}\n\n"
            elif section_type == "code":
                md += f"```\n{content}\n```\n\n"
            elif section_type == "data":
                if isinstance(content, pd.DataFrame):
                    md += content.to_markdown(index=False) + "\n\n"
                elif isinstance(content, dict):
                    md += "```json\n" + json.dumps(content, indent=2) + "\n```\n\n"
                else:
                    md += f"{str(content)}\n\n"
            elif section_type == "table":
                if isinstance(content, pd.DataFrame):
                    md += content.to_markdown(index=False) + "\n\n"

        # Add tables
        if self.tables:
            md += "## Tables\n\n"
            for table in self.tables:
                md += f"### {table['name']}\n\n"
                md += table["data"].to_markdown(index=False) + "\n\n"

        # Add figures
        if self.figures:
            md += "## Figures\n\n"
            for fig in self.figures:
                md += f"![{fig['caption']}]({fig['path']})\n"
                if fig['caption']:
                    md += f"*{fig['caption']}*\n"
                md += "\n"

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md)

        logger.info(f"Generated Markdown report: {output_path}")

    # ========== JSON Generation ==========

    def generate_json(self, output_path: Union[str, Path], pretty: bool = True):
        """
        Generate JSON report.

        Args:
            output_path: Path to save the JSON file
            pretty: Whether to pretty-print the JSON
        """
        report_data = {
            "metadata": self.metadata,
            "sections": [],
            "tables": [],
            "figures": []
        }

        # Add sections
        for section in self.sections:
            section_data = {
                "name": section["name"],
                "type": section["type"]
            }

            content = section["content"]
            if isinstance(content, pd.DataFrame):
                section_data["content"] = content.to_dict('records')
            elif isinstance(content, (dict, list, str, int, float)):
                section_data["content"] = content
            else:
                section_data["content"] = str(content)

            report_data["sections"].append(section_data)

        # Add tables
        for table in self.tables:
            report_data["tables"].append({
                "name": table["name"],
                "data": table["data"].to_dict('records')
            })

        # Add figures
        report_data["figures"] = self.figures

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            if pretty:
                json.dump(report_data, f, indent=2, default=str)
            else:
                json.dump(report_data, f, default=str)

        logger.info(f"Generated JSON report: {output_path}")

    # ========== Multi-format Generation ==========

    def generate_all(self, output_dir: Union[str, Path], basename: str = "report"):
        """
        Generate reports in all supported formats.

        Args:
            output_dir: Directory to save reports
            basename: Base filename (without extension)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate all formats
        self.generate_html(output_dir / f"{basename}.html")
        self.generate_markdown(output_dir / f"{basename}.md")
        self.generate_json(output_dir / f"{basename}.json")

        logger.info(f"Generated all report formats in: {output_dir}")

    # ========== Convenience Methods ==========

    def add_dataframe(self, name: str, df: pd.DataFrame, as_section: bool = False):
        """
        Add a DataFrame to the report.

        Args:
            name: Name for the DataFrame
            df: The DataFrame
            as_section: Whether to add as a section (True) or table (False)
        """
        if as_section:
            self.add_section(name, df, section_type="data")
        else:
            self.add_table(name, df)

    def add_metrics(self, metrics: Dict[str, Any], section_name: str = "Metrics"):
        """
        Add metrics dictionary as a formatted section.

        Args:
            metrics: Dictionary of metrics
            section_name: Name for the metrics section
        """
        # Convert metrics to DataFrame for better formatting
        if all(isinstance(v, dict) for v in metrics.values()):
            # Nested metrics
            df = pd.DataFrame(metrics).T
            self.add_section(section_name, df, section_type="data")
        else:
            # Simple metrics
            df = pd.DataFrame([metrics])
            self.add_section(section_name, df, section_type="data")

    def add_summary(self, summary_dict: Dict[str, Any]):
        """
        Add a summary section with key-value pairs.

        Args:
            summary_dict: Dictionary of summary information
        """
        content = ""
        for key, value in summary_dict.items():
            content += f"**{key}**: {value}\n\n"

        self.add_section("Summary", content, section_type="text")


# Convenience function for quick report generation
def generate_quick_report(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    title: str = "Quick Report",
    format: str = "html"
):
    """
    Generate a quick report from results dictionary.

    Args:
        results: Results dictionary
        output_path: Output path
        title: Report title
        format: Output format ('html', 'markdown', 'json', 'all')
    """
    reporter = UnifiedReporter(title)

    # Add sections based on results content
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            reporter.add_dataframe(key, value)
        elif isinstance(value, dict):
            reporter.add_section(key, value, section_type="data")
        else:
            reporter.add_section(key, str(value), section_type="text")

    # Generate report
    if format == "all":
        output_dir = Path(output_path).parent
        basename = Path(output_path).stem
        reporter.generate_all(output_dir, basename)
    elif format == "html":
        reporter.generate_html(output_path)
    elif format == "markdown":
        reporter.generate_markdown(output_path)
    elif format == "json":
        reporter.generate_json(output_path)
    else:
        raise ValueError(f"Unknown format: {format}")
