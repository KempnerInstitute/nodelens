"""Reporting utilities for generating analysis reports."""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class HTMLReporter:
    """Generates HTML reports."""
    
    def __init__(self, title: str = "Alignment Analysis Report"):
        self.title = title
        self.sections = []
        self.figures = []
    
    def add_section(self, name: str, content: str):
        self.sections.append((name, content))
    
    def add_figure(self, figure_path: str, caption: str = ""):
        self.figures.append((figure_path, caption))
    
    def add_dataframe(self, name: str, df: pd.DataFrame):
        html = df.to_html(classes='metric-table', index=False)
        self.add_section(name, html)
    
    def generate(self, output_path: Union[str, Path]):
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1, h2 {{ color: #333; }}
        .section {{ margin: 20px 0; }}
        .metric-table {{ border-collapse: collapse; width: 100%; }}
        .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .metric-table th {{ background-color: #4CAF50; color: white; }}
        .figure {{ text-align: center; margin: 20px 0; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
"""
        
        for name, content in self.sections:
            html += f'<div class="section"><h2>{name}</h2>{content}</div>'
        
        if self.figures:
            html += '<div class="section"><h2>Visualizations</h2>'
            for path, caption in self.figures:
                html += f'<div class="figure"><img src="{path}" alt="{caption}">'
                if caption:
                    html += f'<p>{caption}</p>'
                html += '</div>'
            html += '</div>'
        
        html += '</body></html>'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)
        
        logger.info(f"Generated HTML report: {output_path}")


class MarkdownReporter:
    """Generates Markdown reports."""
    
    def __init__(self, title: str = "Alignment Analysis Report"):
        self.title = title
        self.sections = []
    
    def add_section(self, name: str, content: str):
        self.sections.append((name, content))
    
    def add_table(self, name: str, df: pd.DataFrame):
        content = df.to_markdown(index=False)
        self.add_section(name, content)
    
    def generate(self, output_path: Union[str, Path]):
        content = f"# {self.title}\n\n"
        content += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        
        for name, section_content in self.sections:
            content += f"## {name}\n\n{section_content}\n\n"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        
        logger.info(f"Generated Markdown report: {output_path}")


class JSONReporter:
    """Generates JSON reports."""
    
    def __init__(self, title: str = "Alignment Analysis Report"):
        self.title = title
        self.data = {
            "title": title,
            "timestamp": datetime.now().isoformat(),
            "sections": {}
        }
    
    def add_section(self, name: str, data: Any):
        if isinstance(data, pd.DataFrame):
            data = data.to_dict('records')
        self.data["sections"][name] = data
    
    def generate(self, output_path: Union[str, Path]):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
        
        logger.info(f"Generated JSON report: {output_path}")
