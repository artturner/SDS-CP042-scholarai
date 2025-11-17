"""JSON exporter for research reports."""

import json
from typing import Optional
from pathlib import Path
from models.report import ResearchReport


class JSONExporter:
    """Export research reports to JSON format."""

    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the JSON exporter.

        Args:
            output_dir: Directory to save exported reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def export(
        self,
        report: ResearchReport,
        filename: Optional[str] = None,
        indent: int = 2
    ) -> str:
        """
        Export a report to JSON.

        Args:
            report: The ResearchReport to export
            filename: Optional filename (defaults to topic-based name)
            indent: JSON indentation level (default: 2)

        Returns:
            Path to the exported file
        """
        # Generate JSON content
        json_content = self._generate_json(report, indent=indent)

        # Determine filename
        if filename is None:
            # Sanitize topic for filename
            safe_topic = "".join(
                c if c.isalnum() or c in (' ', '-', '_') else '_'
                for c in report.topic
            )
            safe_topic = safe_topic[:50]  # Limit length
            filename = f"{safe_topic.strip().replace(' ', '_')}.json"

        # Write to file
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_content)

        return str(filepath)

    def _generate_json(self, report: ResearchReport, indent: int = 2) -> str:
        """Generate JSON content from a report."""
        # Use Pydantic's model_dump to convert to dict
        report_dict = report.model_dump(mode='json')

        # Convert to JSON string with formatting
        json_string = json.dumps(report_dict, indent=indent, ensure_ascii=False)

        return json_string

    def to_string(self, report: ResearchReport, indent: int = 2) -> str:
        """
        Convert report to JSON string without saving to file.

        Args:
            report: The ResearchReport to convert
            indent: JSON indentation level (default: 2)

        Returns:
            JSON formatted string
        """
        return self._generate_json(report, indent=indent)

    def export_compact(
        self,
        report: ResearchReport,
        filename: Optional[str] = None
    ) -> str:
        """
        Export a report to compact JSON (no indentation).

        Args:
            report: The ResearchReport to export
            filename: Optional filename

        Returns:
            Path to the exported file
        """
        return self.export(report, filename=filename, indent=None)


def export_to_json(
    report: ResearchReport,
    output_dir: str = "outputs",
    filename: Optional[str] = None,
    indent: int = 2
) -> str:
    """
    Convenience function to export a report to JSON.

    Args:
        report: The ResearchReport to export
        output_dir: Directory to save the report
        filename: Optional filename
        indent: JSON indentation level

    Returns:
        Path to the exported file
    """
    exporter = JSONExporter(output_dir=output_dir)
    return exporter.export(report, filename=filename, indent=indent)


def to_json(report: ResearchReport, indent: int = 2) -> str:
    """
    Convert report to JSON string.

    Args:
        report: The ResearchReport to convert
        indent: JSON indentation level

    Returns:
        JSON formatted string
    """
    exporter = JSONExporter()
    return exporter.to_string(report, indent=indent)
