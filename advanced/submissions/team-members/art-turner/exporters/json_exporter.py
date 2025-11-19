"""JSON exporter for multi-agent research reports."""

import json
from models.report import MultiAgentReport


def to_json(report: MultiAgentReport, indent: int = 2) -> str:
    """
    Convert a MultiAgentReport to a JSON string.

    Args:
        report: The MultiAgentReport to convert
        indent: Number of spaces for indentation

    Returns:
        JSON string representation of the report
    """
    # Use Pydantic's model_dump for proper serialization
    report_dict = report.model_dump()
    return json.dumps(report_dict, indent=indent, ensure_ascii=False)


def to_dict(report: MultiAgentReport) -> dict:
    """
    Convert a MultiAgentReport to a dictionary.

    Args:
        report: The MultiAgentReport to convert

    Returns:
        Dictionary representation of the report
    """
    return report.model_dump()
