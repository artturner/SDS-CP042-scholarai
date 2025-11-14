"""Data models for research reports.

This module defines the data structures using Pydantic, which provides:
1. Automatic validation - ensures data types are correct
2. Serialization - easy conversion to JSON/dict
3. Documentation - field descriptions are self-documenting
4. Type safety - IDE autocomplete and type checking

Why Pydantic? It's the standard for modern Python data validation and
is used by FastAPI, LangChain, and many other frameworks.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl


class Source(BaseModel):
    """
    Model for a source citation.

    Represents a single web source with all its metadata.
    This is used for the "Top 5 Sources" section of our reports.
    """

    # Field(...) means required field, the ... is Python's Ellipsis indicating no default
    title: str = Field(..., description="Title of the source")
    url: str = Field(..., description="URL of the source")
    snippet: str = Field(..., description="Relevant excerpt or snippet from the source")

    # Optional fields use None as default, indicated by Optional[type]
    score: Optional[float] = Field(None, description="Relevance score (0-1)")

    # This field is filled by the Synthesizer Agent to explain why each source matters
    why_matters: Optional[str] = Field(
        None, description="Explanation of why this source is important"
    )


class KeyFinding(BaseModel):
    """
    Model for a key finding with citation.

    Represents one insight discovered during research, along with the
    sources that support it. This enforces our requirement that all
    findings must be backed by citations.
    """

    # The actual insight or discovery
    finding: str = Field(..., description="The key finding or insight")

    # List of URLs that support this finding
    # default_factory=list creates an empty list if none provided
    # This is safer than using [] as default (which is mutable and can cause bugs)
    citations: List[str] = Field(
        default_factory=list,
        description="List of source URLs supporting this finding"
    )


class ResearchReport(BaseModel):
    """
    Complete research report model.

    This is the main data structure that represents a fully synthesized research report.
    It contains all the components required by the project specification:
    - TL;DR (≤120 words)
    - Key findings with citations
    - Conflicts & caveats
    - Top 5 sources with explanations
    """

    # The original research question or topic from the user
    topic: str = Field(..., description="The research topic or question")

    # TL;DR summary - our specification requires ≤120 words
    # max_length=800 chars is approximately 120 words (avg 6.7 chars/word)
    # Pydantic will validate this doesn't exceed the limit
    tldr: str = Field(
        ...,
        description="TL;DR summary (≤120 words)",
        max_length=800  # Roughly 120 words
    )

    # List of key findings, each with supporting citations
    # Using default_factory ensures each instance gets its own list
    key_findings: List[KeyFinding] = Field(
        default_factory=list,
        description="List of key findings with citations"
    )

    # Discussion of any disagreements between sources or important limitations
    # Empty string default means this section is optional but always present
    conflicts_and_caveats: str = Field(
        default="",
        description="Discussion of conflicts between sources and important caveats"
    )

    # Top 5 most important/relevant sources
    # max_length ensures we don't exceed the 5 source limit from the spec
    top_sources: List[Source] = Field(
        default_factory=list,
        description="Top 5 most relevant sources",
        max_length=5
    )

    # Extra data like timestamp, which model generated it, etc.
    # Using dict allows flexibility without defining every possible field
    metadata: Optional[dict] = Field(
        default_factory=dict,
        description="Additional metadata (timestamp, model used, etc.)"
    )

    def model_dump_summary(self) -> dict:
        """Return a summary version of the report."""
        return {
            "topic": self.topic,
            "tldr": self.tldr,
            "num_findings": len(self.key_findings),
            "num_sources": len(self.top_sources),
        }

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "topic": "Recent advances in large language models",
                "tldr": "Large language models have seen significant advances...",
                "key_findings": [
                    {
                        "finding": "Transformer architecture has become dominant",
                        "citations": ["https://example.com/paper1"]
                    }
                ],
                "conflicts_and_caveats": "Some sources disagree on...",
                "top_sources": [
                    {
                        "title": "Attention Is All You Need",
                        "url": "https://example.com/paper",
                        "snippet": "We propose a new architecture...",
                        "score": 0.95,
                        "why_matters": "Foundational paper introducing transformers"
                    }
                ],
                "metadata": {
                    "timestamp": "2025-01-14T10:30:00Z",
                    "model": "gpt-4-turbo-preview"
                }
            }
        }
