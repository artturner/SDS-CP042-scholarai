"""Synthesizer Agent for generating structured research reports."""

import os
import json
from typing import List, Dict, Optional
from datetime import datetime
from openai import OpenAI
from models.report import ResearchReport, KeyFinding, Source


class SynthesizerAgent:
    """Agent that synthesizes research findings into structured reports."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
    ):
        """
        Initialize the Synthesizer Agent.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY from environment.
            model: OpenAI model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

        # System prompt for the synthesizer
        self.system_prompt = """You are an expert research synthesizer. Your task is to analyze research sources and create structured, comprehensive reports.

Your responsibilities:
1. Read and analyze all provided sources carefully
2. Synthesize information into clear, concise findings
3. Identify conflicts or disagreements between sources
4. Provide accurate citations for all claims
5. Generate a structured report with the following components:
   - TL;DR: A concise summary (≤120 words) of the main insights
   - Key Findings: Bulleted list of main discoveries with citations
   - Conflicts & Caveats: Important disagreements or limitations
   - Top Sources: The 5 most relevant and authoritative sources with explanations

Guidelines:
- Be objective and evidence-based
- Cite sources using their URLs
- Highlight both consensus and disagreements
- Prioritize recent and authoritative sources
- Keep the TL;DR under 120 words
- Each key finding should be supported by at least one citation
- Explain why each top source matters

Format your response as valid JSON matching this structure:
{
  "tldr": "string (≤120 words)",
  "key_findings": [
    {
      "finding": "string",
      "citations": ["url1", "url2"]
    }
  ],
  "conflicts_and_caveats": "string",
  "top_sources": [
    {
      "title": "string",
      "url": "string",
      "snippet": "string",
      "score": float,
      "why_matters": "string"
    }
  ]
}"""

    def synthesize(
        self,
        topic: str,
        sources: List[Dict],
        analysis: Optional[str] = None
    ) -> ResearchReport:
        """
        Synthesize research sources into a structured report.

        Args:
            topic: The research topic
            sources: List of source dictionaries with title, url, snippet, score
            analysis: Optional preliminary analysis from research agent

        Returns:
            ResearchReport object with structured findings
        """
        # Prepare sources for the prompt
        sources_text = self._format_sources_for_prompt(sources)

        # Build user message
        user_message = f"""Topic: {topic}

Sources to analyze:
{sources_text}
"""

        if analysis:
            user_message += f"\nPreliminary analysis:\n{analysis}\n"

        user_message += "\nPlease synthesize these sources into a structured report. Return ONLY valid JSON, no other text."

        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,  # Lower temperature for more consistent output
        )

        # Parse response
        result_text = response.choices[0].message.content
        result_data = json.loads(result_text)

        # Build ResearchReport
        report = self._build_report(topic, result_data, sources)

        return report

    def _format_sources_for_prompt(self, sources: List[Dict]) -> str:
        """Format sources for inclusion in prompt."""
        formatted = []
        for i, source in enumerate(sources, 1):
            formatted.append(
                f"\n[{i}] {source.get('title', 'Untitled')}\n"
                f"    URL: {source.get('url', 'N/A')}\n"
                f"    Score: {source.get('score', 'N/A')}\n"
                f"    Content: {source.get('snippet', 'No content available')}"
            )
        return "\n".join(formatted)

    def _build_report(
        self,
        topic: str,
        result_data: Dict,
        original_sources: List[Dict]
    ) -> ResearchReport:
        """Build a ResearchReport from LLM output."""
        # Parse key findings
        key_findings = [
            KeyFinding(
                finding=kf.get("finding", ""),
                citations=kf.get("citations", [])
            )
            for kf in result_data.get("key_findings", [])
        ]

        # Parse top sources
        top_sources_data = result_data.get("top_sources", [])

        # If LLM didn't provide top sources, use the highest scored original sources
        if not top_sources_data:
            sorted_sources = sorted(
                original_sources,
                key=lambda x: x.get("score", 0),
                reverse=True
            )[:5]
            top_sources_data = [
                {
                    "title": s.get("title", ""),
                    "url": s.get("url", ""),
                    "snippet": s.get("snippet", ""),
                    "score": s.get("score"),
                    "why_matters": "High relevance score"
                }
                for s in sorted_sources
            ]

        top_sources = [
            Source(
                title=s.get("title", ""),
                url=s.get("url", ""),
                snippet=s.get("snippet", ""),
                score=s.get("score"),
                why_matters=s.get("why_matters", "")
            )
            for s in top_sources_data[:5]  # Ensure max 5
        ]

        # Build report
        report = ResearchReport(
            topic=topic,
            tldr=result_data.get("tldr", ""),
            key_findings=key_findings,
            conflicts_and_caveats=result_data.get("conflicts_and_caveats", ""),
            top_sources=top_sources,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "model": self.model,
                "num_sources_analyzed": len(original_sources)
            }
        )

        return report


def create_synthesizer_agent(model: str = "gpt-4-turbo-preview") -> SynthesizerAgent:
    """
    Factory function to create a synthesizer agent.

    Args:
        model: OpenAI model to use

    Returns:
        Initialized SynthesizerAgent instance
    """
    return SynthesizerAgent(model=model)
