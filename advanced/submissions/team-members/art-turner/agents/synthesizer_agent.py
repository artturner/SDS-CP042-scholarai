"""Synthesizer Agent for merging multi-agent research findings.

This agent takes findings from multiple researcher agents and synthesizes
them into a coherent, structured report with executive summary, insights
by subtopic, conflicts/gaps, and consolidated citations.
"""

import os
import json
from typing import List, Optional
from datetime import datetime
from openai import OpenAI
from models.report import (
    SubtopicFindings,
    MultiAgentReport,
    KeyFinding,
    Source
)


class SynthesizerAgent:
    """
    Agent that synthesizes findings from multiple researchers into a unified report.

    This is the final agent in the pipeline. It takes all SubtopicFindings
    and produces a comprehensive MultiAgentReport with:
    - Executive Summary (≤150 words)
    - Key Insights by Subtopic
    - Conflicts or Gaps in Literature
    - Citations & Resource List
    """

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

        self.system_prompt = """You are an expert research synthesizer. Your task is to merge findings from multiple research agents into a comprehensive, coherent report.

You will receive findings from multiple researchers, each focused on a different subtopic. Your responsibilities:

1. Create an Executive Summary (≤150 words) that captures the main insights across all subtopics
2. Identify overall key insights that emerge from combining the research
3. Find consensus points where multiple sources/subtopics agree
4. Identify conflicts, contradictions, or gaps in the literature
5. Select the top 5 most important sources overall

Guidelines:
- Maintain objectivity and evidence-based reasoning
- Highlight both agreements and disagreements between sources
- Cite sources using their URLs
- Keep the executive summary concise but comprehensive
- Each key insight should synthesize across subtopics where possible

Return your synthesis as valid JSON:
{
  "executive_summary": "150 words max, covering main insights from all subtopics",
  "overall_insights": [
    {
      "finding": "Cross-cutting insight that emerges from multiple subtopics",
      "citations": ["url1", "url2"]
    }
  ],
  "consensus_points": [
    "Point where multiple sources agree"
  ],
  "conflicts_and_gaps": "Discussion of contradictions and areas needing more research",
  "top_sources": [
    {
      "title": "Source title",
      "url": "Source URL",
      "snippet": "Key excerpt",
      "score": 0.95,
      "why_matters": "Why this source is particularly important"
    }
  ]
}"""

    def synthesize(
        self,
        topic: str,
        subtopic_findings: List[SubtopicFindings],
        style: str = "Technical",
        tone: str = "Neutral"
    ) -> MultiAgentReport:
        """
        Synthesize multiple subtopic findings into a unified report.

        Args:
            topic: The main research topic
            subtopic_findings: List of findings from each researcher agent
            style: Writing style (Technical/Layperson/Business)
            tone: Tone (Neutral/Advisory)

        Returns:
            MultiAgentReport with synthesized findings
        """
        # Style and tone instructions
        style_instructions = {
            "Technical": "Use technical language and domain-specific terminology.",
            "Layperson": "Use clear, simple language accessible to general audiences.",
            "Business": "Focus on practical implications and actionable insights."
        }

        tone_instructions = {
            "Neutral": "Present findings objectively without recommendations.",
            "Advisory": "Provide insights and recommendations based on findings."
        }

        # Format subtopic findings for the prompt
        findings_text = self._format_findings_for_prompt(subtopic_findings)

        # Build user message
        user_message = f"""Main Research Topic: {topic}

Style: {style_instructions.get(style, style_instructions["Technical"])}
Tone: {tone_instructions.get(tone, tone_instructions["Neutral"])}

Research Findings by Subtopic:
{findings_text}

Please synthesize these findings into a comprehensive report. Return ONLY valid JSON."""

        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        result_text = response.choices[0].message.content
        result_data = json.loads(result_text)

        # Build the report
        report = self._build_report(topic, subtopic_findings, result_data)

        return report

    def _format_findings_for_prompt(self, subtopic_findings: List[SubtopicFindings]) -> str:
        """Format subtopic findings for inclusion in the synthesis prompt."""
        formatted = []

        for i, findings in enumerate(subtopic_findings, 1):
            section = f"\n{'='*50}\nSUBTOPIC {i}: {findings.subtopic}\n{'='*50}\n"
            section += f"\nSummary: {findings.summary}\n"

            if findings.key_insights:
                section += "\nKey Insights:\n"
                for j, insight in enumerate(findings.key_insights, 1):
                    section += f"  {j}. {insight.finding}\n"
                    if insight.citations:
                        section += f"     Citations: {', '.join(insight.citations[:3])}\n"

            if findings.sources:
                section += f"\nSources ({len(findings.sources)} total):\n"
                for source in findings.sources[:5]:  # Top 5 sources per subtopic
                    section += f"  - {source.title}\n"
                    section += f"    URL: {source.url}\n"
                    if source.score:
                        section += f"    Score: {source.score:.3f}\n"

            if findings.researcher_notes:
                section += f"\nResearcher Notes: {findings.researcher_notes}\n"

            formatted.append(section)

        return "\n".join(formatted)

    def _build_report(
        self,
        topic: str,
        subtopic_findings: List[SubtopicFindings],
        result_data: dict
    ) -> MultiAgentReport:
        """Build a MultiAgentReport from synthesizer output."""

        # Parse overall insights
        overall_insights = [
            KeyFinding(
                finding=ins.get("finding", ""),
                citations=ins.get("citations", [])
            )
            for ins in result_data.get("overall_insights", [])
        ]

        # Parse top sources
        top_sources_data = result_data.get("top_sources", [])
        top_sources = [
            Source(
                title=s.get("title", ""),
                url=s.get("url", ""),
                snippet=s.get("snippet", ""),
                score=s.get("score"),
                why_matters=s.get("why_matters", "")
            )
            for s in top_sources_data[:5]
        ]

        # Collect all sources from all researchers
        all_sources = []
        for findings in subtopic_findings:
            all_sources.extend(findings.sources)

        # Remove duplicates by URL
        seen_urls = set()
        unique_sources = []
        for source in all_sources:
            if source.url not in seen_urls:
                seen_urls.add(source.url)
                unique_sources.append(source)

        # Build the report
        report = MultiAgentReport(
            topic=topic,
            subtopics=[f.subtopic for f in subtopic_findings],
            executive_summary=result_data.get("executive_summary", ""),
            subtopic_findings=subtopic_findings,
            overall_insights=overall_insights,
            consensus_points=result_data.get("consensus_points", []),
            conflicts_and_gaps=result_data.get("conflicts_and_gaps", ""),
            all_sources=unique_sources,
            top_sources=top_sources,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "model": self.model,
                "num_researchers": len(subtopic_findings),
                "total_sources": len(unique_sources)
            }
        )

        return report

    def revise(
        self,
        report: MultiAgentReport,
        revision_instructions: str,
        style: str = "Technical",
        tone: str = "Neutral"
    ) -> MultiAgentReport:
        """
        Revise a report based on critic feedback.

        Args:
            report: The original MultiAgentReport to revise
            revision_instructions: Instructions from the Critic Agent
            style: Writing style
            tone: Tone

        Returns:
            Revised MultiAgentReport
        """
        # Style and tone instructions
        style_instructions = {
            "Technical": "Use technical language and domain-specific terminology.",
            "Layperson": "Use clear, simple language accessible to general audiences.",
            "Business": "Focus on practical implications and actionable insights."
        }

        tone_instructions = {
            "Neutral": "Present findings objectively without recommendations.",
            "Advisory": "Provide insights and recommendations based on findings."
        }

        # Format the current report for context
        current_report_text = self._format_report_for_revision(report)

        revision_prompt = f"""You are revising a research report based on critical feedback.

Style: {style_instructions.get(style, style_instructions["Technical"])}
Tone: {tone_instructions.get(tone, tone_instructions["Neutral"])}

CURRENT REPORT:
{current_report_text}

REVISION INSTRUCTIONS FROM CRITIC:
{revision_instructions}

Please revise the report to address the critic's concerns. Maintain the same structure but improve the content based on the feedback. Return ONLY valid JSON with the same structure as before:
{{
  "executive_summary": "Revised executive summary",
  "overall_insights": [...],
  "consensus_points": [...],
  "conflicts_and_gaps": "...",
  "top_sources": [...]
}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": revision_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        result_text = response.choices[0].message.content
        result_data = json.loads(result_text)

        # Build revised report
        revised_report = self._build_report(
            report.topic,
            report.subtopic_findings,
            result_data
        )

        # Increment revision count
        revised_report.revision_count = report.revision_count + 1

        return revised_report

    def _format_report_for_revision(self, report: MultiAgentReport) -> str:
        """Format the current report for revision context."""
        sections = []

        sections.append(f"TOPIC: {report.topic}")
        sections.append(f"\nEXECUTIVE SUMMARY:\n{report.executive_summary}")

        if report.overall_insights:
            sections.append("\nOVERALL INSIGHTS:")
            for i, insight in enumerate(report.overall_insights, 1):
                sections.append(f"  {i}. {insight.finding}")
                if insight.citations:
                    sections.append(f"     Citations: {', '.join(insight.citations)}")

        if report.consensus_points:
            sections.append("\nCONSENSUS POINTS:")
            for point in report.consensus_points:
                sections.append(f"  - {point}")

        if report.conflicts_and_gaps:
            sections.append(f"\nCONFLICTS & GAPS:\n{report.conflicts_and_gaps}")

        if report.top_sources:
            sections.append("\nTOP SOURCES:")
            for source in report.top_sources:
                sections.append(f"  - {source.title} ({source.url})")
                if source.why_matters:
                    sections.append(f"    Why it matters: {source.why_matters}")

        return "\n".join(sections)


def create_synthesizer_agent(model: str = "gpt-4-turbo-preview") -> SynthesizerAgent:
    """
    Factory function to create a synthesizer agent.

    Args:
        model: OpenAI model to use

    Returns:
        Initialized SynthesizerAgent instance
    """
    return SynthesizerAgent(model=model)
