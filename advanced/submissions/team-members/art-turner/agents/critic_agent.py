"""Critic Agent for reviewing and requesting revisions to research reports.

This agent reviews the Synthesizer's output for factual consistency,
citation accuracy, logical coherence, and completeness. It can either
approve the report or request specific revisions.
"""

import os
import json
from typing import Optional
from openai import OpenAI
from models.report import MultiAgentReport, CriticReview, CriticIssue


class CriticAgent:
    """
    Agent that reviews synthesized reports and requests revisions if needed.

    The Critic Agent performs quality control on the final report by checking:
    - Factual consistency between claims and sources
    - Citation accuracy and attribution
    - Logical coherence (no contradictions)
    - Completeness (key findings included)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        strictness: str = "balanced"
    ):
        """
        Initialize the Critic Agent.

        Args:
            api_key: OpenAI API key. If not provided, uses environment variable.
            model: OpenAI model to use
            strictness: Review strictness level (lenient/balanced/strict)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.strictness = strictness

        # Adjust thresholds based on strictness
        strictness_config = {
            "lenient": {
                "description": "Only flag major issues that significantly impact report quality",
                "revision_threshold": "Only request revision for serious factual errors or missing critical information"
            },
            "balanced": {
                "description": "Flag moderate to major issues that affect report reliability",
                "revision_threshold": "Request revision for factual inconsistencies, poor citations, or significant gaps"
            },
            "strict": {
                "description": "Flag all issues including minor ones for maximum quality",
                "revision_threshold": "Request revision for any factual issues, weak citations, or incomplete coverage"
            }
        }

        config = strictness_config.get(strictness, strictness_config["balanced"])

        self.system_prompt = f"""You are a critical reviewer for research reports. Your task is to evaluate the quality and accuracy of synthesized research findings.

Review Strictness: {strictness.upper()}
{config["description"]}

Your review should check for:

1. **Factual Consistency**
   - Do claims in the executive summary match the subtopic findings?
   - Are insights properly supported by the cited sources?
   - Are there any unsupported claims?

2. **Citation Accuracy**
   - Are citations properly attributed to specific claims?
   - Do the cited URLs appear relevant to the claims they support?
   - Are there claims without citations that need them?

3. **Logical Coherence**
   - Are there contradictions between different sections?
   - Does the executive summary accurately represent the overall findings?
   - Is the flow of information logical?

4. **Completeness**
   - Are key findings from each subtopic represented in the synthesis?
   - Are important consensus points and conflicts captured?
   - Is anything significant missing?

After your review, decide:
- **APPROVED**: Report meets quality standards
- **REVISION_NEEDED**: Report has issues that should be fixed

{config["revision_threshold"]}

Return your review as JSON:
{{
  "decision": "APPROVED" or "REVISION_NEEDED",
  "overall_score": 1-10,
  "issues_found": [
    {{
      "category": "factual_consistency|citation_accuracy|logical_coherence|completeness",
      "severity": "minor|moderate|major",
      "description": "Description of the issue",
      "location": "Where in the report (e.g., 'executive_summary', 'subtopic: AI in diagnostics')",
      "suggestion": "How to fix it"
    }}
  ],
  "strengths": ["What the report does well"],
  "revision_instructions": "If REVISION_NEEDED, specific instructions for the Synthesizer to improve the report. Be specific about what needs to change."
}}"""

    def review(self, report: MultiAgentReport) -> CriticReview:
        """
        Review a synthesized report and provide feedback.

        Args:
            report: The MultiAgentReport to review

        Returns:
            CriticReview with decision, issues, and revision instructions
        """
        # Format report for review
        report_text = self._format_report_for_review(report)

        user_message = f"""Please review this research report:

{report_text}

Evaluate the report for factual consistency, citation accuracy, logical coherence, and completeness. Return your review as JSON."""

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

        # Convert issue dicts to CriticIssue objects
        issues = []
        for issue_data in result_data.get("issues_found", []):
            issues.append(CriticIssue(
                category=issue_data.get("category", "completeness"),
                severity=issue_data.get("severity", "moderate"),
                description=issue_data.get("description", ""),
                location=issue_data.get("location", ""),
                suggestion=issue_data.get("suggestion", "")
            ))

        # Build CriticReview
        review = CriticReview(
            decision=result_data.get("decision", "APPROVED"),
            overall_score=result_data.get("overall_score", 7),
            issues_found=issues,
            strengths=result_data.get("strengths", []),
            revision_instructions=result_data.get("revision_instructions", "")
        )

        return review

    def _format_report_for_review(self, report: MultiAgentReport) -> str:
        """Format the report for the critic to review."""
        sections = []

        sections.append(f"TOPIC: {report.topic}")
        sections.append(f"SUBTOPICS: {', '.join(report.subtopics)}")
        sections.append("")

        sections.append("=" * 50)
        sections.append("EXECUTIVE SUMMARY")
        sections.append("=" * 50)
        sections.append(report.executive_summary)
        sections.append("")

        if report.overall_insights:
            sections.append("=" * 50)
            sections.append("OVERALL KEY INSIGHTS")
            sections.append("=" * 50)
            for i, insight in enumerate(report.overall_insights, 1):
                sections.append(f"{i}. {insight.finding}")
                if insight.citations:
                    sections.append(f"   Citations: {', '.join(insight.citations)}")
            sections.append("")

        if report.consensus_points:
            sections.append("=" * 50)
            sections.append("CONSENSUS POINTS")
            sections.append("=" * 50)
            for point in report.consensus_points:
                sections.append(f"- {point}")
            sections.append("")

        sections.append("=" * 50)
        sections.append("SUBTOPIC FINDINGS")
        sections.append("=" * 50)
        for findings in report.subtopic_findings:
            sections.append(f"\n--- {findings.subtopic} ---")
            sections.append(f"Summary: {findings.summary}")
            if findings.key_insights:
                sections.append("Insights:")
                for insight in findings.key_insights:
                    sections.append(f"  - {insight.finding}")
                    if insight.citations:
                        sections.append(f"    Citations: {', '.join(insight.citations)}")
            sections.append("")

        if report.conflicts_and_gaps:
            sections.append("=" * 50)
            sections.append("CONFLICTS & GAPS")
            sections.append("=" * 50)
            sections.append(report.conflicts_and_gaps)
            sections.append("")

        if report.top_sources:
            sections.append("=" * 50)
            sections.append("TOP SOURCES")
            sections.append("=" * 50)
            for source in report.top_sources:
                sections.append(f"- {source.title}")
                sections.append(f"  URL: {source.url}")
                if source.why_matters:
                    sections.append(f"  Why it matters: {source.why_matters}")
            sections.append("")

        return "\n".join(sections)


def create_critic_agent(
    model: str = "gpt-4-turbo-preview",
    strictness: str = "balanced"
) -> CriticAgent:
    """
    Factory function to create a critic agent.

    Args:
        model: OpenAI model to use
        strictness: Review strictness (lenient/balanced/strict)

    Returns:
        Initialized CriticAgent instance
    """
    return CriticAgent(model=model, strictness=strictness)
