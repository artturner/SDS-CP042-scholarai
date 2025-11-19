"""Multi-Agent Orchestrator for coordinating the research workflow.

This module coordinates the entire multi-agent research pipeline:
1. Topic Splitter breaks down the main topic
2. Researcher agents run in parallel on subtopics
3. Synthesizer merges all findings
4. Critic reviews and requests revisions (self-critique loop)
"""

import os
from typing import List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from .topic_splitter import TopicSplitterAgent
from .researcher_agent import ResearcherAgent
from .synthesizer_agent import SynthesizerAgent
from .critic_agent import CriticAgent
from models.report import Subtopic, SubtopicFindings, MultiAgentReport


class MultiAgentOrchestrator:
    """
    Orchestrates the multi-agent research workflow.

    This class coordinates:
    - Topic Splitter: Breaks topic into subtopics
    - Researcher Agents: Research each subtopic in parallel
    - Synthesizer Agent: Merges findings into final report
    - Critic Agent: Reviews and requests revisions (self-critique loop)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        num_subtopics: int = 3,
        max_sources_per_subtopic: int = 8,
        max_workers: int = 3,
        enable_critic: bool = True,
        max_revisions: int = 2,
        critic_strictness: str = "balanced",
    ):
        """
        Initialize the Multi-Agent Orchestrator.

        Args:
            api_key: OpenAI API key. If not provided, uses environment variable.
            model: OpenAI model to use for all agents
            num_subtopics: Number of subtopics to generate (2-4)
            max_sources_per_subtopic: Max sources each researcher should fetch
            max_workers: Max parallel researcher agents
            enable_critic: Whether to enable the Critic Agent review loop
            max_revisions: Maximum number of revision iterations
            critic_strictness: Critic strictness level (lenient/balanced/strict)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.model = model
        self.num_subtopics = num_subtopics
        self.max_sources_per_subtopic = max_sources_per_subtopic
        self.max_workers = max_workers
        self.enable_critic = enable_critic
        self.max_revisions = max_revisions
        self.critic_strictness = critic_strictness

        # Initialize agents
        self.topic_splitter = TopicSplitterAgent(
            api_key=self.api_key,
            model=self.model,
            num_subtopics=self.num_subtopics
        )

        self.synthesizer = SynthesizerAgent(
            api_key=self.api_key,
            model=self.model
        )

        # Initialize critic if enabled
        self.critic = None
        if self.enable_critic:
            self.critic = CriticAgent(
                api_key=self.api_key,
                model=self.model,
                strictness=self.critic_strictness
            )

    def run(
        self,
        topic: str,
        style: str = "Technical",
        tone: str = "Neutral",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> MultiAgentReport:
        """
        Run the complete multi-agent research workflow.

        Args:
            topic: The main research topic
            style: Writing style for the report
            tone: Tone for the report
            progress_callback: Optional callback for progress updates (progress, message)

        Returns:
            MultiAgentReport with synthesized findings
        """
        def update_progress(progress: float, message: str):
            if progress_callback:
                progress_callback(progress, message)

        # Step 1: Split topic into subtopics
        update_progress(0.1, "Analyzing topic and generating subtopics...")
        analysis = self.topic_splitter.analyze_topic(topic)
        subtopics = analysis["subtopics"]

        if not subtopics:
            raise ValueError("Topic splitter failed to generate subtopics")

        update_progress(0.2, f"Generated {len(subtopics)} subtopics. Starting research...")

        # Step 2: Research subtopics in parallel
        subtopic_findings = self._research_in_parallel(
            topic=topic,
            subtopics=subtopics,
            progress_callback=progress_callback
        )

        # Step 3: Synthesize findings
        update_progress(0.7, "Synthesizing findings into final report...")
        report = self.synthesizer.synthesize(
            topic=topic,
            subtopic_findings=subtopic_findings,
            style=style,
            tone=tone
        )

        # Step 4: Critic review and revision loop
        if self.enable_critic and self.critic:
            report = self._run_critic_loop(
                report=report,
                style=style,
                tone=tone,
                progress_callback=progress_callback
            )

        update_progress(1.0, "Research complete!")

        return report

    def _run_critic_loop(
        self,
        report: MultiAgentReport,
        style: str,
        tone: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> MultiAgentReport:
        """
        Run the critic review and revision loop.

        Args:
            report: Initial synthesized report
            style: Writing style
            tone: Tone
            progress_callback: Progress callback

        Returns:
            Final report after critic review (and possible revisions)
        """
        def update_progress(progress: float, message: str):
            if progress_callback:
                progress_callback(progress, message)

        iteration = 0
        current_report = report

        while iteration < self.max_revisions:
            iteration += 1
            update_progress(
                0.8 + (0.15 * iteration / (self.max_revisions + 1)),
                f"Critic reviewing report (iteration {iteration})..."
            )

            # Get critic review
            review = self.critic.review(current_report)
            review.iteration = iteration

            if review.decision == "APPROVED":
                # Report approved - attach review and return
                current_report.critic_review = review
                update_progress(
                    0.95,
                    f"Report approved by critic (score: {review.overall_score}/10)"
                )
                break

            elif review.decision == "REVISION_NEEDED":
                if iteration >= self.max_revisions:
                    # Max revisions reached - attach final review and return
                    current_report.critic_review = review
                    update_progress(
                        0.95,
                        f"Max revisions reached. Final score: {review.overall_score}/10"
                    )
                    break

                # Request revision from synthesizer
                update_progress(
                    0.8 + (0.15 * iteration / (self.max_revisions + 1)),
                    f"Revising report based on critic feedback..."
                )

                current_report = self.synthesizer.revise(
                    report=current_report,
                    revision_instructions=review.revision_instructions,
                    style=style,
                    tone=tone
                )

        return current_report

    def _research_in_parallel(
        self,
        topic: str,
        subtopics: List[Subtopic],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[SubtopicFindings]:
        """
        Run researcher agents in parallel on subtopics.

        Args:
            topic: Main topic for context
            subtopics: List of subtopics to research
            progress_callback: Optional progress callback

        Returns:
            List of SubtopicFindings from all researchers
        """
        findings = []

        def research_subtopic(subtopic: Subtopic, agent_id: str) -> SubtopicFindings:
            """Research a single subtopic."""
            researcher = ResearcherAgent(
                api_key=self.api_key,
                model=self.model,
                max_sources=self.max_sources_per_subtopic,
                agent_id=agent_id
            )
            return researcher.research_subtopic(subtopic, main_topic=topic)

        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all research tasks
            future_to_subtopic = {}
            for i, subtopic in enumerate(subtopics):
                agent_id = f"Researcher {i + 1}"
                future = executor.submit(research_subtopic, subtopic, agent_id)
                future_to_subtopic[future] = (i, subtopic)

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_subtopic):
                idx, subtopic = future_to_subtopic[future]
                try:
                    result = future.result()
                    findings.append((idx, result))
                    completed += 1

                    if progress_callback:
                        progress = 0.2 + (0.6 * completed / len(subtopics))
                        progress_callback(
                            progress,
                            f"Completed research on: {subtopic.name}"
                        )

                except Exception as e:
                    # Create error finding for failed research
                    error_finding = SubtopicFindings(
                        subtopic=subtopic.name,
                        summary=f"Research failed: {str(e)}",
                        key_insights=[],
                        sources=[],
                        researcher_notes=f"Error during research: {str(e)}",
                        metadata={"error": str(e)}
                    )
                    findings.append((idx, error_finding))
                    completed += 1

        # Sort by original index to maintain order
        findings.sort(key=lambda x: x[0])
        return [f for _, f in findings]

    def run_sequential(
        self,
        topic: str,
        style: str = "Technical",
        tone: str = "Neutral",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> MultiAgentReport:
        """
        Run the workflow sequentially (useful for debugging).

        Args:
            topic: The main research topic
            style: Writing style
            tone: Tone
            progress_callback: Optional progress callback

        Returns:
            MultiAgentReport
        """
        def update_progress(progress: float, message: str):
            if progress_callback:
                progress_callback(progress, message)

        # Step 1: Split topic
        update_progress(0.1, "Analyzing topic...")
        analysis = self.topic_splitter.analyze_topic(topic)
        subtopics = analysis["subtopics"]

        # Step 2: Research sequentially
        subtopic_findings = []
        for i, subtopic in enumerate(subtopics):
            progress = 0.2 + (0.6 * i / len(subtopics))
            update_progress(progress, f"Researching: {subtopic.name}")

            researcher = ResearcherAgent(
                api_key=self.api_key,
                model=self.model,
                max_sources=self.max_sources_per_subtopic,
                agent_id=f"Researcher {i + 1}"
            )
            findings = researcher.research_subtopic(subtopic, main_topic=topic)
            subtopic_findings.append(findings)

        # Step 3: Synthesize
        update_progress(0.8, "Synthesizing...")
        report = self.synthesizer.synthesize(
            topic=topic,
            subtopic_findings=subtopic_findings,
            style=style,
            tone=tone
        )

        update_progress(1.0, "Complete!")

        return report


def create_orchestrator(
    model: str = "gpt-4-turbo-preview",
    num_subtopics: int = 3,
    max_sources_per_subtopic: int = 8,
    max_workers: int = 3,
    enable_critic: bool = True,
    max_revisions: int = 2,
    critic_strictness: str = "balanced"
) -> MultiAgentOrchestrator:
    """
    Factory function to create an orchestrator.

    Args:
        model: OpenAI model to use
        num_subtopics: Number of subtopics to generate
        max_sources_per_subtopic: Sources per researcher
        max_workers: Parallel workers
        enable_critic: Enable the Critic Agent
        max_revisions: Max revision iterations
        critic_strictness: Critic strictness level

    Returns:
        Initialized MultiAgentOrchestrator
    """
    return MultiAgentOrchestrator(
        model=model,
        num_subtopics=num_subtopics,
        max_sources_per_subtopic=max_sources_per_subtopic,
        max_workers=max_workers,
        enable_critic=enable_critic,
        max_revisions=max_revisions,
        critic_strictness=critic_strictness
    )
