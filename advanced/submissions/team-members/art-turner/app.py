"""Gradio web interface for ScholarAI Advanced - Multi-Agent Research System."""

import os
import sys
import html
import tempfile
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from agents.orchestrator import MultiAgentOrchestrator
from exporters.markdown_exporter import to_markdown
from exporters.json_exporter import to_json


def run_multi_agent_research(
    topic: str,
    style: str = "Technical",
    tone: str = "Neutral",
    num_subtopics: int = 3,
    max_sources: int = 8,
    enable_critic: bool = True,
    critic_strictness: str = "balanced",
    max_revisions: int = 2,
    progress=gr.Progress()
):
    """
    Run the complete multi-agent research pipeline.

    Args:
        topic: Research topic
        style: Writing style (Technical/Layperson/Business)
        tone: Tone (Neutral/Advisory)
        num_subtopics: Number of subtopics to research
        max_sources: Max sources per subtopic
        enable_critic: Whether to enable critic review
        critic_strictness: Critic strictness level
        max_revisions: Max revision iterations
        progress: Gradio progress tracker

    Returns:
        Tuple of outputs for the UI
    """
    try:
        if not topic or not topic.strip():
            error_html = "<p style='color: red;'>Please enter a research topic.</p>"
            return error_html, "", "", "", "", None, "", ""

        # Create orchestrator
        orchestrator = MultiAgentOrchestrator(
            model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
            num_subtopics=num_subtopics,
            max_sources_per_subtopic=max_sources,
            max_workers=min(num_subtopics, 3),
            enable_critic=enable_critic,
            max_revisions=max_revisions,
            critic_strictness=critic_strictness
        )

        # Progress callback for Gradio
        def progress_callback(prog: float, message: str):
            progress(prog, desc=message)

        # Run the multi-agent workflow
        report = orchestrator.run(
            topic=topic,
            style=style,
            tone=tone,
            progress_callback=progress_callback
        )

        # Generate HTML outputs for each tab
        summary_html = generate_summary_html(report)
        findings_html = generate_findings_html(report)
        sources_html = generate_sources_html(report)
        critic_html = generate_critic_html(report)

        # Generate export formats
        markdown_str = to_markdown(report)
        json_str = to_json(report, indent=2)

        return (
            summary_html,
            report.executive_summary,
            findings_html,
            sources_html,
            critic_html,
            report,
            markdown_str,
            json_str
        )

    except Exception as e:
        error_html = f"""
        <div style='padding: 20px; background: #fee; border-left: 4px solid #e74c3c; border-radius: 4px;'>
            <h3 style='color: #c0392b; margin-top: 0;'>Error</h3>
            <p style='color: #c0392b;'>{html.escape(str(e))}</p>
        </div>
        """
        return error_html, "", "", "", "", None, "", ""


def generate_critic_html(report) -> str:
    """Generate HTML for the critic review tab."""
    if not report.critic_review:
        return """
        <div style='padding: 20px; background: #f8f9fa; border-radius: 8px;'>
            <h2 style='color: #2c3e50; margin-top: 0;'>Critic Review</h2>
            <p style='color: #7f8c8d;'>Critic review was not enabled for this report.</p>
        </div>
        """

    review = report.critic_review

    # Decision badge
    decision_color = "#28a745" if review.decision == "APPROVED" else "#ffc107"
    decision_icon = "&#10004;" if review.decision == "APPROVED" else "&#9888;"

    # Score color
    score = review.overall_score
    if score >= 8:
        score_color = "#28a745"
    elif score >= 6:
        score_color = "#ffc107"
    else:
        score_color = "#dc3545"

    # Strengths
    strengths_html = ""
    if review.strengths:
        strength_items = "".join(
            f"<li style='color: #155724;'>{html.escape(s)}</li>"
            for s in review.strengths
        )
        strengths_html = f"""
        <div style='margin-top: 20px; padding: 15px; background: #d4edda; border-left: 4px solid #28a745; border-radius: 4px;'>
            <h4 style='color: #155724; margin-top: 0;'>Strengths</h4>
            <ul style='margin: 0; padding-left: 20px;'>{strength_items}</ul>
        </div>
        """

    # Issues
    issues_html = ""
    if review.issues_found:
        issues_items = ""
        for issue in review.issues_found:
            severity = issue.severity
            severity_colors = {"minor": "#17a2b8", "moderate": "#ffc107", "major": "#dc3545"}
            sev_color = severity_colors.get(severity, "#6c757d")

            issues_items += f"""
            <div style='margin-bottom: 15px; padding: 10px; background: #f8f9fa; border-radius: 4px; border: 1px solid #dee2e6;'>
                <p style='margin: 0 0 5px 0;'>
                    <span style='background: {sev_color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px;'>{severity.upper()}</span>
                    <span style='color: #495057; font-size: 12px; margin-left: 8px;'>{html.escape(issue.category)}</span>
                </p>
                <p style='margin: 5px 0;'><strong style='color: #212529;'>{html.escape(issue.description)}</strong></p>
                <p style='margin: 5px 0; font-size: 13px;'><strong style='color: #495057;'>Location:</strong> <span style='color: #495057;'>{html.escape(issue.location)}</span></p>
                <p style='margin: 5px 0; font-size: 13px;'><strong style='color: #155724;'>Suggestion:</strong> <span style='color: #155724;'>{html.escape(issue.suggestion)}</span></p>
            </div>
            """

        issues_html = f"""
        <div style='margin-top: 20px;'>
            <h4 style='color: #2c3e50;'>Issues Found ({len(review.issues_found)})</h4>
            {issues_items}
        </div>
        """

    # Revision info
    revision_html = ""
    if report.revision_count > 0:
        revision_html = f"""
        <p style='margin-top: 10px; color: #6c757d;'>
            <strong>Revisions made:</strong> {report.revision_count}
        </p>
        """

    return f"""
    <div style='padding: 20px; background: #f8f9fa; border-radius: 8px;'>
        <h2 style='color: #2c3e50; margin-top: 0;'>Critic Review</h2>

        <div style='display: flex; gap: 20px; margin-bottom: 20px;'>
            <div style='padding: 15px; background: white; border-radius: 8px; flex: 1; text-align: center;'>
                <p style='margin: 0; font-size: 24px; color: {decision_color};'>{decision_icon}</p>
                <p style='margin: 5px 0 0 0; font-weight: bold; color: {decision_color};'>{review.decision}</p>
            </div>
            <div style='padding: 15px; background: white; border-radius: 8px; flex: 1; text-align: center;'>
                <p style='margin: 0; font-size: 24px; color: {score_color};'>{score}/10</p>
                <p style='margin: 5px 0 0 0; color: #6c757d;'>Quality Score</p>
            </div>
            <div style='padding: 15px; background: white; border-radius: 8px; flex: 1; text-align: center;'>
                <p style='margin: 0; font-size: 24px; color: #6c757d;'>{review.iteration}</p>
                <p style='margin: 5px 0 0 0; color: #6c757d;'>Review Iteration</p>
            </div>
        </div>

        {strengths_html}
        {issues_html}
        {revision_html}
    </div>
    """


def generate_summary_html(report) -> str:
    """Generate HTML for the summary tab."""
    escaped_summary = html.escape(report.executive_summary)

    # Consensus points
    consensus_html = ""
    if report.consensus_points:
        consensus_items = "".join(
            f"<li style='color: #2c3e50;'>{html.escape(point)}</li>"
            for point in report.consensus_points
        )
        consensus_html = f"""
        <div style='margin-top: 20px; padding: 15px; background: #d4edda; border-left: 4px solid #28a745; border-radius: 4px;'>
            <h4 style='color: #155724; margin-top: 0;'>Consensus Points</h4>
            <ul style='margin: 0; padding-left: 20px;'>{consensus_items}</ul>
        </div>
        """

    # Overall insights
    insights_html = ""
    if report.overall_insights:
        insights_items = ""
        for insight in report.overall_insights[:5]:
            escaped_finding = html.escape(insight.finding)
            citations_text = ""
            if insight.citations:
                citations_text = f"<br><small style='color: #6c757d;'>Sources: {len(insight.citations)}</small>"
            insights_items += f"<li style='margin-bottom: 10px;'><strong style='color: #2c3e50;'>{escaped_finding}</strong>{citations_text}</li>"

        insights_html = f"""
        <div style='margin-top: 20px;'>
            <h4 style='color: #2c3e50;'>Key Insights</h4>
            <ul style='padding-left: 20px;'>{insights_items}</ul>
        </div>
        """

    return f"""
    <div style='padding: 20px; background: #f8f9fa; border-radius: 8px;'>
        <h2 style='color: #2c3e50; margin-top: 0;'>Executive Summary</h2>
        <p style='font-size: 16px; line-height: 1.6; color: #2c3e50;'>{escaped_summary}</p>
        {consensus_html}
        {insights_html}
        <hr style='margin: 20px 0;'>
        <p style='font-size: 14px;'>
            <strong style='color: #2c3e50;'>Subtopics:</strong> <span style='color: #7f8c8d;'>{len(report.subtopics)}</span><br>
            <strong style='color: #2c3e50;'>Total Sources:</strong> <span style='color: #7f8c8d;'>{len(report.all_sources)}</span>
        </p>
    </div>
    """


def generate_findings_html(report) -> str:
    """Generate HTML for the per-researcher findings tab."""
    html_parts = ["<div style='padding: 20px;'>"]
    html_parts.append("<h2 style='color: #2c3e50;'>Research Findings by Subtopic</h2>")

    for i, findings in enumerate(report.subtopic_findings, 1):
        escaped_subtopic = html.escape(findings.subtopic)
        escaped_summary = html.escape(findings.summary)

        # Key insights for this subtopic
        insights_html = ""
        if findings.key_insights:
            insights_items = ""
            for insight in findings.key_insights:
                escaped_finding = html.escape(insight.finding)
                citations_html = ""
                if insight.citations:
                    citations_links = "".join(
                        f"<a href='{html.escape(url)}' target='_blank' style='color: #3498db; font-size: 12px;'>{html.escape(url[:50])}...</a><br>"
                        for url in insight.citations[:2]
                    )
                    citations_html = f"<div style='margin-top: 5px;'>{citations_links}</div>"
                insights_items += f"""
                <div style='margin-bottom: 15px; padding: 10px; background: white; border-radius: 4px;'>
                    <p style='margin: 0; color: #2c3e50;'>{escaped_finding}</p>
                    {citations_html}
                </div>
                """
            insights_html = f"<div style='margin-top: 15px;'><strong style='color: #34495e;'>Key Insights:</strong>{insights_items}</div>"

        # Sources count
        sources_count = len(findings.sources) if findings.sources else 0

        # Researcher notes
        notes_html = ""
        if findings.researcher_notes:
            escaped_notes = html.escape(findings.researcher_notes)
            notes_html = f"<p style='font-style: italic; color: #7f8c8d; margin-top: 10px;'><strong>Notes:</strong> {escaped_notes}</p>"

        html_parts.append(f"""
        <div style='margin-bottom: 30px; padding: 20px; background: #f8f9fa; border-left: 4px solid #3498db; border-radius: 4px;'>
            <h3 style='color: #2c3e50; margin-top: 0;'>
                <span style='background: #3498db; color: white; padding: 2px 8px; border-radius: 4px; font-size: 14px;'>Subtopic {i}</span>
                {escaped_subtopic}
            </h3>
            <p style='color: #34495e; line-height: 1.6;'><strong>Summary:</strong> {escaped_summary}</p>
            {insights_html}
            {notes_html}
            <p style='margin-top: 15px; color: #7f8c8d; font-size: 12px;'>
                Sources analyzed: {sources_count}
            </p>
        </div>
        """)

    # Conflicts and gaps
    if report.conflicts_and_gaps:
        escaped_conflicts = html.escape(report.conflicts_and_gaps)
        html_parts.append(f"""
        <div style='margin-top: 30px; padding: 20px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px;'>
            <h3 style='color: #856404; margin-top: 0;'>Conflicts & Gaps in Literature</h3>
            <p style='color: #856404;'>{escaped_conflicts}</p>
        </div>
        """)

    html_parts.append("</div>")
    return "".join(html_parts)


def generate_sources_html(report) -> str:
    """Generate HTML for the sources tab."""
    html_parts = ["<div style='padding: 20px;'>"]

    # Top sources
    if report.top_sources:
        html_parts.append("<h2 style='color: #2c3e50;'>Top Sources</h2>")

        for i, source in enumerate(report.top_sources, 1):
            score_color = "#27ae60" if source.score and source.score > 0.95 else "#3498db"
            escaped_title = html.escape(source.title)
            escaped_url = html.escape(source.url)
            escaped_snippet = html.escape(source.snippet[:300]) if source.snippet else ""
            snippet_suffix = '...' if source.snippet and len(source.snippet) > 300 else ''

            why_matters_html = ""
            if source.why_matters:
                escaped_why = html.escape(source.why_matters)
                why_matters_html = f"""
                <p style='background: white; padding: 10px; border-radius: 4px; margin: 10px 0;'>
                    <strong style='color: #2c3e50;'>Why It Matters:</strong>
                    <span style='color: #34495e;'>{escaped_why}</span>
                </p>
                """

            html_parts.append(f"""
            <div style='margin-bottom: 25px; padding: 20px; background: #f8f9fa; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h3 style='color: #2c3e50; margin-top: 0;'>
                    {i}. <a href='{escaped_url}' target='_blank' style='color: #3498db; text-decoration: none;'>{escaped_title}</a>
                </h3>
                <p style='color: #7f8c8d; margin: 10px 0;'>
                    <strong style='color: {score_color};'>Relevance:</strong> {f"{source.score:.4f}" if source.score else 'N/A'}
                </p>
                {why_matters_html}
                <blockquote style='border-left: 3px solid #bdc3c7; padding-left: 15px; margin: 15px 0; color: #7f8c8d; font-style: italic;'>
                    {escaped_snippet}{snippet_suffix}
                </blockquote>
            </div>
            """)

    # All sources by subtopic
    html_parts.append("<h2 style='color: #2c3e50; margin-top: 40px;'>All Sources by Subtopic</h2>")

    for findings in report.subtopic_findings:
        if findings.sources:
            escaped_subtopic = html.escape(findings.subtopic)
            html_parts.append(f"""
            <div style='margin-bottom: 20px;'>
                <h4 style='color: #34495e;'>{escaped_subtopic}</h4>
                <ul style='padding-left: 20px;'>
            """)

            for source in findings.sources[:5]:
                escaped_title = html.escape(source.title)
                escaped_url = html.escape(source.url)
                score_text = f" ({source.score:.3f})" if source.score else ""
                html_parts.append(f"""
                <li style='margin-bottom: 8px;'>
                    <a href='{escaped_url}' target='_blank' style='color: #3498db;'>{escaped_title}</a>
                    <span style='color: #95a5a6; font-size: 12px;'>{score_text}</span>
                </li>
                """)

            html_parts.append("</ul></div>")

    html_parts.append("</div>")
    return "".join(html_parts)


def create_app():
    """Create and configure the Gradio application."""

    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .output-html {
        max-height: 600px;
        overflow-y: auto;
    }
    """

    with gr.Blocks(css=custom_css, title="ScholarAI Advanced - Multi-Agent Research") as app:
        # Header
        gr.Markdown("""
        # ScholarAI Advanced - Multi-Agent Research System
        ### AI-Powered Collaborative Research with Specialized Agents

        Enter a research topic and watch multiple AI agents collaborate to produce a comprehensive research report.
        The system will:
        1. **Split** your topic into focused subtopics
        2. **Research** each subtopic in parallel with specialized agents
        3. **Synthesize** findings into a unified report
        4. **Review** with a Critic Agent that can request revisions
        """)

        # State
        report_state = gr.State(None)
        markdown_state = gr.State("")
        json_state = gr.State("")

        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                topic_input = gr.Textbox(
                    label="Research Topic",
                    placeholder="e.g., 'Impact of artificial intelligence on healthcare' or 'Renewable energy storage solutions'",
                    lines=2
                )

                with gr.Row():
                    style_dropdown = gr.Dropdown(
                        choices=["Technical", "Layperson", "Business"],
                        value="Technical",
                        label="Writing Style",
                        info="Complexity level of the report"
                    )
                    tone_dropdown = gr.Dropdown(
                        choices=["Neutral", "Advisory"],
                        value="Neutral",
                        label="Tone",
                        info="Objective or with recommendations"
                    )

                with gr.Row():
                    num_subtopics_slider = gr.Slider(
                        minimum=2,
                        maximum=4,
                        value=3,
                        step=1,
                        label="Number of Subtopics",
                        info="How many aspects to research"
                    )
                    max_sources_slider = gr.Slider(
                        minimum=5,
                        maximum=15,
                        value=8,
                        step=1,
                        label="Sources per Subtopic",
                        info="Sources each researcher finds"
                    )

                # Critic settings
                with gr.Accordion("Critic Agent Settings", open=False):
                    enable_critic_checkbox = gr.Checkbox(
                        label="Enable Critic Review",
                        value=True,
                        info="Critic reviews and requests revisions"
                    )
                    with gr.Row():
                        critic_strictness_dropdown = gr.Dropdown(
                            choices=["lenient", "balanced", "strict"],
                            value="balanced",
                            label="Strictness",
                            info="How strict the critic reviews"
                        )
                        max_revisions_slider = gr.Slider(
                            minimum=1,
                            maximum=3,
                            value=2,
                            step=1,
                            label="Max Revisions",
                            info="Maximum revision iterations"
                        )

                submit_btn = gr.Button("Start Multi-Agent Research", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("""
                ### How It Works

                **1. Topic Splitter Agent**
                Analyzes your topic and identifies key aspects

                **2. Researcher Agents**
                Multiple agents research subtopics in parallel

                **3. Synthesizer Agent**
                Merges all findings into a coherent report

                **4. Critic Agent**
                Reviews quality and requests revisions

                ### Tips
                - Broad topics work best (they can be split)
                - Technical style for academic work
                - Enable Critic for higher quality reports
                """)

        # Output tabs
        with gr.Tabs():
            with gr.Tab("Summary"):
                summary_output = gr.HTML(label="Executive Summary")
                executive_text = gr.Textbox(label="Summary Text", visible=False)

            with gr.Tab("Research Findings"):
                findings_output = gr.HTML(label="Per-Subtopic Findings")

            with gr.Tab("Sources"):
                sources_output = gr.HTML(label="All Sources")

            with gr.Tab("Critic Review"):
                critic_output = gr.HTML(label="Critic Review")

        # Export section
        gr.Markdown("### Export Options")
        with gr.Row():
            download_md_btn = gr.Button("Download Markdown", variant="secondary")
            download_json_btn = gr.Button("Download JSON", variant="secondary")

        md_file_output = gr.File(label="Markdown Download", visible=True)
        json_file_output = gr.File(label="JSON Download", visible=True)

        with gr.Accordion("Preview Exports", open=False):
            with gr.Tab("Markdown"):
                markdown_preview = gr.Code(
                    label="Markdown",
                    language="markdown",
                    lines=15,
                    interactive=False
                )
            with gr.Tab("JSON"):
                json_preview = gr.Code(
                    label="JSON",
                    language="json",
                    lines=15,
                    interactive=False
                )

        # Event handlers
        submit_btn.click(
            fn=run_multi_agent_research,
            inputs=[
                topic_input,
                style_dropdown,
                tone_dropdown,
                num_subtopics_slider,
                max_sources_slider,
                enable_critic_checkbox,
                critic_strictness_dropdown,
                max_revisions_slider
            ],
            outputs=[
                summary_output,
                executive_text,
                findings_output,
                sources_output,
                critic_output,
                report_state,
                markdown_state,
                json_state
            ]
        ).then(
            fn=lambda md, js: (md, js),
            inputs=[markdown_state, json_state],
            outputs=[markdown_preview, json_preview]
        )

        # Download functions
        def save_markdown(md_content, topic):
            if not md_content:
                return None
            safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in topic)[:30]
            safe_name = safe_name if safe_name else "report"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
                f.write(md_content)
                return f.name

        def save_json(json_content, topic):
            if not json_content:
                return None
            safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in topic)[:30]
            safe_name = safe_name if safe_name else "report"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                f.write(json_content)
                return f.name

        download_md_btn.click(
            fn=save_markdown,
            inputs=[markdown_state, topic_input],
            outputs=[md_file_output]
        )

        download_json_btn.click(
            fn=save_json,
            inputs=[json_state, topic_input],
            outputs=[json_file_output]
        )

        # Footer
        gr.Markdown("""
        ---
        <div style='text-align: center; color: #7f8c8d;'>
            <p>Powered by OpenAI GPT-4 & Tavily Search | ScholarAI Advanced</p>
            <p><em>Multi-Agent Research System - SuperDataScience Community Project</em></p>
        </div>
        """)

    return app


def main():
    """Launch the Gradio interface."""
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment")
        sys.exit(1)

    if not os.getenv("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY not found in environment")
        sys.exit(1)

    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
