"""Gradio web interface for ScholarAI."""

import os
import sys
from pathlib import Path
from datetime import datetime
import gradio as gr
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from agents.research_agent import ResearchAgent
from agents.synthesizer_agent import SynthesizerAgent
from exporters.markdown_exporter import to_markdown
from exporters.json_exporter import to_json


def research_and_synthesize(
    topic: str,
    style: str = "Technical",
    tone: str = "Neutral",
    max_sources: int = 10,
    progress=gr.Progress()
):
    """
    Run the research pipeline with style and tone options.

    Args:
        topic: Research topic or question
        style: Output style (Technical/Layperson)
        tone: Output tone (Neutral/Advisory)
        max_sources: Maximum number of sources to fetch
        progress: Gradio progress tracker

    Returns:
        Tuple of (summary_html, tldr_text, evidence_html, sources_html, report_obj, markdown_str, json_str)
    """
    try:
        if not topic or not topic.strip():
            return (
                "<p style='color: red;'>Please enter a research topic.</p>",
                "",
                "",
                "",
                None,
                "",
                ""
            )

        # Update synthesizer prompt based on style and tone
        style_instruction = {
            "Technical": "Use technical language and domain-specific terminology. Assume the reader has expertise in the field.",
            "Layperson": "Use clear, simple language that anyone can understand. Avoid jargon and explain complex concepts."
        }

        tone_instruction = {
            "Neutral": "Present findings objectively without recommendations.",
            "Advisory": "Provide insights and recommendations based on the findings."
        }

        # Step 1: Research
        progress(0.1, desc="Searching...")
        research_agent = ResearchAgent(
            model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
            max_sources=max_sources
        )
        research_results = research_agent.research(topic)

        # Step 2: Synthesize with custom instructions
        progress(0.5, desc="Synthesizing...")
        synthesizer = SynthesizerAgent(
            model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        )

        # Modify the synthesizer's system prompt temporarily
        original_prompt = synthesizer.system_prompt
        custom_prompt = f"{original_prompt}\n\nSTYLE: {style_instruction[style]}\nTONE: {tone_instruction[tone]}"
        synthesizer.system_prompt = custom_prompt

        report = synthesizer.synthesize(
            topic=topic,
            sources=research_results['sources'],
            analysis=research_results.get('analysis')
        )

        # Restore original prompt
        synthesizer.system_prompt = original_prompt

        # Step 3: Generate outputs
        progress(0.9, desc="Finalizing...")

        # Summary tab - TL;DR
        import html
        escaped_tldr = html.escape(report.tldr)

        summary_html = f"""
        <div style='padding: 20px; background: #f8f9fa; border-radius: 8px;'>
            <h2 style='color: #2c3e50; margin-top: 0;'>Summary</h2>
            <p style='font-size: 16px; line-height: 1.6; color: #2c3e50;'>{escaped_tldr}</p>
            <hr style='margin: 20px 0;'>
            <p style='font-size: 14px;'>
                <strong style='color: #2c3e50;'>Generated:</strong> <span style='color: #7f8c8d;'>{report.metadata.get('timestamp', 'N/A')}</span><br>
                <strong style='color: #2c3e50;'>Sources Analyzed:</strong> <span style='color: #7f8c8d;'>{report.metadata.get('num_sources_analyzed', 'N/A')}</span><br>
                <strong style='color: #2c3e50;'>Key Findings:</strong> <span style='color: #7f8c8d;'>{len(report.key_findings)}</span>
            </p>
        </div>
        """

        # Evidence tab - Key findings
        findings_html = "<div style='padding: 20px;'>"
        findings_html += "<h2 style='color: #2c3e50;'>Key Findings</h2>"

        for i, finding in enumerate(report.key_findings, 1):
            escaped_finding = html.escape(finding.finding)
            findings_html += f"""
            <div style='margin-bottom: 30px; padding: 15px; background: #f8f9fa; border-left: 4px solid #3498db; border-radius: 4px;'>
                <h3 style='color: #2c3e50; margin-top: 0;'>{i}. {escaped_finding}</h3>
                <p style='margin: 10px 0 5px 0; color: #7f8c8d;'><strong>Citations:</strong></p>
                <ul style='margin: 5px 0; padding-left: 20px;'>
            """
            for citation in finding.citations:
                escaped_citation = html.escape(citation)
                findings_html += f"<li><a href='{escaped_citation}' target='_blank' style='color: #3498db;'>{escaped_citation}</a></li>"
            findings_html += "</ul></div>"

        # Add conflicts and caveats
        if report.conflicts_and_caveats:
            escaped_conflicts = html.escape(report.conflicts_and_caveats)
            findings_html += f"""
            <div style='margin-top: 30px; padding: 15px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px;'>
                <h3 style='color: #856404; margin-top: 0;'>⚠️ Conflicts & Caveats</h3>
                <p style='color: #856404;'>{escaped_conflicts}</p>
            </div>
            """

        findings_html += "</div>"

        # Sources tab
        sources_html = "<div style='padding: 20px;'>"
        sources_html += "<h2 style='color: #2c3e50;'>Top Sources</h2>"

        for i, source in enumerate(report.top_sources, 1):
            score_color = "#27ae60" if source.score and source.score > 0.95 else "#3498db"
            escaped_title = html.escape(source.title)
            escaped_url = html.escape(source.url)
            escaped_snippet = html.escape(source.snippet[:300])
            snippet_suffix = '...' if len(source.snippet) > 300 else ''

            why_matters_html = ""
            if source.why_matters:
                escaped_why_matters = html.escape(source.why_matters)
                why_matters_html = f"<p style='background: white; padding: 10px; border-radius: 4px; margin: 10px 0;'><strong style='color: #2c3e50;'>Why It Matters:</strong> <span style='color: #34495e;'>{escaped_why_matters}</span></p>"

            sources_html += f"""
            <div style='margin-bottom: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h3 style='color: #2c3e50; margin-top: 0;'>
                    {i}. <a href='{escaped_url}' target='_blank' style='color: #3498db; text-decoration: none;'>{escaped_title}</a>
                </h3>
                <p style='color: #7f8c8d; margin: 10px 0;'>
                    <strong style='color: {score_color};'>Relevance Score:</strong> {f"{source.score:.4f}" if source.score is not None else 'N/A'}
                </p>
                {why_matters_html}
                <blockquote style='border-left: 3px solid #bdc3c7; padding-left: 15px; margin: 15px 0; color: #7f8c8d; font-style: italic;'>
                    {escaped_snippet}{snippet_suffix}
                </blockquote>
            </div>
            """

        sources_html += "</div>"

        # Generate export formats
        markdown_str = to_markdown(report)
        json_str = to_json(report, indent=2)

        progress(1.0, desc="Complete!")

        return summary_html, report.tldr, findings_html, sources_html, report, markdown_str, json_str

    except Exception as e:
        error_html = f"""
        <div style='padding: 20px; background: #fee; border-left: 4px solid #e74c3c; border-radius: 4px;'>
            <h3 style='color: #c0392b; margin-top: 0;'>❌ Error</h3>
            <p style='color: #c0392b;'>{str(e)}</p>
        </div>
        """
        return error_html, "", "", "", None, "", ""


def create_app():
    """Create and configure the Gradio application."""

    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .output-html {
        max-height: 600px;
        overflow-y: auto;
    }
    """

    with gr.Blocks(css=custom_css, title="ScholarAI Research Assistant") as app:
        # Header
        gr.Markdown("""
        # 🎓 ScholarAI Research Assistant
        ### AI-Powered Academic Research & Synthesis

        Enter a research topic and get a comprehensive report with curated sources, key findings, and citations.
        """)

        # State for storing report data
        report_state = gr.State(None)
        markdown_state = gr.State("")
        json_state = gr.State("")

        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                topic_input = gr.Textbox(
                    label="Research Topic",
                    placeholder="e.g., 'recent advances in quantum computing' or 'impact of climate change on agriculture'",
                    lines=2
                )

                with gr.Row():
                    style_dropdown = gr.Dropdown(
                        choices=["Technical", "Layperson"],
                        value="Technical",
                        label="Writing Style",
                        info="Choose the complexity level of the report"
                    )
                    tone_dropdown = gr.Dropdown(
                        choices=["Neutral", "Advisory"],
                        value="Neutral",
                        label="Tone",
                        info="Objective presentation or with recommendations"
                    )

                max_sources_slider = gr.Slider(
                    minimum=5,
                    maximum=20,
                    value=10,
                    step=1,
                    label="Maximum Sources",
                    info="Number of sources to analyze"
                )

                submit_btn = gr.Button("🔍 Research", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("""
                ### ℹ️ How to Use

                1. **Enter your topic** - Be specific for better results
                2. **Choose style & tone** - Customize the output
                3. **Click Research** - Wait for the AI to analyze sources
                4. **Explore results** - View summary, findings, and sources
                5. **Export** - Download as Markdown or JSON

                ### 📋 Tips
                - Use clear, specific research questions
                - Technical style for academic work
                - Layperson style for general audience
                """)

        # Output section with tabs
        with gr.Tabs() as tabs:
            with gr.Tab("📊 Summary"):
                summary_output = gr.HTML(label="Summary")
                tldr_text = gr.Textbox(label="TL;DR (for copying)", visible=False)
                with gr.Row():
                    copy_tldr_btn = gr.Button("📋 Copy Summary", size="sm")
                    copy_status = gr.Textbox(label="", visible=False, elem_id="copy_status")

            with gr.Tab("🔍 Key Findings"):
                evidence_output = gr.HTML(label="Evidence & Findings")

            with gr.Tab("📚 Sources"):
                sources_output = gr.HTML(label="Top Sources")

        # Export section
        gr.Markdown("### 💾 Export Options")
        with gr.Row():
            download_md_btn = gr.Button("⬇️ Download Markdown", variant="secondary")
            download_json_btn = gr.Button("⬇️ Download JSON", variant="secondary")

        # File outputs for downloads
        md_file_output = gr.File(label="Markdown Download", visible=True)
        json_file_output = gr.File(label="JSON Download", visible=True)

        with gr.Accordion("📄 Preview Exports", open=False):
            with gr.Tab("Markdown"):
                markdown_preview = gr.Code(
                    label="Markdown Content",
                    language="markdown",
                    lines=15,
                    interactive=False
                )
            with gr.Tab("JSON"):
                json_preview = gr.Code(
                    label="JSON Content",
                    language="json",
                    lines=15,
                    interactive=False
                )

        # Event handlers
        submit_btn.click(
            fn=research_and_synthesize,
            inputs=[topic_input, style_dropdown, tone_dropdown, max_sources_slider],
            outputs=[
                summary_output,
                tldr_text,
                evidence_output,
                sources_output,
                report_state,
                markdown_state,
                json_state
            ]
        ).then(
            fn=lambda md, js: (md, js),
            inputs=[markdown_state, json_state],
            outputs=[markdown_preview, json_preview]
        )

        # Copy TL;DR button - copy from tldr_text field
        def copy_to_clipboard(tldr):
            """Return the TLDR for copying."""
            return tldr if tldr else "No summary available yet."

        copy_tldr_btn.click(
            fn=copy_to_clipboard,
            inputs=[tldr_text],
            outputs=[copy_status],
            js="(tldr) => { navigator.clipboard.writeText(tldr); return 'Copied!'; }"
        )

        # Download functions
        def save_markdown(md_content, topic):
            """Save markdown to temporary file for download."""
            if not md_content:
                return None
            import tempfile
            safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in topic)[:30]
            safe_name = safe_name if safe_name else "report"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
                f.write(md_content)
                return f.name

        def save_json(json_content, topic):
            """Save JSON to temporary file for download."""
            if not json_content:
                return None
            import tempfile
            safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in topic)[:30]
            safe_name = safe_name if safe_name else "report"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                f.write(json_content)
                return f.name

        # Download buttons
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
            <p>Powered by OpenAI GPT-4 & Tavily Search | ScholarAI © 2025</p>
            <p><em>Part of the SuperDataScience Community Project</em></p>
        </div>
        """)

    return app


def main():
    """Launch the Gradio interface."""
    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in .env file")
        sys.exit(1)

    if not os.getenv("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY not found in .env file")
        sys.exit(1)

    app = create_app()

    # Launch the app
    # HF Spaces provides its own public URL, so share=False
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # HF Spaces handles public URLs
        show_error=True
    )


if __name__ == "__main__":
    main()
