"""CLI interface for ScholarAI research assistant."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.research_agent import ResearchAgent
from agents.synthesizer_agent import SynthesizerAgent
from exporters.markdown_exporter import export_to_markdown
from exporters.json_exporter import export_to_json


def run_research_pipeline(
    query: str,
    model: str = "gpt-4-turbo-preview",
    max_sources: int = 10,
    export_markdown: bool = True,
    export_json: bool = True,
    output_dir: str = "outputs"
):
    """
    Run the complete research pipeline.

    Args:
        query: The research question or topic
        model: OpenAI model to use
        max_sources: Maximum number of sources to fetch
        export_markdown: Whether to export to Markdown
        export_json: Whether to export to JSON
        output_dir: Directory for exported files

    Returns:
        ResearchReport object
    """
    print("=" * 80)
    print("ScholarAI Research Pipeline")
    print("=" * 80)
    print(f"\nTopic: {query}\n")

    # Step 1: Research Agent
    print("Step 1/3: Searching and curating sources...")
    research_agent = ResearchAgent(model=model, max_sources=max_sources)
    research_results = research_agent.research(query)

    print(f"✓ Found {research_results['total_sources']} sources\n")

    # Step 2: Synthesizer Agent
    print("Step 2/3: Synthesizing findings into structured report...")
    synthesizer = SynthesizerAgent(model=model)
    report = synthesizer.synthesize(
        topic=query,
        sources=research_results['sources'],
        analysis=research_results.get('analysis')
    )

    print("✓ Report generated\n")

    # Step 3: Export
    print("Step 3/3: Exporting report...")
    export_paths = []

    if export_markdown:
        md_path = export_to_markdown(report, output_dir=output_dir)
        export_paths.append(f"Markdown: {md_path}")

    if export_json:
        json_path = export_to_json(report, output_dir=output_dir)
        export_paths.append(f"JSON: {json_path}")

    if export_paths:
        print("✓ Exported to:")
        for path in export_paths:
            print(f"  - {path}")
    print()

    # Display summary
    print("=" * 80)
    print("Report Summary")
    print("=" * 80)
    print(f"\n{report.tldr}\n")
    print(f"Key Findings: {len(report.key_findings)}")
    print(f"Top Sources: {len(report.top_sources)}")
    print("\n" + "=" * 80)

    return report


def main(query: str):
    """
    Run the research pipeline from the command line.

    Args:
        query: The research question or topic to investigate
    """
    try:
        # Check for required API keys
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY not found in environment")
            sys.exit(1)

        if not os.getenv("TAVILY_API_KEY"):
            print("Error: TAVILY_API_KEY not found in environment")
            sys.exit(1)

        # Run the pipeline
        report = run_research_pipeline(
            query=query,
            model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
            max_sources=int(os.getenv("MAX_SEARCH_RESULTS", "10")),
            export_markdown=True,
            export_json=True,
            output_dir="outputs"
        )

        print("\n✅ Research complete!")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py 'Your research question'")
        print("\nExample:")
        print("  python main.py 'recent advances in quantum computing'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    main(query)
