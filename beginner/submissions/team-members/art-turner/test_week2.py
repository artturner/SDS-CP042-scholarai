"""Test script for Week 2 components."""

import os
import sys
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
from exporters.markdown_exporter import to_markdown, export_to_markdown
from exporters.json_exporter import to_json, export_to_json
from models.report import ResearchReport, KeyFinding, Source


def test_data_models():
    """Test the Pydantic data models."""
    print("=" * 80)
    print("Testing Data Models")
    print("=" * 80)

    try:
        # Create sample data
        source = Source(
            title="Test Paper",
            url="https://example.com/paper",
            snippet="This is a test snippet.",
            score=0.95,
            why_matters="Important test paper"
        )

        finding = KeyFinding(
            finding="This is a key finding",
            citations=["https://example.com/paper1", "https://example.com/paper2"]
        )

        report = ResearchReport(
            topic="Test Topic",
            tldr="This is a test summary of the research findings.",
            key_findings=[finding],
            conflicts_and_caveats="Some conflicts exist.",
            top_sources=[source],
            metadata={"test": True}
        )

        print("\n✓ Data models created successfully")
        print(f"  - Report topic: {report.topic}")
        print(f"  - Key findings: {len(report.key_findings)}")
        print(f"  - Top sources: {len(report.top_sources)}")
        print()

        return True

    except Exception as e:
        print(f"\n✗ Data models test failed: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_synthesizer():
    """Test the synthesizer agent."""
    print("=" * 80)
    print("Testing Synthesizer Agent")
    print("=" * 80)

    try:
        # Create mock sources
        sources = [
            {
                "title": "AI Advances in 2024",
                "url": "https://example.com/ai-2024",
                "snippet": "Significant progress in large language models...",
                "score": 0.98
            },
            {
                "title": "Machine Learning Trends",
                "url": "https://example.com/ml-trends",
                "snippet": "New architectures and training techniques...",
                "score": 0.95
            },
        ]

        print("\nSynthesizing report from sources...")
        print("(This may take a moment...)\n")

        synthesizer = SynthesizerAgent(model="gpt-4-turbo-preview")
        report = synthesizer.synthesize(
            topic="Recent advances in AI",
            sources=sources,
            analysis="Preliminary analysis of AI progress"
        )

        print("✓ Report synthesized successfully")
        print(f"  - TL;DR length: {len(report.tldr)} characters")
        print(f"  - Key findings: {len(report.key_findings)}")
        print(f"  - Top sources: {len(report.top_sources)}")
        print(f"\nTL;DR Preview:\n{report.tldr[:200]}...")
        print()

        return True, report

    except Exception as e:
        print(f"\n✗ Synthesizer test failed: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False, None


def test_exporters(report):
    """Test the Markdown and JSON exporters."""
    print("=" * 80)
    print("Testing Exporters")
    print("=" * 80)

    try:
        # Test Markdown
        print("\nTesting Markdown exporter...")
        md_string = to_markdown(report)
        print(f"✓ Markdown string generated ({len(md_string)} characters)")

        md_file = export_to_markdown(report, output_dir="test_outputs")
        print(f"✓ Markdown exported to: {md_file}")

        # Test JSON
        print("\nTesting JSON exporter...")
        json_string = to_json(report)
        print(f"✓ JSON string generated ({len(json_string)} characters)")

        json_file = export_to_json(report, output_dir="test_outputs")
        print(f"✓ JSON exported to: {json_file}")

        print()
        return True

    except Exception as e:
        print(f"\n✗ Exporter test failed: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the complete pipeline."""
    print("=" * 80)
    print("Testing Full Pipeline (Research → Synthesis → Export)")
    print("=" * 80)

    try:
        topic = "artificial intelligence in healthcare"
        print(f"\nResearching: {topic}")
        print("(This may take a minute...)\n")

        # Step 1: Research
        print("Step 1: Researching...")
        research_agent = ResearchAgent(model="gpt-4-turbo-preview", max_sources=5)
        research_results = research_agent.research(topic)
        print(f"✓ Found {research_results['total_sources']} sources")

        # Step 2: Synthesize
        print("\nStep 2: Synthesizing...")
        synthesizer = SynthesizerAgent(model="gpt-4-turbo-preview")
        report = synthesizer.synthesize(
            topic=topic,
            sources=research_results['sources'],
            analysis=research_results.get('analysis')
        )
        print("✓ Report generated")

        # Step 3: Export
        print("\nStep 3: Exporting...")
        md_path = export_to_markdown(report, output_dir="test_outputs")
        json_path = export_to_json(report, output_dir="test_outputs")
        print(f"✓ Exported to Markdown: {md_path}")
        print(f"✓ Exported to JSON: {json_path}")

        print("\n✓ Full pipeline test completed successfully!")
        print(f"\nReport Summary:")
        print(f"  Topic: {report.topic}")
        print(f"  TL;DR: {report.tldr[:150]}...")
        print(f"  Key Findings: {len(report.key_findings)}")
        print(f"  Top Sources: {len(report.top_sources)}")
        print()

        return True

    except Exception as e:
        print(f"\n✗ Full pipeline test failed: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Week 2 Component Tests")
    print("=" * 80 + "\n")

    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("✗ OPENAI_API_KEY not found in environment")
        return

    if not os.getenv("TAVILY_API_KEY"):
        print("✗ TAVILY_API_KEY not found in environment")
        return

    print("✓ Environment variables configured\n")

    # Run tests
    results = []

    # Test 1: Data Models
    results.append(("Data Models", test_data_models()))

    # Test 2: Synthesizer
    synth_passed, report = test_synthesizer()
    results.append(("Synthesizer Agent", synth_passed))

    # Test 3: Exporters (if synthesizer succeeded)
    if synth_passed and report:
        results.append(("Exporters", test_exporters(report)))
    else:
        results.append(("Exporters", False))
        print("⚠️  Skipping exporter tests due to synthesizer failure\n")

    # Test 4: Full Pipeline
    results.append(("Full Pipeline", test_full_pipeline()))

    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")

    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n🎉 All tests passed! Week 2 components are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
