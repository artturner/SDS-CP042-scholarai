"""Test script to verify the multi-agent system setup."""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from models.report import (
            Source, KeyFinding, Subtopic, SubtopicFindings, MultiAgentReport
        )
        print("  [OK] models.report")
    except ImportError as e:
        print(f"  [FAIL] models.report: {e}")
        return False

    try:
        from tools.web_search import WebSearchTool, web_search
        print("  [OK] tools.web_search")
    except ImportError as e:
        print(f"  [FAIL] tools.web_search: {e}")
        return False

    try:
        from agents.topic_splitter import TopicSplitterAgent
        print("  [OK] agents.topic_splitter")
    except ImportError as e:
        print(f"  [FAIL] agents.topic_splitter: {e}")
        return False

    try:
        from agents.researcher_agent import ResearcherAgent
        print("  [OK] agents.researcher_agent")
    except ImportError as e:
        print(f"  [FAIL] agents.researcher_agent: {e}")
        return False

    try:
        from agents.synthesizer_agent import SynthesizerAgent
        print("  [OK] agents.synthesizer_agent")
    except ImportError as e:
        print(f"  [FAIL] agents.synthesizer_agent: {e}")
        return False

    try:
        from agents.orchestrator import MultiAgentOrchestrator
        print("  [OK] agents.orchestrator")
    except ImportError as e:
        print(f"  [FAIL] agents.orchestrator: {e}")
        return False

    try:
        from exporters.markdown_exporter import to_markdown
        from exporters.json_exporter import to_json
        print("  [OK] exporters")
    except ImportError as e:
        print(f"  [FAIL] exporters: {e}")
        return False

    return True


def test_models():
    """Test that models can be instantiated."""
    print("\nTesting models...")

    from models.report import (
        Source, KeyFinding, Subtopic, SubtopicFindings, MultiAgentReport
    )

    try:
        source = Source(
            title="Test Source",
            url="https://example.com",
            snippet="Test snippet"
        )
        print("  [OK] Source")
    except Exception as e:
        print(f"  [FAIL] Source: {e}")
        return False

    try:
        finding = KeyFinding(
            finding="Test finding",
            citations=["https://example.com"]
        )
        print("  [OK] KeyFinding")
    except Exception as e:
        print(f"  [FAIL] KeyFinding: {e}")
        return False

    try:
        subtopic = Subtopic(
            name="Test Subtopic",
            description="Test description",
            search_queries=["query 1", "query 2"]
        )
        print("  [OK] Subtopic")
    except Exception as e:
        print(f"  [FAIL] Subtopic: {e}")
        return False

    try:
        findings = SubtopicFindings(
            subtopic="Test",
            summary="Test summary",
            key_insights=[finding],
            sources=[source]
        )
        print("  [OK] SubtopicFindings")
    except Exception as e:
        print(f"  [FAIL] SubtopicFindings: {e}")
        return False

    try:
        report = MultiAgentReport(
            topic="Test Topic",
            subtopics=["Subtopic 1"],
            executive_summary="Test summary",
            subtopic_findings=[findings],
            overall_insights=[finding],
            all_sources=[source],
            top_sources=[source]
        )
        print("  [OK] MultiAgentReport")
    except Exception as e:
        print(f"  [FAIL] MultiAgentReport: {e}")
        return False

    return True


def test_exporters():
    """Test exporters with a sample report."""
    print("\nTesting exporters...")

    from models.report import (
        Source, KeyFinding, Subtopic, SubtopicFindings, MultiAgentReport
    )
    from exporters.markdown_exporter import to_markdown
    from exporters.json_exporter import to_json

    # Create a sample report
    source = Source(
        title="Test Source",
        url="https://example.com",
        snippet="This is a test snippet for the source.",
        score=0.95,
        why_matters="This is important because it tests the system."
    )

    finding = KeyFinding(
        finding="Test finding about the topic",
        citations=["https://example.com"]
    )

    subtopic_findings = SubtopicFindings(
        subtopic="Test Subtopic",
        summary="This is a summary of the test subtopic.",
        key_insights=[finding],
        sources=[source],
        researcher_notes="Some additional notes"
    )

    report = MultiAgentReport(
        topic="Test Research Topic",
        subtopics=["Test Subtopic"],
        executive_summary="This is the executive summary of the test report.",
        subtopic_findings=[subtopic_findings],
        overall_insights=[finding],
        consensus_points=["Point 1", "Point 2"],
        conflicts_and_gaps="Some conflicts noted.",
        all_sources=[source],
        top_sources=[source],
        metadata={"timestamp": "2025-01-01T00:00:00"}
    )

    try:
        md = to_markdown(report)
        assert len(md) > 100
        assert "Test Research Topic" in md
        print("  [OK] Markdown export")
    except Exception as e:
        print(f"  [FAIL] Markdown export: {e}")
        return False

    try:
        json_str = to_json(report)
        assert len(json_str) > 100
        assert "Test Research Topic" in json_str
        print("  [OK] JSON export")
    except Exception as e:
        print(f"  [FAIL] JSON export: {e}")
        return False

    return True


def test_env_vars():
    """Check for required environment variables."""
    print("\nChecking environment variables...")

    from dotenv import load_dotenv
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if openai_key:
        print(f"  [OK] OPENAI_API_KEY found ({openai_key[:8]}...)")
    else:
        print("  [WARN] OPENAI_API_KEY not found")

    if tavily_key:
        print(f"  [OK] TAVILY_API_KEY found ({tavily_key[:8]}...)")
    else:
        print("  [WARN] TAVILY_API_KEY not found")

    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("ScholarAI Advanced - Setup Test")
    print("=" * 50)

    all_passed = True

    if not test_imports():
        all_passed = False

    if not test_models():
        all_passed = False

    if not test_exporters():
        all_passed = False

    test_env_vars()

    print("\n" + "=" * 50)
    if all_passed:
        print("All tests passed!")
        print("\nTo run the app: python app.py")
    else:
        print("Some tests failed. Please check the errors above.")
    print("=" * 50)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
