"""Test script for Week 1 components."""

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

from tools.web_search import web_search, WebSearchTool
from agents.research_agent import ResearchAgent


def test_web_search():
    """Test the web search tool."""
    print("=" * 80)
    print("Testing Web Search Tool")
    print("=" * 80)

    try:
        # Test basic search
        query = "machine learning transformers"
        print(f"\nSearching for: '{query}'")
        results = web_search(query, k=5)

        print(f"\nFound {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   Score: {result.get('score', 'N/A')}")
            print(f"   Snippet: {result['snippet'][:150]}...")
            print()

        print("✓ Web search tool working correctly!\n")
        return True

    except Exception as e:
        print(f"✗ Web search tool failed: {str(e)}\n")
        return False


def test_research_agent():
    """Test the research agent."""
    print("=" * 80)
    print("Testing Research Agent")
    print("=" * 80)

    try:
        # Initialize agent
        agent = ResearchAgent(
            model="gpt-4-turbo-preview",
            max_sources=5
        )

        # Test research
        topic = "recent advances in large language models"
        print(f"\nResearching: '{topic}'")
        print("(This may take a moment...)\n")

        result = agent.research(topic)

        print(f"\nResearch Results:")
        print(f"Topic: {result['topic']}")
        print(f"Total sources found: {result['total_sources']}")
        print(f"\nAgent Analysis:")
        print(result['analysis'])

        if result['sources']:
            print(f"\nTop Sources:")
            curated = agent.curate_sources(result['sources'], top_n=3)
            for i, source in enumerate(curated, 1):
                print(f"\n{i}. {source['title']}")
                print(f"   URL: {source['url']}")
                print(f"   Score: {source.get('score', 'N/A')}")

        print("\n✓ Research agent working correctly!\n")
        return True

    except Exception as e:
        print(f"✗ Research agent failed: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Week 1 Component Tests")
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
    results.append(("Web Search Tool", test_web_search()))
    results.append(("Research Agent", test_research_agent()))

    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")

    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n🎉 All tests passed! Week 1 components are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
