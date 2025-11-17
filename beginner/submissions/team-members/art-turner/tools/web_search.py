"""Web search tool using Tavily API.

This module provides a wrapper around the Tavily search API, which is specifically
optimized for AI applications. Tavily returns high-quality, relevant search results
with pre-extracted content and relevance scores.
"""

import os
from typing import List, Dict, Optional
# Tavily is an AI-optimized search API that returns clean, structured results
# Better for research than raw Google search because it filters and ranks content
from tavily import TavilyClient


class WebSearchTool:
    """
    Wrapper for Tavily web search API.

    This class encapsulates all interaction with the Tavily search service,
    making it easy to search the web and get structured results.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the web search tool.

        Args:
            api_key: Tavily API key. If not provided, will use TAVILY_API_KEY from environment.

        Raises:
            ValueError: If no API key is found in parameters or environment
        """
        # Try to get API key from parameter first, then from environment variable
        # This pattern allows flexibility: can pass key directly or load from .env
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

        # Fail fast if no API key is available - better to crash early than later
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")

        # Initialize the Tavily client with our API key
        # This client handles all HTTP requests to the Tavily API
        self.client = TavilyClient(api_key=self.api_key)

    def search(
        self,
        query: str,
        max_results: int = 10,
        search_depth: str = "advanced",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Search the web for relevant sources.

        Args:
            query: The search query (e.g., "quantum computing advances 2024")
            max_results: Maximum number of results to return (default: 10)
            search_depth: "basic" for quick results, "advanced" for deeper analysis (default: "advanced")
            include_domains: Optional list of domains to restrict search to (e.g., ["edu", "gov"])
            exclude_domains: Optional list of domains to exclude from results

        Returns:
            List of dictionaries with keys: 'title', 'url', 'snippet', 'score'
            Score is a float 0-1 indicating relevance (1.0 = most relevant)

        Raises:
            RuntimeError: If the search API call fails
        """
        try:
            # Make the API call to Tavily
            # search_depth="advanced" means Tavily will:
            #   - Visit more pages
            #   - Extract more detailed content
            #   - Provide better relevance scoring
            # This is slower but gives higher quality results for research
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )

            # Transform Tavily's response format into our standardized format
            # We normalize the structure so the rest of our app doesn't need to know
            # about Tavily's specific response format
            results = []
            for result in response.get("results", []):
                # .get() with default values ensures we handle missing fields gracefully
                results.append({
                    "title": result.get("title", ""),           # Page title
                    "url": result.get("url", ""),               # Source URL
                    "snippet": result.get("content", ""),       # Extracted text content
                    "score": result.get("score", 0.0),          # Relevance score (0-1)
                })

            return results

        except Exception as e:
            # Wrap any errors in a RuntimeError with context
            # This helps with debugging and error handling upstream
            raise RuntimeError(f"Web search failed: {str(e)}")


def web_search(query: str, k: int = 10) -> List[Dict[str, str]]:
    """
    Convenience function for web search.

    This is a simpler interface that creates a WebSearchTool instance and
    performs a search in one call. Useful when you just need to do a quick search
    without maintaining a WebSearchTool instance.

    Args:
        query: The search query
        k: Number of results to return (default: 10)

    Returns:
        List of dictionaries with keys: 'title', 'url', 'snippet', 'score'

    Example:
        results = web_search("climate change effects", k=5)
        for result in results:
            print(f"{result['title']}: {result['score']}")
    """
    # Create a new WebSearchTool instance (will load API key from environment)
    tool = WebSearchTool()
    # Perform the search and return results
    return tool.search(query, max_results=k)
