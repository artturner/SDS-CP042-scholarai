"""Web search tool using Tavily API.

This module provides a wrapper around the Tavily search API, which is specifically
optimized for AI applications. Tavily returns high-quality, relevant search results
with pre-extracted content and relevance scores.
"""

import os
from typing import List, Dict, Optional
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
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")

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
            query: The search query
            max_results: Maximum number of results to return (default: 10)
            search_depth: "basic" for quick results, "advanced" for deeper analysis
            include_domains: Optional list of domains to restrict search to
            exclude_domains: Optional list of domains to exclude from results

        Returns:
            List of dictionaries with keys: 'title', 'url', 'snippet', 'score'

        Raises:
            RuntimeError: If the search API call fails
        """
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )

            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("content", ""),
                    "score": result.get("score", 0.0),
                })

            return results

        except Exception as e:
            raise RuntimeError(f"Web search failed: {str(e)}")


def web_search(query: str, k: int = 10) -> List[Dict[str, str]]:
    """
    Convenience function for web search.

    Args:
        query: The search query
        k: Number of results to return (default: 10)

    Returns:
        List of dictionaries with keys: 'title', 'url', 'snippet', 'score'
    """
    tool = WebSearchTool()
    return tool.search(query, max_results=k)
