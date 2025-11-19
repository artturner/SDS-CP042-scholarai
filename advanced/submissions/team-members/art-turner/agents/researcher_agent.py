"""Researcher Agent for investigating specific subtopics.

This agent specializes in researching a single subtopic, searching for
relevant sources, and producing structured findings that can be combined
with other researchers' work.
"""

import os
import json
from typing import List, Dict, Optional
from datetime import datetime
from openai import OpenAI
from tools.web_search import web_search
from models.report import Subtopic, SubtopicFindings, KeyFinding, Source


class ResearcherAgent:
    """
    Agent that researches a specific subtopic and returns structured findings.

    Each researcher agent focuses on one subtopic from the Topic Splitter,
    searches for relevant sources, and produces findings that can be
    merged by the Synthesizer Agent.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        max_sources: int = 8,
        agent_id: Optional[str] = None,
    ):
        """
        Initialize the Researcher Agent.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY from environment.
            model: OpenAI model to use
            max_sources: Maximum number of sources to fetch per search
            agent_id: Optional identifier for this researcher (e.g., "Researcher 1")
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_sources = max_sources
        self.agent_id = agent_id or "Researcher"

        # Tool definition for function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for relevant academic sources, articles, and papers on a given topic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query for finding relevant sources",
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of results to return (default: 8)",
                                "default": 8,
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        self.system_prompt = """You are a specialized research agent focused on investigating a specific subtopic. Your task is to search for and analyze sources related to your assigned subtopic.

Your responsibilities:
1. Use the web_search function to find relevant sources
2. Focus your searches on the specific subtopic you've been assigned
3. Analyze the search results for relevance and quality
4. Extract key insights with supporting citations
5. Provide a concise summary of your findings

When searching:
- Use the provided search queries as starting points
- Formulate additional queries if needed
- Look for authoritative, recent sources
- Prioritize academic and professional content

After gathering sources, provide your findings in JSON format:
{
  "summary": "TL;DR of findings for this subtopic (2-3 sentences)",
  "key_insights": [
    {
      "finding": "Main insight or discovery",
      "citations": ["url1", "url2"]
    }
  ],
  "researcher_notes": "Any additional observations or caveats"
}"""

    def research_subtopic(
        self,
        subtopic: Subtopic,
        main_topic: str = ""
    ) -> SubtopicFindings:
        """
        Research a specific subtopic and return structured findings.

        Args:
            subtopic: The Subtopic object with name, description, and search queries
            main_topic: The main research topic for context

        Returns:
            SubtopicFindings with summary, insights, and sources
        """
        # Build context-aware prompt
        context = f"Main Topic: {main_topic}\n" if main_topic else ""

        user_message = f"""{context}Your assigned subtopic: {subtopic.name}

Description: {subtopic.description}

Suggested search queries:
{chr(10).join(f'- {q}' for q in subtopic.search_queries)}

Please search for sources related to this subtopic and provide your findings."""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Initial call to GPT-4
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto",
        )

        # Collect all sources
        all_sources = []
        queries_used = []

        # Agent loop - process tool calls
        while response.choices[0].message.tool_calls:
            messages.append(response.choices[0].message)

            for tool_call in response.choices[0].message.tool_calls:
                if tool_call.function.name == "web_search":
                    args = json.loads(tool_call.function.arguments)
                    query = args.get("query")
                    k = args.get("k", self.max_sources)

                    queries_used.append(query)
                    search_results = web_search(query, k=k)
                    all_sources.extend(search_results)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(search_results),
                    })

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
            )

        # Get final analysis from the agent
        final_message = response.choices[0].message.content

        # Parse the findings
        findings = self._parse_findings(
            subtopic=subtopic,
            sources=all_sources,
            analysis=final_message,
            queries_used=queries_used
        )

        return findings

    def _parse_findings(
        self,
        subtopic: Subtopic,
        sources: List[Dict],
        analysis: str,
        queries_used: List[str]
    ) -> SubtopicFindings:
        """Parse the agent's analysis into structured findings."""

        # Try to extract JSON from the analysis
        try:
            # Find JSON in the response
            json_start = analysis.find('{')
            json_end = analysis.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = analysis[json_start:json_end]
                result_data = json.loads(json_str)
            else:
                result_data = {}
        except json.JSONDecodeError:
            result_data = {}

        # Parse key insights
        key_insights = []
        for insight_data in result_data.get("key_insights", []):
            key_insights.append(KeyFinding(
                finding=insight_data.get("finding", ""),
                citations=insight_data.get("citations", [])
            ))

        # If no insights were parsed, create a default one
        if not key_insights and sources:
            key_insights.append(KeyFinding(
                finding=result_data.get("summary", f"Research findings on {subtopic.name}"),
                citations=[s["url"] for s in sources[:3]]
            ))

        # Convert sources to Source objects
        source_objects = [
            Source(
                title=s.get("title", ""),
                url=s.get("url", ""),
                snippet=s.get("snippet", ""),
                score=s.get("score"),
            )
            for s in sources
        ]

        # Build SubtopicFindings
        findings = SubtopicFindings(
            subtopic=subtopic.name,
            summary=result_data.get("summary", analysis[:500] if analysis else f"Research on {subtopic.name}"),
            key_insights=key_insights,
            sources=source_objects,
            researcher_notes=result_data.get("researcher_notes", ""),
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id,
                "model": self.model,
                "queries_used": queries_used,
                "num_sources": len(sources)
            }
        )

        return findings


def create_researcher_agent(
    model: str = "gpt-4-turbo-preview",
    max_sources: int = 8,
    agent_id: Optional[str] = None
) -> ResearcherAgent:
    """
    Factory function to create a researcher agent.

    Args:
        model: OpenAI model to use
        max_sources: Maximum number of sources to fetch
        agent_id: Optional identifier for this researcher

    Returns:
        Initialized ResearcherAgent instance
    """
    return ResearcherAgent(model=model, max_sources=max_sources, agent_id=agent_id)
