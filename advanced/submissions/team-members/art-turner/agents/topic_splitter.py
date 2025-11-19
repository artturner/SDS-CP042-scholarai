"""Topic Splitter Agent for breaking down research topics into subtopics.

This agent analyzes a broad research topic and breaks it down into 2-3
focused subtopics that can be researched in parallel by specialized
researcher agents.
"""

import os
import json
from typing import List, Optional
from openai import OpenAI
from models.report import Subtopic


class TopicSplitterAgent:
    """
    Agent that splits a research topic into focused subtopics.

    This is the first agent in the multi-agent pipeline. It analyzes
    the user's research topic and identifies 2-3 distinct aspects
    that can be researched independently.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        num_subtopics: int = 3,
    ):
        """
        Initialize the Topic Splitter Agent.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY from environment.
            model: OpenAI model to use
            num_subtopics: Number of subtopics to generate (2-3 recommended)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.num_subtopics = min(max(num_subtopics, 2), 4)  # Clamp between 2-4

        self.system_prompt = f"""You are an expert research strategist. Your task is to analyze a research topic and break it down into {self.num_subtopics} distinct, focused subtopics that can be researched independently.

Your responsibilities:
1. Analyze the main research topic thoroughly
2. Identify {self.num_subtopics} distinct aspects or dimensions of the topic
3. Ensure subtopics are complementary but not overlapping
4. Provide clear descriptions for each subtopic
5. Suggest 2-3 specific search queries for each subtopic

Guidelines:
- Subtopics should cover different angles (e.g., technical, applications, challenges, future directions)
- Each subtopic should be specific enough for focused research
- Search queries should be actionable and likely to find relevant academic/professional sources
- Consider both breadth and depth in your breakdown

Return your analysis as valid JSON with this structure:
{{
  "main_topic_analysis": "Brief analysis of the main topic (1-2 sentences)",
  "subtopics": [
    {{
      "name": "Subtopic name",
      "description": "What this subtopic covers and why it's important",
      "search_queries": ["query 1", "query 2", "query 3"]
    }}
  ]
}}"""

    def split_topic(self, topic: str) -> List[Subtopic]:
        """
        Split a research topic into focused subtopics.

        Args:
            topic: The main research topic to analyze

        Returns:
            List of Subtopic objects with names, descriptions, and search queries
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Analyze this research topic and break it down into {self.num_subtopics} focused subtopics:\n\n{topic}"
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.7,  # Some creativity in topic breakdown
        )

        result_text = response.choices[0].message.content
        result_data = json.loads(result_text)

        # Parse subtopics into Subtopic objects
        subtopics = []
        for st_data in result_data.get("subtopics", []):
            subtopic = Subtopic(
                name=st_data.get("name", ""),
                description=st_data.get("description", ""),
                search_queries=st_data.get("search_queries", [])
            )
            subtopics.append(subtopic)

        return subtopics

    def analyze_topic(self, topic: str) -> dict:
        """
        Analyze a topic and return both subtopics and analysis.

        Args:
            topic: The research topic to analyze

        Returns:
            Dictionary with 'analysis' and 'subtopics' keys
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Analyze this research topic and break it down into {self.num_subtopics} focused subtopics:\n\n{topic}"
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        result_text = response.choices[0].message.content
        result_data = json.loads(result_text)

        # Parse subtopics
        subtopics = []
        for st_data in result_data.get("subtopics", []):
            subtopic = Subtopic(
                name=st_data.get("name", ""),
                description=st_data.get("description", ""),
                search_queries=st_data.get("search_queries", [])
            )
            subtopics.append(subtopic)

        return {
            "analysis": result_data.get("main_topic_analysis", ""),
            "subtopics": subtopics
        }


def create_topic_splitter(
    model: str = "gpt-4-turbo-preview",
    num_subtopics: int = 3
) -> TopicSplitterAgent:
    """
    Factory function to create a topic splitter agent.

    Args:
        model: OpenAI model to use
        num_subtopics: Number of subtopics to generate

    Returns:
        Initialized TopicSplitterAgent instance
    """
    return TopicSplitterAgent(model=model, num_subtopics=num_subtopics)
