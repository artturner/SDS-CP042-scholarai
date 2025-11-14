"""Research Agent using OpenAI Agents SDK.

This agent demonstrates the "function calling" pattern, which is the core
of making LLMs into autonomous agents that can use tools.

KEY CONCEPT: Instead of just generating text, GPT-4 can:
1. Read tool descriptions
2. Decide when to use a tool
3. Generate a function call with parameters
4. Process the results

This is what makes it an "agent" - it has agency to make decisions.
"""

import os
import json
from typing import List, Dict, Optional
from openai import OpenAI
from tools.web_search import web_search


class ResearchAgent:
    """
    Agent that searches the web and curates relevant sources.

    This class wraps GPT-4 and gives it access to the web_search tool.
    The LLM autonomously decides when and how to use this tool.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        max_sources: int = 10,
    ):
        """
        Initialize the Research Agent.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY from environment.
            model: OpenAI model to use (gpt-4-turbo-preview recommended for best results)
            max_sources: Maximum number of sources to fetch per search
        """
        # Load API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Create OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_sources = max_sources

        # ============================================================================
        # CRITICAL SECTION: Tool Definition for Function Calling
        # ============================================================================
        # This JSON structure tells GPT-4 what tools it can use.
        # Think of it as giving GPT-4 a "manual" for the web_search function.
        #
        # When we pass this to GPT-4 along with a message, GPT-4 can:
        # 1. Read the description and understand what the tool does
        # 2. Look at the parameters to know what arguments it needs
        # 3. Decide "I need to search" and generate a function call
        # 4. Return a special message saying "please call web_search with query='...'"
        #
        # We then execute that function and send results back to GPT-4.
        self.tools = [
            {
                "type": "function",  # Type of tool (could also be "retrieval", etc.)
                "function": {
                    # Must match the actual Python function name
                    "name": "web_search",

                    # This description helps GPT-4 decide WHEN to use this tool
                    # Be specific! "Search the web" is clear about what it does
                    "description": "Search the web for relevant academic sources, articles, and papers on a given topic",

                    # Parameters in JSON Schema format
                    # GPT-4 uses this to know what arguments to provide
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query for finding relevant sources",
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of results to return (default: 10)",
                                "default": 10,
                            },
                        },
                        # "query" is required, "k" is optional (has default)
                        "required": ["query"],
                    },
                },
            }
        ]

        # ============================================================================
        # System Prompt: Instructions for the Agent
        # ============================================================================
        # This tells GPT-4 its role and how to behave.
        # The prompt explicitly instructs GPT-4 to USE the web_search tool.
        self.system_prompt = """You are a research assistant tasked with finding and curating relevant sources on a given topic.

Your responsibilities:
1. Use the web_search function to find relevant sources
2. Analyze the search results for relevance and quality
3. Curate the most relevant and reliable sources
4. Provide a structured summary of the sources found

When searching:
- Formulate clear, specific search queries
- Look for academic sources, reputable publications, and authoritative content
- Prioritize recent and well-cited sources
- Consider multiple perspectives on the topic

Return your findings in a structured format with the curated sources."""

    def research(self, topic: str) -> Dict:
        """
        Research a topic by searching the web and curating sources.

        This is where the "agent loop" happens. The flow is:
        1. Send topic to GPT-4 with tool definitions
        2. GPT-4 decides to call web_search
        3. We execute the search
        4. Send results back to GPT-4
        5. GPT-4 analyzes and may search again OR finish
        6. Loop until GPT-4 decides it's done

        Args:
            topic: The research topic or question (e.g., "quantum computing advances")

        Returns:
            Dictionary containing:
            - topic: The original query
            - sources: All sources found (combined from potentially multiple searches)
            - analysis: GPT-4's final analysis/summary
            - total_sources: Count of sources found
        """
        # ============================================================================
        # STEP 1: Build the conversation
        # ============================================================================
        # The "messages" list represents the conversation with GPT-4
        # It starts with system prompt (agent's instructions) and user query
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Research this topic and find the most relevant sources: {topic}",
            },
        ]

        # ============================================================================
        # STEP 2: Initial call to GPT-4
        # ============================================================================
        # We pass:
        # - messages: The conversation so far
        # - tools: The tool definitions (web_search)
        # - tool_choice="auto": Let GPT-4 decide whether to use tools
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto",  # GPT-4 chooses: use tool OR just respond
        )

        # ============================================================================
        # STEP 3: The Agent Loop - Process tool calls until GPT-4 is done
        # ============================================================================
        # GPT-4 can either:
        # A) Return a text response (done thinking)
        # B) Return a tool_call (wants to use web_search)
        #
        # We loop while GPT-4 keeps requesting tool calls
        sources = []
        while response.choices[0].message.tool_calls:
            # GPT-4 wants to call a tool!
            # response.choices[0].message.tool_calls is a list of tool call requests

            # Add GPT-4's message to conversation history
            # This preserves the context of what tools it asked for
            messages.append(response.choices[0].message)

            # ========================================================================
            # STEP 3a: Execute each tool call that GPT-4 requested
            # ========================================================================
            for tool_call in response.choices[0].message.tool_calls:
                if tool_call.function.name == "web_search":
                    # GPT-4 has asked us to call web_search!

                    # Parse the arguments GPT-4 provided
                    # tool_call.function.arguments is a JSON string like:
                    # '{"query": "quantum computing breakthroughs 2024", "k": 10}'
                    args = json.loads(tool_call.function.arguments)
                    query = args.get("query")
                    k = args.get("k", self.max_sources)

                    # Execute the actual search using our web_search function
                    search_results = web_search(query, k=k)

                    # Collect all sources (GPT-4 might search multiple times)
                    sources.extend(search_results)

                    # ================================================================
                    # STEP 3b: Send tool results back to GPT-4
                    # ================================================================
                    # We add a message with role="tool" containing the search results
                    # tool_call_id links this result to the specific tool call
                    messages.append({
                        "role": "tool",  # Special role indicating tool output
                        "tool_call_id": tool_call.id,  # Links to the request
                        "content": json.dumps(search_results),  # The actual results
                    })

            # ========================================================================
            # STEP 3c: Call GPT-4 again with the tool results
            # ========================================================================
            # Now GPT-4 can see:
            # - Original user query
            # - Its tool call request
            # - The tool results
            #
            # GPT-4 will either:
            # - Make another tool call (search again with different query)
            # - Return a final text response (done researching)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # Full conversation including tool results
                tools=self.tools,
                tool_choice="auto",
            )
            # Loop continues if there are more tool_calls, exits otherwise

        # ============================================================================
        # STEP 4: GPT-4 is done calling tools - get final analysis
        # ============================================================================
        # When we exit the while loop, GPT-4 has provided a text response
        # This is its analysis of the sources it found
        final_message = response.choices[0].message.content

        # Return everything in a structured format
        return {
            "topic": topic,
            "sources": sources,                # All sources from all searches
            "analysis": final_message,         # GPT-4's summary
            "total_sources": len(sources),     # How many we found
        }

    def curate_sources(
        self, sources: List[Dict], top_n: int = 5
    ) -> List[Dict]:
        """
        Curate and rank sources by relevance.

        Args:
            sources: List of source dictionaries
            top_n: Number of top sources to return

        Returns:
            List of top N curated sources
        """
        # Sort by score (if available) and return top N
        sorted_sources = sorted(
            sources,
            key=lambda x: x.get("score", 0.0),
            reverse=True
        )
        return sorted_sources[:top_n]


def create_research_agent(
    model: str = "gpt-4-turbo-preview",
    max_sources: int = 10
) -> ResearchAgent:
    """
    Factory function to create a research agent.

    Args:
        model: OpenAI model to use
        max_sources: Maximum number of sources to fetch

    Returns:
        Initialized ResearchAgent instance
    """
    return ResearchAgent(model=model, max_sources=max_sources)
