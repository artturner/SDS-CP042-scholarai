# ScholarAI Presentation Guide

## Quick Reference for Explaining Your Code

### 30-Second Elevator Pitch
"I built an AI research assistant using OpenAI's function calling to create autonomous agents. The system uses two specialized agents: one that searches the web and another that synthesizes findings into structured, cited reports. It's deployed via Gradio as an interactive web app."

### Key Technical Achievements

1. **Implemented Agentic AI** - Not just prompting, but true agents with tool use
2. **Function Calling Pattern** - LLM autonomously decides when/how to use tools
3. **Multi-Agent Architecture** - Separation of concerns (search vs synthesis)
4. **Type-Safe Data Models** - Pydantic ensures data quality
5. **Production-Ready UI** - Full Gradio interface with export functionality

---

## The Three Concepts You MUST Understand

### 1. What Makes It "Agentic"?

**Traditional Approach:**
```python
# Hard-coded, we decide everything
search("topic A")
search("topic B")
# etc...
```

**Our Agentic Approach:**
```python
# LLM decides what to search based on context
research_agent.research("topic")
# Agent might search multiple queries it determines are relevant
```

**The Difference:** The LLM has *agency* - it makes decisions about what actions to take.

### 2. How Function Calling Works

**The Pattern:**
1. **Define** tools in JSON format (what they do, what parameters they take)
2. **Send** to GPT-4 along with user query
3. **GPT-4 decides** "I need to use tool X with parameters Y"
4. **We execute** the actual function
5. **Send results** back to GPT-4
6. **Loop** until GPT-4 is satisfied

**The Code:**
```python
# 1. Define tool
tools = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web...",
        "parameters": {...}
    }
}]

# 2. Call GPT-4 with tools
response = client.chat.completions.create(
    messages=messages,
    tools=tools,                    # ← Give GPT-4 the "manual"
    tool_choice="auto"              # ← Let it decide
)

# 3. Check if GPT-4 wants to use a tool
if response.choices[0].message.tool_calls:
    # GPT-4 returned: "call web_search with query='...'"
    # We execute it and send results back
```

### 3. Why Two Agents?

**Research Agent:**
- **Job:** Find relevant sources
- **Tools:** web_search
- **Output:** List of sources + preliminary analysis

**Synthesizer Agent:**
- **Job:** Analyze sources and create structured report
- **Tools:** None (pure analysis)
- **Output:** ResearchReport with TL;DR, findings, citations

**Why separate?**
- **Specialized prompts** - Each agent has different instructions
- **Different capabilities** - One uses tools, one doesn't
- **Easier testing** - Test search and synthesis independently
- **Scalability** - Could easily add more agents (fact-checker, summarizer, etc.)

---

## Demo Flow - What to Show

### 1. Start with the Web Interface (app.py running)
- "Here's the final product - let's submit a query"
- Show a research query executing
- Point out the progress bar (shows agent working)
- Show the three tabs (Summary, Findings, Sources)
- Demonstrate export buttons

### 2. Explain the Architecture
- "Behind the scenes, this uses a two-agent system"
- Show the flow diagram (from UNDERSTANDING.md)
- Explain: User → Research Agent → Synthesizer → Output

### 3. Show the Research Agent Code (research_agent.py)
- Point to the tool definition (lines 67-98)
- Explain: "This is like giving GPT-4 a manual for web_search"
- Show the agent loop (lines 179-235)
- Explain: "GPT-4 decides to search, we execute, send results back, repeat"

### 4. Show a Tool Call Example
- "Let me show you what GPT-4 actually returns when it wants to use a tool"
- Example JSON:
```json
{
  "tool_calls": [{
    "id": "call_abc123",
    "function": {
      "name": "web_search",
      "arguments": "{\"query\": \"quantum computing advances 2024\", \"k\": 10}"
    }
  }]
}
```
- "We parse this, call web_search('quantum computing advances 2024', k=10), and return results"

### 5. Show the Data Models (models/report.py)
- "I used Pydantic for type safety and validation"
- Show ResearchReport structure
- Explain: "This ensures every report has all required fields"

### 6. Show the Synthesizer (briefly)
- "The synthesizer uses JSON mode to ensure structured output"
- Point to `response_format={"type": "json_object"}`
- "GPT-4 must return valid JSON matching our schema"

---

## Questions You Might Get

### Q: "Why not just use ChatGPT with search?"
**A:** ChatGPT can search, but I wanted to:
- Learn how agents work under the hood
- Customize the search behavior
- Control the output format precisely
- Build a production application

### Q: "How is this different from LangChain?"
**A:** LangChain is a framework that provides agent abstractions. I built this using the OpenAI SDK directly to:
- Understand the fundamentals
- Have full control over the agent loop
- Keep dependencies minimal

### Q: "Could this work with other LLMs?"
**A:** Yes! The pattern works with:
- Anthropic Claude (tool use API)
- Google Gemini (function calling)
- Open source models that support function calling

The code would need minor changes to adapt to different APIs.

### Q: "How do you prevent hallucinations?"
**A:**
- **Citation requirement** - Every finding must cite sources
- **Web search** - Grounded in real web content
- **Validation** - Pydantic ensures output structure
- **Transparency** - User can click sources to verify

### Q: "What was the hardest part?"
**A:** Understanding the agent loop! It's a mental shift from:
- "I write code that does X"
to
- "I write code that tells an AI how to do X, then the AI decides the steps"

The function calling mechanism took time to grasp, but once it clicked, everything made sense.

---

## Talking Points - What Makes This Good

1. **Follows Best Practices**
   - Type hints throughout
   - Pydantic for validation
   - Environment variables for secrets
   - Separation of concerns

2. **Production-Ready**
   - Error handling
   - Progress feedback
   - Multiple export formats
   - User-friendly interface

3. **Well-Tested**
   - Test suites for each component
   - Integration tests
   - All tests passing

4. **Documented**
   - Inline comments explaining WHY
   - README with setup instructions
   - UNDERSTANDING.md for concepts
   - This presentation guide

5. **Extensible**
   - Easy to add more agents
   - Easy to add more tools
   - Easy to change LLM providers

---

## Red Flags to Avoid in Your Presentation

❌ "The AI just does it magically"
✅ "The LLM reads the tool definition, decides to call it, and I execute the function"

❌ "I don't really know how it works"
✅ "The function calling loop works by sending tool definitions, getting back tool call requests, executing them, and sending results"

❌ "I copied this from a tutorial"
✅ "I implemented the function calling pattern and customized it for research use"

---

## Final Tip

**Be honest about what you understand and what you learned.**

Good response to "Did you build this yourself?":
> "I built this with AI assistance - similar to how developers use Stack Overflow or documentation. The key is that I understand every line: I can explain the function calling mechanism, why I chose two agents, and how the data flows through the system. I learned a ton about agentic AI patterns in the process."

**Confidence comes from understanding, not from claiming you did everything alone.**
