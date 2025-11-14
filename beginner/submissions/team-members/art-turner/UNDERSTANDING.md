# Understanding the ScholarAI Codebase

This document explains the key concepts and architecture to help you understand and present the code confidently.

## Core Architecture

### The Agent Pattern

**What is an Agent?**
An "agent" is an LLM (like GPT-4) that can:
1. Understand what tools it has available
2. Decide when to use those tools
3. Call the tools with appropriate parameters
4. Interpret the results and decide next steps

This is different from simple "prompt engineering" because the LLM autonomously decides WHEN and HOW to use tools.

### Our Two-Agent Pipeline

```
User Query
    ↓
Research Agent (uses web_search tool)
    → Searches the web
    → Gets results
    → Curates relevant sources
    ↓
Synthesizer Agent (no tools, pure analysis)
    → Analyzes all sources
    → Generates structured report
    ↓
Final Report (exported as MD/JSON)
```

## Key Files Explained

### 1. `tools/web_search.py` - The Foundation
**What it does:** Wraps the Tavily API to search the web

**Why we need it:**
- Raw Google search returns HTML pages
- Tavily is AI-optimized: returns clean text + relevance scores
- Perfect for feeding to an LLM

**Key concept:**
```python
results = web_search("quantum computing", k=10)
# Returns: [{"title": "...", "url": "...", "snippet": "...", "score": 0.95}, ...]
```

### 2. `models/report.py` - Data Structures
**What it does:** Defines the shape of our data using Pydantic

**Why Pydantic?**
- **Validation:** Ensures data types are correct
- **Serialization:** Easy JSON export
- **Self-documenting:** Field descriptions explain the data

**Key models:**
- `Source`: One web source with URL, title, snippet, score
- `KeyFinding`: One insight + citations
- `ResearchReport`: The complete report structure

### 3. `agents/research_agent.py` - The Smart Searcher
**What it does:** An LLM that can search the web autonomously

**How it works:**
1. We give GPT-4 a "tool definition" describing the `web_search` function
2. GPT-4 reads the user's query and decides: "I need to search for X"
3. GPT-4 returns a "function call" request (JSON format)
4. We execute `web_search(X)` and give results back to GPT-4
5. GPT-4 analyzes results and returns curated sources

**The Key Innovation - Function Calling:**
```python
# We tell GPT-4 about our tool:
tools = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web...",
        "parameters": {...}  # Defines what args it takes
    }
}]

# GPT-4 can now "call" this function autonomously!
```

**Why this is powerful:**
- GPT-4 decides WHAT to search for based on the query
- GPT-4 can search multiple times if needed
- GPT-4 filters and ranks results intelligently

### 4. `agents/synthesizer_agent.py` - The Analyst
**What it does:** Takes raw sources and creates a structured report

**How it differs from Research Agent:**
- NO tools - just analysis
- Uses JSON mode to ensure structured output
- Identifies conflicts, extracts findings, explains importance

**JSON Mode Explained:**
```python
response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    response_format={"type": "json_object"},  # Forces JSON output
    messages=[...]
)
```

This ensures GPT-4 ALWAYS returns valid JSON matching our report structure.

### 5. `app.py` - The User Interface
**What it does:** Gradio web interface

**Key Gradio Concepts:**
- `gr.Textbox`: Input field
- `gr.Dropdown`: Selection menu
- `gr.HTML`: Display formatted HTML
- `gr.Button.click()`: Connect button to Python function
- `gr.State`: Store data between interactions

**The Flow:**
1. User clicks "Research" button
2. `research_and_synthesize()` function runs:
   - Shows progress bar
   - Calls Research Agent
   - Calls Synthesizer Agent
   - Returns formatted HTML
3. Gradio updates the UI automatically

### 6. `exporters/*.py` - Output Generation
**Simple converters:**
- Take a `ResearchReport` object
- Convert to Markdown or JSON format
- Save to file

Uses Python f-strings for templating.

## Important Technical Decisions

### Why OpenAI Function Calling?
**Alternative:** We could have hard-coded "search for X, Y, Z"
**Our choice:** Let GPT-4 decide what to search
**Benefit:** More intelligent, adapts to different query types

### Why Pydantic Models?
**Alternative:** Plain dictionaries
**Our choice:** Typed models with validation
**Benefit:** Catches errors early, self-documenting, IDE support

### Why Two Agents Instead of One?
**Alternative:** One agent does search AND synthesis
**Our choice:** Separation of concerns
**Benefit:**
- Research Agent: Specialized for finding sources
- Synthesizer Agent: Specialized for analysis
- Each can use different prompts/settings
- Easier to test and debug

### Why Gradio Instead of Streamlit?
**Both are good choices**
**Gradio advantages:**
- Better for ML/AI demos
- Built-in progress bars
- Easier state management
- Better for Hugging Face Spaces

## Common Questions

### Q: How does the agent know to use the web_search tool?
**A:** We provide a JSON "tool definition" that describes:
- Function name
- What it does
- What parameters it accepts

GPT-4 reads this and understands: "When I need to search, I should call web_search(query, k)".

### Q: What if GPT-4 doesn't call the tool?
**A:** The system prompt instructs it to. We say: "Use web_search to find sources". GPT-4 is good at following instructions.

### Q: How do we know the output is structured correctly?
**A:**
1. **Synthesizer:** JSON mode forces valid JSON
2. **Pydantic:** Validates the structure matches our models
3. **If validation fails:** We get a clear error message

### Q: Why use environment variables for API keys?
**A:**
- **Security:** Never commit keys to git
- **Flexibility:** Different keys for dev/prod
- **Standard practice:** Industry best practice

### Q: Could we use a different LLM (not OpenAI)?
**A:** Yes! The pattern works with:
- Anthropic Claude (via tool use)
- Google Gemini (via function calling)
- Open source models (via function calling)

You'd just need to change the client initialization and API calls.

## Key Takeaways for Presenting

1. **Agents are LLMs with tools** - This is the core concept
2. **Function calling enables autonomy** - LLM decides when/how to use tools
3. **Separation of concerns** - Research vs Synthesis
4. **Type safety with Pydantic** - Ensures data quality
5. **Gradio for rapid prototyping** - Quick, professional UI

## The "Aha!" Moment

**Traditional approach:**
```python
# Hard-coded, inflexible
results1 = search("quantum computing")
results2 = search("quantum computing applications")
results3 = search("quantum computing 2024")
# etc...
```

**Agent approach:**
```python
# LLM decides what to search based on context
agent.research("quantum computing")
# Agent might search:
# - "quantum computing fundamentals"
# - "quantum computing recent advances"
# - "quantum vs classical computing comparison"
# All decided autonomously!
```

This is what makes it "agentic" - the LLM has agency to make decisions.
