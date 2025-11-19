---
title: ScholarAI Advanced
emoji: "🎓"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
---

# ScholarAI Advanced - Multi-Agent Research System

A multi-agent AI system that collaborates to perform comprehensive research synthesis. Multiple specialized agents work together to break down topics, research subtopics in parallel, synthesize findings, and review quality through a self-critique loop.

## Architecture

```
User Topic → Topic Splitter → [Researcher 1] → Synthesizer → Critic → Report
                            → [Researcher 2] →              ↓
                            → [Researcher 3] →        (revision loop)
```

### Agents

1. **Topic Splitter Agent**: Analyzes the main topic and breaks it into 2-4 focused subtopics
2. **Researcher Agents**: Run in parallel, each researching one subtopic using web search
3. **Synthesizer Agent**: Merges all findings into a coherent report
4. **Critic Agent**: Reviews for quality and can request revisions (self-critique loop)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
OPENAI_MODEL=gpt-4-turbo-preview
```

### 3. Run the Application

```bash
python app.py
```

The Gradio interface will launch at `http://localhost:7860`

## Usage

1. **Enter a research topic** - Broad topics work best
2. **Configure settings**:
   - Style: Technical, Layperson, or Business
   - Tone: Neutral or Advisory
   - Number of subtopics: 2-4
   - Sources per subtopic: 5-15
   - Critic settings: Enable/disable, strictness, max revisions
3. **Click "Start Multi-Agent Research"**
4. **View results** in tabs:
   - Summary: Executive summary and key insights
   - Research Findings: Detailed findings per subtopic
   - Sources: All citations organized by subtopic
   - Critic Review: Quality score, issues, and strengths
5. **Export** as Markdown or JSON

## Features

- **Parallel Processing**: Researcher agents run concurrently
- **Self-Critique Loop**: Critic reviews and requests revisions
- **Structured Output**: Consistent format with citations
- **Multiple Styles**: Technical, Layperson, or Business
- **Export Options**: Markdown or JSON download
- **Progress Tracking**: Real-time updates

## Project Structure

```
art-turner/
├── agents/
│   ├── topic_splitter.py    # Breaks topics into subtopics
│   ├── researcher_agent.py  # Researches subtopics
│   ├── synthesizer_agent.py # Merges findings
│   ├── critic_agent.py      # Reviews quality
│   └── orchestrator.py      # Coordinates workflow
├── models/
│   └── report.py            # Data models
├── tools/
│   └── web_search.py        # Tavily search
├── exporters/
│   ├── markdown_exporter.py
│   └── json_exporter.py
├── app.py                   # Gradio interface
└── requirements.txt
```

## Requirements

- Python 3.9+
- OpenAI API key (GPT-4 Turbo recommended)
- Tavily API key for web search

## License

Part of the SuperDataScience Community Project - ScholarAI Advanced Track
