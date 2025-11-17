---
title: ScholarAI Research Assistant
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
license: mit
---

# ScholarAI - Beginner Track Submission

**Author:** Art Turner
**Track:** Beginner
**Project:** AI-Powered Research Assistant
**Status:** ✅ Complete

## Overview

ScholarAI is an AI-powered research assistant that searches the web, curates relevant sources, and synthesizes findings into structured, citation-backed reports. Built as part of the SuperDataScience Community Project, this application demonstrates the power of agentic AI systems for academic research.

## Features

### 🔍 Research Capabilities
- **Intelligent Web Search**: Leverages Tavily API for AI-optimized search results
- **Multi-Source Analysis**: Analyzes up to 20 sources per query
- **Relevance Scoring**: Automatically ranks sources by relevance (0-1 scale)
- **Citation Tracking**: Maintains accurate citations for all findings

### 🤖 AI Agents
- **Research Agent**: Uses OpenAI Agents SDK with function calling to search and curate sources
- **Synthesizer Agent**: Generates structured reports with:
  - TL;DR summaries (≤120 words)
  - Key findings with citations
  - Conflicts & caveats analysis
  - Top 5 most relevant sources with explanations

### 🎨 User Interface
- **Interactive Gradio Web UI** with:
  - Three-tab output display (Summary, Findings, Sources)
  - Style controls (Technical vs. Layperson)
  - Tone settings (Neutral vs. Advisory)
  - Configurable source limits (5-20)
  - Real-time progress tracking
  - One-click export to Markdown/JSON
  - Copy-to-clipboard functionality

### 📤 Export Formats
- **Markdown**: Beautifully formatted reports for documentation
- **JSON**: Structured data for programmatic use

## Setup

### Prerequisites

- Python 3.9 or higher
- OpenAI API key
- Tavily API key

### Installation

1. Navigate to this directory:
```bash
cd beginner/submissions/team-members/art-turner
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Usage

### Web Interface (Recommended)

Launch the Gradio web interface:

```bash
python app.py
```

Then open your browser to **http://localhost:7860**

#### Using the Interface:
1. Enter your research topic in the text box
2. Select your preferred **Style** (Technical/Layperson)
3. Choose a **Tone** (Neutral/Advisory)
4. Adjust the maximum number of sources (5-20)
5. Click **Research** and wait for results
6. Explore the three tabs:
   - **Summary**: TL;DR and metadata
   - **Key Findings**: Detailed findings with citations
   - **Sources**: Top sources with relevance scores
7. Use **Copy Summary** or **Download** buttons to save results

### Command Line Interface

For quick research from the terminal:

```bash
python main.py "recent advances in quantum computing"
```

The CLI will:
- Search for sources
- Generate a report
- Export to both Markdown and JSON in the `outputs/` directory
- Display a summary in the terminal

### Examples

**Academic Research:**
```bash
python main.py "transformer architecture in natural language processing"
```

**Current Events:**
```bash
python main.py "impact of AI on healthcare in 2024"
```

**Technical Topics:**
```bash
python main.py "blockchain consensus mechanisms comparison"
```

## Project Structure

```
art-turner/
├── agents/              # AI agent implementations
│   ├── research_agent.py
│   └── synthesizer_agent.py
├── tools/               # Web search and utility tools
│   └── web_search.py
├── models/              # Data models for reports
│   └── report.py
├── exporters/           # Export functionality (MD/JSON)
│   ├── markdown_exporter.py
│   └── json_exporter.py
├── app.py              # Gradio web interface
├── main.py             # CLI interface
├── .env.example        # Environment variables template
├── .gitignore          # Git ignore rules
├── pyproject.toml      # Project dependencies
├── README.md           # This file
└── REPORT.md           # Project report
```

## Configuration

Key settings in `.env`:
- `OPENAI_API_KEY`: Your OpenAI API key
- `TAVILY_API_KEY`: Your Tavily API key
- `OPENAI_MODEL`: Model to use (default: gpt-4-turbo-preview)
- `MAX_SEARCH_RESULTS`: Number of search results to fetch (default: 10)
- `MAX_FINAL_SOURCES`: Number of sources in final report (default: 5)

## Testing

Run the test suites to verify all components:

### Week 1 Tests (Web Search + Research Agent):
```bash
python test_week1.py
```

### Week 2 Tests (Synthesizer + Exporters + Pipeline):
```bash
python test_week2.py
```

### Gradio App Test:
```bash
python test_gradio.py
```

## Development

### Install Development Dependencies:
```bash
pip install -e ".[dev]"
```

### Code Formatting:
```bash
# Format code with Black
black .

# Lint with Ruff
ruff check .
```

### Project Architecture

```
Research Pipeline:
User Query → Research Agent → Web Search (Tavily) → Source Curation
           → Synthesizer Agent → Structured Report → Export (MD/JSON)

Agents:
- ResearchAgent: Function-calling LLM that uses web_search tool
- SynthesizerAgent: Report generation with JSON-mode output

Data Flow:
1. User provides topic + preferences (style/tone)
2. ResearchAgent searches and curates N sources
3. SynthesizerAgent analyzes sources and generates report
4. Exporters convert to Markdown/JSON
5. Gradio UI displays in formatted tabs
```

## Deployment

### Local Deployment
The app runs locally by default on port 7860. For production:

1. Update `.env` with production API keys
2. Set appropriate rate limits
3. Consider adding authentication

### Hugging Face Spaces

To deploy to Hugging Face Spaces:

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Select **Gradio** as the SDK
3. Upload all files except:
   - `.env` (add secrets in Space settings)
   - `test_*.py` files
   - `outputs/` directory
4. Add secrets in Space settings:
   - `OPENAI_API_KEY`
   - `TAVILY_API_KEY`
5. The app will launch automatically

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install -e .

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t scholarai .
docker run -p 7860:7860 --env-file .env scholarai
```

### Environment Variables for Deployment
```bash
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
OPENAI_MODEL=gpt-4-turbo-preview  # Optional
MAX_SEARCH_RESULTS=10              # Optional
```

## Troubleshooting

**Issue: API Key Errors**
- Verify `.env` file exists and contains valid keys
- Check API key permissions and quotas

**Issue: Unicode Errors on Windows**
- The code includes Windows encoding fixes
- Ensure console supports UTF-8

**Issue: Slow Performance**
- Reduce `MAX_SEARCH_RESULTS` in UI slider
- Use `gpt-3.5-turbo` instead of GPT-4 (faster, cheaper)

**Issue: Import Errors**
- Ensure you're in the correct directory
- Run `pip install -e .` to install in editable mode

## Performance Metrics

Typical performance (may vary based on API latency):
- **Research Phase**: 5-15 seconds
- **Synthesis Phase**: 10-20 seconds
- **Total Pipeline**: 15-35 seconds
- **API Costs**: ~$0.05-0.15 per query (GPT-4 Turbo)

## Future Enhancements

Potential improvements:
- [ ] Add source credibility scoring
- [ ] Implement caching for repeated queries
- [ ] Support multiple search APIs
- [ ] Add conversation history
- [ ] PDF export option
- [ ] Multi-language support
- [ ] Comparative analysis mode (compare multiple topics)

## License

Part of the SuperDataScience Community Project - ScholarAI

## Acknowledgments

- **SuperDataScience Community** - Project initiative and support
- **OpenAI** - GPT-4 and Agents SDK
- **Tavily** - AI-optimized search API
- **Gradio** - Web UI framework
- **Contributors** - Community feedback and testing

## Contact

For questions or issues:
- Open an issue on the project repository
- Contact: shaheer@superdatascience.com
