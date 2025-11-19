# ScholarAI Advanced Track - Project Report

## Project Overview

This project implements a multi-agent research system where specialized AI agents collaborate to perform comprehensive literature research and synthesis. The system demonstrates multi-agent orchestration patterns, parallel execution, context sharing between agents, and a self-critique loop for quality improvement.

---

## Agent Roles and Interactions

### 1. Topic Splitter Agent

**Role**: Strategic research planning

**Responsibilities**:
- Analyzes broad research topics
- Identifies 2-4 distinct, complementary subtopics
- Generates specific search queries for each subtopic
- Ensures comprehensive coverage without overlap

**Example Prompt**:
```
Analyze this research topic and break it down into 3 focused subtopics:
"Impact of artificial intelligence on healthcare"
```

**Example Output**:
```json
{
  "main_topic_analysis": "AI in healthcare spans diagnostics, treatment, and operations",
  "subtopics": [
    {
      "name": "AI in Medical Diagnostics",
      "description": "Machine learning applications in disease detection and imaging",
      "search_queries": ["AI medical imaging diagnosis", "deep learning radiology", "ML pathology detection"]
    },
    {
      "name": "AI in Drug Discovery",
      "description": "Computational approaches to pharmaceutical research",
      "search_queries": ["AI drug discovery", "machine learning pharmaceutical", "computational drug design"]
    },
    {
      "name": "AI in Patient Care Management",
      "description": "AI systems for treatment planning and monitoring",
      "search_queries": ["AI patient monitoring", "predictive healthcare analytics", "AI treatment planning"]
    }
  ]
}
```

---

### 2. Researcher Agents

**Role**: Specialized subtopic investigation

**Responsibilities**:
- Execute web searches using provided queries
- Analyze and curate relevant sources
- Extract key insights with citations
- Identify important findings for their subtopic

**Key Design**: Multiple researcher agents run in parallel using `ThreadPoolExecutor`, each assigned to one subtopic. This significantly reduces total research time.

**Example Output** (SubtopicFindings):
```json
{
  "subtopic": "AI in Medical Diagnostics",
  "summary": "Deep learning models show promise in medical imaging, with some outperforming radiologists in specific cancer detection tasks.",
  "key_insights": [
    {
      "finding": "CNN-based systems achieve 94% accuracy in detecting diabetic retinopathy",
      "citations": ["https://example.com/study1"]
    }
  ],
  "sources": [...],
  "researcher_notes": "Strong evidence for imaging applications, less for general diagnostics"
}
```

---

### 3. Synthesizer Agent

**Role**: Cross-subtopic integration and report generation

**Responsibilities**:
- Merge findings from all researchers
- Identify consensus points across subtopics
- Highlight conflicts and gaps in literature
- Generate executive summary (≤150 words)
- Select top 5 most important sources overall

**Style Adaptation**: Supports three writing styles:
- **Technical**: Domain-specific terminology for experts
- **Layperson**: Clear, jargon-free language
- **Business**: Focus on practical implications

**Tone Options**:
- **Neutral**: Objective presentation of findings
- **Advisory**: Includes recommendations

---

### 4. Critic Agent

**Role**: Quality assurance through self-critique

**Responsibilities**:
- Review synthesized reports for quality issues
- Check factual consistency, citation accuracy, logical coherence, completeness
- Provide quality score (1-10)
- Request revisions with specific instructions if needed

**Strictness Levels**:
- **Lenient**: Only flags major issues
- **Balanced**: Moderate to major issues (default)
- **Strict**: All issues including minor ones

**Self-Critique Loop**:
```
Synthesizer → Critic Review
                 ↓
         APPROVED? → Done
                 ↓ no
         Synthesizer Revises
                 ↓
         Critic Re-reviews
                 ↓
         (up to max_revisions)
```

**Example Review Output**:
```json
{
  "decision": "REVISION_NEEDED",
  "overall_score": 7,
  "issues_found": [
    {
      "category": "citation_accuracy",
      "severity": "moderate",
      "description": "Executive summary claims lack direct citations",
      "location": "executive_summary",
      "suggestion": "Add inline citations for key statistics"
    }
  ],
  "strengths": ["Good coverage of subtopics", "Clear structure"],
  "revision_instructions": "Add citations to support claims in executive summary..."
}
```

---

## Orchestration Design

### Parallel Execution Pattern

The orchestrator uses a **parallel fan-out, sequential fan-in** pattern:

```python
# Fan-out: Parallel research
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(research, subtopic) for subtopic in subtopics]

# Fan-in: Collect results
findings = [future.result() for future in as_completed(futures)]
```

### Context Sharing

Each agent receives appropriate context:
- **Researchers**: Main topic + subtopic details + search queries
- **Synthesizer**: All subtopic findings + style/tone preferences
- **Critic**: Complete report for review

### Error Handling

Failed researcher agents don't crash the pipeline:
```python
except Exception as e:
    error_finding = SubtopicFindings(
        subtopic=subtopic.name,
        summary=f"Research failed: {str(e)}",
        ...
    )
```

---

## Data Models

### Key Models

1. **Subtopic**: Name, description, search queries
2. **SubtopicFindings**: Per-researcher output with sources and insights
3. **CriticReview**: Decision, score, issues, revision instructions
4. **MultiAgentReport**: Complete synthesis with all components

### Report Structure

```python
MultiAgentReport:
  - topic: str
  - subtopics: List[str]
  - executive_summary: str (≤150 words)
  - subtopic_findings: List[SubtopicFindings]
  - overall_insights: List[KeyFinding]
  - consensus_points: List[str]
  - conflicts_and_gaps: str
  - all_sources: List[Source]
  - top_sources: List[Source] (max 5)
  - critic_review: Optional[CriticReview]
  - revision_count: int
```

---

## Gradio Interface

### Input Controls
- Research topic text field
- Style dropdown (Technical/Layperson/Business)
- Tone dropdown (Neutral/Advisory)
- Number of subtopics slider (2-4)
- Sources per subtopic slider (5-15)
- Critic settings accordion (enable, strictness, max revisions)

### Output Tabs
1. **Summary**: Executive summary, consensus points, key insights
2. **Research Findings**: Per-subtopic details with citations
3. **Sources**: Top sources with relevance scores and explanations
4. **Critic Review**: Quality score, strengths, issues found

### Export Options
- Download as Markdown
- Download as JSON
- Preview exports in collapsible section

---

## Challenges and Solutions

### 1. Coordination Complexity

**Challenge**: Managing state across multiple agents with different outputs.

**Solution**: Pydantic models enforce structure, and the orchestrator maintains clear data flow between stages.

### 2. Context Window Management

**Challenge**: Large amounts of source content can exceed context limits.

**Solution**: Each researcher summarizes findings before passing to synthesizer, and sources are truncated to key snippets.

### 3. Consistency in Synthesis

**Challenge**: Ensuring synthesizer accurately represents all researcher findings.

**Solution**: Critic agent specifically checks for completeness and flags missing key points.

### 4. Revision Loop Termination

**Challenge**: Preventing infinite revision loops.

**Solution**: Hard limit on max_revisions (default 2) and score-based approval threshold.

### 5. Parallel Execution on Windows

**Challenge**: ThreadPoolExecutor behavior on Windows.

**Solution**: Explicit thread management and proper exception handling per future.

---

## Reflections

### What Worked Well

1. **Parallel execution** dramatically improved research speed (3x faster than sequential)
2. **Self-critique loop** caught issues like missing citations and unsupported claims
3. **Structured output** made synthesis consistent and exportable
4. **Style/tone controls** added flexibility for different use cases

### Areas for Improvement

1. **Source deduplication** could be smarter (currently URL-based only)
2. **Critic could be more specific** in revision instructions
3. **Could add citation verification** by fetching and checking source content
4. **Memory/caching** for repeated searches on similar topics

### Lessons Learned

1. **Agent specialization** improves quality - each agent has clear, focused responsibilities
2. **Self-critique is powerful** - the revision loop caught many quality issues
3. **Structured prompts** with JSON output are more reliable than free-form text
4. **Progress feedback** is essential for long-running multi-agent tasks

---

## Future Enhancements

1. **Hierarchical agents**: Sub-researchers for very complex subtopics
2. **Source verification**: Fetch and validate cited URLs
3. **Comparative analysis**: Compare findings across different runs
4. **Custom critic rules**: User-defined quality criteria
5. **Agent memory**: Remember previous research for follow-up queries

---

## Deployment

The application is deployed to Hugging Face Spaces with:
- Pinned Gradio version (4.19.2) and Pydantic 2.0.3 for compatibility
- Environment variables for API keys (set as HF Secrets)
- Automatic scaling based on usage

---

## Conclusion

This project demonstrates the power of multi-agent AI systems for complex research tasks. By decomposing the problem into specialized agents with clear roles, implementing parallel execution for efficiency, and adding a self-critique loop for quality assurance, the system produces comprehensive, well-structured research reports that would take hours to compile manually.

The self-critique pattern in particular is highly reusable for any AI system that produces output requiring quality review, and the parallel orchestration pattern can be applied to any task that can be decomposed into independent subtasks.
