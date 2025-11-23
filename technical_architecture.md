# 🏗️ Deep Research Agent - Technical Architecture

## 1. System Overview

The Deep Research Agent is a stateful, multi-agent AI system designed to perform autonomous open-source intelligence (OSINT) research. It leverages **LangGraph** for orchestration, **Google Vertex AI (Gemini)** and **OpenAI (GPT-4)** for reasoning, and **SerpApi** for real-time data retrieval.

## 2. Core Architecture (LangGraph)

The system is built as a stateful graph where nodes represent agents or tools, and edges represent the flow of control.

### 2.1. Graph Flow

1.  **Start**: User input triggers the graph.
2.  **Researcher Node**:
    - Analyzes current findings.
    - Generates search queries to fill information gaps.
    - **Loop**: Can loop back to itself (via Tools) up to 3 times.
3.  **Tools Node**:
    - Executes Google Search queries.
    - Returns structured results (Title, Link, Snippet).
4.  **Analyst Node**:
    - Synthesizes all findings.
    - Calculates a **Risk Score** (0-10) based on a rubric.
    - Generates a drafted report with inline citations.
5.  **Reviewer Node**:
    - Acts as Quality Control.
    - Reviews the Analyst's report for accuracy, consistency, and completeness.
    - **Feedback Loop**: Can reject the report and send it back to the Analyst (max 2 revisions).
6.  **End**: Final report is displayed to the user.

### 2.2. State Management (`src/state.py`)

The system maintains a global state object (`AgentState`) that persists across the graph execution.

```python
class AgentState(TypedDict):
    research_topic: str       # The subject of the research
    findings: List[str]       # Accumulated search results
    search_queries: List[str] # History of queries to prevent duplication
    iterations: int           # Counter for research loops (Max 3)
    risk_score: float         # Extracted numerical risk assessment (0-10)
    report: str               # The final markdown report
    messages: List[BaseMessage] # Chat history for context
    researcher_model: str     # Selected model for research
    analyst_model: str        # Selected model for analysis
    revisions: int            # Counter for review loops (Max 2)
```

## 3. Agents & Components

### 3.1. Researcher (`src/agents/researcher.py`)

- **Role**: The "Gatherer".
- **Logic**:
  - **Consecutive Search Strategy**: Receives previous findings and queries to build upon them.
  - **Dynamic Query Refinement**: Prompted to generate _new_ queries based on what is missing.
  - **Infinite Loop Prevention**: Strictly enforces a limit of 3 iterations. If reached, it stops searching and forces a summary.
- **Tools**: Google Search.

### 3.2. Analyst (`src/agents/analyst.py`)

- **Role**: The "Synthesizer".
- **Logic**:
  - **Risk Scoring**: Uses regex to extract a numerical score (0-10) from the generated text.
  - **Citations**: Formats citations as `[Short Name](URL)` (e.g., `[Reuters](...)`) instead of full URLs.
  - **Revisions**: Tracks the number of revisions to prevent infinite review loops.

### 3.3. Reviewer (`src/agents/reviewer.py`)

- **Role**: The "Gatekeeper" / Quality Control.
- **Logic**:
  - **Adversarial Evaluation**: Checks for hallucinations and logic errors.
  - **Rate Limit Handling**: Implements conditional truncation for GPT models (20k chars) to avoid `429` errors, while allowing full context for Gemini.

### 3.4. Google Search (`src/tools/search.py`)

- **Wrapper**: Uses `SerpApi`.
- **Output**: Returns a JSON string containing `title`, `link`, and `snippet` for each result. This structured data allows the Analyst to create accurate citations.

### 3.5. User Interface (`app.py`)

- **Framework**: Chainlit.
- **Concierge**: A lightweight routing agent that distinguishes between "chitchat" and "research requests."
- **Streaming**: Uses `graph.astream()` to provide real-time feedback.
  - **Step Visualization**: Custom `cl.Step` objects show the user exactly what the agent is doing.
  - **Deferred Output**: The final report is hidden until the Reviewer approves.

## 4. Directory Structure

```
src/
├── agents/          # Agent definitions (prompts & logic)
│   ├── researcher.py
│   ├── analyst.py
│   └── reviewer.py
├── tools/           # Tool definitions
│   └── search.py
├── utils/           # Helpers (logging, etc.)
├── config.py        # Environment & Configuration
├── graph.py         # LangGraph definition
└── state.py         # State schema
└app.py           # Chainlit application (Entry Point)
```

## 5. Future Improvements

- **Parallel Search**: Update the Researcher to generate multiple queries that can be executed in parallel batches.
- **Vector Memory**: Implement a vector database (e.g., Pinecone, Chroma) to store findings for long-term memory across different sessions and also more context engineering.
