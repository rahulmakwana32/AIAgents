# AI Agent System: An Agentic Workflow Architecture

**Architected for Context, Observability, and Action.**

## 📖 The "Why": Engineering Beyond Chatbots

As software engineers, we spend approximately 70% of our time on "toil"—context switching between terminals, browser tabs, logs, and documentation. We hold complex mental models in our heads, only to lose them the moment we have to search for a syntax error.

This project is not just another "AI Chatbot." It is an architectural pattern for **Agentic Workflows**. It is designed to bridge the gap between *reasoning* (the LLM) and *doing* (the tools), solving the fundamental problem of **Context Retention** in automated systems.

We built this to answer a specific question: *How do we build a system that can investigate a root cause across multiple domains (Web, Terminal, Logs) without losing the thread of the investigation?*

## 🏗️ Architecture & Design Decisions

We chose a **Client-Server Architecture** utilizing the **Model Context Protocol (MCP)**. Here is why:

### 1. Decoupling "Brain" from "Body" (MCP)
*   **The Problem**: Tightly coupling LLM logic with tool execution leads to monolithic, fragile codebases. If you want to swap OpenAI for Gemini, or add a new tool, you shouldn't have to rewrite your entire agent loop.
*   **The Solution**: We use **MCP** to treat tools as microservices.
    *   **Server (`server.py`)**: The "Body." It exposes capabilities (Splunk, Browser, Terminal) but has no opinion on how they are used. It is stateless and robust.
    *   **Client (`client.py`)**: The "Brain." It handles state, history, and reasoning. It consumes the tools exposed by the server.
*   **Benefit**: This separation of concerns allows us to scale tools independently of the reasoning engine.

### 2. Solving State & Context Retention
*   **The Problem**: Most AI agents are "goldfish"—they perform an action and forget the result immediately after summarizing it. This makes multi-step debugging (e.g., "Get logs" -> "Summarize" -> "Why was it high?") impossible.
*   **The Solution**: **Source-Aware Context Routing**.
    *   We implemented a custom history manager in the client that tracks the *provenance* of data.
    *   When the agent needs to answer a follow-up question, it doesn't just look at the previous summary; it retrieves the original raw data artifact from the history.
*   **Benefit**: The agent maintains a "Mental Model" of the investigation, similar to a senior engineer keeping a debug session in their head.

### 3. The "Generic" Transformation Pattern
*   **The Problem**: Building specific tools for every data format (JSON to CSV, XML to HTML) is unmaintainable.
*   **The Solution**: A **Generic Action Tool** that leverages the LLM's reasoning capabilities to perform ad-hoc transformations on *existing context*.
*   **Benefit**: We write the tool once, and it solves infinite formatting problems.

## 🛠️ System Capabilities

The system exposes four primary interfaces, designed to mimic the workflow of a Site Reliability Engineer (SRE):

1.  **`run_splunk_query` (The Analyst)**
    *   **Capability**: Executes SPL against enterprise logs.
    *   **Intelligence**: Automatically performs anomaly detection on results. It doesn't just return data; it returns *insights* (e.g., "Detected 30% spike in 500 errors").
    *   **Resilience**: Includes self-healing logic to correct invalid SPL syntax automatically.

2.  **`run_browser_task` (The Observer)**
    *   **Capability**: A visible, non-headless browser instance.
    *   **Use Case**: Verifying frontend behavior, reading documentation, or checking external dashboards.

3.  **`run_terminal_task` (The Operator)**
    *   **Capability**: Safe execution of local shell commands.
    *   **Use Case**: Checking disk usage, verifying running processes, or managing files.

4.  **`run_generic_action` (The Transformer)**
    *   **Capability**: Post-processing of tool outputs.
    *   **Use Case**: "Turn that Splunk output into a Markdown table for my report."

## 🚀 Getting Started

Designed for Python 3.11+.

### Prerequisites
*   **Python 3.11+** (Required for `browser-use`)
*   **API Keys**: Google Gemini (Recommended for speed/cost) or OpenAI.
*   **Splunk Credentials**: If using the Splunk tool.

### Installation

1.  **Clone & Environment**
    ```bash
    git clone <repo>
    cd ai-browser-agent
    
    # BRANCHING STRATEGY
    # Use 'release' for the latest working project (Episode 2 features)
    # Use 'main' for the stable, production-ready version
    git checkout release
    
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Dependencies**
    ```bash
    pip install -r requirements.txt
    playwright install
    ```

3.  **Configuration (`.env`)**
    ```env
    GEMINI_API_KEY=...
    SPLUNK_HOST=localhost
    SPLUNK_PASSWORD=...
    ```

### Running the Agent
```bash
python client.py
```
*The client will automatically spawn the MCP server subprocess.*

## 👨‍💻 For the Junior Developer: How to Read This Code

*   Start with **`client.py`**: Look at the `main` loop. Notice how we capture `user_input`, decide on an `intent`, and then dispatch to a tool. This is the "Event Loop" pattern.
*   Look at **`server.py`**: Notice the `@mcp.tool()` decorators. This is how we expose Python functions as API endpoints for the brain to use.
*   **Experiment**: Try modifying the `system_instruction` in `client.py` to change the agent's personality.

 
