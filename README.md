# AI Browser Agent

An intelligent, visible browser agent built with [browser-use](https://github.com/browser-use/browser-use) and [LangChain](https://github.com/langchain-ai/langchain). This agent can control a web browser to perform tasks based on natural language instructions, supporting both Google Gemini and OpenAI models.

## Features

*   **Multi-LLM Support**: Choose between Google Gemini (Vision & Reasoning) and OpenAI GPT-4o.
*   **Visible Execution**: Runs the browser in non-headless mode (`headless=False`) so you can watch the agent work.
*   **Interactive Loop**: Persistent session allows you to issue multiple consecutive commands to the same browser instance.
*   **Robust Error Handling**: Keeps the session alive even if an individual task fails.

## Prerequisites

*   Python 3.11 or higher
*   A Google Cloud Project with Gemini API access OR an OpenAI API Key.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ai-browser-agent
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install browser-use langchain-google-genai langchain-openai python-dotenv
    ```
    *(Note: Ensure you have `playwright` installed and browsers downloaded)*
    ```bash
    playwright install
    ```

## Configuration

1.  Create a `.env` file in the project root:
    ```bash
    touch .env
    ```

2.  Add your API keys to `.env`:
    ```env
    GEMINI_API_KEY=your_google_api_key_here
    OPENAI_API_KEY=your_openai_api_key_here
    ```

## Usage

1.  **Run the agent:**
    ```bash
    python visible_agent.py
    ```

2.  **Select your LLM:**
    When prompted, enter `gemini` or `openai`.

3.  **Give instructions:**
    Type your task at the `>>` prompt.
    *   Example: *"Go to google.com and search for 'Agentic AI'"*
    *   Example: *"Click on the first result and summarize the page"*

4.  **Exit:**
    Type `QUIT` to close the browser and end the session.

## Project Structure

*   `visible_agent.py`: Main entry point containing the agent logic and interactive loop.
*   `server.py`: MCP server implementation exposing the browser agent as a tool.
*   `.env`: Stores sensitive API keys (not committed to version control).

## MCP Server Usage

You can run this project as an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server to use the browser agent from MCP-compliant clients (like Claude Desktop).

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the server:**
    ```bash
    # Run with mcp CLI (recommended for development)
    mcp run server.py

    # OR run directly with python
    python server.py
    ```

3.  **Configure in Claude Desktop:**
    Add the following to your `claude_desktop_config.json`:
    ```json
    {
      "mcpServers": {
        "ai-browser-agent": {
          "command": "python",
          "args": ["/absolute/path/to/ai-browser-agent/server.py"]
        }
      }
    }
    ```

## Custom Client Usage

You can also use the provided Python client to interact with the server programmatically or via a CLI.

```bash
python client.py
```
This will connect to the local `server.py`.

*   **Browser Task**: Type your instruction normally (e.g., "Search for AI").
*   **Terminal Task**: Prefix with `term:` (e.g., `term: ls -la`).



