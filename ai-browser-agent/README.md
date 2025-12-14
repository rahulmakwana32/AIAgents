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
*   `.env`: Stores sensitive API keys (not committed to version control).
