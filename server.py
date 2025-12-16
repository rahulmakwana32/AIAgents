import os
import asyncio
import sys
import logging
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Configure logging to write to stderr to avoid polluting stdout (used for JSON-RPC)
logging.basicConfig(level=logging.INFO, stream=sys.stderr)

# Browser-Use and LangChain Imports
from browser_use import Agent, Browser, BrowserProfile
# Browser-Use Wrappers (Required for Agent)
from browser_use.llm import ChatGoogle as BrowserChatGoogle
from browser_use.llm import ChatOpenAI as BrowserChatOpenAI

# Standard LangChain Models (Required for Terminal Tool / HumanMessage)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

# Load environment variables
load_dotenv()

# Initialize FastMCP Server
mcp = FastMCP("AI Browser Agent")

# Global Browser Instance
_browser: Browser | None = None

def initialize_browser_llm(llm_choice: str) -> BaseChatModel:
    """Initializes the LLM for browser-use (requires specific wrappers)."""
    if llm_choice.lower() == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found.")
        return BrowserChatGoogle(model="gemini-2.5-flash", api_key=api_key, temperature=0)
    elif llm_choice.lower() == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found.")
        return BrowserChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
    else:
        raise ValueError("Invalid LLM choice.")

def initialize_terminal_llm(llm_choice: str) -> BaseChatModel:
    """Initializes the LLM for terminal observation (standard LangChain)."""
    if llm_choice.lower() == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found.")
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
    elif llm_choice.lower() == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found.")
        return ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
    else:
        raise ValueError("Invalid LLM choice.")

async def get_browser() -> Browser:
    """Returns a persistent browser instance, creating one if needed."""
    global _browser
    if _browser is None:
        browser_config = BrowserProfile(
            headless=False,
            slow_mo=500,
            viewport_width=1280,
            viewport_height=900
        )
        _browser = Browser(browser_profile=browser_config)
        await _browser.start()
    return _browser

@mcp.tool()
async def run_browser_task(task: str, model_provider: str = "gemini") -> str:
    """
    Executes a task in a visible browser using the specified AI model.
    
    Args:
        task: The natural language instruction for the browser agent (e.g., "Go to google.com and search for AI").
        model_provider: The LLM provider to use. Options: "gemini" (default) or "openai".
    """
    try:
        llm = initialize_browser_llm(model_provider)
        browser = await get_browser()
        
        agent = Agent(
            task=task,
            llm=llm,
            browser=browser,
            max_actions_per_step=1,
            use_vision=True
        )
        
        # Run the agent with a step limit to prevent infinite loops
        result = await agent.run(max_steps=15)
        
        final_answer = "Agent run completed, but no final answer was extracted."
        try:
            if result.final_answer:
                final_answer = result.final_answer
        except AttributeError:
            pass
            
        return final_answer

    except Exception as e:
        return f"Error executing task: {str(e)}"

@mcp.tool()
async def run_terminal_task(command: str, model_provider: str = "gemini") -> str:
    """
    Executes a terminal command and provides an AI observation of the output.
    
    Args:
        command: The terminal command to execute (e.g., "ls -la", "git status").
        model_provider: The LLM provider to use for observation. Options: "gemini" (default) or "openai".
    """
    try:
        # Execute the command
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        output = stdout.decode().strip()
        error = stderr.decode().strip()
        
        full_output = f"STDOUT:\n{output}\n\nSTDERR:\n{error}" if error else output
        
        # Get AI Observation
        llm = initialize_terminal_llm(model_provider)
        prompt = f"""
        Analyze the following terminal output for the command '{command}':
        
        ```
        {full_output}
        ```
        
        Provide a concise observation of what this output means.
        """
        
        from langchain_core.messages import HumanMessage
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        observation = response.content
        
        return f"COMMAND: {command}\n\nOUTPUT:\n{full_output}\n\nOBSERVATION:\n{observation}"

    except Exception as e:
        return f"Error executing terminal command: {str(e)}"

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
