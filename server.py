import os
import asyncio
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Browser-Use and LangChain Imports
from browser_use import Agent, Browser, BrowserProfile
from browser_use.llm import ChatGoogle, ChatOpenAI
from langchain_core.language_models import BaseChatModel

# Load environment variables
load_dotenv()

# Initialize FastMCP Server
mcp = FastMCP("AI Browser Agent")

# Global Browser Instance
_browser: Browser | None = None

def initialize_llm(llm_choice: str) -> BaseChatModel:
    """Initializes the selected LLM (Gemini or OpenAI)."""
    if llm_choice.lower() == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        return ChatGoogle(model="gemini-2.5-flash", api_key=api_key, temperature=0)
    elif llm_choice.lower() == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        return ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
    else:
        raise ValueError("Invalid LLM choice. Please choose 'Gemini' or 'OpenAI'.")

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
        llm = initialize_llm(model_provider)
        browser = await get_browser()
        
        agent = Agent(
            task=task,
            llm=llm,
            browser=browser
        )
        
        result = await agent.run()
        
        final_answer = "Agent run completed, but no final answer was extracted."
        try:
            if result.final_answer:
                final_answer = result.final_answer
        except AttributeError:
            pass
            
        return final_answer

    except Exception as e:
        return f"Error executing task: {str(e)}"

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
