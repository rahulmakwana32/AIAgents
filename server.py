import os
import asyncio
import sys
import logging
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import splunklib.client as client
import splunklib.results as results
import json

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
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)
    elif llm_choice.lower() == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found.")
        return ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
    else:
        raise ValueError("Invalid LLM choice.")

def initialize_generic_llm(llm_choice: str) -> BaseChatModel:
    """Initializes the LLM for generic actions (standard LangChain)."""
    # We can reuse the terminal LLM init as it returns a standard ChatModel
    return initialize_terminal_llm(llm_choice)

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

@mcp.tool()
async def run_splunk_query(query: str, earliest_time: str = "-24h", latest_time: str = "now", model_provider: str = "gemini") -> str:
    """
    Executes a search query against a Splunk instance and provides AI insights.
    
    Args:
        query: The Splunk search query (e.g., "search index=_internal | head 5").
        earliest_time: The earliest time for the search (default "-24h").
        latest_time: The latest time for the search (default "now").
        model_provider: The LLM provider to use for insights. Options: "gemini" (default) or "openai".
    """
    try:
        host = os.getenv("SPLUNK_HOST", "localhost")
        port = os.getenv("SPLUNK_PORT", "8089") # Default to 8089 if not set
        username = os.getenv("SPLUNK_USERNAME")
        password = os.getenv("SPLUNK_PASSWORD")
        token = os.getenv("SPLUNK_TOKEN")
        
        if not username and not token:
             return "Error: SPLUNK_USERNAME/PASSWORD or SPLUNK_TOKEN not found in environment."
            
        service_args = {
            "host": host,
            "port": int(port),
            "verify": False
        }
        
        if token:
            service_args["token"] = token
        else:
            service_args["username"] = username
            service_args["password"] = password
            
        print(f"Connecting to Splunk at {host}:{port}...")
        service = client.connect(**service_args)
        
        if not query.startswith("search") and not query.startswith("|"):
            query = f"search {query}"
            
        search_kwargs = {
            "earliest_time": earliest_time,
            "latest_time": latest_time,
            "output_mode": "json"
        }
        
        print(f"Running query: '{query}' with args: {search_kwargs}...")
        # oneshot returns a stream
        response_stream = service.jobs.oneshot(query, **search_kwargs)
        
        # Read the stream
        reader = results.JSONResultsReader(response_stream)
        
        results_list = []
        for item in reader:
            if isinstance(item, results.Message):
                # Diagnostic messages
                results_list.append(f"[Message] {item.type}: {item.message}")
            elif isinstance(item, dict):
                # Normal result
                results_list.append(item)
                
        if not results_list:
            return "No results found."
            
        # Format results nicely (limit to first 10 for brevity)
        formatted_results = json.dumps(results_list[:10], indent=2)
        count = len(results_list)
        
        result_str = f"Found {count} results (showing first 10):\n{formatted_results}"
        
        # Generate Insights
        try:
            llm = initialize_generic_llm(model_provider)
            prompt = f"""
            Analyze the following Splunk search results for the query: '{query}'
            
            Results:
            ```json
            {formatted_results}
            ```
            
            Provide 2-3 bullet points of key insights or patterns found in this data. Be concise.
            If the data is just a list of fields or metadata, just summarize what is available.
            """
            from langchain_core.messages import HumanMessage
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            insight = response.content.strip()
            
            return f"{result_str}\n\n### AI Insights:\n{insight}"
        except Exception as e:
            print(f"Failed to generate insights: {e}")
            return result_str

    except Exception as e:
        return f"Error executing Splunk query: {str(e)}"

@mcp.tool()
async def run_generic_action(data: str, instruction: str, model_provider: str = "gemini") -> str:
    """
    Performs a generic action or transformation on the provided data based on natural language instructions.
    
    Args:
        data: The input data to process (e.g., previous tool output).
        instruction: The instruction for what to do with the data (e.g., "convert to HTML table", "summarize this").
        model_provider: The LLM provider to use. Options: "gemini" (default) or "openai".
    """
    try:
        llm = initialize_generic_llm(model_provider)
        
        prompt = f"""
        You are a helpful data processing assistant.
        
        DATA:
        ```
        {data}
        ```
        
        INSTRUCTION:
        {instruction}
        
        Perform the instruction on the data.  
        """
        
        from langchain_core.messages import HumanMessage
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content
        
    except Exception as e:
        return f"Error executing generic action: {str(e)}"        

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
