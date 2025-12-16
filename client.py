import asyncio
import sys
import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Import standard LangChain models for routing
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

def initialize_routing_llm(llm_choice: str):
    """Initializes the LLM used for routing decisions."""
    if llm_choice == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        # Use standard ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=api_key, 
            temperature=0
        )
    elif llm_choice == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
            model="gpt-4o-mini", 
            api_key=api_key, 
            temperature=0
        )
    return None

# ... (imports)

async def route_intent(user_input: str, llm) -> str:
    print("Routing Intent... using. ",llm)
    """Decides if the input is a BROWSER task or a TERMINAL task."""
    # We rely entirely on the LLM to decide based on the user prompt.
    
    prompt_text = f"""
    Classify the following user request into one of two categories:
    
    1. BROWSER: 
       - Browsing the web, searching Google, visiting URLs.
       - Extracting data from websites, summarizing web pages.
       - Interacting with web elements (clicking, typing on a page).
       
    2. TERMINAL: 
       - Running local shell commands (ls, cd, git, grep, etc.).
       - Managing local files and directories (creating, deleting, listing).
       - Checking system status, local processes, or desktop content.
       - Any request that implies looking at the local computer's storage or state.

    User Request: "{user_input}"

    Return ONLY the word "BROWSER" or "TERMINAL".
    """
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt_text)])
        print("Routing LLM Response:", response.content.strip().upper())
        return response.content.strip().upper()
    except Exception as e:
        print("Routing LLM failed. Defaulting to 'BROWSER'.")
        print(f"Exception details: {e}")
        return "BROWSER" # Default fallback

async def generate_shell_command(user_input: str, llm) -> str:
    """Converts a natural language request into a shell command."""
    prompt_text = f"""
    Convert the following natural language request into a single valid shell command (for macOS/Linux).
    Return ONLY the command, no markdown, no explanations.
    
    Request: "{user_input}"
    """
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt_text)])
        command = response.content.strip()
        # Remove markdown code blocks if present
        if command.startswith("```"):
            command = command.strip("`").replace("bash", "").replace("sh", "").strip()
        return command
    except Exception as e:
        print(f"Command generation failed: {e}")
        return user_input # Fallback to raw input

async def main():
    # --- LLM Selection ---
    print("--- AI Browser Agent Client ---")
    llm_choice = input("Enter your preferred LLM (Gemini/OpenAI): ").strip().lower()
    if llm_choice not in ["gemini", "openai"]:
        print("Invalid choice. Defaulting to 'gemini'.")
        llm_choice = "gemini"
    print(f"Using: {llm_choice.capitalize()}\n")

    # Initialize Routing LLM
    routing_llm = initialize_routing_llm(llm_choice)

    # Define server parameters
    server_script = os.path.join(os.path.dirname(__file__), "server.py")
    
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script],
        env=None 
    )

    print(f"Connecting to server at: {server_script}...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("\nConnected to MCP Server!")
            print("Available Tools:", [tool.name for tool in tools.tools])
            
            print("\n" + "="*50)
            print("MCP CLIENT - BROWSER AGENT")
            print("Type 'QUIT' to exit.")
            print("="*50 + "\n")

            while True:
                user_input = input(">> ").strip()
                
                if user_input.upper() == 'QUIT':
                    break
                
                if not user_input:
                    continue

                # Smart Routing
                intent = await route_intent(user_input, routing_llm)
                print(f"Detected Intent: {intent}")

                if intent == "TERMINAL":
                    tool_name = "run_terminal_task"
                    # If it has "term:" prefix, use the rest as command.
                    # If it's natural language (no prefix), generate the command.
                    if user_input.startswith("term:"):
                        cmd = user_input[5:].strip()
                    else:
                        print("Generating shell command...")
                        cmd = await generate_shell_command(user_input, routing_llm)
                    
                    tool_args = {
                        "command": cmd,
                        "model_provider": llm_choice
                    }
                    print(f"Executing terminal command: '{cmd}'...")
                else:
                    tool_name = "run_browser_task"
                    tool_args = {
                        "task": user_input,
                        "model_provider": llm_choice
                    }
                    print(f"Sending task to browser agent: '{user_input}'...")
                
                try:
                    result = await session.call_tool(
                        name=tool_name,
                        arguments=tool_args
                    )
                    
                    print("\n" + "#"*50)
                    print("RESULT:")
                    # The result content is a list of TextContent or ImageContent
                    for content in result.content:
                        if content.type == 'text':
                            print(content.text)
                        else:
                            print(f"[{content.type} content]")
                    print("#"*50 + "\n")
                    
                except Exception as e:
                    print(f"\n[ERROR] Tool execution failed: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nFatal Error: {e}")
