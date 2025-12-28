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
            model="gemini-2.5-flash", 
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

async def route_intent(user_input: str, llm, history: list = None) -> str:
    print("Routing Intent... using. ",llm)
    """Decides if the input is a BROWSER task or a TERMINAL task."""
    # We rely entirely on the LLM to decide based on the user prompt.
    
    history_text = ""
    if history:
        history_text = "Previous interactions:\n"
        for h in history:
            history_text += f"- User: {h['user']}\n  Assistant: {h.get('assistant', '')}\n"

    prompt_text = f"""
    Classify the following user request into one of two categories:
    
    {history_text}

    1. BROWSER: 
       - Browsing the web, searching Google, visiting URLs.
       - Extracting data from websites, summarizing web pages.
       - Interacting with web elements (clicking, typing on a page).
       
    2. TERMINAL: 
       - Running local shell commands (ls, cd, git, grep, etc.).
       - Managing local files and directories (creating, deleting, listing).
       - Checking system status, local processes, or desktop content.
       - Any request that implies looking at the local computer's storage or state.

    3. SPLUNK:
       - Searching Splunk logs, running Splunk queries.
       - Analyzing logs, finding errors in logs, checking Splunk indexes.
       - Any request mentioning "Splunk" or looking for log data.

    4. GENERIC:
       - Transforming data, formatting output, summarizing previous results.
       - "Convert this to HTML", "Make a table from that", "Summarize the logs".
       - Any request that acts on *previous* output rather than fetching new data.

    User Request: "{user_input}"

    Return ONLY the word "BROWSER", "TERMINAL", "SPLUNK", or "GENERIC".
    """
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt_text)])
        print("Routing LLM Response:", response.content.strip().upper())
        return response.content.strip().upper()
    except Exception as e:
        print("Routing LLM failed. Defaulting to 'BROWSER'.")
        print(f"Exception details: {e}")
        return "BROWSER" # Default fallback

async def generate_shell_command(user_input: str, llm, history: list = None) -> str:
    """Converts a natural language request into a shell command."""
    
    history_text = ""
    if history:
        history_text = "Previous interactions:\n"
        for h in history:
            history_text += f"- User: {h['user']}\n  Assistant: {h.get('assistant', '')}\n"

    prompt_text = f"""
    Convert the following natural language request into a single valid shell command (for macOS/Linux).
    Return ONLY the command, no markdown, no explanations.
    
    {history_text}

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

async def generate_splunk_query(user_input: str, llm, history: list = None) -> str:
    """Converts a natural language request into a Splunk SPL query, considering history."""
    
    history_text = ""
    if history:
        history_text = "Previous interactions:\n"
        for h in history:
            history_text += f"- User: {h['user']}\n  Assistant: {h.get('assistant', '')}\n"
    
    prompt_text = f"""
    Convert the following natural language request into a valid Splunk SPL query.
    Return ONLY the query, no markdown, no explanations.
    
    {history_text}
    
    IMPORTANT RULES:
    1. If the request is a follow-up (e.g., "put it in a table", "sort by time"), MODIFY the previous SPL query from the history.
    2. If the request is a CORRECTION (e.g., "I meant index=main", "typo in field name"), apply the correction to the PREVIOUS query, preserving the rest of the logic (e.g., keep "| fieldsummary").
    3. Do NOT use 'outputformat', 'outputcsv', 'outputjson', or any CLI-specific formatting commands. The system handles output format.
    4. If the user asks for "html" or "json" format, IGNORE the format request and just return the data query.
    5. For "table" commands, you MUST specify fields (e.g., "| table index count").
    6. SPL SYNTAX WARNING: Do NOT use `top` inside `stats`. `stats` only supports aggregations (avg, max, sum, count, values, list). Use `top` as a separate pipe command AFTER `stats` or instead of it.
       - INVALID: | stats count, top(user)
       - VALID: | top user
       - VALID: | stats count by user | sort -count
    
    Common SPL patterns:
    - "get all indexes": | eventcount summarize=false index=* | dedup index | fields index
    - "search index X": search index=X
    - "errors in last 15 mins": search index=* error earliest=-15m
    - Follow-up "table format": [Previous Query] | table field1 field2
    - Correction example: 
       User: "index=main" (after previous query "index=wrong | stats count")
       Result: "search index=main | stats count"
    
    Request: "{user_input}"
    """
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt_text)])
        query = response.content.strip()
        # Remove markdown code blocks if present
        if query.startswith("```"):
            query = query.strip("`").replace("splunk", "").strip()
            
        # Failsafe: Remove known invalid commands that the LLM might hallucinate
        forbidden_commands = ["outputformat", "display", "outputcsv", "outputjson"]
        for cmd in forbidden_commands:
            # Check for pipe-prefixed command
            if f"| {cmd}" in query:
                print(f"Warning: Removed invalid command '{cmd}' from query.")
                query = query.split(f"| {cmd}")[0].strip()
        
        # Failsafe: Fix common "stats ... top(...)" hallucination
        if "| stats" in query and "top(" in query:
            # Check if top is used inside stats (heuristic)
            stats_part = query.split("| stats")[1].split("|")[0]
            if "top(" in stats_part:
                print("Warning: Detected invalid 'top()' inside 'stats'. Replacing with 'values()'.")
                query = query.replace("top(", "values(")
        
        return query
        
        return query
    except Exception as e:
        print(f"Splunk query generation failed: {e}")
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

            chat_history = []

            while True:
                user_input = input(">> ").strip()
                
                if user_input.upper() == 'QUIT':
                    break
                
                if not user_input:
                    continue

                # Smart Routing
                intent = await route_intent(user_input, routing_llm, chat_history)
                print(f"Detected Intent: {intent}")

                tool_output = ""

                if intent == "TERMINAL":
                    tool_name = "run_terminal_task"
                    # If it has "term:" prefix, use the rest as command.
                    # If it's natural language (no prefix), generate the command.
                    if user_input.startswith("term:"):
                        cmd = user_input[5:].strip()
                    else:
                        print("Generating shell command...")
                        cmd = await generate_shell_command(user_input, routing_llm, chat_history)
                    
                    tool_args = {
                        "command": cmd,
                        "model_provider": llm_choice
                    }
                    print(f"Executing terminal command: '{cmd}'...")
                    tool_output = f"Executed command: {cmd}"
                elif intent == "SPLUNK":
                    tool_name = "run_splunk_query"
                    print("Generating Splunk query...")
                    spl_query = await generate_splunk_query(user_input, routing_llm, chat_history)
                    
                    tool_args = {
                        "query": spl_query
                    }
                    print(f"Executing Splunk query: '{spl_query}'...")
                    tool_output = f"Executed SPL: {spl_query}"
                elif intent == "GENERIC":
                    tool_name = "run_generic_action"
                    
                    # Construct data context with both immediate history and original data source
                    data_context = ""
                    
                    # 1. Get immediate last output
                    last_turn = chat_history[-1] if chat_history else None
                    if last_turn:
                        data_context += f"PREVIOUS OUTPUT:\n{last_turn.get('assistant', '')}\n\n"
                    
                    # 2. Find last raw data (Splunk/Terminal)
                    raw_data = ""
                    for turn in reversed(chat_history):
                        t_name = turn.get("tool", "")
                        if t_name in ["run_splunk_query", "run_terminal_task", "run_browser_task", "run_generic_action"]:
                            # Avoid duplication if the last turn WAS the raw data
                            if turn != last_turn:
                                raw_data = turn.get("assistant", "")
                            break
                    
                    if raw_data:
                        data_context += f"ORIGINAL DATA SOURCE:\n{raw_data}\n"
                    
                    if not data_context:
                        data_context = "No previous output found."

                    tool_args = {
                        "data": data_context,
                        "instruction": user_input,
                        "model_provider": llm_choice
                    }
                    print(f"Executing Generic Action on previous output...")
                    tool_output = f"Generic Action: {user_input}"
                else:
                    tool_name = "run_browser_task"
                    tool_args = {
                        "task": user_input,
                        "model_provider": llm_choice
                    }
                    print(f"Sending task to browser agent: '{user_input}'...")
                    tool_output = f"Browser Task: {user_input}"
                
                try:
                    result = await session.call_tool(
                        name=tool_name,
                        arguments=tool_args
                    )
                    
                    print("\n" + "#"*50)
                    print("RESULT:")
                    # The result content is a list of TextContent or ImageContent
                    full_result_text = ""
                    for content in result.content:
                        if content.type == 'text':
                            print(content.text)
                            full_result_text += content.text + "\n"
                        else:
                            print(f"[{content.type} content]")
                            full_result_text += f"[{content.type} content]\n"
                    print("#"*50 + "\n")

                    print("#"*50 + "\n")

                    # Update history with the ACTUAL result content and the TOOL NAME
                    chat_history.append({
                        "user": user_input, 
                        "assistant": full_result_text,
                        "tool": tool_name
                    })
                    # Keep history manageable
                    if len(chat_history) > 10:
                        chat_history.pop(0)
                    
                except Exception as e:
                    print(f"\n[ERROR] Tool execution failed: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nFatal Error: {e}")
