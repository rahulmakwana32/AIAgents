import os
import asyncio
from dotenv import load_dotenv
import sys 

# Browser-Use and LangChain Imports
from browser_use import Agent, Browser, BrowserProfile 
# Required custom LLM wrappers for browser-use compatibility
from browser_use.llm import ChatGoogle, ChatOpenAI 
from langchain_core.language_models import BaseChatModel

# Load environment variables from .env file
load_dotenv()

# --- 1. LLM Selection Function (No Change) ---

def initialize_llm(llm_choice: str) -> BaseChatModel:
    """Initializes the selected LLM (Gemini or OpenAI) using browser_use wrappers."""
    
    if llm_choice.lower() == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        print("💡 Using Google Gemini 2.5 Flash wrapper for vision and reasoning.")
        return ChatGoogle(
            model="gemini-2.5-flash", 
            api_key=api_key,
            temperature=0
        )
        
    elif llm_choice.lower() == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        print("💡 Using OpenAI GPT-4o wrapper for vision and reasoning.")
        return ChatOpenAI(
            model="gpt-4o-mini", 
            api_key=api_key,
            temperature=0
        )
        
    else:
        raise ValueError("Invalid LLM choice. Please choose 'Gemini' or 'OpenAI'.")

# --- 2. Asynchronous Input Helper (No Change) ---

async def ainput(prompt: str) -> str:
    """Runs the blocking input() function in a separate thread to prevent blocking the asyncio loop."""
    return await asyncio.to_thread(input, prompt)


# --- 3. Main Execution Function (FIX APPLIED HERE) ---

async def run_visible_agent():
    """Sets up the visible browser and runs the AI agent in a continuous, persistent loop."""
    
    # --- 3.1 Setup and Browser Launch ---
    print("--- AI Browser Agent Setup ---")
    
    llm_choice = input("Enter your preferred LLM (Gemini/OpenAI): ").strip().lower()
    try:
        llm = initialize_llm(llm_choice)
    except Exception as e:
        print(f"Setup Error: {e}")
        return

    browser_config = BrowserProfile( 
        headless=False,
        slow_mo=500,
        viewport_width=1280,
        viewport_height=900
    )

    browser = Browser(browser_profile=browser_config)
    print("Starting visible browser session...")
    await browser.start()

    print("\n" + "="*50)
    print("BROWSER READY. ENTER TASKS BELOW.")
    print("Type 'QUIT' to close the browser and exit.")
    print("="*50 + "\n")
    
    # --- 3.2 Interactive Task Loop (Persistent Session) ---
    
    while True:
        task = await ainput(">> ")
        
        if task.upper() == 'QUIT':
            break

        if not task.strip():
            print("Task cannot be empty. Please enter instructions.")
            continue

        try:
            agent = Agent(
                task=task,
                llm=llm,
                browser=browser 
            )

            print("\n" + "~"*50)
            print(f"AGENT ACTION: Executing task '{task[:50]}...'")
            print("~"*50)
            
            # Run the agent until it finds the final answer
            result = await agent.run()

            # --- Final Output (FIXED) ---
            print("\n" + "#"*50)
            print("AGENT FINAL ANSWER:")
            
            # FIX: Use .final_answer property instead of .text
            final_answer = "Agent run completed, but no final answer was extracted."
            try:
                final_answer = result.final_answer
                if final_answer is None:
                    final_answer = "Agent run failed or finished without extracting a final text answer."
            except AttributeError:
                final_answer = "ERROR: Could not retrieve final answer. The library's result structure may have changed again."
            
            print(final_answer)
            print("#"*50 + "\n")
            
            # Conversational prompt
            print("✅ Task completed successfully. The browser is ready for your next instruction.")
            print("What is your next action? (Type 'QUIT' to exit)")
            

        except Exception as e:
            # Handle the exception and keep the session open
            print(f"\n[ERROR] An agent execution error occurred: {e}")
            print("The browser remains open. Please try a different task or 'QUIT'.")

    # --- 3.3 Cleanup ---
    await browser.stop()
    print("\nBrowser closed. Exiting agent.")


# --- 4. Run the Async Main Function (No Change) ---

if __name__ == "__main__":
    try:
        asyncio.run(run_visible_agent())
    except KeyboardInterrupt:
        print("\nProgram interrupted. Attempting graceful exit...")
    except Exception as e:
        print(f"A final error occurred: {e}")