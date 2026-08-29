import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load settings from .env, overriding any existing system variables
load_dotenv(override=True)

# 2. Retrieve variables from .env
API_KEY = os.getenv("GEMINI_API_KEY")
PROXY_URL = os.getenv("PROXY_URL")  # e.g., socks5://127.0.0.1:10808

if not API_KEY:
    raise ValueError("ERROR: GEMINI_API_KEY not found in .env!")

# 3. Configure the proxy via environment variables
if PROXY_URL:
    print(f"[*] Using proxy: {PROXY_URL}")
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL
    os.environ["ALL_PROXY"] = PROXY_URL
    
    # Clean up any bad socks configurations that Windows might inject
    for key in ['http_proxy', 'https_proxy', 'all_proxy']:
        os.environ[key] = PROXY_URL
else:
    print("[!] NO PROXY SET. Request is going directly.")

# 4. Initialize the Google Client
client = genai.Client(api_key=API_KEY)

def refactor_blueprint_code(flat_code: str) -> dict:
    """
    Sends flat code to Gemini and returns a JSON containing 
    the refactored code and synthetic user prompts.
    """
    prompt = f"""
    Here is an automatically decompiled, flat Factorio blueprint code:
    
    ```python
    {flat_code}
    ```
    
    Do the following:
    1. Refactor the code. Find repeating patterns (belt lines, furnace rows, etc.) and replace them with `for` loops and variables.
    2. Create 3 human-like user prompts (in English and Russian) that a player might use to request this exact blueprint.
    
    Return the result strictly in this JSON format:
    {{
        "refactored_code": "string containing the full python code",
        "user_prompts": ["prompt 1", "prompt 2", "prompt 3"]
    }}
    """
    
    config = types.GenerateContentConfig(
        system_instruction=(
            "You are an expert in Python and the game Factorio. Your task is to take 'flat' blueprint "
            "generation code using the `factorio-draftsman` library and refactor it. "
            "Use `for` loops, mathematical calculations, and variables. "
            "The code must be compact, logical, and algorithmic. Preserve the exact geometry and classes. "
            "RETURN THE RESPONSE STRICTLY IN JSON FORMAT."
        ),
        response_mime_type="application/json",
    )
    
    model_name = 'gemini-3.5-flash-lite'
    print(f"[*] Sending request to {model_name}...")
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config
        )
        # Parse the JSON string from the model
        return json.loads(response.text)
    except Exception as e:
        print(f"[!] API Request Error: {e}")
        return None

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Get the directory where this script (ai_refactor.py) is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define absolute paths for input and output files
    input_file_path = os.path.join(script_dir, "decompiled_test.py")
    output_file_path = os.path.join(script_dir, "refactored_output.py")
    
    # Check if the decompiled file actually exists
    if not os.path.exists(input_file_path):
        print(f"[!] Error: Input file not found at {input_file_path}")
        print("Please make sure you have generated it using your decompiler first.")
        exit(1)
        
    print(f"[*] Reading flat code from: {input_file_path}")
    
    # Read the flat code
    with open(input_file_path, "r", encoding="utf-8") as f:
        flat_code = f.read()
        
    # Send to Gemini
    result = refactor_blueprint_code(flat_code)
    
    if result:
        print("\n=== GENERATED PROMPTS ===")
        for p in result['user_prompts']:
            print(f"- {p}")
            
        # Prepare the content for the new file
        # We save the prompts as a Python docstring at the top of the file
        output_content = '"""\nSynthetic User Prompts for this blueprint:\n'
        for p in result['user_prompts']:
            output_content += f"- {p}\n"
        output_content += '"""\n\n'
        
        # Append the AI-generated code
        output_content += result['refactored_code']
        
        # Write to the new file
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(output_content)
            
        print(f"\n[*] SUCCESS! Refactored code and prompts saved to:\n{output_file_path}")