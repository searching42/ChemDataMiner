import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from src.config import FILES

# 加载环境变量
load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = "xxx"    #将xxx替换成llm api的网址

SYSTEM_PROMPT = """
You are a chemist. Extract structured data from the table.
Output valid JSON only. Format: {"data": [{"original_name": "...", "smiles": "...", "properties": {"HOMO": "..."}}]}
Context is provided to help resolve aliases to SMILES.
"""

def process_line(client, line):
    record = json.loads(line)
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Table:\n{record.get('table_content')}\nContext:\n{record.get('extracted_context')}"}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return {**record, "extracted_data": result.get("data", [])}
    except Exception as e:
        return {**record, "error": str(e)}

def run():
    print(">>> STEP 3: Running LLM Inference...")
    if not API_KEY:
        raise ValueError("Please set DEEPSEEK_API_KEY in .env file")
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    with open(FILES['llm_input'], 'r') as f:
        lines = f.readlines()
        
    with open(FILES['llm_output'], 'w') as out:
        for line in tqdm(lines):
            res = process_line(client, line)
            out.write(json.dumps(res) + "\n")

if __name__ == "__main__":
    run()