import json
import ast
import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
input_file = DATA_DIR / "dataset.jsonl"
output_file = DATA_DIR / "dataset_cleaned.jsonl"

def clean_dataset():
    valid_count = 0
    invalid_count = 0

    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                record = json.loads(line)
                assistant_code = None
                for msg in record.get("messages", []):
                    if msg.get("role") == "assistant":
                        assistant_code = msg.get("content")
                        break
                
                if assistant_code:
                    try:
                        ast.parse(assistant_code)
                        fout.write(line)
                        valid_count += 1
                    except SyntaxError:
                        invalid_count += 1
                else:
                    invalid_count += 1
            except json.JSONDecodeError:
                pass

    print(f"Valid: {valid_count}, Invalid: {invalid_count}")
    os.replace(output_file, input_file)
    print("Dataset cleaned successfully.")

if __name__ == "__main__":
    clean_dataset()
