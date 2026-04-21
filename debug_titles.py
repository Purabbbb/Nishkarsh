import json
from difflib import SequenceMatcher

# Load both files
with open(r'c:\Users\bhati\Downloads\inputs.json', 'r', encoding='utf-8') as f:
    inputs = json.load(f)

with open(r'c:\Users\bhati\OneDrive\Desktop\outputs - Copy.json', 'r', encoding='utf-8') as f:
    outputs = json.load(f)

print(f"Inputs: {len(inputs)} items")
print(f"Outputs: {len(outputs)} items")

# Show first 5 input titles and file_names
for d in inputs[:5]:
    print(f"  ID {d['id']}: title='{d['title'][:60]}' | file='{d['file_name'][:60]}'")

# Show first 5 output titles
print("\nFirst 5 output titles:")
for d in outputs[:5]:
    print(f"  '{d['title'][:80]}'")
