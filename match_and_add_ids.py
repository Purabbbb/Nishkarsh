import json
from difflib import SequenceMatcher
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load both files
with open(r'c:\Users\bhati\Downloads\inputs.json', 'r', encoding='utf-8') as f:
    inputs = json.load(f)

with open(r'c:\Users\bhati\OneDrive\Desktop\outputs - Copy.json', 'r', encoding='utf-8') as f:
    outputs = json.load(f)

print(f"Inputs: {len(inputs)} items")
print(f"Outputs: {len(outputs)} items")

# Build lookup from file_name (without .pdf) to id
def normalize(s):
    return s.strip().lower().replace('.pdf', '').replace('-', ' ').replace('  ', ' ')

input_lookup = {}
for item in inputs:
    norm = normalize(item['file_name'])
    input_lookup[norm] = item['id']

# Match and add IDs
matched = 0
unmatched = []

for out_item in outputs:
    out_title = out_item['title']
    out_norm = normalize(out_title)
    
    # Try exact normalized match
    if out_norm in input_lookup:
        out_item['id'] = input_lookup[out_norm]
        matched += 1
        continue
    
    # Try fuzzy match against file_names
    best_ratio = 0
    best_id = None
    best_fname = None
    for inp_item in inputs:
        inp_norm = normalize(inp_item['file_name'])
        ratio = SequenceMatcher(None, out_norm, inp_norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_id = inp_item['id']
            best_fname = inp_item['file_name']
    
    if best_ratio >= 0.7:
        out_item['id'] = best_id
        matched += 1
        print(f"  Fuzzy match ({best_ratio:.2f}): '{out_title[:70]}' -> ID {best_id}")
    else:
        unmatched.append((out_title, best_ratio, best_fname))
        print(f"  NO MATCH ({best_ratio:.2f}): '{out_title[:70]}' ~~ '{best_fname}'")

print(f"\nMatched: {matched}/{len(outputs)}")
if unmatched:
    print(f"Unmatched: {len(unmatched)}")

# Reorder keys so 'id' comes first
reordered = []
for item in outputs:
    if 'id' in item:
        new_item = {'id': item['id']}
        for k, v in item.items():
            if k != 'id':
                new_item[k] = v
        reordered.append(new_item)
    else:
        reordered.append(item)

# Save the updated outputs file
with open(r'c:\Users\bhati\OneDrive\Desktop\outputs - Copy.json', 'w', encoding='utf-8') as f:
    json.dump(reordered, f, indent=2, ensure_ascii=False)

print("\nSaved updated 'outputs - Copy.json' with IDs!")
