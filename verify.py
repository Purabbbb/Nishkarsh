import json, sys
sys.stdout.reconfigure(encoding='utf-8')

data = json.load(open(r'c:\Users\bhati\OneDrive\Desktop\outputs - Copy.json', 'r', encoding='utf-8'))
print(f"Total items: {len(data)}")
print(f"All have 'id': {all('id' in d for d in data)}")
print(f"Keys: {list(data[0].keys())}")

# Show a few samples
for d in data[:3]:
    print(f"  id={d['id']}, title='{d['title'][:60]}'")

# Check for duplicate IDs
ids = [d['id'] for d in data]
from collections import Counter
dupes = {k: v for k, v in Counter(ids).items() if v > 1}
if dupes:
    print(f"\nDuplicate IDs found: {dupes}")
    for dup_id, count in dupes.items():
        for d in data:
            if d['id'] == dup_id:
                print(f"  ID {dup_id}: '{d['title'][:60]}'")
else:
    print("\nNo duplicate IDs.")
