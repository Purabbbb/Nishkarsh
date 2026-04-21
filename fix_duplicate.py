import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'c:\Users\bhati\OneDrive\Desktop\outputs - Copy.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Find the duplicate BioMDSum entries
biomdsum_items = [(i, d) for i, d in enumerate(data) if d['id'] == 22]
print(f"Found {len(biomdsum_items)} items with ID 22:")
for idx, item in biomdsum_items:
    print(f"  Index {idx}: title='{item['title'][:70]}'")
    print(f"    summary starts: '{item['summary'][:100]}'")
    print()

# Check if summaries are identical
if biomdsum_items[0][1]['summary'] == biomdsum_items[1][1]['summary']:
    print("Summaries are IDENTICAL - removing duplicate")
else:
    print("Summaries DIFFER - keeping both but flagging")
    print(f"  First:  '{biomdsum_items[0][1]['summary'][:150]}'")
    print(f"  Second: '{biomdsum_items[1][1]['summary'][:150]}'")

# Remove the second duplicate
second_idx = biomdsum_items[1][0]
data.pop(second_idx)
print(f"\nRemoved duplicate at index {second_idx}")
print(f"New total: {len(data)}")

with open(r'c:\Users\bhati\OneDrive\Desktop\outputs - Copy.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Saved!")
