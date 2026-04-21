import json

with open(r'c:\Users\bhati\Downloads\inputs.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total items: {len(data)}")
print(f"Keys in first item: {list(data[0].keys())}")
print(f"Keys in last item: {list(data[-1].keys())}")

ids = [d['id'] for d in data]
print(f"ID range: {min(ids)}-{max(ids)}")

# Check for missing IDs
expected_ids = set(range(min(ids), max(ids) + 1))
actual_ids = set(ids)
missing = expected_ids - actual_ids
duplicates = [i for i in ids if ids.count(i) > 1]
print(f"Missing IDs: {missing if missing else 'None'}")
print(f"Duplicate IDs: {set(duplicates) if duplicates else 'None'}")

# Check consistent keys
keys0 = set(data[0].keys())
for d in data:
    if set(d.keys()) != keys0:
        print(f"  Item {d.get('id', '?')}: has keys {set(d.keys())} instead of {keys0}")

# Check for empty or null fields
for d in data:
    for k, v in d.items():
        if v is None or (isinstance(v, str) and v.strip() == ''):
            print(f"  Item {d['id']}: field '{k}' is empty/null")

# Check for truncated text
for d in data:
    text = d.get('text', '')
    if len(text) < 100:
        print(f"  Item {d['id']}: text is very short ({len(text)} chars)")

print("Done.")
