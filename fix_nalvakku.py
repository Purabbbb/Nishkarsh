import json, sys
sys.stdout.reconfigure(encoding='utf-8')

# Fix the one unmatched item
with open(r'c:\Users\bhati\OneDrive\Desktop\outputs - Copy.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Find the unmatched "Nalvakku" entry and check the inputs for its ID
with open(r'c:\Users\bhati\Downloads\inputs.json', 'r', encoding='utf-8') as f:
    inputs = json.load(f)

# Find Nalvakku in inputs
for inp in inputs:
    if 'nalvakku' in inp['file_name'].lower():
        print(f"Found in inputs: ID={inp['id']}, file={inp['file_name']}")
        nalvakku_id = inp['id']
        break

# Fix in outputs
for item in data:
    if 'id' not in item:
        print(f"Fixing: '{item['title']}'")
        item['id'] = nalvakku_id
        # Reorder
        new_item = {'id': item['id']}
        for k, v in item.items():
            if k != 'id':
                new_item[k] = v
        data[data.index(item)] = new_item

# Verify all have IDs
missing = [d for d in data if 'id' not in d]
print(f"Items without ID: {len(missing)}")

with open(r'c:\Users\bhati\OneDrive\Desktop\outputs - Copy.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Fixed and saved!")
