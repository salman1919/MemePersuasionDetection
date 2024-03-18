import json

# Read the JSON file
with open('output.json', 'r', encoding="utf8") as file:
    data1 = json.load(file)

# Extract IDs from the JSON data
ids_pre = [entry['id'] for entry in data1]

# Print the IDs
print("IDs extracted from JSON file:")
print(ids_pre)


# Read the JSON file
with open('.\dataset\dev_gold_labels\dev_subtask2a_en.json', 'r', encoding="utf8") as file:
    data2 = json.load(file)

# Extract IDs from the JSON data
ids_gold = [entry['id'] for entry in data2]
# Print the IDs
print("IDs extracted from JSON file:")
print(ids_gold)


for id in ids_pre:
    if id in ids_gold:
        print(id)