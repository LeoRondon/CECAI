import json

with open("cec_rules.json", "r", encoding="utf-8") as f:
    rules = json.load(f)

query = input("Search CEC: ").lower()

results = []

for rule in rules:
    text = f"{rule['rule']} {rule['title']} {rule['text']}".lower()

    if query in text:
        results.append(rule)

print(f"\nFound {len(results)} results\n")

for rule in results[:10]:
    print("=" * 70)
    print(f"{rule['rule']} — {rule['title']}")
    print(rule["text"][:500])

