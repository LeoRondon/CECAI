import json
import requests
import time

INPUT_FILE = "cec_rules.json"
OUTPUT_FILE = "cec_embeddings.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    rules = json.load(f)

results = []

start = time.time()

for i, rule in enumerate(rules, 1):
    text = f"{rule['rule']} {rule['title']} {rule['text']}"

    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": text
        }
    )

    embedding = response.json()["embeddings"][0]

    results.append({
        "rule": rule["rule"],
        "title": rule["title"],
        "text": rule["text"],
        "embedding": embedding
    })

    if i % 100 == 0:
        print(f"Processed {i}/{len(rules)} rules")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f)

elapsed = time.time() - start

print(f"\nFinished in {elapsed:.1f} seconds")
print(f"Saved to {OUTPUT_FILE}")
