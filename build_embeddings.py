import json
import requests
import time

INPUT_FILE = "cec_rules.json"
OUTPUT_FILE = "cec_embeddings.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    rules = json.load(f)

# Reuse embeddings from the previous build when a rule's text is unchanged,
# so only new or edited rules are sent to the embedding model.
cache = {}
try:
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for old in json.load(f):
            cache[(old["rule"], old["title"], old["text"])] = old["embedding"]
except (FileNotFoundError, json.JSONDecodeError):
    pass

results = []
reused = 0

start = time.time()

for i, rule in enumerate(rules, 1):
    text = f"{rule['rule']} {rule['title']} {rule['text']}"

    key = (rule["rule"], rule["title"], rule["text"])
    if key in cache:
        results.append({
            "rule": rule["rule"],
            "title": rule["title"],
            "text": rule["text"],
            "embedding": cache[key]
        })
        reused += 1
        continue

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
