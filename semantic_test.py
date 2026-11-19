import json
import requests
import math


def embed(text):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": text
        }
    )
    return response.json()["embeddings"][0]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))

    return dot / (mag_a * mag_b)


# Load CEC rules
with open("cec_rules.json", "r", encoding="utf-8") as f:
    rules = json.load(f)

query = "How do I ground a separately derived AC system?"

print("Creating query embedding...")
query_embedding = embed(query)

results = []

for rule in rules:
    text = f"{rule['rule']} {rule['title']} {rule['text']}"

    rule_embedding = embed(text)

    similarity = cosine_similarity(
        query_embedding,
        rule_embedding
    )

    results.append((similarity, rule))


results.sort(reverse=True, key=lambda x: x[0])

print("\nTop results:\n")

for similarity, rule in results[:10]:
    print("=" * 70)
    print(f"Similarity: {similarity:.4f}")
    print(f"{rule['rule']} — {rule['title']}")

