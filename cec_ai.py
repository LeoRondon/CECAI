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


with open("cec_embeddings.json", "r", encoding="utf-8") as f:
    rules = json.load(f)


question = input("CEC Question: ")

question_embedding = embed(question)

results = []

for rule in rules:
    similarity = cosine_similarity(
        question_embedding,
        rule["embedding"]
    )

    results.append((similarity, rule))


results.sort(reverse=True, key=lambda x: x[0])

top_rules = results[:5]


context = ""

for similarity, rule in top_rules:
    context += f"""
Rule {rule['rule']} — {rule['title']}

{rule['text']}

"""


prompt = f"""
You are a Canadian Electrical Code assistant.

Answer the user's question using ONLY the CEC material provided below.

IMPORTANT RULES:

1. Do not use outside electrical knowledge.
2. Do not invent requirements, numbers, exceptions, definitions, or rule references.
3. Do not infer that a retrieved rule contains a requirement unless that requirement is explicitly present in the supplied text.
4. Prefer the rule that directly answers the question over rules that are merely related.
5. When describing an exception, state only what the supplied CEC text actually says.
6. Identify the specific CEC rule number that supports each important requirement.
7. If the supplied material does not contain enough information to answer confidently, say so.
8. Do not cite a rule simply because it was retrieved. Cite it only if its supplied text supports the statement.
Do not invent requirements, numbers, exceptions, or rules.

CEC material:

{context}

User question:

{question}
"""


response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen3:8b",
        "prompt": prompt,
        "stream": False
    }
)

data = response.json()

print("\n" + "=" * 70)
print("CEC ASSISTANT")
print("=" * 70)
print(data["response"])

print("\nRelevant rules:")
for similarity, rule in top_rules:
    print(f"- {rule['rule']} — {rule['title']} ({similarity:.3f})")
