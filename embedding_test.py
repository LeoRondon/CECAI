import requests

response = requests.post(
    "http://localhost:11434/api/embed",
    json={
        "model": "nomic-embed-text",
        "input": "How do I ground a separately derived AC system?"
    }
)

data = response.json()

embedding = data["embeddings"][0]

print("Embedding dimensions:", len(embedding))
print("First 10 values:", embedding[:10])
