import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen3:8b",
        "prompt": "Explain Ohm's law to an electrician in simple terms.",
        "stream": False
    }
)

data = response.json()

print(data["response"])
