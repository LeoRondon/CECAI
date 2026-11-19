"""Embed every rule in cec_rules.json and save the result to cec_embeddings.json.

Embeddings from a previous build are reused when a rule's text is unchanged,
so only new or edited rules are sent to the embedding model.
"""

from __future__ import annotations

import json
import sys
import time

from cec import ollama
from cec.config import settings


def main() -> int:
    with open(settings.rules_file, encoding="utf-8") as f:
        rules = json.load(f)

    cache: dict[tuple[str, str, str], list[float]] = {}
    try:
        with open(settings.embeddings_file, encoding="utf-8") as f:
            for old in json.load(f):
                cache[(old["rule"], old["title"], old["text"])] = old["embedding"]
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    results = []
    reused = 0
    start = time.time()

    try:
        for i, rule in enumerate(rules, 1):
            key = (rule["rule"], rule["title"], rule["text"])
            embedding = cache.get(key)
            if embedding is not None:
                reused += 1
            else:
                embedding = ollama.embed_one(f"{rule['rule']} {rule['title']} {rule['text']}")
            results.append(
                {"rule": rule["rule"], "title": rule["title"], "text": rule["text"], "embedding": embedding}
            )
            if i % 100 == 0:
                print(f"Processed {i}/{len(rules)} rules")
    except ollama.OllamaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    with open(settings.embeddings_file, "w", encoding="utf-8") as f:
        json.dump(results, f)

    print(f"\nFinished in {time.time() - start:.1f} seconds ({reused} reused)")
    print(f"Saved to {settings.embeddings_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
