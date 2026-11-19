"""Show the rules retrieved for a question, without calling the chat model.

Usage:  python -m scripts.semantic_search "How do I ground a separately derived AC system?"
"""

from __future__ import annotations

import sys

from cec.ollama import OllamaError
from cec.retrieval import RuleIndex


def main() -> int:
    query = " ".join(sys.argv[1:]) or input("Ask the CEC: ")
    try:
        matches = RuleIndex.load().search(query, k=10)
    except OllamaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("\nTop results:\n")
    for m in matches:
        print("=" * 70)
        print(f"Similarity: {m.score:.4f}")
        print(f"{m.rule.rule} — {m.rule.title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
