"""Smoke-test the Ollama setup: both models respond.

Usage:  python -m scripts.check_ollama
"""

from __future__ import annotations

import sys

from cec import ollama
from cec.config import settings


def main() -> int:
    print(f"Ollama: {settings.ollama_host}")
    try:
        vector = ollama.embed_one("How do I ground a separately derived AC system?")
        print(f"✓ {settings.embed_model}: {len(vector)}-dimension embeddings")
        reply = ollama.generate("Reply with the single word: ready")
        print(f"✓ {settings.chat_model}: {reply[:80]!r}")
    except ollama.OllamaError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
