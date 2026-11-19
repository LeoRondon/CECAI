"""Command-line CEC Assistant.

Usage:
    python cec_ai.py "What size grounding conductor is required?"
    python cec_ai.py            # prompts for a question
"""

from __future__ import annotations

import argparse
import sys

from cec import ollama
from cec.config import settings
from cec.prompts import build_prompt
from cec.retrieval import RuleIndex


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the Canadian Electrical Code.")
    parser.add_argument("question", nargs="*", help="question to ask")
    parser.add_argument("-k", "--top-k", type=int, default=settings.top_k, help="rules to retrieve")
    parser.add_argument("-m", "--model", default=settings.chat_model, help="Ollama chat model")
    args = parser.parse_args()

    question = " ".join(args.question) or input("CEC Question: ")
    if not question.strip():
        parser.error("please provide a question")

    try:
        matches = RuleIndex.load().search(question, args.top_k)
        print("\n" + "=" * 70 + "\nCEC ASSISTANT\n" + "=" * 70)
        started = False
        for chunk in ollama.generate_stream(build_prompt(question, matches), args.model):
            if chunk.kind == "answer":
                text = chunk.text if started else chunk.text.lstrip()
                started = started or bool(text)
                print(text, end="", flush=True)
        print()
    except ollama.OllamaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("\nRelevant rules:")
    for m in matches:
        print(f"- {m.rule.rule} — {m.rule.title} ({m.score:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
