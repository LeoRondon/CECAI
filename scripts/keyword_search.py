"""Plain substring search over cec_rules.json (no Ollama needed).

Usage:  python -m scripts.keyword_search "ground fault"
"""

from __future__ import annotations

import json
import sys

from cec.config import settings


def main() -> None:
    query = (" ".join(sys.argv[1:]) or input("Search CEC: ")).lower()
    with open(settings.rules_file, encoding="utf-8") as f:
        rules = json.load(f)

    results = [r for r in rules if query in f"{r['rule']} {r['title']} {r['text']}".lower()]
    print(f"\nFound {len(results)} results\n")
    for rule in results[:10]:
        print("=" * 70)
        print(f"{rule['rule']} — {rule['title']}")
        print(rule["text"][:500])


if __name__ == "__main__":
    main()
