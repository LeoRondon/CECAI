import json

import numpy as np
import pytest

from cec.prompts import build_prompt
from cec.retrieval import Match, Rule, RuleIndex


def make_index() -> RuleIndex:
    rules = [Rule("1-000", "A", "alpha"), Rule("2-000", "B", "beta"), Rule("3-000", "C", "gamma")]
    vectors = np.array([[1, 0, 0], [0, 2, 0], [1, 1, 0]], dtype=float)
    return RuleIndex(rules, vectors)


def test_search_orders_by_cosine_similarity():
    matches = make_index().search_vector([0, 1, 0], k=3)
    assert [m.rule.rule for m in matches] == ["2-000", "3-000", "1-000"]
    assert matches[0].score == pytest.approx(1.0)
    assert matches[1].score == pytest.approx(1 / np.sqrt(2))
    assert matches[2].score == pytest.approx(0.0)


def test_search_matches_pure_python_cosine():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(50, 16))
    rules = [Rule(str(i), "", "") for i in range(50)]
    query = rng.normal(size=16)

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    expected = sorted(range(50), key=lambda i: cosine(query, vectors[i]), reverse=True)[:5]
    matches = RuleIndex(rules, vectors).search_vector(query, k=5)
    assert [int(m.rule.rule) for m in matches] == expected


def test_k_is_capped_at_index_size():
    assert len(make_index().search_vector([1, 0, 0], k=50)) == 3


def test_load_from_json(tmp_path):
    path = tmp_path / "emb.json"
    path.write_text(json.dumps([{"rule": "8-100", "title": "T", "text": "x", "embedding": [0.0, 1.0]}]))
    index = RuleIndex.load(path)
    assert len(index) == 1
    assert index.search_vector([0, 1], k=1)[0].rule.rule == "8-100"


def test_prompt_contains_rules_and_question():
    prompt = build_prompt("  Why? ", [Match(Rule("4-004", "Ampacity", "Body text"), 0.9)])
    assert "Rule 4-004 — Ampacity\n\nBody text" in prompt
    assert prompt.rstrip().endswith("USER QUESTION:\n\nWhy?")
