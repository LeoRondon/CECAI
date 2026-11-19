import json

import pytest
import requests

from cec import ollama
from cec.ollama import Chunk, OllamaError, ThinkSplitter


def split(pieces: list[str]) -> list[Chunk]:
    splitter = ThinkSplitter()
    out = [c for p in pieces for c in splitter.feed(p)] + splitter.flush()
    # Merge adjacent chunks of the same kind to compare content, not chunking.
    merged: list[Chunk] = []
    for c in out:
        if merged and merged[-1].kind == c.kind:
            merged[-1] = Chunk(c.kind, merged[-1].text + c.text)
        else:
            merged.append(c)
    return merged


def test_plain_text_passes_through():
    assert split(["Hello ", "world"]) == [Chunk("answer", "Hello world")]


def test_think_block_is_separated():
    assert split(["<think>hmm</think>Answer"]) == [Chunk("thinking", "hmm"), Chunk("answer", "Answer")]


@pytest.mark.parametrize("size", [1, 2, 3, 5])
def test_tags_split_across_chunks(size):
    text = "<think>step one</think>\n\nFinal a < b"
    pieces = [text[i : i + size] for i in range(0, len(text), size)]
    assert split(pieces) == [Chunk("thinking", "step one"), Chunk("answer", "\n\nFinal a < b")]


def test_partial_tag_lookalike_is_released_on_flush():
    assert split(["x <thi"]) == [Chunk("answer", "x <thi")]


class FakeResponse:
    def __init__(self, status: int, lines: list[dict] | None = None, body: dict | None = None):
        self.status_code = status
        self._lines = [json.dumps(x).encode() for x in lines or []]
        self._body = body or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body

    def iter_lines(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_generate_stream_handles_thinking_field(monkeypatch):
    lines = [
        {"thinking": "reason", "response": ""},
        {"response": "Hi"},
        {"response": " there", "done": True},
    ]
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(200, lines))
    chunks = list(ollama.generate_stream("p", think=True))
    assert chunks == [Chunk("thinking", "reason"), Chunk("answer", "Hi"), Chunk("answer", " there")]


def test_generate_retries_without_think_for_unsupported_models(monkeypatch):
    calls = []

    def fake_post(url, json, **kwargs):
        calls.append(json)
        if "think" in json:
            return FakeResponse(400, body={"error": '"llama3" does not support thinking'})
        return FakeResponse(200, [{"response": "ok", "done": True}])

    monkeypatch.setattr(requests, "post", fake_post)
    assert ollama.generate("p", "llama3") == "ok"
    assert len(calls) == 2


def test_connection_error_is_friendly(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError()

    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(OllamaError, match="Can't reach Ollama"):
        ollama.embed_one("x")


def test_missing_model_suggests_pull(monkeypatch):
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: FakeResponse(404, body={"error": 'model "qwen3:8b" not found'})
    )
    with pytest.raises(OllamaError, match="ollama pull qwen3:8b"):
        ollama.generate("p", "qwen3:8b")
