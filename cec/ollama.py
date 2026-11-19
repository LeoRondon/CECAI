"""Thin client for the Ollama HTTP API with friendly error messages."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import requests

from cec.config import settings

ChunkKind = Literal["thinking", "answer"]


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error."""


@dataclass(frozen=True)
class Chunk:
    kind: ChunkKind
    text: str


def _post(path: str, payload: dict, *, stream: bool = False, timeout: int | None = None):
    url = f"{settings.ollama_host}{path}"
    try:
        response = requests.post(
            url,
            json=payload,
            stream=stream,
            # (connect timeout, read timeout)
            timeout=(5, timeout or settings.timeout),
        )
    except requests.ConnectionError as exc:
        raise OllamaError(
            f"Can't reach Ollama at {settings.ollama_host}. "
            "Make sure it is running (`ollama serve`) or set OLLAMA_HOST."
        ) from exc
    except requests.Timeout as exc:
        raise OllamaError(f"Ollama at {settings.ollama_host} timed out.") from exc

    if response.status_code != 200:
        try:
            message = response.json().get("error", response.text)
        except ValueError:
            message = response.text
        model = payload.get("model", "")
        if response.status_code == 404 and "not found" in message.lower():
            raise OllamaError(f"Model `{model}` isn't installed. Run `ollama pull {model}`.")
        raise OllamaError(f"Ollama error ({response.status_code}): {message}")
    return response


def embed(text: str | list[str], model: str | None = None) -> list[list[float]]:
    """Return one embedding per input text."""
    response = _post("/api/embed", {"model": model or settings.embed_model, "input": text})
    return response.json()["embeddings"]


def embed_one(text: str, model: str | None = None) -> list[float]:
    return embed(text, model)[0]


class ThinkSplitter:
    """Split a token stream into reasoning and answer text.

    Some models (e.g. qwen3) wrap their reasoning in ``<think>…</think>``
    inside the response itself. Tags can be split across stream chunks, so
    text that might be the start of a tag is held back until it's resolved.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._thinking = False

    def feed(self, text: str) -> list[Chunk]:
        self._buffer += text
        out: list[Chunk] = []
        while self._buffer:
            tag = self.CLOSE if self._thinking else self.OPEN
            kind: ChunkKind = "thinking" if self._thinking else "answer"
            index = self._buffer.find(tag)
            if index >= 0:
                if index:
                    out.append(Chunk(kind, self._buffer[:index]))
                self._buffer = self._buffer[index + len(tag):]
                self._thinking = not self._thinking
                continue
            # Hold back a suffix that could be the beginning of the tag.
            keep = next(
                (n for n in range(len(tag) - 1, 0, -1) if self._buffer.endswith(tag[:n])),
                0,
            )
            emit = self._buffer[: len(self._buffer) - keep]
            if emit:
                out.append(Chunk(kind, emit))
            self._buffer = self._buffer[len(emit):]
            break
        return out

    def flush(self) -> list[Chunk]:
        rest, self._buffer = self._buffer, ""
        return [Chunk("thinking" if self._thinking else "answer", rest)] if rest else []


def generate_stream(
    prompt: str, model: str | None = None, *, think: bool = False
) -> Iterator[Chunk]:
    """Stream a completion, yielding reasoning and answer chunks separately."""
    payload = {"model": model or settings.chat_model, "prompt": prompt, "stream": True, "think": think}
    try:
        response = _post("/api/generate", payload, stream=True)
    except OllamaError as exc:
        # Models without a thinking mode reject the `think` option.
        if "does not support thinking" not in str(exc):
            raise
        payload.pop("think")
        response = _post("/api/generate", payload, stream=True)

    splitter = ThinkSplitter()
    with response:
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if "error" in data:
                raise OllamaError(f"Ollama error: {data['error']}")
            if data.get("thinking"):
                yield Chunk("thinking", data["thinking"])
            if data.get("response"):
                yield from splitter.feed(data["response"])
            if data.get("done"):
                break
    yield from splitter.flush()


def generate(prompt: str, model: str | None = None) -> str:
    """Return the full answer text (reasoning removed)."""
    return "".join(c.text for c in generate_stream(prompt, model) if c.kind == "answer").strip()
