"""Runtime settings, read from environment variables with sensible defaults.

| Variable              | Default                   | Purpose                               |
|-----------------------|---------------------------|---------------------------------------|
| ``OLLAMA_HOST``       | ``http://localhost:11434`` | Base URL of the Ollama server        |
| ``CEC_CHAT_MODEL``    | ``qwen3:8b``              | Model that writes the answer          |
| ``CEC_EMBED_MODEL``   | ``nomic-embed-text``      | Model used for embeddings             |
| ``CEC_TOP_K``         | ``5``                     | Number of rules retrieved per question |
| ``CEC_TIMEOUT``       | ``300``                   | Seconds to wait on Ollama responses   |
| ``CEC_EMBEDDINGS_FILE`` | ``cec_embeddings.json`` | Pre-computed rule embeddings          |
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _path_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


@dataclass(frozen=True)
class Settings:
    ollama_host: str
    chat_model: str
    embed_model: str
    top_k: int
    timeout: int
    rules_file: Path
    embeddings_file: Path
    cec_text_file: Path

    @classmethod
    def from_env(cls) -> Settings:
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        return cls(
            ollama_host=host,
            chat_model=os.environ.get("CEC_CHAT_MODEL", "qwen3:8b"),
            embed_model=os.environ.get("CEC_EMBED_MODEL", "nomic-embed-text"),
            top_k=_int_env("CEC_TOP_K", 5),
            timeout=_int_env("CEC_TIMEOUT", 300),
            rules_file=_path_env("CEC_RULES_FILE", ROOT / "cec_rules.json"),
            embeddings_file=_path_env("CEC_EMBEDDINGS_FILE", ROOT / "cec_embeddings.json"),
            cec_text_file=_path_env("CEC_TEXT_FILE", ROOT / "pdf" / "cec.txt"),
        )


settings = Settings.from_env()
