"""File database for prompt-keyed T5 conditioning caches."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from scail2_inference.conditioning import (
    NEGATIVE_CONTEXT,
    TEXT_CONTEXT,
    build_t5_metadata,
    load_t5_cache,
    save_t5_cache,
)

EMPTY_NEGATIVE_PROMPT = ""


def normalize_prompt(prompt: str) -> str:
    """Return the exact prompt representation used for hashing and encoding."""
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    normalized = prompt.strip()
    if not normalized:
        raise ValueError("prompt cannot be empty")
    return normalized


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(normalize_prompt(prompt).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class T5CacheRecord:
    prompt_hash: str
    path: Path
    cache_hit: bool


class T5CacheDatabase:
    """Store immutable T5 cache files under a SHA-256 directory index.

    The owning HTTP service serializes all calls. This class therefore needs no
    process or thread locking; atomic cache publication protects restart safety.
    """

    def __init__(
        self,
        root: Path,
        *,
        profile: str,
        text_len: int,
        t5_checkpoint: Path,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.text_len = int(text_len)
        self.t5_checkpoint = Path(t5_checkpoint).resolve(strict=True)

    def path_for_hash(self, prompt_hash: str) -> Path:
        if (
            len(prompt_hash) != 64
            or any(character not in "0123456789abcdef" for character in prompt_hash)
        ):
            raise ValueError("prompt hash must be 64 lowercase hexadecimal characters")
        return self.root / prompt_hash[:2] / f"{prompt_hash}.safetensors"

    def _expected_metadata(self, prompt_hash: str) -> dict[str, str]:
        from scail2_inference.conditioning import expected_t5_metadata

        expected = expected_t5_metadata(
            profile=self.profile,
            text_len=self.text_len,
            t5_checkpoint=self.t5_checkpoint,
        )
        expected.update(
            {
                "prompt_sha256": prompt_hash,
                "negative_prompt_sha256": hashlib.sha256(b"").hexdigest(),
            }
        )
        return expected

    def lookup_hash(self, prompt_hash: str) -> T5CacheRecord | None:
        path = self.path_for_hash(prompt_hash)
        if not path.is_file():
            return None
        load_t5_cache(path, self._expected_metadata(prompt_hash))
        return T5CacheRecord(prompt_hash=prompt_hash, path=path, cache_hit=True)

    def get_or_create(
        self,
        prompt: str,
        producer: Callable[[str], Mapping[str, object]],
    ) -> T5CacheRecord:
        normalized = normalize_prompt(prompt)
        prompt_hash = prompt_sha256(normalized)
        try:
            existing = self.lookup_hash(prompt_hash)
        except (OSError, RuntimeError, TypeError, ValueError):
            existing = None
        if existing is not None:
            return existing

        tensors = producer(normalized)
        if set(tensors) != {TEXT_CONTEXT, NEGATIVE_CONTEXT}:
            raise ValueError("T5 producer returned unexpected tensors")
        metadata = build_t5_metadata(
            prompt=normalized,
            negative_prompt=EMPTY_NEGATIVE_PROMPT,
            profile=self.profile,
            text_len=self.text_len,
            t5_checkpoint=self.t5_checkpoint,
        )
        path = self.path_for_hash(prompt_hash)
        save_t5_cache(path, tensors, metadata, overwrite=True)
        load_t5_cache(path, self._expected_metadata(prompt_hash))
        return T5CacheRecord(prompt_hash=prompt_hash, path=path, cache_hit=False)

    def statistics(self) -> dict[str, int]:
        files = list(self.root.glob("[0-9a-f][0-9a-f]/*.safetensors"))
        return {
            "files": len(files),
            "bytes": sum(path.stat().st_size for path in files),
        }
