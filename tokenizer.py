from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable


class SentenceSplitter:
    _pattern = re.compile(r"(?<=[.!?])\s+|\n+")

    @staticmethod
    def split(text: str) -> list[str]:
        parts = [p.strip() for p in SentenceSplitter._pattern.split(text) if p.strip()]
        return parts or ([text.strip()] if text.strip() else [])


@dataclass
class TokenBatch:
    sentences: list[str]
    token_ids: list[int]
    attention_mask: list[int]
    embeddings: list[list[float]]
    raw_tokens: list[str]


class QueryTokenizer:
    def __init__(self, tokenizer_name: str, max_length: int = 512) -> None:
        self.backend_name = tokenizer_name
        self.backend = None
        self.max_length = max_length

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'QueryTokenizer':
        name = config.get('model', {}).get('fallback_model_name') or config.get('model', {}).get('model_name') or 'fallback-tokenizer'
        return cls(name)

    def tokenize(self, text: str) -> TokenBatch:
        sentences = SentenceSplitter.split(text)
        raw_tokens = re.findall(r'[A-Za-z0-9_]+|[^\w\s]', text)[: self.max_length]
        token_ids = [self._stable_id(tok) for tok in raw_tokens]
        attention_mask = [1] * len(token_ids)
        embeddings = self._id_embeddings(token_ids)
        return TokenBatch(sentences, token_ids, attention_mask, embeddings, raw_tokens)

    @staticmethod
    def _stable_id(token: str) -> int:
        digest = hashlib.sha256(token.encode('utf-8')).digest()
        return int.from_bytes(digest[:4], 'big') % 32000

    @staticmethod
    def _id_embeddings(token_ids: list[int], dim: int = 64) -> list[list[float]]:
        vectors: list[list[float]] = []
        for token_id in token_ids:
            # Deterministic pseudo-embedding from token id.
            x = token_id + 13579
            row = []
            for i in range(dim):
                x = (1103515245 * x + 12345 + i) & 0x7FFFFFFF
                row.append(((x % 2000) / 1000.0) - 1.0)
            vectors.append(row)
        return vectors


class TextEmbedder:
    def __init__(self, model_name: str, cache_size: int = 2048) -> None:
        self.model_name = model_name
        self.cache_size = cache_size
        self._cache: dict[str, list[float]] = {}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'TextEmbedder':
        return cls(config.get('retrieval', {}).get('embedding_model', 'simple-hash-embedding'))

    def embed_text(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        vec = self._fallback(text)
        if len(self._cache) >= self.cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[text] = vec
        return vec

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]

    @staticmethod
    def _fallback(text: str, dim: int = 384) -> list[float]:
        vec = [0.0] * dim
        for tok in re.findall(r'[A-Za-z0-9_]+', text.lower()):
            idx = _stable_hash(tok) % dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def _stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8], 'big')
