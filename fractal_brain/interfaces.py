"""Shared interfaces for encoder-decoder fusion models."""
from __future__ import annotations

from .math_utils import Vector


class Encoder:
    """Minimal encoder contract."""
    def encode(self, token_ids):
        raise NotImplementedError


class Decoder:
    """Minimal decoder contract."""
    def decode(self, latent: Vector, prompt_ids=None, max_new_tokens: int = 32):
        raise NotImplementedError
