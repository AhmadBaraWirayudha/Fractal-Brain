"""Encoder-decoder fusion model with an autoregressive decoder."""
from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Sequence

from .interfaces import Encoder, Decoder
from .math_utils import Matrix, Vector, softmax, sample_multinomial


class NativeEncoder(Encoder):
    """Pure-Python encoder that maps token ids to a latent vector.

    It uses a learned token embedding matrix and an output projection.
    """

    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = Matrix.he_init(vocab_size, d_model)
        self.W_latent = Matrix.he_init(d_model, d_model)

    def encode(self, token_ids: Sequence[int]) -> Vector:
        if not token_ids:
            return Vector.zeros(self.d_model)
        rows = [self.embedding.data[i] for i in token_ids]
        mean = Vector([sum(col) / len(rows) for col in zip(*rows)])
        return self.W_latent.linear(mean)


class NativeAutoregressiveDecoder(Decoder):
    """Pure-Python autoregressive decoder that samples tokens from latent state."""

    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.W_context = Matrix.he_init(d_model, d_model)
        self.W_out = Matrix.he_init(d_model, vocab_size)

    def _logits(self, latent: Vector, prev_token: Optional[int] = None) -> Vector:
        context = self.W_context.linear(latent)
        if prev_token is not None:
            # Mild autoregressive feedback: each previous token nudges the context.
            feedback = Vector([(prev_token % 11) / 11.0 for _ in range(self.d_model)])
            context = Vector([a + b for a, b in zip(context, feedback)])
        return self.W_out.linear(context)

    def decode(self, latent: Vector, prompt_ids=None, max_new_tokens: int = 32, eos_id: Optional[int] = None):
        if not isinstance(latent, Vector):
            latent = Vector(latent)
        generated = list(prompt_ids or [])
        prev = generated[-1] if generated else None
        for _ in range(max_new_tokens):
            logits = self._logits(latent, prev)
            probs = softmax(logits)
            next_id = sample_multinomial(probs)
            generated.append(next_id)
            prev = next_id
            if eos_id is not None and next_id == eos_id:
                break
        return generated


class RobertaAdapter(Encoder):
    """Adapter for an external RoBERTa-style encoder.

    Pass either a callable `encode(token_ids) -> vector-like`, or an object exposing
    `.encode(...)` / `.forward(...)` / `.__call__(...)`.
    """

    def __init__(self, model):
        self.model = model

    def encode(self, token_ids):
        if hasattr(self.model, 'encode'):
            out = self.model.encode(token_ids)
        elif hasattr(self.model, 'forward'):
            out = self.model.forward(token_ids)
        else:
            out = self.model(token_ids)
        return out if isinstance(out, Vector) else Vector(out)


class GPTAdapter(Decoder):
    """Adapter for an external GPT-style decoder.

    It is intentionally model-agnostic: the caller owns the real model weights.
    """

    def __init__(self, model):
        self.model = model

    def decode(self, latent: Vector, prompt_ids=None, max_new_tokens: int = 32):
        if hasattr(self.model, 'decode'):
            return self.model.decode(latent, prompt_ids=prompt_ids, max_new_tokens=max_new_tokens)
        if hasattr(self.model, 'generate'):
            return self.model.generate(latent, prompt_ids=prompt_ids, max_new_tokens=max_new_tokens)
        return self.model(latent, prompt_ids=prompt_ids, max_new_tokens=max_new_tokens)


class FusionModel:
    """Encoder-decoder hybrid with optional memory fusion and autoregressive output."""

    def __init__(self, encoder: Encoder, decoder: Decoder, fusion: Optional[Callable[[Vector], Vector]] = None):
        self.encoder = encoder
        self.decoder = decoder
        self.fusion = fusion

    def encode(self, token_ids):
        latent = self.encoder.encode(token_ids)
        return latent if isinstance(latent, Vector) else Vector(latent)

    def generate(self, token_ids, prompt_ids=None, max_new_tokens: int = 32, eos_id: Optional[int] = None):
        latent = self.encode(token_ids)
        if self.fusion is not None:
            latent = self.fusion(latent)
            if not isinstance(latent, Vector):
                latent = Vector(latent)
        return self.decoder.decode(latent, prompt_ids=prompt_ids, max_new_tokens=max_new_tokens, eos_id=eos_id)

    def generate_text(self, token_ids, tokenizer, prompt_ids=None, max_new_tokens: int = 32, eos_id: Optional[int] = None):
        out_ids = self.generate(token_ids, prompt_ids=prompt_ids, max_new_tokens=max_new_tokens, eos_id=eos_id)
        return tokenizer.decode(out_ids)
