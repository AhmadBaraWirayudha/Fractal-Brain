"""
fractal_brain/tokenizer.py
A from-scratch, dependency-free byte-pair-encoding (BPE) tokenizer.

Covers the "Tokenizer", "Vocabulary builder", and "Text cleaning / normalization" items
from To-Do.md's data pipeline section. No external dependencies, consistent with the
rest of the library.

Honest limitations (documented rather than silently glossed over):
  - Splits words into Unicode *code points*, not grapheme clusters, so some combining
    characters/emoji may split in surprising ways. Fine for plain English/ASCII-ish text,
    which is what this project's examples use.
  - decode() is an approximate detokenizer, not an exact inverse of encode(): whitespace
    and punctuation spacing are reconstructed with a simple heuristic, not preserved
    byte-for-byte. A fully exact round trip (as in byte-level BPE, e.g. GPT-2's) would
    need whitespace itself folded into the vocabulary, which is a larger undertaking than
    is warranted here -- see To-Do.md.
"""
import re
import json
from collections import Counter

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
EOW = "</w>"     # end-of-word marker: keeps "cat" and "cats" from bleeding into each
                 # other's merges, and doubles as the word-boundary marker for decode()
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]

_WORD_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_NO_SPACE_BEFORE_RE = re.compile(r"\s+([.,!?;:\)\]\}'\"])")


def normalize_text(text, lowercase=True, collapse_whitespace=True):
    """Basic, dependency-free text cleaning."""
    if lowercase:
        text = text.lower()
    if collapse_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
    return text


def _word_tokenize(text):
    """Split into word-runs and standalone punctuation characters."""
    return _WORD_RE.findall(text)


def _merge_symbols(symbols, pair, merged):
    """Replace every adjacent occurrence of `pair` in `symbols` with the single symbol `merged`."""
    out = []
    i = 0
    n = len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


class BPETokenizer:
    """
    Byte-pair-encoding tokenizer, trained from raw text, zero external dependencies.

    Usage:
        tok = BPETokenizer()
        tok.train(["a corpus of raw text", "more raw text"], vocab_size=500)
        ids = tok.encode("some new text")
        text = tok.decode(ids)
        tok.save("vocab.json"); tok2 = BPETokenizer.load("vocab.json")
    """
    def __init__(self, lowercase=True):
        self.lowercase = lowercase
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = []          # list of (str, str), in the order they were learned
        self._merge_rank = {}     # (str, str) -> rank; lower rank = learned earlier = applied first
        self._word_cache = {}     # word -> list[str] subword pieces, memoized for encode() speed

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def train(self, corpus, vocab_size=1000, min_frequency=2):
        """
        corpus: list of raw text strings.
        vocab_size: target vocabulary size (special tokens + base characters count
                    towards this; training stops early if the corpus is too small/
                    repetitive to reach it, or once no pair reaches min_frequency).
        """
        if vocab_size <= len(SPECIAL_TOKENS):
            raise ValueError(f"vocab_size must be > {len(SPECIAL_TOKENS)} (the number of special tokens)")

        word_freq = Counter()
        for text in corpus:
            text = normalize_text(text, lowercase=self.lowercase)
            word_freq.update(_word_tokenize(text))

        # each distinct word starts as a list of characters + an end-of-word marker
        splits = {word: list(word) + [EOW] for word in word_freq}

        base_symbols = set()
        for symbols in splits.values():
            base_symbols.update(symbols)
        vocab = list(SPECIAL_TOKENS) + sorted(base_symbols)

        merges = []
        while len(vocab) < vocab_size:
            pair_counts = Counter()
            for word, freq in word_freq.items():
                symbols = splits[word]
                for i in range(len(symbols) - 1):
                    pair_counts[(symbols[i], symbols[i + 1])] += freq
            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_frequency:
                break
            merged = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            vocab.append(merged)
            for word in splits:
                splits[word] = _merge_symbols(splits[word], best_pair, merged)

        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        self.merges = merges
        self._merge_rank = {pair: rank for rank, pair in enumerate(merges)}
        self._word_cache = {}
        return self

    def _split_word(self, word):
        cached = self._word_cache.get(word)
        if cached is not None:
            return cached
        symbols = list(word) + [EOW]
        while len(symbols) > 1:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            ranked = [(self._merge_rank[p], i) for i, p in enumerate(pairs) if p in self._merge_rank]
            if not ranked:
                break
            _, i = min(ranked)   # apply the earliest-learned (highest-priority) merge present
            symbols = symbols[:i] + [symbols[i] + symbols[i + 1]] + symbols[i + 2:]
        self._word_cache[word] = symbols
        return symbols

    def encode(self, text, add_bos=False, add_eos=False):
        """text (str) -> list[int]"""
        if not self.token_to_id:
            raise RuntimeError("Tokenizer has no vocabulary yet -- call train() or load() first")
        text = normalize_text(text, lowercase=self.lowercase)
        unk_id = self.token_to_id[UNK]
        ids = [self.token_to_id[BOS]] if add_bos else []
        for word in _word_tokenize(text):
            for sym in self._split_word(word):
                ids.append(self.token_to_id.get(sym, unk_id))
        if add_eos:
            ids.append(self.token_to_id[EOS])
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        """list[int] -> str. Approximate detokenization -- see module docstring."""
        specials = set(SPECIAL_TOKENS)
        pieces = []
        for i in token_ids:
            tok = self.id_to_token.get(i, UNK)
            if skip_special_tokens and tok in specials:
                continue
            pieces.append(tok)
        text = "".join(pieces).replace(EOW, " ").strip()
        text = _NO_SPACE_BEFORE_RE.sub(r"\1", text)   # "word ." -> "word."
        return text

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "lowercase": self.lowercase,
                "token_to_id": self.token_to_id,
                "merges": self.merges,
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(lowercase=data["lowercase"])
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = {i: t for t, i in tok.token_to_id.items()}
        tok.merges = [tuple(p) for p in data["merges"]]
        tok._merge_rank = {pair: rank for rank, pair in enumerate(tok.merges)}
        tok._word_cache = {}
        return tok
