"""
fractal_brain/dataset.py
A minimal, dependency-free dataset loader: turns raw text into sliding-window
(context_token_ids, target_one_hot) next-token-prediction examples using a tokenizer,
with train/validation/test splitting.

Covers the "Dataset loader" and "Train / validation / test split" items from
To-Do.md's data pipeline section.

Deliberately produces one example at a time, matching FractalBrain.step()'s current
single-sequence interface -- batching/padding is a separate, later To-Do item, since it
needs changes to FractalBrain itself (which processes one sequence per call today), not
just to data loading.
"""
import random


class TextDataset:
    """
    Wraps a tokenizer + raw text into (context_token_ids, target_one_hot) examples.

    tokenizer: any object with .encode(text) -> list[int] and a .vocab_size property
               (e.g. tokenizer.BPETokenizer).
    texts: list of raw text strings (one per document/paragraph/line -- windows never
           cross a text boundary).
    context_length: number of tokens of context per example.
    stride: step between consecutive windows (1 = every possible window, context_length
            = non-overlapping windows).
    """
    def __init__(self, tokenizer, texts, context_length=8, stride=1):
        if context_length < 1:
            raise ValueError("context_length must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.context_length = context_length
        self.examples = []   # list of (context_token_ids, target_token_id)
        for text in texts:
            ids = tokenizer.encode(text)
            for start in range(0, len(ids) - context_length, stride):
                context = ids[start:start + context_length]
                target_id = ids[start + context_length]
                self.examples.append((context, target_id))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        context, target_id = self.examples[idx]
        target = [0.0] * self.vocab_size
        target[target_id] = 1.0
        return context, target

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def split(self, train_frac=0.8, val_frac=0.1, seed=0):
        """
        Shuffle example indices and split into (train, val, test) views. Returns
        DatasetView objects (not plain lists) so each retains .vocab_size and supports
        the same __len__/__getitem__/__iter__ interface as this dataset.
        """
        if not (0 < train_frac < 1) or not (0 <= val_frac < 1) or train_frac + val_frac > 1:
            raise ValueError("require 0 < train_frac < 1, 0 <= val_frac, train_frac + val_frac <= 1")
        idxs = list(range(len(self.examples)))
        random.Random(seed).shuffle(idxs)
        n = len(idxs)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_idx = idxs[:n_train]
        val_idx = idxs[n_train:n_train + n_val]
        test_idx = idxs[n_train + n_val:]
        return DatasetView(self, train_idx), DatasetView(self, val_idx), DatasetView(self, test_idx)


class DatasetView:
    """A read-only view over a subset of a TextDataset's examples, addressed by index."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)
        self.vocab_size = dataset.vocab_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def shuffle(self, seed=None):
        """Shuffle this view's index order in place (e.g. once per training epoch)."""
        random.Random(seed).shuffle(self.indices)
