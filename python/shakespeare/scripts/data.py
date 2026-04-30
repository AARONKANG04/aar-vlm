import json
from pathlib import Path

import numpy as np


class CharTokenizer:
    def __init__(self, vocab):
        self.vocab = list(vocab)
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.itos = {i: c for i, c in enumerate(self.vocab)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f)["vocab"])

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab, "vocab_size": len(self.vocab)}, f)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, s):
        return np.array([self.stoi[c] for c in s], dtype=np.int64)

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


class TokenWindowDataset:
    def __init__(self, tokens, max_seq_len):
        self.tokens = tokens
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.tokens) - self.max_seq_len

    def __getitem__(self, i):
        x = np.asarray(self.tokens[i : i + self.max_seq_len], dtype=np.int64)
        y = np.asarray(self.tokens[i + 1 : i + self.max_seq_len + 1], dtype=np.int64)
        return {"input": x, "target": y}


def default_collate(batch):
    keys = batch[0].keys()
    return {k: np.stack([b[k] for b in batch]) for k in keys}


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=True,
                 seed=0, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn if collate_fn is not None else default_collate
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        n = len(self.dataset)
        idx = self.rng.permutation(n) if self.shuffle else np.arange(n)
        bs = self.batch_size
        end = (n // bs) * bs if self.drop_last else n
        for start in range(0, end, bs):
            batch_idx = idx[start : start + bs]
            samples = [self.dataset[int(i)] for i in batch_idx]
            yield self.collate_fn(samples)
