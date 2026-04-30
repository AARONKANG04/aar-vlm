import numpy as np

import aargrad as ag
from aargrad import nn
from aargrad.nn import Module


class TransformerBlock(Module):
    def __init__(self, d_model, n_heads, dropout, rng):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiHeadAttention(d_model, n_heads, causal=True, bias=False, rng=rng)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=True, rng=rng)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * d_model, d_model, bias=True, rng=rng)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = ag.add(x, self.drop(self.attn(self.ln1(x))))
        h = self.gelu(self.fc1(self.ln2(x)))
        x = ag.add(x, self.drop(self.fc2(h)))
        return x


class GPT(Module):
    def __init__(self, vocab_size, max_seq_len, n_layer, n_head, d_model,
                 dropout=0.0, rng=None):
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.dropout = dropout

        self.tok_emb = nn.Embedding(vocab_size, d_model, rng=rng)
        self.pos_emb = nn.Embedding(max_seq_len, d_model, rng=rng)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_head, dropout, rng) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False, rng=rng)

    def forward(self, ids):
        B, T = ids.shape
        if T > self.max_seq_len:
            raise ValueError(f"sequence length {T} > max_seq_len {self.max_seq_len}")
        positions_np = np.tile(np.arange(T, dtype=np.int64), (B, 1))
        positions = ag.from_numpy(positions_np).to(ids.device)
        x = ag.add(self.tok_emb(ids), self.pos_emb(positions))
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)
