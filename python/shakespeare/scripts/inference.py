import argparse
import json
from pathlib import Path

import numpy as np

import aargrad as ag

from data import CharTokenizer
from model import GPT

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "data"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="path prefix; expects <prefix>.npz and <prefix>.json")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help="dir containing meta.json")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_model(prefix, device):
    prefix = Path(prefix)
    with open(prefix.with_suffix(".json")) as f:
        cfg = json.load(f)
    model = GPT(
        vocab_size=cfg["vocab_size"],
        max_seq_len=cfg["max_seq_len"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        d_model=cfg["d_model"],
        dropout=0.0,
    )
    model.to(device)
    data = np.load(prefix.with_suffix(".npz"))
    keys = sorted(data.files, key=lambda k: int(k[1:]))
    arrays = [data[k] for k in keys]
    params = list(model.parameters())
    if len(params) != len(arrays):
        raise ValueError(f"param count mismatch: model has {len(params)}, ckpt has {len(arrays)}")
    for p, arr in zip(params, arrays):
        new_t = ag.from_numpy(np.ascontiguousarray(arr)).to(device)
        new_t.requires_grad = True
        p.tensor = new_t
    return model, cfg


def softmax_np(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def sample(model, tok, prompt, max_new, temperature, top_k, device, rng):
    max_seq_len = model.max_seq_len
    ids = tok.encode(prompt).tolist() if len(prompt) > 0 else [0]
    print(prompt, end="", flush=True)
    model.eval()
    for _ in range(max_new):
        ctx = ids[-max_seq_len:]
        x = ag.from_numpy(np.array([ctx], dtype=np.int64)).to(device)
        logits = model(x)
        logits_np = ag.to_numpy(logits.to(ag.Device.CPU))[0, -1, :].astype(np.float64)
        logits_np = logits_np / max(temperature, 1e-8)
        if top_k is not None and top_k < len(logits_np):
            kth = np.partition(logits_np, -top_k)[-top_k]
            logits_np = np.where(logits_np >= kth, logits_np, -np.inf)
        probs = softmax_np(logits_np)
        next_id = int(rng.choice(len(probs), p=probs))
        ids.append(next_id)
        print(tok.itos[next_id], end="", flush=True)
    print()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = ag.Device.CUDA if args.device == "cuda" else ag.Device.CPU
    tok = CharTokenizer.load(args.data_dir / "meta.json")
    model, _ = load_model(args.checkpoint, device)
    sample(model, tok, args.prompt, args.max_new_tokens,
           args.temperature, args.top_k, device, rng)


if __name__ == "__main__":
    main()
