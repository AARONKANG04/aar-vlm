import argparse
import json
import time
from pathlib import Path

import numpy as np

import aargrad as ag
from aargrad import nn
from aargrad.optim import AdamW

from data import CharTokenizer, TokenWindowDataset, DataLoader
from model import GPT

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "data"
DEFAULT_OUT_DIR = SCRIPT_DIR.parent / "data"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--betas", nargs=2, type=float, default=[0.9, 0.95])
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--save-interval", type=int, default=500)
    return p.parse_args()


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def to_device_batch(batch, device):
    x = ag.from_numpy(batch["input"]).to(device)
    y = ag.from_numpy(batch["target"]).to(device)
    return x, y


def save_checkpoint(model, args, vocab_size, prefix):
    prefix = Path(prefix)
    arrays = [ag.to_numpy(p.tensor.to(ag.Device.CPU)) for p in model.parameters()]
    np.savez(prefix.with_suffix(".npz"), **{f"p{i}": a for i, a in enumerate(arrays)})
    cfg = {
        "vocab_size": vocab_size,
        "max_seq_len": args.max_seq_len,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "d_model": args.d_model,
        "dropout": args.dropout,
    }
    with open(prefix.with_suffix(".json"), "w") as f:
        json.dump(cfg, f, indent=2)


def estimate_loss(model, loader, ce, device, n_batches):
    model.eval()
    losses = []
    it = iter(loader)
    for _ in range(n_batches):
        batch = next(it, None)
        if batch is None:
            break
        x, y = to_device_batch(batch, device)
        logits = model(x)
        loss = ce(logits, y)
        losses.append(ag.to_numpy(loss.to(ag.Device.CPU)).item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ag.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tok = CharTokenizer.load(args.data_dir / "meta.json")
    train_tokens = np.memmap(args.data_dir / "train.bin", dtype=np.int64, mode="r")
    val_tokens = np.memmap(args.data_dir / "val.bin", dtype=np.int64, mode="r")

    train_ds = TokenWindowDataset(train_tokens, args.max_seq_len)
    val_ds = TokenWindowDataset(val_tokens, args.max_seq_len)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, seed=args.seed)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=True, seed=args.seed + 1)

    device = ag.Device.CUDA if args.device == "cuda" else ag.Device.CPU

    model = GPT(
        vocab_size=tok.vocab_size,
        max_seq_len=args.max_seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout,
        rng=rng,
    )
    model.to(device)

    optimizer = AdamW(
        list(model.parameters()),
        lr=args.lr,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
    )
    ce = nn.CrossEntropyLoss()

    n_params = sum(int(np.prod(p.tensor.shape)) for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params on {args.device}")
    print(f"train tokens: {len(train_tokens):,} | val tokens: {len(val_tokens):,}")

    train_iter = cycle(train_loader)
    t0 = time.time()
    for it in range(1, args.max_iters + 1):
        batch = next(train_iter)
        x, y = to_device_batch(batch, device)

        logits = model(x)
        loss = ce(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % args.log_interval == 0:
            dt = time.time() - t0
            l = ag.to_numpy(loss.to(ag.Device.CPU)).item()
            print(f"iter {it:5d} | loss {l:.4f} | {dt/it*1000:.1f} ms/it")

        if it % args.eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, ce, device, args.eval_batches)
            print(f"  eval | val loss {val_loss:.4f}")

        if it % args.save_interval == 0:
            prefix = args.out_dir / f"ckpt_{it}"
            save_checkpoint(model, args, tok.vocab_size, prefix)
            print(f"  saved {prefix}.npz / {prefix}.json")

    final = args.out_dir / "ckpt_final"
    save_checkpoint(model, args, tok.vocab_size, final)
    print(f"saved {final}.npz / {final}.json")


if __name__ == "__main__":
    main()
