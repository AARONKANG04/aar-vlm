import argparse
import json
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from tqdm import tqdm

import aargrad as ag
from aargrad import nn
from aargrad.optim import AdamW

from data import CharTokenizer, TokenWindowDataset, DataLoader
from model import GPT


class StageTimer:
    def __init__(self, sync_fn, warmup):
        self._sync = sync_fn
        self._warmup = warmup
        self._times = defaultdict(list)
        self._counts = defaultdict(int)
        self._order = []

    @contextmanager
    def __call__(self, name):
        if name not in self._counts:
            self._order.append(name)
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._counts[name] += 1
            if self._counts[name] > self._warmup:
                self._times[name].append(dt_ms)

    def report(self):
        if not self._times:
            print("no profile data")
            return
        n = min(len(self._times[k]) for k in self._order)
        per_step_total = [sum(self._times[k][i] for k in self._order) for i in range(n)]

        def stats(values):
            v = sorted(values)
            med = v[len(v) // 2]
            p99 = v[min(len(v) - 1, int(len(v) * 0.99))]
            return med, p99

        total_med, total_p99 = stats(per_step_total)
        print()
        print(f"{'stage':<12} {'ms_med':>10} {'%':>6}  {'ms_p99':>10}")
        for name in self._order:
            med, p99 = stats(self._times[name])
            pct = (med / total_med * 100.0) if total_med else 0.0
            print(f"{name:<12} {med:>10.3f} {pct:>5.1f}  {p99:>10.3f}")
        print("-" * 42)
        print(f"{'total':<12} {total_med:>10.3f} {100.0:>5.1f}  {total_p99:>10.3f}")

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
    p.add_argument("--profile", action="store_true", help="profile training step stages then exit")
    p.add_argument("--profile-warmup", type=int, default=10)
    p.add_argument("--profile-steps", type=int, default=50)
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

    if args.profile:
        timer = StageTimer(ag.cuda_synchronize, warmup=args.profile_warmup)
        n_total = args.profile_warmup + args.profile_steps
        pbar = tqdm(range(1, n_total + 1), desc="profile", dynamic_ncols=True)
        for _ in pbar:
            batch = next(train_iter)
            x, y = to_device_batch(batch, device)

            with timer("forward"):
                logits = model(x)
            with timer("loss"):
                loss = ce(logits, y)
            with timer("backward"):
                optimizer.zero_grad()
                loss.backward()
            with timer("optimizer"):
                optimizer.step()
        timer.report()
        return

    pbar = tqdm(range(1, args.max_iters + 1), desc="train", dynamic_ncols=True)
    for it in pbar:
        batch = next(train_iter)
        x, y = to_device_batch(batch, device)

        logits = model(x)
        loss = ce(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % args.log_interval == 0:
            l = ag.to_numpy(loss.to(ag.Device.CPU)).item()
            pbar.set_postfix(loss=f"{l:.4f}")

        if it % args.eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, ce, device, args.eval_batches)
            pbar.write(f"iter {it} | val loss {val_loss:.4f}")

        if it % args.save_interval == 0:
            prefix = args.out_dir / f"ckpt_{it}"
            save_checkpoint(model, args, tok.vocab_size, prefix)
            pbar.write(f"iter {it} | saved {prefix}.npz")

    final = args.out_dir / "ckpt_final"
    save_checkpoint(model, args, tok.vocab_size, final)
    pbar.write(f"saved {final}.npz / {final}.json")


if __name__ == "__main__":
    main()
