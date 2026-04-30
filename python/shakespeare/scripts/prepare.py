import argparse
import urllib.request
from pathlib import Path

from data import CharTokenizer

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "data"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--url", default=SHAKESPEARE_URL)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.data_dir / "input.txt"
    if not input_path.exists() or args.force:
        print(f"downloading {args.url} -> {input_path}")
        urllib.request.urlretrieve(args.url, input_path)
    text = input_path.read_text()
    print(f"text: {len(text)} chars")

    tok = CharTokenizer.from_text(text)
    print(f"vocab: {tok.vocab_size} unique chars")
    tok.save(args.data_dir / "meta.json")

    ids = tok.encode(text)
    n_val = max(1, int(len(ids) * args.val_frac))
    train_ids = ids[:-n_val]
    val_ids = ids[-n_val:]
    train_path = args.data_dir / "train.bin"
    val_path = args.data_dir / "val.bin"
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    print(f"train: {len(train_ids):,} tokens -> {train_path}")
    print(f"val:   {len(val_ids):,} tokens -> {val_path}")


if __name__ == "__main__":
    main()
