"""The six correctness checks

Run:

    python checks.py

Every check prints PASS/FAIL. The output belongs in the README.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from decode import parse_prediction
from model.transformer import Transformer
from starter.embeddings import TokenEmbedding
from starter.tokenizer import PAD_ID, read_pairs
from train import transformer_lr, WARMUP_STEPS, D_MODEL

DEFAULT_CHECKPOINT = "best_model.pt"
TOKENIZER = "data/sql_sp.model"
WIKISQL_DIR = Path("WikiSQL")
WIKISQL_DATA_DIR = Path("Wikidata")
BATCH = 8
SOURCE_LEN = 40
TARGET_LEN = 20
LR_CHECK_STEP = 20000

_results = []


def record(name, passed, detail):
    _results.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return passed


def build_model(checkpoint_path, device):
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = Transformer(
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        heads=config["heads"],
        token_embedding=TokenEmbedding(
            sp.get_piece_size(), config["d_model"], PAD_ID
        ),
        ffn_dim=config["d_ff"],
        max_len=512,
        dropout=config["dropout"],
        vocab_size=sp.get_piece_size(),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, sp, checkpoint


def random_batch(device, rows=BATCH, src_len=SOURCE_LEN, tgt_len=TARGET_LEN):
    src = torch.randint(4, 200, (rows, src_len), device=device)
    tgt = torch.randint(4, 200, (rows, tgt_len), device=device)
    src[:, -3:] = PAD_ID
    tgt[:, -3:] = PAD_ID
    return src, tgt


# ---------------------------------------------------------------- check 1
def check_causal_mask(model, device):
    """Changing the last decoder token must not change earlier positions."""
    _, tgt = random_batch(device)
    mutated = tgt.clone()
    mutated[:, -1] = (mutated[:, -1] + 17) % 500 + 4

    src, _ = random_batch(device)
    with torch.no_grad():
        base, _ = model(src, tgt)
        changed, _ = model(src, mutated)

    diff = (base[:, :-1] - changed[:, :-1]).abs().max().item()
    return record(
        "Causal mask",
        diff < 1e-5,
        f"max |delta| over earlier positions = {diff:.3e} (target < 1e-5)",
    )


# ---------------------------------------------------------------- check 2
def check_padding_mask(model, device):
    """Extra <pad> tokens in the source must not change the output."""
    src, tgt = random_batch(device, src_len=SOURCE_LEN)
    padded = torch.full(
        (src.size(0), SOURCE_LEN + 7), PAD_ID, dtype=torch.long, device=device
    )
    padded[:, :SOURCE_LEN] = src

    with torch.no_grad():
        base, _ = model(src, tgt)
        extra, _ = model(padded, tgt)

    diff = (base - extra[:, :SOURCE_LEN]).abs().max().item()
    return record(
        "Padding mask",
        diff < 1e-5,
        f"max |delta| after 7 extra <pad> = {diff:.3e} (target < 1e-5)",
    )


# ---------------------------------------------------------------- check 3
def check_attention_rows(model, device):
    """Attention weights sum to 1 over the unmasked positions."""
    src, tgt = random_batch(device)
    with torch.no_grad():
        _, attention = model(src, tgt)

    # attention: (B, heads, T, S) from the last decoder layer.
    rows = attention.sum(dim=-1)
    worst = (rows - 1.0).abs().max().item()
    return record(
        "Attention rows sum to 1",
        worst < 1e-5,
        f"max |row sum - 1| = {worst:.3e} (target < 1e-5)",
    )


# ---------------------------------------------------------------- check 4
def check_weight_sharing(model):
    """Output projection weight and embedding weight must be the same tensor."""
    same = model.output_projection.weight is model.encoder.input.tok.emb.weight
    return record(
        "Weight sharing",
        same,
        "output_projection.weight is embedding.weight" if same else "tensors differ",
    )


# ---------------------------------------------------------------- check 5
def check_lr_schedule():
    """Linear warmup for 4000 steps, then decay as step^-0.5."""
    peak = transformer_lr(WARMUP_STEPS, D_MODEL, WARMUP_STEPS)
    early = transformer_lr(WARMUP_STEPS // 8, D_MODEL, WARMUP_STEPS)
    late = transformer_lr(LR_CHECK_STEP, D_MODEL, WARMUP_STEPS)
    expected_late = D_MODEL ** -0.5 * LR_CHECK_STEP ** -0.5

    linear = abs(early * 8 - peak) < 1e-9
    decaying = late < peak
    correct_form = abs(late - expected_late) < 1e-12

    # The plotted figure lives in results/figures/, written by tables.py;
    # this check only asserts the shape of the curve.
    passed = linear and decaying and correct_form
    return record(
        "Learning-rate schedule",
        passed,
        f"peak@4000 = {peak:.3e}, value@{LR_CHECK_STEP} = {late:.3e} "
        f"(expected {expected_late:.3e}); linear ramp = {linear}, "
        f"decays after warmup = {decaying}",
    )


# ---------------------------------------------------------------- check 6
def gold_round_trip(sp, device, limit=None):
    """Gold targets -> parser -> official evaluator. Must exceed 99%."""
    if not (WIKISQL_DIR / "evaluate.py").exists():
        return record(
            "Gold round-trip",
            False,
            "WikiSQL/evaluate.py not found (git clone https://github.com/salesforce/WikiSQL WikiSQL)",
        )

    pairs = read_pairs("data/dev_pairs.jsonl")
    if limit:
        pairs = pairs[:limit]

    # Scratch file only -- the evaluator needs a path, but this prediction
    # file is a by-product of the check, not a deliverable, so it must not
    # land in results/.
    handle, name = tempfile.mkstemp(suffix=".jsonl")
    os.close(handle)
    path = Path(name)
    unparseable = 0
    try:
        with path.open("w", encoding="utf-8") as stream:
            for pair in pairs:
                query = parse_prediction(pair["tgt"])
                if query is None:
                    unparseable += 1
                    stream.write(json.dumps({"error": "parse"}) + "\n")
                else:
                    stream.write(json.dumps({"query": query}) + "\n")

        environment = dict(os.environ, PYTHONUTF8="1")
        run = subprocess.run(
            [
                sys.executable,
                str(WIKISQL_DIR / "evaluate.py"),
                str(WIKISQL_DATA_DIR / "dev.jsonl"),
                str(WIKISQL_DATA_DIR / "dev.db"),
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=environment,
        )
    finally:
        path.unlink(missing_ok=True)
    if run.returncode != 0:
        return record(
            "Gold round-trip",
            False,
            "official evaluator failed: " + run.stderr.strip()[:300],
        )

    metrics = json.loads(run.stdout[run.stdout.find("{"):])
    accuracy = metrics["ex_accuracy"]
    return record(
        "Gold round-trip",
        accuracy > 0.99 and unparseable == 0,
        f"execution accuracy = {accuracy:.2%} over {len(pairs)} gold targets, "
        f"{unparseable} unparseable (target > 99%, 0 unparseable)",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gold-limit", type=int, help="Subsample the gold round-trip.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sp, _ = build_model(args.checkpoint, device)

    print(f"Checkpoint: {args.checkpoint}  Device: {device}\n")
    check_causal_mask(model, device)
    check_padding_mask(model, device)
    check_attention_rows(model, device)
    check_weight_sharing(model)
    check_lr_schedule()
    print()
    gold_round_trip(sp, device, args.gold_limit)

    failed = [name for name, passed, _ in _results if not passed]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed.")
    if failed:
        print("Failed: " + ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
