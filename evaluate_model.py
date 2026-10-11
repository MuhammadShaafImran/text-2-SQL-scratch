"""Decode a split and write WikiSQL prediction files.

Task 4.3: one JSON line per input row, in input-file order, never skipped.
Malformed generations are written as {"error": "parse"}.

Task 5.1: logical-form and execution accuracy come from the official
WikiSQL evaluator (``WikiSQL/evaluate.py``), never from this module.
The spec is explicit: "Use the evaluator shipped with WikiSQL; do not
write your own."  This script only orchestrates decoding, writes the
prediction file, and reports the component accuracies required by
Task 5.2 plus the parse-failure rate.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn

from decode import beam_search, greedy_decode, parse_prediction
from model.transformer import Transformer
from starter.dataset import make_loader
from starter.embeddings import TokenEmbedding
from starter.tokenizer import PAD_ID, read_pairs


DEFAULT_TOKENIZER = "data/sql_sp.model"
DEFAULT_BEST_CHECKPOINT = "best_model.pt"
DEFAULT_LAST_CHECKPOINT = "last_model.pt"
DEFAULT_BATCH_SIZE = 64
MAX_LEN = 512
DEFAULT_OUTPUT_DIR = Path("results")
WIKISQL_DIR = Path("WikiSQL")
WIKISQL_DATA_DIR = Path("Wikidata")


def choose_checkpoint(path):
    if path:
        return Path(path)
    best = Path(DEFAULT_BEST_CHECKPOINT)
    last = Path(DEFAULT_LAST_CHECKPOINT)
    # Task 3.5: report the checkpoint with the lowest dev loss.
    return best if best.exists() else last


def build_model(checkpoint, sp, device):
    config = checkpoint["config"]
    vocab_size = sp.get_piece_size()
    model = Transformer(
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        heads=config["heads"],
        token_embedding=TokenEmbedding(vocab_size, config["d_model"], PAD_ID),
        ffn_dim=config["d_ff"],
        max_len=MAX_LEN,
        dropout=config["dropout"],
        vocab_size=vocab_size,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def normalize_text(text):
    return " ".join(text.lower().strip().split())


def normalize_conditions(conditions):
    return sorted(
        (int(column), int(operator), normalize_text(str(value)))
        for column, operator, value in conditions
    )


def decode_example(model, pair, sp, device, strategy, beam_size):
    src_ids = torch.tensor(
        [sp.encode(pair["src"], out_type=int)],
        dtype=torch.long,
        device=device,
    )
    if strategy == "beam":
        token_ids = beam_search(
            model,
            src_ids,
            device,
            beam_size=beam_size,
        )
        return token_ids, sp.decode(token_ids)
    return greedy_decode(model, src_ids, sp, device)


def write_predictions(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def component_counts(gold, predicted):
    """Task 5.2 accuracies: sel column, agg, and the WHERE clause."""
    counts = {"select_column_exact": 0, "aggregation_exact": 0, "conditions_exact": 0}
    if gold is None or predicted is None:
        return counts
    counts["select_column_exact"] = int(gold["sel"] == predicted["sel"])
    counts["aggregation_exact"] = int(gold["agg"] == predicted["agg"])
    counts["conditions_exact"] = int(
        normalize_conditions(gold["conds"]) == normalize_conditions(predicted["conds"])
    )
    return counts


def decode_split(model, pairs, sp, device, strategy, beam_size):
    """Decode every pair, returning per-example records and component totals."""
    totals = {
        "parse_valid": 0,
        "select_column_exact": 0,
        "aggregation_exact": 0,
        "conditions_exact": 0,
    }
    records = []
    rows = []

    for pair in pairs:
        _, predicted = decode_example(model, pair, sp, device, strategy, beam_size)
        gold_query = parse_prediction(pair["tgt"])
        predicted_query = parse_prediction(predicted)

        if predicted_query is None:
            # Task 4.3: never skip a line.
            records.append({"error": "parse"})
        else:
            records.append({"query": predicted_query})

        totals["parse_valid"] += int(predicted_query is not None)
        component = component_counts(gold_query, predicted_query)
        for key, value in component.items():
            totals[key] += value
        rows.append((pair, predicted))

    return records, totals, rows


def official_metrics(source_file, db_file, pred_file):
    """Run the evaluator shipped with WikiSQL and return its two numbers."""
    if not (WIKISQL_DIR / "evaluate.py").exists():
        return None
    if not source_file.exists() or not db_file.exists():
        return None

    environment = dict(os.environ, PYTHONUTF8="1")
    result = subprocess.run(
        [
            sys.executable,
            str(WIKISQL_DIR / "evaluate.py"),
            str(source_file),
            str(db_file),
            str(pred_file),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    if result.returncode != 0:
        print("Official evaluator failed:\n" + result.stderr, file=sys.stderr)
        return None

    start = result.stdout.find("{")
    if start < 0:
        return None
    return json.loads(result.stdout[start:])


def teacher_forced_metrics(model, loader, device, label_smoothing):
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_ID,
        label_smoothing=label_smoothing,
        reduction="sum",
    )
    total_loss = 0.0
    correct_tokens = 0
    token_count = 0

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        logits, _ = model(src, tgt[:, :-1])
        target = tgt[:, 1:]
        total_loss += criterion(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
        ).item()
        non_padding = target.ne(PAD_ID)
        correct_tokens += (
            logits.argmax(dim=-1).eq(target) & non_padding
        ).sum().item()
        token_count += non_padding.sum().item()

    loss = total_loss / token_count if token_count else float("inf")
    return loss, math.exp(min(loss, 20.0)), (
        correct_tokens / token_count if token_count else 0.0
    )


def parameter_count(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Decode a split into WikiSQL predictions.")
    parser.add_argument("--checkpoint", help="Checkpoint path.")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, help="Decode only the first N examples (debug).")
    parser.add_argument("--split", default="data/dev_pairs.jsonl")
    parser.add_argument("--strategy", choices=("greedy", "beam"), default="greedy")
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-official",
        action="store_true",
        help="Skip WikiSQL/evaluate.py even if it is present.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = choose_checkpoint(args.checkpoint)

    for path, label in (
        (checkpoint_path, "Checkpoint"),
        (Path(args.tokenizer), "Tokenizer"),
        (Path(args.split), "Evaluation split"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    model = build_model(checkpoint, sp, device)
    split_name = Path(args.split).stem.replace("_pairs", "")
    pairs = read_pairs(args.split)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    loader = make_loader(args.split, sp, train=False, batch_size=args.batch_size)
    if args.limit is not None:
        dataset = torch.utils.data.Subset(
            loader.dataset,
            range(min(args.limit, len(loader.dataset))),
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=loader.collate_fn,
        )

    label_smoothing = checkpoint.get("config", {}).get("label_smoothing", 0.0)
    loss, perplexity, token_accuracy = teacher_forced_metrics(
        model, loader, device, label_smoothing
    )
    records, totals, _ = decode_split(
        model, pairs, sp, device, args.strategy, args.beam_size
    )

    # Task 4.3: one line per row of the source file, same order.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_partial{len(pairs)}" if args.limit is not None else ""
    prediction_path = (
        args.output_dir / f"{split_name}_{args.strategy}{suffix}_predictions.jsonl"
    )
    write_predictions(prediction_path, records)

    total = len(pairs) or 1
    parse_failures = 1.0 - totals["parse_valid"] / total
    metrics = {
        "split": split_name,
        "strategy": args.strategy,
        "examples": len(pairs),
        "parse_failure_rate": parse_failures,
        "select_column_exact": totals["select_column_exact"] / total,
        "aggregation_exact": totals["aggregation_exact"] / total,
        "conditions_exact": totals["conditions_exact"] / total,
    }

    # Task 5.1 / 3.1: official metrics, only when the whole split was decoded.
    # Logical-form and execution accuracy come from WikiSQL/evaluate.py only --
    # this module computes no accuracy of its own beyond the Task 5.2 components.
    if not args.skip_official and args.limit is None:
        official = official_metrics(
            WIKISQL_DATA_DIR / f"{split_name}.jsonl",
            WIKISQL_DATA_DIR / f"{split_name}.db",
            prediction_path,
        )
        if official:
            metrics["lf_accuracy"] = official["lf_accuracy"]
            metrics["ex_accuracy"] = official["ex_accuracy"]

    metrics_path = (
        args.output_dir / f"{split_name}_{args.strategy}{suffix}_metrics.json"
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Examples: {len(pairs)}")
    print(f"Teacher-forced loss: {loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Teacher-forced token accuracy: {token_accuracy:.2%}")
    print(f"Parse-failure rate: {parse_failures:.2%}")
    print(f"sel column exact: {metrics['select_column_exact']:.2%}")
    print(f"agg exact: {metrics['aggregation_exact']:.2%}")
    print(f"WHERE clause exact: {metrics['conditions_exact']:.2%}")
    if "lf_accuracy" in metrics:
        print(f"Official logical-form accuracy: {metrics['lf_accuracy']:.2%}")
        print(f"Official execution accuracy: {metrics['ex_accuracy']:.2%}")
    else:
        print("Official metrics: not available (run WikiSQL/evaluate.py separately)")
    print(f"Strategy: {args.strategy}")
    print(f"Trainable parameters: {parameter_count(model):,}")
    print(f"Predictions: {prediction_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
