"""Write the four result tables, the loss figure, and the sample list.

Run after training and after evaluate_model.py:

    python tables.py
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sentencepiece as spm

from decode import parse_prediction, query_to_sql
from starter.tokenizer import BOS_ID, EOS_ID, read_pairs
from train import transformer_lr, WARMUP_STEPS, D_MODEL, EPOCHS, BATCH_SIZE

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables.md"
SAMPLES = RESULTS / "samples.md"
TRAIN_LOG = RESULTS / "train_log.jsonl"
DATA = Path("data")
WIKISQL_DATA = Path("Wikidata")

MAX_SRC = 160
MAX_TGT = 64

SPLITS = ("train", "dev", "test")


def split_stats(split, sp):
    """Table 1: pairs, length statistics over every pair, and pairs dropped.

    Lengths are plain BPE token counts, computed over ALL pairs so the max
    can exceed max_src/max_tgt -- that is exactly why the last row exists.
    """
    pairs = read_pairs(DATA / f"{split}_pairs.jsonl")
    source_lengths = []
    target_lengths = []
    dropped = 0

    for pair in pairs:
        source_lengths.append(len(sp.encode(pair["src"])))
        target_lengths.append(len(sp.encode(pair["tgt"])))
        # Same rule as starter/dataset.py (train=True).
        src = sp.encode(pair["src"]) + [EOS_ID]
        tgt = [BOS_ID] + sp.encode(pair["tgt"]) + [EOS_ID]
        if len(src) > MAX_SRC or len(tgt) > MAX_TGT:
            dropped += 1

    return {
        "pairs": len(pairs),
        "source_mean": sum(source_lengths) / len(source_lengths),
        "source_max": max(source_lengths),
        "target_mean": sum(target_lengths) / len(target_lengths),
        "target_max": max(target_lengths),
        "dropped": dropped,
    }


def read_log(path=None):
    log_path = Path(path) if path else TRAIN_LOG
    if not log_path.exists():
        return []
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_metrics(split, strategy):
    path = RESULTS / f"{split}_{strategy}_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def prediction_lines(split, strategy):
    path = RESULTS / f"{split}_{strategy}_predictions.jsonl"
    if not path.exists():
        return None
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def headers(split):
    path = WIKISQL_DATA / f"{split}.tables.jsonl"
    if not path.exists():
        return {}
    table_map = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            table = json.loads(line)
            table_map[table["id"]] = table["header"]
    return table_map


def table1(stats):
    lines = [
        "## Table 1 - Data",
        "",
        "| | Train | Dev | Test |",
        "|---|---:|---:|---:|",
        "| Pairs | " + " | ".join(f"{stats[s]['pairs']:,}" for s in SPLITS) + " |",
        "| Mean source length (tokens) | "
        + " | ".join(f"{stats[s]['source_mean']:.2f}" for s in SPLITS)
        + " |",
        "| Max source length (tokens) | "
        + " | ".join(str(stats[s]["source_max"]) for s in SPLITS)
        + " |",
        "| Mean target length (tokens) | "
        + " | ".join(f"{stats[s]['target_mean']:.2f}" for s in SPLITS)
        + " |",
        "| Max target length (tokens) | "
        + " | ".join(str(stats[s]["target_max"]) for s in SPLITS)
        + " |",
        "| Pairs dropped as too long | "
        + str(stats["train"]["dropped"])
        + " | - | - |",
        "",
    ]
    return lines


def table2(log, params, checkpoint_epoch, best_dev_loss):
    lines = [
        "## Table 2 - Model and training",
        "",
        "| | |",
        "|---|---|",
        f"| Trainable parameters | {params:,} |",
    ]
    epochs = [entry["epoch"] for entry in log]
    trained = max(epochs) if epochs else checkpoint_epoch
    lines.append(f"| Epochs trained / best epoch | {trained} / {checkpoint_epoch} |")
    lines.append(f"| Best dev loss | {best_dev_loss:.4f} |")

    total_seconds = sum(entry.get("seconds", 0.0) for entry in log)
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    gpu = log[0].get("gpu", "unknown") if log else "not recorded"
    # Only claim a batch size the log actually saw; the current checkpoint
    # predates the log and was trained at a different value.
    batch = log[0].get("batch_size", "n/a") if log else "n/a (no training log)"
    lines.append(f"| Training time | {hours}h {minutes}m {seconds}s |")
    lines.append(f"| GPU | {gpu} |")
    lines.append(f"| Batch size | {batch} |")
    lines.append(f"| Epochs (target) | {EPOCHS} |")
    lines.append("")
    return lines


def table3():
    lines = [
        "## Table 3 - Official metrics",
        "",
        "Produced by `WikiSQL/evaluate.py` (the evaluator shipped with WikiSQL).",
        "",
        "| Split | Decoding | Logical form (%) | Execution (%) | Parse failures (%) |",
        "|---|---|---:|---:|---:|",
    ]
    rows = (
        ("dev", "greedy", "Dev greedy"),
        ("dev", "beam", "Dev beam (4)"),
        ("test", "greedy", "Test greedy"),
        ("test", "beam", "Test beam (4)"),
    )
    any_row = False
    for split, strategy, label in rows:
        data = load_metrics(split, strategy)
        if not data or data.get("lf_accuracy") is None:
            continue
        any_row = True
        lines.append(
            f"| {label} | {strategy} | {data['lf_accuracy'] * 100:.2f} | "
            f"{data['ex_accuracy'] * 100:.2f} | "
            f"{data.get('parse_failure_rate', 0.0) * 100:.2f} |"
        )
    if not any_row:
        lines.append("| - | - | n/a | n/a | n/a |")
    lines.append("")
    return lines


def table4():
    lines = [
        "## Table 4 - Component accuracy (dev)",
        "",
        "| Component | Correct (%) |",
        "|---|---:|",
    ]
    keys = (
        ("select_column_exact", "sel column correct"),
        ("aggregation_exact", "agg correct"),
        ("conditions_exact", "WHERE clause correct"),
    )
    data = load_metrics("dev", "beam") or load_metrics("dev", "greedy")
    if not data:
        lines.append("| - | n/a |")
    else:
        for key, label in keys:
            value = data.get(key)
            lines.append(
                f"| {label} | {value * 100:.2f} |"
                if value is not None
                else f"| {label} | n/a |"
            )
    lines.append("")
    return lines


def figure_loss(log):
    if not log:
        return None
    FIGURES.mkdir(parents=True, exist_ok=True)
    epochs = [entry["epoch"] for entry in log]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(epochs, [entry["train_loss"] for entry in log], marker="o", label="train loss")
    axis.plot(epochs, [entry["dev_loss"] for entry in log], marker="s", label="dev loss")
    best = min(log, key=lambda entry: entry["dev_loss"])
    axis.scatter([best["epoch"]], [best["dev_loss"]], s=140, facecolors="none",
                edgecolors="tab:red", zorder=5, label="best dev loss")
    axis.set_xlabel("epoch")
    axis.set_ylabel("cross-entropy loss")
    axis.set_title("Training and dev loss per epoch")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    path = FIGURES / "loss_curves.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def figure_lr(log):
    FIGURES.mkdir(parents=True, exist_ok=True)
    last_step = max((entry["global_step"] for entry in log), default=20000)
    steps = list(range(1, last_step + 1))
    values = [transformer_lr(step, D_MODEL, WARMUP_STEPS) for step in steps]
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(steps, values, linewidth=2)
    axis.axvline(WARMUP_STEPS, color="tab:red", linestyle="--", label="warmup = 4000")
    axis.set_xlabel("step")
    axis.set_ylabel("learning rate")
    axis.set_title("Learning-rate schedule (Eq. 3)")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    path = FIGURES / "learning_rate_schedule.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def classify_failure(gold, predicted):
    if gold is None:
        return "gold did not parse"
    if predicted is None:
        return "parse failure"
    if gold["sel"] != predicted["sel"]:
        return "wrong column"
    if gold["agg"] != predicted["agg"]:
        return "wrong aggregation"
    gold_conds = gold["conds"]
    pred_conds = predicted["conds"]
    if len(pred_conds) < len(gold_conds):
        return "missing condition"
    if len(pred_conds) > len(gold_conds):
        return "extra condition"
    if {tuple(c) for c in gold_conds} != {tuple(c) for c in pred_conds}:
        return "wrong value"
    return "unknown"


def write_samples(correct_count=5, wrong_count=5):
    """Task 4.6: ten dev examples, five correct and five wrong."""
    pairs = read_pairs(DATA / "dev_pairs.jsonl")
    predictions = prediction_lines("dev", "beam") or prediction_lines("dev", "greedy")
    if predictions is None:
        print("No dev prediction file; run evaluate_model.py first.")
        return None
    table_headers = headers("dev")

    correct, wrong = [], []
    for index, (pair, prediction) in enumerate(zip(pairs, predictions)):
        gold = parse_prediction(pair["tgt"])
        predicted = (
            parse_prediction_from_record(prediction)
        )
        entry = (index, pair, gold, predicted)
        if gold is not None and predicted is not None and gold == predicted:
            correct.append(entry)
        elif len(wrong) < wrong_count:
            wrong.append(entry)

    chosen = correct[:correct_count] + wrong[:wrong_count]
    lines = [
        "# Qualitative samples",
        "",
        f"{len(correct[:correct_count])} correct and "
        f"{len(wrong[:wrong_count])} incorrect dev examples "
        f"(of {len(pairs):,} dev rows).",
        "",
    ]
    for index, pair, gold, predicted in chosen:
        question = pair["src"].split(" <sep> ", 1)[0]
        header = table_headers.get(pair["table_id"], [])
        status = (
            "CORRECT"
            if gold == predicted
            else f"WRONG ({classify_failure(gold, predicted)})"
        )
        lines.extend(
            [
                f"### Example {index} - {status}",
                "",
                f"- **Question:** {question}",
                f"- **Gold SQL:** `{query_to_sql(gold, header) or '-'}`",
                f"- **Model SQL:** `{query_to_sql(predicted, header) or '-'}`",
                "",
            ]
        )
    SAMPLES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return SAMPLES


def parse_prediction_from_record(record):
    if "error" in record:
        return None
    return record.get("query")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=int, default=None,
                        help="Trainable parameter count (defaults to checkpoint).")
    parser.add_argument("--checkpoint-epoch", type=int, default=None)
    parser.add_argument("--best-dev-loss", type=float, default=None)
    parser.add_argument(
        "--log",
        default=None,
        help=(
            "Training log to summarise (default results/train_log.jsonl). Point it "
            "at a tagged log, e.g. results/train_log_60ep.jsonl, to describe an "
            "ablation run instead of the spec run."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="best_model.pt",
        help="Checkpoint to read epoch / dev loss / parameter count from.",
    )
    args = parser.parse_args()

    params = args.params
    checkpoint_epoch = args.checkpoint_epoch
    best_dev_loss = args.best_dev_loss

    sp = spm.SentencePieceProcessor(model_file=str(DATA / "sql_sp.model"))

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_epoch = checkpoint.get("epoch", checkpoint_epoch)
        best_dev_loss = checkpoint.get("best_dev_loss", best_dev_loss)
        if params is None:
            # Count through the model, not the state dict: the embedding and
            # the output projection are one tensor, and summing state-dict
            # entries counts it twice.
            from evaluate_model import build_model, parameter_count

            model = build_model(checkpoint, sp, torch.device("cpu"))
            params = parameter_count(model)

    log = read_log(args.log)
    stats = {}
    for split in SPLITS:
        stats[split] = split_stats(split, sp)

    lines = ["# Results", ""]
    lines += table1(stats)
    lines += table2(
        log,
        params or 0,
        checkpoint_epoch or 0,
        best_dev_loss if best_dev_loss is not None else float("nan"),
    )
    lines += table3()
    lines += table4()

    loss_figure = figure_loss(log)
    lr_figure = figure_lr(log)
    if loss_figure:
        lines += ["## Figures", "", f"- Training and dev loss: `{loss_figure}`", ""]
    if lr_figure:
        lines += [f"- Learning-rate schedule: `{lr_figure}`", ""]

    RESULTS.mkdir(parents=True, exist_ok=True)
    TABLES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {TABLES}")
    if loss_figure:
        print(f"Wrote {loss_figure}")
    if lr_figure:
        print(f"Wrote {lr_figure}")

    samples = write_samples()
    if samples:
        print(f"Wrote {samples}")


if __name__ == "__main__":
    main()
