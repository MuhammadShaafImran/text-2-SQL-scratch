"""Generate artifacts and a decoder cross-attention plot."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from evaluate_model import build_model, parameter_count
from starter.embeddings import PositionalEncoding
from starter.tokenizer import BOS_ID, EOS_ID, load_tokenizer, read_pairs


def token_lengths(path, sp):
    pairs = read_pairs(path)
    source_lengths = [len(sp.encode(pair["src"], out_type=int)) for pair in pairs]
    target_lengths = [len(sp.encode(pair["tgt"], out_type=int)) for pair in pairs]

    def summary(values):
        return {
            "count": len(values),
            "mean": sum(values) / len(values) if values else 0.0,
            "maximum": max(values) if values else 0,
        }

    return {"source": summary(source_lengths), "target": summary(target_lengths)}


@torch.no_grad()
def attention_for_example(model, pair, sp, device):
    source_ids = torch.tensor(
        [sp.encode(pair["src"], out_type=int) + [EOS_ID]],
        dtype=torch.long,
        device=device,
    )
    generated = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
    weights = []

    for _ in range(64):
        logits, cross_attention = model(source_ids, generated)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        weights.append(cross_attention[0, :, -1, :].detach().cpu())
        generated = torch.cat((generated, next_token), dim=1)
        if next_token.item() == EOS_ID:
            break

    token_ids = generated[0, 1:].tolist()
    if EOS_ID in token_ids:
        token_ids = token_ids[:token_ids.index(EOS_ID)]
    attention = torch.stack(weights[: len(token_ids)]).mean(dim=1)
    source_tokens = [sp.id_to_piece(i) for i in source_ids[0].tolist()]
    target_tokens = [sp.id_to_piece(i) for i in token_ids]
    return attention.numpy(), source_tokens, target_tokens


def plot_attention(matrix, source_tokens, target_tokens, output_path):
    figure, axis = plt.subplots(
        figsize=(max(10, len(source_tokens) * 0.24), max(5, len(target_tokens) * 0.3))
    )
    image = axis.imshow(matrix, aspect="auto", cmap="magma")
    axis.set_xticks(range(len(source_tokens)))
    axis.set_xticklabels(source_tokens, rotation=90, fontsize=7)
    axis.set_yticks(range(len(target_tokens)))
    axis.set_yticklabels(target_tokens, fontsize=8)
    axis.set_xlabel("Source tokens")
    axis.set_ylabel("Generated target tokens")
    axis.set_title("Decoder cross-attention, last layer, head average")
    figure.colorbar(image, ax=axis, label="attention weight")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_report(output_dir, stats, params, attention_path):
    report = output_dir / "analysis_report.md"
    lines = [
        "# Text-to-SQL analysis",
        "",
        "## Starter statistics",
        "",
        "| Split | Pairs | Source mean | Source max | Target mean | Target max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, values in stats.items():
        lines.append(
            f"| {split} | {values['source']['count']:,} | "
            f"{values['source']['mean']:.2f} | {values['source']['maximum']} | "
            f"{values['target']['mean']:.2f} | {values['target']['maximum']} |"
        )
    lines.extend(
        [
            "",
            "## Model facts",
            "",
            f"- Trainable parameters: **{params:,}**.",
            "- Configuration: d_model=256, heads=4, encoder/decoder layers=3/3, "
            "feed-forward size=1024, dropout=0.1, post-norm LayerNorm.",
            "- `starter/check_starter.py` reports `(B, S)` and `(B, T)` token "
            "batches, followed by `(B, S, 256)` and `(B, T-1, 256)` embeddings.",
            "",
            "## Positional encoding",
            "",
            "The heatmap shows alternating sinusoidal patterns across positions "
            "and embedding dimensions. Low-frequency dimensions change slowly while "
            "high-frequency dimensions change rapidly, giving every position a "
            "distinct representation that can be extrapolated beyond training "
            "positions.",
            "",
            "## Cross-attention",
            "",
            f"Generated figure: `{attention_path.name}`. It averages the heads "
            "from the final decoder layer for one dev example.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--tokenizer", default="data/sql_sp.model")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation"))
    parser.add_argument("--example", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = load_tokenizer(args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint, sp, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        split: token_lengths(f"data/{split}_pairs.jsonl", sp)
        for split in ("train", "dev", "test")
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(
            {
                "starter_statistics": stats,
                "trainable_parameters": parameter_count(model),
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": checkpoint.get("epoch"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    positional = PositionalEncoding(256).pe[0, :100].numpy()
    figure, axis = plt.subplots(figsize=(12, 4))
    image = axis.imshow(positional, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axis.set_title("Sinusoidal positional encoding (100 positions x 256 dimensions)")
    axis.set_xlabel("embedding dimension")
    axis.set_ylabel("position")
    figure.colorbar(image, ax=axis, label="value")
    figure.tight_layout()
    figure.savefig(args.output_dir / "positional_encoding.png", dpi=180)
    plt.close(figure)

    pairs = read_pairs("data/dev_pairs.jsonl")
    if not 0 <= args.example < len(pairs):
        raise ValueError(f"--example must be between 0 and {len(pairs) - 1}")
    matrix, source_tokens, target_tokens = attention_for_example(
        model, pairs[args.example], sp, device
    )
    attention_path = args.output_dir / "cross_attention_dev.png"
    plot_attention(matrix, source_tokens, target_tokens, attention_path)
    write_report(
        args.output_dir,
        stats,
        parameter_count(model),
        attention_path,
    )
    print(f"Wrote analysis artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
