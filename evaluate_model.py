import argparse
import math
import re
from pathlib import Path

import torch
import torch.nn as nn

from decode import greedy_decode, parse_prediction
from model.transformer import Transformer
from starter.dataset import make_loader
from starter.embeddings import TokenEmbedding
from starter.tokenizer import PAD_ID, load_tokenizer, read_pairs


DEFAULT_TOKENIZER = "data/sql_sp.model"
DEFAULT_SPLIT = "data/test_pairs.jsonl"
DEFAULT_BEST_CHECKPOINT = "best_model.pt"
DEFAULT_LAST_CHECKPOINT = "last_model.pt"
DEFAULT_BATCH_SIZE = 48
MAX_LEN = 512


def choose_checkpoint(path):
    if path:
        return Path(path)
    last = Path(DEFAULT_LAST_CHECKPOINT)
    return last if last.exists() else Path(DEFAULT_BEST_CHECKPOINT)


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


@torch.no_grad()
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


@torch.no_grad()
def generation_metrics(model, pairs, sp, device):
    counts = {
        "exact_match": 0,
        "parse_valid": 0,
        "logical_form_exact": 0,
        "select_column_exact": 0,
        "aggregation_exact": 0,
        "conditions_exact": 0,
    }
    predictions = []

    for pair in pairs:
        src_ids = torch.tensor(
            [sp.encode(pair["src"], out_type=int)],
            dtype=torch.long,
            device=device,
        )
        _, predicted = greedy_decode(model, src_ids, sp, device)
        gold_query = parse_prediction(pair["tgt"])
        predicted_query = parse_prediction(predicted)

        counts["exact_match"] += (
            normalize_text(pair["tgt"]) == normalize_text(predicted)
        )
        counts["parse_valid"] += predicted_query is not None
        counts["logical_form_exact"] += (
            gold_query is not None and gold_query == predicted_query
        )
        if gold_query is not None and predicted_query is not None:
            counts["select_column_exact"] += gold_query["sel"] == predicted_query["sel"]
            counts["aggregation_exact"] += gold_query["agg"] == predicted_query["agg"]
            counts["conditions_exact"] += gold_query["conds"] == predicted_query["conds"]
        predictions.append((pair["src"], pair["tgt"], predicted))

    total = len(pairs)
    return {
        name: value / total if total else 0.0
        for name, value in counts.items()
    }, predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a text-to-SQL checkpoint.")
    parser.add_argument("--checkpoint", help="Checkpoint path.")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N examples.")
    parser.add_argument("--samples", type=int, default=5)
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
    sp = load_tokenizer(args.tokenizer)
    model = build_model(checkpoint, sp, device)
    pairs = read_pairs(args.split)
    if args.limit is not None:
        pairs = pairs[:args.limit]

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
    metrics, predictions = generation_metrics(model, pairs, sp, device)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Examples: {len(pairs)}")
    print(f"Teacher-forced loss: {loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Teacher-forced token accuracy: {token_accuracy:.2%}")
    print(f"Greedy exact-match accuracy: {metrics['exact_match']:.2%}")
    print(f"Parse-valid accuracy: {metrics['parse_valid']:.2%}")
    print(f"Logical-form exact accuracy: {metrics['logical_form_exact']:.2%}")
    print(f"Selected-column accuracy: {metrics['select_column_exact']:.2%}")
    print(f"Aggregation accuracy: {metrics['aggregation_exact']:.2%}")
    print(f"Conditions exact accuracy: {metrics['conditions_exact']:.2%}")

    for index, (source, gold, predicted) in enumerate(
        predictions[:max(args.samples, 0)], start=1
    ):
        print(f"\nSample {index}")
        print(f"Question:  {source.split(' <sep> ', 1)[0]}")
        print(f"Expected:  {gold}")
        print(f"Generated: {predicted}")


if __name__ == "__main__":
    main()