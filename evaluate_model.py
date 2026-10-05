import argparse
import json
import math
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn

from decode import beam_search, greedy_decode, parse_prediction
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
DEFAULT_OUTPUT_DIR = Path("results")


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


def normalize_conditions(conditions):
    return sorted(
        (int(column), int(operator), normalize_text(str(value)))
        for column, operator, value in conditions
    )


def logical_form_equal(expected, predicted):
    if expected is None or predicted is None:
        return False
    return (
        expected["sel"] == predicted["sel"]
        and expected["agg"] == predicted["agg"]
        and normalize_conditions(expected["conds"])
        == normalize_conditions(predicted["conds"])
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


@lru_cache(maxsize=None)
def table_lookup(split):
    tables_path = Path("Wikidata") / f"{split}.tables.jsonl"
    tables = {}
    with tables_path.open(encoding="utf-8") as handle:
        for line in handle:
            table = json.loads(line)
            tables[table["id"]] = table
    return tables


def execution_equal(split, pair, expected, predicted):
    if expected is None or predicted is None:
        return False
    tables = table_lookup(split)
    table = tables.get(pair["table_id"])
    if table is None:
        return False
    header = table["header"]
    database = Path("Wikidata") / f"{split}.db"
    table_name = f'table_{table["id"].replace("-", "_")}'

    def executable_sql(query):
        if query["sel"] < 0 or query["sel"] >= len(header):
            return None
        aggregation = ""
        if query["agg"]:
            if query["agg"] >= len(("none", "max", "min", "count", "sum", "avg")):
                return None
            aggregation = ("MAX", "MIN", "COUNT", "SUM", "AVG")[query["agg"] - 1]
        selected = f'"{header[query["sel"]].replace(chr(34), chr(34) * 2)}"'
        expression = f"{aggregation}({selected})" if aggregation else selected
        sql = f'SELECT {expression} FROM "{table_name}"'
        conditions = []
        for column, operator, value in query["conds"]:
            if column < 0 or column >= len(header) or operator not in (0, 1, 2):
                return None
            identifier = f'"{header[column].replace(chr(34), chr(34) * 2)}"'
            operator_text = ("=", ">", "<")[operator]
            escaped_value = str(value).replace("'", "''")
            conditions.append(f"{identifier} {operator_text} '{escaped_value}'")
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        return sql

    expected_sql = executable_sql(expected)
    predicted_sql = executable_sql(predicted)
    if expected_sql is None or predicted_sql is None:
        return False
    try:
        with sqlite3.connect(database) as connection:
            expected_rows = connection.execute(expected_sql).fetchall()
            predicted_rows = connection.execute(predicted_sql).fetchall()
    except sqlite3.Error:
        return False
    return sorted(expected_rows) == sorted(predicted_rows)


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
def generation_metrics(model, pairs, sp, device, split, strategy, beam_size):
    counts = {
        "exact_match": 0,
        "parse_valid": 0,
        "logical_form_exact": 0,
        "execution_exact": 0,
        "select_column_exact": 0,
        "aggregation_exact": 0,
        "conditions_exact": 0,
    }
    predictions = []

    for pair in pairs:
        _, predicted = decode_example(
            model, pair, sp, device, strategy, beam_size
        )
        gold_query = parse_prediction(pair["tgt"])
        predicted_query = parse_prediction(predicted)

        counts["exact_match"] += (
            normalize_text(pair["tgt"]) == normalize_text(predicted)
        )
        counts["parse_valid"] += predicted_query is not None
        counts["logical_form_exact"] += logical_form_equal(
            gold_query, predicted_query
        )
        counts["execution_exact"] += execution_equal(
            split, pair, gold_query, predicted_query
        )
        if gold_query is not None and predicted_query is not None:
            counts["select_column_exact"] += gold_query["sel"] == predicted_query["sel"]
            counts["aggregation_exact"] += gold_query["agg"] == predicted_query["agg"]
            counts["conditions_exact"] += normalize_conditions(
                gold_query["conds"]
            ) == normalize_conditions(predicted_query["conds"])
        predictions.append((pair["src"], pair["tgt"], predicted))

    total = len(pairs)
    return {
        name: value / total if total else 0.0
        for name, value in counts.items()
    }, predictions


def prediction_records(pairs, predictions):
    records = []
    for pair, (_, _, text) in zip(pairs, predictions):
        parsed = parse_prediction(text)
        records.append(
            {"query": parsed} if parsed is not None else {"error": "parse"}
        )
    return records


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a text-to-SQL checkpoint.")
    parser.add_argument("--checkpoint", help="Checkpoint path.")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N examples.")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--strategy", choices=("greedy", "beam"), default="greedy")
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    split_name = Path(args.split).stem.replace("_pairs", "")
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
    metrics, predictions = generation_metrics(
        model,
        pairs,
        sp,
        device,
        split_name,
        args.strategy,
        args.beam_size,
    )
    prediction_path = args.output_dir / (
        f"{split_name}_{args.strategy}_predictions.jsonl"
    )
    write_predictions(prediction_path, prediction_records(pairs, predictions))
    metrics_path = args.output_dir / (
        f"{split_name}_{args.strategy}_metrics.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

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
    print(f"Execution exact accuracy: {metrics['execution_exact']:.2%}")
    print(f"Selected-column accuracy: {metrics['select_column_exact']:.2%}")
    print(f"Aggregation accuracy: {metrics['aggregation_exact']:.2%}")
    print(f"Conditions exact accuracy: {metrics['conditions_exact']:.2%}")
    print(f"Strategy: {args.strategy}")
    print(f"Trainable parameters: {parameter_count(model):,}")
    print(f"Predictions: {prediction_path}")
    print(f"Metrics: {metrics_path}")

    for index, (source, gold, predicted) in enumerate(
        predictions[:max(args.samples, 0)], start=1
    ):
        print(f"\nSample {index}")
        print(f"Question:  {source.split(' <sep> ', 1)[0]}")
        print(f"Expected:  {gold}")
        print(f"Generated: {predicted}")


if __name__ == "__main__":
    main()