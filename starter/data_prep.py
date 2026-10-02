import json
from pathlib import Path

DATA_DIR = Path("Wikidata")
OUT_DIR = Path("data")
AGG_OPS = ["", "MAX ", "MIN ", "COUNT ", "SUM ", "AVG"]
COND_OPS = ["=", ">", "<"]
MAX_COLS = 64  # column tokens <c0> ... <c63>


def load_split(split):
    """Return (examples, tables) for 'train', 'dev' or 'test'."""
    tables = {}
    with open(DATA_DIR / f"{split}.tables.jsonl", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            tables[t["id"]] = t
    with open(DATA_DIR / f"{split}.jsonl", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    return examples, tables


def encode_source(question, header):
    """question <sep> <c0> col name <c1> col name ..."""
    cols = " ".join(f"<c{i}> {name}" for i, name in enumerate(header))
    return f"{question.strip()} <sep> {cols}".lower()


def encode_target(sql):
    """{'sel','agg','conds'} -> 'select count <c3> where <c1> = kim manners'"""
    out = ["select "]
    if sql["agg"]:
        out.append(AGG_OPS[sql["agg"]].lower())
    out.append(f"<c{sql['sel']}>")

    for i, (col, op, val) in enumerate(sql["conds"]):
        out += [" where " if i == 0 else "and", f"<c{col}>", COND_OPS[op], str(val)]
    return " ".join(out).lower()


def build_pairs(split):
    examples, tables = load_split(split)
    pairs = []
    for ex in examples:
        header = tables[ex["table_id"]]["header"]
        pairs.append(
            {
                "table_id": ex["table_id"],
                "src": encode_source(ex["question"], header),
                "tgt": encode_target(ex["sql"]),
            }
        )
    return pairs


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "dev", "test"]:
        pairs = build_pairs(split)
        file_path = OUT_DIR / f"{split}_pairs.jsonl"
        with open(f"{file_path}", "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"{split}: {len(pairs)} pairs")
    print(pairs[0]["src"])
    print(pairs[0]["tgt"])