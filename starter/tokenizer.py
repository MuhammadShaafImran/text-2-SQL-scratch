import json
import sentencepiece as spm
from starter.data_prep import MAX_COLS
from pathlib import Path

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL = ["<sep>"] + [f"<c{i}>" for i in range(MAX_COLS)]
OUT_DIR = Path('data')

def read_pairs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def train_tokenizer(train_path=str(OUT_DIR / "train_pairs.jsonl"), vocab_size=8000):
    pairs = read_pairs(train_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = OUT_DIR / "spm_corpus.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        for p in pairs:  # ONE shared vocabulary for source and target, as in the original Transformer
            f.write(p["src"] + "\n")
            f.write(p["tgt"] + "\n")
    spm.SentencePieceTrainer.train(
        input=str(file_path),
        model_prefix=str(OUT_DIR / "sql_sp"),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        user_defined_symbols=SPECIAL,  # never split <sep>, <c0>, ...
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
    )
    return spm.SentencePieceProcessor(model_file=str(OUT_DIR / "sql_sp.model"))


if __name__ == "__main__":
    data = Path('data')
    sp = train_tokenizer(str(data/'train_pairs.jsonl'))
    p = read_pairs(data/"dev_pairs.jsonl")[0]
    print(sp.encode(p["src"], out_type=str))
    print(sp.encode(p["tgt"], out_type=str))
    print(sp.decode(sp.encode(p["tgt"])) == p["tgt"])
