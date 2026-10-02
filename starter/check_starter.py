import sentencepiece as spm
from starter.dataset import make_loader
from starter.embeddings import TokenEmbedding, InputLayer
from starter.tokenizer import PAD_ID, OUT_DIR

sp = spm.SentencePieceProcessor(model_file=str(OUT_DIR / "sql_sp.model"))
train_dl = make_loader(str(OUT_DIR / "train_pairs.jsonl"), sp, train=True)
dev_dl = make_loader(str(OUT_DIR / "dev_pairs.jsonl"), sp, train=False)

src, tgt = next(iter(train_dl))
print("src", tuple(src.shape), "tgt", tuple(tgt.shape))

d_model = 256
shared = TokenEmbedding(sp.get_piece_size(), d_model, PAD_ID)
enc_in = InputLayer(shared, d_model)  # encoder input
dec_in = InputLayer(shared, d_model)  # decoder input (same weights)

x = enc_in(src)  # (B, S, 256) -> goes into YOUR encoder
y = dec_in(tgt[:, :-1])  # (B, T-1, 256) -> goes into YOUR decoder
print(" encoder input ", tuple(x.shape), " decoder input ", tuple(y.shape))
