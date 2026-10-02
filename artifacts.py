import sentencepiece as spm
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from starter.dataset import make_loader
from starter.embeddings import PositionalEncoding
from starter.data_prep import OUT_DIR

sp = spm.SentencePieceProcessor(model_file=str(OUT_DIR / "sql_sp.model"))
train_dl = make_loader(str(OUT_DIR / "train_pairs.jsonl"), sp, train=True)
src, tgt = next(iter(train_dl))

D_MODEL, N_POS = 256, 100
positional_encoding = PositionalEncoding(D_MODEL).pe[0, :N_POS].numpy()  # (100, 256)

fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(
    positional_encoding,
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    aspect="auto",
    interpolation="nearest",
)
ax.set_title(f"Sinusoidal positional encoding ({N_POS} positions x {D_MODEL} dims)")
ax.set_xlabel("embedding dimension i")
ax.set_ylabel("position p")
fig.colorbar(im, ax=ax, label="PE value")
fig.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "pe_heatmap.png", dpi=150)
print("PE matrix", tuple(positional_encoding.shape), "-> ", OUT_DIR / "pe_heatmap.png")

