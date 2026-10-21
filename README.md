# Text-to-SQL with the Transformer (built and trained from scratch)

An encoder–decoder Transformer from *Attention Is All You Need* (Vaswani et al., 2017),
implemented from basic PyTorch layers, trained from random initialisation on
[WikiSQL](https://github.com/salesforce/WikiSQL) to turn an English question plus a
table's column names into a SQL query.

**No pretrained weights, no `nn.Transformer`, no `nn.MultiheadAttention`,
no Hugging Face `transformers`.**

---

## Repository layout

```
starter/                 the four given files + check_starter.py (unchanged)
model/attention.py       scaled dot-product and multi-head attention
model/layers.py          feed-forward, encoder layer, decoder layer
model/transformer.py     full model and masks
model/helper.py          padding mask and causal mask
train.py                 Task 3  – training
decode.py                Task 4  – greedy, beam search, parser, readable SQL
evaluate_model.py        Task 5  – prediction files, component accuracy
checks.py                section 3.2 correctness checks
tables.py                section 4  – Tables 1-4, figures, samples
analysis.py              positional-encoding heat-map and cross-attention map
train_colab.ipynb        Colab training notebook
results/                 prediction files, tables, figures, samples.md
app.py                   backend that serves the front end
UI/                      front end
```

---

## Quick start (local)

```powershell
# 1. environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. data: download data.tar.bz2 from salesforce/WikiSQL, then
tar xjf data.tar.bz2          # creates data/
Move-Item data Wikidata       # this repo reads Wikidata/

# 3. official evaluator (needed for the real accuracy numbers)
git clone https://github.com/salesforce/WikiSQL WikiSQL
git -C WikiSQL checkout HEAD -- evaluate.py lib

# 4. build pairs + shared 8000-piece BPE vocabulary
python -m starter.data_prep
python -m starter.tokenizer
python -m starter.check_starter

# 5. train (20 epochs, batch 64, GPU if available)
python train.py

# 6. correctness checks – all six must pass
python checks.py

# 7. decode and score
python evaluate_model.py --split data/dev_pairs.jsonl --strategy greedy
python evaluate_model.py --split data/dev_pairs.jsonl --strategy beam --beam-size 4
python tables.py
python analysis.py --checkpoint best_model.pt --example 0
```

Run the test split **once**, at the very end, with your final decoding choice:

```powershell
python evaluate_model.py --split data/test_pairs.jsonl --strategy beam --beam-size 4
```

> Set `PYTHONUTF8=1` on Windows. The official evaluator opens files without an
> encoding, which otherwise fails on non-ASCII questions.

---

## Training on Kaggle

Open [`train_colab.ipynb`](./train_colab.ipynb) as a Kaggle notebook. In the notebook
settings (right panel) select **Accelerator → GPU (T4 x2 or P100)** and enable
**Internet**, then run the cells in order.

The notebook clones this repository, fetches WikiSQL, rebuilds `data/`, trains for
20 epochs at batch size 64, runs the six correctness checks, and copies
`best_model.pt` / `last_model.pt` / the training log / `results/` into
`/kaggle/working/artifacts/` — Kaggle saves that folder automatically as the
notebook's Output. Roughly 30–60 minutes on a T4.

> Push your latest commits first: the notebook clones the GitHub repo, and the
> clone must contain `checks.py` and `tables.py` or cells 8 and 11 will fail.

## Serving the model from a Colab T4

[`inference_ngrok.ipynb`](./inference_ngrok.ipynb) runs in Colab (T4 GPU): mount
Drive with `best_model.pt` + `data/sql_sp.model`, and it loads your trained model,
serves the `UI/` front end and `/api/query` (greedy **and** beam) under Flask, and
exposes it through an ngrok tunnel.

---

## Model configuration

| | |
|---|---|
| `d_model` | 256 |
| Heads | 4 (`d_k = d_v = 64`) |
| Encoder / decoder layers | 3 / 3 |
| Feed-forward inner size | 1024 |
| Dropout | 0.1 (embeddings + every sublayer) |
| Normalisation | post-norm, `LayerNorm(x + Sublayer(x))` |
| Weight sharing | encoder embedding = decoder embedding = output projection |
| Optimiser | Adam, β₁ = 0.9, β₂ = 0.98, ε = 1e-9 |
| LR schedule | Eq. 3, warmup 4000 |
| Loss | cross-entropy, label smoothing ε = 0.1, `<pad>` ignored |
| Batch size / epochs | 64 / 20 |

---

## Evaluation

Logical-form and execution accuracy come from the evaluator shipped with WikiSQL —
this repository never computes them itself.

```powershell
$env:PYTHONUTF8 = "1"
python WikiSQL\evaluate.py Wikidata\dev.jsonl Wikidata\dev.db results\dev_greedy_predictions.jsonl
```

It prints two numbers:

- **Logical-form accuracy** – predicted query equals the gold query (sel, agg, condition set)
- **Execution accuracy** – the predicted query, run on the real table, returns the same result

`evaluate_model.py` calls this automatically and also reports the parse-failure rate
and the Task 5.2 component accuracies (sel column / agg / WHERE clause).

Prediction files contain exactly one JSON line per source row, in source order.
Unparseable generations are written as `{"error": "parse"}` — lines are never skipped.

---

## Correctness checks (section 3.2)

Output of `python checks.py`:

```
Checkpoint: best_model.pt  Device: cuda

[PASS] Causal mask: max |delta| over earlier positions = 0.000e+00 (target < 1e-5)
[PASS] Padding mask: max |delta| after 7 extra <pad> = 0.000e+00 (target < 1e-5)
[PASS] Attention rows sum to 1: max |row sum - 1| = 1.788e-07 (target < 1e-5)
[PASS] Weight sharing: output_projection.weight is embedding.weight
[PASS] Learning-rate schedule: peak@4000 = 9.882e-04, value@20000 = 4.419e-04
       (expected 4.419e-04); linear ramp = True, decays after warmup = True

[PASS] Gold round-trip: execution accuracy = 100.00% over 8421 gold targets,
       0 unparseable (target > 99%, 0 unparseable)

6/6 checks passed.
```

The gold round-trip feeds every dev gold target through `parse_prediction` and the
official evaluator — it is the check that catches parser and evaluator bugs.

---

## Results

Generated by `python tables.py` into [`results/tables.md`](./results/tables.md),
with figures in `results/figures/` and ten qualitative examples in
[`results/samples.md`](./results/samples.md).

---

## Front end

The interface takes a question and a comma-separated list of column names and
returns the generated SQL with real column names.

```powershell
$env:HF_TOKEN = "..."
python app.py
# open http://localhost:5000/query.html
```

<!-- FRONTEND_SCREENSHOT: add a screenshot of the running UI here -->
