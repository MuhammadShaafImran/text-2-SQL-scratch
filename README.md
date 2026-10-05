# GenAI project text-2-SQL from scratch

Download data from https://github.com/salesforce/WikiSQL and place it in following folder structure
WikiData/
- all copied data from wikisql
data/
- get the filtered out data from starter code

## Colab + Google Drive + ngrok inference

Open [inference_ngrok.ipynb](./inference_ngrok.ipynb) in Google Colab. Put only
`best_model.pt` and `data/sql_sp.model` in a Google Drive folder. The notebook
clones the Python source from this GitHub repository, loads the model artifacts
from Drive, starts `POST /api/query`, and prints an ngrok URL.

The deployed landing page accepts the printed endpoint URL. Paste either the
ngrok base URL or its `/api/query` URL into the **Inference endpoint** field.
The value is stored in the browser and passed to Query Studio automatically.

## Deploy the UI to Vercel

This project deploys only the frontend to Vercel. PyTorch inference remains in
the Google Colab notebook and the temporary ngrok URL remains the backend.

1. Run [inference_ngrok.ipynb](./inference_ngrok.ipynb) in Google Colab.
2. Keep the Colab runtime running and copy its printed ngrok URL.
3. Push this repository to GitHub, excluding model checkpoints from the
   frontend deployment.
4. Import the repository into Vercel. No build command is required.
5. Open the deployed Vercel URL and paste the Colab URL into **Inference
   endpoint**.
6. Click **USE ENDPOINT**, then open Query Studio.

The [vercel.json](./vercel.json) file maps the repository root to `UI/landing.html`
and maps the UI assets so the existing relative links work on Vercel. The
endpoint is stored in the visitor's browser only; each browser must configure
the current Colab/ngrok endpoint separately. A new ngrok URL must be entered
again after the tunnel restarts.

## Evaluation and analysis

The evaluation script produces WikiSQL-compatible logical-form and execution
metrics, component accuracies, and one JSONL prediction per input row. It supports
both required decoding strategies:

```powershell
python evaluate_model.py --split data/dev_pairs.jsonl --strategy greedy
python evaluate_model.py --split data/dev_pairs.jsonl --strategy beam --beam-size 4
python evaluate_model.py --split data/test_pairs.jsonl --strategy beam --beam-size 4
```

Outputs are written under `results/` as `*_predictions.jsonl` and
`*_metrics.json`. A malformed generation is represented by `{"error": "parse"}`;
valid generations use the required `{"query": {...}}` shape.

Generate the starter statistics, parameter count, positional-encoding figure,
cross-attention heatmap, and a short analysis report with:

```powershell
python analysis.py --checkpoint best_model.pt --example 0
```

The report and figures are written to `results/`. The cross-attention plot is
the averaged final decoder-layer attention for the selected dev example.

For a direct link, open the query page with the `api` query parameter:

```text
http://localhost:5000/query.html?api=https%3A%2F%2FYOUR-NGROK-URL.ngrok-free.app%2Fapi%2Fquery
```

Alternatively, set `window.INFERENCE_API_URL` before loading `script.js`. The
UI keeps using `/api/query` by default when no override is set.

## Hosted model endpoint

The UI can be served with the Hugging Face-hosted model through `app.py`.
Set `HF_TOKEN` before starting the server:

```powershell
$env:HF_TOKEN = "your-hugging-face-token"
python app.py
```

Open `http://localhost:5000/query.html`. The UI sends questions to
`POST /api/query`; the endpoint calls `shaaf257/text-2-sql-scratch` and returns
the generated SQL in the `sql` field.
