# GenAI project text-2-SQL from scratch

Download data from https://github.com/salesforce/WikiSQL and place it in following folder structure
WikiData/
- all copied data from wikisql
data/
- get the filtered out data from starter code

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
