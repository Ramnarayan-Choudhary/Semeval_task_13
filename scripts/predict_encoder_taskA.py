import os, sys, pandas as pd, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

TASK_DIR  = os.environ.get("TASK_DIR", "task_A/task_A")
SPLIT     = os.environ.get("SPLIT", "test")  # "test" or "dev"
MODEL_DIR = os.environ.get("MODEL_DIR", "task_A/outputs/enc_taskA/best")
OUT_CSV   = os.environ.get("OUT_CSV", f"task_A/outputs/submission_taskA_enc_{SPLIT}.csv")
ID_COL    = os.environ.get("ID_COL")  # optional override

split_file = "test.parquet" if SPLIT == "test" else "dev.parquet"
ds = load_dataset("parquet", data_files=f"{TASK_DIR}/{split_file}")["train"]
cols = list(ds.column_names)

# pick an id column if present
if ID_COL and ID_COL in cols:
    id_col = ID_COL
else:
    candidates = [c for c in ["id","ID","Id","idx","index","row_id","sample_id"] if c in cols]
    id_col = candidates[0] if candidates else None
    if ID_COL and ID_COL not in cols:
        print(f"[warn] ID_COL={ID_COL} not in columns {cols}", file=sys.stderr)

tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

rows = []
for i, ex in enumerate(ds):
    enc = tok(ex["code"], return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        pred = int(torch.argmax(logits, dim=-1).item())
    ex_id = ex[id_col] if id_col is not None else i  # fallback: row index
    rows.append({"id": ex_id, "label": pred})

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print("Wrote", OUT_CSV)
