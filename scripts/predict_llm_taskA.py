import os, json, torch, pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

TASK_DIR  = os.environ.get("TASK_DIR", "task_A/task_A")
MODEL_DIR = os.environ.get("MODEL_DIR","task_A/outputs/llm_lora_taskA/best")
OUT_CSV   = os.environ.get("OUT_CSV", "task_A/outputs/submission_taskA_llm.csv")
SPLIT     = os.environ.get("SPLIT", "dev")  # "dev" or "test"

# Find base model name
base_txt = os.path.join(MODEL_DIR, "base_model.txt")
if os.path.exists(base_txt):
    with open(base_txt) as f: BASE = f.read().strip()
else:
    BASE = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, padding_side="left")
if tok.pad_token is None: tok.pad_token = tok.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base  = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, MODEL_DIR).to(device).eval()

VALID = ["0","1"]
SYS = "You are a precise annotator. Reply with exactly one label id."
USR = ("Task: Given the following code, predict the label id.\n"
       f"Allowed label ids: {', '.join(VALID)}.\n\n"
       "Code:\n{code}\n\nAnswer with a single id only.")

split_file = "dev.parquet" if SPLIT=="dev" else "test.parquet"
ds = load_dataset("parquet", data_files=f"{TASK_DIR}/{split_file}")["train"]
rows=[]
for i, ex in enumerate(ds):
    msgs=[{"role":"system","content":SYS},{"role":"user","content":USR.format(code=ex["code"])}]
    inps = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(inps, max_new_tokens=3, do_sample=False, eos_token_id=tok.eos_token_id)
    text = tok.decode(out[0, inps.shape[1]:], skip_special_tokens=True).strip()
    token = "".join(ch for ch in text if ch.isdigit())
    pred = int(token) if token in VALID else (1 if "1" in text else 0)
    rows.append({"id": i, "label": pred})  # HF dev/test here has no 'id'; use row index

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print("Wrote", OUT_CSV)
