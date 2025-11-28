import os, json, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

TASK_DIR = os.environ.get("TASK_DIR", "task_A/task_A")
BASE     = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
OUT_DIR  = os.environ.get("OUT_DIR",  "task_A/outputs/llm_lora_taskA")

SEQ      = int(os.environ.get("SEQ", 2048))
BSZ      = int(os.environ.get("BSZ", 2))
GA       = int(os.environ.get("GA", 8))
LR       = float(os.environ.get("LR", 2e-4))
EPOCHS   = int(os.environ.get("EPOCHS", 1))
WARMUP   = float(os.environ.get("WARMUP", 0.1))

train = load_dataset("parquet", data_files=f"{TASK_DIR}/train.parquet")["train"]
dev   = load_dataset("parquet", data_files=f"{TASK_DIR}/dev.parquet")["train"]

# Allowed label IDs for Task A
VALID = ["0", "1"]

SYS = "You are a precise annotator. Reply with exactly one label id."
USR = ("Task: Given the following code, predict the label id.\n"
       f"Allowed label ids: {', '.join(VALID)}.\n\n"
       "Code:\n{code}\n\nAnswer with a single id only.")

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True, padding_side="left")
if tok.pad_token is None: tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")

lora = LoraConfig(
    r=int(os.environ.get("LORA_R",16)),
    lora_alpha=int(os.environ.get("LORA_ALPHA",32)),
    lora_dropout=float(os.environ.get("LORA_DROPOUT",0.05)),
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

def render(ex):
    msgs=[{"role":"system","content":SYS},
          {"role":"user","content":USR.format(code=ex["code"])},
          {"role":"assistant","content":str(ex["label"])}]
    return {"text": tok.apply_chat_template(msgs, tokenize=False)}

train = train.map(render)
dev   = dev.map(render)

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BSZ,
    per_device_eval_batch_size=max(1, BSZ*2),
    gradient_accumulation_steps=GA,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    warmup_ratio=WARMUP,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    bf16=True,
    gradient_checkpointing=True,
    save_total_limit=2,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    peft_config=lora,
    train_dataset=train,
    eval_dataset=dev,
    dataset_text_field="text",
    max_seq_length=SEQ,
    packing=False,
    args=args
)

trainer.train()
# Save the LoRA adapter + tokenizer to OUT_DIR/best
best_dir = os.path.join(OUT_DIR, "best")
trainer.model.save_pretrained(best_dir)
tok.save_pretrained(best_dir)

# record base model so prediction can re-load it
with open(os.path.join(best_dir, "base_model.txt"), "w") as f:
    f.write(BASE)
print("Saved adapter to:", best_dir)
