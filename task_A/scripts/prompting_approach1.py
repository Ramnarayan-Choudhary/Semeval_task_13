import os
import random
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# Quick config (override with env vars if needed)
MODEL_NAME = os.getenv("HF_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "1") == "1"
CHECKPOINT_FILE = "predictions_checkpoint.csv"
OUTPUT_FILE = "final_predictions.csv"
SAMPLE_SUBMISSION_FILE = os.getenv("SAMPLE_SUBMISSION", "")

FEW_SHOT_COUNT = 4
FEW_SHOT_POOL_SIZE = 4000
MAX_CODE_SNIPPET_CHARS = 500
MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 6
TEMPERATURE = 0.0
TOP_P = 0.9
RANDOM_SEED = 42

SYSTEM_PROMPT = (
    "You are a concise code forensics assistant. Decide if a snippet is human-written (0) "
    "or machine-generated (1). Look for template-like comments, uniform formatting, generic "
    "naming, over-engineered structure, and exhaustive edge handling. Respond with only 0 or 1."
)


def truncate_code(code: str, max_chars: int = MAX_CODE_SNIPPET_CHARS) -> str:
    code = (code or "").strip()
    if len(code) <= max_chars:
        return code
    return code[:max_chars] + "\n... [truncated]"


def tokenize_code(code: str) -> set[str]:
    """Lightweight tokenization that strips strings/comments for Jaccard overlap."""
    code = re.sub(r'\"\"\".*?\"\"\"', "", code, flags=re.DOTALL)
    code = re.sub(r"\'\'\'.*?\'\'\'", "", code, flags=re.DOTALL)
    code = re.sub(r'\".*?\"', "", code)
    code = re.sub(r"\'.*?\'", "", code)
    code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    # keep operands and punctuation; avoid bad ranges by not over-escaping
    # simple token grabber; no escaping tricks to avoid regex range issues
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[-+*/%=<>!&|^~]+|[\[\]{}();,.]", code)
    return {t.lower() for t in tokens}


def jaccard(s1: set[str], s2: set[str]) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def create_detection_prompt(code_snippet: str, language: str, few_shot_examples: List[Dict[str, Any]]) -> str:
    header = (
        "Classify code as human (0) or AI (1). Focus on naming quirks, formatting rhythm, "
        "comment tone, and whether control flow looks hand-written or templated."
    )
    blocks = []
    for i, example in enumerate(few_shot_examples, 1):
        label = int(example.get("label", 1))
        lang = example.get("language", "Unknown")
        author = "Human" if label == 0 else "AI"
        blocks.append(
            f"Example {i} [{lang}] label={label} ({author}):\n"
            f"```{lang.lower()}\n{truncate_code(example.get('code', ''))}\n```"
        )

    prompt = (
        f"{header}\n\n"
        f"Few-shot references:\n" + "\n\n".join(blocks) + "\n\n"
        f"Now classify this {language} snippet:\n"
        f"```{language.lower()}\n{truncate_code(code_snippet)}\n```\n"
        "Return only 0 or 1."
    )
    return prompt


def format_prompt_for_model(user_prompt: str, tokenizer: AutoTokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "apply_chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"{SYSTEM_PROMPT}\n\n{user_prompt}\nAssistant:"


def extract_prediction(response_text: str) -> int:
    response_text = (response_text or "").strip()
    for char in response_text:
        if char in ["0", "1"]:
            return int(char)
    lower_response = response_text.lower()
    if "human" in lower_response and "ai" not in lower_response:
        return 0
    if any(word in lower_response for word in ["ai", "machine", "generated"]):
        return 1
    print(f"Warning: unclear response '{response_text}', defaulting to 1")
    return 1


def load_checkpoint() -> pd.DataFrame:
    if os.path.exists(CHECKPOINT_FILE):
        df = pd.read_csv(CHECKPOINT_FILE)
        print(f"Loaded {len(df)} predictions from checkpoint")
        return df
    return pd.DataFrame(columns=["ID", "label"])


def save_checkpoint(predictions_df: pd.DataFrame) -> None:
    predictions_df.to_csv(CHECKPOINT_FILE, index=False)


def resolve_ids(test_df: pd.DataFrame, sample_submission_path: Path) -> List[Any]:
    if "ID" in test_df.columns:
        return test_df["ID"].tolist()
    if "id" in test_df.columns:
        return test_df["id"].tolist()
    if sample_submission_path.exists():
        sample_df = pd.read_csv(sample_submission_path)
        id_col = "ID" if "ID" in sample_df.columns else "id"
        if len(sample_df) == len(test_df):
            return sample_df[id_col].tolist()
        print("Warning: sample submission row count mismatch; falling back to row indices.")
    return list(range(len(test_df)))


def build_train_pool(train_data: List[Dict[str, Any]], pool_size: int) -> List[Dict[str, Any]]:
    rng = random.Random(RANDOM_SEED)
    if pool_size and pool_size < len(train_data):
        sampled = rng.sample(train_data, pool_size)
    else:
        sampled = train_data
    return [{"data": ex, "tokens": tokenize_code(ex.get("code", ""))} for ex in sampled]


def prepare_few_shot_examples(
    tokenized_train: List[Dict[str, Any]], test_tokens: set[str], count: int
) -> List[Dict[str, Any]]:
    # prioritize same-language examples and higher token overlap
    scored = []
    for entry in tokenized_train:
        sim = jaccard(test_tokens, entry["tokens"])
        lang_match = 0.05 if entry["data"].get("language") else 0.0
        scored.append((sim + lang_match, entry["data"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_candidates = [ex for _, ex in scored[: max(count * 10, count * 4)]]

    human_examples = [ex for ex in top_candidates if int(ex.get("label", 1)) == 0]
    ai_examples = [ex for ex in top_candidates if int(ex.get("label", 1)) == 1]

    selected: List[Dict[str, Any]] = []
    human_need = count // 2
    ai_need = count - human_need

    selected.extend(human_examples[:human_need])
    selected.extend(ai_examples[:ai_need])

    if len(selected) < count:
        for ex in top_candidates:
            if ex not in selected:
                selected.append(ex)
            if len(selected) >= count:
                break

    random.shuffle(selected)
    return selected[:count]


def _resolve_model_name(name: str) -> str:
    path = Path(os.path.expanduser(name))
    if path.exists():
        return str(path)
    return name


def load_model() -> tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    model_id = _resolve_model_name(MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=LOCAL_FILES_ONLY)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=LOCAL_FILES_ONLY)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_label(
    prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: torch.device
) -> int:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        padding=False,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=TEMPERATURE > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return extract_prediction(generated)


def run_inference(
    test_data: List[Dict[str, Any]],
    ids: List[Any],
    tokenized_train: List[Dict[str, Any]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    checkpoint_df: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    results: List[Dict[str, Any]] = []
    processed_ids = set(checkpoint_df["ID"].tolist()) if not checkpoint_df.empty else set()

    print(f"Starting inference on {len(test_data)} samples...")
    print(f"Already processed: {len(processed_ids)} samples")

    for idx, sample in enumerate(tqdm(test_data)):
        sample_id = ids[idx] if idx < len(ids) else idx

        if sample_id in processed_ids:
            existing_pred = checkpoint_df[checkpoint_df["ID"] == sample_id]["label"].values[0]
            results.append({"ID": sample_id, "label": int(existing_pred)})
            continue

        test_tokens = tokenize_code(sample.get("code", ""))
        few_shot_examples = prepare_few_shot_examples(tokenized_train, test_tokens, FEW_SHOT_COUNT)
        user_prompt = create_detection_prompt(sample.get("code", ""), sample.get("language", "Unknown"), few_shot_examples)
        formatted_prompt = format_prompt_for_model(user_prompt, tokenizer)

        try:
            prediction = generate_label(formatted_prompt, model, tokenizer, device)
        except Exception as exc:
            print(f"\nError generating for sample {sample_id}: {exc}")
            prediction = 1

        results.append({"ID": sample_id, "label": prediction})

        if (idx + 1) % 10 == 0:
            temp_df = pd.DataFrame(results)
            all_predictions = pd.concat([checkpoint_df, temp_df], ignore_index=True)
            save_checkpoint(all_predictions)
            print(f"\nCheckpoint saved at {idx + 1} samples")

    return pd.DataFrame(results)


def evaluate_predictions(predictions_df: pd.DataFrame, test_data: List[Dict[str, Any]], ids: List[Any]) -> float:
    test_df = pd.DataFrame(test_data).copy()
    test_df["ID"] = ids
    merged = test_df.merge(predictions_df, on="ID", suffixes=("_true", "_pred"))

    # coerce to numeric and drop anything outside {0,1} to avoid sklearn target errors
    merged["label_true"] = pd.to_numeric(merged["label_true"], errors="coerce")
    merged["label_pred"] = pd.to_numeric(merged["label_pred"], errors="coerce")
    merged = merged[merged["label_pred"].isin([0, 1]) & merged["label_true"].isin([0, 1])]

    if merged.empty:
        print("No valid rows to evaluate after filtering labels; skipping evaluation.")
        return 0.0

    y_true = merged["label_true"].values
    y_pred = merged["label_pred"].values
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Human", "AI"]))

    print("\nPer-Language Performance:")
    for lang in merged["language"].unique():
        lang_data = merged[merged["language"] == lang]
        lang_f1 = f1_score(lang_data["label_true"], lang_data["label_pred"], average="macro")
        print(f"  {lang}: Macro F1 = {lang_f1:.4f}")

    return macro_f1


def main() -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "task_A"
    train_path = data_dir / "train.parquet"
    test_path = data_dir / "test.parquet"
    sample_submission_path = Path(SAMPLE_SUBMISSION_FILE) if SAMPLE_SUBMISSION_FILE else data_dir / "sample_submission.csv"

    if not train_path.exists():
        print(f"Error: train file not found at {train_path}")
        return
    if not test_path.exists():
        print(f"Error: test file not found at {test_path}")
        return

    try:
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
    except Exception as exc:
        print(f"Error reading parquet files: {exc}")
        return

    train_data = train_df.to_dict(orient="records")
    test_data = test_df.to_dict(orient="records")

    ids = resolve_ids(test_df, sample_submission_path)
    if len(ids) != len(test_data):
        print(f"Warning: ID count {len(ids)} does not match test rows {len(test_data)}")

    tokenized_train = build_train_pool(train_data, FEW_SHOT_POOL_SIZE)
    print(f"Using {len(tokenized_train)} training rows (of {len(train_data)}) for few-shot retrieval")

    try:
        model, tokenizer, device = load_model()
        print(f"\nLoaded model {MODEL_NAME} on {device}")
    except Exception as exc:
        print(f"Error loading model '{MODEL_NAME}': {exc}")
        return

    checkpoint_df = load_checkpoint()
    predictions_df = run_inference(test_data, ids, tokenized_train, model, tokenizer, checkpoint_df, device)

    final_predictions = pd.concat([checkpoint_df, predictions_df], ignore_index=True)
    final_predictions = final_predictions.drop_duplicates(subset=["ID"], keep="last")

    final_predictions.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFinal predictions saved to {OUTPUT_FILE}")

    test_df_for_eval = pd.DataFrame(test_data)
    if "label" in test_df_for_eval.columns:
        evaluate_predictions(final_predictions, test_data, ids)
    else:
        print("\nNo ground truth labels in test set. Skipping evaluation.")

    print("\nDone!")


if __name__ == "__main__":
    main()
