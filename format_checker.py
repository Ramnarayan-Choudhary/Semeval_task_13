import argparse
import os
import json
import pandas as pd

def check_format(pred_path, task, main_path):
    if not pred_path.endswith(".csv"):
        raise ValueError("Submission must be a .csv file.")

    df = pd.read_csv(pred_path)

    if "ID" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'id' and 'label' columns.")

    label_path = os.path.join(main_path, os.path.join(f"task_{task.upper()}", "id_to_label.json"))
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Could not find label mapping at {label_path}")
    
    with open(label_path) as f:
        valid_labels = set(json.load(f).keys())

    invalid = set(map(str, df["label"])) - valid_labels
    if invalid:
        raise ValueError(f"Invalid label IDs found: {invalid}")
    
    print("✅ Format check passed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to the predictions .csv file")
    parser.add_argument("--task", required=True, help="Task letter: A, B, or C")
    parser.add_argument("--main_path", required=False, help="Path to the root of the project", default='.')

    args = parser.parse_args()

    check_format(args.predictions, args.task, args.main_path)
