from __future__ import annotations
from pathlib import Path
import torch
import json
import sys
from config import parse_args
from case_analysize.eval_utils import evaluate_model_batched, load_model_and_tokenizer
from Data.data import get_amc23_questions, get_aime25_questions, get_gsm8k_questions 

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Run Mode] Single GPU: {device}")

    model_dir = Path(args.core.model_dir).expanduser().as_posix()
    model, tokenizer = load_model_and_tokenizer(
        directory_path=model_dir,
        hf_token="HF_TOKEN", 
        device=device,
        load_in_4bit=bool(args.core.load_in_4bit)
    )

    ds_name = args.core.dataset_name.lower()
    if ds_name == "amc23":
        test_dataset = get_amc23_questions(args.core.test_dataset_split)
    elif ds_name == "aime25":
        test_dataset = get_aime25_questions(args.core.test_dataset_split)
    else:
        test_dataset = get_gsm8k_questions(args.core.test_dataset_split)

    print(f"Starting detailed analysis on {len(test_dataset)} problems...")

    results = evaluate_model_batched(
        model=model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        device=device,
        num_samples=8
    )

    output_path = f"case_analysis_DWCAL_{ds_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\nEvaluation Complete. Metrics: {results['metrics']}")
    print(f"Detailed data saved to {output_path}")

if __name__ == "__main__":
    main()
