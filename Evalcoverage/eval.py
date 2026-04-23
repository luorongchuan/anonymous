from __future__ import annotations
from pathlib import Path
import torch
import os
import torch.distributed as dist
from torch.utils.data import Subset
import json
import argparse

from Evalcoverage.eval_utils import evaluate_model_batched, load_model_and_tokenizer
from Data.data import (
    get_gsm8k_questions,
    get_aime25_questions,
    get_math500_questions,
    get_olympiadbench_questions,
    get_amc23_questions,
    get_minervamath_questions,
    get_aquarat_questions,
    get_livemathbench_questions,
)

def parse_args_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="模型路径")
    parser.add_argument("--dataset_name", type=str, required=True, help="数据集名称")
    parser.add_argument("--test_dataset_split", type=str, default="test")
    parser.add_argument("--load_in_4bit", type=int, default=0)
    parser.add_argument("--num_rollouts", type=int, default=16)
    return parser.parse_args()

def main():
    args = parse_args_cli()

    rank = 0
    world_size = 1
    local_rank = 0
    is_main_process = True
    device = None

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        try:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            is_main_process = (rank == 0)
            if is_main_process:
                print(f"[Distributed] Initialized with {world_size} GPUs. Local Rank: {local_rank}")
        except Exception as e:
            print(f"[Rank {rank}] Failed to init process group: {e}")
            raise
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if is_main_process:
            print(f"[Single GPU] Running on {device}")

    if is_main_process:
        print(f"Loading dataset: {args.dataset_name} ...")

    dataset_name = args.dataset_name.lower()
    split = args.test_dataset_split

    if dataset_name == "gsm8k":
        test_dataset = get_gsm8k_questions(split)
    elif dataset_name == "aime25":
        test_dataset = get_aime25_questions(split)
    elif dataset_name == "math500":
        test_dataset = get_math500_questions(split)
    elif dataset_name == "olympiadbench":
        test_dataset = get_olympiadbench_questions()
    elif dataset_name == "amc23":
        test_dataset = get_amc23_questions(split)
    elif dataset_name == "aquarat":
        test_dataset = get_aquarat_questions(split)
    elif dataset_name == "minervamath":
        test_dataset = get_minervamath_questions(split)
    elif dataset_name == "livemathbench":
        test_dataset = get_livemathbench_questions(split)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    num_samples = len(test_dataset)
    if world_size > 1:
        samples_per_gpu = (num_samples + world_size - 1) // world_size
        start_idx = rank * samples_per_gpu
        end_idx = min(start_idx + samples_per_gpu, num_samples)
        if start_idx >= num_samples:
            test_dataset = Subset(test_dataset, [])
            if is_main_process:
                print(f"[Rank {rank}] No data to process")
        else:
            test_dataset = Subset(test_dataset, range(start_idx, end_idx))
    else:
        if is_main_process:
            print(f"[Single GPU] Processing all {num_samples} samples")

    if is_main_process:
        print(f"Loading model from {args.model_dir} ...")
    model, tokenizer = load_model_and_tokenizer(
        directory_path=args.model_dir,
        hf_token=None,
        device=device,
        load_in_4bit=bool(args.load_in_4bit)
    )
    model.eval()

    num_rollouts = 16
    correct_counts = evaluate_model_batched(
        model=model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        batch_size=8,
        device=device,
        progress=is_main_process,
        num_samples=num_rollouts
    )

    if is_main_process:
        out_file = f"correct_counts.json"
        with open(out_file, "w") as f:
            json.dump(correct_counts, f)
        print(f"correct_counts saved to {out_file}")

    solvable = [c > 0 for c in correct_counts]
    if is_main_process:
        print("Solvable dict prepared. Total solvable:", sum(solvable))

    if world_size > 1:
        dist.destroy_process_group()
        if is_main_process:
            print("[Rank 0] Process group destroyed.")

if __name__ == "__main__":
    main()
