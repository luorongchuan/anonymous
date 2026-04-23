from __future__ import annotations
from pathlib import Path
import torch
import os
import torch.distributed as dist
from torch.utils.data import Subset

from config import parse_args
from Evalmargin.eval_utils import evaluate_model_batched, load_model_and_tokenizer
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

def main() -> None:
    args = parse_args()

    # ----------------------------
    # 1. 初始化分布式环境
    # ----------------------------
    rank = 0
    world_size = 1
    local_rank = 0
    is_main_process = True
    device = None

    # 检测是否由 torchrun/accelerate 启动
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
        # 单卡模式 fallback
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if is_main_process:
            print(f"[Single GPU] Running on {device}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.core.model_dir).expanduser()
    out = out_dir.as_posix()
    hf_token = "HF_TOKEN"

    # ----------------------------
    # 2. Dataset loading
    # ----------------------------
    if is_main_process:
        print(f"Loading dataset: {args.core.dataset_name} ...")
    
    if args.core.dataset_name.lower() == "gsm8k":
        test_dataset = get_gsm8k_questions(args.core.test_dataset_split)
    elif args.core.dataset_name.lower() == "aime25":
        test_dataset = get_aime25_questions(args.core.test_dataset_split)
    elif args.core.dataset_name.lower() == "math500":
        test_dataset = get_math500_questions(args.core.test_dataset_split)
    elif args.core.dataset_name.lower() == "olympiadbench":
        test_dataset = get_olympiadbench_questions()
    elif args.core.dataset_name.lower() == "amc23":
        test_dataset = get_amc23_questions(args.core.test_dataset_split)
    elif args.core.dataset_name.lower() == "aquarat":
        test_dataset = get_aquarat_questions(args.core.test_dataset_split)
    elif args.core.dataset_name.lower() == "minervamath":
        test_dataset = get_minervamath_questions(args.core.test_dataset_split)
    elif args.core.dataset_name.lower() == "livemathbench":
        test_dataset = get_livemathbench_questions(args.core.test_dataset_split)
    else:
        raise ValueError(f"Unknown dataset: {args.core.dataset_name}")

    # ----------------------------
    # 3. 数据分片 (核心逻辑)
    # ----------------------------
    num_samples = len(test_dataset)
    
    if world_size > 1:
        # 向上取整，确保所有数据都被覆盖
        samples_per_gpu = (num_samples + world_size - 1) // world_size
        start_idx = rank * samples_per_gpu
        end_idx = min(start_idx + samples_per_gpu, num_samples)
        
        # 如果当前 rank 超出了数据范围（例如数据少于卡数），则设为空
        if start_idx >= num_samples:
            test_dataset = Subset(test_dataset, [])
            if is_main_process:
                print(f"[Rank {rank}] No data to process (start_idx {start_idx} >= total {num_samples})")
        else:
            test_dataset = Subset(test_dataset, range(start_idx, end_idx))
            if is_main_process and rank == 0:
                print(f"[Rank 0] Total: {num_samples}, Split: {samples_per_gpu}/GPU")
                print(f"[Rank 0] This rank processing: [{start_idx}:{end_idx}] ({len(test_dataset)} samples)")
    else:
        if is_main_process:
            print(f"[Single GPU] Processing all {num_samples} samples.")

    # ----------------------------
    # 4. Model loading
    # ----------------------------
    # 注意：这里使用的是上面正确分配的 device (包含 local_rank)
    # 绝对不能再重新定义 device = torch.device("cuda"...)
    if is_main_process:
        print(f"[Rank {rank}] Loading model to {device} ...")
        
    model, tokenizer = load_model_and_tokenizer(
        directory_path=out,
        device=device,
        load_in_4bit=bool(args.core.load_in_4bit),
        hf_token=hf_token,
    )
    model.eval()

    # ----------------------------
    # 5. Evaluation
    # ----------------------------
    # 构造唯一的输出后缀，防止文件冲突
    # 注意：你需要去 Eval/eval_utils.py 里修改函数签名以接收这个参数，
    # 或者在该函数内部自动检测 rank。
    output_suffix = f"_rank{rank}" if world_size > 1 else ""
    
    if is_main_process:
        print(f"[Rank {rank}] Starting evaluation...")

    try:
        evaluate_model_batched(
            model=model,
            tokenizer=tokenizer,
            dataset=test_dataset,
            batch_size=1, 
            device=device,
            progress=is_main_process, # 只有主进程显示进度条
            # 如果 eval_utils 还没改，这行可能会报错，请先注释掉，见下方说明
            # result_suffix=output_suffix 
        )
    except Exception as e:
        print(f"[Rank {rank}] Evaluation failed: {e}")
        raise
    finally:
        # ----------------------------
        # 6. 清理
        # ----------------------------
        if world_size > 1:
            dist.destroy_process_group()
            if is_main_process:
                print("[Rank 0] Process group destroyed.")

if __name__ == "__main__":
    main()