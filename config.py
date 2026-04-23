from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from dataclasses import asdict, is_dataclass
from typing import Optional

@dataclass
class CoreConfig:
   
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_rank: int = 16
    max_seq_length: int = 2048
    load_in_4bit: int = 0
    model_dir: str = ""  # required via CLI
    dataset_name: str = (
        "gsm8k"  # {"gsm8k", "aime25", "math500", "olympiadbench", "amc23", "minervamath", "math", "aquarat", "livemathbench", "dapo_math"}
    )
    dataset_split: str = "train"
    test_dataset_split: str = "test"
    trainer_type: str = "grpo"  # {"grpo", "amir_grpo"}
    calibration: bool = False


@dataclass
class TrainingConfig:
   
    learning_rate: float = 5e-6
    weight_decay: float = 0.1
    max_grad_norm: float = 0.1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_steps: int = 1000
    seed: int = 0


@dataclass
class SchedulerOptimConfig:
  
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    optim: str = "adamw_8bit"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99


@dataclass
class GenerationConfig:
   

    num_generations: int = 8
    max_prompt_length: int = 1024
    max_completion_length: int = 1024


@dataclass
class AlgorithmConfig:
   
    loss_type: str = "grpo"
    epsilon: float = 0.20
    epsilon_high: float = 0.20
    mask_truncated_completions: int = 0
    scale_rewards: str = "group"
    importance_sampling_level: str = "token"


@dataclass
class DPOConfig:
   

    lambda_strong: float = 0.01
    reward_margin: float = 2.0  
    beta_strong: float = 0.2
    pair_mode: str = "all"      
    max_pairs_per_group: Optional[int] = None
    ref_free: bool = True

    alpha_rank: float = 0.5
    alpha_len: float = 0.5
    alpha_group: float = 0.5
    w_max: float = 3.5
    group_quality_threshold: float = 0.5

    weak_margin: float = 0.15
    lambda_weak: float = 0.01
    weak_warmup_steps: int = 100
    beta_weak: float = 0.05

    dpo_chunk_size: int = 8


@dataclass
class LoggingConfig:
  
    logging_steps: int = 1
    save_steps: int = 50
    report_to: str = "wandb"
    wandb_api_key: Optional[str] = None


@dataclass
class Config:
  
    core: CoreConfig
    training: TrainingConfig
    sched_optim: SchedulerOptimConfig
    generation: GenerationConfig
    algorithm: AlgorithmConfig
    dpo: DPOConfig
    logging: LoggingConfig

def _positive_int(value: str) -> int:
  
    i = int(value)
    if i <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0")
    return i


def _non_negative_float(value: str) -> float:
   
    f = float(value)
    if f < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0")
    return f


def build_parser() -> argparse.ArgumentParser:
   
    parser = argparse.ArgumentParser(description="GRPO / GRPO+DPO Training CLI")

    core = parser.add_argument_group("Core / Model & Data")
    core.add_argument(
        "--model_name",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model ID or local path.",
    )
    core.add_argument(
        "--lora_rank",
        type=_positive_int,
        default=16,
        help="LoRA rank. Use 0 to disable LoRA in downstream code if supported.",
    )
    core.add_argument(
        "--max_seq_length",
        type=_positive_int,
        default=2048,
        help="Maximum packed sequence length.",
    )
    core.add_argument(
        "--load_in_4bit",
        type=int,
        choices=[0, 1],
        default=0,
        help="Load model weights in 4-bit (1) or full precision (0).",
    )
    core.add_argument(
        "--model_dir", required=True, help="Output directory for checkpoints and logs."
    )
    core.add_argument(
        "--dataset_name",
        required=True,
        choices=[
            "gsm8k",
            "aime25",
            "math500",
            "olympiadbench",
            "amc23",
            "minervamath",
            "math",
            "aquarat",
            "livemathbench",
            "dapo_math",
        ],
        help="Dataset used for training/eval.",
    )
    core.add_argument(
        "--dataset_split",
        default="train",
        help="Split used for training. Typically 'train'.",
    )
    core.add_argument(
        "--test_dataset_split", default="test", help="Split used for evaluation."
    )
    core.add_argument(
        "--trainer_type",
        choices=["grpo", "dwcal_grpo"],
        default="grpo",
        required=False,
        help="'grpo' = vanilla GRPO; 'amir_grpo' = GRPO with DPO regularization.",
    )
    core.add_argument(
        "--calibration",
        action="store_true",
        help="Enable confidence calibration requiring <analysis> and <confidence> formats.",
    )

    # --- Training ---
    train = parser.add_argument_group("Training")
    train.add_argument(
        "--learning_rate",
        type=_non_negative_float,
        default=5e-6,
        help="Base learning rate for optimizer.",
    )
    train.add_argument(
        "--weight_decay",
        type=_non_negative_float,
        default=0.1,
        help="AdamW weight decay.",
    )
    train.add_argument(
        "--max_grad_norm",
        type=_non_negative_float,
        default=0.1,
        help="Gradient clipping value (0 disables clipping).",
    )
    train.add_argument(
        "--per_device_train_batch_size",
        type=_positive_int,
        default=1,
        help="weak-batch size per device.",
    )
    train.add_argument(
        "--gradient_accumulation_steps",
        type=_positive_int,
        default=4,
        help="Number of accumulation steps to reach the effective batch size.",
    )
    train.add_argument(
        "--max_steps",
        type=_positive_int,
        default=1000,
        help="Total number of optimizer steps.",
    )
    train.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )

    so = parser.add_argument_group("Scheduler & Optimizer")
    so.add_argument(
        "--lr_scheduler_type",
        default="cosine",
        help="LR schedule type (e.g., 'cosine', 'linear', 'constant').",
    )
    so.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (0.0 - 1.0) of total training steps.",
    )
    so.add_argument(
        "--optim",
        default="adamw_8bit",
        help="Optimizer identifier (e.g., 'adamw', 'adamw_8bit').",
    )
    so.add_argument("--adam_beta1", type=float, default=0.9, help="Adam β1.")
    so.add_argument("--adam_beta2", type=float, default=0.99, help="Adam β2.")

    gen = parser.add_argument_group("Generation (for GRPO rollouts)")
    gen.add_argument(
        "--num_generations",
        type=_positive_int,
        default=8,
        help="Number of sampled completions per prompt during rollout.",
    )
    gen.add_argument(
        "--max_prompt_length",
        type=_positive_int,
        default=1024,
        help="Maximum prompt token length for generation.",
    )
    gen.add_argument(
        "--max_completion_length",
        type=_positive_int,
        default=1024,
        help="Maximum completion token length for generation.",
    )

    algo = parser.add_argument_group("RL Algorithm")
    algo.add_argument(
        "--loss_type",
        default="grpo",
        help="Loss identifier. (e.g., 'dr_grpo', 'grpo', 'dapo')",
    )
    algo.add_argument(
        "--epsilon",
        type=float,
        default=0.20,
        help="Lower clipping factor for advantage normalization.",
    )
    algo.add_argument(
        "--epsilon_high",
        type=float,
        default=0.20,
        help="Upper clipping factor for advantage normalization.",
    )
    algo.add_argument(
        "--mask_truncated_completions",
        type=int,
        choices=[0, 1],
        default=0,
        help="Whether to exclude truncated or incomplete completions from loss and reward computation.",
    )
    algo.add_argument(
        "--scale_rewards",
        type=str,
        default="group",
        help="Specifies how to normalize or scale rewards across samples. Options: ['group', 'batch', 'none'].",
    )
    algo.add_argument(
        "--importance_sampling_level",
        type=str,
        default="token",
        help="Determines the granularity at which importance sampling is applied. Options: ['token', 'sequence'].",
    )

    dpo = parser.add_argument_group(
        "DPO Integration (only if trainer_type='amir_grpo')"
    )
    dpo.add_argument(
        "--lambda_strong",
        type=float,
        default=0.01,
        help="Interpolation weight for DPO pairwise term relative to GRPO loss.",
    )
    dpo.add_argument(
        "--reward_margin",
        type=float,
        default=2.0,
        help="Score difference threshold to form a preference pair.",
    )
    dpo.add_argument(
        "--beta_strong",
        type=float,
        default=0.2,
        help="Inverse temperature for DPO logistic transform.",
    )
    dpo.add_argument(
        "--pair_mode",
        default="all",
        help="Pair mining strategy (e.g., 'all', 'topk').",
    )
    dpo.add_argument(
        "--max_pairs_per_group",
        type=lambda s: None if s.lower() == "none" else _positive_int(s),
        default=None,
        help="Optional cap on pairs per prompt group. Use 'None' to disable.",
    )
    dpo.add_argument(
        "--ref_free",
        type=lambda x: str(x).lower() in {"true", "1", "yes"},
        default=True,
        help="Whether to use the policy as its own reference (true/false).",
    )

    dpo.add_argument(
        "--alpha_rank", type=float, default=0.5,
        help="Weight for rank-based quality in strong branch."
    )
    dpo.add_argument(
        "--alpha_len", type=float, default=0.5,
        help="Weight for length penalty in strong branch."
    )
    dpo.add_argument(
        "--alpha_group", type=float, default=0.5,
        help="Weight for group quality in strong branch."
    )
    dpo.add_argument(
        "--w_max", type=float, default=3.5,
        help="Maximum weight cap for strong branch pairs."
    )
    dpo.add_argument(
        "--group_quality_threshold", type=float, default=0.5,
        help="Threshold tau for group quality calculation."
    )

    dpo.add_argument(
        "--weak_margin", type=float, default=0.15,
        help="Reward gap threshold for micro-preference pairs."
    )
    dpo.add_argument(
        "--lambda_weak", type=float, default=0.005,
        help="Maximum interpolation weight for micro branch."
    )
    dpo.add_argument(
        "--weak_warmup_steps", type=int, default=500,
        help="Warmup steps for micro branch lambda."
    )
    dpo.add_argument(
        "--beta_weak", type=float, default=0.05,
        help="Inverse temperature for micro branch DPO loss."
    )

    dpo.add_argument(
        "--dpo_chunk_size", type=int, default=8,
        help="Chunk size for computing logps in DPO regularization to save memory."
    )

    log = parser.add_argument_group("Logging & Checkpointing")
    log.add_argument(
        "--logging_steps",
        type=_positive_int,
        default=1,
        help="Log metrics every N optimizer steps.",
    )
    log.add_argument(
        "--save_steps",
        type=_positive_int,
        default=50,
        help="Save a checkpoint every N optimizer steps.",
    )
    log.add_argument(
        "--report_to",
        default="wandb",
        help="Reporting backend, e.g., 'wandb', 'tensorboard', or 'none'.",
    )
    log.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="W&B API key for authentication. ",
    )

    return parser


def _namespace_to_config(ns: argparse.Namespace) -> Config:
 
    core = CoreConfig(
        model_name=ns.model_name,
        lora_rank=ns.lora_rank,
        max_seq_length=ns.max_seq_length,
        load_in_4bit=ns.load_in_4bit,
        model_dir=ns.model_dir,
        dataset_name=ns.dataset_name,
        dataset_split=ns.dataset_split,
        test_dataset_split=ns.test_dataset_split,
        trainer_type=ns.trainer_type,
        calibration=ns.calibration,
    )

    training = TrainingConfig(
        learning_rate=ns.learning_rate,
        weight_decay=ns.weight_decay,
        max_grad_norm=ns.max_grad_norm,
        per_device_train_batch_size=ns.per_device_train_batch_size,
        gradient_accumulation_steps=ns.gradient_accumulation_steps,
        max_steps=ns.max_steps,
        seed=ns.seed,
    )

    sched_optim = SchedulerOptimConfig(
        lr_scheduler_type=ns.lr_scheduler_type,
        warmup_ratio=ns.warmup_ratio,
        optim=ns.optim,
        adam_beta1=ns.adam_beta1,
        adam_beta2=ns.adam_beta2,
    )

    generation = GenerationConfig(
        num_generations=ns.num_generations,
        max_prompt_length=ns.max_prompt_length,
        max_completion_length=ns.max_completion_length,
    )

    algorithm = AlgorithmConfig(
        loss_type=ns.loss_type,
        epsilon=ns.epsilon,
        epsilon_high=ns.epsilon_high,
        mask_truncated_completions=ns.mask_truncated_completions,
        scale_rewards=ns.scale_rewards,
        importance_sampling_level=ns.importance_sampling_level,
    )

    dpo = DPOConfig(
        lambda_strong=ns.lambda_strong,
        reward_margin=ns.reward_margin,
        beta_strong=ns.beta_strong,
        pair_mode=ns.pair_mode, # 对应修改后的参数名
        max_pairs_per_group=ns.max_pairs_per_group,
        ref_free=ns.ref_free,
        # [NEW] Pass new DWCAL params
        alpha_rank=ns.alpha_rank,
        alpha_len=ns.alpha_len,
        alpha_group=ns.alpha_group,
        w_max=ns.w_max,
        group_quality_threshold=ns.group_quality_threshold,
        weak_margin=ns.weak_margin,
        lambda_weak=ns.lambda_weak,
        weak_warmup_steps=ns.weak_warmup_steps,
        beta_weak=ns.beta_weak,
    )

    logging = LoggingConfig(
        logging_steps=ns.logging_steps,
        save_steps=ns.save_steps,
        report_to=ns.report_to,
    )

    return Config(
        core=core,
        training=training,
        sched_optim=sched_optim,
        generation=generation,
        algorithm=algorithm,
        dpo=dpo,
        logging=logging,
    )


def parse_args(argv: Optional[list[str]] = None) -> Config:
  

    parser = build_parser()
    ns = parser.parse_args(argv)
    return _namespace_to_config(ns)


def get_parser() -> argparse.ArgumentParser:
 
    return build_parser()


def save_config_files(args: Config, out_dir: Path) -> None:
   
    payload = asdict(args) if is_dataclass(args) else args
    json_path = out_dir / "config.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
