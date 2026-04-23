from __future__ import annotations

from pathlib import Path
import wandb
from huggingface_hub import HfApi, create_repo

from config import parse_args, save_config_files
from Train.models import bf16_fp16_flags, load_ref_model, load_train_model
from Train.rewards import (
    brier_score,
    correctness_reward_func,
    expression_correctness_reward_func,
    int_reward_func,
    strict_format_reward_func,
    strict_format_reward_func_with_calib,
    xmlcount_reward_func,
)
from trl import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

# Local: hybrid trainer that adds DPO regularization
from Train.trainer import DWCAL_GRPO_Trainer

# Datasets
from Data.data import (
    get_gsm8k_questions,
    get_math500_questions,
    get_dapo_math_questions,
)

def build_grpo_config(args) -> GRPOConfig:
    bf16, fp16 = bf16_fp16_flags()
    return GRPOConfig(
        # Training
        learning_rate=args.training.learning_rate,
        weight_decay=args.training.weight_decay,
        max_grad_norm=args.training.max_grad_norm,
        per_device_train_batch_size=args.training.per_device_train_batch_size,
        gradient_accumulation_steps=args.training.gradient_accumulation_steps,
        max_steps=args.training.max_steps,
        seed=args.training.seed,
        # Scheduler & Optimizer
        lr_scheduler_type=args.sched_optim.lr_scheduler_type,
        warmup_ratio=args.sched_optim.warmup_ratio,
        optim=args.sched_optim.optim,
        adam_beta1=args.sched_optim.adam_beta1,
        adam_beta2=args.sched_optim.adam_beta2,
        # Precision
        bf16=bf16,
        fp16=fp16,
        # Generation
        num_generations=args.generation.num_generations,
        max_prompt_length=args.generation.max_prompt_length,
        max_completion_length=args.generation.max_completion_length,
        # Algorithm
        loss_type=args.algorithm.loss_type,
        epsilon=args.algorithm.epsilon,
        epsilon_high=args.algorithm.epsilon_high,
        mask_truncated_completions=bool(args.algorithm.mask_truncated_completions),
        scale_rewards=args.algorithm.scale_rewards,
        importance_sampling_level=args.algorithm.importance_sampling_level,
        # Logging / IO
        logging_steps=args.logging.logging_steps,
        save_steps=args.logging.save_steps,
        report_to=args.logging.report_to,
        run_name=f"{args.core.model_dir}/output",
        output_dir=args.core.model_dir,
    )

def main() -> None:
    args = parse_args()

    # WandB Login
    if args.logging.wandb_api_key:
        wandb.login(args.logging.wandb_api_key)

    # Output path
    out = Path(args.core.model_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    training_args = build_grpo_config(args)

    # Model loading
    model_name = args.core.model_name
    lora_rank = args.core.lora_rank
    max_seq_length = args.core.max_seq_length
    load_in_4bit = bool(args.core.load_in_4bit)

    model, tokenizer = load_train_model(
        model_name=model_name,
        lora_rank=lora_rank,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    # Dataset loading
    dataset_name = args.core.dataset_name.lower()
    dataset_split = args.core.dataset_split

    if dataset_name == "gsm8k":
        train_dataset = get_gsm8k_questions(dataset_split)
    elif dataset_name == "math":
        train_dataset = get_math500_questions(dataset_split)
    elif dataset_name == "dapo_math":
        train_dataset = get_dapo_math_questions(dataset_split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Reference model
    ref_free = args.dpo.ref_free
    ref_model = None
    if not ref_free:
        ref_model = load_ref_model(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )

    # Reward functions
    use_calibration = args.core.calibration
    if use_calibration:
        reward_funcs = [
            brier_score,
            xmlcount_reward_func,
            strict_format_reward_func_with_calib,
            int_reward_func,
            correctness_reward_func,
        ]
    else:
        reward_funcs = [
            xmlcount_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]

    # Trainer selection
    trainer_type = args.core.trainer_type
    TrainerCls = DWCAL_GRPO_Trainer if trainer_type == "dwcal_grpo" else GRPOTrainer

    trainer_kwargs = dict(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
    )

    if TrainerCls is DWCAL_GRPO_Trainer:
        trainer_kwargs.update(
            ref_model=ref_model,
            # [Existing AMIR Args] - 全部改为从 args 读取
            lambda_strong=getattr(args.dpo, 'lambda_strong', 0.01),
            reward_margin=getattr(args.dpo, 'reward_margin', 2.0),
            beta_strong=getattr(args.dpo, 'beta_strong', 0.2),
            pair_mode=getattr(args.dpo, 'pair_mode', 'all'),
            max_pairs_per_group=getattr(args.dpo, 'max_pairs_per_group', None),
            dpo_chunk_size=getattr(args.dpo, 'dpo_chunk_size', 8), # 【修正】不再硬编码
            ref_free=args.dpo.ref_free,
            
            # [DWCAL Strong Branch Args]
            alpha_rank=getattr(args.dpo, 'alpha_rank', 0.5),
            alpha_len=getattr(args.dpo, 'alpha_len', 0.5),
            alpha_group=getattr(args.dpo, 'alpha_group', 0.5),
            w_max=getattr(args.dpo, 'w_max', 3.5),
            group_quality_threshold=getattr(args.dpo, 'group_quality_threshold', 0.5),

            # [DWCAL Micro Branch Args]
            weak_margin=getattr(args.dpo, 'weak_margin', 0.15),
            lambda_weak=getattr(args.dpo, 'lambda_weak', 0.005),
            weak_warmup_steps=getattr(args.dpo, 'weak_warmup_steps', 500),
            beta_weak=getattr(args.dpo, 'beta_weak', 0.05),
        )

    trainer = TrainerCls(**trainer_kwargs)
    trainer.train()

    # Save artifacts
    model.save_pretrained(out.as_posix())
    tokenizer.save_pretrained(out.as_posix())
    model.config.save_pretrained(out.as_posix())
    save_config_files(args, out)

    # Push to Hub (可选，如果不需要推送到 HF 可注释掉)
    # 注意：请确保替换真实的 HF_TOKEN 和 USERNAME，或者在环境变量中设置
    hf_token = "HF_TOKEN" 
    user = "HF-USERNAME"
    
    if hf_token != "HF_TOKEN" and user != "HF-USERNAME":
        private = False
        repo_name = out.name
        repo_id = f"{user}/{repo_name}"

        api = HfApi(token=hf_token)
        create_repo(repo_id=repo_id, repo_type="model", private=private, token=hf_token, exist_ok=True)
        api.upload_folder(
            folder_path=out.as_posix(),
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["checkpoint-*", "checkpoint-*/**", "global_step*", "global_step*/**"],
        )
    else:
        print("[Info] Skipping HF Hub upload. Please set HF_TOKEN and HF-USERNAME in train.py if needed.")

if __name__ == "__main__":
    main()