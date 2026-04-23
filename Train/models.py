from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported

TARGET_MODULES: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

__all__ = [
    "TARGET_MODULES",
    "load_train_model",
    "load_ref_model",
    "bf16_fp16_flags",
]


def load_train_model(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    lora_rank: int = 16,
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
    *,
    target_modules: Sequence[str] | None = None,
    seed: int = 0,
):
    
    tm = list(target_modules) if target_modules is not None else TARGET_MODULES

    # Choose dtype: prefer bf16 when available
    bf16, use_fp16 = bf16_fp16_flags()
    dtype = torch.bfloat16 if bf16 else torch.float16

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        max_lora_rank=lora_rank,
        dtype=dtype,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=tm,
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    setattr(model, "is_loaded_in_4bit", False)
    setattr(model, "is_loaded_in_8bit", False)

    model.config.use_cache = False
    return model, tokenizer


def load_ref_model(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
):
    

    bf16, _ = bf16_fp16_flags()
    dtype = torch.bfloat16 if bf16 else torch.float16

    ref_model, _ = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=dtype,
        max_lora_rank=0,  # ensure NO LoRA heads
    )
    ref_model.config.use_cache = True
    ref_model.eval()
    return ref_model


def bf16_fp16_flags() -> Tuple[bool, bool]:

    bf16 = is_bfloat16_supported()
    return bf16, (not bf16)
