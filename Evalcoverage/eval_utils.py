from __future__ import annotations

import math
import torch.distributed as dist 
from typing import List, Optional, Sequence, Tuple, Union, Dict
import os
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from peft import PeftModel

from Data.math_grader import answer_tag_reward_fn_for_orz

def _get_device(preferred: Optional[torch.device] = None) -> torch.device:
   
    if preferred is not None:
        return preferred
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _choose_dtype() -> torch.dtype:
   
    if torch.cuda.is_available():
        # bf16 if the hardware supports it; otherwise fp16
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        return torch.float16
    return torch.float32

def _estimate_pass_at_k(
    correct_counts: List[int],
    n_samples: int,
    ks: Sequence[int],
) -> Dict[int, float]:
    
    pass_at_k: Dict[int, float] = {}

    num_problems = len(correct_counts)
    if num_problems == 0:
        return {k: 0.0 for k in ks}

    for k in ks:
        if k > n_samples:
            # Not defined (can't choose k unique programs out of < k samples)
            pass_at_k[k] = 0.0
            continue

        total = 0.0
        denom = math.comb(n_samples, k)
        for c in correct_counts:
            if c == 0:
                # No correct samples for this problem → contributes 0
                continue
            if c >= n_samples:
                # All samples correct → contributes 1
                total += 1.0
                continue

            num = math.comb(n_samples - c, k)
            total += 1.0 - (num / denom)

        pass_at_k[k] = total / num_problems

    return pass_at_k

def load_model_and_tokenizer(
    directory_path: str,
    hf_token: str,
    *,
    device: Optional[torch.device] = None,
    load_in_4bit: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
   
    device = _get_device(device)
    dtype = dtype or _choose_dtype()

    model = AutoModelForCausalLM.from_pretrained(
        directory_path,
        load_in_4bit=load_in_4bit,
        dtype=dtype,
        device_map=device,
        token=hf_token,
    )


    if os.path.exists(os.path.join(directory_path, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, directory_path, token=hf_token)
    else:
        print(f"Warning: {directory_path} is not a PEFT adapter directory (missing adapter_config.json). Loading base model only.")
    tokenizer = AutoTokenizer.from_pretrained(directory_path, token=hf_token)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    model.eval()
    return model, tokenizer


def _apply_chat_template_if_available(
    tokenizer: PreTrainedTokenizerBase, prompt: Union[str, Sequence[dict]]
) -> str:
  
    if hasattr(tokenizer, "apply_chat_template") and not isinstance(prompt, str):
        # Expecting a list of chat messages (OpenAI-style dicts)
        return tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
    elif hasattr(tokenizer, "apply_chat_template") and isinstance(prompt, str):
        # Some tokenizers can still accept raw strings; keep your original behavior
        return tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
    else:
        return str(prompt)


def generate_batch(
    prompts: List[Union[str, Sequence[dict]]],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    *,
    device: Optional[torch.device] = None,
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_p: float = 1.0,
    num_return_sequences: int = 2,
) -> List[str]:
   
    device = _get_device(device)

    # Apply chat template on CPU
    input_texts = [_apply_chat_template_if_available(tokenizer, p) for p in prompts]

    # Tokenize on CPU
    enc = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
        )

    del enc

    # Move outputs back to CPU ASAP
    outputs = outputs.to("cpu")

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    del outputs

    return decoded

def evaluate_model_batched(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset,
    *,
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    device: Optional[torch.device] = None,
    progress: bool = True,
    num_samples: int = 16,
    ks: Sequence[int] = (1, 2, 4),
) -> Dict[int, float]:
    device = _get_device(device)

    if next(model.parameters()).device.type != device.type:
        model.to(device)
    model.eval()

    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1

    buf_prompts: List[Union[str, Sequence[dict]]] = []
    buf_gts: List[Union[str, float, int, list]] = []
    per_problem_correct_counts: List[int] = []

    iterator = enumerate(dataset)
    if progress and rank == 0:
        try:
            n = len(dataset)
        except TypeError:
            n = None
        iterator = enumerate(tqdm(dataset, total=n, desc=f"Rank {rank} Eval"))
    elif progress:

        iterator = enumerate(tqdm(dataset, desc=f"Rank {rank} Eval"))

    for i, sample in iterator:
        if max_samples is not None and i >= max_samples:
            break

        buf_prompts.append(sample["prompt"])
        buf_gts.append(sample["answer"])
        flush = (
            len(buf_prompts) == batch_size
            or (max_samples is not None and i + 1 == max_samples)
            or (
                len(buf_prompts)
                and (i + 1 == len(dataset) if hasattr(dataset, "__len__") else False)
            )
        )
        if not flush:
            continue

        outputs = generate_batch(
            prompts=buf_prompts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            num_return_sequences=num_samples,
            do_sample=True,
            temperature=0.6,
        )

        num_prompts = len(buf_prompts)

        for k_idx in range(num_prompts):
            gt = buf_gts[k_idx]
            start_index = k_idx * num_samples
            end_index = start_index + num_samples
            ensemble_outputs = outputs[start_index:end_index]

            correct_count = 0
            for pred_text in ensemble_outputs:
                _, reward = answer_tag_reward_fn_for_orz(pred_text, gt, fast=False)
                if reward == 1.0:
                    correct_count += 1

            per_problem_correct_counts.append(correct_count)

        buf_prompts.clear()
        buf_gts.clear()
        del outputs

    all_correct_counts: List[List[int]] = [None] * world_size if is_distributed else [per_problem_correct_counts]
    
    if is_distributed:

        dist.gather_object(
            per_problem_correct_counts, 
            all_correct_counts if dist.get_rank() == 0 else None, 
            dst=0
        )

        dist.barrier()

    if rank == 0:
        if is_distributed:
            final_counts = [c for sublist in all_correct_counts for c in sublist]
            print(f"[Rank 0] Gathered results from {world_size} GPUs. Total problems: {len(final_counts)}")
        else:
            final_counts = per_problem_correct_counts
            print(f"[Single GPU] Total problems: {len(final_counts)}")

        pass_at_k = _estimate_pass_at_k(
            correct_counts=final_counts,
            n_samples=num_samples,
            ks=ks,
        )

        num_problems = len(final_counts)
        print("-" * 40)
        print(f"Final Evaluation Results on {num_problems} problems ({num_samples} samples each):")
        for k in ks:
            print(f"pass@{k}: {pass_at_k[k] * 100:.2f}%")
        print("-" * 40)
     
        return final_counts
    else:

        if is_distributed:
            dist.barrier()
        return {}
