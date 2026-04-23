from __future__ import annotations

import math
import torch.distributed as dist 
from typing import List, Optional, Sequence, Tuple, Union, Dict
import numpy as np
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

    model = PeftModel.from_pretrained(model, directory_path, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(directory_path, token=hf_token)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    model.eval()
    return model, tokenizer

def _apply_chat_template_if_available(
    tokenizer: PreTrainedTokenizerBase, prompt: Union[str, Sequence[dict]]
) -> str:
  
    if hasattr(tokenizer, "apply_chat_template") and not isinstance(prompt, str):

        return tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
    elif hasattr(tokenizer, "apply_chat_template") and isinstance(prompt, str):

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

    input_texts = [_apply_chat_template_if_available(tokenizer, p) for p in prompts]

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
    outputs = outputs.to("cpu")
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    del outputs
    return decoded

def evaluate_model_batched(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset,
    *,
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    device: Optional[torch.device] = None,
    progress: bool = True,
    num_samples: int = 8,
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
    per_problem_margins: List[float] = []  

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
            temperature=0.2,
        )

        num_prompts = len(buf_prompts)
        
        batch_trajectories = [] 

        for k_idx in range(num_prompts):
            gt = buf_gts[k_idx]
            start_index = k_idx * num_samples
            end_index = start_index + num_samples
            ensemble_outputs = outputs[start_index:end_index]

            correct_count = 0
            
            for pred_text in ensemble_outputs:
                _, reward = answer_tag_reward_fn_for_orz(pred_text, gt, fast=False)
                is_correct = (reward == 1.0)
                
                if is_correct:
                    correct_count += 1
                
                batch_trajectories.append({
                    "text": pred_text,
                    "is_correct": is_correct,
                    "problem_idx": k_idx
                })

            per_problem_correct_counts.append(correct_count)

        if batch_trajectories:
            all_texts = [t["text"] for t in batch_trajectories]

            enc = tokenizer(
                all_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            with torch.inference_mode():
                logits = model(**enc).logits

            shift_logits = logits[:, :-1, :]
            shift_labels = enc.input_ids[:, 1:]

            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            shift_mask = (shift_labels != pad_token_id).float()

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            
            token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            
            token_log_probs = token_log_probs * shift_mask
            
            sum_log_probs = token_log_probs.sum(dim=1)  # [batch]
            lengths = shift_mask.sum(dim=1).clamp(min=1) # [batch], 防止除以0
            ln_logprobs = (sum_log_probs / lengths).cpu().tolist() # 转回 CPU list


            problem_correct_lps = {idx: [] for idx in range(num_prompts)}
            problem_wrong_lps = {idx: [] for idx in range(num_prompts)}

            for idx, ln_lp in enumerate(ln_logprobs):
                traj_info = batch_trajectories[idx]
                p_idx = traj_info["problem_idx"]
                
                if traj_info["is_correct"]:
                    problem_correct_lps[p_idx].append(ln_lp)
                else:
                    problem_wrong_lps[p_idx].append(ln_lp)

            for k_idx in range(num_prompts):
                correct_lps = problem_correct_lps[k_idx]
                wrong_lps = problem_wrong_lps[k_idx]
                
                margin = 0.0

                if len(correct_lps) > 0 and len(wrong_lps) > 0:
                    avg_correct = np.mean(correct_lps)
                    avg_wrong = np.mean(wrong_lps)
                    margin = float(avg_correct - avg_wrong)
                    per_problem_margins.append(margin)
                else:

                    per_problem_margins.append(float('nan'))

        # Clear buffers
        buf_prompts.clear()
        buf_gts.clear()
        del outputs
        del batch_trajectories
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    all_correct_counts: List[List[int]] = [None] * world_size if is_distributed else [per_problem_correct_counts]
    all_margins: List[List[float]] = [None] * world_size if is_distributed else [per_problem_margins]
    
    if is_distributed:
        # Gather correct counts
        dist.gather_object(
            per_problem_correct_counts, 
            all_correct_counts if dist.get_rank() == 0 else None, 
            dst=0
        )
        # Gather margins
        dist.gather_object(
            per_problem_margins, 
            all_margins if dist.get_rank() == 0 else None, 
            dst=0
        )
        
        dist.barrier()

    if rank == 0:
        if is_distributed:
            final_counts = [c for sublist in all_correct_counts for c in sublist]
            final_margins = [m for sublist in all_margins for m in sublist]
            print(f"[Rank 0] Gathered results from {world_size} GPUs. Total problems: {len(final_counts)}")
        else:
            final_counts = per_problem_correct_counts
            final_margins = per_problem_margins
            print(f"[Single GPU] Total problems: {len(final_counts)}")

        print("\n" + "="*40)
        print("=== Preference Margin Results ===")
        print("="*40)

        print(f"Final Margins: {final_margins}")
        valid_margins = [m for m in final_margins if not math.isnan(m)]
        print(f"Valid Margins (non-NaN): {valid_margins}")

        all_correct_count = sum(1 for m in final_margins if math.isnan(m) and final_counts[final_margins.index(m)] == num_samples)

        stats_all_correct = 0
        stats_all_wrong = 0
        stats_mixed = 0
        
        for i, count in enumerate(final_counts):
            if count == num_samples:
                stats_all_correct += 1
            elif count == 0:
                stats_all_wrong += 1
            else:
                stats_mixed += 1

        print(f"\n[Diagnosis] Total Problems: {len(final_counts)}")
        print(f"  - All Correct (8/8): {stats_all_correct} problems -> Margin undefined (no wrong samples)")
        print(f"  - All Wrong   (0/8): {stats_all_wrong} problems -> Margin undefined (no correct samples)")
        print(f"  - Mixed Results:     {stats_mixed} problems -> Margin computable")

        if len(valid_margins) > 0:
            print(f"\n>>> Computing margins for {len(valid_margins)} mixed problems...")
            
            print("\n--- Margin per problem (First 20 shown) ---")

            display_limit = min(40, len(valid_margins))
            for i in range(display_limit):
                m = valid_margins[i]
                print(f"Problem #{i}: margin = {m:.4f}")
            
            if len(valid_margins) > 20:
                print(f"... and {len(valid_margins) - 20} more.")
           
            print(f"\n>>> Average Preference Margin: {np.mean(valid_margins):.4f}")
            print(f">>> Std Dev of Margin: {np.std(valid_margins):.4f}")
            
        else:
            print("\n❌ No valid margins could be computed!")
            print("   Reason: Every single problem was either 100% correct or 0% correct.")
            print("   This often happens with small datasets or extreme model performance.")

            print("\n[Debug] Detailed breakdown of first 5 problems:")
            for i in range(min(5, len(final_counts))):
                c = final_counts[i]
                status = "ALL RIGHT" if c == num_samples else ("ALL WRONG" if c == 0 else "MIXED")
                print(f"  Problem {i}: {c}/{num_samples} correct -> [{status}]")
                
            print("\n💡 Suggestion:")
            print("   1. If 'All Wrong' is high: The model might be failing to follow format (check grader).")
            print("   2. If 'All Right' is high: The dataset is too easy, or the grader is too loose.")
            print("   3. Try increasing --temperature to 1.0 or 1.2 to force more diversity/errors.")

        print("\n" + "="*40)
        print("=== Pass@k Results ===")
        print("="*40)
        
        pass_at_k = _estimate_pass_at_k(
            correct_counts=final_counts,
            n_samples=num_samples,
            ks=ks,
        )

        num_problems = len(final_counts)
        print(f"Total Evaluation Problems: {num_problems}")
        for k in ks:
            print(f"pass@{k}: {pass_at_k[k] * 100:.2f}%")
        print("-" * 40)
