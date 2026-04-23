from __future__ import annotations
import math
from typing import List, Optional, Sequence, Tuple, Union, Dict
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
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        return torch.float16
    return torch.float32

def _estimate_pass_at_k(correct_counts: List[int], n_samples: int, ks: Sequence[int]) -> Dict[int, float]:
    pass_at_k: Dict[int, float] = {}
    num_problems = len(correct_counts)
    if num_problems == 0: return {k: 0.0 for k in ks}
    for k in ks:
        if k > n_samples:
            pass_at_k[k] = 0.0
            continue
        total = 0.0
        denom = math.comb(n_samples, k)
        for c in correct_counts:
            if c == 0: continue
            if c >= n_samples:
                total += 1.0
                continue
            num = math.comb(n_samples - c, k)
            total += 1.0 - (num / denom)
        pass_at_k[k] = total / num_problems
    return pass_at_k

def load_model_and_tokenizer(directory_path: str, hf_token: str, *, device: Optional[torch.device] = None, load_in_4bit: bool = False, dtype: Optional[torch.dtype] = None) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    device = _get_device(device)
    dtype = dtype or _choose_dtype()
    model = AutoModelForCausalLM.from_pretrained(directory_path, load_in_4bit=load_in_4bit, dtype=dtype, device_map=device, token=hf_token)
    model = PeftModel.from_pretrained(model, directory_path, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(directory_path, token=hf_token)
    if hasattr(model.config, "use_cache"): model.config.use_cache = True
    model.eval()
    return model, tokenizer

def _apply_chat_template_if_available(tokenizer: PreTrainedTokenizerBase, prompt: Union[str, Sequence[dict]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return str(prompt)

def generate_batch(prompts: List[Union[str, Sequence[dict]]], tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel, *, device: Optional[torch.device] = None, max_new_tokens: int = 1024, do_sample: bool = True, temperature: float = 0.6, num_return_sequences: int = 8) -> Tuple[List[str], List[int], List[float]]:
    device = _get_device(device)
    input_texts = [_apply_chat_template_if_available(tokenizer, p) for p in prompts]
    enc = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            output_scores=True,
            return_dict_in_generate=True
        )
    
    input_len = enc["input_ids"].shape[1]
    gen_sequences = outputs.sequences[:, input_len:]
    stacked_scores = torch.stack(outputs.scores, dim=1)
    log_probs = torch.log_softmax(stacked_scores, dim=-1)
    target_log_probs = torch.gather(log_probs, index=gen_sequences.unsqueeze(-1), dim=-1).squeeze(-1)
    
    mean_log_probs = []
    gen_lengths = []
    for i in range(gen_sequences.shape[0]):
        mask = (gen_sequences[i] != tokenizer.pad_token_id)
        length = mask.sum().item()
        gen_lengths.append(length)
        mean_lp = target_log_probs[i, mask].mean().item() if length > 0 else 0.0
        mean_log_probs.append(mean_lp)

    decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return decoded, gen_lengths, mean_log_probs

def evaluate_model_batched(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, dataset, *, batch_size: int = 1, device: Optional[torch.device] = None, progress: bool = True, num_samples: int = 8, ks: Sequence[int] = (1, 2, 4)) -> Dict:
    device = _get_device(device)
    model.eval()
    detailed_results = {}
    per_problem_correct_counts = []

    for i, sample in enumerate(tqdm(dataset, desc="Eval") if progress else enumerate(dataset)):
        prompt = sample["prompt"]
        gt = sample["answer"]

        outputs, gen_lengths, mean_lps = generate_batch(prompts=[prompt], tokenizer=tokenizer, model=model, device=device, num_return_sequences=num_samples)

        completions_info = []
        correct_count = 0
        avg_len = sum(gen_lengths) / num_samples

        print("\n" + "#"*80)
        print(f" PROBLEM {i} ".center(80, "#"))
        print(f"QUERY: {prompt}")
        print(f"GROUND TRUTH: {gt}")
        print("-" * 80)

        for idx, (text, length, lp) in enumerate(zip(outputs, gen_lengths, mean_lps)):

            extracted_ans, reward = answer_tag_reward_fn_for_orz(text, gt, fast=False)
            is_correct = (reward == 1.0)
            if is_correct: correct_count += 1
            
            status = "✅ [CORRECT]" if is_correct else "❌ [WRONG]"
            ratio = length / avg_len if avg_len > 0 else 1.0
            
            print(f"--- Sample {idx} | {status} | Len: {length} | LogP: {lp:.4f} | Ratio: {ratio:.2f} ---")
            print(f"EXTRACTED ANSWER: {extracted_ans}")
            print(f"FULL COMPLETION:\n{text}")
            print("-" * 40)
            
            completions_info.append({
                "text": text, "is_correct": is_correct, "extracted": extracted_ans,
                "length": length, "mean_log_prob": lp, "length_ratio": ratio
            })

        print(f"RESULT SUMMARY: {correct_count}/{num_samples} Correct | Avg Group Len: {avg_len:.1f}")
        print("#"*80 + "\n")
        # ------------------------

        per_problem_correct_counts.append(correct_count)
        detailed_results[f"prob_{i}"] = {
            "question": prompt, "gt": gt, "correct_count": correct_count,
            "avg_length": avg_len, "completions": completions_info
        }

    pass_at_k = _estimate_pass_at_k(per_problem_correct_counts, num_samples, ks)
    return {"metrics": pass_at_k, "details": detailed_results}
