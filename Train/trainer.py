from __future__ import annotations

import torch
import torch.nn.functional as F
from contextlib import nullcontext
from typing import List, Tuple, Optional, Dict, Any
from trl.trainer.grpo_trainer import GRPOTrainer
from torch.cuda.amp import autocast
import math

class DWCAL_GRPO_Trainer(GRPOTrainer):
    def __init__(
        self,
        *args,
        lambda_strong: float = 0.01,
        reward_margin: float = 2.0, 
        pair_mode: str = "all",
        max_pairs_per_group: Optional[int] = None,
        beta_strong: float = 0.2,
        ref_free: bool = True,
        dpo_chunk_size: int = 8,

        alpha_rank: float = 0.5,
        alpha_len: float = 0.5,
        alpha_group: float = 0.5,
        w_max: float = 2,
        group_quality_threshold: float = 0.5,

        weak_margin: float = 0.15,
        lambda_weak: float = 0.01,
        weak_warmup_steps: int = 100,
        beta_weak: float = 0.05,
        
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        
        self.lambda_strong = float(lambda_strong)
        self.reward_margin = float(reward_margin)
        self.pair_mode = pair_mode
        self.max_pairs_per_group = max_pairs_per_group
        self.beta_strong = float(beta_strong)
        self.ref_free = bool(ref_free)
        self.dpo_chunk_size = int(dpo_chunk_size)
        
        # Store DWCAL Strong Branch params
        self.alpha_rank = alpha_rank
        self.alpha_len = alpha_len
        self.alpha_group = alpha_group
        self.w_max = w_max
        self.group_quality_threshold = group_quality_threshold

        self.weak_margin = float(weak_margin)
        self.lambda_weak_initial = float(lambda_weak)  
        self.lambda_weak = self.lambda_weak_initial    
        self.lambda_weak_min = 0.001                            
        self.weak_warmup_steps = int(weak_warmup_steps)
        self.beta_weak = float(beta_weak)

        self._dpo_cache = None
        self.global_step = 0
        self.current_lambda_weak = 0.0

        if not hasattr(self, "_peft_has_been_casted_to_bf16"):
            self._peft_has_been_casted_to_bf16 = False

    def _amp_ctx(self):
        return ( 
            autocast(self.accelerator.device.type)
            if self._peft_has_been_casted_to_bf16
            else nullcontext()
        )

    def _build_contrastive_pairs(self, rewards: torch.Tensor, group_size: int) -> List[Dict[str, Any]]:
        if rewards.numel() == 0 or group_size <= 1:
            return [] 

        B = rewards.numel() // group_size
        pairs: List[Dict[str, Any]] = []
        rewards_np = rewards.cpu().numpy()

        for b in range(B):
            start = b * group_size
            r_group = rewards_np[start : start + group_size]
            order = sorted(range(group_size), key=lambda k: r_group[k], reverse=True)
            
            strong_count = 0
            micro_count = 0
            candidates = []

            if self.pair_mode == "topk":
                top_local = order[0]
                top_val = r_group[top_local]
                for o in order[1:]:
                    diff = top_val - r_group[o]
                    if diff > 1e-6:
                        candidates.append((diff, top_local, o))
            else:
                for ii in range(group_size):
                    i_local = order[ii]
                    for jj in range(ii + 1, group_size):
                        j_local = order[jj]
                        diff = r_group[i_local] - r_group[j_local]
                        if diff > 1e-6:
                            candidates.append([diff, i_local, j_local])
            
            for diff, i_local, j_local in candidates:
                p_type = None
                if diff > self.reward_margin:
                    if self.max_pairs_per_group is None or strong_count < self.max_pairs_per_group:
                        p_type = 'strong'
                        strong_count += 1
                else: 
                    if self.max_pairs_per_group is None or micro_count < self.max_pairs_per_group:
                        p_type = 'micro'
                        micro_count += 1
                
                if p_type:
                    pairs.append({
                        'chosen_idx': start + i_local,
                        'rejected_idx': start + j_local,
                        'type': p_type,
                        'gap': diff,
                        'r_chosen': r_group[i_local],
                        'r_rejected': r_group[j_local]
                    })
        return pairs

    @staticmethod
    def _make_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_lens: torch.Tensor, label_pad: int = -100) -> torch.Tensor:
        N, T = input_ids.shape
        labels = input_ids.clone()
        arange = torch.arange(T, device=input_ids.device).unsqueeze(0)
        is_prompt = arange < prompt_lens.unsqueeze(1)
        labels[is_prompt] = label_pad
        labels[attention_mask == 0] = label_pad
        return labels

    @staticmethod
    def _fallback_get_batch_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n, t, v = logits.shape
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
        logp = logits.log_softmax(-1)
        gather_mask = labels.ne(-100)
        safe_labels = labels.masked_fill(~gather_mask, 0)
        tok_logps = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        tok_logps = tok_logps * gather_mask.to(tok_logps.dtype)
        return tok_logps.sum(dim=1)

    def _get_dpo_logps(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self._fallback_get_batch_logps(logits, labels)

    def _compute_seq_logps_microbatch(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, chunk_size: int, requires_grad: bool = True) -> torch.Tensor:
        device = input_ids.device
        N = input_ids.size(0)
        seq_logps = torch.empty(N, device=device, dtype=torch.float32)
        outer_ctx = nullcontext if requires_grad else torch.no_grad

        with outer_ctx():
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                ids_chunk = input_ids[start:end]
                mask_chunk = attention_mask[start:end]
                labels_chunk = labels[start:end]

                with self.accelerator.autocast():
                    out_chunk = model(input_ids=ids_chunk, attention_mask=mask_chunk)

                logps_chunk = self._get_dpo_logps(out_chunk.logits, labels_chunk)
                comp_lens_chunk = (labels_chunk != -100).sum(dim=1).clamp_min(1)
                logps_chunk = logps_chunk / comp_lens_chunk
                seq_logps[start:end] = logps_chunk

                del out_chunk, logps_chunk, comp_lens_chunk

        return seq_logps

    def _generate_and_score_completions(self, inputs):
        base_out = super()._generate_and_score_completions(inputs)
        device = self.accelerator.device

        prompt_ids = base_out["prompt_ids"]
        prompt_mask = base_out["prompt_mask"]
        completion_ids = base_out["completion_ids"]
        completion_mask = base_out["completion_mask"]

        seq_len = min(prompt_ids.size(1) + completion_ids.size(1), prompt_mask.size(1) + completion_mask.size(1))

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)[:, :seq_len]
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(device)[:, :seq_len]
        prompt_lens = torch.min(prompt_mask.sum(dim=1), torch.tensor(seq_len, device=device))

        prompt_ids_cpu = prompt_ids.detach().cpu()
        completion_ids_cpu = completion_ids.detach().cpu()
        prompts_text = self.processing_class.batch_decode(prompt_ids_cpu, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids_cpu, skip_special_tokens=True)
        prompts = [x["prompt"] for x in inputs]

        if isinstance(prompts[0], list):
            completions = []
            for prompt, completion in zip(prompts, completions_text, strict=True):
                last_msg = prompt[-1] if (isinstance(prompt, list) and len(prompt) > 0) else None
                bootstrap = ""
                if last_msg and last_msg.get("role") == "assistant":
                    bootstrap = last_msg.get("content", "")
                    if isinstance(bootstrap, list):
                        bootstrap = "".join(part.get("text", "") for part in bootstrap if part.get("type") == "text")
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, [ids.tolist() for ids in completion_ids])
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        self._dpo_cache = {
            "rewards": rewards.detach().cpu(),
            "prompt_completion_ids": prompt_completion_ids.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu(),
            "prompt_lens": prompt_lens.detach().cpu(),
            "group_size": int(self.num_generations),
            "completion_lengths": completion_mask.sum(dim=1).detach().cpu(),
        }
        
        del rewards, rewards_per_func, prompts_text, completions_text, prompts, completions
        del prompt_completion_ids, attention_mask, prompt_lens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return base_out

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self.global_step += 1
        
        grpo_loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)

        if (self.lambda_strong == 0.0 and self.lambda_weak == 0.0) or self._dpo_cache is None:
            return grpo_loss

        ctx = self._dpo_cache
        dev = model.device
        
        rewards = ctx["rewards"].to(dev, non_blocking=True)
        input_ids_all = ctx["prompt_completion_ids"].to(dev, non_blocking=True)
        attention_mask_all = ctx["attention_mask"].to(dev, non_blocking=True)
        prompt_lens_all = ctx["prompt_lens"].to(dev, non_blocking=True)
        comp_lengths_all = ctx["completion_lengths"].to(dev, non_blocking=True)
        G = int(ctx["group_size"])

        pairs = self._build_contrastive_pairs(rewards, G)
        if not pairs:
            self._dpo_cache = None
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return grpo_loss

        strong_pairs = [p for p in pairs if p['type'] == 'strong']
        micro_pairs = [p for p in pairs if p['type'] == 'micro']

        unique_indices = set()
        for p in strong_pairs + micro_pairs:
            unique_indices.add(p['chosen_idx'])
            unique_indices.add(p['rejected_idx'])
        
        if not unique_indices:
            return grpo_loss
            
        sorted_indices = sorted(list(unique_indices))
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
        
        unique_input_ids = input_ids_all[sorted_indices]
        unique_attention_mask = attention_mask_all[sorted_indices]
        unique_prompt_lens = prompt_lens_all[sorted_indices]
        unique_labels = self._make_labels(unique_input_ids, unique_attention_mask, unique_prompt_lens)
        
        seq_logps_unique = self._compute_seq_logps_microbatch(
            model, unique_input_ids, unique_attention_mask, unique_labels, 
            self.dpo_chunk_size, requires_grad=True
        )
        
        def get_pair_logps(pair_list):
            if not pair_list:
                return None, None
            chosen_logps = []
            reject_logps = []
            for p in pair_list:
                c_idx = idx_map[p['chosen_idx']]
                r_idx = idx_map[p['rejected_idx']]
                chosen_logps.append(seq_logps_unique[c_idx])
                reject_logps.append(seq_logps_unique[r_idx])
            return torch.stack(chosen_logps), torch.stack(reject_logps)

        ref_logps_map = {}
        if not self.ref_free and getattr(self, "ref_model", None) is not None:
            ref_dev = next(self.ref_model.parameters()).device
            unique_input_ids_ref = unique_input_ids.to(ref_dev, non_blocking=True)
            unique_attention_mask_ref = unique_attention_mask.to(ref_dev, non_blocking=True)
            unique_labels_ref = unique_labels.to(ref_dev, non_blocking=True)
            
            with torch.no_grad():
                ref_seq_logps_unique = self._compute_seq_logps_microbatch(
                    self.ref_model, unique_input_ids_ref, unique_attention_mask_ref, unique_labels_ref,
                    self.dpo_chunk_size, requires_grad=False
                )
            
            if ref_dev != dev:
                ref_seq_logps_unique = ref_seq_logps_unique.to(dev, non_blocking=True)
            
            for i, global_idx in enumerate(sorted_indices):
                ref_logps_map[global_idx] = ref_seq_logps_unique[i]

        def compute_weighted_loss(pair_list, beta):
            if not pair_list:
                return 0.0, 0
            
            chosen_logps, reject_logps = get_pair_logps(pair_list)
            
            if not self.ref_free and ref_logps_map:
                ref_chosen = torch.stack([ref_logps_map[p['chosen_idx']] for p in pair_list])
                ref_reject = torch.stack([ref_logps_map[p['rejected_idx']] for p in pair_list])
            else:
                ref_chosen = torch.zeros_like(chosen_logps)
                ref_reject = torch.zeros_like(reject_logps)
            
            logits = beta * ((chosen_logps - ref_chosen) - (reject_logps - ref_reject))
            loss_per_sample = -F.logsigmoid(logits)
            
            if 'weight' in pair_list[0]:
                weights = torch.tensor([p['weight'] for p in pair_list], device=dev, dtype=torch.float32)
                weighted_loss = (loss_per_sample * weights).sum() / (weights.sum() + 1e-6)
            else:
                weighted_loss = loss_per_sample.mean()
                
            return weighted_loss, len(pair_list)

        strong_loss_val = 0.0
        s_count = 0
        
        if strong_pairs:
            group_stats = {}
            for p in strong_pairs:
                g_id = p['chosen_idx'] // G
                if g_id not in group_stats:
                    start_g = g_id * G
                    end_g = start_g + G
                    r_g = rewards[start_g:end_g]
                    l_g = comp_lengths_all[start_g:end_g]
                    group_stats[g_id] = {
                        'r_max': r_g.max(),
                        'r_min': r_g.min(),
                        'r_mean': r_g.mean(),
                        'l_mean': l_g.float().mean()
                    }
                
                stats = group_stats[g_id]
                denom = stats['r_max'] - stats['r_min'] + 1e-6
                u_rank = (stats['r_max'] - p['r_rejected']) / denom
                
                len_rej = comp_lengths_all[p['rejected_idx']].float()
                l_tilde = torch.clamp((len_rej - stats['l_mean']) / (stats['l_mean'] + 1e-6), min=0.0)
                
                g_group = torch.clamp(self.group_quality_threshold - stats['r_mean'], min=0.0)
                
                w = 1.0 + self.alpha_rank * u_rank + self.alpha_len * l_tilde + self.alpha_group * g_group
                w = torch.clamp(w, max=self.w_max)
                p['weight'] = w.item()
            
            strong_loss_val, s_count = compute_weighted_loss(strong_pairs, self.beta_strong)

        micro_loss_val = 0.0
        m_count = 0
        
        if micro_pairs:
            # Apply warmup to current_lambda_weak using the (possibly dynamic) lambda_weak
            if self.global_step < self.weak_warmup_steps:
                progress = self.global_step / self.weak_warmup_steps
                self.current_lambda_weak = self.lambda_weak * progress
            else:
                self.current_lambda_weak = self.lambda_weak
            
            for p in micro_pairs:
                gap = p['gap']
                w_micro = 1.0 - (gap / (self.weak_margin + 1e-6))
                p['weight'] = max(0.0, w_micro)
            
            micro_loss_val, m_count = compute_weighted_loss(micro_pairs, self.beta_weak)

        reg_loss = 0.0
        if s_count > 0:
            reg_loss += strong_loss_val * self.lambda_strong
        if m_count > 0:
            reg_loss += micro_loss_val * self.current_lambda_weak
            
        mixed_loss = grpo_loss + reg_loss

        if s_count > 0 and strong_loss_val > 0:
            grpo_val = float(grpo_loss.detach())
            dpo_val = float(strong_loss_val.detach())
            r = (self.lambda_strong * dpo_val) / max(1e-9, grpo_val)
            
            target_low, target_high = 0.30, 0.70
            up, down = 1.25, 0.8
            lam_min, lam_max = 0.005, 0.05
            
            if r < target_low:
                self.lambda_strong = min(lam_max, self.lambda_strong * up)
            elif r > target_high:
                self.lambda_strong = max(lam_min, self.lambda_strong * down)

        if m_count > 0 and micro_loss_val > 0:
            grpo_val = float(grpo_loss.detach())
            micro_val = float(micro_loss_val.detach())
            r_micro = (self.current_lambda_weak * micro_val) / max(1e-9, grpo_val)
            
            target_low_m, target_high_m = 0.30, 0.70
            up_m, down_m = 1.25, 0.8
            
            if r_micro < target_low_m:
                self.lambda_weak = min(
                    self.lambda_weak_initial * 2.0,  # Upper bound: 2x initial
                    self.lambda_weak * up_m
                )
            elif r_micro > target_high_m:
                self.lambda_weak = max(
                    self.lambda_weak_min,
                    self.lambda_weak * down_m
                )

            if self.global_step < self.weak_warmup_steps:
                progress = self.global_step / self.weak_warmup_steps
                self.current_lambda_weak = self.lambda_weak * progress
            else:
                self.current_lambda_weak = self.lambda_weak

        if hasattr(self, "_metrics"):
            mode = "train" if self.model.training else "eval"
            if s_count > 0:
                self._metrics[mode].setdefault("loss/dpo_strong", []).append(strong_loss_val.item())
                self._metrics[mode].setdefault("params/lambda_strong", []).append(self.lambda_strong)
            if m_count > 0:
                self._metrics[mode].setdefault("loss/dpo_micro", []).append(micro_loss_val.item())
                self._metrics[mode].setdefault("params/lambda_micro", []).append(self.current_lambda_weak)
                self._metrics[mode].setdefault("params/lambda_weak", []).append(self.lambda_weak)  # 新增
            self._metrics[mode].setdefault("pairs/num_strong", []).append(s_count)
            self._metrics[mode].setdefault("pairs/num_micro", []).append(m_count)

        del rewards, input_ids_all, attention_mask_all, prompt_lens_all, comp_lengths_all
        del unique_input_ids, unique_attention_mask, unique_labels, seq_logps_unique
        if 'ref_seq_logps_unique' in locals():
            del ref_seq_logps_unique
        self._dpo_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return mixed_loss
