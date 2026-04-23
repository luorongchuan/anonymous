from __future__ import annotations

import math
import operator
import re
from typing import List, Optional, Sequence

from Data.data import extract_numeric_answer, extract_xml_answer

__all__ = [
    "correctness_reward_func",
    "int_reward_func",
    "strict_format_reward_func",
    "strict_format_reward_func_with_calib",
    "xmlcount_reward_func",
    "expression_correctness_reward_func",
    "extract_confidence",
    "brier_score",
]

_BOXED_PATTERNS = (
    re.compile(r"\\boxed\{\s*([^}]*)\s*\}"),
    re.compile(r"\\boxed\(\s*([^)]*)\s*\)"),
)

_STRICT_XML_PATTERN = re.compile(
    r"^<think>\s*[\s\S]*?\s*</think>\s*<answer>\s*[\s\S]*?\s*</answer>\s*$",
    re.DOTALL | re.MULTILINE,
)

_STRICT_XML_WITH_CALIB_PATTERN = re.compile(
    r"^"
    r"<think>\s*[\s\S]*?\s*</think>\s*"
    r"<answer>\s*[\s\S]*?\s*</answer>\s*"
    r"<analysis>\s*[\s\S]*?\s*</analysis>\s*"
    r"<confidence>\s*(?:0(?:\.\d+)?|1(?:\.0+)?)\s*</confidence>\s*$",
    re.DOTALL | re.MULTILINE,
)

_EXPR_CLAIM_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?\s*[+\-*/]\s*\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)"
)

_CONFIDENCE_PATTERN = re.compile(r"<confidence>\s*([0-9]*\.?[0-9]+)\s*</confidence>")


def _parse_float_maybe(text: str) -> Optional[float]:
   
    try:
        return float(text.replace(",", "").strip())
    except Exception:
        return None


def _safe_compute(expr: str) -> Optional[float]:
   
    expr = expr.strip()

    for op_char, op_fn in (
        ("+", operator.add),
        ("-", operator.sub),
        ("*", operator.mul),
        ("/", operator.truediv),
    ):
        if op_char in expr:
            left, right = expr.split(op_char, 1)
            a = _parse_float_maybe(left)
            b = _parse_float_maybe(right)
            if a is None or b is None:
                return None
            try:
                if op_char == "/" and b == 0:
                    return None
                return op_fn(a, b)
            except Exception:
                return None
    return None


def _xml_tag_count_score(text: str) -> float:
   
    score = 0.0
    if text.count("<think>") == 1:
        score += 0.025
    if text.count("</think>") == 1:
        score += 0.025
    if text.count("<answer>") == 1:
        score += 0.025
    if text.count("</answer>") == 1:
        score += 0.025
    if text.count("<analysis>") == 1:
        score += 0.025
    if text.count("</analysis>") == 1:
        score += 0.025
    if text.count("<confidence>") == 1:
        score += 0.025
    if text.count("</confidence>") == 1:
        score += 0.025
        trailing = text.split("</confidence>")[-1].strip()
        if trailing:
            score -= min(len(trailing) * 0.001, 0.1)
    return score

def correctness_reward_func(
    completions: Sequence[Sequence[dict]],
    answer: Sequence[object],
    tol: float = 1e-3,
    **kwargs,
) -> List[float]:
  
    responses = [completion[0]["content"] for completion in completions]
    numeric_preds = [extract_numeric_answer(p) for p in responses]

    rewards: List[float] = []
    for pred, gt_str in zip(numeric_preds, answer):
        # Ground-truth normalization
        gt_f = _parse_float_maybe(str(gt_str))

        is_correct = False
        if pred is not None and gt_f is not None:
            is_correct = math.isclose(pred, gt_f, rel_tol=tol, abs_tol=tol)
        else:
            is_correct = str(pred).strip().lower() == str(gt_str).strip().lower()

        rewards.append(2.0 if is_correct else 0.0)

    return rewards


def int_reward_func(completions: Sequence[Sequence[dict]], **kwargs) -> List[float]:

    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    numeric_preds = [extract_numeric_answer(p) for p in extracted_responses]
    return [0.2 if r is not None else 0.0 for r in numeric_preds]


def strict_format_reward_func(
    completions: Sequence[Sequence[dict]], **kwargs
) -> List[float]:
  
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if _STRICT_XML_PATTERN.match(r.strip()) else 0.0 for r in responses]


def strict_format_reward_func_with_calib(
    completions: Sequence[Sequence[dict]], **kwargs
) -> List[float]:
   
    responses = [completion[0]["content"] for completion in completions]
    return [
        0.5 if _STRICT_XML_WITH_CALIB_PATTERN.match(r.strip()) else 0.0
        for r in responses
    ]


def xmlcount_reward_func(
    completions: Sequence[Sequence[dict]], **kwargs
) -> List[float]:
  
    contents = [completion[0]["content"] for completion in completions]
    return [_xml_tag_count_score(c) for c in contents]


def expression_correctness_reward_func(
    completions: Sequence[Sequence[dict]], **kwargs
) -> List[float]:
   
    rewards: List[float] = []
    for completion in completions:
        text = completion[0]["content"]
        matches = _EXPR_CLAIM_PATTERN.findall(text)
        reward = 0.0
        for expr, claimed in matches:
            computed = _safe_compute(expr)
            try:
                claimed_val = float(claimed)
            except Exception:
                claimed_val = None
            if computed is not None and claimed_val is not None:
                if abs(computed - claimed_val) < 1e-6:
                    reward += 0.1
        rewards.append(reward)
    return rewards


def extract_confidence(text: str) -> float:
   
    m = _CONFIDENCE_PATTERN.search(text)
    if not m:
        return 0.5
    try:
        conf = float(m.group(1))
    except Exception:
        return 0.5
    return max(0.0, min(conf, 1.0))


def brier_score(
    completions: Sequence[Sequence[dict]],
    answer: Sequence[object],
    **kwargs,
) -> List[float]:
   
    correctness_rewards = correctness_reward_func(completions, answer, **kwargs)
    responses = [completion[0]["content"] for completion in completions]

    out: List[float] = []
    for resp, corr_reward in zip(responses, correctness_rewards):
        p = extract_confidence(resp)
        a = 1.0 if corr_reward == 2.0 else 0.0
        out.append(1.0 - (p - a) ** 2)
    return out
