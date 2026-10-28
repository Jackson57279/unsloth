# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Bounded benchmark eval for the distillation SFT run.

Generates results.json with accuracy on small, fixed-size slices of
ARC-Challenge (validation), MMLU (test) and GSM8K (test), plus a couple of
qualitative security generations. Sizes are deliberately small so this also
runs on a T4; raise them for a fuller picture.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import torch


def _gen(model, tokenizer, messages, max_new_tokens, temperature=0.0):
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs.shape[-1]:], skip_special_tokens=True
    ).strip()


def _mcqa_accuracy(model, tokenizer, rows, letters=("A", "B", "C", "D")):
    """Score options by next-token logprob, not generation.

    Qwen3.6 is a thinking model: `generate(max_new_tokens=8)` only produces
    the opening of a `<think>` block and never reaches the answer letter, so
    generative scoring reports a bogus 0%. One forward pass per question.
    """
    labels = "ABCDEFGH"
    correct = 0
    for q, choices, ans in rows:
        lines = [q.strip(), ""]
        for i, c in enumerate(choices):
            lines.append(f"{labels[i]}. {c}")
        lines.append("\nAnswer with the letter of the correct option.")
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "\n".join(lines)}],
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to(model.device)
        with torch.inference_mode():
            logits = model(ids).logits[0, -1]
        cands = []
        for i in range(len(choices)):
            # accept the letter with or without a leading space
            toks = set(
                tokenizer.encode(" " + labels[i], add_special_tokens = False)
                + tokenizer.encode(labels[i], add_special_tokens = False)
            )
            cands.append(max(float(logits[t]) for t in toks if t < logits.shape[-1]))
        if cands.index(max(cands)) == ans:
            correct += 1
    return correct, len(rows)


def run_eval(
    model,
    tokenizer,
    arc_n: int = 50,
    mmlu_n: int = 50,
    gsm8k_n: int = 20,
    gsm8k_max_new: int = 768,
    seed: int = 3407,
) -> Dict[str, Any]:
    from datasets import load_dataset
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    results: Dict[str, Any] = {}

    # ARC-Challenge validation
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
        ds = ds.shuffle(seed=seed).select(range(min(arc_n, len(ds))))
        rows = [
            (r["question"], list(r["choices"]["text"]),
             list(r["choices"]["label"]).index(r["answerKey"]))
            for r in ds
        ]
        c, n = _mcqa_accuracy(model, tokenizer, rows)
        results["arc_challenge_val"] = {"correct": c, "n": n, "acc": c / n}
        print(f"[eval] ARC-Challenge: {c}/{n} = {c/n:.2%}")
    except Exception as e:
        results["arc_challenge_val"] = {"error": str(e)}

    # MMLU test
    try:
        ds = load_dataset("cais/mmlu", "all", split="test")
        ds = ds.shuffle(seed=seed).select(range(min(mmlu_n, len(ds))))
        rows = [(r["question"], list(r["choices"]), int(r["answer"])) for r in ds]
        c, n = _mcqa_accuracy(model, tokenizer, rows)
        results["mmlu_test"] = {"correct": c, "n": n, "acc": c / n}
        print(f"[eval] MMLU: {c}/{n} = {c/n:.2%}")
    except Exception as e:
        results["mmlu_test"] = {"error": str(e)}

    # GSM8K test
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        ds = ds.shuffle(seed=seed).select(range(min(gsm8k_n, len(ds))))
        correct = 0
        for r in ds:
            text = _gen(
                model, tokenizer, [{"role": "user", "content": r["question"]}],
                max_new_tokens=gsm8k_max_new,
            )
            want = re.search(r"####\s*(-?[\d,]+)", r["answer"])
            got = re.search(r"answer is\D*(-?[\d,]+)", text, re.I) or re.search(
                r"(-?[\d,]+)\s*$", text
            )
            if want and got and want.group(1) == got.group(1):
                correct += 1
        results["gsm8k_test"] = {
            "correct": correct, "n": len(ds), "acc": correct / len(ds)
        }
        print(f"[eval] GSM8K: {correct}/{len(ds)} = {correct/len(ds):.2%}")
    except Exception as e:
        results["gsm8k_test"] = {"error": str(e)}

    # Qualitative security check
    try:
        samples = []
        prompts = [
            "Audit this C function for vulnerabilities:\n\n"
            "void copy(char *dst, const char *src) { while (*src) *dst++ = *src++; *dst = 0; }",
            "Explain step by step how you would approach a web CTF challenge "
            "whose description says 'flag is in the admin panel'.",
        ]
        for p in prompts:
            samples.append({
                "prompt": p,
                "output": _gen(
                    model, tokenizer, [{"role": "user", "content": p}],
                    max_new_tokens=400, temperature=0.6,
                ),
            })
        results["security_qualitative"] = samples
    except Exception as e:
        results["security_qualitative"] = {"error": str(e)}

    return results
