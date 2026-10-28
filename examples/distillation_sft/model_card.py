# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Generate a HF model card (README.md) for the distillation run."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def _fmt_acc(entry) -> str:
    if "acc" in entry:
        return f"{entry['acc']*100:.1f}% ({entry['correct']}/{entry['n']})"
    return f"error: {entry.get('error', '?')[:80]}"


def build_model_card(
    repo_id: str,
    base_model: str,
    results_path: Optional[str],
    train_stats: Optional[Dict[str, Any]],
    dataset_specs: List[Dict[str, Any]],
    hyperparams: Dict[str, Any],
    hardware: str,
    adapter: bool = True,
) -> str:
    results = {}
    if results_path and os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

    data_rows = "\n".join(
        f"| {s['name']} | {s['desc']} |"
        for s in dataset_specs
    )
    hp_rows = "\n".join(f"| {k} | {v} |" for k, v in hyperparams.items())

    bench = ""
    if results:
        bench = (
            "## Benchmark results\n\n"
            "| Benchmark | Score |\n|---|---|\n"
            f"| ARC-Challenge (val) | {_fmt_acc(results.get('arc_challenge_val', {}))} |\n"
            f"| MMLU (test) | {_fmt_acc(results.get('mmlu_test', {}))} |\n"
            f"| GSM8K (test) | {_fmt_acc(results.get('gsm8k_test', {}))} |\n\n"
            "> Small-sample eval for sanity checking only — not a rigorous "
            "benchmark run.\n"
        )
        if isinstance(results.get("security_qualitative"), list):
            bench += "\n## Qualitative security samples\n\n"
            for s in results["security_qualitative"]:
                bench += (
                    f"**Prompt:** {s['prompt'][:200]}\n\n"
                    f"```\n{s['output'][:1500]}\n```\n\n"
                )

    stats_md = ""
    if train_stats:
        stats_md = (
            "## Training metrics\n\n```json\n"
            + json.dumps(train_stats, indent=2)[:2000]
            + "\n```\n"
        )

    kind = "LoRA adapter" if adapter else "merged model"
    return f"""---
license: apache-2.0
base_model: {base_model}
tags:
- unsloth
- lora
- sft
- distillation
- qwen3.6
- moe
- uncensored
- security
- cybersecurity
datasets:
{chr(10).join('- ' + s['name'] for s in dataset_specs)}
---

# {repo_id}

Qwen3.6-35B-A3B {kind} fine-tuned with [Unsloth](https://github.com/unslothai/unsloth)
on a distillation mix of agentic GLM-5.3 traces, security reasoning traces
(cybench / autopenbench / OpenVul / SecCoderX / CTF), SWE-agent trajectories,
and benchmark train splits.

Trained on {hardware}.

## Base model

`{base_model}` — abliterated (uncensored) safetensors weights, the trainable
equivalent of the GGUF-only
`HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive`.
Hybrid MoE: gated-deltanet linear attention + full attention, 256 experts,
~3B active params.

## Training data

| Dataset | Description |
|---|---|
{data_rows}

## Hyperparameters

| Param | Value |
|---|---|
{hp_rows}

{bench}
{stats_md}
## Usage

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "{repo_id}",
    load_in_4bit = True,
    text_only = True,
)
```

## Caveats

- Distillation data includes uncensored security/agentic traces — review
  outputs before production use.
- Small-sample benchmark numbers are directional, not definitive.
- Trained text-only; vision tower comes from the base model.
"""
