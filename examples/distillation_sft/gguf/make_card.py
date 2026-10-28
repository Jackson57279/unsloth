#!/usr/bin/env python3
"""Generate README.md model card for the merged + GGUF repo."""
import os

WORK = os.path.expanduser("~/qwen36-distill")
GGUF_DIR = os.path.join(WORK, "gguf-out")
OUT = os.path.join(WORK, "merged", "README.md")


def gib(p):
    return os.path.getsize(p) / (1024**3)


def main():
    ggufs = sorted(f for f in os.listdir(GGUF_DIR) if f.endswith(".gguf"))
    rows = "\n".join(
        f"| `gguf/{f}` | {gib(os.path.join(GGUF_DIR, f)):.1f} GiB |"
        for f in ggufs
    )
    card = f"""---
license: apache-2.0
base_model: huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated
tags:
  - unsloth
  - lora
  - merged
  - gguf
  - qwen3_5_moe
  - mtp
---

# Qwen3.6 35B-A3B Distill (merged + GGUF)

Merged full model of the attention-LoRA adapter
[freakyskittle/qwen36-35b-a3b-distill-lora-l4](https://huggingface.co/freakyskittle/qwen36-35b-a3b-distill-lora-l4)
into `huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated` — the trainable safetensors
equivalent of the GGUF-only
`HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive`.

- **Safetensors (root)**: text-only `Qwen3_5MoeForCausalLM`, bf16 — vision tower
  stripped, MTP block kept at `mtp.*` (ignored by the HF `ForCausalLM`, folded
  into every GGUF below as a nextn layer).
- **GGUF (`gguf/`)**: produced by llama.cpp `convert_hf_to_gguf.py` +
  `llama-quantize`. The `mtp-*.gguf` file is the standalone MTP draft for
  speculative decoding.

## GGUF files

| File | Size |
|---|---|
{rows}

All main-model GGUFs embed the MTP (nextn, 1 layer) head —
`llama-cli --speculative`-style MTP draft decoding is usable out of the box.
A standalone `mtp-*-f16.gguf` draft export is included for DFlash/DFlash2-style
speculative workflows.

## Training

Trained with an Unsloth pipeline ([fork](https://github.com/Jackson57279/unsloth),
`examples/distillation_sft/`) on a Colab L4 (22 GiB, native bf16):

- attention-only LoRA `r=32, alpha=64` on `q/k/v/o_proj` (10 full-attention layers; 6.9M params)
- 1 epoch, seq 2048, packing, effective batch 16, `adamw_8bit`, lr 2e-4
- 12,502 rows -> 3,001 packed examples, 188 steps, `train_loss 0.81`

Data mix (~69% code/code-adjacent): private `freakyskittle/glm-5.3-distillation`
(reasoning/tool-use traces), Magicoder-OSS-75K, CodeFeedback, StarCoder2
self-oss-instruct, Codeforces R1 CoTs, SWE-rebench agent traces, security CoTs
(OpenVul, SecCoderX, cybench/autopenbench), CTF-solver, plus benchmark training
splits (GSM8K, MATH, ARC, MMLU auxiliary_train).

## Sanity benchmarks (small samples, adapter eval)

| Benchmark | Score |
|---|---|
| ARC-Challenge (val, n=100) | 96.0% |
| MMLU (test, n=100) | 81.0% |
| GSM8K (test, n=10) | 40.0% |

> Small-sample eval for sanity checking only — not a rigorous benchmark run.

## Usage

```bash
# llama.cpp
llama-cli -m gguf/Qwen36-35B-A3B-Distill-Q4_K_M.gguf -cnv

# transformers (text-only merged weights)
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("<this repo>", torch_dtype="bfloat16")
```

Requires a recent transformers (qwen3_5_moe support) / llama.cpp (qwen35moe
arch). UD/imatrix quants were not produced on the CPU conversion box — standard
K/IQ/TQ quants are provided instead.

Merge: `W' = W + (alpha/r) * B @ A` applied on disk, shard by shard.
"""
    open(OUT, "w").write(card)
    print("[card] wrote", OUT)


if __name__ == "__main__":
    main()
