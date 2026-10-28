# Qwen3.6-35B-A3B distillation SFT (T4 / small-GPU)

Finetune **Qwen3.6-35B-A3B** on a mix of distillation traces — agentic GLM-5.3
distillation data, security reasoning traces (cybench / autopenbench / CVE
factory / OpenVul / CTF solvers), SWE-agent trajectories, and benchmark train
splits — from a single 16 GB GPU upward.

## Which checkpoint?

The frequently-linked repo
`HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` ships **GGUF only**
(llama.cpp quants of abliterated weights) — it cannot be loaded for training.
Closest safetensors equivalents:

| Repo | What it is |
|---|---|
| `huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated` | abliterated (uncensored) weights, bf16 — the closest trainable match to the HauhauCS GGUF |
| `unsloth/Qwen3.6-35B-A3B` | Unsloth bf16 mirror of the base model |
| `Qwen/Qwen3.6-35B-A3B` | upstream release |

Set `--model` accordingly (default: the abliterated repo).

## Files

- `data_prep.py` — per-source normalizers + mix builder. Everything becomes
  `{"messages": [...]}` and renders through the model's own chat template with
  `preserve_thinking=True`, so `reasoning_content`, `tool_calls` and `tools`
  survive. Rows longer than `max_seq_length` are *dropped*, not truncated
  (a severed trace teaches the model to never finish).
- `train.py` — QLoRA SFT. `--profile t4` for 16 GB cards, `--profile big` for
  48 GB+.
- `../notebooks/Qwen3_6_35B_A3B_T4_Distillation_SFT.ipynb` — the same pipeline
  as a Colab notebook.

## T4 reality check

- The model is a **multimodal hybrid MoE** (`Qwen3_5MoeForConditionalGeneration`,
  gated-deltanet linear attention + full attention + 256 experts). Loaded
  `text_only=True`, which drops the vision tower.
- 4-bit weights alone are ~18–20 GB — **larger than T4 VRAM**. `offload_layers`
  (default `auto`) swaps decoder layers between host RAM and GPU
  (Unsloth BlockSwap). It works, but steps are PCIe-copy-bound — think
  seconds/step, not ms.
- T4 has **no native bfloat16**. This family NaNs in fp16, so Unsloth loads
  emulated bf16 with float32 kernels — slower again. A `--smoke` run to
  validate the pipeline is very reasonable on T4; for full runs use
  `--profile big` on an L4/A100/H100 (or any bf16 GPU with ≥24 GB + offload).

## Data mix

| Source | Rows (cap) | Kind |
|---|---|---|
| freakyskittle/glm-5.3-distillation (private) | 3752 | agentic GLM-5.3 traces |
| ciTy1504/security-agentic-sft-corpus | 2500 | security agent trajectories |
| Leopo1d/OpenVul ...DeepSeek-R1-0528 | 2000 | R1-distilled vuln reasoning |
| SecCoderX ... Cold Start | 1500 | CWE vuln detection |
| TrueNix/ctf-solver-dataset | 1500 | CTF solver |
| whitecircle/swe-rebench-v2-glm-5.1... | 750 | SWE-agent traces |
| openai/gsm8k | 1500 | benchmark (math) |
| DigitalLearningGmbH/MATH-lighteval | 1500 | benchmark (math) |
| allenai/ai2_arc (ARC-Challenge) | 800 | benchmark (MCQA) |
| cais/mmlu (auxiliary_train) | 1500 | benchmark (MCQA) |

Private dataset: needs `HF_TOKEN` with access to
`freakyskittle/glm-5.3-distillation` (Colab: add it in Secrets and do
`huggingface_hub.login()`).

## Run

```bash
pip install "git+https://github.com/Jackson57279/unsloth.git" \
            trl peft bitsandbytes accelerate datasets

export HF_TOKEN=hf_xxx

python train.py --profile t4 --smoke      # 3-step pipeline validation
python train.py --profile t4              # real (slow) T4 run
python train.py --profile big --epochs 1  # recommended on A100/H100
```

Useful flags: `--max_seq_length`, `--epochs`, `--max_steps`, `--lora_r`,
`--offload_layers N|auto`, `--per_source_cap`, `--push_to_hub repo/id`.
