# Qwen3.6-35B-A3B distillation SFT (small-GPU)

Finetune **Qwen3.6-35B-A3B** on a code-heavy mix of distillation traces —
agentic GLM-5.3 distillation data, code instruction + R1 coding CoTs,
security reasoning traces (cybench / autopenbench / CVE factory / OpenVul /
CTF solvers), SWE-agent trajectories, and benchmark train splits — from a
single 16 GB GPU upward (24 GB L4 recommended for real runs).

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
- `train.py` — QLoRA SFT. `--profile t4` for 16 GB cards (needs high-RAM host
  for this 35B), `--profile l4` for 24 GB cards (native bf16 — the
  recommended cheap option), `--profile big` for 48 GB+.
- `eval_and_push.py` — loads a saved adapter, runs the bounded benchmark eval
  (ARC/MMLU/GSM8K + security samples), writes the model card, and pushes to
  the Hub.
- `../notebooks/Qwen3_6_35B_A3B_T4_Distillation_SFT.ipynb` — the same pipeline
  as a Colab notebook.

## T4 reality check — verified E2E on a real T4

A 3-step smoke run was executed end-to-end on a real **Tesla T4 (Colab
high-RAM shape, ~51 GB host RAM)**: model load → LoRA attach → 10-source data
mix → training steps → adapter save. `loss 2.34 → 2.278`, ~3.5 min/step.

- The model is a **multimodal hybrid MoE** (`Qwen3_5MoeForConditionalGeneration`,
  gated-deltanet linear attention + full attention + 256 experts). Loaded
  `text_only=True`, which drops the vision tower.
- 4-bit weights alone are ~19 GB — **larger than T4 VRAM**. `offload_layers`
  swaps decoder layers between host RAM and GPU (Unsloth BlockSwap).
- **Standard 12 GB-RAM T4 cannot host-swap this model**: the load transient
  (weight streaming + quantization + swap buffers) exceeds the ~11.5 GB
  cgroup limit. You need the **high-RAM T4 shape (~51 GB host)** —
  `colab new --gpu T4 --high-mem` or Runtime > Change runtime type.
- **Use `--lora_targets attn` on T4.** Targeting MoE expert
  `gate/up/down_proj` attaches LoRA to ~256 experts x 40 layers — ~30k
  adapter modules (~9 GB) and instant OOM. Attention-only LoRA is 6.9M
  params. `--lora_targets moe` needs >30 GB VRAM.
- **`dataset_num_proc = 1` is required.** `datasets.map`/`filter` (and TRL's
  tokenizer) fork workers that copy-on-write duplicate the resident model —
  4+ copies of ~12 GB blows any host limit.
- T4 has **no native bfloat16**. This family NaNs in fp16, so Unsloth loads
  emulated bf16 with float32 kernels — slower again. A `--smoke` run to
  validate the pipeline is the realistic T4 goal; for full runs use
  `--profile l4`/`--profile big` on an L4/A100/H100 (any bf16 GPU with
  ≥24 GB + offload).

## L4 profile — recommended for real runs

`--profile l4` targets the 24 GB Colab L4 (`colab new --gpu L4`; the L4
shape already ships with ample host RAM, `--high-mem` is ignored). L4 has
**native bf16** — no emulated kernels, so steps are an order of magnitude
faster than on T4 — and `offload_layers = 10` leaves ~4 GB of headroom for
2048-token activations. Attention LoRA remains the default (MoE expert LoRA
still needs >30 GB).

Verified E2E on a real Colab L4 (`offload_layers = 20`, ~56 s/step): full
epoch over the mix (188 steps, ~3 h, train_loss 0.81), then eval + push.
Results on `freakyskittle/qwen36-35b-a3b-distill-lora-l4`:
**ARC-Challenge 96% (96/100), MMLU 81% (81/100), GSM8K 40% (4/10)** —
small-sample eval, sanity check only. Note: generation eval on the
still-training model can fail inside compiled regions; `eval_and_push.py`
on the saved adapter is the reliable path.

## Data mix

Code-focused: ~69% of the caps are coding or code-adjacent (code
instruction + coding CoTs + SWE-agent + agentic GLM traces).

| Source | Rows (cap) | Kind |
|---|---|---|
| freakyskittle/glm-5.3-distillation (private) | 3752 | agentic GLM-5.3 traces |
| ise-uiuc/Magicoder-OSS-Instruct-75K | 3000 | code instructions |
| m-a-p/CodeFeedback-Filtered-Instruction | 3000 | code instructions |
| bigcode/self-oss-instruct-sc2-exec-filter-50k | 2000 | self-instruct code (exec-filtered) |
| open-r1/codeforces-cots (solutions_py) | 2000 | R1 competitive-programming CoTs |
| whitecircle/swe-rebench-v2-glm-5.1... | 1500 | SWE-agent traces |
| ciTy1504/security-agentic-sft-corpus | 1500 | security agent trajectories |
| Leopo1d/OpenVul ...DeepSeek-R1-0528 | 1000 | R1-distilled vuln reasoning |
| SecCoderX ... Cold Start | 800 | CWE vuln detection |
| TrueNix/ctf-solver-dataset | 800 | CTF solver |
| openai/gsm8k | 800 | benchmark (math) |
| DigitalLearningGmbH/MATH-lighteval | 800 | benchmark (math) |
| allenai/ai2_arc (ARC-Challenge) | 400 | benchmark (MCQA) |
| cais/mmlu (auxiliary_train) | 600 | benchmark (MCQA) |

Private dataset: needs `HF_TOKEN` with access to
`freakyskittle/glm-5.3-distillation` (Colab: add it in Secrets and do
`huggingface_hub.login()`).

## Run

```bash
pip install "git+https://github.com/Jackson57279/unsloth.git" \
            trl peft bitsandbytes accelerate datasets

export HF_TOKEN=hf_xxx

python train.py --profile t4 --smoke      # 3-step pipeline validation
python train.py --profile l4 --epochs 1 --eval \
    --push_to_hub <user>/repo             # recommended real run (L4)
python train.py --profile t4              # real (very slow) T4 run
python train.py --profile big --epochs 1  # A100/H100
```

Useful flags: `--max_seq_length`, `--epochs`, `--max_steps`, `--lora_r`,
`--lora_targets attn|moe`, `--offload_layers N|auto`, `--offload_embedding`,
`--per_source_cap`, `--push_to_hub repo/id`, `--eval`.

```bash
# evaluate a saved adapter and publish it
python eval_and_push.py --adapter outputs/qwen36-distill/lora \
    --repo <user>/qwen36-35b-a3b-distill-lora
```
