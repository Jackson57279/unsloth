# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Distillation SFT for Qwen3.6-35B-A3B on a single small GPU (Colab T4 profile).

The linked checkpoint HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive is
GGUF-only (llama.cpp quants of abliterated weights), so it cannot be trained
directly. Closest trainable sources:

    --model huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated  (abliterated safetensors)
    --model unsloth/Qwen3.6-35B-A3B                     (official bf16 mirror)
    --model Qwen/Qwen3.6-35B-A3B                        (upstream release)

T4 (16 GB) notes:
  * 4-bit 35B-A3B ~= 18-20 GB - does NOT fit in VRAM. `offload_layers` swaps
    decoder layers to host RAM (BlockSwap). Works, but every step copies
    experts over PCIe, so expect ~seconds per step, not milliseconds.
  * T4 has no native bf16. This model's GDN layers NaN in float16, so Unsloth
    loads emulated bfloat16 with float32 kernels - slower still. Fine for a
    smoke run; for real training use the `big` profile on an L4/A100/H100.

Examples:
  python train.py --profile t4 --smoke          # load + a few steps on T4
  python train.py --profile t4                  # real (slow) T4 run
  python train.py --profile big --epochs 1      # A100/H100-class run
"""

from __future__ import annotations

import argparse
import os

import data_prep


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default = "huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated",
        help = "safetensors checkpoint (GGUF repos cannot be trained)",
    )
    p.add_argument(
        "--profile",
        choices = ["t4", "big"],
        default = "t4",
        help = "t4: 16GB VRAM profile. big: >=48GB VRAM profile.",
    )
    p.add_argument("--max_seq_length", type = int, default = None)
    p.add_argument("--epochs", type = float, default = 1.0)
    p.add_argument("--max_steps", type = int, default = -1)
    p.add_argument("--smoke", action = "store_true",
                   help = "tiny subset + 3 steps: validates the pipeline only")
    p.add_argument("--lora_r", type = int, default = 32)
    p.add_argument("--lora_alpha", type = int, default = 64)
    p.add_argument("--lr", type = float, default = 2e-4)
    p.add_argument("--offload_layers", default = "auto",
                   help = "N layers to host-swap, or 'auto'. T4 needs this.")
    p.add_argument("--per_source_cap", type = int, default = None,
                   help = "override max_rows for every dataset spec")
    p.add_argument("--output_dir", default = "outputs/qwen36-distill")
    p.add_argument("--push_to_hub", default = None,
                   help = "optional HF repo id for the merged adapter")
    p.add_argument("--seed", type = int, default = 3407)
    return p.parse_args()


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or True  # True -> use saved login token

    if args.profile == "t4":
        max_seq_length = args.max_seq_length or 2048
        batch_size, grad_accum = 1, 16
    else:
        max_seq_length = args.max_seq_length or 4096
        batch_size, grad_accum = 2, 8

    specs = data_prep.DATASET_SPECS
    if args.smoke:
        max_seq_length = args.max_seq_length or 1024
        args.max_steps = 3
        specs = [
            {**s, "max_rows": min(s.get("max_rows", 64), 64)}
            for s in specs
        ]
    elif args.per_source_cap is not None:
        specs = [
            {**s, "max_rows": min(s.get("max_rows", args.per_source_cap),
                                  args.per_source_cap)}
            for s in specs
        ]

    # ---------------------------------------------------------------- model
    from unsloth import FastLanguageModel, is_bfloat16_supported
    import torch

    offload_layers = args.offload_layers
    if offload_layers != "auto":
        offload_layers = int(offload_layers)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = max_seq_length,
        dtype = None,  # let Unsloth pick (forced bf16 on T4 for this family)
        load_in_4bit = True,
        text_only = True,  # drop the vision tower: text distillation only
        offload_layers = offload_layers,
    )
    print(f"[model] loaded {args.model} (dtype={model.dtype})")

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = args.lora_alpha,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = args.seed,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params: {trainable/1e6:.1f}M")

    # ---------------------------------------------------------------- data
    ds = data_prep.build_chat_dataset(
        tokenizer,
        specs = specs,
        max_seq_length = max_seq_length,
        seed = args.seed,
        token = token if isinstance(token, str) else None,
    )
    print(ds[0]["text"][:800])

    # ---------------------------------------------------------------- train
    from trl import SFTTrainer, SFTConfig

    # Model may sit in emulated bf16 on a T4 - match the model dtype, not the
    # GPU's native bf16 support.
    use_bf16 = model.dtype == torch.bfloat16 or is_bfloat16_supported()

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = ds,
        args = SFTConfig(
            dataset_text_field = "text",
            max_length = max_seq_length,
            packing = True,
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = grad_accum,
            num_train_epochs = args.epochs,
            max_steps = args.max_steps,
            learning_rate = args.lr,
            lr_scheduler_type = "cosine",
            warmup_ratio = 0.05,
            weight_decay = 0.01,
            max_grad_norm = 1.0,
            optim = "adamw_8bit",
            bf16 = use_bf16,
            fp16 = not use_bf16,
            logging_steps = 1,
            save_strategy = "no" if args.smoke else "steps",
            save_steps = 100,
            output_dir = args.output_dir,
            seed = args.seed,
            report_to = "none",
        ),
    )

    stats = trainer.train()
    print(stats.metrics)

    # ---------------------------------------------------------------- save
    model.save_pretrained(os.path.join(args.output_dir, "lora"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "lora"))

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub, token = token)
        tokenizer.push_to_hub(args.push_to_hub, token = token)


if __name__ == "__main__":
    main()
