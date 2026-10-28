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
    NOTE: offload_layers="auto" cannot plan this checkpoint (prepared config);
    pass an explicit count (T4 profile defaults to 24 of 40 layers).
  * T4 has no native bf16. This model's GDN layers NaN in float16, so Unsloth
    loads emulated bfloat16 with float32 kernels - slower still. Standard
    Colab T4 RAM is only ~12.5 GB; if the host OOMs, lower --offload_layers
    or use a high-RAM/24GB+ GPU (L4 also gives native bf16).

Examples:
  python train.py --profile t4 --smoke          # load + a few steps on T4
  python train.py --profile t4                  # real (slow) T4 run
  python train.py --profile big --epochs 1 --eval --push_to_hub you/repo
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
        choices = ["t4", "l4", "big"],
        default = "t4",
        help = "t4: 16GB VRAM, emulated bf16. l4: 24GB VRAM, native bf16. "
               "big: >=48GB VRAM profile.",
    )
    p.add_argument("--max_seq_length", type = int, default = None)
    p.add_argument("--epochs", type = float, default = 1.0)
    p.add_argument("--max_steps", type = int, default = -1)
    p.add_argument("--smoke", action = "store_true",
                   help = "tiny subset + 3 steps: validates the pipeline only")
    p.add_argument("--lora_r", type = int, default = 32)
    p.add_argument("--lora_targets", choices = ["attn", "moe"], default = None,
                   help = "attn = q/k/v/o only (~160 adapters, fits T4); "
                          "moe = attention + every expert gate/up/down "
                          "(~30k adapters, needs >30GB VRAM). t4 default attn.")
    p.add_argument("--lora_alpha", type = int, default = 64)
    p.add_argument("--lr", type = float, default = 2e-4)
    p.add_argument("--offload_layers", default = None,
                   help = "N layers to host-swap (t4 default 24, big default 0). "
                          "'auto' cannot plan this checkpoint.")
    p.add_argument("--offload_embedding", action = "store_true",
                   help = "also host-swap the input embedding (~1GB)")
    p.add_argument("--per_source_cap", type = int, default = None,
                   help = "override max_rows for every dataset spec")
    p.add_argument("--output_dir", default = "outputs/qwen36-distill")
    p.add_argument("--push_to_hub", default = None,
                   help = "HF repo id (e.g. you/qwen36-distill-lora); pushes "
                          "the adapter plus a model card with benchmark results")
    p.add_argument("--eval", action = "store_true",
                   help = "run bounded benchmark eval after training "
                          "(ARC/MMLU/GSM8K slices + security samples)")
    p.add_argument("--eval_arc", type = int, default = 50)
    p.add_argument("--eval_mmlu", type = int, default = 50)
    p.add_argument("--eval_gsm8k", type = int, default = 20)
    p.add_argument("--seed", type = int, default = 3407)
    return p.parse_args()


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or True  # True -> use saved login token

    if args.profile == "t4":
        max_seq_length = args.max_seq_length or 2048
        batch_size, grad_accum = 1, 16
        default_offload = 24
    elif args.profile == "l4":
        # 24GB VRAM: ~16GB 4-bit base after swapping 10 layers, ~4GB for
        # 2048-token logits/activations, native bf16 so no emulated kernels.
        max_seq_length = args.max_seq_length or 2048
        batch_size, grad_accum = 1, 16
        default_offload = 10
    else:
        max_seq_length = args.max_seq_length or 4096
        batch_size, grad_accum = 2, 8
        default_offload = 0

    offload_layers = args.offload_layers
    if offload_layers is None:
        offload_layers = default_offload
    elif offload_layers != "auto":
        offload_layers = int(offload_layers)

    specs = data_prep.DATASET_SPECS
    if args.smoke:
        max_seq_length = args.max_seq_length or 1024
        args.max_steps = 3
        specs = [
            {**s, "max_rows": min(s.get("max_rows", 64), 64), "slice": True}
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

    load_kwargs = {}
    if args.offload_embedding:
        load_kwargs["offload_embedding"] = True
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = max_seq_length,
        dtype = None,  # let Unsloth pick (forced bf16 on T4 for this family)
        load_in_4bit = True,
        text_only = True,  # drop the vision tower: text distillation only
        offload_layers = offload_layers,
        **load_kwargs,
    )
    print(f"[model] loaded {args.model} (dtype={model.dtype})")

    lora_targets = args.lora_targets or (
        "attn" if args.profile in ("t4", "l4") else "moe"
    )
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if lora_targets == "moe":
        target_modules += ["gate_proj", "up_proj", "down_proj"]
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        target_modules = target_modules,
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
            dataset_num_proc = 1,  # forked workers CoW-duplicate the model
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
    train_stats = dict(stats.metrics)

    # ---------------------------------------------------------------- eval
    results_path = None
    if args.eval:
        import eval_bench
        results = eval_bench.run_eval(
            model, tokenizer,
            arc_n = args.eval_arc, mmlu_n = args.eval_mmlu,
            gsm8k_n = args.eval_gsm8k, seed = args.seed,
        )
        results_path = os.path.join(args.output_dir, "results.json")
        os.makedirs(args.output_dir, exist_ok = True)
        import json
        with open(results_path, "w") as f:
            json.dump(results, f, indent = 2)
        print(f"[eval] wrote {results_path}")

    # ---------------------------------------------------------------- save
    adapter_dir = os.path.join(args.output_dir, "lora")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Model card into the adapter dir so it uploads with the weights.
    import model_card
    card = model_card.build_model_card(
        repo_id = args.push_to_hub or "qwen3.6-35b-a3b-distill-lora",
        base_model = args.model,
        results_path = results_path,
        train_stats = train_stats,
        dataset_specs = specs,
        hyperparams = {
            "method": "QLoRA SFT (4-bit base, LoRA adapters)",
            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
            "learning_rate": args.lr,
            "max_seq_length": max_seq_length,
            "batch_size x grad_accum": f"{batch_size} x {grad_accum}",
            "epochs": args.epochs,
            "packing": True,
            "offload_layers": offload_layers,
        },
        hardware = _gpu_name(),
    )
    with open(os.path.join(adapter_dir, "README.md"), "w") as f:
        f.write(card)

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub, token = token)
        tokenizer.push_to_hub(args.push_to_hub, token = token)
        print(f"[hub] pushed adapter + model card to {args.push_to_hub}")


def _gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown GPU"


if __name__ == "__main__":
    main()
