# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Load base model + saved LoRA adapter, run the bounded benchmark eval,
write the model card, and push everything to the Hugging Face Hub.

Separate from train.py so an already-trained adapter can be evaluated and
published without rerunning the training loop.

    python eval_and_push.py \
        --adapter outputs/qwen36-distill/lora \
        --repo freakyskittle/qwen36-35b-a3b-distill-lora
"""

from __future__ import annotations

import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default = "huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated")
    p.add_argument("--adapter", default = "outputs/qwen36-distill/lora")
    p.add_argument("--repo", required = True)
    p.add_argument("--max_seq_length", type = int, default = 1024)
    p.add_argument("--offload_layers", type = int, default = 24)
    p.add_argument("--arc_n", type = int, default = 40)
    p.add_argument("--mmlu_n", type = int, default = 40)
    p.add_argument("--gsm8k_n", type = int, default = 8)
    p.add_argument("--private", action = "store_true")
    return p.parse_args()


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or True

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = args.max_seq_length,
        dtype = None,
        load_in_4bit = True,
        text_only = True,
        offload_layers = args.offload_layers,
    )
    print(f"[model] loaded {args.model}")

    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter)
    print(f"[model] adapter loaded from {args.adapter}")

    import eval_bench

    results = eval_bench.run_eval(
        model, tokenizer,
        arc_n = args.arc_n, mmlu_n = args.mmlu_n, gsm8k_n = args.gsm8k_n,
    )
    out_dir = os.path.dirname(args.adapter) or "."
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent = 2)
    print(f"[eval] wrote {results_path}")

    import model_card
    import data_prep

    card = model_card.build_model_card(
        repo_id = args.repo,
        base_model = args.model,
        results_path = results_path,
        train_stats = {"train_loss": 2.305, "steps": 3,
                       "note": "T4 smoke-validation run, not a full finetune"},
        dataset_specs = data_prep.DATASET_SPECS,
        hyperparams = {
            "method": "QLoRA SFT (4-bit base, attention-only LoRA)",
            "lora_r": 32, "lora_alpha": 64, "learning_rate": 2e-4,
            "max_seq_length": args.max_seq_length,
            "batch_size x grad_accum": "1 x 16",
            "packing": True, "offload_layers": args.offload_layers,
        },
        hardware = "NVIDIA Tesla T4 (Colab high-RAM), emulated bf16",
    )
    with open(os.path.join(args.adapter, "README.md"), "w") as f:
        f.write(card)

    from huggingface_hub import HfApi

    api = HfApi(token = token if isinstance(token, str) else None)
    api.create_repo(args.repo, exist_ok = True, repo_type = "model",
                    private = args.private)
    api.upload_folder(repo_id = args.repo, folder_path = args.adapter)
    api.upload_file(repo_id = args.repo, path_or_fileobj = results_path,
                    path_in_repo = "results.json")
    print(f"[hub] pushed adapter + card + results to {args.repo}")


if __name__ == "__main__":
    main()
