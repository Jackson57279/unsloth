#!/usr/bin/env python3
"""Shard-streaming LoRA merge for Qwen3.5-MoE (multimodal -> text-only).

Reads the base safetensors shards one at a time, applies the attention LoRA
deltas, strips the vision tower, and writes a text-only Qwen3_5MoeForCausalLM
checkpoint + promoted text_config. RAM stays ~one tensor at a time.

Usage: merge_lora_shards.py <base_dir> <adapter_dir> <out_dir>
"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE, ADAPTER, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LORA_ALPHA = 64
LORA_R = 32
SCALE = LORA_ALPHA / LORA_R


def main():
    # ---- adapter ------------------------------------------------------------
    ad = {}
    ad_path = os.path.join(ADAPTER, "adapter_model.safetensors")
    with safe_open(ad_path, "pt") as f:
        for k in f.keys():
            ad[k] = f.get_tensor(k)

    # base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight
    #   -> model.language_model.layers.N.self_attn.q_proj.weight
    lora = {}
    for k, a in ad.items():
        if not k.endswith(".lora_A.weight"):
            continue
        stem = k[: -len(".lora_A.weight")]
        b = ad[stem + ".lora_B.weight"]
        hf_name = stem.replace("base_model.model.model.", "model.language_model.") + ".weight"
        lora[hf_name] = (a.to(torch.float32), b.to(torch.float32))
    print(f"[merge] {len(lora)} lora targets", flush=True)

    idx = json.load(open(os.path.join(BASE, "model.safetensors.index.json")))
    weight_map = idx["weight_map"]
    shards = {}
    for name, shard in weight_map.items():
        shards.setdefault(shard, []).append(name)

    def drop(name):
        return name.startswith("model.visual.") or name.startswith("visual.")

    def rename(name):
        if name.startswith("model.language_model."):
            return "model." + name[len("model.language_model."):]
        return name  # mtp.*, lm_head.weight keep their names

    os.makedirs(OUT, exist_ok=True)
    new_map, merged_n = {}, 0
    for shard in sorted(shards):
        out = {}
        with safe_open(os.path.join(BASE, shard), "pt") as f:
            for name in shards[shard]:
                if drop(name):
                    continue
                t = f.get_tensor(name)
                if name in lora:
                    a, b = lora.pop(name)
                    t = (t.to(torch.float32) + SCALE * (b @ a)).to(t.dtype)
                    merged_n += 1
                out[rename(name)] = t
        save_file(out, os.path.join(OUT, shard), metadata={"format": "pt"})
        for name in shards[shard]:
            if not drop(name):
                new_map[rename(name)] = shard
        print(f"[merge] {shard} done ({len(out)} tensors)", flush=True)

    if lora:
        raise RuntimeError(f"unmatched lora targets: {sorted(lora)[:8]}")
    print(f"[merge] applied {merged_n} lora deltas", flush=True)

    json.dump(
        {"metadata": {"total_size": idx.get("metadata", {}).get("total_size", 0)}, "weight_map": new_map},
        open(os.path.join(OUT, "model.safetensors.index.json"), "w"),
        indent=2,
    )

    # ---- config: promote text_config -> root ---------------------------------
    cfg = json.load(open(os.path.join(BASE, "config.json")))
    tc = cfg["text_config"]
    tc["architectures"] = ["Qwen3_5MoeForCausalLM"]
    json.dump(tc, open(os.path.join(OUT, "config.json"), "w"), indent=2)
    for extra in ("generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(BASE, extra)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUT, extra))
    chat = os.path.join(ADAPTER, "chat_template.jinja")
    if os.path.exists(chat):
        shutil.copy(chat, os.path.join(OUT, "chat_template.jinja"))
    print("[merge] done", flush=True)


if __name__ == "__main__":
    main()
