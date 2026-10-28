"""Capture target hidden states at DFlash tap layers for the distillation corpus.

Loads the merged model bf16 on the A100, renders training rows through the
chat template, tokenizes, runs forward with output_hidden_states, and stores
concatenated tap-layer hiddens + token ids as .pt shards.

usage: capture.py <n_rows_per_src> <max_len> <out_dir> [profile]
"""
import os, sys, time, json, random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "freakyskittle/qwen36-35b-a3b-distill"
TAPS = [1, 10, 19, 28, 37]          # build_target_layer_ids(40, 5); +1 offset applied at index time
N_PER_SRC = int(sys.argv[1]) if len(sys.argv) > 1 else 120
MAX_LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
OUT = sys.argv[3] if len(sys.argv) > 3 else "/content/dflash/data"

sys.path.insert(0, "/content/dflash")
import data_prep  # noqa: E402


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    print("[cap] tokenizer ok", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    print("[cap] model loaded", flush=True)

    # the draft borrows the target's embedding + lm_head at train/serve time
    emb = model.get_input_embeddings().weight.detach().half().cpu()
    head = model.get_output_embeddings().weight.detach().half().cpu()
    torch.save({"embed": emb, "lm_head": head}, f"{OUT}/target_parts.pt")
    print(f"[cap] saved target parts: emb {tuple(emb.shape)} head {tuple(head.shape)}", flush=True)

    # --- corpus: data_prep mix, scaled to N_PER_SRC per source ---------------
    import copy
    specs = copy.deepcopy(data_prep.DATASET_SPECS)
    for spec in specs:
        spec["max_rows"] = N_PER_SRC
    ds = data_prep.build_chat_dataset(
        tok, specs=specs, max_seq_length=MAX_LEN,
        token=os.environ.get("HF_TOKEN"))
    texts = [r["text"] for r in ds]
    print(f"[cap] corpus: {len(texts)} rendered rows", flush=True)

    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    shard, shard_i, done = [], 0, 0
    with torch.inference_mode():
        for text in texts:
            ids = tok(text, add_special_tokens=False).input_ids
            if not (64 <= len(ids) <= MAX_LEN):
                continue
            input_ids = torch.tensor([ids], device="cuda")
            out = model(input_ids, output_hidden_states=True)
            hs = out.hidden_states  # tuple(len 41): [0]=emb, [i+1]=layer i out
            taps = torch.cat([hs[t + 1][0] for t in TAPS], dim=-1)  # [L, 5*2048]
            shard.append({
                "ids": ids,
                "target_hidden": taps.half().cpu(),  # fp16 to halve disk
            })
            done += 1
            if done % 25 == 0:
                el = time.time() - t0
                print(f"[cap] {done}/{len(texts)} ({el/done:.1f}s/seq)", flush=True)
            if len(shard) >= 50:
                torch.save(shard, f"{OUT}/shard_{shard_i:03d}.pt")
                shard, shard_i = [], shard_i + 1
        if shard:
            torch.save(shard, f"{OUT}/shard_{shard_i:03d}.pt")
    json.dump({"taps": TAPS, "n_seqs": done, "max_len": MAX_LEN},
              open(f"{OUT}/meta.json", "w"))
    print(f"[cap] done: {done} seqs, {shard_i + 1} shards, {(time.time()-t0)/60:.0f}min", flush=True)


if __name__ == "__main__":
    main()
