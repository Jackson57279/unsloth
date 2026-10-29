# DFlash2 draft training for qwen36-35b-a3b-distill

Trains a [`DFlash2DraftModel`](https://github.com/z-lab/dflash) block-diffusion
draft for the merged target (`freakyskittle/qwen36-35b-a3b-distill`) from
scratch — no upstream checkpoint needed.

Pipeline:

1. **Capture** — `capture_hiddens.py <rows_per_src> <max_len> <out_dir>`
   loads the merged model bf16, renders the `data_prep.py` mix through the chat
   template, and stores token ids + concatenated hidden states at tap layers
   `[1, 10, 19, 28, 37]` (= `build_target_layer_ids(40, 5)`) per sequence, plus
   the target's frozen `embed`/`lm_head` (`target_parts.pt`). ~0.5 s/seq on an
   A100.

2. **Train** — `train_draft.py <data_dir> <out_dir> [--steps N --bs B --lr LR
   --save_every K]` builds a 5-layer bidirectional draft (hidden 2048, block 8,
   mask 248077, dynamic grouped causal conv k=2/g=16, candidate selector
   rank 256 top-k 16) and optimizes masked-block cross-entropy + a
   teacher-forced selector loss. Borrows the target's embedding + LM head at
   train/serve time.

3. **Convert** — llama.cpp supports the arch directly:
   `convert_hf_to_gguf.py draft_dir --target-model-dir merged_repo
   --outfile draft.gguf` (pulls tokenizer + embeddings/head refs from the
   target; `mask_embedding.pt` is unnecessary when the target's mask row is
   trained).

Gotchas discovered the hard way:

- **`GroupedDynamicCausalConv.base_kernel` is `nn.Parameter(torch.empty())`**
  and `_init_weights` never covers it — fresh inits read uninitialized memory,
  which is sometimes NaN bit patterns → nondeterministic NaN forward passes.
  `train_draft.py` initializes it deterministically.
- **Train the draft in fp32.** bf16 params + a 248k-way softmax plateau then
  diverge; fp32 trains cleanly at 0.2–0.6 s/step, then cast to bf16 for the
  checkpoint to match upstream dtype.
- Draft eval metric = **top-k containment** (fraction of masked block
  positions whose true token sits in the draft top-k), which maps to
  speculative acceptance. top-1 alone understates the draft.

Result for this target: 10k steps, loss 8.3 → 3.9, held-out containment
top1 22.6% / top3 34.0% / top8 44.9% / top16 53.5%. Published at
`freakyskittle/qwen36-35b-a3b-distill-dflash2` (+ `...-DFlash2-F16.gguf` in the
`-GGUF` repo).

Resilience: shards upload fine to a private HF dataset repo mid-capture (see
the session's watcher pattern) — Colab session reclaim was survived this way.
