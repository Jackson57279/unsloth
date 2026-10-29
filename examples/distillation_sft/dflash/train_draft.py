"""Train a DFlash2DraftModel against captured target hidden states.

Loss = masked-block CE (all block positions) + selector CE (teacher-forced
candidate path), with the target's frozen embedding producing noise_embedding
and its frozen lm_head producing logits — matching dflash inference exactly.

usage: train_draft.py <data_dir> <out_dir> [--steps N]
"""
import argparse, glob, json, math, os, random, time

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from dflash.model import DFlash2DraftModel

MASK_ID = 248077
BLOCK = 8
CTX = 256
TAPS = [1, 10, 19, 28, 37]
H = 2048
SEL_TOPK = 16
SEL_LAMBDA = 0.5


def build_config():
    return Qwen3Config(
        architectures=["DFlash2DraftModel"],
        model_type="qwen3",
        is_causal=False,
        hidden_size=H,
        num_hidden_layers=5,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=128,
        intermediate_size=7168,
        hidden_act="silu",
        layer_types=["sliding_attention"] * 5,
        sliding_window=2048,
        max_window_layers=5,
        use_sliding_window=True,
        vocab_size=248320,
        num_target_layers=40,
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 10000000, "rope_type": "default"},
        eos_token_id=248044,
        pad_token_id=248044,
        tie_word_embeddings=False,
        dtype="bfloat16",
        auto_map={"AutoModel": "dflash.DFlash2DraftModel"},
        dflash_config={
            "block_size": BLOCK,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": MASK_ID,
            "selector_rank": 256,
            "selector_top_k": SEL_TOPK,
            "target_layer_ids": TAPS,
        },
    )


def make_sample(seq, emb, mask_vec, p):
    """One training sample: ctx hiddens + masked block, at block start p."""
    ids = seq["ids"]
    ctx = seq["target_hidden"][p - CTX:p].to(torch.float32).cuda()  # [W, 5H]
    noise = torch.cat([
        emb[ids[p]].unsqueeze(0),                    # anchor token emb
        mask_vec.unsqueeze(0).expand(BLOCK - 1, -1), # mask emb x (B-1)
    ]).cuda()                                       # [B, H]
    labels = torch.tensor(ids[p + 1:p + 1 + BLOCK], device="cuda")
    positions = torch.arange(p - CTX, p + BLOCK, device="cuda")
    return ctx, noise, labels, positions


def selector_loss(model, hidden, logits, labels, anchor_id):
    """Teacher-forced candidate-path loss (mirrors CandidateSelector.select)."""
    sel = model.candidate_selector
    unary, cand = logits.topk(SEL_TOPK, dim=-1, sorted=False)   # [1,B,K]
    hp = sel.hidden_projection(hidden)                          # [1,B,R]
    preds = torch.cat([torch.tensor([anchor_id], device="cuda"),
                       labels[:-1]])                            # true predecessors
    pair = torch.einsum(
        "br,bkr->bk",
        sel.predecessor_codebook(preds) * hp[0],
        sel.successor_codebook(cand[0]),
    )
    scores = unary[0] + pair                                    # [B,K]
    tgt = (cand[0] == labels[:, None])
    valid = tgt.any(-1)
    if not valid.any():
        return torch.zeros((), device="cuda")
    idx = tgt[valid].float().argmax(-1)
    return F.cross_entropy(scores[valid], idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save_every", type=int, default=0)
    args = ap.parse_args()

    parts = torch.load(f"{args.data_dir}/target_parts.pt")
    emb = parts["embed"].to(torch.float32).cuda()
    head_w = parts["lm_head"].to(torch.float32).cuda()
    lm_head = lambda h: F.linear(h, head_w)  # noqa: E731 — compute_logits calls head(hidden)
    mask_vec = emb[MASK_ID]
    emb.requires_grad_(False)

    seqs = []
    for f in sorted(glob.glob(f"{args.data_dir}/shard_*.pt")):
        seqs.extend(s for s in torch.load(f, weights_only=False)
                    if len(s["ids"]) > CTX + BLOCK + 2)
    random.Random(0).shuffle(seqs)
    n_val = max(8, len(seqs) // 20)
    val, seqs = seqs[:n_val], seqs[n_val:]
    print(f"[train] {len(seqs)} train seqs, {n_val} val", flush=True)

    torch.manual_seed(0)
    model = DFlash2DraftModel(build_config()).to(torch.float32).cuda()
    # GroupedDynamicCausalConv.base_kernel is nn.Parameter(torch.empty(...)):
    # _init_weights never touches it, so fresh inits read uninitialized memory
    # (nondeterministic NaNs). Initialize deterministically.
    with torch.no_grad():
        for layer in model.layers:
            for conv in (layer.attention_conv, layer.mlp_conv):
                torch.nn.init.normal_(conv.base_kernel, mean=0.0, std=0.02)
    model.train()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[train] draft params: {n_params:.0f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(50, args.steps // 20)

    def lr_mult(s):
        if s < warm:
            return (s + 1) / warm
        t = (s - warm) / max(1, args.steps - warm)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * t)) / 2

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)

    def batch(rng):
        ctx_l, noise_l, lab_l, pos_l, anc = [], [], [], [], []
        for _ in range(args.bs):
            s = rng.choice(seqs)
            p = rng.randint(CTX, len(s["ids"]) - BLOCK - 1)
            c, n, l, pos = make_sample(s, emb, mask_vec, p)
            ctx_l.append(c); noise_l.append(n); lab_l.append(l)
            pos_l.append(pos); anc.append(s["ids"][p])
        return (torch.stack(ctx_l), torch.stack(noise_l), torch.stack(lab_l),
                torch.stack(pos_l), anc)

    rng = random.Random(1)
    nan_skips = 0
    t0, log_loss, log_sel = time.time(), 0.0, 0.0
    for step in range(1, args.steps + 1):
        ctx, noise, labels, positions, anchors = batch(rng)
        hidden = model(noise_embedding=noise, target_hidden=ctx,
                       position_ids=positions)
        logits = model.compute_logits(hidden, lm_head)
        loss = F.cross_entropy(
            logits[0].float(), labels[0], reduction="sum")
        sel = selector_loss(model, hidden, logits, labels[0], anchors[0])
        for b in range(1, ctx.shape[0]):
            loss = loss + F.cross_entropy(
                logits[b].float(), labels[b], reduction="sum")
            sel = sel + selector_loss(model, hidden[b:b+1], logits[b:b+1],
                                      labels[b], anchors[b])
        loss = loss / (ctx.shape[0] * BLOCK)
        loss = loss + SEL_LAMBDA * sel / ctx.shape[0]
        if not torch.isfinite(loss):
            nan_skips += 1
            if nan_skips > 30:
                raise RuntimeError(f"loss not finite for {nan_skips} steps")
            opt.zero_grad(set_to_none=True)
            continue
        nan_skips = 0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        log_loss += loss.item(); log_sel += sel.item()
        if step % 50 == 0:
            el = time.time() - t0
            print(f"[train] {step}/{args.steps} loss {log_loss/50:.3f} "
                  f"sel {log_sel/50:.3f} ({el/step:.2f}s/st)", flush=True)
            log_loss, log_sel = 0.0, 0.0
        if args.save_every and step % args.save_every == 0 and step < args.steps:
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save({"step": step, "model": model.state_dict(),
                        "opt": opt.state_dict()},
                       f"{args.out_dir}/checkpoint.pt")
            print(f"[train] checkpoint saved at step {step}", flush=True)

    # save first: eval must never be able to lose the trained weights
    os.makedirs(args.out_dir, exist_ok=True)
    model.config.auto_map = {"AutoModel": "dflash.DFlash2DraftModel"}
    model = model.to(torch.bfloat16)
    head_w = head_w.to(torch.bfloat16)
    model.save_pretrained(args.out_dir)
    print(f"[train] saved to {args.out_dir}", flush=True)

    # --- eval: block containment at k=1/3/8/16 on held-out seqs --------------
    model.eval()
    hits = {1: 0, 3: 0, 8: 0, 16: 0}
    n = 0
    with torch.inference_mode():
        for s in val:
            mid = (len(s["ids"]) - BLOCK - 1) // 2
            for p in dict.fromkeys([CTX, max(CTX, mid)]):
                if p >= len(s["ids"]) - BLOCK - 1:
                    continue
                c, nse, lab, pos = make_sample(s, emb, mask_vec, p)
                h = model(noise_embedding=nse.to(model.dtype)[None],
                          target_hidden=c.to(model.dtype)[None],
                          position_ids=pos[None])
                lg = model.compute_logits(h, lm_head)[0].float()
                for k in hits:
                    hits[k] += (lg.topk(k, -1).indices == lab[:, None]) \
                        .any(-1).sum().item()
                n += BLOCK
    msg = " ".join(f"top{k} {v/n:.1%}" for k, v in hits.items())
    print(f"[eval] block containment: {msg} on {n} positions", flush=True)


if __name__ == "__main__":
    main()
