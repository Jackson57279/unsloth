# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Dataset normalizers for distillation SFT on Qwen3.6-35B-A3B.

Every loader produces rows of {"messages": [...], "tools": [...]?} so a single
chat-template pass can render the whole mix. Security / agentic / benchmark
sources each keep their own normalizer; add a spec + a function to extend.

Sources
-------
- freakyskittle/glm-5.3-distillation  (PRIVATE: needs HF token)
    Agentic GLM-5.3 distillation traces. `messages` with role/content plus
    optional reasoning_content / tool_calls / name fields - the Qwen3.6 chat
    template renders all of them natively.
- ciTy1504/security-agentic-sft-corpus
    Agentic security trajectories (cybench / autopenbench / bughunter /
    cve_factory / cti). `turns` list of {role, text}.
- Leopo1d/OpenVul_Distilled_Vulnerability_Reasoning_CoTs_from_DeepSeek-R1-0528
    R1-distilled C/C++ vulnerability reasoning CoTs. `query` (chat list) +
    `multi_generations` (list of <think> completions); we take generation 0.
- SecCoderX/SecCoderX_Reasoning_Vulnerability_Detection_SFT_Cold_Start_Dataset
    CWE vuln-detection SFT pairs. instruction/output, output starts with
    <think> - kept verbatim so the model learns think->answer format.
- TrueNix/ctf-solver-dataset
    CTF solver SFT `messages` (chat format already).
- whitecircle/swe-rebench-v2-glm-5.1-pi-agent-successful-traces
    Successful GLM-5.1 SWE-agent traces. `messages` + `tools` columns.
- Benchmarks (train splits only, formatted as QA chat):
    openai/gsm8k, DigitalLearningGmbH/MATH-lighteval,
    allenai/ai2_arc (ARC-Challenge), cais/mmlu (auxiliary_train).
"""

from __future__ import annotations

import json
import random
from typing import Any, Callable, Dict, List, Optional


def _clean_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Coerce tool_calls into the shape the Qwen chat template expects.

    The template iterates ``tool_call.arguments | items`` and reads
    ``tool_call.name`` directly. OpenAI-style traces instead nest
    ``{"function": {"name": ..., "arguments": ...}}`` and encode
    ``arguments`` as a JSON *string* - both crash the Jinja render.
    """
    out = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        m = dict(m)
        tcs = m.get("tool_calls")
        if tcs:
            fixed = []
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                if isinstance(fn, dict):
                    tc = {
                        **tc,
                        "name": tc.get("name") or fn.get("name"),
                        "arguments": tc.get("arguments", fn.get("arguments")),
                    }
                args = tc.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = None
                if args is not None and not isinstance(args, dict):
                    args = {"value": args}
                fixed.append({**tc, "arguments": args if args is not None else {}})
            m["tool_calls"] = fixed
        out.append(m)
    return out

# ---------------------------------------------------------------------------
# Mix specification. `max_rows` caps each source BEFORE rendering; `weight`
# scales how many rows are actually kept relative to `max_rows` (fraction).
# Tune for your GPU budget - a T4 run should stay near the defaults below.
# ---------------------------------------------------------------------------

DATASET_SPECS: List[Dict[str, Any]] = [
    # --- your private distillation set (HF token required) ---
    dict(
        name = "freakyskittle/glm-5.3-distillation",
        kind = "messages",
        max_rows = 3752,
        desc = "GLM-5.3 agentic distillation (private)",
    ),
    # --- security reasoning / traces ---
    dict(
        name = "ciTy1504/security-agentic-sft-corpus",
        kind = "city1504_turns",
        loader = "jsonl_files",
        max_rows = 2500,
        desc = "agentic security trajectories (cybench/autopenbench/CVE)",
    ),
    dict(
        name = "Leopo1d/OpenVul_Distilled_Vulnerability_Reasoning_CoTs_from_DeepSeek-R1-0528",
        kind = "openvul",
        max_rows = 2000,
        desc = "R1-distilled vulnerability reasoning CoTs",
    ),
    dict(
        name = "SecCoderX/SecCoderX_Reasoning_Vulnerability_Detection_SFT_Cold_Start_Dataset",
        kind = "instruction_output",
        max_rows = 1500,
        desc = "CWE vulnerability-detection reasoning",
    ),
    dict(
        name = "TrueNix/ctf-solver-dataset",
        kind = "messages",
        max_rows = 1500,
        desc = "CTF solver traces",
    ),
    # --- other agent traces ---
    dict(
        name = "whitecircle/swe-rebench-v2-glm-5.1-pi-agent-successful-traces",
        kind = "messages",
        max_rows = 750,
        desc = "GLM-5.1 SWE-agent successful traces",
    ),
    # --- benchmarks (train splits, formatted as chat) ---
    dict(
        name = "openai/gsm8k", config = "main", split = "train",
        kind = "gsm8k", max_rows = 1500,
        desc = "GSM8K math word problems",
    ),
    dict(
        name = "DigitalLearningGmbH/MATH-lighteval", config = "default", split = "train",
        kind = "math_lighteval", max_rows = 1500,
        desc = "MATH competition problems",
    ),
    dict(
        name = "allenai/ai2_arc", config = "ARC-Challenge", split = "train",
        kind = "arc", max_rows = 800,
        desc = "ARC-Challenge science MCQA",
    ),
    dict(
        name = "cais/mmlu", config = "all", split = "auxiliary_train",
        kind = "mmlu", max_rows = 1500,
        desc = "MMLU auxiliary train QA",
    ),
]

# ---------------------------------------------------------------------------
# Tolerant loader: `datasets`'s json builder aborts the whole split when any
# file has an invalid surrogate pair (seen in ciTy1504/autopenbench). Reading
# the .jsonl files ourselves drops only the bad lines.
# ---------------------------------------------------------------------------

def _strip_surrogates(o):
    if isinstance(o, str):
        return o.encode("utf-8", errors = "replace").decode("utf-8")
    if isinstance(o, list):
        return [_strip_surrogates(x) for x in o]
    if isinstance(o, dict):
        return {k: _strip_surrogates(v) for k, v in o.items()}
    return o


def _load_jsonl_files_tolerant(spec, token = None):
    from datasets import Dataset
    from huggingface_hub import hf_hub_download, list_repo_files

    repo = spec["name"]
    files = [
        f for f in list_repo_files(repo, repo_type = "dataset", token = token)
        if f.endswith((".jsonl", ".json"))
    ]
    rows = []
    for f in files:
        path = hf_hub_download(repo, f, repo_type = "dataset", token = token)
        if f.endswith(".json"):
            try:
                obj = json.load(open(path, encoding = "utf-8", errors = "replace"))
                rows.extend(_strip_surrogates(r) for r in
                            (obj if isinstance(obj, list) else [obj]))
            except Exception:
                pass
            continue
        with open(path, encoding = "utf-8", errors = "replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(_strip_surrogates(json.loads(line)))
                except json.JSONDecodeError:
                    continue
    if not rows:
        raise RuntimeError(f"no .jsonl rows loaded from {repo}")
    # Plain list, not Dataset: the files' schemas disagree (e.g. `score`
    # is int in one file and 'payments+app-owners' in another), so a
    # uniform pyarrow table can't be built. Normalizers only need `turns`.
    return rows


# ---------------------------------------------------------------------------
# Normalizers: row -> {"messages": [...], "tools": [...]?} or None to drop.
# ---------------------------------------------------------------------------

def _msg(role: str, content: Optional[str], **extra) -> Dict[str, Any]:
    m = {"role": role, "content": content or ""}
    m.update({k: v for k, v in extra.items() if v is not None})
    return m


def norm_messages(row, spec):
    """Already chat-shaped (`messages` column). Keeps reasoning_content /
    tool_calls / name / tool_call_id - the Qwen3.6 template reads them."""
    msgs = row.get("messages")
    if not msgs or not isinstance(msgs, (list, tuple)):
        return None
    out = {"messages": [dict(m) for m in msgs if isinstance(m, dict) and m.get("role")]}
    if not out["messages"]:
        return None
    if row.get("tools"):
        out["tools"] = row["tools"]
    return out


def norm_city1504_turns(row, spec):
    """turns: [{role, text}] -> messages [{role, content}]."""
    turns = row.get("turns")
    if not turns:
        return None
    msgs = [
        _msg(t.get("role", "user"), t.get("text"))
        for t in turns
        if isinstance(t, dict) and t.get("role")
    ]
    if len(msgs) < 2:
        return None
    return {"messages": msgs}


def norm_openvul(row, spec):
    """query: chat list; multi_generations: list of CoT strings -> append
    generation[0] as the assistant turn. The <think>...</think> text inside
    the generation is re-split by the Qwen3.6 template into reasoning."""
    query = row.get("query")
    gens = row.get("multi_generations")
    if not query or not gens:
        return None
    msgs = [dict(m) for m in query if isinstance(m, dict) and m.get("role")]
    gen = gens[0]
    if not isinstance(gen, str) or not gen.strip():
        return None
    msgs.append(_msg("assistant", gen.strip()))
    return {"messages": msgs}


def norm_instruction_output(row, spec):
    ins, out = row.get("instruction"), row.get("output")
    if not ins or not out:
        return None
    return {
        "messages": [
            _msg("user", ins.strip()),
            _msg("assistant", out.strip()),
        ]
    }


def norm_gsm8k(row, spec):
    q, a = row.get("question"), row.get("answer")
    if not q or not a:
        return None
    # '#### 42' -> explicit final-answer phrasing the model can reuse.
    a = a.replace("####", "The final answer is").strip()
    return {"messages": [_msg("user", q.strip()), _msg("assistant", a)]}


def norm_math_lighteval(row, spec):
    q, a = row.get("problem"), row.get("solution")
    if not q or not a:
        return None
    return {"messages": [_msg("user", q.strip()), _msg("assistant", a.strip())]}


def _mcqa_messages(question: str, choices: List[str], answer_idx: int):
    labels = "ABCDEFGH"
    lines = [question.strip(), ""]
    for i, c in enumerate(choices):
        lines.append(f"{labels[i]}. {c}")
    lines.append("\nAnswer with the letter of the correct option.")
    ans = f"{labels[answer_idx]}. {choices[answer_idx]}"
    return [_msg("user", "\n".join(lines)), _msg("assistant", ans)]


def norm_arc(row, spec):
    q, ch, key = row.get("question"), row.get("choices"), row.get("answerKey")
    if not q or not ch or not key:
        return None
    texts, labels = ch.get("text", []), ch.get("label", [])
    if key not in labels:
        return None
    msgs = _mcqa_messages(q, list(texts), labels.index(key))
    return {"messages": msgs}


def norm_mmlu(row, spec):
    q, ch, a = row.get("question"), row.get("choices"), row.get("answer")
    if not q or not ch or a is None or int(a) >= len(ch):
        return None
    msgs = _mcqa_messages(q, list(ch), int(a))
    return {"messages": msgs}


NORMALIZERS: Dict[str, Callable] = {
    "messages": norm_messages,
    "city1504_turns": norm_city1504_turns,
    "openvul": norm_openvul,
    "instruction_output": norm_instruction_output,
    "gsm8k": norm_gsm8k,
    "math_lighteval": norm_math_lighteval,
    "arc": norm_arc,
    "mmlu": norm_mmlu,
}


# ---------------------------------------------------------------------------
# Mix builder.
# ---------------------------------------------------------------------------

def build_chat_dataset(
    tokenizer,
    specs: Optional[List[Dict[str, Any]]] = None,
    max_seq_length: int = 2048,
    seed: int = 3407,
    token = None,
    verbose: bool = True,
):
    """Load, normalize, chat-template-render and mix all sources.

    Returns a datasets.Dataset with a single `text` column (fully rendered),
    filtered to <= max_seq_length tokens and shuffled.
    """
    from datasets import load_dataset, concatenate_datasets

    specs = specs or DATASET_SPECS
    rendered = []

    for spec in specs:
        kind = spec["kind"]
        normalize = NORMALIZERS[kind]
        try:
            if spec.get("loader") == "jsonl_files":
                ds = _load_jsonl_files_tolerant(spec, token = token)
            else:
                split = spec.get("split", "train")
                if spec.get("slice"):
                    # Prefix-slice inside the split string so `datasets` only
                    # generates what the cap can use (MMLU aux_train is ~100k
                    # rows; generating all of it for a 64-row cap wastes ~1h).
                    n = min(spec.get("max_rows", 64) * 8, 8000)
                    split = f"{split}[:{n}]"
                ds = load_dataset(
                    spec["name"],
                    spec.get("config"),
                    split = split,
                    token = token,
                )
        except Exception as e:
            print(f"[data] SKIP {spec['name']}: {e}")
            continue

        n = min(spec.get("max_rows", len(ds)), len(ds))
        if isinstance(ds, list):
            rnd = random.Random(seed)
            rnd.shuffle(ds)
            rows = ds[:n]
        else:
            rows = ds.shuffle(seed = seed).select(range(n)) if len(ds) > n else ds

        kept = []
        for row in rows:
            out = normalize(row, spec)
            if out is not None:
                kept.append(out)
        if not kept:
            print(f"[data] SKIP {spec['name']}: normalizer kept 0 rows")
            continue

        texts = []
        skipped = 0
        # Pre-render cap: rows far beyond max_seq_length would be filtered
        # post-tokenize anyway; rendering a 20k-message trace first costs GBs
        # of RAM. ~4 chars/token -> 6x chars is a strict superset of the cap.
        char_cap = max_seq_length * 6
        for item in kept:
            messages = _clean_messages(item["messages"])
            chars = sum(
                len(v) for m in messages for v in m.values()
                if isinstance(v, str)
            )
            if item.get("tools"):
                chars += len(json.dumps(item["tools"]))
            if chars > char_cap:
                skipped += 1
                continue
            try:
                texts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tools = item.get("tools"),
                        tokenize = False,
                        add_generation_prompt = False,
                        # keep <think> on every assistant turn, not just the
                        # last one - distillation needs the full trace.
                        preserve_thinking = True,
                    )
                )
            except TypeError:
                # Older transformers without preserve_thinking kwarg.
                try:
                    texts.append(
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize = False,
                            add_generation_prompt = False,
                        )
                    )
                except Exception:
                    skipped += 1
            except Exception:
                # Unrenderable row (bad tool_calls, odd content) - skip it
                # rather than killing the whole mix.
                skipped += 1
        rendered.append(texts)
        if verbose:
            print(
                f"[data] {spec['name']}: {len(texts)} rows"
                + (f" (+{skipped} unrenderable skipped)" if skipped else "")
                + f" ({spec['desc']})"
            )

    if not rendered:
        raise RuntimeError("No dataset rows survived normalization.")

    from datasets import Dataset

    flat = [t for texts in rendered for t in texts]
    ds = Dataset.from_dict({"text": flat})

    # Drop rows that do not fit: truncating agent traces mid-thought teaches
    # the model to never finish an answer, so filter rather than truncate.
    def fits(batch):
        return {
            "fits": [
                len(ids) <= max_seq_length
                for ids in tokenizer(batch["text"], add_special_tokens = False)[
                    "input_ids"
                ]
            ]
        }

    before = len(ds)
    # num_proc stays 1: forking workers while a multi-GB model is resident
    # copy-on-write duplicates it and can blow the host cgroup limit.
    ds = ds.map(fits, batched = True, batch_size = 256)
    ds = ds.filter(lambda x: x["fits"])
    ds = ds.remove_columns("fits")
    if verbose:
        print(
            f"[data] total {before} -> {len(ds)} rows fit in "
            f"{max_seq_length} tokens"
        )
    return ds.shuffle(seed = seed)
