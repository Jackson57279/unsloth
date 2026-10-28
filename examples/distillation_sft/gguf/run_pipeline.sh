#!/usr/bin/env bash
# Merge -> GGUF -> quantize -> upload pipeline for qwen36-35b-a3b-distill.
# Runs fully on this machine (CPU). Log: pipeline.log
# Companion scripts live in the same dir: merge_lora_shards.py, make_card.py, upload_repo.py
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1   # multi-chunk transfer; install hf_transfer first
WORK="$HOME/qwen36-distill"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$WORK/base" "$WORK/adapter" "$WORK/merged" "$WORK/gguf-out"
cd "$WORK"

step(){ printf '\n##### %s #####\n' "$*"; date; }

step "python deps"
[ -d .venv ] || uv venv --python 3.12 .venv
# shellcheck disable=SC1091
. .venv/bin/activate
uv pip install -q torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -q safetensors "huggingface_hub[cli]" numpy

step "download base (~72GB)"
hf download huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated --local-dir "$WORK/base"

step "download adapter"
hf download freakyskittle/qwen36-35b-a3b-distill-lora-l4 --local-dir "$WORK/adapter"

step "merge LoRA -> text-only safetensors"
if [ -f "$WORK/merged/model.safetensors.index.json" ] && [ -f "$WORK/merged/config.json" ]; then
    echo "[skip] merged/ already exists"
else
    python3 "$SCRIPT_DIR/merge_lora_shards.py" "$WORK/base" "$WORK/adapter" "$WORK/merged"
fi
rm -rf "$WORK/base/.cache"

step "clone + build llama.cpp"
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
uv pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
cmake -B llama.cpp/build -S llama.cpp -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build -j"$(nproc)" --target llama-quantize

step "convert f16 master (MTP embedded as nextn layer)"
if [ -f "$WORK/gguf-out/Qwen36-35B-A3B-Distill-f16.gguf" ]; then
    echo "[skip] f16 master exists"
else
    python3 llama.cpp/convert_hf_to_gguf.py "$WORK/merged" \
        --outfile "$WORK/gguf-out/Qwen36-35B-A3B-Distill-f16.gguf" --outtype f16
fi

step "export standalone MTP draft gguf"
if ls "$WORK/gguf-out/"mtp-*.gguf >/dev/null 2>&1; then
    echo "[skip] mtp draft exists"
else
    python3 llama.cpp/convert_hf_to_gguf.py "$WORK/merged" \
        --mtp --outfile "$WORK/gguf-out/" --outtype f16
fi

step "quantize"
for q in Q4_K_M Q4_K_S IQ4_XS Q6_K Q8_0 TQ1_0 TQ2_0; do
    if [ -f "$WORK/gguf-out/Qwen36-35B-A3B-Distill-$q.gguf" ]; then
        echo "[skip] $q exists"
        continue
    fi
    echo "-- $q"
    ./llama.cpp/build/bin/llama-quantize \
        "$WORK/gguf-out/Qwen36-35B-A3B-Distill-f16.gguf" \
        "$WORK/gguf-out/Qwen36-35B-A3B-Distill-$q.gguf" "$q"
done
rm -f "$WORK/gguf-out/Qwen36-35B-A3B-Distill-f16.gguf"  # master is intermediate only

step "model card"
python3 "$SCRIPT_DIR/make_card.py"

step "upload to HF"
python3 "$SCRIPT_DIR/upload_repo.py"

step "PIPELINE DONE"
