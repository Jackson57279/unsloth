#!/usr/bin/env python3
"""Upload merged safetensors + gguf/ + card to the HF repo."""
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # before huggingface_hub import

from huggingface_hub import HfApi

WORK = os.path.expanduser("~/qwen36-distill")
REPO = "freakyskittle/qwen36-35b-a3b-distill"

api = HfApi()
api.create_repo(REPO, repo_type="model", exist_ok=True)
print(f"[hub] repo {REPO} ready")


def upload(folder, path_in_repo):
    for attempt in range(3):
        try:
            api.upload_folder(repo_id=REPO, repo_type="model",
                              folder_path=folder, path_in_repo=path_in_repo)
            return
        except Exception as e:
            print(f"[hub] upload attempt {attempt + 1} failed: {e}", flush=True)
            if attempt == 2:
                raise


# GGUFs first — most useful, available sooner
upload(os.path.join(WORK, "gguf-out"), "gguf")
print("[hub] gguf/ uploaded")
upload(os.path.join(WORK, "merged"), "")
print("[hub] merged safetensors + README uploaded")
print("[hub] done:", f"https://huggingface.co/{REPO}")
