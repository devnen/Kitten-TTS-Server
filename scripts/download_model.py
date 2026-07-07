#!/usr/bin/env python3
"""Download and bake the configured KittenTTS default model into the image.

The baked layout is intentionally flat and deterministic:
    /app/model_cache/baked/<repo-name>/config.json
    /app/model_cache/baked/<repo-name>/<onnx-file-from-config>
    /app/model_cache/baked/<repo-name>/<voices-file-from-config>
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_REPO_ID = "KittenML/kitten-tts-nano-0.1"
DEFAULT_BAKED_ROOT = "/app/model_cache/baked"


def _repo_name(repo_id: str) -> str:
    return repo_id.rstrip("/").split("/")[-1]


def _copy_to_baked(src: str, dest_dir: Path, filename: str) -> Path:
    dest = dest_dir / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def main() -> None:
    repo_id = os.environ.get("KITTEN_TTS_DEFAULT_MODEL", DEFAULT_REPO_ID)
    baked_root = Path(os.environ.get("KITTEN_TTS_BAKED_MODEL_ROOT", DEFAULT_BAKED_ROOT))
    cache_dir = Path(os.environ.get("KITTEN_TTS_BUILD_HF_CACHE", "/tmp/kitten_tts_hf_cache"))
    baked_dir = baked_root / _repo_name(repo_id)

    print(f"Baking KittenTTS model {repo_id} into {baked_dir}")
    baked_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    config_src = hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir=str(cache_dir))
    config_dest = _copy_to_baked(config_src, baked_dir, "config.json")

    with open(config_dest, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    model_filename = model_config.get("model_file")
    voices_filename = model_config.get("voices")
    missing_keys = [key for key, value in {"model_file": model_filename, "voices": voices_filename}.items() if not value]
    if missing_keys:
        raise RuntimeError(f"config.json is missing required key(s): {', '.join(missing_keys)}")

    for filename in (model_filename, voices_filename):
        src = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(cache_dir))
        dest = _copy_to_baked(src, baked_dir, filename)
        print(f"Baked {filename} -> {dest}")

    print(f"Model bake complete: {baked_dir}")


if __name__ == "__main__":
    main()
