"""
Copy the Qwen2.5 tokenizer files into out/tokenizer/ so swift-transformers'
AutoTokenizer.from(modelFolder:) can find them at runtime.

VibeVoice-Realtime-0.5B reuses Qwen2.5-0.5B's tokenizer verbatim, so we pull
it from `Qwen/Qwen2.5-0.5B` on the Hub.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

OUT = Path(__file__).resolve().parents[1] / "out" / "tokenizer"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Downloading tokenizer files from Qwen/Qwen2.5-0.5B…")
    local = snapshot_download(
        repo_id="Qwen/Qwen2.5-0.5B",
        allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "generation_config.json"],
    )
    local = Path(local)
    for f in local.iterdir():
        if f.is_file():
            shutil.copy2(f, OUT / f.name)
            print(f"  {f.name}")
    print(f"Tokenizer bundled at {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
