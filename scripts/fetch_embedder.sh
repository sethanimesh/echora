#!/usr/bin/env bash
#
# Fetch the sentence encoder used for personal-history retrieval.
#
# The bundle follows the same discipline as the recognizer: weights pinned to one
# revision, hashes written down, and nothing downloaded at request time. Without
# it Echora still runs -- the personal layer reports itself unavailable and every
# message is produced exactly as it was before that layer existed.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="sentence-transformers/all-MiniLM-L6-v2"
REVISION="c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
BUNDLE="$ROOT/models/echora-minilm-l6-v2"

[ -x "$ROOT/.venv/bin/python" ] || { echo "Run scripts/setup_local.sh first" >&2; exit 1; }

"$ROOT/.venv/bin/python" - "$REPO" "$REVISION" "$BUNDLE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo, revision, bundle = sys.argv[1], sys.argv[2], Path(sys.argv[3])
encoder = bundle / "encoder"
encoder.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo,
    revision=revision,
    local_dir=str(encoder),
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ],
)

digests = {}
for name in sorted(p.name for p in encoder.iterdir() if p.is_file()):
    digests[name] = hashlib.sha256((encoder / name).read_bytes()).hexdigest()

(bundle / "CHECKSUMS.sha256").write_text(
    "".join(f"{digest}  encoder/{name}\n" for name, digest in digests.items()), encoding="utf-8"
)
(bundle / "manifest.json").write_text(
    json.dumps(
        {
            "name": "echora-minilm-l6-v2",
            "foundation": repo,
            "foundation_revision": revision,
            "pooling": "mean",
            "normalize": True,
            "dimension": json.loads((encoder / "config.json").read_text())["hidden_size"],
            "max_tokens": 128,
            "purpose": "Retrieval over one speaker's accepted messages. Never used for ASR.",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(f"ECHORA_EMBEDDER_READY {bundle}")
PY

cd "$BUNDLE" && shasum -a 256 -c CHECKSUMS.sha256
