"""Stage the fixed foundation gate with cloud-valid audio paths."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cloud-root", type=Path, default=Path("/workspace/echora/gate"))
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.manifest.read_text().splitlines()
        if line.strip()
    ]
    audio_dir = args.output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for row in rows:
        source = Path(row["audio_filepath"])
        filename = f"{row['id']}{source.suffix.lower()}"
        destination = audio_dir / filename
        shutil.copy2(source, destination)
        cloud_row = dict(row)
        cloud_row["audio_filepath"] = str(args.cloud_root / "audio" / filename)
        staged.append(cloud_row)

    output_manifest = args.output / "foundation_gate_400.jsonl"
    output_manifest.write_text("".join(json.dumps(row) + "\n" for row in staged))
    total_bytes = sum(path.stat().st_size for path in audio_dir.iterdir())
    print(f"staged {len(staged)} utterances ({total_bytes} bytes) at {args.output}")


if __name__ == "__main__":
    main()
