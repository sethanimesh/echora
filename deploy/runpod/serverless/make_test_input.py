"""Create a RunPod SDK test input from a local audio file."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path


if len(sys.argv) != 3:
    raise SystemExit("usage: make_test_input.py AUDIO OUTPUT_JSON")
audio, output = map(Path, sys.argv[1:])
payload = {
    "input": {
        "audio_base64": base64.b64encode(audio.read_bytes()).decode("ascii"),
        "filename": audio.name,
        "beams": 5,
    }
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(output)
