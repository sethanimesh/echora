"""Record a clip from the Mac microphone into research/benchmarks/clips/.

    python record.py                          # timestamped name, stop with ENTER
    python record.py water                    # saves clips/water.wav
    python record.py water -t "i would like some water"
    python record.py --list                   # show available input devices
    python record.py water -d 2               # use input device index 2

The -t text is stored next to the clip as a .txt sidecar, and every run_*.py
script prints it above the transcript so you can see what the model got wrong.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime

from common import CLIPS_DIR, SAMPLE_RATE


def audio_devices() -> list[str]:
    proc = subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    )
    # ffmpeg prints the device list to stderr and then exits non-zero. That is normal.
    devices, printing = [], False
    for line in proc.stderr.splitlines():
        if "AVFoundation audio devices" in line:
            printing = True
        elif "AVFoundation video devices" in line:
            printing = False
        elif printing and "] [" in line:
            devices.append(line.split("] ", 1)[-1])
    return devices


def default_device() -> int:
    """Prefer the built-in mic. Index 0 is often a paired iPhone, which will not record."""
    for index, name in enumerate(audio_devices()):
        if "MacBook" in name or "Built-in" in name:
            return index
    return 0


def record(path, device_index: int) -> None:
    cmd = [
        "ffmpeg", "-y", "-v", "quiet",
        "-f", "avfoundation", "-i", f":{device_index}",
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    print("  recording... press ENTER to stop")
    try:
        input()
    except KeyboardInterrupt:
        pass
    # 'q' is ffmpeg's graceful stop; it finalises the WAV header before exiting.
    try:
        proc.stdin.write(b"q")
        proc.stdin.flush()
        proc.wait(timeout=5)
    except (BrokenPipeError, subprocess.TimeoutExpired):
        proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", nargs="?", help="file name without extension (default: timestamp)")
    parser.add_argument("-t", "--text", help="the sentence you intend to say, saved as a reference")
    parser.add_argument("-d", "--device-index", type=int, help="ffmpeg avfoundation audio device index")
    parser.add_argument("--list", action="store_true", help="list input devices and exit")
    args = parser.parse_args()

    if args.list:
        for index, name in enumerate(audio_devices()):
            print(f"  [{index}] {name}")
        return

    device_index = args.device_index if args.device_index is not None else default_device()

    CLIPS_DIR.mkdir(exist_ok=True)
    name = args.name or datetime.now().strftime("%Y%m%d-%H%M%S")
    path = CLIPS_DIR / f"{name}.wav"

    if args.text:
        print(f'\n  say:  "{args.text}"')
    record(path, device_index)

    if not path.is_file() or path.stat().st_size < 1000:
        sys.exit(
            "\n  Nothing was recorded.\n"
            "  Check that your terminal has microphone access in\n"
            "  System Settings > Privacy & Security > Microphone,\n"
            "  and that the device index is right (python record.py --list)."
        )

    if args.text:
        path.with_suffix(".txt").write_text(args.text.strip() + "\n")
    print(f"  saved {path}  ({path.stat().st_size / 1024:.0f} KB)\n")


if __name__ == "__main__":
    main()
