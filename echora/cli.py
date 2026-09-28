"""Command-line interface for trying and debugging the pipeline.

    echora "Main kal office mein meeting attend karunga."
    echora                          # interactive REPL
    echora -e "..."                 # explain: per-token labels and who decided
    echora -c offline "..."         # force a specific classifier
    echo "..." | echora -            # read stdin

The explain mode is the point of this tool. When output is wrong the useful
question is never "what did it say" but "which stage decided that, and on what
evidence" -- so ``-e`` shows the label, the deciding stage, and the Devanagari
candidates for every token.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from echora.config import Config
from echora.core.model import Label, Source
from echora.pipeline import Pipeline, Result
from echora.tts.fish_audio import (
    AlignmentSegment, SpeechError, SpeechOptions, SpeechStyle,
)

# ANSI colours, suppressed when piped or when NO_COLOR is set.
_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    import os

    if not _TTY or os.environ.get("NO_COLOR"):
        return text
    return f"\033[{code}m{text}\033[0m"


DIM = lambda s: _c("2", s)          # noqa: E731
BOLD = lambda s: _c("1", s)         # noqa: E731
RED = lambda s: _c("31", s)         # noqa: E731
GREEN = lambda s: _c("32", s)       # noqa: E731
YELLOW = lambda s: _c("33", s)      # noqa: E731
BLUE = lambda s: _c("34", s)        # noqa: E731
MAGENTA = lambda s: _c("35", s)     # noqa: E731
CYAN = lambda s: _c("36", s)        # noqa: E731

def _width(text: str) -> int:
    """Terminal columns a string occupies.

    Emoji and CJK are double-width, and combining marks (common in Devanagari)
    are zero-width, so len() misaligns the explain table.
    """
    import unicodedata

    total = 0
    for ch in text:
        if unicodedata.combining(ch):
            continue
        total += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return total


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


_LABEL_COLOUR = {
    Label.HI: GREEN,
    Label.EN: BLUE,
    Label.NAME: MAGENTA,
    Label.NUMBER: CYAN,
    Label.CODE_URL: CYAN,
    Label.OTHER: DIM,
    Label.AMBIGUOUS: YELLOW,
}


def explain(result: Result) -> None:
    """Print the per-token decision table."""
    print()
    print(f"  {DIM('input ')} {result.raw}")
    print(f"  {BOLD('output')} {result.text}")
    if result.speech != result.text:
        print(f"  {DIM('speech')} {result.speech}")
    print()

    header = f"  {'token':<16} {'label':<11} {'decided by':<12} candidates"
    print(BOLD(header))
    print(DIM("  " + "-" * (len(header) - 2)))

    for seg in result.utterance.segments:
        if not seg.is_word and seg.source is Source.NONWORD:
            continue  # whitespace/punctuation: preserved, not interesting here
        colour = _LABEL_COLOUR.get(seg.label, str)
        source = seg.source.value if seg.source else "-"
        candidates = ", ".join(seg.candidates[:3]) if seg.candidates else ""
        arrow = f" -> {BOLD(seg.output)}" if seg.output else ""
        confidence = f" ({seg.confidence})" if seg.confidence else ""
        print(
            f"  {_pad(seg.text[:16], 16)} {colour(f'{seg.label.value:<11}')} "
            f"{DIM(f'{source:<12}')} {DIM(candidates)}{arrow}{confidence}"
        )

    print()
    reached = result.llm_spans
    total_words = sum(1 for _ in result.utterance.words())
    if total_words:
        share = reached / total_words
        print(
            f"  {DIM('spans sent to classifier:')} {reached}/{total_words} "
            f"({share:.0%})"
            + (f"  {DIM('via')} {result.classifier_used}"
               if result.classifier_used else "")
        )


def render(
    result: Result, elapsed_ms: float, args, audio_meta: dict | None = None
) -> None:
    if args.json:
        payload = {
            "raw": result.raw,
            "text": result.text,
            "speech": result.speech,
            "llm_spans": result.llm_spans,
            "classifier": result.classifier_used,
            "elapsed_ms": round(elapsed_ms, 1),
            "labels": [
                {"text": s.text, "label": s.label.value,
                 "source": s.source.value if s.source else None,
                 "output": s.output}
                for s in result.utterance.words()
            ],
        }
        if audio_meta is not None:
            payload["audio"] = audio_meta
        print(json.dumps(payload, ensure_ascii=False))
    elif args.explain:
        explain(result)
        print(f"  {DIM('elapsed:')} {elapsed_ms:.0f} ms")
    elif args.speech:
        print(result.speech)
    else:
        print(result.text)


def _fish_requested(args) -> bool:
    return bool(args.play or args.audio_out or args.timestamps_out)


def _resolve_audio_format(args) -> str:
    inferred = None
    if args.audio_out is not None:
        inferred = args.audio_out.suffix.lower().removeprefix(".")
        if inferred not in ("opus", "mp3", "wav", "pcm"):
            raise ValueError(
                "audio output must end in .opus, .mp3, .wav, or .pcm"
            )
    if args.audio_format and inferred and args.audio_format != inferred:
        raise ValueError(
            f"--format {args.audio_format} conflicts with {args.audio_out.suffix}"
        )
    return args.audio_format or inferred or "opus"


def _speech_options(args, config: Config) -> SpeechOptions:
    return SpeechOptions(
        voice_id=args.voice or config.fish_voice_id,
        style=SpeechStyle(args.style) if args.style else None,
        direction=args.direction,
        speed=args.speed if args.speed is not None else 1.0,
        volume=args.volume if args.volume is not None else 0.0,
        latency=args.latency or "balanced",
        format=_resolve_audio_format(args),
    )


def _alignment_json(alignment: tuple[AlignmentSegment, ...]) -> bytes:
    payload = {
        "segments": [
            {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end,
                "chunk_seq": segment.chunk_seq,
            }
            for segment in alignment
        ]
    }
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    """Write beside the destination and expose it only when complete."""
    parent = path.parent
    if not parent.exists():
        raise SpeechError(f"output directory does not exist: {parent}", kind="validation")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    os.close(fd)
    temp_path = Path(temporary)
    try:
        temp_path.write_bytes(data)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _set_dotenv_value(path: Path, key: str, value: str) -> None:
    """Atomically add or replace one dotenv value, preserving every other line."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    prefix = f"{key}="
    replacement = f"{key}={value}"
    found = False
    output: list[str] = []
    for line in lines:
        if line.strip().startswith(prefix):
            if not found:
                output.append(replacement)
                found = True
            continue
        output.append(line)
    if not found:
        output.append(replacement)
    _atomic_write(path, ("\n".join(output) + "\n").encode("utf-8"))


def _highlight_stream(events, *, enabled: bool, collector: dict):
    """Tee events to playback while showing newly aligned words on stderr."""
    seen: dict[int, int] = {}
    for event in events:
        collector["audio"].append(event.audio)
        if event.alignment is not None:
            collector["snapshots"][event.chunk_seq] = event.alignment
            if enabled:
                start = min(seen.get(event.chunk_seq, 0), len(event.alignment))
                for segment in event.alignment[start:]:
                    print(f"{segment.text} ", end="", file=sys.stderr, flush=True)
                seen[event.chunk_seq] = len(event.alignment)
        yield event.audio
    if enabled:
        print(file=sys.stderr)


def _run_synthesis(result: Result, args, config: Config) -> dict:
    options = _speech_options(args, config)
    synthesizer = config.build_synthesizer()
    collector: dict = {"audio": [], "snapshots": {}}
    chunks = _highlight_stream(
        synthesizer.stream(result.speech, options),
        enabled=bool(args.play and not args.no_highlight),
        collector=collector,
    )

    if args.play:
        try:
            from fishaudio.utils import play
        except ImportError as exc:
            raise SpeechError(
                "playback support is not installed; install echora[fish]",
                kind="dependency",
            ) from exc
        try:
            play(chunks)
        except SpeechError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize playback dependencies
            raise SpeechError(f"audio playback failed: {exc}", kind="playback") from exc
    else:
        for _ in chunks:
            pass

    audio = b"".join(collector["audio"])
    if not audio:
        raise SpeechError("Fish Audio returned no audio", kind="protocol")
    alignment = tuple(
        segment
        for chunk_seq in sorted(collector["snapshots"])
        for segment in collector["snapshots"][chunk_seq]
    )
    if args.timestamps_out is not None and not alignment:
        raise SpeechError(
            "Fish Audio returned no timestamp alignment", kind="protocol"
        )
    if args.audio_out is not None:
        _atomic_write(args.audio_out, audio)
    if args.timestamps_out is not None:
        _atomic_write(args.timestamps_out, _alignment_json(alignment))

    return {
        "format": options.format,
        "bytes": len(audio),
        "voice_id": options.voice_id,
        "style": options.style.value if options.style else "custom",
        "direction": options.direction,
        "latency": options.latency,
        "audio_path": str(args.audio_out) if args.audio_out else None,
        "timestamps_path": (
            str(args.timestamps_out) if args.timestamps_out else None
        ),
        "alignment_segments": len(alignment),
    }


def _process_text(pipeline: Pipeline, text: str, args, config: Config) -> int:
    started = time.perf_counter()
    result = pipeline.run(text)
    elapsed = (time.perf_counter() - started) * 1000
    if not _fish_requested(args):
        render(result, elapsed, args)
        return 0
    try:
        audio_meta = _run_synthesis(result, args, config)
    except (SpeechError, ValueError) as exc:
        if args.json:
            render(
                result,
                elapsed,
                args,
                audio_meta={"error": str(exc), "kind": getattr(exc, "kind", "validation")},
            )
        else:
            print(result.speech)
        print(RED(f"  speech error: {exc}"), file=sys.stderr)
        return 1
    render(result, elapsed, args, audio_meta=audio_meta)
    return 0


def repl(pipeline: Pipeline, args, config: Config) -> None:
    print(BOLD("Echora") + DIM("  — Hinglish → TTS preprocessing"))
    print(DIM("  Type a sentence. /explain toggles detail, /quit exits.\n"))
    explain_mode = args.explain

    while True:
        try:
            line = input(BOLD("> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("/quit", "/q", "/exit"):
            return
        if line in ("/explain", "/e"):
            explain_mode = not explain_mode
            print(DIM(f"  explain {'on' if explain_mode else 'off'}"))
            continue
        if line == "/help":
            print(DIM("  /explain toggle detail   /quit exit"))
            continue

        try:
            status = _process_text(
                pipeline, line, replace_args(args, explain=explain_mode), config
            )
        except Exception as exc:  # noqa: BLE001 - a REPL must not die on one input
            print(RED(f"  error: {exc}"))
            continue
        if status:
            continue


def replace_args(args, **kw):
    """argparse.Namespace has no _replace; emulate it for the REPL toggle."""
    clone = argparse.Namespace(**vars(args))
    for key, value in kw.items():
        setattr(clone, key, value)
    return clone


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="echora",
        description="Convert Roman-script Hindi to Devanagari, preserving "
                    "English, names, numbers, URLs and punctuation.",
        epilog="With no TEXT argument, starts an interactive session.",
    )
    parser.add_argument("--env-file", type=Path, default=None,
                        help="explicit library environment file (none read by default)")
    parser.add_argument("text", nargs="*",
                        help="text to process; '-' reads stdin")
    parser.add_argument("-e", "--explain", action="store_true",
                        help="show per-token labels and which stage decided")
    parser.add_argument("-s", "--speech", action="store_true",
                        help="print the TTS-normalised form instead")
    parser.add_argument("--json", action="store_true",
                        help="machine-readable output")
    parser.add_argument("-c", "--classifier",
                        choices=["auto", "ollama", "groq", "offline"],
                        help="override ECHORA_CLASSIFIER")
    parser.add_argument("-m", "--model", help="override the model id")
    parser.add_argument("--timeout", type=float, help="classifier timeout (s)")
    parser.add_argument("--no-cache", action="store_true",
                        help="disable the decision cache")
    parser.add_argument("--warm", action="store_true",
                        help="preload the local model before starting")
    parser.add_argument("--play", action="store_true",
                        help="stream and play Fish Audio speech")
    parser.add_argument("--audio-out", type=Path, metavar="PATH",
                        help="atomically save generated audio")
    parser.add_argument("--timestamps-out", type=Path, metavar="PATH",
                        help="atomically save timestamp alignment JSON")
    parser.add_argument("--voice", help="Fish voice ID; overrides the configured default")
    direction = parser.add_mutually_exclusive_group()
    direction.add_argument(
        "--style", choices=[style.value for style in SpeechStyle],
        help="assistive delivery style (default: simple)",
    )
    direction.add_argument(
        "--direction", help="custom one-line Fish delivery direction (no brackets)"
    )
    parser.add_argument("--speed", type=float, help="speech speed, 0.5 to 2.0")
    parser.add_argument("--volume", type=float, help="volume adjustment, -20 to 20 dB")
    parser.add_argument("--latency", choices=["low", "balanced", "normal"],
                        help="Fish latency/quality mode")
    parser.add_argument("--format", dest="audio_format",
                        choices=["opus", "mp3", "wav", "pcm"],
                        help="generated audio format")
    parser.add_argument("--no-highlight", action="store_true",
                        help="disable approximate word progress during playback")
    parser.add_argument("--enroll-voice", action="append", type=Path, metavar="PATH",
                        help="voice sample to upload; repeat for multiple samples")
    parser.add_argument("--voice-transcript", action="append", default=[], metavar="TEXT",
                        help="exact sample transcript; repeat in sample order")
    parser.add_argument("--voice-title", help="title for the enrolled private voice")
    parser.add_argument("--confirm-consent", action="store_true",
                        help="confirm permission non-interactively for voice enrollment")
    parser.add_argument("--set-default", action="store_true",
                        help="save an enrolled voice ID to .env")
    return parser


def _run_enrollment(args, config: Config) -> int:
    paths = args.enroll_voice or []
    if args.text:
        raise ValueError("voice enrollment does not accept message text")
    if _fish_requested(args):
        raise ValueError("voice enrollment cannot be combined with speech output flags")
    if not args.voice_title:
        raise ValueError("--voice-title is required for enrollment")
    if len(paths) != len(args.voice_transcript):
        raise ValueError("each --enroll-voice requires one --voice-transcript")

    if not args.confirm_consent:
        print("Voice samples to upload to Fish Audio as a private voice:", file=sys.stderr)
        for path in paths:
            print(f"  {path}", file=sys.stderr)
        if not sys.stdin.isatty():
            raise ValueError(
                "interactive consent requires a terminal; use --confirm-consent "
                "only after obtaining permission"
            )
        answer = input(
            "Do you confirm you have permission to clone this voice? [y/N] "
        ).strip().lower()
        if answer not in ("y", "yes"):
            raise ValueError("voice enrollment cancelled")

    enrollment = config.build_synthesizer().enroll_voice(
        paths, args.voice_transcript, title=args.voice_title
    )
    if args.set_default:
        _set_dotenv_value(Path(".env"), "ECHORA_FISH_VOICE_ID", enrollment.id)
    if args.json:
        print(json.dumps({
            "id": enrollment.id,
            "title": enrollment.title,
            "state": enrollment.state,
            "default_saved": bool(args.set_default),
        }, ensure_ascii=False))
    else:
        print(enrollment.id)
        print(f"  state: {enrollment.state}", file=sys.stderr)
        if args.set_default:
            print("  saved as ECHORA_FISH_VOICE_ID in .env", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = Config.from_env(dotenv=args.env_file)
    if args.classifier:
        config.classifier = args.classifier
    if args.model:
        # One flag sets whichever vendor is actually in play.
        if config.classifier == "groq":
            config.groq_model = args.model
        else:
            config.ollama_model = args.model
    if args.timeout:
        config.timeout = args.timeout
    if args.no_cache:
        config.cache = False

    if args.enroll_voice:
        try:
            return _run_enrollment(args, config)
        except (SpeechError, ValueError) as exc:
            print(RED(f"  enrollment error: {exc}"), file=sys.stderr)
            return 1
    if args.set_default or args.confirm_consent or args.voice_title or args.voice_transcript:
        parser.error("voice enrollment options require --enroll-voice")

    speech_options_used = any((
        args.voice, args.style, args.direction, args.speed is not None,
        args.volume is not None, args.latency, args.audio_format,
    ))
    if speech_options_used and not _fish_requested(args):
        parser.error("Fish speech options require --play, --audio-out, or --timestamps-out")
    if (args.audio_out or args.timestamps_out) and (
        not args.text or args.text == ["-"]
    ):
        parser.error("file output requires one-shot TEXT, not stdin or REPL mode")
    if _fish_requested(args):
        try:
            _speech_options(args, config)
        except ValueError as exc:
            parser.error(str(exc))

    pipeline = Pipeline.from_config(config)

    if args.warm:
        from echora.classify.ollama_classifier import OllamaSpanClassifier

        started = time.perf_counter()
        warmed = False
        # Reach through the decorator chain for anything that can preload.
        stack = [pipeline._classifier]
        while stack:
            node = stack.pop()
            if isinstance(node, OllamaSpanClassifier):
                node.warm_up()
                warmed = True
            for attr in ("_inner", "_classifiers"):
                value = getattr(node, attr, None)
                if isinstance(value, list):
                    stack.extend(value)
                elif value is not None:
                    stack.append(value)
        if warmed:
            print(DIM(f"  warmed in {(time.perf_counter() - started) * 1000:.0f} ms"),
                  file=sys.stderr)

    # stdin
    if args.text == ["-"] or (not args.text and not sys.stdin.isatty()):
        for line in sys.stdin:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if _process_text(pipeline, line, args, config):
                return 1
        return 0

    if not args.text:
        repl(pipeline, args, config)
        return 0

    text = " ".join(args.text)
    return _process_text(pipeline, text, args, config)


if __name__ == "__main__":
    raise SystemExit(main())
