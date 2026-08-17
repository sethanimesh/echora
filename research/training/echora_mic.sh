#!/usr/bin/env bash
# Record a voice on the Mac, run it through the command-v3 literal-ASR server on
# the Pod, and print the raw top-k alternatives for every loaded adapter.
#
# The server must already be running on the Pod:
#   /workspace/echora/research/training/cloud/start_command_v3_server.sh
#
# Examples:
#   ./research/training/echora_mic.sh --label animesh --ref "I want water"
#   ./research/training/echora_mic.sh --seconds 5 --label helper-voice
#   ./research/training/echora_mic.sh --file research/benchmarks/clips/20260815-135957.wav --ref "I water"
#   ./research/training/echora_mic.sh --list-devices
#
# Alternatives are raw beam-search output.  Probabilities are relative search
# scores, not calibrated confidence.  No ranking or semantic repair is applied.
set -euo pipefail

POD_HOST="${ECHORA_POD_HOST:-202.181.159.233}"
POD_PORT="${ECHORA_POD_PORT:-19317}"
POD_USER="${ECHORA_POD_USER:-root}"
INBOX=/workspace/echora/serve/inbox
OUTBOX=/workspace/echora/serve/outbox

# avfoundation renumbers devices whenever one connects or disconnects, so the
# default is matched by name and only falls back to an index.
DEVICE="${ECHORA_AUDIO_DEVICE:-MacBook Pro Microphone}"
SECONDS_LIMIT=""
LABEL=""
REFERENCE=""
SOURCE_FILE=""
ADAPTERS="base,v1,v3"
BEAMS=5
KEEP=0
SESSION_DIR="${ECHORA_SESSION_DIR:-$HOME/echora-voice-tests}"

CONTROL="$HOME/.ssh/echora-pod-%r@%h:%p"
SSH_OPTS=(-o StrictHostKeyChecking=no -o ControlMaster=auto
          -o "ControlPath=$CONTROL" -o ControlPersist=300 -o ConnectTimeout=20)

pod_ssh() { ssh "${SSH_OPTS[@]}" -p "$POD_PORT" "$POD_USER@$POD_HOST" "$@"; }
pod_scp() { scp "${SSH_OPTS[@]}" -P "$POD_PORT" "$@"; }

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

audio_devices() {
  ffmpeg -f avfoundation -list_devices true -i "" 2>&1 \
    | sed -n '/AVFoundation audio devices/,$p' \
    | sed 's/^\[[^]]*\] //' | grep -E '^\[[0-9]+\]' || true
}

# Accepts an index or a case-insensitive name fragment; prints "index<TAB>name".
resolve_device() {
  local want="$1" listing line
  listing="$(audio_devices)"
  [ -n "$listing" ] || { echo "no audio input devices found" >&2; return 1; }
  case "$want" in
    ''|*[!0-9]*) line="$(printf '%s\n' "$listing" | grep -i -- "$want" | head -1)" ;;
    *)           line="$(printf '%s\n' "$listing" | grep -E "^\[$want\]" | head -1)" ;;
  esac
  if [ -z "$line" ]; then
    echo "no audio device matched '$want'. Available:" >&2
    printf '%s\n' "$listing" >&2
    return 1
  fi
  printf '%s\t%s\n' \
    "$(printf '%s' "$line" | sed -E 's/^\[([0-9]+)\].*/\1/')" \
    "$(printf '%s' "$line" | sed -E 's/^\[[0-9]+\] //')"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --device)        DEVICE="$2"; shift 2 ;;
    --seconds)       SECONDS_LIMIT="$2"; shift 2 ;;
    --label)         LABEL="$2"; shift 2 ;;
    --ref)           REFERENCE="$2"; shift 2 ;;
    --file)          SOURCE_FILE="$2"; shift 2 ;;
    --adapters)      ADAPTERS="$2"; shift 2 ;;
    --beams)         BEAMS="$2"; shift 2 ;;
    --keep)          KEEP=1; shift ;;
    --list-devices)  audio_devices; exit 0 ;;
    -h|--help)       usage ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

command -v ffmpeg >/dev/null || { echo "ffmpeg is required (brew install ffmpeg)" >&2; exit 1; }

if ! pod_ssh "test -f $OUTBOX/SERVER_READY" 2>/dev/null; then
  echo "The Pod server is not ready." >&2
  echo "Start it on the Pod with:" >&2
  echo "  /workspace/echora/research/training/cloud/start_command_v3_server.sh" >&2
  exit 1
fi

mkdir -p "$SESSION_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
CLEAN_LABEL="$(printf '%s' "${LABEL:-clip}" | tr -cs 'A-Za-z0-9._-' '-' | sed 's/^-*//; s/-*$//')"
STEM="${CLEAN_LABEL:-clip}-$STAMP"
WAV="$SESSION_DIR/$STEM.wav"

if [ -n "$SOURCE_FILE" ]; then
  [ -f "$SOURCE_FILE" ] || { echo "no such file: $SOURCE_FILE" >&2; exit 1; }
  echo "converting $SOURCE_FILE to 16 kHz mono"
  ffmpeg -v error -i "$SOURCE_FILE" -ac 1 -ar 16000 -c:a pcm_s16le -y "$WAV"
else
  RESOLVED="$(resolve_device "$DEVICE")" || exit 1
  DEVICE_INDEX="${RESOLVED%%	*}"
  DEVICE_NAME="${RESOLVED#*	}"
  echo "microphone: [$DEVICE_INDEX] $DEVICE_NAME"
  if [ -n "$SECONDS_LIMIT" ]; then
    echo "speak now; stopping automatically after ${SECONDS_LIMIT}s"
    ffmpeg -v error -f avfoundation -i ":$DEVICE_INDEX" -ac 1 -ar 16000 \
      -c:a pcm_s16le -t "$SECONDS_LIMIT" -y "$WAV"
  elif [ -t 0 ]; then
    echo "speak now; press Enter to stop"
    ffmpeg -v error -f avfoundation -i ":$DEVICE_INDEX" -ac 1 -ar 16000 \
      -c:a pcm_s16le -y "$WAV" </dev/null &
    FFMPEG_PID=$!
    read -r _ || true
    kill -INT "$FFMPEG_PID" 2>/dev/null || true
    wait "$FFMPEG_PID" 2>/dev/null || true
  else
    echo "no terminal available; defaulting to a 6 second recording"
    ffmpeg -v error -f avfoundation -i ":$DEVICE_INDEX" -ac 1 -ar 16000 \
      -c:a pcm_s16le -t 6 -y "$WAV"
  fi
fi

[ -s "$WAV" ] || { echo "recording produced no audio" >&2; exit 1; }
echo "saved $WAV ($(du -h "$WAV" | cut -f1))"

SIDECAR="$SESSION_DIR/$STEM.request.json"
/usr/bin/python3 - "$SIDECAR" "$STEM" "$LABEL" "$REFERENCE" "$ADAPTERS" "$BEAMS" <<'PY'
import json, sys
path, stem, label, reference, adapters, beams = sys.argv[1:7]
json.dump({
    "label": label or stem,
    "reference": reference,
    "adapters": [a.strip() for a in adapters.split(",") if a.strip()],
    "beams": int(beams),
}, open(path, "w"), indent=2)
PY

# The server triggers on the .wav, so the request sidecar must land first and
# the audio must arrive under a temporary name and then be renamed atomically.
pod_scp "$SIDECAR" "$POD_USER@$POD_HOST:$INBOX/$STEM.json" >/dev/null
pod_scp "$WAV" "$POD_USER@$POD_HOST:$INBOX/$STEM.wav.part" >/dev/null
pod_ssh "mv $INBOX/$STEM.wav.part $INBOX/$STEM.wav"

printf 'transcribing'
RESULT=""
for _ in $(seq 1 150); do
  if pod_ssh "test -f $OUTBOX/$STEM.json" 2>/dev/null; then
    RESULT="$(pod_ssh "cat $OUTBOX/$STEM.json")"
    break
  fi
  printf '.'
  sleep 2
done
echo

if [ -z "$RESULT" ]; then
  echo "timed out waiting for a result; check the server log on the Pod:" >&2
  echo "  tail -n 40 /workspace/echora/serve/server.log" >&2
  exit 1
fi

SHOW="$(dirname "$0")/echora_show_result.py"
RESULT_FILE="$SESSION_DIR/$STEM.result.json"
printf '%s\n' "$RESULT" >"$RESULT_FILE"
/usr/bin/python3 "$SHOW" <"$RESULT_FILE"

rm -f "$SIDECAR"
echo
echo "audio  $WAV"
echo "json   $RESULT_FILE"
