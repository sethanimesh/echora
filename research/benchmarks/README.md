# TORGO fine-tune bench

Four TORGO-fine-tuned ASR checkpoints, one script each, run manually against
your own recordings.

## Setup

Already done. The venv lives in `research/benchmarks/.venv`. If you ever need to rebuild it:

    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

Give your terminal microphone access before the first recording:
System Settings > Privacy & Security > Microphone.

## Record

    .venv/bin/python record.py water -t "i would like some water please"

Saves `clips/water.wav` and `clips/water.txt`. Press ENTER to stop recording.
The `-t` text is optional; when present every run script prints it above the
transcript so you can see what the model got wrong.

    .venv/bin/python record.py --list     # input devices, if the default is wrong

You can also drop your own wav/mp3/m4a/flac files straight into `clips/`.

## Run

Each script loads one model and transcribes every clip in `clips/`:

    .venv/bin/python run_dbarbera.py
    .venv/bin/python run_ruch9265.py
    .venv/bin/python run_macarious.py
    .venv/bin/python run_sqrk_mms.py
    .venv/bin/python run_parakeet.py      # stock Parakeet TDT, not a TORGO fine-tune
    .venv/bin/python run_parakeet_ctc.py  # stock Parakeet CTC, not a TORGO fine-tune
    .venv/bin/python run_qwen3_asr.py      # official Qwen3-ASR 1.7B foundation

The NeMo-only official 1.1B Parakeet foundation runner is intended for the
Linux cloud environment:

    python run_parakeet_tdt_1_1b.py --device cuda

Results also land in `results/<model>.tsv` so you can diff runs later.

Once clips have `.txt` references, every run prints literal WER/CER. Compare
saved runs at any time with:

    .venv/bin/python metrics.py results/*.tsv

Use `--tag zero-shot-20260815` (or a model/data version) to preserve a snapshot
instead of overwriting the default result. Top-k oracle WER is reported only
when a decoder supplies raw alternatives; it never replaces top-1 WER.

The native Transformers Parakeet runners are currently greedy-only, so their
`alternatives` columns remain empty. Do not manufacture variants with an LLM;
verified decoder beams will be added with the NeMo training setup.

## Foundation-only decision gate

For the exact L40S setup, upload, run, result-download, comparison, and next
fine-tuning plan, use [`cloud/README.md`](cloud/README.md). The cloud scripts are
fully local under `research/benchmarks/cloud/` and can be transferred with the benchmark.

No community fine-tuned weights participate in model selection or training.
Before starting a paid instance, verify the exact SSD files against their
published SHA-256 digests:

    .venv/bin/python verify_foundation_weights.py

Build the fixed 400-utterance screen with 50 duration-stratified examples from
each dysarthric speaker:

    .venv/bin/python make_foundation_gate.py \
      --source ../data/derived/torgo/dysarthric.jsonl \
      --output ../data/derived/torgo/foundation_gate_400.jsonl

On the cloud GPU, run both official foundations on that exact manifest:

    python evaluate_manifest.py --runner parakeet_tdt_1_1b --device cuda \
      --manifest ../data/derived/torgo/foundation_gate_400.jsonl \
      --output results/foundation_gate/parakeet_tdt_1_1b.tsv --batch-size 4

    python evaluate_manifest.py --runner qwen3_asr --device cuda \
      --manifest ../data/derived/torgo/foundation_gate_400.jsonl \
      --output results/foundation_gate/qwen3_asr_1_7b.tsv --batch-size 4

Then compare literal ASR output on identical utterance IDs:

    python compare_foundations.py \
      results/foundation_gate/parakeet_tdt_1_1b.tsv \
      results/foundation_gate/qwen3_asr_1_7b.tsv

The gate uses speaker-macro WER as its primary metric and guards against the
deletions and empty transcripts that are especially harmful for slow speech.
It does not use an LLM, semantic correction, or the Echora candidate ranker.

To create an ignored stress suite that inserts 0.5, 1.0, and 2.0 second pauses
at low-energy internal boundaries while preserving the transcript:

    .venv/bin/python make_pause_stress.py

Run any model with `--clips ../data/derived/pause_stress/1000ms`. Result TSVs
also record total and maximum internal low-energy gaps for later slicing. These
synthetic clips test pause sensitivity; they are not evidence of clinical
generalization and must never be mixed into clean test scores.

Flags: `--clips <dir>` to point somewhere else, `--device mps` to try the GPU
(CPU is the default because a few ops fall back and MPS is not always faster
at these model sizes).

## The models

| Script | Checkpoint | Architecture | Size | Reported WER |
|---|---|---|---|---|
| `run_dbarbera.py` | dbarbera/whisper-small-torgo-dysarthria-lora | whisper-small + LoRA | 1 GB | 40.7% dysarthric, 16.9% overall |
| `run_ruch9265.py` | ruch9265/distil-whisper-torgo | whisper-tiny, full FT | 150 MB | not reported |
| `run_macarious.py` | macarious/torgo_xlsr_finetune_M04 | wav2vec2-XLSR-53 + CTC | 1.3 GB | 27.4% (speaker M04 held out) |
| `run_sqrk_mms.py` | sqrk/torgo-mms1ball-Nov29 | MMS-1B-all + CTC | 3.9 GB | 42.0% |
| `run_parakeet.py` | nvidia/parakeet-tdt-0.6b-v3 | FastConformer + TDT | 2.5 GB | n/a, not TORGO-adapted |
| `run_parakeet_ctc.py` | nvidia/parakeet-ctc-1.1b | FastConformer + CTC | 4.3 GB | n/a, not TORGO-adapted |
| `run_parakeet_tdt_1_1b.py` | nvidia/parakeet-tdt-1.1b | FastConformer + TDT | 4.3 GB | n/a, official foundation |
| `run_qwen3_asr.py` | Qwen/Qwen3-ASR-1.7B-hf | AuT encoder + Qwen3 decoder | 3.6 GB | n/a, official foundation |

`run_parakeet.py` is the odd one out: stock NVIDIA weights with no dysarthric
adaptation at all. It is here because plan.md names Parakeet-TDT the leading
fine-tuning candidate, and you want to see the un-adapted starting point. Note
it uses 0.6b-v3 rather than the tdt-1.1b named in plan.md, because the 1.1b repo
ships only a .nemo archive that requires the NVIDIA NeMo toolkit. 0.6b-v3 ships
safetensors, loads natively in transformers, and scores better on the Open ASR
leaderboard.

Public community atypical-speech checkpoints now exist, but this project
deliberately excludes them from the foundation comparison and subsequent
training because their source conditions do not match the target population.

`run_parakeet_ctc.py` is the CTC counterpart, there so you can read a transducer
and a CTC decoding of the same clip side by side. The model actually worth
having is nvidia/parakeet-tdt_ctc-1.1b, a hybrid carrying both heads on one
shared encoder -- disagreement between two heads over one encoder is a clean
uncertainty signal, which two separate models cannot give you. Every hybrid
tdt_ctc checkpoint is .nemo-only and needs the NVIDIA NeMo toolkit, so that one
is a cloud-GPU job rather than something this bench can run.

## ASR boundary

This benchmark ends at literal recognition. TDT beam hypotheses and CTC token
posteriors are allowed because they are emitted by the ASR decoder. Personal
context, semantic cleanup, an LLM, and Echora's message candidate ranker cannot
affect transcripts or metrics here. Those remain a downstream confirmation
layer.

These are the four that survived a trust filter over ~100 TORGO repos on the
Hub. Everything excluded was excluded for a stated reason:

- `sqrk/torgo-whisper-lg-3-Nov29` claims ~5% WER on TORGO. Published
  speaker-independent results sit near 38%. That gap is the signature of
  utterance overlap between train and test, since TORGO speakers read the same
  prompt lists. Its sibling MMS run reports 42%, which is credible, so that is
  the one included.
- `zorbbbb/whisper-small-lora-torgo`, all `kesbeast23/*` — model card is an
  unfilled template. No stated splits, no way to know what the weights are.
- `Menhaz/*`, `RSTV-24/*`, `p29ris/*` — same families as models already here,
  with less documentation.
- The other ~60 `macarious`/`yip-i`/`jindaznb` repos are per-speaker variants of
  the same leave-one-out study. `_M04` is included as the family representative.

Adding one back is a copy of the closest run script with the model id changed.

## What to expect

These were fine-tuned on 15 speakers reading fixed prompt lists, so on your
voice the Whisper ones will likely look fine (they inherit general English from
the base model and the TORGO adaptation is a thin layer on top) while the CTC
ones will look phonetic and rough. On a synthetic test clip of "i would like
some water please":

    dbarbera    I would like some water please
    ruch9265    I would like some water please
    macarious   i willike som water plaze
    sqrk_mms    <unk> will like some water please.

That difference is the point rather than a bug. The CTC models have no language
model, so they transcribe what was actually said instead of smoothing it into
fluent English. For a confirmation loop that matters: a model that fails
visibly is safer than one that confidently invents a plausible sentence.
