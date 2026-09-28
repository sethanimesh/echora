# Facial-expression comparison experiment

Implemented 5 September 2026. Current status: **experiment inactive; Gemini-only
facial analysis selected by the user** after the first real-face trial. The
local model and reports are retained. The trial establishes agreement and timing,
not a general accuracy ranking.

## What is active

`ECHORA_FACE_MODE=gemini` is set in the ignored `the repository root .env`. Local
models are not checked, loaded or run for current recordings. The paired flow
described below remains available for a future explicit experiment.

During comparison mode, recording remains one action: audio and three camera snapshots are submitted
together. Whisper transcription is unchanged. Gemini voice analysis and the
paired facial check run concurrently after transcription. Within the facial
check, both providers receive the **identical three snapshots** concurrently.

Gemini remains the declared facial source for the one automatic tone/pace
recommendation during this experiment. EmotiEffLib is an evaluation result; it is
not silently substituted on disagreement or failure. The user never has to
choose between voice and face. Applying delivery still does not start speech;
The unified app now follows its selected immediate-speech policy after the current result is ready; unresolved wording still requires a choice.

The existing camera sampler retains the last three completed snapshots from
distinct video times, sampled every 300 ms while recording. These capture the
end of a recording, not a full temporal history. Neither identity continuity
nor liveness is established. No new camera permission or separate recording is
needed for comparison mode.

## Local model and preprocessing

- Expression classifier: **EmotiEffLib `enet_b0_8_best_afew`**, full ONNX network.
  It runs directly in ONNX Runtime; the EmotiEffLib Python package is not needed.
  Upstream RGB cropping/preprocessing is preserved: OpenCV resize to 224×224,
  divide by 255, ImageNet mean `[.485,.456,.406]`, standard deviation
  `[.229,.224,.225]`, float32 NCHW. Outputs are eight logits, normalized with
  stable softmax. No identity embedding or face registration is performed.
- Detection: **OpenCV YuNet `face_detection_yunet_2026may`**, dynamic-size ONNX
  compatible with OpenCV 5. Its box is cropped before expression classification.
- Engine: CPU, two ONNX inference threads; OpenCV uses one thread. Telemetry is
  disabled. Verified weights preload at backend startup, with a 45-second bound.
- Pinned source revisions, byte lengths, URLs and SHA-256 hashes: [models.json](models.json).
  Weights total approximately 16.3 MB and are stored in ignored
  `communication/data/face-models`. They are verified before loading and are
  never downloaded during a recording.
- Local requirements: [requirements-face.txt](../../backend/requirements-face.txt).
  macOS arm64 / Python 3.14.7 was used for this test. EmotiEffLib is Apache-2.0;
  YuNet is MIT. License copies are alongside this document.

Source references:
[EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib),
[pinned preprocessing](https://github.com/sb-ai-lab/EmotiEffLib/blob/520a051c64cd191521e5934655314e769a319684/emotiefflib/facial_analysis.py),
[YuNet](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet).

## Initial quality and confidence rules

These are **prototype thresholds**, not calibrated guarantees:

1. JPEG only, at most 256 KiB per image and 1,638,400 decoded pixels; three images.
2. Exactly one YuNet detection at its 0.8 detection threshold. Missing/multiple
   detections do not enter the expression classifier.
3. Face crop at least 80 pixels in both dimensions and at least 88% of the
   predicted box inside the image. Mean grayscale brightness 25–235, standard
   deviation at least 12, and Laplacian variance at least 18.
4. A class must have softmax score ≥0.65 and lead the next class by ≥0.20.
   These are **model scores**, not probabilities that a person feels an emotion.
5. Every frame must have an acceptable view, and two of three must agree on a
   usable delivery cue. The `Happiness` class becomes `positive_expression`,
   not a claim of happiness or a measurement of smile intensity. `Neutral`
   becomes a neutral cue. Other labels are kept in the diagnostic result but
   do not yet trigger delivery styling. A high happiness score is never called
   a broad smile.

Current delivery fusion is still the bounded rule described in the main
repository README. Face cues style playback pace; they do not measure actual
speech speed. A neutral or limited expression cannot imply slow speech. Emotion
classifiers can misread speaking movements, atypical movement, posed expressions
or photos; no claim of disability-specific validation or liveness is made.

## Modes and reproducible setup

From the repository root:

```sh
.venv/bin/python -m pip install -r communication/backend/requirements-face.txt
.venv/bin/python communication/experiments/facial-cues/setup_models.py
```

The setup script verifies an existing asset or downloads its exact pinned URL
and checks the hash before installing. It does not fetch moving model versions.
The current pinned runtime installs without conflicting with the existing
communication dependencies (`pip check` passed).

Set `ECHORA_FACE_MODE` in `the repository root .env`, then restart the backend:

| Mode | Facial image processing | Delivery source |
| --- | --- | --- |
| `gemini` | Gemini only; original default | Gemini facial cue + voice |
| `compare` | Local EmotiEffLib and Gemini, same frames | Gemini facial cue + voice; local result evaluated |
| `local` | Local model only; no images sent to Gemini | Local facial cue + voice |

Audio ASR and voice analysis remain cloud-based in every mode. No automatic
provider fallback or retry exists. A provider failure leaves the other result
and transcript available. Stopped/replaced jobs cannot publish late results.
Local processing is serialized with a non-queueing lock and a 15-second request
wait bound. Cancellation discards its result; an already-running native call
may finish its bounded three-image work before releasing the lock.

A command-line trial accepts three explicitly supplied JPEG files:

```sh
.venv/bin/python communication/experiments/facial-cues/compare.py frame1.jpg frame2.jpg frame3.jpg --mode local --output /tmp/face-local.json
```

`--mode compare` or `--mode gemini` **uploads those three files to Gemini** using
the configured key. The default is local. Neither mode opens a camera. API-based
app trials retain the existing per-process request limits; this standalone
experiment command makes one explicit trial and does not use app counters.

## User trial

1. Refresh Echora at `http://127.0.0.1:3000/`.
2. Enable **Include facial cues while I speak**. The notice states that both
   facial models will check the same snapshots.
3. Record “Please bring me water” for about 4–6 seconds with an ordinary or
   neutral expression. Stop, then **Transcribe & suggest delivery**.
4. Open **Facial model comparison** below Your delivery. Read the result and
   time for each model. Label the test after analysis; that label is never sent
   to either model.
5. Apply the single suggested delivery and Confirm & speak if desired. Mark
   how the speech sounded, then **Save comparison report**.
6. Repeat with a comfortable smile, keeping it through the end of the phrase.
   Later add head movement, poor lighting and out-of-view controls. Do not
   force facial movements; an unavailable cue is a valid outcome.

The saved JSON is intentionally limited to model results, timing, test label
and listening feedback. It excludes images, audio, transcript and identity.
Images exist transiently in browser/server memory; nothing automatically writes
them to app storage or logs. Closing/reloading the app loses unsaved comparison
results. Gemini retention follows the provider's policies. Local mode performs
no inference-time networking for images.

## Recorded validation

- **256 backend tests passed**, including routing, paired concurrency, separate
  failures, cancellation, crop/quality rules, score gating, image bounds,
  normalization and preservation of transcript/confirmation behavior.
- **18 targeted frontend tests passed**: shared capture, automatic fusion,
  comparison display, source-selection removal, export field whitelist and
  unchanged confirmation rules. Types, lint and production build passed.
- Native classifier execution checked the actual ONNX input/output and
  preprocessing using a uniform synthetic crop; this is not a face-accuracy test.
- Real local detector rejected blank JPEGs without a tone. The first library
  import/load in the development process took about 19 seconds; subsequent
  startup preload completed in under the tool's one-second initial window.
- Paired HTTP trial through port 5173, using existing synthetic speech and
  three blank JPEGs: transcript retained; voice neutral/standard; **both facial
  checks returned no face and no tone**. Local check **69 ms**, Gemini facial
  check **5,091 ms**, complete submission **5.897 s**. Results are in
  [verification.json](verification.json). This timing covers no-face rejection,
  not expression classification latency, and is a single measurement.

No user's camera was opened by the development tools. A first user-submitted
real-face report is recorded below; its intended expression and listening
feedback are still unlabelled. Agreement between models is not accuracy;
compare against the user's labelled test, inspect abstentions and timing, and
use separate recordings for any later threshold tuning and evaluation. Do not
promote the local model based solely on the blank-image test or reported scores.


## First user-submitted real-face trial — 5 September 2026

[Original exported report](reports/2026-09-05T152748-user-trial.json), timestamp
15:27:48.997 UTC (20:57:48.997 India time). The report is preserved as supplied;
no images or audio accompanied it.

| Check | Local EmotiEffLib | Gemini Flash |
| --- | --- | --- |
| View | Clear face in all three frames | Clear face |
| Cue | Positive expression in all three frames | Smile |
| Facial tone | Warm | Warm |
| Processing time | 154 ms | 5,764 ms |

All three local frames passed the existing score and margin thresholds. Scores
were 0.7798, 0.8098 and 0.7216; these are uncalibrated classifier scores, not
accuracy measurements. Local processing was 37.4 times faster than Gemini's
facial check in this single trial (5.610 seconds less). This does not measure
end-to-end recording/voice latency or establish general speed/accuracy.

Interpretation: successful real-face processing and stable agreement across the
three sampled frames. The trial remains `unlabelled`, and listening feedback is
`not_listened`; the user's intended expression and preferred tone/pace therefore
remain unknown. No accuracy claim, threshold change or provider promotion is
justified yet. Gemini remains the delivery baseline. Next: confirm whether this
was an intentional smile, then run a labelled neutral-expression control with
the same phrase and record listening feedback.
