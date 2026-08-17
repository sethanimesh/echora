# Echora Speaker-Independent Dysarthric ASR Plan

## Summary

Build on an open foundation model; do not train an ASR model from scratch. Benchmark current foundations first, then fine-tune the two strongest under controlled, speaker-disjoint evaluation.

The initial population will be English speakers with motor dysarthria, particularly patterns represented by cerebral palsy, ALS, Parkinsonian disorders, and severe dysarthria. Do not combine aphasia into the acoustic model: aphasia affects language formulation, while dysarthria affects speech production; apraxia is another distinct motor-planning disorder that may coexist after stroke. [ASHA describes these distinctions](https://www.asha.org/practice-portal/clinical-topics/dysarthria-in-adults/).

Recommended model candidates:

1. NVIDIA Parakeet-TDT 1.1B as the leading fine-tuning candidate because its transducer decoder naturally supports N-best decoding and confidence extraction. A Parakeet-TDT fine-tune won the 2025 Speech Accessibility Challenge, reducing WER from 17.82% to 8.11%. [Challenge system](https://www.isca-archive.org/interspeech_2025/takahashi25_interspeech.html), [model card](https://huggingface.co/nvidia/parakeet-tdt-1.1b).
2. Qwen3-ASR 1.7B as the current general-quality challenger. Its official project includes fine-tuning support, but its ability to preserve dysarthric disfluencies and produce useful uncertainty must be tested. [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR).
3. Whisper large-v3 and large-v3-turbo as established baselines and fallbacks. [Whisper model card](https://huggingface.co/openai/whisper-large-v3-turbo).

Use a cloud NVIDIA GPU for training and the M2 Mac for data inspection and analysis. The first runtime may remain cloud-hosted.

## Experiment-First Implementation

### 1. Data and licensing gate

Create a versioned manifest containing source, license, checksum, speaker, condition, severity when available, prompt type, transcript, duration, and deterministic split assignment.

Use:

- **homeService** immediately: English, severe dysarthria, realistic home recordings, CC BY-NC-SA 4.0. [Official corpus](https://mini.dcs.shef.ac.uk/resources/homeservice-corpus/).
- **TORGO** only after confirming that an unaffiliated personal research project satisfies its “academic, non-profit” condition. It provides English CP/ALS dysarthric speech, controls, read material, and spontaneous speech. Do not treat unofficial Hugging Face mirrors as changing the original license. [Official TORGO source](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html).
- A small, balanced **Common Voice English 26** subset—including Indian/South Asian English—to prevent catastrophic forgetting of ordinary English. The current release is CC0. [Mozilla dataset](https://mozilladatacollective.com/datasets/cmqim2hn800ssnr07gvmpcnwu).
- Optional later: apply for HeyJay restricted access; it includes Parkinson’s, ALS, ataxia, dystonia, and some stroke speech, but it is not required for the first experiment. [HeyJay access page](https://www.icpsr.umich.edu/web/ICPSR/studies/39448/datadocumentation).

Exclude from the initial dependency chain:

- UA-Speech, because access is limited to government and academic labs. [Access terms](https://speechtechnology.web.illinois.edu/uaspeech/).
- Speech Accessibility Project data, because its agreement requires an organizational recipient and a separate authorized signatory. [Access requirements](https://speechaccessibilityproject.beckman.illinois.edu/conduct-research-through-the-project).
- AphasiaBank, because clinical access requires established institutional researchers. [Access levels](https://talkbank.org/aphasia/access.html).

Keep verbatim transcripts: repetitions, restarts, and partial words must not be silently normalized away.

### 2. Zero-shot baseline experiment

Run Parakeet-TDT 1.1B, Qwen3-ASR 1.7B, Whisper large-v3, and Whisper large-v3-turbo on identical held-out audio.

- Convert audio to 16 kHz mono without removing internal pauses; use conservative endpointing because slow speech and long pauses may be meaningful.
- Use nested leave-one-speaker-out evaluation. Add a phrase-disjoint challenge split where repeated prompts permit it.
- Decode up to five unique hypotheses using beam search. Deduplicate exact normalized matches but never invent artificial diversity.
- Report per-speaker and macro-average WER, CER, top-5 oracle WER, disfluency preservation, deletion/substitution rates, silence hallucinations, real-time factor, and GPU memory.
- Separately test normal English to detect a model that performs well only because the clinical corpus is small or prompt-constrained.
- Stop after publishing the reproducible baseline report and predictions. These results determine the two models entering fine-tuning.

Selection is lexicographic: lowest macro dysarthric WER; if models are within one percentage point, lowest top-5 oracle WER; then best disfluency preservation; then latency.

### 3. Controlled foundation-model fine-tuning

Fine-tune the two baseline winners rather than adopting an undocumented community dysarthria checkpoint.

- Balance batches by speaker and corpus so one prolific speaker cannot dominate.
- Mix approximately 80–90% dysarthric data with 10–20% ordinary English retention data.
- Compare a parameter-efficient adapter run with a partial/full fine-tune using early stopping. Use the same outer speaker folds, three seeds, and untouched test speakers.
- Apply conservative augmentation: room impulse responses, realistic background noise, gain changes, and mild speed perturbation. Do not use synthetic “dysarthria conversion” in the first iteration because it may encode unrealistic pathology.
- Preserve the original foundation checkpoint and every experiment configuration for reproducibility.
- Select the final acoustic model using the same lexicographic criteria as the baseline experiment.

Fine-tuning passes only if it produces at least 15% relative macro-WER improvement over the best zero-shot model, improves top-5 oracle WER, and degrades ordinary-English WER by no more than two absolute percentage points.

### 4. ASR decoder alternatives and uncertainty (not the message candidate ranker)

Generate 5-best internal hypotheses and expose the best three useful candidates to the confirmation UI.

This section remains inside the recognizer: alternatives must come from CTC,
TDT/RNN-T, or another acoustic decoder. Echora's later semantic/personal-context
candidate ranker consumes these outputs but is not part of ASR training,
decoding, model selection, or WER/CER reporting.

- Length-normalize beam scores and convert them into within-utterance evidence weights.
- Align N-best hypotheses into a word confusion network.
- Mark words uncertain when hypotheses disagree or when token posterior/margin is weak.
- Temperature-calibrate candidate and word scores exclusively on held-out development speakers.
- Choose the uncertain-word threshold on development data to recover at least 80% of word errors while flagging no more than 40% of all words.
- If calibration does not reach ECE ≤ 0.08, expose the value as an `evidence_score`, not as a probability or confidence claim.
- Always require user confirmation before speaking or sending a message, regardless of model confidence.

## Public Interface

Return a stable speech-layer object:

```json
{
  "utterance_id": "uuid",
  "model_version": "string",
  "language": "en",
  "candidates": [
    {
      "rank": 1,
      "text": "verbatim transcript",
      "evidence_score": 0.0,
      "calibrated_confidence": null
    }
  ],
  "uncertain_spans": [
    {
      "start_word": 2,
      "end_word": 2,
      "start_ms": 820,
      "end_ms": 1210,
      "observed": "water",
      "confidence": 0.61,
      "alternatives": [
        {"text": "quarter", "confidence": 0.24}
      ]
    }
  ],
  "audio_quality": {
    "low_snr": false,
    "clipped": false,
    "too_short": false
  },
  "confirmation_required": true
}
```

`evidence_score` is comparable only among candidates from the same utterance and model version. `calibrated_confidence` remains `null` until calibration acceptance tests pass. The later meaning/context layer consumes this object but must retain the original acoustic hypotheses and scores alongside any cleaned message proposal.

## Test and Acceptance Plan

- Unit tests for transcript normalization, beam deduplication, N-best alignment, uncertain-span extraction, score calibration, and serialization.
- Integration tests for empty audio, silence, clipping, noise, very slow speech, long internal pauses, repetitions, fragments, and fewer than three supported hypotheses.
- Regression fixtures containing model output and expected metrics for each frozen evaluation fold.
- Speaker leakage and prompt leakage checks before every training run.
- Report WER/CER by speaker, corpus, severity where available, and utterance type rather than only one aggregate score.
- Require top-5 oracle WER to be at least 15% relatively lower than top-1 WER; otherwise the alternatives are not adding enough recoverable information.
- Target cloud inference real-time factor ≤ 0.25 and post-utterance p95 latency under two seconds for 15-second messages.
- Do not claim effectiveness for stroke, aphasia, apraxia, or real-world deployment until an appropriate held-out population has been evaluated.

## Assumptions

- Version one targets English, short 1–15 second everyday messages, and speaker-independent motor-dysarthria recognition.
- The acoustic transcript remains verbatim; interpretation and cleanup belong to the meaning layer.
- No per-user acoustic personalization or onboarding recordings will be used.
- Training data may be non-commercial because this is a personal prototype. Any later commercial release requires a fresh license audit and potentially retraining without restricted corpora.
- With the currently accessible datasets, the defensible initial claim is “English motor-dysarthria prototype,” not “best model for all stroke survivors.”
