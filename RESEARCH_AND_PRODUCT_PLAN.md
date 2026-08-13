# Echora: research and product plan

**Working concept:** A user-controlled, personalized communication aid for adults with aphasia and potentially co-occurring apraxia of speech or dysarthria.

**Plan date:** 13 August 2026

## 1. Product definition

Echora helps a person communicate when their spoken attempt is difficult for another person or a standard speech recognizer to understand.

The primary loop is:

1. The user presses and holds a large talk control (with an optional tap-to-start/tap-to-stop access mode).
2. Recording continues until the user deliberately releases or stops it. Silence never ends the recording.
3. One or more speech recognizers produce an N-best hypothesis set, with acoustic confidence and token alternatives where available.
4. A constrained interpretation service uses those hypotheses, the current conversation, and user-approved personal information to construct a small set of semantically distinct candidate messages.
5. The user selects, edits, retries, or rejects all candidates.
6. Only the confirmed message is spoken through text-to-speech (TTS).
7. The audio, alternatives, chosen message, and corrections become a consented training event for future ranking and personalization.

The system is an augmentative and alternative communication (AAC) aid. It is not a diagnostic tool, a replacement for a speech-language pathologist (SLP), or an autonomous conversational agent.

### The most important product correction

The LLM should not invent an *answer to* the user. It should propose clear versions of the message the user may be trying to say. If the intended utterance is “tea,” a valid output may be “I would like some tea,” not a chatbot reply about tea. The user remains the author and final speaker.

## 2. Clinical and interaction assumptions

Aphasia affects language production and/or comprehension. Unclear articulation can instead, or additionally, result from apraxia of speech or dysarthria. These conditions can co-occur but require different forms of personalization. Recruitment, evaluation, and model analysis must therefore record the user's communication profile rather than treating “aphasic speech” as one homogeneous acoustic category.

ASHA recommends multimodal communication, communication-partner training, confirmation, summarization, drawing, written keywords, and choices as part of supported conversation for adults with aphasia. Echora should complement speech with text, personally meaningful photos or symbols, replay, gesture/pointing, and caregiver/partner support when the user wants it.

The target cohort for the first study should be narrow enough to learn from:

- Adults with chronic expressive aphasia who can reliably indicate a choice between two or three items, independently or through an agreed access method.
- Stratification for co-occurring apraxia/dysarthria, comprehension, reading ability, motor access, vision, hearing, primary language, and fatigue.
- Exclusion from the first efficacy claim—not from later product access—when the confirmation interface itself is not yet reliable. A later version should support symbols, partner-assisted scanning, switch access, and other modalities.

## 3. Experience design

### Recording state

- A single high-contrast talk control occupies the main target area.
- Give immediate visual and haptic feedback: “Recording,” elapsed time, and a waveform that does not imply the person should speak faster.
- Manual release/stop is the only endpoint. Voice activity detection may be used *after release* for analysis or noise estimation, but must not truncate pauses or discard audio.
- Offer both hold-to-talk and tap-to-start/tap-to-stop. Holding can be inaccessible after stroke-related weakness.
- Preserve the local recording if the network fails; allow retrying transcription without asking the person to speak again.
- Provide “Cancel,” “Add more,” and “Play my recording” controls. “Add more” appends another deliberate recording to the same thought.

### Candidate state

Start with **two candidates plus “Neither”**, but make this adaptive. Two choices can reduce overload; two wrong choices can also create forced consent.

Each candidate card should have:

- One short message in large, plain text.
- Optional key-word highlighting and a personally relevant image/symbol.
- A preview button that reads only that candidate privately or at low volume/headphones.
- An edit route using keyboard, word/picture tiles, or partner-assisted correction.

Always provide:

- **Neither / try again**
- **Use exactly what I said/transcript** when useful
- **Cancel**
- A way to compare the differing word or phrase when candidates are similar

Do not display “72% likely” merely because an LLM or recognizer emitted a score. Show “Likely” only after probabilities have been calibrated against that user's confirmed history. Before calibration, neutral labels such as “Option 1” and “Option 2” are more honest.

### Speaking state

- TTS runs only after an affirmative user action.
- Keep the confirmed text visible while it is spoken.
- Provide pause, stop, replay, volume, and speed controls.
- Let the user select a voice that fits their identity. Voice cloning is optional, separately consented, revocable, and never required.
- Maintain an offline system-voice fallback for loss of connectivity.
- Pin emergency and self-advocacy phrases locally, without an LLM: for example, “Please give me time,” “I understand you,” and user-defined medical/emergency statements.

## 4. System architecture

```text
manual audio capture
        |
        v
audio quality checks ----------> local encrypted audio/event store
        |
        v
ASR ensemble or personalized ASR
        |
        +--> N-best text + acoustic scores + token alternatives
        |
        v
candidate builder
  - preserves evidence from audio hypotheses
  - retrieves approved user memory
  - uses recent conversation context
  - generates/ranks semantically distinct messages
        |
        v
two candidate cards + neither/edit/retry
        |
        v
explicit user confirmation ------> TTS
        |
        v
append-only feedback event ------> safe personalization pipeline
```

### 4.1 Audio capture and preprocessing

- Capture mono PCM at 16 kHz or better before encoding; retain an unmodified source during the active transaction.
- Apply conservative denoising and gain normalization as alternate inference views, not destructive replacements. Benchmark raw and processed audio.
- Detect clipping, very low input, overlapping speakers, and excessive noise. Report a simple actionable state rather than rejecting slow or interrupted speech.
- Do not assume every silence is accidental. Long pauses are signal-bearing for this population.

### 4.2 ASR strategy

Use a staged strategy rather than betting the product on one recognizer:

1. **Baseline bake-off:** Evaluate at least one strong cloud recognizer and two locally deployable speech foundation models on the actual pilot cohort. Require N-best output or simulate diversity through beam decoding, model ensembles, and alternate preprocessing.
2. **Atypical-speech group model:** Adapt a pretrained model on appropriately licensed disordered-speech data. Keep train/validation/test splits speaker-disjoint and phrase-disjoint.
3. **Per-user adapter:** Periodically train a small speaker adapter or parameter-efficient layer set from confirmed audio/text pairs. Keep the general model fixed and version every adapter.
4. **Personal vocabulary biasing:** Maintain pronunciation examples and contextual bias for user-approved names, places, foods, medications, and recurring phrases.

Google's Project Euphonia research found large gains from personalization: one study reported 62% relative WER improvement for ALS speech and found that 71% of the eventual improvement came from five minutes of training audio. A larger study of 432 people with disordered speech reported a median WER of 4.6% for personalized models versus 31% for speaker-independent models on short phrases. These are strong reasons to personalize, but they are not direct performance promises for aphasia, spontaneous conversation, other languages, or Echora's cohort.

The product must preserve uncertainty. Store an N-best list or lattice with acoustic scores rather than passing only a polished 1-best transcript to the LLM. Research on rescoring and cross-modal error correction supports using alternate hypotheses, but text-only “correction” can introduce fluent words unsupported by the audio. Echora's candidate builder should therefore retain acoustic evidence and be trained to abstain.

### 4.3 Candidate builder and constrained LLM

Inputs:

- N-best ASR hypotheses, token alternatives, and recognizer scores
- User-approved vocabulary, people, places, routines, and style examples
- The last few confirmed conversational turns, with speaker labels
- Optional context deliberately supplied by the user, such as “at the doctor”
- Negative constraints: rejected candidates and critical entities that must not be changed

Outputs should follow a strict schema:

```json
{
  "candidates": [
    {
      "text": "...",
      "evidence_hypotheses": [0, 2],
      "uncertain_spans": ["..."],
      "critical_tokens": ["..."]
    }
  ],
  "abstain": false,
  "reason_code": "sufficient_evidence"
}
```

Guardrails:

- Generate zero to three candidates, then semantically cluster and return at most two distinct choices initially.
- Never create facts, people, negation, numbers, dosage, money amounts, dates, addresses, consent, or emergency status without support in the ASR evidence or explicit user memory/context.
- Prefer short, literal, first-person messages. Do not make them more polite, emotional, grammatical, or verbose unless this matches the user's approved style.
- If top candidates differ only cosmetically, show one. If the unresolved difference changes meaning, highlight it.
- Abstain when evidence is insufficient. “I’m not sure—please try again or choose words/pictures” is a successful safety behavior.
- Validate the schema, run contradiction and critical-token checks, and filter unsafe or unsupported output before display.

Use a lightweight reranker before generative rewriting whenever an existing ASR hypothesis already expresses the likely meaning. Generation is for repairing fragments or combining evidence, not a mandatory stylistic pass.

### 4.4 Ranking and confidence

A candidate score can combine:

- normalized acoustic likelihood
- ASR agreement across models/preprocessing views
- semantic support across the N-best set
- match to the user's vocabulary and approved style
- fit with the current conversational topic
- personalized selection/rejection history
- penalties for unsupported or safety-critical changes

Train the first personalized component as a ranker, not a free-running generator. Pairwise preference data (“A selected over B,” “neither,” edited result) is more data-efficient and easier to audit. Calibrate the final score per cohort and, once enough examples exist, per user using held-out events. Track expected calibration error and Brier score; do not equate softmax values with real-world correctness.

## 5. Memory design

Memory should be explicit, editable, exportable, and divided by function.

### A. Acoustic memory

Confirmed `(audio, transcript)` pairs, recording conditions, and model version. This trains or retrieves the user's speech patterns. A rejected interpretation is not a positive transcript unless the user supplies the correction.

### B. Lexical/knowledge memory

User-approved names, places, relationships, routines, medicines, interests, and preferred phrases. Each fact has provenance, date, confidence, scope, and a delete/edit control.

### C. Voice and style memory

Examples of how the person wants messages worded: concise or expressive, formal or casual, preferred greetings, dialect, bilingual choices, humor, and phrases they never want suggested. This is separate from biographical facts.

### D. Ephemeral conversational memory

Recent confirmed turns, current location/context if deliberately shared, and conversation partner. This expires quickly and should not become a durable personal fact by default.

### Feedback event

For every interaction, append:

```text
audio_id, timestamp, context_id, asr_model_versions,
n_best_hypotheses_and_scores, displayed_candidates,
selected_candidate | neither | retry | edited_text,
time_to_selection, number_of_repairs, tts_spoken,
consent_scope, user_profile_version
```

Do not update production behavior immediately from a single event. Batch candidate updates, test them against a frozen personal validation set, compare critical-error and selection metrics, then promote or roll back the version. Weight recent events enough to adapt to recovery or progression, while detecting rather than blindly learning temporary noise.

## 6. Bootstrapping personalization

Google Project Relate asked users to record 500 phrases and allowed custom cards for proper nouns, but currently is not accepting new sign-ups. Its exact model and private Euphonia corpus cannot simply be reused. Euphonia does provide research code/resources, and the Speech Accessibility Project offers data through an organizational data-use agreement.

Echora should reduce onboarding fatigue through progressive bootstrapping:

### Session 1: useful immediately (10–15 minutes)

- Choose access mode, text size, image support, voice, and language.
- Add 10–20 essential phrases and 10–20 high-value names/places/items.
- Record only what the person can comfortably produce. Allow rest and multiple sessions.
- Establish a speaker-independent baseline and report its quality honestly.

### First week: active learning (approximately 5–15 minutes total speech)

- Select recording prompts that maximize phonetic/lexical coverage and daily usefulness.
- Mix prompted phrases with picture descriptions, personally meaningful speech, and natural conversation. Read-speech-only adaptation may not transfer to spontaneous aphasic speech.
- Ask for repetitions of important names in several phrases, following the useful pattern in Project Relate custom cards.
- Prefer prompts the user can genuinely say; do not label a failed attempt as though it exactly matches printed text.
- Use SLP/user/caregiver verification only with consent and retain who supplied each label.

### Passive improvement

- Convert every confirmed real interaction into a potential training example.
- Use active learning to ask for a correction only when it is likely to improve the model and the user is not in a time-critical conversation.
- Offer an optional “practice and improve” session separate from normal communication.

### Optional historical sources

With explicit permission, import the user's sent messages, writing samples, phrase books, and recordings to learn vocabulary and style. Pre-aphasia material may help represent identity, but the user must decide whether that past style still represents them. Caregiver-provided profiles are suggestions pending user approval, not ground truth.

## 7. Data and model resources

- **AphasiaBank:** Valuable for aphasic discourse research, but access is protected and intended for consortium researchers/clinicians. Confirm the permitted use before any product training.
- **Speech Accessibility Project:** Includes dysarthria, apraxia, dysphonia, and other atypical speech and is available to organizations under a data-use agreement. Check diagnosis/language coverage against the intended cohort.
- **Project Euphonia:** Its dataset is not available to outside teams or companies. Use its published findings and open-source research resources, not an assumed data entitlement.
- **Participant-collected Echora corpus:** This becomes the most relevant resource. Obtain informed, revocable consent separately for providing the service, personalization, human review, and research/model improvement.

Dataset design must represent severity, etiology, age, gender, accent/dialect, primary language, code-switching, device/microphone, room noise, fatigue, prompted versus spontaneous speech, and co-occurring motor-speech conditions. Publish model cards and subgroup results; a good overall WER can hide a product that fails its most affected users.

## 8. Privacy, agency, and safety

- Default to encrypted local storage and on-device retrieval of the user profile. Upload only what the selected processing mode requires.
- Provide separate toggles for cloud transcription, cloud LLM interpretation, saving audio, personalized training, caregiver access, and research donation.
- Let users inspect, edit, export, and delete each durable memory and all recordings.
- Use short retention for raw cloud audio; document deletion behavior and subprocessor use.
- Never train a shared model from personal conversations without separate opt-in consent.
- Encrypt in transit and at rest; use tenant isolation, least privilege, audit logs, key rotation, and signed model versions.
- Never speak automatically. The user must be able to stop TTS instantly.
- Treat identity, negation, numbers, medication, finances, legal consent, emergencies, and intimate content as critical spans requiring stronger confirmation.
- Provide a private-screen mode so candidates are not exposed to the conversation partner before selection.
- Include a “do not learn from this” action for sensitive or anomalous conversations.

Research with AAC users shows that LLM suggestions may save time and effort but can impede communication when they fail to reflect the user's style. Studies specifically with people with aphasia also report intent mismatches and added cognitive burden from changing AI outputs. Agency and consistency are product-quality metrics, not merely ethics copy.

## 9. Evaluation plan

### Offline technical metrics

Evaluate at the utterance, speaker, subgroup, and critical-token levels:

- WER and character error rate
- meaning-preservation rating, with blinded human evaluation as the reference
- top-1 and top-2 intended-message recall
- oracle N-best WER/meaning recall (whether the correct answer was available at all)
- candidate precision and abstention quality
- exact accuracy for names, numbers, negation, medication, and emergency terms
- hallucination/unsupported-fact rate
- expected calibration error and Brier score
- median/P90 latency, failure rate, battery, network, and memory use
- personalized improvement versus the same user's unpersonalized baseline

WER alone is insufficient. Google research found that an LLM-based meaning-preservation assessment tracked human judgments better than common text metrics for disordered-speech transcription. Echora should use a similar metric for scalable iteration, while retaining human raters as the gold standard and auditing the evaluator for bias.

### User-centered outcomes

- successful communication of intended meaning
- time from press to spoken message
- number of taps, retries, and repair turns
- selection accuracy and “neither” rate
- fatigue and perceived cognitive load
- frustration, trust, autonomy, and authenticity of voice
- conversation-partner comprehension
- voluntary continued use and abandonment
- performance by fatigue level, environment, topic, and conversation partner

### Study sequence

1. **Co-design:** 8–12 people with varied aphasia profiles, 4–6 communication partners, and 3–5 SLPs. Test paper/Wizard-of-Oz candidates before building model complexity.
2. **Formative lab evaluation:** 12–20 users; within-subject comparison of 1-best transcript, two unpersonalized candidates, and two personalized candidates. Counterbalance conditions.
3. **Home feasibility pilot:** 15–30 users for 6–8 weeks. Measure learning curves, fatigue, spontaneous use, repairs, and privacy expectations.
4. **Larger prospective evaluation:** Pre-register primary outcomes and power analysis; use participant-level splits and compare against each person's existing communication method, not only generic ASR.

Compensate participants, use aphasia-accessible consent materials, permit supported consent without transferring authorship, build in breaks, and create a protocol for distress or fatigue. Obtain ethics/IRB review for research and specialist advice on medical-device classification in each launch jurisdiction.

## 10. Delivery roadmap

### Phase 0 — Discovery and protocol (3–4 weeks)

- Recruit an SLP/AAC clinical lead and paid lived-experience advisory group.
- Define the first cohort, contexts, languages, consent model, and critical-risk taxonomy.
- Create task scripts, paper prototypes, data schema, and evaluation protocol.
- Verify dataset licenses and vendor data-retention terms.

**Exit:** Users can reliably understand and operate the proposed record/choose/speak interaction in low-fidelity tests.

### Phase 1 — Thin vertical prototype (4–6 weeks)

- Manual push/tap recording, local buffering, one baseline ASR, N-best capture, constrained candidate generation, two-card confirmation, TTS, and neither/retry/edit.
- Implement audit events and strict no-speech-before-confirmation rule.
- Run Wizard-of-Oz and internal latency/failure testing with synthetic/non-sensitive data before participant recordings.

**Exit:** End-to-end task works, including offline/network failure recovery, with no automatic endpointing.

### Phase 2 — Baseline cohort study (4–6 weeks)

- Run ASR bake-off on consented pilot audio.
- Establish top-2 recall, semantic error, hallucination, critical-token, latency, and usability baselines.
- Refine visual cards, symbols/photos, access modes, and abstention thresholds.

**Exit:** Predefined safety thresholds are met and the top-2 system materially improves task success over a 1-best transcript for the target cohort.

### Phase 3 — Memory and personalization (6–10 weeks)

- Add explicit profile/memory editor, personal vocabulary, recent-context expiration, event review, and deletion/export.
- Train a personalized ranker first; then experiment with per-user ASR adapters.
- Add staged onboarding and active learning, version promotion, validation, and rollback.

**Exit:** Held-out personal data shows improvement without worse critical errors or subgroup regressions.

### Phase 4 — Home pilot and hardening (8–12 weeks)

- Deploy to a small cohort with SLP onboarding and support.
- Add monitoring, encrypted synchronization if needed, offline TTS, accessibility QA, incident response, and model cards.
- Analyze adoption, fatigue, selection behavior, drift, and real conversational repairs.

**Exit:** Users choose to continue using it; reliability, privacy, and communication-success targets are met in daily life.

### Phase 5 — Expansion

- Broader comprehension/severity profiles and alternative access methods
- More languages and code-switching, each separately evaluated
- On-device ASR/LLM paths and federated or local personalization where feasible
- Optional camera/photo/scene support and conversation preparation
- Carefully studied endpoint assistance that suggests—not triggers—recording completion

## 11. Initial success gates

Set final thresholds with the advisory group, but useful starting gates are:

- 100% of audible TTS messages have an explicit confirmation event.
- Zero accepted critical facts invented by the interpretation layer in the safety test set.
- At least 90% top-2 intended-message recall on the defined, non-critical pilot task set, reported with confidence intervals and subgroup breakdowns.
- A meaningful reduction in repair turns or message-completion time versus the user's baseline method, without worse perceived agency/authenticity.
- P90 release-to-candidates latency below 4 seconds on the supported network/device class; local capture and recovery always remain available.
- Every memory is traceable, editable, exportable, and deletable.

These are product gates, not claims that the system will work for every person with aphasia.

## 12. Team

Minimum core team:

- product lead
- SLP/AAC clinical lead
- lived-experience advisors and communication partners
- mobile/accessibility engineer
- speech ML engineer
- applied LLM/ML engineer
- backend/privacy engineer
- UX researcher/designer experienced in cognitive accessibility
- security/privacy and regulatory/ethics support

Do not defer user research until after model selection. The confirmation interface is part of the safety system and must be developed with the people who will use it.

## 13. Research basis and reading list

1. [ASHA: Aphasia practice portal](https://www.asha.org/practice-portal/clinical-topics/aphasia/) — clinical scope, multimodal communication, partner training, AAC, and supported conversation.
2. [ASHA: Communication Access Resources](https://www.asha.org/practice/communication-access/communication-aids/) — plain language, highlighted keywords, and questions with choices.
3. [Shor et al. (2019), Personalizing ASR for Dysarthric and Accented Speech with Limited Data](https://www.isca-archive.org/interspeech_2019/shor19_interspeech.html) — limited-data personalized adaptation.
4. [Green et al. (2021), Personalized Models Outperforming Human Listeners on Short Phrases](https://www.isca-archive.org/interspeech_2021/green21_interspeech.html) — large-cohort personalized disordered-speech ASR results and limitations.
5. [Le & Mower Provost (2016), Improving Automatic Recognition of Aphasic Speech with AphasiaBank](https://www.isca-archive.org/interspeech_2016/le16b_interspeech.html) — aphasia-specific recognition and adaptation.
6. [Zwilling et al. (2025), Speech Accessibility Project best practices](https://www.isca-archive.org/interspeech_2025/zwilling25_interspeech.html) — collection and curation of diverse disordered speech.
7. [Project Euphonia](https://sites.research.google/euphonia/about/) and [Project Relate help](https://sites.research.google/relate/help/) — available research resources, dataset limits, 500-phrase onboarding, custom cards, and current availability.
8. [Kang et al. (2024), Transformer-based ASR N-best Rescoring and Rewriting](https://www.isca-archive.org/interspeech_2024/kang24c_interspeech.html) — use of full N-best context.
9. [Radhakrishnan et al. (2023), Whispering LLaMA](https://aclanthology.org/2023.emnlp-main.618/) — cross-modal acoustic and linguistic evidence for ASR correction.
10. [Tomanek et al. (2024), LLMs as a Proxy for Human Evaluation of Disordered Speech Transcription](https://research.google/pubs/large-language-models-as-a-proxy-for-human-evaluation-in-assessing-the-comprehensibility-of-disordered-speech-transcription/) — meaning-preservation evaluation beyond WER.
11. [Valencia et al. (2023), “The less I type, the better”](https://discovery.ucl.ac.uk/id/eprint/10185220/) — benefits and agency/style risks of LLM suggestions for AAC users.
12. [Mao et al. (2025), Design Probes for AI-Driven AAC in Aphasia](https://arxiv.org/abs/2504.09435) — multimodal verification, intent mismatch, personalization, and cognitive load with people with aphasia.
13. [Koul & Harding (2017), visual scene versus grid displays](https://www.tandfonline.com/doi/abs/10.1080/02687038.2016.1274874) — evidence that contextual photographic scenes may require fewer cognitive resources for some people with chronic aphasia.
14. [AphasiaBank](https://talkbank.org/aphasia/) and [Speech Accessibility Project data access](https://speechaccessibilityproject.beckman.illinois.edu/conduct-research-through-the-project) — access and use constraints.

## 14. First decisions to make

Before implementation, the product team must explicitly decide:

1. The exact first-user cohort and primary language.
2. Whether the first release is face-to-face communication, remote messaging, or both.
3. Which operations must work offline.
4. Whether raw audio leaves the device, for how long, and under which consent.
5. How users who cannot read candidate sentences will verify intent.
6. Which safety-critical domains are excluded from the first pilot.
7. The comparator and primary outcome for the first study.

The recommended first scope is face-to-face, short-message communication in one language, for adults with chronic expressive aphasia who can independently confirm two large text-plus-audio choices, with medical, financial, legal, and emergency inference excluded. This scope is narrow enough to evaluate responsibly and broad enough to demonstrate the core value.
