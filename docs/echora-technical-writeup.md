# Echora: a practical communication pipeline for difficult-to-understand speech

Echora is a local-first communication assistant for people whose speech is hard for a typical speech recognizer to understand after stroke or similar neurological injury. It turns uncertain speech into a small number of useful communication choices, keeps the original recognition evidence visible, and leaves the decision with the speaker when the system is unsure.

## 1. Starting with speech recognition

We started with Qwen3-ASR 1.7B, a general speech-recognition model. It struggles with dysarthric and very short, effortful utterances. For this use case, one polished-looking transcript is not enough: its most important word may be wrong.

We fine-tuned selected audio layers, the multimodal projector, and small LoRA adapters on genuine TORGO dysarthric recordings plus controlled two- and three-word command compositions from those same speakers. The model was trained to produce literal wording, including incomplete or unusual word sequences, rather than fill in what it thought the speaker meant.

The deployed epoch-7 `echora-qwen3-asr-command-v3` adapter produces up to five literal hypotheses (beam-search alternatives), not five invented interpretations. On the protected command test, the exact phrase appeared somewhere in the top five for 36.46% of utterances, compared with 16.04% for the earlier adapter. This is why Echora uses a set of alternatives rather than trusting only the first transcript.

## 2. Turning alternatives into communication options

The ASR output is evidence, not the final message. Echora sends the aligned alternatives to one structured Groq call, with lightweight context such as home, care, or outdoors. The model returns a literal reading assembled only from recognized words and a short, natural message for each plausible meaning.

We then apply deterministic checks outside the language model. A reading is rejected if it contains a word absent from the ASR alternatives. Important terms must be supported by the beams, and personal wording is accepted only when there is an explicit, valid anchor for it. This stops a fluent response silently becoming a fabricated message.

The local adapted English route uses a separate acoustic verification decision. Its scorer reads frozen Qwen features and literal text; scoped profile context enters bounded fusion afterward. Learned automatic selection requires identity-checked artifacts and a successful acceptance audit. Missing, uncalibrated, or rejected artifacts leave the route asking the speaker to choose, even when the wording pass produces only one message. Other recognizers retain a labelled legacy decision route. All distinct literal hypotheses remain selectable, including when the cloud wording step fails. See [verification and memory](verification-and-memory.md) for the experiment and acceptance criteria.

Who the speaker is talking to also changes the message form. A familiar carer or family member can act on a need, so the message can be direct: “I want to use the washroom.” A stranger can mainly answer, point, or serve, so a place becomes “Where is the washroom?” and a thing uses the complete service request “Could you please get me …, please.”

## 3. Personalisation without treating a profile as evidence

Personal context is optional, stored locally, and has two separate purposes:

- A profile can help choose between words the recognizer already heard. For example, `[marge | march | large] tea` can resolve to a known carer called Marge, but only because “Marge” was already one of the recognition alternatives.
- A profile can add an explicitly defined detail, such as turning “coffee” into “Madras filter coffee.” This is allowed only when the profile declared that exact wording and the anchor's conservative spelling family holds enough search weight. Once eligible, code applies it deterministically rather than asking the language model whether to keep it.

Profiles and explicitly remembered wording use the local SQLite store. The last approved message may supply a narrow follow-up for at most ten minutes within the same context, with a visible reference, Forget control, and profile/scope checks. Only an explicit Remember action persists the exact displayed message under the selected profile. Arrival, selection and playback never save history automatically, and remembered wording is not verified acoustic training truth.

## 4. Speaking the result

Once a completed suggestion is resolved, it may speak automatically. Selecting an alternative speaks the chosen displayed words. Internal authorization binds playback to the current revision; edits, Stop, new work, and context changes invalidate pending playback, and restoring a session never replays old speech. Device speech and configured Groq/Fish providers support delivery; provider failure retains the message and offers device speech. `ECHORA_SPEECH_AUTOPLAY=false` suppresses arrival speech while retaining explicit choices. Provider-backed features send their required inputs to the selected provider.

## What worked

- Fine-tuning for literal, short-command recognition improved protected command WER to 51.58%, versus 72.58% for the earlier adapter. It also preserved normal-speech performance at 5.23% WER.
- Returning five ASR alternatives materially improved the chance that the exact short command was available to the downstream pipeline.
- The grounded message layer produces evidence-supported wording while retaining literal alternatives. Automated fidelity and lifecycle checks exercise these contracts; they do not establish usability or speech quality for an individual.
- Keeping recognition, message generation, personalisation, and speech synthesis as separate layers makes failure safer: each layer can fall back without pretending to know more than it does.

## What did not work, or remains limited

- The recognizer is not reliable enough to be treated as a universal dysarthric-ASR solution. Protected TORGO WER was 55.02%, and the exact phrase is often absent even from the top five alternatives.
- The known phrase “I water” is still missed entirely by the deployed model’s top five. In cases like this, the interface must preserve alternatives and ask rather than confidently guess.
- The command data are controlled compositions, not newly recorded natural conversations. They help model pauses, stretching, and short word sequences, but they do not add new speakers or prove real-world performance.
- Beam-search weights are useful relative evidence inside one recognition run, but they are not calibrated confidence scores. We therefore do not use a numerical confidence threshold to decide whether a message is safe.
- Personalisation improves relevance only when the underlying recognition contains enough evidence. It cannot recover a word the model never produced.

## Further technical detail

The [architecture](architecture.md), [evaluation](evaluation.md), and [failure analysis](failure-analysis.md) connect the implementation to measured results and limitations. Model adaptation, acoustic verification, grounded wording, scoped context, and revision-bound playback remain separate mechanisms with inspectable contracts.
