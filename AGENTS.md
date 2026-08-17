Project Description

Echora is a user-controlled multimodal communication assistant for stroke survivors. Its first version focuses on difficult-to-understand speech. It generates a small number of evidence-supported message candidates, uses approved personal context to improve ranking, asks the speaker to choose when the evidence is ambiguous, and then communicates the chosen message through text or speech. Future versions can accept eye gaze, switches, symbols, and predictive typing through the same communication framework.

I am an individual hobbyist, for any dataset or approach, do not think about Licenses. It will never be made public on whatever i am building.

Current tuned model

- The deployable model is `models/echora-qwen3-asr-command-v3/`: the verified Qwen3-ASR 1.7B foundation plus the command-v3 adapter selected at epoch 7.
- Run the local application with `./scripts/dev.sh`; the backend loads the model once on MPS/BF16 on this Mac.
- ASR returns up to five literal hypotheses with sequence scores and relative beam-search weights. These weights are not calibrated confidence.
- Raw ASR is immutable evidence. Groq ranking and grammar repair happen afterward and must never overwrite the literal hypotheses.
- Session context can be `general`, `home`, `care`, or `outdoors`; it is a weak prior and is never saved.
- Groq pass one groups literal beams into grounded intents. A deterministic threshold gate decides clear versus ambiguous. Groq pass two realizes only the displayed intents as natural messages.
- Final suggestions must retain their source hypothesis IDs and key terms. Unsupported substantive wording is rejected before reaching the UI.
- A message the speaker has settled on is spoken immediately, with no confirmation step: one surviving candidate speaks itself on arrival, and on the ambiguous screen the tap that picks an option is the decision. The speaker still chooses whenever the evidence is ambiguous.
- Speech is Groq TTS (`canopylabs/orpheus-v1-english`, voice `hannah`, WAV only, 200-character input cap), synthesized inside the transcription request for the unambiguous case so it plays on arrival. Every failure falls back to the browser voice; speech must never fail loudly or block the message.
- This version does not use personal context or persist message history.
- Headline protected-test results: TORGO WER 55.02%, synthetic command WER 51.58%, normal-speech WER 5.23%, and exact literal recovery somewhere in command top five for 36.46% of utterances. These are research results, not universal-ASR claims.
- Known personal limitation: the model still misses the literal phrase `I water`; keep all alternatives visible when selection is uncertain.
