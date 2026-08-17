# Application architecture

The browser sends one completed recording to the local FastAPI API. FastAPI normalizes it to 16 kHz mono audio and invokes exactly one configured ASR backend: local MPS, a persistent Pod, or RunPod Serverless.

The ASR worker returns literal hypotheses only. The local API then runs a contextual Groq interpreter. Alongside the beams it receives a slot alignment that aligns them position by position, so stable words and unresolved words are separated before any judgement is made, plus the per-term share of search weight and of beams. It groups supporting hypothesis IDs, assigns a constrained speech act, and extracts key terms that must occur in the cited literal evidence. A key term is retained only when it holds a strict majority of the grouped beams by both search weight and beam count, so grouping more beams no longer discards the words that distinguish them. General, Home, Hospital/care, and Outdoors are session-only settings and are treated as weak priors.

A deterministic clarity gate combines beam margin, normalized entropy, grouped search weight, semantic evidence strength, and explicit speech-act cues. A speech-act cue alone cannot clear an interpretation whose grouped beams offer competing alternatives for the same slot; that contested case stays ambiguous. A clear interpretation proceeds as one message; genuine uncertainty exposes up to three interpreted alternatives. The search features are relative evidence, not calibrated confidence.

A second Groq request realizes every displayed interpretation as natural communication wording. A deterministic lexical grounding check rejects new substantive terms or dropped key terms and substitutes a conservative grounded message. The original ASR hypotheses are never mutated.

The UI primarily shows post-chain communication suggestions and keeps literal evidence expandable and attached to every option. A suggested message is editable and is never treated as an ASR evaluation result. Once the speaker has settled on a message it is spoken without a further confirmation step: an unambiguous result is synthesized during the transcription request and plays on arrival, and choosing among ambiguous options speaks the chosen one. Groq TTS failures fall back to the browser's own voice rather than surfacing an error.

Remote GPU workers never receive the Groq key and never perform semantic repair. They expose the same raw-ASR schema as the local engine.
