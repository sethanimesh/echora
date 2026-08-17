# Application architecture

The browser sends one completed recording to the local FastAPI API. FastAPI normalizes it to 16 kHz mono audio and invokes exactly one configured ASR backend: local MPS, a persistent Pod, or RunPod Serverless.

The ASR worker returns literal hypotheses only. The local API then runs a contextual Groq interpreter. Alongside the beams it receives a slot alignment that aligns them position by position, so stable words and unresolved words are separated before any judgement is made, plus the per-term share of search weight and of beams. It groups supporting hypothesis IDs, assigns a constrained speech act, and extracts key terms that must occur in the cited literal evidence. A key term is retained only when it holds a strict majority of the grouped beams by both search weight and beam count, so grouping more beams no longer discards the words that distinguish them. General, Home, Hospital/care, and Outdoors are session-only settings and are treated as weak priors.

A deterministic clarity gate combines beam margin, normalized entropy, grouped search weight, semantic evidence strength, and explicit speech-act cues. A speech-act cue alone cannot clear an interpretation whose grouped beams offer competing alternatives for the same slot; that contested case stays ambiguous. A clear interpretation proceeds as one message; genuine uncertainty exposes up to three interpreted alternatives. The search features are relative evidence, not calibrated confidence.

A second Groq request realizes every displayed interpretation as natural communication wording. A deterministic lexical grounding check rejects new substantive terms or dropped key terms and substitutes a conservative grounded message. The original ASR hypotheses are never mutated.

The UI primarily shows post-chain communication suggestions and keeps literal evidence expandable and attached to every option. A suggested message is editable and is never treated as an ASR evaluation result. Once the speaker has settled on a message it is spoken without a further confirmation step: an unambiguous result is synthesized during the transcription request and plays on arrival, and choosing among ambiguous options speaks the chosen one. Groq TTS failures fall back to the browser's own voice rather than surfacing an error.

When a profile is selected, a personal brief is assembled before the Groq call and appended after the
evidence keys, so the model reads what was heard before it reads anything about who was speaking. The
brief carries three things: known words, restricted to entries the beams actually produced; declared
details, restricted to anchors sitting in a stable slot; and up to four past accepted messages
retrieved by cosine over a local mean-pooled encoder, scored down by age, mismatched setting and
mismatched time of day. An un-personalized request sends a byte-identical payload to the one sent
before this layer existed.

The grounding check on the reading is unchanged. Specializations are audited beside it: a declared
detail survives only if the profile declared that exact wording, its source matches, its anchor is in
the reading and in a stable slot, and the wording appears in the message. A refused detail is stripped
back to the plain wording rather than failing the message. Separately, any profile word that no beam
produced and no licensed detail explains rejects the option outright, which closes a gap the reading
check cannot see: a message may legitimately contain words no beam carried, so `Marge` could otherwise
reach the speaker's mouth from a reading of `march`. The plain wording is derived in code by
substituting each applied surface back, so reverting is mechanical rather than promised.

Accepted messages are consolidated on write: an acceptance merges into the nearest entry above a
similarity threshold, taking its count up and its vector toward the new phrasing, and the store is
bounded by the same score retrieval ranks with. Shipped personas are read-only; a profile is copied
into a gitignored live store on first use. Every failure in this layer returns nothing and is silent.

Remote GPU workers never receive the Groq key and never perform semantic repair. They expose the same raw-ASR schema as the local engine.
