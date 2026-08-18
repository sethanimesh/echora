# Application architecture

The browser sends one completed recording to the local FastAPI API. FastAPI normalizes it to 16 kHz mono audio and invokes exactly one configured ASR backend: local MPS, a persistent Pod, or RunPod Serverless.

The ASR worker returns literal hypotheses only. The local API then runs a contextual Groq interpreter -- one call, not two. Alongside the beams it receives a slot alignment that aligns them position by position, so stable words and unresolved words are separated before any judgement is made, plus the per-term share of search weight and of beams. It groups supporting hypothesis IDs, assigns a constrained speech act, and extracts key terms that must occur in the cited literal evidence. A key term is retained only when it holds a strict majority of the grouped beams by both search weight and beam count, so grouping more beams no longer discards the words that distinguish them. General, Home, Hospital/care, and Outdoors are session-only settings and are treated as weak priors. Beside the setting travels one other closed value: whether the person being spoken to knows the speaker. That is what decides the act rather than the words -- a familiar listener can fetch and do, so a need is stated to them, while an unfamiliar one can only answer, so the same need is asked. It is two values, orthogonal to the four settings and never a fifth one; Outdoors defaults to unfamiliar and everything else to familiar. A place declares which applies where, a profile may say what a setting usually means for that speaker, and the setting's own default stands when neither does. The setting reaches the model through one guidance sentence keyed by the pair, and the evidence keys around it are unchanged. The speaker reaches them through named places, which resolve to one of those four and never extend the set: a place the speaker adds declares which of the four it borrows, so its prior and its retrieval pool are that built-in's. Places, and the switch that lets location choose between them, persist in `data/personal/settings.json`; the setting a given utterance was spoken in is still session-only.

That one call returns both parts of every option: the reading, which is the recognizer's own words with one chosen per position, and the message, which is that reading realized as natural communication wording. Clear versus ambiguous is then decided in code, after the grounding filters have dropped what they drop: one surviving message is a clear interpretation, and anything else exposes up to three alternatives. A deterministic lexical grounding check rejects a reading containing a word no beam produced, and a separate audit drops any option carrying profile wording that neither a beam nor a licensed detail explains. The original ASR hypotheses are never mutated. The search features are relative evidence, not calibrated confidence.

The UI primarily shows post-chain communication suggestions and keeps literal evidence expandable and attached to every option. A suggested message is editable and is never treated as an ASR evaluation result. Once the speaker has settled on a message it is spoken without a further confirmation step: an unambiguous result is synthesized during the transcription request and plays on arrival, and choosing among ambiguous options speaks the chosen one. Groq TTS failures fall back to the browser's own voice rather than surfacing an error.

When a profile is selected, a personal brief is assembled before the Groq call and appended after the
evidence keys, so the model reads what was heard before it reads anything about who was speaking. The
brief carries three things: known words, restricted to entries the beams actually produced and to the
settings that entry applies in; declared details, restricted to anchors sitting in a stable slot and
likewise to their own settings; and up to four past accepted messages retrieved by cosine over a
local mean-pooled encoder, scored down by age, mismatched setting, mismatched listener and mismatched
time of day. The listener penalty is the harshest of those, because an example addressed to a
different kind of person is not merely less relevant -- its shape is wrong, and shape is what a
few-shot example teaches. An un-personalized request sends a byte-identical payload to the one sent
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
similarity threshold *that was accepted to the same kind of listener*, taking its count up and its
vector toward the new phrasing, and the store is bounded by the same score retrieval ranks with. The
listener gate is what lets per-place phrasing accumulate at all -- a merge takes the longer wording
and stamps the newcomer's setting over the old one, so without it a stranger-facing message would
quietly absorb the one it was meant to sit beside. Shipped personas are read-only; a profile is copied
into a gitignored live store on first use. Every failure in this layer returns nothing and is silent.

Remote GPU workers never receive the Groq key and never perform semantic repair. They expose the same raw-ASR schema as the local engine.

Location, when the speaker turns it on, only chooses among places they tagged themselves. A place is tagged by
standing in it and asking the browser once for its coordinates -- there is no address lookup, no map provider, and
no outbound request. Matching runs in the browser against a radius, widened by the reported accuracy of the fix
because an indoor reading is routinely tens of metres out, so coordinates never reach the API and the transcription
request is unchanged: it still carries only the borrowed context. Detection is confined to the idle screen, names
the place it picked, and yields permanently to a tap. Every failure -- permission refused, no fix, nothing tagged,
an unreadable document -- leaves the speaker choosing by hand, exactly as before the layer existed.
