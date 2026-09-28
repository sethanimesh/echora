# Two roleplay scripts

Two speakers, three settings each, the same words in all three. Nothing here is
a new feature: every difference below comes out of the listener stance in
`backend/app/messaging/groq_chain.py`, the `settings` list on a specialization,
or the precedence chain place → profile → setting default.

## Read this before you run anything

These plays make claims about three layers, and the layers fail independently.
An earlier version of this document wrote all three as one flat "You say /
Expected" table, which is why it read as broken: a row would miss, and there was
no way to tell which layer had missed or whether anything had missed at all.

**The act** — statement or question, longer or shorter. Decided by the setting
and the listener together, and *deterministic given both*. Every act row in
these plays reproduces. If one does not, that is a real finding.

**The detail** — a profile's own wording (`tea` → `my Lipton tea with milk`) in
place of a word that was heard. Three conditions, all required: the profile
declared it, it is in scope for this setting, **and the beams settled the word it
rides on** — the anchor must hold at least 75% of its slot's search weight
(`ECHORA_PERSONAL_ANCHOR_SHARE`). That third condition belongs to the recognizer,
not to you. A detail row is a conditional claim, and the condition is not one the
script can hold still while you speak into a microphone.

**The ambiguity** — two options rather than one. The disambiguation prior is a
filter over what the recognizer produced; it can only ever *choose among* those
words. It cannot manufacture a choice, so "make it offer me Donna and Dawn" is
not something you can do by pronouncing a name a particular way. Either both
words are in the beams or no layer of this system can put them there.

So there are two lanes, and the reason there are two is that one lane cannot
separate those layers.

### Lane one: the fixed lane

```
./scripts/roleplay.py --list          # the acts, and what a run costs
./scripts/roleplay.py --act One:II
```

The beams are written into the script with explicit search weights, so the
evidence cannot move. Only the setting, the listener and the profile change.
This is the lane the scoping and precedence claims live in: when the evidence is
held still, a detail that appears at home and not on the ward appeared because of
the scope and nothing else. It is also the only lane that can stage a genuine
ambiguity on purpose.

Every scene is one Groq call and the free tier is about 200k tokens a day, so
run an act at a time rather than the whole thing. `--list` prints the cost first.

### Lane two: the voice lane

```
./scripts/dev.sh                      # then read the lines aloud
./scripts/roleplay.py --act One:II --voice   # or let `say` stand in for you
```

The whole path, recognizer included. This is the honest end-to-end test and the
one that tells you whether the thing works for a person. It cannot demonstrate
scoping, because a line the recognizer half-hears drops the detail for a reason
that has nothing to do with scope — and it will half-hear, because that is what
these plays are about.

Say the lines the way the character would: Grace gets one or two content words
out and stops. Tomás says a whole sentence, slurred and effortful. Do not
enunciate for the machine.

### Reading a miss

When a detail does not appear, open **Evidence**. It now shows the stance the
message was actually written under — which is not always the one the screen
implies, because a place may declare nothing and let the profile decide — and
what the profile contributed. The panel says it in words; `./scripts/roleplay.py`
prints the same two counts as `offered= applied=`. Either way:

- **nothing offered** — the detail was never put in front of the model. Either
  it is scoped out of this setting, or the anchor did not hold enough of its
  slot. Check the beams: `soup` at 0.82 and `soup` at 0.30 are two different
  tests, and only the fixed lane lets you choose which one you are running.
- **offered and applied** — the detail is inserted deterministically in code,
  marked, and **Say it
  plainly instead** reverts it in one tap.

And one trap worth knowing. When the Groq call fails — a rate limit is the usual
reason — the result comes back as **ambiguous with the raw beams as the
options**, which looks exactly like a genuine ambiguity. The warning line above
the options says so, and `./scripts/roleplay.py` prints `DID NOT RUN`. Read it
before you write down that the disambiguation is broken.

Finally: the "expected" wording is what the stance asks for, not a string the
code guarantees. The model writes it. Judge a run on **shape** — statement or
question, longer or shorter, detail present or absent — not on matching the text
letter for letter.

### Setting up the interface

- **Who is speaking?** picks the profile. It also moves the place to the
  profile's default setting — but only if you have not already tapped a place
  yourself, because a tap outranks a profile.
- The chips under the microphone are **places**, not settings. `Home`, `Care`
  and `Outdoors` ship. Each one carries a listener, set in **Places** on the
  place itself, and it has three values, not two. *Leave it to whoever is
  speaking* is the shipped state and the only one that lets a profile be heard;
  the other two are declarations and outrank the profile.
- One surviving candidate speaks itself the moment it arrives. Two or three and
  you get the choice screen; the tap that picks one is the decision.

---

# Play One — Grace Okonkwo

*61, alone in a council flat in Manchester. Two years after a stroke. One or
two content words at a time, and she tires by the end of a sentence. Nigerian
accent the recognizer handles poorly. Independent-minded, slightly formal, asks
plainly with please.*

**Set up:** Who is speaking? → Grace. The place should land on `Home` by itself.

## Act I — Home. The carer has just come through the door.

The listener is **familiar**: Donna knows where the kitchen is and can be sent
to it. Grace states what she needs.

| You say | Expected | Holds when |
|---|---|---|
| "washroom" | *I need the washroom.* | Always. A need stated to someone who can act on it. |
| "tea" | *I need tea.* | **Generic on purpose.** Donna knows which tea she drinks, so the detail is scoped away from home. Contrast with Act III. |
| "soup" | *the egusi soup from the freezer* | `soup` is settled. Scoped `home` only — this is the one place it can appear. |
| "cream" | *Could you put on my leg cream from the blue tub?* | `cream` is settled. Scoped `home, care`. The blue tub matters; the white one is for hands. |
| "buzzer" | *Someone is at the buzzer.* | Always. Lexicon, not a specialization, so no anchor share to satisfy. |

**The ambiguous beat.** In the fixed lane this act runs the same word twice,
with the weight arranged two different ways.

With `donna 0.38 · dawn 0.34 · dorn 0.16 · danna 0.12` you should get **two**
options, *Donna.* and *Dawn.* Two, not four: `dorn` is a declared alias of Dawn
and `danna` of Donna, and the profile is what folds them. Two, not one: the
weight is genuinely split between two real people, so the system must ask rather
than commit — a single option is spoken the moment it arrives, and a wrong commit
is already said.

With `dawn 0.81 · dorn 0.11 · darn 0.08` you should get **one**. `donna` is not
in the evidence, and nothing in this system can put it there. That is not a
failure to disambiguate; it is the grounding rule working.

This is the beat the old script could not stage. It asked you to say "Dawn" into
a microphone and then recorded "it did not ask me" as a failure, when what had
actually happened is that the recognizer never produced `donna` — and no prior,
however good, is allowed to invent it.

## Act II — Care. Same words, respite ward.

Tap the **Care** chip. Still **familiar** — a nurse on duty can act.

| You say | Expected | What changed |
|---|---|---|
| "washroom" | *I need the washroom.* | Nothing. Both settings are familiar, so the act is the same. |
| "tea" | *I need my **Lipton tea with milk**.* | **Changed.** A rotating agency rota does not know which tea. Marked; **Say it plainly instead** → *my tea*. |
| "soup" | *Please bring me soup.* | **Changed.** Scoped `home` only, so the egusi is not offered on the ward. |
| "cream" | *my leg cream from the blue tub* | Still specialized. |

The soup line is the demonstration, and it only demonstrates anything if you can
rule out the other reason a detail goes missing. That is why this act runs
**"soup" a second time, at home, with the anchor unsettled** (`suit 0.34 · soup
0.30 · sleep 0.20 · suite 0.16`). Same setting as Act I, same word in the beams,
no detail — because `soup` holds 30% of its slot and a detail may not ride on a
word the recognizer is not sure of. Put the two rows side by side and the trace
reads the same (`offered=0`) while the cause is different, which is exactly why
the beams are printed.

The ward freezer does not have her egusi in it, so the detail is not offered —
but the audit vocabulary still holds the word, so if the model wrote "egusi"
here anyway the option would be rejected rather than shown. Scoping narrows what
is offered; it never widens what is accepted.

## Act III — Outdoors. The bus stop, then the chemist.

Tap **Outdoors**. Nothing is declared and Grace's profile says nothing about
outdoors, so the chain falls to the setting's default: **unfamiliar**.

| You say | Expected | What changed |
|---|---|---|
| "washroom" | *Where is the washroom?* | **Changed.** A stranger cannot take her; they can point. |
| "help" | *Could you help me?* | Asked directly, no preamble. |
| "tea" | *Could you please get me my **Lipton tea with milk**, please.* | **Changed twice.** The act became a complete counter request, and the detail appears — a counter cannot know which tea, so this is where the words are needed. |
| "water" | *Could you please get me water, please.* | A complete request over a counter, not a statement of want. |

The tea row shows the two layers working together: the listener changes a bare
need into a complete request, while the profile supplies the exact tea. The
detail is not decoration: it is what lets a stranger fulfil the request rather
than guess. Sales-clerk ratings of AAC users put information-bearing messages
above empty politeness in a time-pressured exchange (Bedrosian, Hoag & McCoy,
2008): what a stranger needs
is the thing being asked for, first time, without a preamble.

So the tea detail is scoped to exactly the settings where it is *needed*. Donna
has made her tea a hundred times and "my tea" is enough; a rotating ward rota and
a counter have not, and there "Lipton with milk" is the message rather than an
embellishment on it. The complete polite request must preserve those specifics.

That is per-detail, and it points different ways for different details. `soup` is
scoped `home` only, in the opposite direction, because "the egusi soup from the
freezer" means something in her own kitchen and nothing at all to a café. The
question a scope answers is not "how formal is this setting" — it is "does the
person in front of her need these words, and can they act on them?"

**Failure to watch for:** *"Excuse me, I'm sorry to bother you, but I wonder if
you could possibly tell me where the washroom is?"* That is the model being
polite at her expense while a stranger waits. Longer outdoors than at home is
always a bug.

## Act IV — Outdoors, but Ngozi has come up from London.

Grace is outdoors and her daughter is beside her. Same street, different act.

1. Open **Places**, find `Outdoors`, set it to **People here know me**. This is
   editable on a built-in on purpose — a speaker who only ever goes out with
   family needs exactly this — and it is a *declaration*, so it outranks
   whatever the profile would have said.
2. Back on the idle screen, tap **Outdoors**.

| You say | Expected | Why |
|---|---|---|
| "washroom" | *I need the washroom.* | Outdoors, and back to a statement. The setting did not change; the listener did. |
| "tea" | *Please bring my **Lipton tea with milk**.* | Same detail as Act III, stated rather than asked. The setting keeps the detail; the listener sets the act. |

Read the two rows together and the two knobs come apart cleanly. Against Act III:
the street did not change and neither did the detail, because the *setting* keeps
it — only the act moved, from a request put to a stranger to a need stated to her
daughter. Against Act I: the listener is familiar in both, and the detail is
present here and absent there, because the *setting* changed. Setting governs
what is said; listener governs how it is put.

Set `Outdoors` back to **Leave it to whoever is speaking** before you move on.

**Optional fifth scene.** Add a place called `Church` in **Places**, borrowing
`outdoors`, **People here know me** — she is out of the flat but among people
who know her. Then add `Chemist`, borrowing `outdoors`, **People here do not
know me**. Say "help" at each. Two custom places, one borrowed setting, two
different acts.

---

# Play Two — Tomás

*45, Brazilian-born, working part-time in Lisbon. Fourteen months after a
brainstem stroke. Mild dysarthria only: his language and judgement are entirely
intact, his articulation is not. He says whole adult sentences and does not want
them shortened for him. Alone at his desk or out in the city most of the day;
Inês is home in the evening.*

**Set up:** Who is speaking? → Tomás. His default is `general`, so no place chip
is selected. That is correct — General is the absence of a place, not a fourth
one.

## Act I — Home. Evening, Inês in the kitchen.

Tap **Home**. Listener **familiar**.

| You say | Expected | Why |
|---|---|---|
| "I need my tablet" | *Could you bring my baclofen tablet, the evening one?* | Scoped `home, care`. The evening dose is a different dose, which is exactly why the detail exists. |
| "where is my headset" | *Where is my noise cancelling headset?* | Scoped `home, general`. The other one picks up the whole room. |
| "coffee" | *I would like a short black coffee, no sugar.* | No `settings` list at all, so it applies everywhere. |
| "the standup is at ten" | *The standup is at ten.* | Lexicon only. His grammar is intact; the message should not be simplified for him. |

**Failure to watch for:** a two-word telegram. Grace's brevity is not his. If
"I need my tablet before the standup" comes back as *"Tablet."*, the system has
flattened a speaker who has all his words.

In the voice lane, expect the headset line to be the flakiest in either play:
`headset` reliably comes back as `head set` or `head sit`, and once `headset` is
not in the beams the detail has no anchor to ride on and the message can lose the
word entirely. That is a recognizer result, not a scoping one, and it is why the
fixed lane exists.

## Act II — Care. The neuro clinic, waiting on the physio.

Tap **Care**. Still **familiar**.

| You say | Expected | What changed |
|---|---|---|
| "I need my tablet" | *my baclofen tablet, the evening one* | Unchanged: `home, care`. |
| "where is my headset" | *Where is my headset?* | **Changed.** Scoped `home, general` — the clinic is neither, so the detail drops. |
| "coffee" | *a short black coffee, no sugar* | Unchanged: unscoped. |

Three specializations, three different scopes, one setting change, and the beams
held still. This is the cleanest place in either play to see that scoping is
per-detail and not a global switch.

## Act III — Outdoors. Cais do Sodré, then the counter.

Tap **Outdoors**. Nothing declared, and Tomás's profile says nothing about
outdoors, so: **unfamiliar**.

| You say | Expected | What changed |
|---|---|---|
| "which platform, Cascais" | *Which platform for the Cascais train?* | The `train` specialization is unscoped, so `the Cascais train` survives — and it matters, because the Sintra line leaves the same platform. |
| "coffee" | *A short black coffee, no sugar, please.* | The detail is forced once its anchor gate holds. |
| "washroom" | *Where is the washroom?* | The house example. Asked, not stated. |
| "I need my tablet" | *Could I have my tablet?* | **Changed.** `home, care` scoping drops the detail, and a stranger cannot be sent for his medication anyway. If the message still says "baclofen", that is a scoping failure worth recording. |

The **Cascais** line is the one that carries the point, and it carries it
reliably: the detail is *content he supplied* and survives, while the
statement→question flip is *act*, which the listener owns. Content and act move
in opposite directions in the same message.

The **coffee** line now holds whenever its anchor gate holds. Once the profile
offers "a short black coffee, no sugar", code applies the detail after Groq has
formed the message; the model cannot trim it as decoration.

## Act IV — The precedence chain, in three taps.

Tomás goes to the office with Duarte twice a week — outdoors, but not among
strangers.

1. **Places** → add `Office`, borrowing `outdoors`, **People here know me**.
2. Tap `Office`. Say "coffee" → a statement, because the place *declared* a
   familiar listener. Say the same thing on the `Outdoors` chip → a counter
   request. One borrowed setting, two places, two acts.
3. Now the layer underneath. Switch to **Krishnan** and tap `Outdoors`. His
   profile carries `listener_by_setting: {outdoors: familiar}` — he only goes out
   with Meera or Priya — so he gets the familiar register outdoors without any
   place having said so.

That is the whole chain, strongest first: **what the place declares**, then
**what the profile says that setting usually means for this speaker**, then
**the setting's own default**. Every step falls back, so a speaker with no
profile and no places behaves exactly as the setting alone always did.

Step 3 only works because `Outdoors` is left at **Leave it to whoever is
speaking**. A place that declares a listener — even one that happens to match
the setting's own default — outranks the profile and the middle rung is never
reached. If you set `Outdoors` to *People here do not know me* in Act IV of Play
One and forgot to put it back, Krishnan will get the unfamiliar register and the
chain will look broken.

---

## What to write down after a run

- Did the **act** change between familiar and unfamiliar? Statement at home,
  location question or complete service request among strangers.
- Did an unfamiliar request for a thing use the complete **Could you please get
  me …, please** form, while a location remained a direct question?
- Did each **specialization** appear only in the settings it lists — with the
  anchor share held constant — and did **Say it plainly instead** put the plain
  wording back in one tap?
- When the evidence was genuinely split between two real words, did it **ask**
  rather than pick? And when two spellings named the same person, did it fold
  them into one option?
- Are the literal hypotheses still on screen, unedited, beside the message?
- Before recording any of the above as a failure: did the Groq call actually
  run? A rate limit returns the raw beams as options and says so in the warning
  line.

Two known limitations to expect rather than chase: the model still mishears the
literal phrase *I water*, and `headset` rarely survives the recognizer as one
word. Keep every alternative visible when selection is uncertain.
