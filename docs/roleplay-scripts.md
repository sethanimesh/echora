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
prints the same three counts as `offered= applied= refused=`. Either way:

- **nothing offered** — the detail was never put in front of the model. Either
  it is scoped out of this setting, or the anchor did not hold enough of its
  slot. Check the beams: `soup` at 0.82 and `soup` at 0.30 are two different
  tests, and only the fixed lane lets you choose which one you are running.
- **offered, not used, refused** — the model declared the detail and then wrote
  a message without it, so the code stripped it back to the plain wording. The
  model chose brevity; nothing is broken.
- **offered and used** — the detail is in the message, marked, and **Say it
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
| "tea" | *I would like my Lipton tea with milk.* | `tea` is settled in the beams. Scoped `home, care`, so it is offered here. Marked; **Say it plainly instead** → *my tea*. |
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
| "tea" | *my Lipton tea with milk* | Still specialized: `home, care` covers here. |
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
| "tea" | *Tea, please.* | **Changed twice.** The specialization is scoped out, and the act became a counter request. |
| "water" | *Some water, please.* | Over a counter, not a statement of want. |

Now the part that is easy to get backwards: **among strangers the message gets
shorter, not longer.** No "I'm sorry to trouble you", no explaining that she has
had a stroke, no softening. Sales-clerk ratings of AAC users put short and
information-bearing above politeness in a time-pressured exchange (Bedrosian,
Hoag & McCoy, 2008), and people with aphasia are if anything faster with
unfamiliar partners than familiar ones (Doedens et al., 2021).

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
| "tea" | *I would like a tea.* — or *Tea, please.* | The detail stays away either way: still scoped `home, care`. The act is the soft half of this row; see below. |

Judge Act IV on **washroom**, not on tea, and the reason is worth knowing. The
listener changes the act only where the words carry an act to change. `washroom`
is a place, and "I need the washroom." and "Where is the washroom?" are visibly
two different things to do. `tea` is a bare noun with no verb in it, and "Tea,
please." is an ordinary thing to say to your own daughter as well as across a
counter — so the model lands on it under either listener, and that is not wrong.
The tea row still earns its place because it shows the *detail* staying away
while the setting is outdoors; it is just not the row to test the act with.

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
| "coffee" | *A short black coffee, no sugar, please.* — **or** *Some coffee, please.* | See below. |
| "washroom" | *Where is the washroom?* | The house example. Asked, not stated. |
| "I need my tablet" | *Could I have my tablet?* | **Changed.** `home, care` scoping drops the detail, and a stranger cannot be sent for his medication anyway. If the message still says "baclofen", that is a scoping failure worth recording. |

The **Cascais** line is the one that carries the point, and it carries it
reliably: the detail is *content he supplied* and survives, while the
statement→question flip is *act*, which the listener owns. Content and act move
in opposite directions in the same message.

The **coffee** line makes the same point and does **not** hold reliably, so it is
written here with both outcomes. The unfamiliar stance tells the model to carry
"the thing being asked for and nothing else", and about half the time it reads
"a short black coffee, no sugar" as elaboration and trims it — you will see
`offered=1 applied=0 refused=1` in the trace, meaning the model declared the
detail and then wrote a message without it. The prompt now says in as many words
that a detail is *what* is being asked for rather than decoration around it, and
that the instruction to be short never trims one, which helps and does not settle
it. Record it as a known wobble, not as a scoping failure.

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
  question or short request among strangers.
- Was the unfamiliar message **shorter** than the familiar one? Longer is a bug,
  every time.
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
