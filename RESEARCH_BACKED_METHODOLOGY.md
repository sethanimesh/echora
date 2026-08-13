# Echora: research-backed phased methodology

This document explains the idea as a sequence of research-backed experiments rather than as a software architecture. Each phase produces something that can be tested with real recordings. The next phase is added only if the previous idea demonstrably helps.

## Overall hypothesis

A person with aphasia may produce speech that a conventional recognizer interprets incorrectly. Instead of trusting one transcript, Echora should preserve several plausible interpretations, use only user-approved personal context to rank them, ask the person to confirm the intended message, and learn from that confirmation over time.

The conceptual loop is:

```text
person controls when the utterance is complete
                    |
                    v
       produce several speech interpretations
                    |
                    v
 use approved personal and conversational context
                    |
                    v
      present probable meaningfully different messages if unsure
                    |
                    v
          person confirms or rejects them
                    |
          +---------+---------+
          |                   |
          v                   v
    speak the message    improve future ranking
```

Two different personalization problems must be kept separate:

1. **Speech personalization:** learning how this person pronounces words.
2. **Meaning personalization:** learning which words, people, topics, and styles are relevant to this person.

Combining these from the start would make failures hard to diagnose. The phases below isolate them.

## Phase 1 — Complete-utterance baseline

### Research question

Can a standard speech recognizer preserve enough of the user's intended message when the person—not an automatic silence detector—decides when the attempt is complete?

### Why this is supported

People with post-stroke aphasia have been found to pause more frequently and for longer durations than control speakers. Pauses can reflect word retrieval, planning, monitoring, and memory rather than the end of a message. This supports retaining the entire recording until deliberate release instead of using a normal conversational silence threshold. See [Angelopoulou et al., silent pauses in post-stroke aphasia](https://pmc.ncbi.nlm.nih.gov/articles/PMC11047180/) and [Silent pauses in aphasia](https://pubmed.ncbi.nlm.nih.gov/29634961/).

Automatic recognition of aphasic speech is known to be difficult, and work using AphasiaBank showed that aphasia-specific data can improve recognition over an out-of-domain recognizer. This means the standard recognizer is a necessary baseline, not the expected final solution. See [Le and Mower Provost (2016)](https://www.isca-archive.org/interspeech_2016/le16b_interspeech.html).

### Phase deliverable

A testable flow in which the user holds or toggles recording, deliberately ends it, receives one transcript, confirms or edits it, and then hears the confirmed message through text-to-speech.

No LLM interpretation and no memory are used yet.

<!-- ### Test method

- Collect a small consented set of prompted and spontaneous messages from the initial target group.
- Include messages with long internal pauses, repetitions, self-corrections, and incomplete grammar.
- Ask the speaker or an agreed communication partner to establish the intended reference message.
- Measure whether the transcript preserves the intended meaning, not just whether every word matches.
- Record how often editing or re-recording is needed. -->

### Decision to continue

Continue even if the standard recognizer is weak, provided the complete recordings and reference meanings can be collected reliably. These become the benchmark for later phases.

## Phase 2 — Preserve multiple probable speech interpretations

### Research question

When the first transcript is wrong, is the intended message often present among the recognizer's next-best alternatives?

### Why this is supported

Speech recognition systems naturally contain uncertainty. Research on N-best rescoring shows that considering several hypotheses can outperform reliance on the single top result. A transformer that examined the complete N-best set and could both rerank and rewrite it reported up to an average 8.6% relative WER reduction over ASR alone. See [Kang, Van Gysel, and Siu (2024)](https://www.isca-archive.org/interspeech_2024/kang24c_interspeech.html).

Earlier work also demonstrated a practical pattern in which users select the correct item from an N-best list and those selections later improve correction. See [Bohus et al. (2008)](https://aclanthology.org/W08-0103/).

### Phase deliverable

For every recording, the system returns several genuinely different probable transcriptions rather than one supposedly certain transcript. For example:

```text
1. I want tea.
2. I want to eat.
3. I won't eat.
```

At this phase, the alternatives come from speech recognition only. The system does not yet rewrite them.

### Test method

Compare:

- How often the first transcript contains the intended meaning.
- How often the intended meaning appears anywhere in the top two alternatives.
- How often it appears anywhere in the top five alternatives.
- Which important differences are recovered: names, actions, objects, and negation.

Use speaker-level reporting so good performance for easier speakers does not conceal failure for severely affected speakers.

### Decision to continue

Proceed to LLM interpretation only if the multiple-hypothesis set contains the intended meaning materially more often than the first transcript alone. If it does not, improve the speech-recognition stage first; an LLM cannot reliably recover acoustic information that was never preserved.

## Phase 3 — LLM interpretation constrained by speech evidence

### Research question

Can an LLM convert fragmented or competing transcriptions into two clear message candidates while remaining faithful to the speech evidence?

### Why this is supported

Research shows that language models can rescore and correct speech hypotheses. Cross-modal work combining acoustic and language information has also improved recognition, supporting the use of both speech evidence and linguistic plausibility rather than text-only correction. See [Whispering LLaMA, Radhakrishnan et al. (2023)](https://aclanthology.org/2023.emnlp-main.618/) and [Kang et al. (2024)](https://www.isca-archive.org/interspeech_2024/kang24c_interspeech.html).

However, fluent correction can invent plausible language. Therefore, the LLM must be treated as an interpretation proposer, not an authority. It must point back to the speech hypotheses supporting each candidate and be allowed to say that the evidence is insufficient.

### Phase deliverable

The system converts the speech alternatives into:

- No candidate when evidence is insufficient.
- One candidate when the alternatives mean essentially the same thing.
- Two candidates when two meanings remain plausible.
- A permanent “Neither” path.

Candidates must represent the user's message, not answer the user. “Tea” may become “I would like some tea,” but not “Tea is a popular drink.”

### Test method

Create a blinded comparison among:

1. The first ASR transcript.
2. The two best raw ASR alternatives.
3. The two LLM-interpreted candidates.

Measure:

- Whether the intended meaning is present in the two options.
- Whether either option introduces information unsupported by the recording.
- Errors involving people, negation, numbers, medicines, money, and emergencies.
- Whether the system correctly declines to guess.

Meaning should be judged by the speaker whenever possible, supported by trained human raters. Google research found that LLM-based meaning-preservation evaluation correlated better with human judgment than standard word metrics, but human assessment remains the reference. See [Tomanek et al. (2024)](https://research.google/pubs/large-language-models-as-a-proxy-for-human-evaluation-in-assessing-the-comprehensibility-of-disordered-speech-transcription/).

### Decision to continue

The LLM phase must improve intended-message coverage over raw alternatives without increasing unsupported critical information. A fluent-looking output is not evidence of success.

## Phase 4 — Add user-approved personal context

### Research question

Does knowing the user's own people, places, routines, vocabulary, and communication style help distinguish acoustically similar interpretations?

### Why this is supported

Google's Project Relate allowed people to record custom cards for names and other personal vocabulary, recommending several contextual recordings of important proper nouns. Its approach reflects the importance of personally relevant vocabulary in atypical-speech recognition. See the [Project Relate help materials](https://sites.research.google/relate/help/).

Work with AAC users found that LLM suggestions could reduce effort, but participants wanted suggestions to reflect their own communication style and preferences. See [Valencia et al. (2023), “The less I type, the better”](https://discovery.ucl.ac.uk/id/eprint/10185220/).

A 2025 study involving people with aphasia found value in personalized, contextual, and multimodal AI support, while also observing intent mismatches and cognitive burden when suggestions did not fit. See [Mao et al. (2025)](https://arxiv.org/abs/2504.09435).

### Phase deliverable

The system can use a small, user-approved memory consisting of:

- Important names and places.
- Recurring objects and activities.
- Frequently used messages.
- Examples of the person's preferred wording.
- Only the recent confirmed conversation for temporary context.

The system must be able to explain which memory item influenced a suggestion. Information mentioned once is not automatically turned into a permanent fact.

### Test method

Run each ambiguous recording three ways:

1. Without personal information.
2. With the correct person's approved information.
3. With deliberately irrelevant or conflicting information.

The correct profile should improve recognition of relevant names and likely meanings. Irrelevant context must not override strong speech evidence. This experiment distinguishes genuine contextual benefit from an LLM that merely follows whatever biography it is shown.

### Decision to continue

Proceed only if approved personal context improves personal-word or intended-message accuracy without increasing confident but unsupported guesses.

## Phase 5 — Learn from the user's selections

### Research question

Can the system predict better next time by learning which displayed option the user selected, edited, or rejected?

### Why this is supported

Bohus et al. demonstrated that clicks on N-best recognition choices can serve as feedback for learning future corrections. The important transferable idea is that a selection provides comparative evidence: the selected interpretation was preferred over the alternatives shown at that moment. See [Bohus et al. (2008)](https://aclanthology.org/W08-0103/).

Research on personalized language models also supports using lightweight representations of individual preferences rather than immediately retraining an entire language model. See [Personalized Language Modeling from Personalized Human Feedback](https://openreview.net/forum?id=bqUsdBeRjQ).

### Phase deliverable

The system remembers interaction outcomes:

- Which two choices were displayed.
- Which one was selected.
- Whether the user edited it.
- Whether both were rejected.
- The final confirmed wording.
- The relevant topic and approved context.

Initially, this history changes only the order and selection of future candidates. It does not yet retrain how the user's voice is transcribed.

### Test method

Replay later interactions twice:

1. Using the common, non-personalized ranking.
2. Using ranking learned from that user's earlier confirmations.

Use time order: train from earlier interactions and test on later ones. Randomly mixing one person's nearly identical messages between training and testing would exaggerate performance.

Measure whether personalization increases first-choice and top-two accuracy, while reducing “Neither,” editing, and response time.

### Decision to continue

Adopt learned ranking only when it helps on later, unseen interactions and does not increase critical errors. One accidental selection should never change the system immediately; improvement should be evaluated in batches before release.

## Phase 6 — Learn the person's speech patterns

### Research question

After enough confirmed recordings have accumulated, does adapting the speech recognizer to the individual improve the quality of the alternatives supplied to the LLM?

### Why this is supported

Google's work on personalized recognition for non-standard speech reported a 62% relative WER improvement for speakers with ALS and found that 71% of the eventual improvement was obtained from only five minutes of training speech. See [Shor et al. (2019)](https://www.isca-archive.org/interspeech_2019/shor19_interspeech.html).

A larger Euphonia study involving 432 people with disordered speech reported a median WER of 4.6% for personalized models compared with 31% for speaker-independent models on short phrases. Improvements were especially large for more severely affected speakers. See [Green et al. (2021)](https://www.isca-archive.org/interspeech_2021/green21_interspeech.html).

These results strongly justify individual adaptation, but they do not guarantee the same gains for aphasia or spontaneous conversation. Aphasia-specific research nevertheless shows that adaptation using AphasiaBank can improve recognition, supporting a careful cohort-specific experiment. See [Le and Mower Provost (2016)](https://www.isca-archive.org/interspeech_2016/le16b_interspeech.html).

### Phase deliverable

A personalized speech-recognition version trained from that person's confirmed audio-message pairs. The general recognizer remains available for comparison and fallback.

### Bootstrapping method

Do not require 500 recordings before providing value. Instead, measure the learning curve:

- General recognizer with no personal recordings.
- Approximately five minutes of verified speech.
- Ten minutes.
- Twenty minutes or the user's comfortable limit.
- Continued learning from normal, confirmed use.

Include:

- Functional everyday phrases.
- Important names placed in several different phrases.
- Prompted phrases for sound coverage.
- Spontaneous messages for realistic transfer.
- Multiple sessions and recording conditions.

Only recordings with a reliable confirmed meaning should teach the speech model. Rejected or uncertain messages are not valid training labels.

### Test method

Hold back later recordings that were never used for adaptation. Compare:

1. General speech recognition.
2. A broader atypical-speech recognizer, if available.
3. The person's personalized recognizer.

Measure first-transcript accuracy, top-two/top-five intended-message coverage, important-word accuracy, and downstream candidate selection. Test both familiar and previously unseen phrases.

### Decision to continue

Use the personal recognizer only for people whose held-out results improve. More personalization is not automatically better; insufficient or poor-quality recordings may make a model worse.

## Phase 7 — Complete communication evaluation

### Research question

Does the full method help a person communicate their intended message more accurately or efficiently than a standard transcript or their existing method?

### Why this is supported

Word accuracy alone can misrepresent the practical effect of a transcription error. A small error can either preserve the message or reverse it. Meaning-preservation research therefore supports evaluating whether another person understands what the speaker intended. See [Tomanek et al. (2024)](https://research.google/pubs/large-language-models-as-a-proxy-for-human-evaluation-in-assessing-the-comprehensibility-of-disordered-speech-transcription/).

Research with AAC users also warns that generated suggestions can save effort while reducing authorship or authenticity when they do not match the individual. Communication success, agency, and perceived ownership must therefore be evaluated together. See [Valencia et al. (2023)](https://discovery.ucl.ac.uk/id/eprint/10185220/) and [Mao et al. (2025)](https://arxiv.org/abs/2504.09435).

### Phase deliverable

A pilot-ready full method:

```text
manual complete recording
 -> personalized probable interpretations
 -> evidence-constrained message candidates
 -> approved personal context
 -> user confirmation or rejection
 -> text-to-speech
 -> learning from confirmed use
```

### Test method

Compare three conditions for the same participant:

1. Their existing communication approach or a plain transcript.
2. Multiple candidates without personalization.
3. Multiple candidates with speech and meaning personalization.

Use prompted tasks and real conversations. Measure:

- Was the intended meaning successfully communicated?
- How long did it take from starting speech to speaking the confirmed message?
- How many attempts, edits, and repair turns were required?
- How frequently were both suggestions rejected?
- Did the person feel that the final message was theirs?
- Did accuracy change with fatigue, topic, environment, or conversation partner?
- Did continued use improve later performance?

### Decision to continue

The full system succeeds only if it improves communication while preserving control and authorship. Faster output that says the wrong thing is a failure.

## What each phase proves

| Phase | Idea under test | Testable output |
|---|---|---|
| 1 | Manual completion preserves slow, pause-filled speech | One transcript and confirmed TTS |
| 2 | Alternative speech hypotheses recover missed meaning | Ranked probable transcriptions |
| 3 | An LLM can clarify alternatives without inventing intent | Zero, one, or two evidence-backed messages |
| 4 | Approved personal context resolves ambiguity | Profile-aware candidates |
| 5 | User selections improve later ranking | Personalized option ordering |
| 6 | Confirmed recordings improve recognition of the individual | Personalized speech model |
| 7 | The complete method improves real communication | Comparative pilot results |

## Core research measurements

Across all phases, retain the same principal measurements:

- **First-choice intended-message accuracy:** Was the correct meaning presented first?
- **Top-two intended-message accuracy:** Was it present in either choice?
- **Available-evidence accuracy:** Was the intended meaning present anywhere in the speech-recognition alternatives?
- **Meaning preservation:** Would a listener understand the intended message?
- **Unsupported-content rate:** Did the system add a fact not justified by speech or approved context?
- **Critical-word accuracy:** Names, actions, negation, numbers, medicines, money, and emergencies.
- **Rejection and repair:** How often did the user choose neither, edit, or try again?
- **Communication effort:** Time, number of actions, fatigue, and frustration.
- **Agency:** Did the person feel the final message represented what they wanted to say?

The most informative sequence is therefore:

```text
prove that alternatives help
        before
prove that the LLM helps
        before
prove that context helps
        before
learn from selections
        before
retrain speech recognition
```

This makes every improvement attributable to a specific research-backed idea and prevents a complex final system from hiding which part is actually helping or harming the user.
