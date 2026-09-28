# Problem and methodology

A recognizer can turn incomplete speech into incorrect words. A second model can make that error more convincing. Echora investigates literal alternatives, constrained composition and speaker choice as an inspectable communication workflow. It does not establish the speaker's intended meaning independently of the speaker.

## Alternatives and constraints

| Approach | Attraction | Limitation |
| --- | --- | --- |
| Speak the top ASR result | Simple and fast | Conceals other readings when rank one is wrong |
| Freely rewrite a transcript | Fluent output | Can introduce unsupported meaning |
| Let history determine wording | Familiar vocabulary | Habits can contradict current words |
| Require a separate confirmation button | Explicit approval | Extra interaction effort |
| Separate web/native services | Independent development | Divergent cancellation and memory behavior |

The implementation uses immutable evidence, one adapted-speech composition pass, bounded verification/context fusion, and a shared revision lifecycle. Benefit to real speakers still requires participant evaluation. [Decisions](adr/README.md).

## Observable success criteria

| Dimension | Pass condition | Evidence |
| --- | --- | --- |
| Evidence integrity | Literal hypotheses survive composition and remain selectable | Backend fidelity/verification and client tests |
| Speech control | Stop, edits and stale revisions cannot authorize old speech | Shared lifecycle and native contract tests |
| Memory control | Only explicit Remember persists displayed wording | Profile/memory tests |
| Learned automatic selection | Identity, calibration and acceptance gates pass | Current audit fails; route remains advisory |
| Recognition | Matched literal WER/CER, coverage and retention comparisons | Foundation and adapter artifacts |
| Speaker usefulness | Accurate intended messages with manageable effort | Not yet established |

## Evaluation layers

1. Score recognition before context or generation. WER counts substitutions, deletions and insertions relative to reference words; speaker-macro WER weights speakers equally.
2. Separate checkpoint selection and protected evaluation. Preserve seeds, configurations, hashes and the distinction between original speech and composed commands.
3. Compare verification against ASR rank one and context/scorer ablations, including all-beams-wrong inputs. Count independent prompt groups.
4. Evaluate wording through the production boundary. Keep automated grading limits and pending human ratings visible.
5. Test interaction contracts separately from model quality. Mocked providers verify behavior under controlled responses, not provider accuracy.

[Evaluation](evaluation.md) links results to evidence. [Limitations](limitations.md) describes overlap and selection exposure.
