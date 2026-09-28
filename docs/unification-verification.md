# Unification verification — 28 September 2026

This records the earlier application merge. Subsequent acoustic verification and explicit-memory behavior are documented in [verification-and-memory.md](verification-and-memory.md); this page's test totals and runtime observations describe that earlier checkpoint.

The canonical app is `echora/` within the workspace. Both original code bases and the pre-merge working files are preserved outside the new runtime. The second project's saved profile database contained no user profile rows; its built-in sample profiles are retained. Six first-project profiles were imported. A repeat migration dry run reported six already present, zero new imports, and zero conflicts.

## Automated checks

| Area | Result |
|---|---|
| Shared backend, recognition/composition, profiles, language library and native API | 636 passed, 2 skipped |
| Web behavior | 67 passed |
| Web type check and production build | Passed, including startup retry changes |
| Mobile behavior and TypeScript | 14 passed; types passed |
| Native API integration | 12 passed, included in the 636 total |
| iOS export | Passed |
| Profile migration repeat dry run | 6 skipped as already imported; 0 conflicts |
| Source whitespace check | Passed |

Run `.venv/bin/python scripts/check_unified.py` for the combined Python check. It blocks outgoing connections, substitutes dummy credentials, disables dotenv loading, and uses temporary profile storage. The skips cover the optional Haystack dependency and a live Fish integration check. The optional Haystack adapter was also checked separately with the previously installed framework environment (four mocked tests passed). No live provider quality is inferred from mocked tests.

The full web lint command still reports 27 inherited findings in generic UI components, accessibility code and older tests. Lint on the changed communication and playback code passes. Build warnings include large client chunks and framework route classification; they did not prevent the production build. Python reports one upstream test-client deprecation warning.

## Running application

The final local API loaded the tuned Qwen model successfully. The web page rendered through port 3000 and connected to the shared backend on 8000. `scripts/smoke_unified.py` passed against the running app, checking the proxy, model readiness, typed input provenance, exact-text confirmation, edit invalidation, stale-revision rejection and cancellation. It requests no inference or speech.

Browser checks confirmed that the merged app connects, exposes both recognizers, loads an imported profile, accepts typed text, and reaches device playback. A longer synthetic message was stopped while playing; the UI showed “Speech stopped” and re-enabled editing. The typed evidence label no longer displays a null model name. Session, profile and saved-place startup reads now retry transient connection failures with bounded backoff and cancellation while the model loads, without replaying restored messages. Device playback state was observed; audible voice quality was not assessed.

## Preserved optional tools

The local facial runtime loaded the preserved, checksum-verified ONNX assets from the canonical project. The selected facial mode remains Gemini; the optional local runtime is not silently substituted. Gemini delivery features require configuration before use.

The standalone gaze runtime uses the copied `.venv-gaze` plus two local MNN libraries verified against the installed wheel's hashes. Synthetic checks returned 258 finite features, changing predictions for changing inputs, and no-face rejection (5.73 ms median inference in that check). The proxy check through port 3000 passed local-origin acceptance, foreign-origin rejection, required headers, blank/invalid image handling, calibration validation, and Stop/session invalidation. The temporary gaze service was stopped after testing.

No camera was opened. These checks do not measure a person's gaze accuracy, facial-cue reliability, or accessibility suitability.

## Remaining device and provider evaluation

Real microphone capture, adapted/Whisper transcription quality on new speech, cloud voice quality, camera capture and native hardware playback have not been exercised during this merge. Browser gaze, facial controls and Fish controls remain web features; native uses the same message policies with its existing adapted-recognition and Groq/device voice controls. No cloud inference or speech requests were made by the verification above.
