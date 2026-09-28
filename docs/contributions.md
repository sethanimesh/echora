# Contributions and attribution

Echora is maintained in the `sethanimesh/echora` repository as a personal project. The repository makes the following project-specific work inspectable: speech adaptation/evaluation tooling, evidence-preserving composition, contextual profiles, verification gates, and the shared message/speech lifecycle with web/native clients.

This inventory describes what exists, not a verified assertion that one person manually authored every component. A detailed personal-versus-assisted contribution breakdown has not been supplied. Git records include AI-assisted development checkpoints; tool assistance and upstream work should not be represented as unaided original authorship.

## Upstream components

- Qwen supplies the pretrained ASR foundation; Echora contains adaptation and application integration, not a foundation model trained from scratch.
- Parakeet and other recognizers appear as baselines or research runners.
- TORGO and Common Voice supply speech data; controlled command compositions are derived examples, not newly recruited participants.
- FastAPI, React/Vinext, Expo, PyTorch, Transformers, MiniLM and the UI/accessibility libraries provide substantial infrastructure.
- Groq, Fish and optional Gemini provide selected hosted features. Device speech is provided by the client platform.

Existing third-party notices remain with their components. No repository-wide license grant is introduced by this documentation update. [Model identities](../models/echora-qwen3-asr-command-v3/manifest.json) and dependency manifests identify the technical dependencies.
