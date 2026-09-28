# Echora 2.0 core reuse

Original source: `Archive/Echora 2.0/echora` in the robot workspace. `SOURCE.json` records SHA-256 hashes of those original files. This standalone export includes the lexicon, segmentation, span model, lexical classifier, classification prompt/schema, English sentence guard, transliterator and TTS normalizer; the original workspace is not required.

Changes to copied Python: imports and the data resource point to this package; `Pipeline.from_config` and its config import are omitted. There is no runtime dependency on Archive, Ollama, Haystack, Fish Audio, or the archived provider/model defaults. The retained heuristic/fallback helpers support the original unit tests; the application never instantiates the fallback chain.

`../hinglish.py` is the application adapter: async Groq classification using `providers.TEXT_MODEL` (currently `openai/gpt-oss-120b`) only for ambiguous spans, bounded requests and explicit failure, strict ID/count/reading validation, and the original English guard. After a gold-set case caused the model to label an unlisted word, the adapter added a closed ID enum and an exact-count reminder, without changing the archived core prompt. The pronunciation normalizer is applied around protected names/code/URLs rather than across them. The readable message remains separate from speech text, which must be reviewed before confirmation.

The original tests for classification and preservation are copied with import changes into `../tests/test_archive_*.py`. Dated evaluation reports are historical. `communication/evals/check_hinglish.py` reads the bundled `evals/fixtures/hinglish_gold.jsonl` and records new checks by current model and date. See [standalone setup](../../../VOICE_APP.md).
