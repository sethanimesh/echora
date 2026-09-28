"""CLI and configuration tests."""
import json
import subprocess
import sys

import pytest

from echora.config import (
    DEFAULT_FISH_VOICE_ID, DEFAULT_GROQ_MODEL, DEFAULT_OLLAMA_MODEL,
    DEFAULT_TIMEOUT, Config,
)


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "echora.cli", *args],
        capture_output=True, text=True, timeout=120,
    )


# --- Config -----------------------------------------------------------------

def test_defaults_are_values_not_slot_descriptors(monkeypatch):
    """Regression: `slots=True` makes `cls.field` a member_descriptor.

    Reading defaults that way silently produced a garbage model name, the vendor
    failed to construct, and the chain degraded to the offline heuristic with no
    visible error -- output looked plausible and was wrong.
    """
    for var in ("ECHORA_OLLAMA_MODEL", "ECHORA_GROQ_MODEL", "ECHORA_TIMEOUT",
                "ECHORA_OLLAMA_HOST"):
        monkeypatch.delenv(var, raising=False)

    config = Config.from_env(dotenv=None)
    assert config.ollama_model == DEFAULT_OLLAMA_MODEL
    assert config.groq_model == DEFAULT_GROQ_MODEL
    assert config.timeout == DEFAULT_TIMEOUT
    assert config.fish_voice_id == DEFAULT_FISH_VOICE_ID
    assert isinstance(config.ollama_model, str)
    assert isinstance(config.timeout, float)


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("ECHORA_OLLAMA_MODEL", "some-other-model")
    monkeypatch.setenv("ECHORA_CLASSIFIER", "offline")
    monkeypatch.setenv("FISH_API_KEY", "fish-key")
    monkeypatch.setenv("ECHORA_FISH_VOICE_ID", "voice-id")
    monkeypatch.setenv("ECHORA_FISH_TIMEOUT", "45")
    config = Config.from_env(dotenv=None)
    assert config.ollama_model == "some-other-model"
    assert config.classifier == "offline"
    assert config.fish_api_key == "fish-key"
    assert config.fish_voice_id == "voice-id"
    assert config.fish_timeout == 45


def test_fish_key_legacy_alias(monkeypatch):
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    monkeypatch.setenv("FISH_AUDIO_API_KEY", "legacy-fish-key")
    config = Config.from_env(dotenv=None)
    assert config.fish_api_key == "legacy-fish-key"


def test_explicitly_requested_vendor_fails_loudly(monkeypatch):
    """Asking for groq and silently getting the heuristic hides real breakage."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    config = Config(classifier="groq")
    with pytest.raises(RuntimeError, match="Groq classifier requested"):
        config.build_classifier()


def test_auto_mode_degrades_quietly(monkeypatch):
    """In auto mode a missing vendor must not stop the device from starting."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    classifier = Config(classifier="auto", ollama_host="http://127.0.0.1:1").build_classifier()
    from echora.classify.port import SpanQuery
    assert classifier.classify("Main ghar hai", [SpanQuery(0, "Main", 0, 4)])


# --- CLI --------------------------------------------------------------------

def test_cli_one_shot():
    result = run_cli("-c", "offline", "Main kal jaunga")
    assert result.returncode == 0
    assert "कल" in result.stdout


def test_cli_preserves_english():
    result = run_cli("-c", "offline", "The main office is closed today")
    assert result.stdout.strip() == "The main office is closed today"


def test_cli_json_output_is_valid_and_unescaped():
    result = run_cli("-c", "offline", "--json", "Main kal jaunga")
    payload = json.loads(result.stdout)
    assert payload["raw"] == "Main kal jaunga"
    assert "कल" in payload["text"]
    assert payload["classifier"] == "HeuristicSpanClassifier", (
        "provenance must survive the decorator chain"
    )
    assert any(entry["label"] == "HI" for entry in payload["labels"])


def test_cli_explain_shows_decision_provenance():
    result = run_cli("-c", "offline", "-e", "Main kal jaunga")
    assert "decided by" in result.stdout
    assert "LEXICON" in result.stdout
    assert "spans sent to classifier" in result.stdout


def test_cli_speech_mode_applies_tts_normalisation():
    result = run_cli("-c", "offline", "-s", "API ka 5kg order")
    assert "A P I" in result.stdout
    assert "kilogram" in result.stdout


def test_cli_reads_stdin():
    result = subprocess.run(
        [sys.executable, "-m", "echora.cli", "-c", "offline", "-"],
        input="Main kal jaunga\nThe main office\n",
        capture_output=True, text=True, timeout=120,
    )
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 2
    assert "कल" in lines[0]
    assert lines[1] == "The main office"


def test_cli_handles_empty_and_odd_input():
    for text in ["!!!", "   ", "😊"]:
        assert run_cli("-c", "offline", text).returncode == 0
