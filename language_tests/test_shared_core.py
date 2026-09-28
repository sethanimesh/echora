"""The library and live app share models, algorithms and the lexicon."""
from importlib import import_module, resources
from pathlib import Path
from echora.config import Config
from echora.pipeline import Pipeline, Result
from communication.backend.hinglish_core.pipeline import Pipeline as CorePipeline, Result as CoreResult


def test_modules_are_identity_preserving_aliases():
    for name in ['core.model', 'detect.segment', 'detect.spans', 'classify.lexical', 'classify.port', 'classify.prompt', 'translit.transliterator', 'tts.normalize']:
        assert import_module(f'echora.{name}') is import_module(f'communication.backend.hinglish_core.{name}')
    assert Result is CoreResult
    assert Pipeline.run is CorePipeline.run


def test_lexicon_resource_has_one_authoritative_location():
    assert resources.files('echora.data').joinpath('hi_roman2dev.json') == resources.files('communication.backend.hinglish_core.data').joinpath('hi_roman2dev.json')
    assert not Path('echora/data/hi_roman2dev.json').exists()


def test_config_does_not_read_an_implicit_environment_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('ECHORA_CLASSIFIER', raising=False)
    (tmp_path / '.env').write_text('ECHORA_CLASSIFIER=unexpected\n')
    assert Config.from_env().classifier == 'auto'
    assert Config.from_env(dotenv=tmp_path / '.env').classifier == 'unexpected'
