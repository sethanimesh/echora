import pytest

@pytest.fixture(autouse=True)
def fixed_test_face_mode(monkeypatch):
    # Local developer selection must not send real snapshots or incur cloud calls
    # in existing mocked tests. Comparison tests explicitly select their mode.
    monkeypatch.setenv('ECHORA_FACE_MODE', 'gemini')
