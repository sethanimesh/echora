#!/usr/bin/env python3
"""Run backend contract tests with no provider access or private profile store."""
import os
from pathlib import Path
import socket
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path[:0] = [str(ROOT / 'backend'), str(ROOT)]
os.environ.update(PYTHON_DOTENV_DISABLED='1', GROQ_API_KEY='offline-test-key',
                  GROQ='', GEMINI_API_KEY='', GOOGLE_API_KEY='', FISH_API_KEY='',
                  FISH_AUDIO_API_KEY='', FISH_REFERENCE_ID='', ECHORA_FACE_MODE='gemini')
sys.dont_write_bytecode = True


def blocked(*args, **kwargs):
    raise AssertionError('External networking is disabled during local verification')


socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.create_connection = blocked

with tempfile.TemporaryDirectory(prefix='echora-check-') as temporary:
    os.environ['ECHORA_PROFILE_DB'] = str(Path(temporary) / 'profiles.sqlite3')
    import pytest
    raise SystemExit(pytest.main(['-q', '-W', 'ignore::DeprecationWarning', '-p', 'no:cacheprovider',
                                 'backend/tests', 'communication/backend/tests', 'language_tests',
                                 'mobile/tests/test_native.py', *sys.argv[1:]]))
