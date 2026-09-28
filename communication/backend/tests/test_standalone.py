"""The unified project owns its contracts and configuration without siblings."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize('process_override', [False, True])
def test_backend_owns_models_and_environment(tmp_path, process_override):
    backend = Path(__file__).resolve().parents[1]
    project = tmp_path / 'unified'
    shutil.copytree(backend, project / 'communication' / 'backend',
                    ignore=shutil.ignore_patterns('tests', '__pycache__', '*.pyc'))
    shutil.copytree(backend.parents[1] / 'backend' / 'app', project / 'backend' / 'app',
                    ignore=shutil.ignore_patterns('tests', '__pycache__', '*.pyc'))
    # Values in a neighboring project must never configure this application.
    (tmp_path / '.env').write_text(
        'GROQ_API_KEY=parent-groq\nGEMINI_API_KEY=parent-gemini\n'
        'FISH_API_KEY=parent-fish\nFISH_REFERENCE_ID=parent-voice\n')
    (project / 'communication' / '.env').write_text(
        'GROQ_API_KEY=obsolete-voice-groq\nGEMINI_API_KEY=obsolete-voice-gemini\n'
        'FISH_API_KEY=obsolete-voice-fish\nFISH_REFERENCE_ID=obsolete-voice-reference\n')
    (project / '.env').write_text(
        'GROQ_API_KEY=unified-groq\nGEMINI_API_KEY=\n'
        'FISH_API_KEY=unified-fish\nFISH_REFERENCE_ID=unified-reference\n')
    env = os.environ.copy()
    for name in ('GROQ_API_KEY', 'GEMINI_API_KEY', 'FISH_API_KEY',
                 'FISH_REFERENCE_ID', 'PYTHON_DOTENV_DISABLED'):
        env.pop(name, None)
    if process_override:
        env['GROQ_API_KEY'] = 'process-groq'
    check = textwrap.dedent('''
        import builtins
        import os
        from pathlib import Path
        import sys

        root = Path(sys.argv[1])
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / 'backend'))
        real_import = builtins.__import__
        def no_robot(name, *args, **kwargs):
            if name == 'robot' or name.startswith('robot.'):
                raise AssertionError('Voice backend attempted a robot import')
            return real_import(name, *args, **kwargs)
        builtins.__import__ = no_robot

        from communication.backend import app, cloud_models, delivery_cues, fish
        assert Path(app.__file__).is_relative_to(root)
        assert app.ENV_FILE == root / '.env'
        assert Path(cloud_models.__file__).is_relative_to(root)
        assert cloud_models.GEMINI_MODEL == 'gemini-3.5-flash-lite'
        assert cloud_models.GEMINI_THINKING_LEVEL == 'MINIMAL'
        assert delivery_cues.MODEL == cloud_models.GEMINI_MODEL
        assert os.environ['GROQ_API_KEY'] == sys.argv[2]
        assert not os.environ.get('GEMINI_API_KEY', '').strip()
        assert fish.configuration() == ('unified-fish', 'unified-reference')
    ''')
    result = subprocess.run(
        [sys.executable, '-I', '-c', check, str(project),
         'process-groq' if process_override else 'unified-groq'],
        env=env, cwd=project, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
