"""Select a fixed facial provider or run a paired experiment on identical frames."""
import asyncio
import os
from . import face_cues as gemini, local_faces
from .providers import ProviderFailure

MAX_FRAME, MAX_UPLOAD, CALL_LIMIT = gemini.MAX_FRAME, gemini.MAX_UPLOAD, gemini.CALL_LIMIT


def mode():
    return os.getenv('ECHORA_FACE_MODE', 'gemini').strip().lower()


def status():
    selected = mode()
    local_ready = bool(local_faces.configured()) if selected in {'local', 'compare'} else False
    cloud_ready = gemini.configured()
    return {'configured': cloud_ready if selected == 'gemini' else local_ready if selected == 'local' else local_ready and cloud_ready if selected == 'compare' else False,
            'mode': selected, 'model': local_faces.MODEL if selected == 'local' else gemini.MODEL,
            'local_ready': local_ready, 'comparison': selected == 'compare', 'message': local_faces.startup_error if selected in {'local', 'compare'} else None}


async def prepare():
    if mode() in {'local', 'compare'} and local_faces.configured():
        try:
            await asyncio.wait_for(asyncio.to_thread(local_faces.warmup), timeout=45)
        except Exception:
            local_faces.startup_error = 'The local facial model could not load. Voice recording remains available; no model was substituted.'


def configured():
    return status()['configured']


def agreement(local, cloud):
    def style(result):
        if result.get('state') != 'ready' or result.get('visibility') != 'clear_face': return None
        return {'positive_expression': 'positive', 'smile': 'positive', 'broad_smile': 'positive', 'neutral': 'neutral'}.get(result.get('cue'))
    a, b = style(local), style(cloud)
    return 'not_comparable' if a is None or b is None else 'same_style' if a == b else 'different_style'


async def suggest(frames):
    selected = mode()
    if selected == 'gemini': return await gemini.suggest(frames)
    if selected == 'local': return await local_faces.suggest(frames)
    if selected != 'compare':
        raise ProviderFailure('face_mode', 'The configured facial analysis mode is invalid.')
    async def run(provider):
        try:
            return {'state': 'ready', **await provider(list(frames))}
        except ProviderFailure as exc:
            return {'state': 'error', 'message': exc.message}
        except Exception:
            return {'state': 'error', 'message': 'This facial check could not finish.'}
    local, cloud = await asyncio.gather(run(local_faces.suggest), run(gemini.suggest))
    # Gemini is the declared baseline during evaluation. Never silently promote
    # the experimental local result or change the baseline after a cloud failure.
    baseline = {k:v for k,v in cloud.items() if k != 'state'} if cloud['state'] == 'ready' else {
        'visibility': 'obscured', 'cue': 'unclear', 'tone': None, 'model': gemini.MODEL}
    return {**baseline, 'comparison': {'local': local, 'gemini': cloud,
            'agreement': agreement(local, cloud), 'selected': 'gemini', 'policy': 'paired-face-1'}}
