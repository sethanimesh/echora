"""Local communication lifecycle; only explicitly remembered wording persists."""
import asyncio
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
import hashlib
import os
import re
from pathlib import Path
import secrets
import sqlite3
import time
from types import SimpleNamespace
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from . import providers
from . import context as personal
from . import hinglish
from . import conversation
from . import wording_fidelity
from . import fish
from . import delivery_cues
from . import face_analysis as face_cues
from . import recognition

ENV_FILE = Path(__file__).resolve().parents[2] / '.env'
load_dotenv(ENV_FILE)
fish.load_configuration([ENV_FILE])
delivery_cues.load_configuration([ENV_FILE])
ORIGINS = {'http://127.0.0.1:5173', 'http://localhost:5173', 'http://127.0.0.1:8766',
           'http://127.0.0.1:3000', 'http://localhost:3000', 'http://127.0.0.1:8000', 'http://localhost:8000'}
MAX_AUDIO = 20 * 1024 * 1024
BUSY = {'transcribing', 'drafting', 'preparing', 'synthesizing', 'suggesting', 'checking_face', 'analyzing_delivery'}
# Limits apply to this local process; restarting resets the request counter.
CALL_LIMIT = max(1, min(int(os.getenv('ECHORA_CALL_LIMIT', '100')), 500))
calls_used = 0
fish_calls_used = 0
FISH_CALL_LIMIT = 20
cue_calls_used = 0
face_calls_used = 0

@dataclass
class Session:
    job: dict | None = None
    sequence: int = 0
    events: deque = field(default_factory=lambda: deque(maxlen=64))
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    source_audio_hash: str | None = None
    audio: dict | None = None
    touched: float = field(default_factory=time.monotonic)
    memory: dict | None = None
    memory_candidate: dict | None = None

    def emit(self, kind):
        self.sequence += 1
        self.events.append({'id': self.sequence, 'type': kind, 'timestamp': time.time(), 'job': self.job.copy() if self.job else None})
        self.changed.set()

sessions: dict[str, Session] = {}

@asynccontextmanager
async def lifespan(app):
    await face_cues.prepare()
    async def prune():
        while True:
            await asyncio.sleep(60)
            for key, s in list(sessions.items()):
                if s.memory and time.monotonic() - s.memory['at'] > conversation.TTL:
                    s.memory = None
                if s.memory_candidate and time.monotonic() - s.memory_candidate['at'] > conversation.TTL:
                    s.memory_candidate = None
                if time.monotonic() - s.touched > 1800:
                    if s.task: s.task.cancel()
                    del sessions[key]
    cleanup = asyncio.create_task(prune())
    yield
    cleanup.cancel()
    tasks = [s.task for s in sessions.values() if s.task]
    for task in tasks: task.cancel()
    await asyncio.gather(cleanup, *tasks, return_exceptions=True)

app = FastAPI(title='Echora communication', lifespan=lifespan, docs_url=None, redoc_url=None)

@app.middleware('http')
async def local_boundary(request: Request, call_next):
    host = request.headers.get('host', '').split(':')[0]
    if host not in {'127.0.0.1', 'localhost', 'testserver'}:
        return JSONResponse({'detail': 'This prototype is available on this computer only.'}, status_code=403)
    if request.headers.get('origin') and request.headers['origin'] not in ORIGINS:
        return JSONResponse({'detail': 'Unrecognized origin.'}, status_code=403)
    if request.method not in {'GET','HEAD','OPTIONS'} and request.headers.get('x-echora-client') != '1':
        return JSONResponse({'detail': 'Missing application request header.'}, status_code=403)
    # Bound uploads before multipart parsing, including requests without Content-Length.
    face_upload = request.url.path.endswith('/face-suggestion')
    if request.url.path == '/api/audio' or request.url.path.endswith('/delivery-suggestion') or face_upload:
        total = 0
        chunks = []
        async for chunk in request.stream():
            total += len(chunk)
            if total > (face_cues.MAX_UPLOAD if face_upload else MAX_AUDIO + (face_cues.MAX_UPLOAD if request.url.path == '/api/audio' else 65536)):
                return JSONResponse({'detail': 'The camera snapshots are too large.' if face_upload else 'Use a recording smaller than 20 MB.'}, status_code=413)
            chunks.append(chunk)
        request._body = b''.join(chunks)
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

def session(request):
    s = sessions.get(request.cookies.get('echora_session', ''))
    if not s: raise HTTPException(401, 'Your session ended. Refresh this page to start again.')
    s.touched = time.monotonic()
    return s

def current(s, job_id, revision=None):
    if not s.job or s.job['id'] != job_id: raise HTTPException(404, 'This message is no longer active.')
    if revision is not None and s.job['revision'] != revision: raise HTTPException(409, 'The message changed. Review the latest version.')
    return s.job


def retain_literal_choices(job, candidates):
    evidence = job.get('evidence') or {}
    if evidence.get('recognizer') != 'adapted':
        return candidates
    raw = SimpleNamespace(hypotheses=[SimpleNamespace(**item) for item in evidence.get('hypotheses', [])])
    return recognition.literal_candidates(raw, candidates)


def reset_context_decision(job):
    """A frozen machine decision cannot authorize wording in a different scope."""
    prior = next((item for item in job.get('candidates', [])
                  if item['id'] == job.get('selected_candidate_id')), None)
    ranking = job.get('ranking') or {}
    if ranking.get('route') == 'verified':
        job['ranking'] = {**ranking, 'status': 'stale_context', 'decision': 'ambiguous',
                          'selected_hypothesis_id': None, 'reason': 'Personal or conversation context changed.'}
    choices = []
    if job.get('decision_source') == 'user':
        choices = [{**(prior or {}), 'id': f"chosen-{job['revision'] + 1}", 'text': job['source_text'],
                    'reading': (prior or {}).get('reading', job['source_text']),
                    'source_hypothesis_ids': (prior or {}).get('source_hypothesis_ids', []),
                    'source_literals': (prior or {}).get('source_literals', []),
                    'available': True, 'based_on_user_edit': prior is None}]
    job['candidates'] = retain_literal_choices(job, choices)
    job['selected_candidate_id'] = choices[0]['id'] if choices else None
    if ranking.get('route') == 'verified' and not choices:
        job['question'] = 'Context changed. Choose the words you mean or edit them.'


def invalidate_memory_context(profile_id, deleted_source_ids, profile_deleted=False):
    """Deletion wins over pending generation, authorization and queued audio."""
    for s in list(sessions.values()):
        j = s.job
        for name in ('memory', 'memory_candidate'):
            remembered = getattr(s, name)
            if remembered and (remembered.get('scope') or [None])[0] == profile_id:
                setattr(s, name, None)
        if not j or ((j.get('context') or {}).get('selection') or {}).get('profile_id') != profile_id:
            continue
        dependencies = (j.get('retrieval') or {}).get('dependencies', [])
        affected = (profile_deleted or 'profile/' + profile_id in deleted_source_ids or bool(j.get('conversation'))
                    or bool({d['source_id'] for d in dependencies} & set(deleted_source_ids)))
        if not affected:
            continue
        if s.task and not s.task.done(): s.task.cancel()
        s.audio = None
        authored = (j.get('wording_source') == 'user_edit' or
                    (j['modality'] in {'text', 'phrase'} and not j.get('metadata', {}).get('wording')))
        evidence = j.get('evidence') or {}
        literals = evidence.get('hypotheses') or []
        if not authored:
            selected = (j.get('ranking') or {}).get('selected_hypothesis_id')
            literal = next((h for h in literals if h['id'] == selected), literals[0] if literals else {})
            j['text'] = j['source_text'] = literal.get('literal_text', j['original'])
        j.update(status='review', confirmed=None, prepared_speech=None, auto_speak_revision=None,
                 retrieval=None, context_trace=None, conversation=None,
                 candidates=retain_literal_choices(j, []) if not authored else [],
                 selected_candidate_id=None, options=[], error=None,
                 question='' if authored else 'Personal context changed. Review or edit the recognized words.')
        for flag in ('_speech_when_ready', '_explicit_speech_when_ready', '_defer_auto_speak'):
            j.pop(flag, None)
        if j.get('ranking'):
            j['ranking'] = {**j['ranking'], 'status': 'stale_context', 'decision': 'ambiguous',
                            'selected_hypothesis_id': None, 'reason': 'Referenced personal context changed.'}
        if profile_deleted or 'profile/' + profile_id in deleted_source_ids:
            from .profile_adapter import resolve_snapshot
            selection = {**j['context']['selection'], 'profile_id': '', 'profile_revision': 0,
                         'audience_id': '', 'moment_id': ''}
            j['context'] = resolve_snapshot(selection, None)
        j['revision'] += 1
        s.emit('personal_context_invalidated')


def check_memory_dependencies(s, j):
    result = j.get('retrieval')
    profile = (j.get('context') or {}).get('profile') or {}
    if not profile and not (result or {}).get('profile_id'):
        return
    from . import memory
    snapshot = {'profile_id': profile.get('id'), 'profile_revision': profile.get('revision'), 'dependencies': []}
    if result and result.get('profile_id'):
        snapshot = result
    if not memory.dependencies_valid(snapshot):
        profile_id = snapshot['profile_id']
        try:
            personal.get_profile(profile_id)
            deleted = False
        except HTTPException as exc:
            if exc.status_code != 404: raise
            deleted = True
        invalidate_memory_context(profile_id, {'profile/' + profile_id,
                                  *(d['source_id'] for d in snapshot.get('dependencies', []))}, deleted)
        raise HTTPException(409, 'Personal context changed. Review the current words before speaking.')

def charge():
    global calls_used
    if calls_used >= CALL_LIMIT: raise HTTPException(429, 'The local test request limit was reached. No additional inference was started.')
    providers.api_key()
    calls_used += 1

def begin(s, text, modality, language='auto'):
    s.memory_candidate = conversation.candidate(s.job, s.memory)
    if s.task and not s.task.done(): s.task.cancel()
    s.audio = None
    s.source_audio_hash = None
    s.job = {'id': secrets.token_hex(12), 'created_at': time.time(), 'revision': 1, 'status': 'review', 'modality': modality,
        'original': text, 'source_text': text, 'text': text, 'question': '', 'options': [], 'answers': [], 'error': None,
        'language': language, 'output_language': 'original', 'metadata': {}, 'confirmed': None,
        'context': None, 'context_trace': None, 'prepared_speech': None,
        'conversation': None, 'independent': False, 'output_script': 'auto',
        'evidence': {'kind': modality, 'backend': 'user', 'model': None,
                     'hypotheses': [{'id': 'input', 'literal_text': text}] if text else []},
        'candidates': [], 'auto_speak_revision': None, 'ranking': None, 'retrieval': None}
    s.emit('message_created')
    return s.job

@app.exception_handler(providers.ProviderFailure)
async def provider_exception(request, exc):
    return JSONResponse({'detail': exc.message, 'code': exc.code}, status_code=503)

@app.exception_handler(sqlite3.Error)
async def profile_storage_exception(request, exc):
    return JSONResponse({'detail': 'Local profiles could not be read or saved. Your current words are still available.'}, status_code=503)

@app.get('/api/session')
async def get_session(request: Request, response: Response):
    key = request.cookies.get('echora_session')
    if key not in sessions:
        if len(sessions) >= 30: raise HTTPException(429, 'Too many local sessions. Close unused tabs and try later.')
        key = secrets.token_urlsafe(32)
        sessions[key] = Session()
        response.set_cookie('echora_session', key, httponly=True, samesite='strict', path='/api')
    s = sessions[key]
    s.touched = time.monotonic()
    return {'job': s.job, 'sequence': s.sequence, 'configured': bool(os.getenv('GROQ_API_KEY', '').strip()),
        'recognition': recognition.status(getattr(app.state, 'runtime', None)),
        'max_audio_seconds': getattr(getattr(getattr(app.state, 'runtime', None), 'settings', None), 'max_audio_seconds', 60),
        'max_upload_bytes': MAX_AUDIO,
        'models': {'transcription': providers.ASR_MODEL, 'wording': providers.TEXT_MODEL},
        'calls_used': calls_used, 'call_limit': CALL_LIMIT, 'fish': fish.status(),
        'face_cues': face_cues.status(),
        'delivery_cues': {**delivery_cues.status(), 'calls_used': cue_calls_used, 'call_limit': delivery_cues.CALL_LIMIT}}

@app.get('/api/events')
async def events(request: Request):
    s = session(request)
    try: cursor = int(request.headers.get('last-event-id', '-1'))
    except ValueError: cursor = -1
    async def stream():
        nonlocal cursor
        while not await request.is_disconnected():
            s.touched = time.monotonic()
            if s not in sessions.values(): break
            s.changed.clear()
            if cursor < 0 or cursor > s.sequence or (s.events and cursor < s.events[0]['id'] - 1):
                cursor = s.sequence
                data = {'id': cursor, 'type': 'snapshot', 'timestamp': time.time(), 'job': s.job}
                yield f'id: {cursor}\nevent: state\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'
            else:
                for event in list(s.events):
                    if event['id'] > cursor:
                        cursor = event['id']
                        yield f'id: {cursor}\nevent: state\ndata: {json.dumps(event, ensure_ascii=False)}\n\n'
            try: await asyncio.wait_for(s.changed.wait(), timeout=12)
            except TimeoutError: yield ': heartbeat\n\n'
    return StreamingResponse(stream(), media_type='text/event-stream', headers={'X-Accel-Buffering':'no'})

class TextInput(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    kind: Literal['text', 'phrase'] = 'text'

@app.post('/api/messages')
async def text_message(body: TextInput, request: Request):
    if not body.text.strip(): raise HTTPException(422, 'Enter a few words first.')
    return begin(session(request), body.text.strip(), body.kind)


def mark_for_speech(j, *, explicit=False):
    settings = getattr(getattr(app.state, 'runtime', None), 'settings', None)
    ranking = j.get('ranking') or {}
    if (not explicit and ranking.get('route') == 'verified' and j.get('decision_source') != 'user'
            and (ranking.get('decision') != 'selected' or not j.get('selected_candidate_id'))):
        return
    if not explicit and not getattr(settings, 'speech_autoplay', True):
        return
    if j['status'] == 'review' and j['text'].strip() and not j['question'] and not j.get('error'):
        if j.get('_defer_auto_speak'):
            j['_speech_when_ready'] = True
            j['_explicit_speech_when_ready'] = explicit
        else:
            j['auto_speak_revision'] = j['revision']


def finish_optional_analysis(j):
    j.pop('_defer_auto_speak', None)
    explicit = j.pop('_explicit_speech_when_ready', False)
    if j.pop('_speech_when_ready', False): mark_for_speech(j, explicit=explicit)


def launch_optional_analysis(s, j, audio, suggest_delivery, frames):
    if not j['text'].strip(): return
    if frames:
        launch_recording_analysis(s, j, audio, frames, suggest_delivery)
    elif suggest_delivery:
        try:
            launch_delivery_suggestion(s, j, audio)
        except HTTPException as exc:
            j['delivery_suggestion'] = {'id': secrets.token_hex(12), 'revision': j['revision'],
                'state': 'error', 'message': str(exc.detail)}
            finish_optional_analysis(j)
            s.emit('delivery_suggestion_finished')
    else:
        finish_optional_analysis(j)


async def adapted_job(s, job_id, audio, filename, runtime, suggest_delivery=False, frames=None):
    revision = current(s, job_id)['revision']
    try:
        raw, metadata = await recognition.transcribe(runtime, audio, filename)
        if not s.job or s.job['id'] != job_id or s.job['status'] != 'transcribing': return
        j = s.job
        text = next(item.literal_text for item in raw.hypotheses if item.literal_text.strip())
        j.update(original=text, source_text=text, text=text, evidence=recognition.evidence(raw),
                 metadata={'transcription': {key: value for key, value in metadata.items() if key != 'acoustic_scores'}},
                 candidates=recognition.literal_candidates(raw, []), status='drafting')
        j['revision'] += 1
        revision = j['revision']
        s.emit('transcription_finished')
        # Competing acoustic readings must not silently borrow an old subject.
        same_reading = len({item.literal_text.strip().casefold() for item in raw.hypotheses}) == 1
        linked = conversation.reference(s.memory_candidate, text, j.get('context'), j.get('independent')) if same_reading else None
        j['conversation'] = linked
        if same_reading and conversation.followup(text) and not linked:
            j.update(status='review', question='What should this refer to? Add the item or message you mean.')
            j['revision'] += 1
            s.emit('draft_ready')
            return
        ranking, retrieved = await recognition.rank(runtime, raw, j, metadata.get('acoustic_scores'), linked)
        if not s.job or s.job['id'] != job_id or s.job['revision'] != revision or s.job['status'] != 'drafting': return
        j.update(ranking=ranking, retrieval=retrieved)
        if ranking.get('decision') == 'selected':
            selected = next((h for h in raw.hypotheses if h.id == ranking.get('selected_hypothesis_id')), None)
            if selected: j['source_text'] = selected.literal_text
        check_memory_dependencies(s, j)
        candidates, result, resolved = await recognition.compose(runtime, raw, j, linked, charge=charge)
        if not s.job or s.job['id'] != job_id or s.job['revision'] != revision or s.job['status'] != 'drafting': return
        j = s.job
        check_memory_dependencies(s, j)
        generated = [item for item in candidates if item.get('available')]
        verified = ranking.get('route') == 'verified'
        recommended = None
        if verified and ranking.get('decision') == 'selected':
            recommended = next((item for item in generated if ranking.get('selected_hypothesis_id') in item['source_hypothesis_ids']), None)
        elif not verified and len(generated) == 1:
            recommended = generated[0]
        j.update(candidates=recognition.literal_candidates(raw, candidates), status='review', warnings=list(result.warnings))
        j['metadata']['wording'] = {'model': result.ranker.assistant_model or 'Grounded message assistance',
                                    'elapsed_ms': round(result.ranking_seconds * 1000)}
        j['revision'] += 1
        if result.ranker.source == 'unavailable':
            j['error'] = {'code': 'wording_unavailable', 'message': result.ranker.reason}
        elif recommended is not None:
            j['text'] = recommended['text']
            j['selected_candidate_id'] = recommended['id']
            j['decision_source'] = 'verification' if verified else 'legacy'
            if resolved.get('requires_personal_choice'):
                rule = next(rule for rule in resolved.get('personal_wording', []) if rule['mode'] == 'ask')
                plain = j['text']
                preferred = re.sub(r'(?<!\w)' + re.escape(rule['anchor']) + r'(?!\w)', lambda _: rule['wording'], plain, count=1, flags=re.I)
                j.update(text=text, question=f"Would you like {rule['wording']}, or keep your original wording?", options=list(dict.fromkeys([preferred, plain])))
            elif needs_hinglish(j):
                j['status'] = 'preparing'
                s.emit('pronunciation_started')
                await speech_job(s, job_id)
                if j.get('prepared_speech'): mark_for_speech(j)
            else:
                mark_for_speech(j)
        else:
            j['question'] = 'Which message would you like to say?'
        s.emit('draft_ready')
        launch_optional_analysis(s, j, audio, suggest_delivery, frames)
    except providers.ProviderFailure as exc:
        fail(s, job_id, exc, revision=revision)
    except HTTPException as exc:
        fail(s, job_id, providers.ProviderFailure('request_limit', str(exc.detail)), revision=revision)
    except Exception:
        fail(s, job_id, providers.ProviderFailure('adapted_message_failed', 'Message assistance could not finish. Your recognized words are kept; retry or edit them.'), revision=revision)

async def transcribe_job(s, job_id, audio, filename, content_type, language, suggest_delivery=False, frames=None):
    revision = current(s, job_id)['revision']
    try:
        text, metadata = await providers.transcribe(audio, filename, content_type, language)
        if not s.job or s.job['id'] != job_id or s.job['status'] != 'transcribing': return
        j = current(s, job_id)
        j.update(original=text, source_text=text, text=text, status='review', metadata={'transcription': metadata})
        j['evidence'] = {'kind': 'audio', 'backend': 'whisper', 'model': metadata.get('model', providers.ASR_MODEL),
                         'hypotheses': [{'id': 'whisper-1', 'literal_text': text}]}
        j['ranking'] = {'route': 'legacy', 'status': 'unsupported', 'decision': 'ambiguous',
                        'selected_hypothesis_id': None, 'scores': [], 'artifacts': {},
                        'reason': 'Acoustic verification is available for local adapted English recognition.'}
        j['revision'] += 1
        revision = j['revision']
        if not text: j['error'] = {'code': 'no_speech', 'message': 'No words were transcribed. Try again or type your message.'}
        s.emit('transcription_finished')
        if text and getattr(app.state, 'runtime', None) is not None:
            charge()
            j['status'] = 'drafting'
            s.emit('draft_started')
            await draft_job(s, job_id)
        if s.job and s.job['id'] == job_id and j['status'] == 'review':
            launch_optional_analysis(s, j, audio, suggest_delivery, frames)
    except providers.ProviderFailure as exc:
        fail(s, job_id, exc, revision=revision)
    except HTTPException as exc:
        fail(s, job_id, providers.ProviderFailure('request_limit', str(exc.detail)), revision=revision)
    finally:
        audio = b''

@app.post('/api/audio')
async def audio_message(request: Request, file: UploadFile = File(...), language: str = Form('auto'), suggest_delivery: bool = Form(False), frames: list[UploadFile] = File(default=[]), recognition_backend: str = Form(''), context: str = Form(''), output_language: str = Form('original'), script: str = Form('auto')):
    s = session(request)
    runtime = getattr(app.state, 'runtime', None)
    backend = recognition_backend or ('adapted' if runtime is not None else 'whisper')
    if backend not in {'adapted', 'whisper'}: raise HTTPException(422, 'Choose an available recognizer.')
    if output_language not in {'original', 'English', 'Hindi/Hinglish'} or script not in {'auto', 'latin', 'devanagari'}:
        raise HTTPException(422, 'Unsupported message language or writing choice.')
    try:
        snapshot = personal.snapshot(personal.Selection.model_validate_json(context), runtime) if context else None
    except ValueError:
        raise HTTPException(422, 'The selected context could not be read.')
    if backend == 'adapted' and (runtime is None or getattr(runtime, 'asr', None) is None):
        raise HTTPException(503, 'Adapted recognition is unavailable. Choose another recognizer explicitly.')
    if language not in {'auto','en','hi'}: raise HTTPException(422, 'Unsupported language setting.')
    ext = Path(file.filename or '').suffix.lower()
    if ext not in {'.webm','.wav','.mp3','.mp4','.m4a','.ogg','.flac','.mpeg','.mpga'}:
        raise HTTPException(422, 'Choose a WAV, WebM, MP3, MP4, M4A, OGG, or FLAC recording.')
    audio = await file.read(MAX_AUDIO + 1)
    await file.close()
    if not audio or len(audio)>MAX_AUDIO: raise HTTPException(413, 'Use a nonempty recording smaller than 20 MB.')
    images = await read_snapshots(frames) if frames else []
    if backend == 'whisper': charge()
    j = begin(s, '', 'audio', language)
    j.update(context=snapshot, output_language=output_language, output_script=script,
             _defer_auto_speak=bool(runtime is not None and (suggest_delivery or images)))
    s.source_audio_hash = hashlib.sha256(audio).hexdigest()
    j['status'] = 'transcribing'
    s.emit('transcription_started')
    if backend == 'adapted':
        s.task = asyncio.create_task(adapted_job(s, j['id'], audio, 'recording'+ext, runtime, suggest_delivery, images))
    else:
        s.task = asyncio.create_task(transcribe_job(s, j['id'], audio, 'recording'+ext, file.content_type or 'application/octet-stream', language, suggest_delivery, images))
    return j

def fail(s, job_id, exc, stage=None, *, revision=None):
    if (s.job and s.job['id'] == job_id and s.job['status'] != 'cancelled'
            and (revision is None or s.job['revision'] == revision)):
        s.job.update(status='review', auto_speak_revision=None, error={'code': exc.code, 'message': exc.message, 'stage': stage})
        s.job.pop('_speech_when_ready', None)
        s.job.pop('_explicit_speech_when_ready', None)
        s.job.pop('_defer_auto_speak', None)
        s.job['revision'] += 1
        s.emit('inference_failed')

class DraftInput(BaseModel):
    revision: int
    output_language: Literal['original','English','Hindi/Hinglish'] = 'original'
    answer: str = Field(default='', max_length=300)
    context: personal.Selection | None = None
    independent: bool = False
    script: Literal['auto', 'latin', 'devanagari'] = 'auto'

def needs_hinglish(j):
    if j['output_language'] == 'English': return False
    detected = str(j.get('metadata', {}).get('transcription', {}).get('language', '')).lower()
    return j['output_language'] == 'Hindi/Hinglish' or j['language'] == 'hi' or detected in {'hi', 'hindi'} or any('\u0900' <= c <= '\u097f' for c in j['text'])

async def draft_job(s, job_id):
    try:
        j = current(s, job_id)
        try:
            check_memory_dependencies(s, j)
        except HTTPException:
            return
        revision = j['revision']
        brief = personal.brief(j.get('context'), j['source_text'], j['answers'], getattr(app.state, 'runtime', None))
        linked = conversation.reference(s.memory_candidate, j['source_text'], j.get('context'), j.get('independent'))
        if linked:
            brief['conversation_reference'] = linked['text']
            # Follow-ups modify the approved message, not saved usual orders.
            brief['personal_wording'] = []
            brief['approved_message'] = ''
        brief['output_script'] = j.get('output_script', 'auto')
        brief['protected_terms'] = wording_fidelity.terms(j.get('context'))
        brief['input_evidence'] = j.get('evidence')
        if j.get('selected_candidate_id'):
            brief['selected_candidate_id'] = j['selected_candidate_id']
        j['conversation'] = linked
        approved = brief.get('approved_message')
        if conversation.followup(j['source_text']) and not linked and not j['answers'] and not j.get('independent'):
            result = {'text': j['source_text'], 'question': 'What should this refer to? Add the item or message you mean.', 'options': []}
            metadata = {'model': 'Conversation boundary', 'elapsed_ms': 0}
        elif approved and j.get('output_script', 'auto') == 'auto' and j.get('context') and j['output_language'] == (j['context']['profile'] or {}).get('language'):
            result = {'text': approved, 'question': '', 'options': []}
            metadata = {'model': 'Saved personal message', 'elapsed_ms': 0}
        else:
            result, metadata = await providers.draft(approved or j['source_text'], j['answers'], j['output_language'], brief)
        result, applied = personal.finish(result, brief)
        if approved: applied.append({'anchor': brief['moment_cue'], 'wording': approved})
        if result['question']: result['text'] = j['source_text']
        if not s.job or s.job['id'] != job_id or s.job['revision'] != revision or s.job['status'] != 'drafting': return
        j = s.job
        try:
            check_memory_dependencies(s, j)
        except HTTPException:
            return
        j.update(**result, status='review')
        j['metadata'] = {**j['metadata'], 'wording': metadata}
        j['context_trace'] = {'scenario': brief.get('scenario', 'general'),
            'recipient': brief.get('recipient', ''), 'listener': brief.get('listener', 'unspecified'),
            'situation': brief.get('situation', ''), 'applied_details': applied,
            'moment_title': brief.get('moment_title', ''),
            'profile_label': (j.get('context', {}).get('profile') or {}).get('label', '') if j.get('context') else ''}
        j['revision'] += 1
        prior_candidate = next((item for item in j.get('candidates', []) if item['id'] == j.get('selected_candidate_id')), None)
        evidence_items = (j.get('evidence') or {}).get('hypotheses', [])
        source_ids = prior_candidate['source_hypothesis_ids'] if prior_candidate else [item['id'] for item in evidence_items] if j.get('wording_source') != 'user_edit' else []
        source_literals = prior_candidate['source_literals'] if prior_candidate else [item['literal_text'] for item in evidence_items] if j.get('wording_source') != 'user_edit' else []
        completed = [] if j['question'] else [{
            'id': f"draft-{j['revision']}", 'text': j['text'], 'reading': j['source_text'],
            'source_hypothesis_ids': source_ids, 'source_literals': source_literals,
            'available': True, 'based_on_user_edit': j.get('wording_source') == 'user_edit',
        }]
        j['candidates'] = retain_literal_choices(j, completed)
        j['selected_candidate_id'] = completed[0]['id'] if completed else None
        if not j['question'] and needs_hinglish(j):
            j['status'] = 'preparing'
            s.emit('pronunciation_started')
            await speech_job(s, job_id)
            if j.get('prepared_speech'):
                mark_for_speech(j)
                s.emit('speech_authorized')
            return
        mark_for_speech(j)
        s.emit('draft_ready')
    except providers.ProviderFailure as exc:
        fail(s, job_id, exc, revision=revision)

    except ValueError:
        fail(s, job_id, providers.ProviderFailure('context_draft', 'The model did not handle your personal wording correctly. Use your original words or try again. No other model was used.'), revision=revision)

@app.post('/api/messages/{job_id}/draft')
async def improve(job_id: str, body: DraftInput, request: Request):
    s = session(request)
    j = current(s, job_id, body.revision)
    if j['status'] in BUSY: raise HTTPException(409, 'Wait for the current request, or cancel it.')
    if not j['original']: raise HTTPException(422, 'No words to improve yet.')
    if len(j['answers']) >= 3: raise HTTPException(422, 'Please edit your message directly after three clarification rounds.')
    if len(j.get('candidates', [])) > 1 and not j.get('selected_candidate_id') and not body.answer.strip():
        raise HTTPException(409, 'Choose an alternative or edit your words before asking for new wording.')
    ranking = j.get('ranking') or {}
    if (ranking.get('route') == 'verified' and ranking.get('decision') != 'selected'
            and j.get('decision_source') != 'user' and not body.answer.strip()):
        raise HTTPException(409, 'Choose an alternative or edit the words to resolve the interpretation first.')
    runtime = getattr(app.state, 'runtime', None)
    snapshot = personal.snapshot(body.context, runtime) if body.context else j.get('context')
    changed_context = snapshot != j.get('context')
    if changed_context:
        if body.answer.strip(): raise HTTPException(409, 'Context changed. Draft the message again before answering.')
        if ranking.get('route') == 'verified' and j.get('decision_source') != 'user':
            j.update(context=snapshot, text=j['source_text'], answers=[], options=[], retrieval=None,
                     confirmed=None, prepared_speech=None, auto_speak_revision=None, context_trace=None,
                     conversation=None, status='review', error=None)
            reset_context_decision(j)
            j['revision'] += 1
            s.audio = None
            s.emit('context_changed')
            return j
    brief = personal.brief(snapshot, j['source_text'], [] if changed_context else j['answers'], runtime)
    local_message = brief.get('approved_message') and not body.answer.strip() and snapshot and body.output_language == (snapshot['profile'] or {}).get('language')
    if body.script != 'auto': local_message = False
    if not local_message: charge()
    if changed_context:
        j.update(answers=[], text=j['source_text'], question='', options=[], retrieval=None)
        j['metadata'] = {key: value for key, value in j['metadata'].items() if key != 'wording'}
    if body.answer.strip():
        j['answers'] = [*j['answers'], {'question': j['question'], 'answer': body.answer.strip()}]
        j['decision_source'] = 'user'
    j.update(status='drafting', output_language=body.output_language, confirmed=None, error=None, auto_speak_revision=None,
             context=snapshot, context_trace=None, prepared_speech=None,
             independent=body.independent, output_script=body.script)
    j['metadata'] = {k: v for k, v in j['metadata'].items() if k != 'pronunciation'}
    j['revision'] += 1
    s.emit('draft_started')
    s.task = asyncio.create_task(draft_job(s, job_id))
    return j

class RevisionInput(BaseModel):
    revision: int


class ConversationInput(RevisionInput):
    action: Literal['separate', 'forget']


@app.post('/api/messages/{job_id}/conversation')
async def change_conversation(job_id: str, body: ConversationInput, request: Request):
    s = session(request)
    j = current(s, job_id, body.revision)
    if j['status'] in BUSY: raise HTTPException(409, 'Finish or stop the current request first.')
    s.memory_candidate = None
    if body.action == 'forget': s.memory = None
    s.audio = None
    j.update(text=j['source_text'], conversation=None, independent=True, answers=[], question='', options=[],
             confirmed=None, prepared_speech=None, context_trace=None, status='review', error=None, auto_speak_revision=None, retrieval=None)
    reset_context_decision(j)
    j['metadata'] = {k:v for k,v in j['metadata'].items() if k not in {'wording', 'pronunciation'}}
    j['revision'] += 1
    s.emit('conversation_cleared')
    return j

async def delivery_suggestion_job(s, job_id, revision, token, audio):
    def active_suggestion():
        j = s.job
        return j and j['id'] == job_id and j['revision'] == revision and j['status'] == 'suggesting' and j.get('delivery_suggestion', {}).get('id') == token
    try:
        result = await delivery_cues.suggest(audio)
        if not active_suggestion(): return
        s.job['delivery_suggestion'] = {'id': token, 'revision': revision, 'state': 'ready', **result}
    except providers.ProviderFailure as exc:
        if not active_suggestion(): return
        s.job['delivery_suggestion'] = {'id': token, 'revision': revision, 'state': 'error', 'message': exc.message}
    except Exception:
        if not active_suggestion(): return
        s.job['delivery_suggestion'] = {'id': token, 'revision': revision, 'state': 'error', 'message': 'Voice analysis could not finish. Your words and tone are unchanged. No other model was used.'}
    finally:
        audio = b''
    if active_suggestion():
        s.job['status'] = 'confirmed' if s.job['confirmed'] else 'review'
        finish_optional_analysis(s.job)
        s.emit('delivery_suggestion_finished')


@app.post('/api/messages/{job_id}/delivery-suggestion')
async def suggest_delivery(job_id: str, request: Request, revision: int = Form(...), file: UploadFile = File(...)):
    s = session(request)
    j = current(s, job_id, revision)
    if j['status'] in BUSY or j['status'] == 'cancelled' or not j['text'].strip():
        raise HTTPException(409, 'Finish your current message before suggesting a delivery.')
    audio = await file.read(MAX_AUDIO + 1)
    await file.close()
    # Recheck after reading: a new recording or edit may have arrived meanwhile.
    j = current(s, job_id, revision)
    if j['status'] in BUSY or j['status'] == 'cancelled':
        raise HTTPException(409, 'The message changed. Try again when ready.')
    if not audio or len(audio) > MAX_AUDIO:
        raise HTTPException(413, 'Use a nonempty recording smaller than 20 MB.')
    if not s.source_audio_hash or hashlib.sha256(audio).hexdigest() != s.source_audio_hash:
        raise HTTPException(409, 'Use the same recording that created this message. Record and transcribe again if needed.')
    return launch_delivery_suggestion(s, j, audio)


def launch_delivery_suggestion(s, j, audio):
    global cue_calls_used
    if not delivery_cues.status()['configured']:
        raise HTTPException(503, 'Voice suggestions need a Gemini API key and FFmpeg. Manual tone selection still works.')
    if cue_calls_used >= delivery_cues.CALL_LIMIT:
        raise HTTPException(429, 'The local voice suggestion limit was reached. Select a tone yourself for now.')
    cue_calls_used += 1
    token = secrets.token_hex(12)
    j['delivery_suggestion'] = {'id': token, 'revision': j['revision'], 'state': 'analyzing'}
    j['status'] = 'suggesting'
    s.emit('delivery_suggestion_started')
    s.task = asyncio.create_task(delivery_suggestion_job(s, j['id'], j['revision'], token, audio))
    return j


class SuggestionStop(RevisionInput):
    suggestion_id: str


@app.post('/api/messages/{job_id}/delivery-suggestion/stop')
async def stop_delivery_suggestion(job_id: str, body: SuggestionStop, request: Request):
    s = session(request)
    j = current(s, job_id, body.revision)
    suggestion = j.get('delivery_suggestion') or {}
    if suggestion.get('id') != body.suggestion_id:
        raise HTTPException(409, 'This voice suggestion is no longer active.')
    if j['status'] == 'suggesting':
        if s.task: s.task.cancel()
        j['status'] = 'confirmed' if j['confirmed'] else 'review'
        j['delivery_suggestion'] = {**suggestion, 'state': 'stopped'}
        j.pop('_speech_when_ready', None)
        j.pop('_explicit_speech_when_ready', None)
        j.pop('_defer_auto_speak', None)
        s.emit('delivery_suggestion_stopped')
    return j


async def read_snapshots(frames):
    if len(frames) != 3: raise HTTPException(422, 'Capture three fresh snapshots.')
    images = []
    for frame in frames:
        data = await frame.read(face_cues.MAX_FRAME + 1)
        await frame.close()
        if len(data) > face_cues.MAX_FRAME: raise HTTPException(413, 'The camera snapshot is too large.')
        if not data.startswith(b'\xff\xd8\xff') or not data.endswith(b'\xff\xd9'):
            raise HTTPException(422, 'Use fresh JPEG camera snapshots.')
        images.append(data)
    return images


async def face_suggestion_job(s, job_id, revision, token, frames):
    def active_face():
        j = s.job
        return j and j['id'] == job_id and j['revision'] == revision and j['status'] == 'checking_face' and j.get('face_suggestion', {}).get('id') == token
    try:
        result = await face_cues.suggest(frames)
        if not active_face(): return
        s.job['face_suggestion'] = {'id': token, 'revision': revision, 'state': 'ready', **result}
    except providers.ProviderFailure as exc:
        if not active_face(): return
        s.job['face_suggestion'] = {'id': token, 'revision': revision, 'state': 'error', 'message': exc.message}
    except Exception:
        if not active_face(): return
        s.job['face_suggestion'] = {'id': token, 'revision': revision, 'state': 'error', 'message': 'Camera analysis could not finish. Your words and delivery are unchanged.'}
    finally:
        frames.clear()
    if active_face():
        s.job['status'] = 'confirmed' if s.job['confirmed'] else 'review'
        s.emit('face_suggestion_finished')


@app.post('/api/messages/{job_id}/face-suggestion')
async def suggest_face(job_id: str, request: Request, revision: int = Form(...), frames: list[UploadFile] = File(...)):
    global face_calls_used
    s = session(request)
    j = current(s, job_id, revision)
    if j['status'] in BUSY or j['status'] == 'cancelled' or not j['text'].strip():
        raise HTTPException(409, 'Finish your message before checking facial cues.')
    images = await read_snapshots(frames)
    j = current(s, job_id, revision)
    if j['status'] in BUSY or j['status'] == 'cancelled': raise HTTPException(409, 'The message changed. Capture again when ready.')
    if not face_cues.configured(): raise HTTPException(503, 'The selected facial analysis mode is not configured.')
    if face_calls_used >= face_cues.CALL_LIMIT: raise HTTPException(429, 'The local camera analysis limit was reached. Use voice or manual tone selection.')
    face_calls_used += 1
    token = secrets.token_hex(12)
    j['face_suggestion'] = {'id': token, 'revision': revision, 'state': 'analyzing'}
    j['status'] = 'checking_face'
    s.emit('face_suggestion_started')
    s.task = asyncio.create_task(face_suggestion_job(s, job_id, revision, token, images))
    return j


@app.post('/api/messages/{job_id}/face-suggestion/stop')
async def stop_face_suggestion(job_id: str, body: SuggestionStop, request: Request):
    s = session(request)
    j = current(s, job_id, body.revision)
    suggestion = j.get('face_suggestion') or {}
    if suggestion.get('id') != body.suggestion_id: raise HTTPException(409, 'This camera check is no longer active.')
    if j['status'] == 'checking_face':
        if s.task: s.task.cancel()
        j['status'] = 'confirmed' if j['confirmed'] else 'review'
        j['face_suggestion'] = {**suggestion, 'state': 'stopped'}
        s.emit('face_suggestion_stopped')
    return j


def launch_recording_analysis(s, j, audio, frames, include_voice):
    global cue_calls_used, face_calls_used
    token = secrets.token_hex(12)
    revision = j['revision']
    j['analysis_id'] = token
    j['status'] = 'analyzing_delivery'
    jobs = []
    def prepare(field, configured, used, limit, message):
        if not configured or used >= limit:
            j[field] = {'id': token, 'revision': revision, 'state': 'error', 'message': message}
            return False
        j[field] = {'id': token, 'revision': revision, 'state': 'analyzing'}
        return True
    if include_voice and prepare('delivery_suggestion', delivery_cues.status()['configured'], cue_calls_used, delivery_cues.CALL_LIMIT,
                                  'Voice delivery analysis is not configured or its local limit was reached. Your transcript is ready.'):
        cue_calls_used += 1
        jobs.append(('delivery_suggestion', delivery_cues.suggest, audio))
    if prepare('face_suggestion', face_cues.configured(), face_calls_used, face_cues.CALL_LIMIT,
               'Facial cue analysis is not configured or its local limit was reached. Your transcript is ready.'):
        face_calls_used += 1
        jobs.append(('face_suggestion', face_cues.suggest, frames))
    s.emit('recording_analysis_started')
    s.task = asyncio.create_task(recording_analysis_job(s, j['id'], revision, token, jobs))


async def recording_analysis_job(s, job_id, revision, token, jobs):
    def active_analysis():
        j = s.job
        return j and j['id'] == job_id and j['revision'] == revision and j['analysis_id'] == token and j['status'] == 'analyzing_delivery'
    async def run(field, provider, data):
        try:
            result = await provider(data)
            record = {'state': 'ready', **result}
        except providers.ProviderFailure as exc:
            record = {'state': 'error', 'message': exc.message}
        except Exception:
            record = {'state': 'error', 'message': 'This delivery check could not finish. Your transcript and settings are unchanged.'}
        if active_analysis():
            s.job[field] = {'id': token, 'revision': revision, **record}
            s.emit('recording_cue_finished')
    try:
        await asyncio.gather(*(run(*job) for job in jobs))
        if active_analysis():
            s.job['status'] = 'review'
            finish_optional_analysis(s.job)
            s.emit('recording_analysis_finished')
    finally:
        jobs.clear()


class AnalysisStop(RevisionInput):
    analysis_id: str


@app.post('/api/messages/{job_id}/recording-analysis/stop')
async def stop_recording_analysis(job_id: str, body: AnalysisStop, request: Request):
    s = session(request)
    j = current(s, job_id, body.revision)
    if j.get('analysis_id') != body.analysis_id: raise HTTPException(409, 'This analysis is no longer active.')
    if j['status'] == 'analyzing_delivery':
        if s.task: s.task.cancel()
        j['status'] = 'review'
        j.pop('_speech_when_ready', None)
        j.pop('_explicit_speech_when_ready', None)
        j.pop('_defer_auto_speak', None)
        for field in ('delivery_suggestion', 'face_suggestion'):
            if j.get(field, {}).get('state') == 'analyzing':
                j[field] = {**j[field], 'state': 'stopped'}
        s.emit('recording_analysis_stopped')
    return j


class EditInput(RevisionInput):
    text: str = Field(min_length=1, max_length=2000)
    selection: bool = False


class ChooseInput(RevisionInput):
    candidate_id: str = Field(min_length=1, max_length=100)


@app.post('/api/messages/{job_id}/choose')
async def choose_candidate(job_id: str, body: ChooseInput, request: Request):
    s = session(request)
    j = current(s, job_id, body.revision)
    check_memory_dependencies(s, j)
    if j['status'] in BUSY or j['status'] == 'cancelled':
        raise HTTPException(409, 'Finish or stop the current request before choosing.')
    if j['question'] and j['options']:
        raise HTTPException(409, 'Choose an option from the current clarification first.')
    candidate = next((item for item in j.get('candidates', []) if item['id'] == body.candidate_id), None)
    if candidate is None: raise HTTPException(404, 'This alternative is no longer available.')
    s.audio = None
    j.update(text=candidate['text'], source_text=candidate['text'], selected_candidate_id=candidate['id'],
             question='', options=[], confirmed=None, prepared_speech=None, status='review', auto_speak_revision=None,
             decision_source='user')
    j['revision'] += 1
    # A tap approves the exact visible words, including an explicitly labelled
    # raw alternative after wording assistance failed. Arrival never does so.
    j['error'] = None
    mark_for_speech(j, explicit=True)
    s.emit('candidate_selected')
    return j

class SpeechDelivery(BaseModel):
    # User-selected delivery intent, separate from message wording and ASR.
    model_config = {'extra': 'forbid'}
    tone: Literal['neutral', 'warm', 'cheerful', 'firm'] = 'neutral'
    rate: float = Field(default=0.9, ge=0.5, le=1.5, allow_inf_nan=False, strict=True)

class ConfirmInput(RevisionInput):
    output_language: Literal['original', 'English', 'Hindi/Hinglish'] | None = None
    context: personal.Selection | None = None
    pronunciation: Literal['auto', 'original'] = 'auto'
    delivery: SpeechDelivery = Field(default_factory=SpeechDelivery)

@app.post('/api/messages/{job_id}/edit')
async def edit(job_id: str, body: EditInput, request: Request):
    s = session(request)
    j = current(s, job_id, body.revision)
    if j['status'] in BUSY: raise HTTPException(409, 'Cancel the pending request before editing.')
    if not body.text.strip(): raise HTTPException(422, 'Enter a few words first.')
    if body.selection and body.text.strip() not in j['options']:
        raise HTTPException(409, 'Choose an option from the current question.')
    if body.selection:
        selection = (j.get('context') or {}).get('selection') or {}
        try:
            wording_fidelity.validate_selection(
                body.text.strip(), j['source_text'], j['answers'], j['output_language'],
                {'output_script': j.get('output_script', 'auto'),
                 'protected_terms': wording_fidelity.terms(j.get('context')),
                 'recipient': selection.get('recipient', ''),
                 'conversation_reference': (j.get('conversation') or {}).get('text', '')}, j['options'])
        except ValueError as exc:
            raise HTTPException(422, 'This option changed a protected name or writing choice. Edit the words before speaking.') from exc
    s.audio = None
    j.update(text=body.text.strip(), source_text=body.text.strip(), answers=[], confirmed=None, status='review', question='', options=[], error=None, context_trace=None, prepared_speech=None,
             candidates=[], selected_candidate_id=None, auto_speak_revision=None, wording_source='user_edit',
             decision_source='user', retrieval=None)
    j['conversation'] = None
    j['metadata'] = {k: v for k, v in j['metadata'].items() if k != 'pronunciation'}
    j['revision'] += 1
    if (j.get('evidence') or {}).get('recognizer') == 'adapted':
        # The edit settles the displayed wording without asserting that any
        # original acoustic beam said it. Keep those beams separately available.
        authored = {'id': f"edited-{j['revision']}", 'text': j['text'], 'reading': j['text'],
                    'source_hypothesis_ids': [], 'source_literals': [], 'available': True,
                    'based_on_user_edit': True, 'retrieved_sources': [], 'contextual_additions': []}
        j['candidates'] = retain_literal_choices(j, [authored])
        j['selected_candidate_id'] = authored['id']
    if body.selection: mark_for_speech(j, explicit=True)
    s.emit('message_edited')
    return j

@app.post('/api/messages/{job_id}/confirm')
async def confirm(job_id: str, body: ConfirmInput, request: Request):
    s = session(request)
    j = current(s, job_id, body.revision)
    check_memory_dependencies(s, j)
    if j['status'] in BUSY or j['status'] == 'cancelled' or not j['text'].strip():
        raise HTTPException(409, 'Review a completed message before confirming it.')
    if j['question']: raise HTTPException(409, 'Choose an option or edit the message to resolve the question first.')
    ranking = j.get('ranking') or {}
    if (ranking.get('route') == 'verified' and j.get('decision_source') != 'user'
            and (ranking.get('decision') != 'selected' or not j.get('selected_candidate_id'))):
        raise HTTPException(409, 'Choose an alternative or edit the words before speaking.')
    snapshot = personal.snapshot(body.context, getattr(app.state, 'runtime', None)) if body.context else j.get('context')
    if j.get('context') and snapshot != j['context']:
        raise HTTPException(409, 'Context changed. Review the message with the new context first.')
    output_language = body.output_language if body.output_language is not None else j['output_language']
    if snapshot != j.get('context') or output_language != j['output_language']:
        j.update(context=snapshot, output_language=output_language, prepared_speech=None, confirmed=None)
        j['metadata'] = {k: v for k, v in j['metadata'].items() if k != 'pronunciation'}
        j['revision'] += 1
    prepared = j.get('prepared_speech')
    valid_preparation = prepared and prepared['revision'] == j['revision'] and prepared['display_text'] == j['text']
    if body.pronunciation == 'auto' and needs_hinglish(j) and not valid_preparation:
        j.update(status='preparing', confirmed=None, prepared_speech=None, error=None)
        j['revision'] += 1
        s.emit('pronunciation_started')
        s.task = asyncio.create_task(speech_job(s, job_id))
        try:
            await asyncio.shield(s.task)
        except asyncio.CancelledError:
            raise HTTPException(409, 'Speech preparation was cancelled. Nothing was spoken.')
        j = current(s, job_id)
        if j['status'] != 'review' or not j.get('prepared_speech'):
            raise HTTPException(503, (j.get('error') or {}).get('message', 'Speech could not be prepared. Your message is still here.'))
    delivery = {**body.delivery.model_dump(), 'source': 'user'}
    if j.get('confirmed') and (j['confirmed'].get('pronunciation', 'auto') != body.pronunciation
                              or j['confirmed'].get('delivery') != delivery):
        j['confirmed'] = None
    if not j['confirmed']:
        s.audio = None
        prepared = j.get('prepared_speech')
        speech_text = prepared['speech_text'] if body.pronunciation == 'auto' and prepared and prepared['revision'] == j['revision'] and prepared['display_text'] == j['text'] else j['text']
        j['confirmed'] = {'version': j['revision'], 'text': j['text'], 'speech_text': speech_text, 'pronunciation': body.pronunciation, 'timestamp': time.time(), 'id': secrets.token_hex(12), 'context': j.get('context'), 'delivery': delivery}
        j['status'] = 'confirmed'
        s.memory = conversation.remember(j)
        s.emit('message_confirmed')
    return j

class AudioInput(RevisionInput):
    confirmation_id: str = Field(min_length=1, max_length=64)


def confirmed_audio(s, job_id, body):
    j = current(s, job_id, body.revision)
    check_memory_dependencies(s, j)
    if not j.get('confirmed') or j['confirmed']['id'] != body.confirmation_id or j['status'] == 'cancelled':
        raise HTTPException(409, 'Review and confirm the current message before generating its voice.')
    return j


async def synthesize_job(s, job_id, confirmation_id, body, cache_key):
    def active():
        return (s.job and s.job['id'] == job_id and s.job['status'] == 'synthesizing'
                and (s.job.get('confirmed') or {}).get('id') == confirmation_id
                and s.task is asyncio.current_task())
    try:
        data, metadata = await fish.synthesize(body)
        if not active(): return
        j = s.job
        s.audio = {'key': cache_key, 'data': data}
        j.update(status='confirmed', synthesis={'state': 'ready', 'confirmation_id': confirmation_id, **metadata})
        s.emit('audio_ready')
    except providers.ProviderFailure as exc:
        if not active(): return
        j = s.job
        j.update(status='confirmed', synthesis={'state': 'error', 'confirmation_id': confirmation_id,
                                              'message': exc.message, 'code': exc.code})
        s.emit('synthesis_failed')
    except Exception:
        # Never leave the UI busy or expose transport diagnostics/credentials.
        if active():
            s.job.update(status='confirmed', synthesis={'state': 'error', 'confirmation_id': confirmation_id,
                'message': 'Voice generation failed unexpectedly. Choose device voice or retry explicitly.', 'code': 'fish_failed'})
            s.emit('synthesis_failed')


@app.post('/api/messages/{job_id}/audio')
async def generate_audio(job_id: str, body: AudioInput, request: Request):
    global fish_calls_used
    s = session(request)
    j = confirmed_audio(s, job_id, body)
    if j['status'] in BUSY:
        raise HTTPException(409, 'Voice preparation is already running. Wait or stop it first.')
    if not fish.status()['configured']:
        raise HTTPException(503, 'Fish Audio needs a local API key and voice ID. Device voice remains available.')
    _, voice = fish.configuration()
    cache_key = (body.confirmation_id, fish.MODEL, voice)
    if not s.audio or s.audio['key'] != cache_key:
        payload = fish.payload(j['confirmed'], voice)
        if fish_calls_used >= FISH_CALL_LIMIT:
            raise HTTPException(429, 'The local Fish Audio trial limit was reached. No further synthesis was started.')
        fish_calls_used += 1
        j.update(status='synthesizing', synthesis={'state': 'generating', 'confirmation_id': body.confirmation_id})
        s.emit('synthesis_started')
        task = asyncio.create_task(synthesize_job(s, job_id, body.confirmation_id, payload, cache_key))
        s.task = task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            raise HTTPException(409, 'Voice generation stopped. Nothing was played.')
    j = confirmed_audio(s, job_id, body)
    if not s.audio or s.audio['key'] != cache_key:
        raise HTTPException(503, j.get('synthesis', {}).get('message', 'Voice generation stopped. Nothing was played.'))
    return Response(s.audio['data'], media_type='audio/mpeg',
                    headers={'X-Confirmation-ID': body.confirmation_id, 'X-Voice-Model': fish.MODEL})


@app.post('/api/messages/{job_id}/audio/stop')
async def stop_audio(job_id: str, body: AudioInput, request: Request):
    s = session(request)
    j = confirmed_audio(s, job_id, body)
    if j['status'] == 'synthesizing':
        if s.task: s.task.cancel()
        s.audio = None
        j.update(status='confirmed', synthesis={'state': 'stopped', 'confirmation_id': body.confirmation_id})
        s.emit('synthesis_stopped')
    return j

@app.post('/api/messages/{job_id}/cancel')
async def cancel(job_id: str, request: Request):
    s = session(request)
    j = current(s, job_id)
    if s.task: s.task.cancel()
    s.audio = None
    j.update(status='cancelled', confirmed=None, prepared_speech=None, auto_speak_revision=None)
    for field in ('_speech_when_ready', '_explicit_speech_when_ready', '_defer_auto_speak'):
        j.pop(field, None)
    j['revision'] += 1
    s.emit('message_cancelled')
    return j

@app.get('/api/profiles')
async def profiles(request: Request):
    session(request)
    return personal.list_profiles()

@app.post('/api/profiles')
async def save_profile(body: personal.Profile, request: Request):
    session(request)
    return personal.save_profile(body)

class ContextInput(RevisionInput):
    context: personal.Selection

@app.post('/api/messages/{job_id}/context')
async def change_context(job_id: str, body: ContextInput, request: Request):
    s = session(request)
    j = current(s, job_id, body.revision)
    if j['status'] in BUSY: raise HTTPException(409, 'Finish or cancel this request before changing context.')
    snapshot = personal.snapshot(body.context, getattr(app.state, 'runtime', None))
    j.update(context=snapshot, context_trace=None, text=j['source_text'], answers=[], confirmed=None, prepared_speech=None,
             status='review', question='', options=[], error=None, conversation=None, auto_speak_revision=None, retrieval=None)
    reset_context_decision(j)
    j['metadata'] = {key: value for key, value in j['metadata'].items() if key not in {'wording', 'pronunciation'}}
    j['revision'] += 1
    s.emit('context_changed')
    return j

async def speech_job(s, job_id):
    try:
        j = current(s, job_id)
        revision = j['revision']
        profile = (j.get('context') or {}).get('profile') or {}
        names = wording_fidelity.terms(j.get('context'))
        protected = set(names)
        prepared = await hinglish.prepare(j['text'], protected, charge)
        if not s.job or s.job['id'] != job_id or s.job['revision'] != revision or s.job['status'] != 'preparing': return
        j = s.job
        try:
            check_memory_dependencies(s, j)
        except HTTPException:
            return
        j['revision'] += 1
        j.update(status='review', prepared_speech={**prepared, 'revision': j['revision']})
        j['metadata'] = {**j['metadata'], 'pronunciation': prepared['metadata']}
        s.emit('pronunciation_ready')
    except providers.ProviderFailure as exc:
        fail(s, job_id, exc, 'pronunciation', revision=revision)
    except HTTPException as exc:
        fail(s, job_id, providers.ProviderFailure('hinglish_limit', str(exc.detail)), 'pronunciation', revision=revision)

class SpeechInput(RevisionInput):
    context: personal.Selection | None = None

@app.post('/api/messages/{job_id}/speech')
async def prepare_speech(job_id: str, body: SpeechInput, request: Request):
    s = session(request)
    j = current(s, job_id, body.revision)
    if j['status'] in BUSY or j['status'] == 'cancelled' or not j['text'].strip() or j['question']:
        raise HTTPException(409, 'Finish reviewing the message before preparing pronunciation.')
    if body.context:
        snapshot = personal.snapshot(body.context, getattr(app.state, 'runtime', None))
        if snapshot != j.get('context'): j['context_trace'] = None
        j['context'] = snapshot
    j.update(status='preparing', confirmed=None, error=None, prepared_speech=None, auto_speak_revision=None)
    j['revision'] += 1
    s.emit('pronunciation_started')
    s.task = asyncio.create_task(speech_job(s, job_id))
    return j

@app.post('/api/messages/{job_id}/speech/reset')
async def reset_speech(job_id: str, body: RevisionInput, request: Request):
    s = session(request)
    j = current(s, job_id, body.revision)
    if j['status'] in BUSY: raise HTTPException(409, 'Finish or cancel the pending request first.')
    j.update(prepared_speech=None, confirmed=None, status='review', auto_speak_revision=None)
    j['revision'] += 1
    j['metadata'] = {k: v for k, v in j['metadata'].items() if k != 'pronunciation'}
    s.emit('pronunciation_removed')
    return j


# Both transports share these handlers; storage approval never calls confirm.
from . import memory, memory_routes
memory.set_invalidator(invalidate_memory_context)
app.include_router(memory_routes.router(session, current))
