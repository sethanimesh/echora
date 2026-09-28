"""Native transport for the same revision-bound communication application.

An opaque, in-memory session token replaces browser cookies. Every operation
still calls the shared ownership/revision checks; browser origins stay bounded.
"""
from http.cookies import SimpleCookie

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field

from communication.backend import app as communication, context
from communication.backend.profile_adapter import as_legacy_profile, legacy_summary, merge_legacy_profile
from app.personal import UserProfileInput
from app.schemas import AudienceResolutionRequest


async def boundary(request: Request, response: Response):
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    origin = request.headers.get('origin')
    if origin and origin not in communication.ORIGINS:
        raise HTTPException(403, 'Unrecognized origin.')
    if request.method not in {'GET', 'HEAD', 'OPTIONS'} and request.headers.get('x-echora-client') != '1':
        raise HTTPException(403, 'Missing application request header.')


router = APIRouter(prefix='/api/v1/communication', dependencies=[Depends(boundary)])


def native_request(request):
    token = request.headers.get('x-echora-session', '')
    if token and (len(token) > 100 or any(char not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for char in token)):
        raise HTTPException(401, 'Your session ended. Start a new session.')
    scope = dict(request.scope)
    scope['headers'] = [(key, value) for key, value in scope['headers'] if key.lower() != b'cookie']
    scope['headers'].append((b'cookie', ('echora_session=' + token).encode()))
    return Request(scope, receive=request.receive)


@router.get('/session')
async def bootstrap(request: Request):
    translated, response = native_request(request), Response()
    result = await communication.get_session(translated, response)
    cookie = SimpleCookie()
    cookie.load(response.headers.get('set-cookie', ''))
    token = cookie['echora_session'].value if 'echora_session' in cookie else request.headers.get('x-echora-session', '')
    return {**result, 'session_token': token, 'reference_ttl_seconds': 600}


@router.get('/state')
async def state(request: Request):
    session = communication.session(native_request(request))
    return {'job': session.job, 'sequence': session.sequence}


@router.post('/audio')
async def audio(request: Request, file: UploadFile = File(...), selection: str = Form(''), recognition_backend: str = Form('adapted')):
    translated = native_request(request)
    communication.session(translated)
    try:
        snapshot = context.snapshot(context.Selection.model_validate_json(selection)) if selection else None
    except ValueError:
        raise HTTPException(422, 'The selected context could not be read.')
    # Native has no per-message language control, so honor the selected shared
    # profile. The shared handler rechecks its revision before starting work.
    language = ((snapshot or {}).get('profile') or {}).get('language', 'original')
    return await communication.audio_message(translated, file=file, language='auto',
        suggest_delivery=False, frames=[], recognition_backend=recognition_backend, context=selection,
        output_language=language, script='auto')


@router.post('/messages')
async def text_message(body: communication.TextInput, request: Request):
    return await communication.text_message(body, native_request(request))


@router.post('/messages/{job_id}/choose')
async def choose(job_id: str, body: communication.ChooseInput, request: Request):
    return await communication.choose_candidate(job_id, body, native_request(request))


@router.post('/messages/{job_id}/edit')
async def edit(job_id: str, body: communication.EditInput, request: Request):
    return await communication.edit(job_id, body, native_request(request))


@router.post('/messages/{job_id}/confirm')
async def confirm(job_id: str, body: communication.ConfirmInput, request: Request):
    return await communication.confirm(job_id, body, native_request(request))


@router.post('/messages/{job_id}/cancel')
async def cancel(job_id: str, request: Request):
    return await communication.cancel(job_id, native_request(request))


@router.post('/messages/{job_id}/conversation')
async def conversation(job_id: str, body: communication.ConversationInput, request: Request):
    return await communication.change_conversation(job_id, body, native_request(request))


@router.post('/messages/{job_id}/speech')
async def speech(job_id: str, body: communication.AudioInput, request: Request):
    session = communication.session(native_request(request))
    job = communication.confirmed_audio(session, job_id, body)
    confirmed = dict(job['confirmed'])
    text = confirmed['speech_text']
    engine = getattr(request.app.state, 'speech', None)
    # Never play a truncated provider clip for a longer displayed message.
    # The device speaks the complete authorized text when the provider cannot.
    try:
        clip = await engine.synthesize(text) if engine is not None and len(text) <= 200 else None
    except Exception:
        clip = None
    latest = communication.confirmed_audio(session, job_id, body)
    if latest['confirmed']['id'] != confirmed['id']:
        raise HTTPException(409, 'The message changed before speech was ready.')
    return {'job': latest, 'speech_text': text, 'speech': clip.model_dump() if clip else None}


def _profile(profile_id):
    try:
        return context.get_profile(profile_id)
    except HTTPException as error:
        if error.status_code == 404 and profile_id == 'user':
            return context.Profile(id='user', label='You')
        raise


def _legacy(document):
    return {**as_legacy_profile(document).model_dump(), 'revision': document.revision}


@router.get('/profiles')
async def profiles(request: Request):
    communication.session(native_request(request))
    documents = [context.Profile.model_validate(value) for value in context.list_profiles()]
    if not any(document.id == 'user' for document in documents):
        documents.append(_profile('user'))
    return [{**legacy_summary(document).model_dump(), 'revision': document.revision} for document in documents]


@router.get('/profiles/{profile_id}')
async def profile(profile_id: str, request: Request):
    communication.session(native_request(request))
    return _legacy(_profile(profile_id))


class NativeProfileSave(BaseModel):
    revision: int = Field(ge=0)
    profile: UserProfileInput


@router.put('/profiles/{profile_id}')
async def save_profile(profile_id: str, body: NativeProfileSave, request: Request):
    communication.session(native_request(request))
    original = _profile(profile_id)
    if original.revision != body.revision:
        raise HTTPException(409, 'This profile changed. Reload before saving.')
    merged = merge_legacy_profile(original, body.profile)
    if original.sample or (profile_id == 'user' and original.revision == 0):
        merged = merged.model_copy(update={'id': '', 'revision': 0, 'sample': False})
    saved = context.Profile.model_validate(context.save_profile(merged))
    return {'profile': _legacy(saved), 'refused': []}


class NativeAudienceRequest(AudienceResolutionRequest):
    place_id: str = ''


@router.post('/audience/resolve')
async def resolve_audience(body: NativeAudienceRequest, request: Request):
    communication.session(native_request(request))
    from communication.backend.profile_adapter import resolve_snapshot
    profile = _profile(body.persona) if body.persona else None
    selection = context.Selection(scenario='outside' if body.context == 'outdoors' else body.context,
        core_context=body.context, audience_id=body.audience, declared_listener=body.declared_listener, place_id=body.place_id)
    data = profile.model_dump() if profile else None
    return resolve_snapshot(selection.model_dump(), data)['resolved_audience']


from communication.backend import memory_routes
router.include_router(memory_routes.router(communication.session, communication.current,
    request_adapter=native_request, prefix=''))
