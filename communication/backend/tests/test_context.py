import asyncio
import pytest
from fastapi.testclient import TestClient
from communication.backend import app as module, context, providers

HEADERS = {'X-Echora-Client': '1'}

@pytest.fixture
def client(monkeypatch, tmp_path):
    async def prepare(text, protected, charge):
        return {'display_text': text, 'pronunciation_text': text, 'speech_text': text, 'changes': [], 'metadata': {'model': 'test', 'elapsed_ms': 0, 'ambiguous_spans': 0}}
    monkeypatch.setattr(module.hinglish, 'prepare', prepare)
    module.sessions.clear()
    monkeypatch.setattr(module, 'calls_used', 0)
    monkeypatch.setenv('GROQ_API_KEY', 'test-only')
    monkeypatch.setenv('ECHORA_PROFILE_DB', str(tmp_path / 'profiles.sqlite3'))
    with TestClient(module.app) as c:
        c.get('/api/session')
        yield c

def selection(scenario='cafe', **values):
    return context.Selection(profile_id='sample-tea', profile_revision=1, scenario=scenario, **values)

def test_scoping_qualifiers_negation_and_answers():
    snap = context.snapshot(selection())
    assert context.brief(snap, 'tea', [])['personal_wording'][0]['wording'] == 'Lipton green tea'
    assert not context.brief(context.snapshot(selection('home')), 'tea', [])['personal_wording']
    for text in ['black tea', 'no tea', 'I do not want tea', 'two teas', 'coffee', 'tea or coffee']:
        assert not context.brief(snap, text, [])['personal_wording']
    assert not context.brief(snap, 'tea', [{'answer': 'black tea'}])['personal_wording']
    assert not context.brief(context.snapshot(selection(use_personal_wording=False)), 'tea', [])['personal_wording']

def test_same_anchor_can_have_different_scenarios():
    rules = [context.Rule(anchor='tea', wording='green tea', scenarios=['cafe']), context.Rule(anchor='tea', wording='ginger tea', scenarios=['home'])]
    context.Profile(label='Test', rules=rules)
    with pytest.raises(ValueError):
        context.Profile(label='Test', rules=rules + [context.Rule(anchor='tea', wording='black tea', scenarios=['home'])])

def test_profiles_saved_with_revision_and_sample_isolation(client):
    sample = client.get('/api/profiles').json()[0]
    assert client.post('/api/profiles', json=sample, headers=HEADERS).status_code == 422
    custom = {**sample, 'id': '', 'revision': 0, 'sample': False, 'label': 'My words'}
    saved = client.post('/api/profiles', json=custom, headers=HEADERS).json()
    assert saved['revision'] == 1
    assert context.get_profile(saved['id']).label == 'My words'
    changed = client.post('/api/profiles', json={**saved, 'label': 'New words'}, headers=HEADERS).json()
    assert changed['revision'] == 2
    assert client.post('/api/profiles', json=saved, headers=HEADERS).status_code == 409
    assert context.get_profile('sample-tea').sample
    with pytest.raises(Exception) as error:
        context.snapshot(context.Selection(profile_id=saved['id'], profile_revision=1))
    assert error.value.status_code == 409

def test_context_draft_and_context_change_invalidate_confirmation(client, monkeypatch):
    calls = []
    async def draft(text, answers, language, brief):
        calls.append(brief)
        return {'text': 'Please bring me tea.', 'question': '', 'options': []}, {'model': providers.TEXT_MODEL, 'elapsed_ms': 5}
    monkeypatch.setattr(providers, 'draft', draft)
    j = client.post('/api/messages', json={'text': 'tea'}, headers=HEADERS).json()
    path = f"/api/messages/{j['id']}"
    client.post(path + '/draft', json={'revision': 1, 'context': selection().model_dump()}, headers=HEADERS)
    j = client.get('/api/session').json()['job']
    assert j['text'] == 'Please bring me Lipton green tea.'
    assert j['original'] == j['source_text'] == 'tea'
    assert j['context_trace']['applied_details'][0]['wording'] == 'Lipton green tea'
    confirmed = client.post(path + '/confirm', json={'revision': j['revision']}, headers=HEADERS).json()
    assert confirmed['confirmed']['context']['selection']['scenario'] == 'cafe'
    switched = client.post(path + '/context', json={'revision': j['revision'], 'context': selection('home').model_dump()}, headers=HEADERS).json()
    assert switched['confirmed'] is None
    assert switched['text'] == 'tea'
    assert switched['context_trace'] is None
    assert client.post(path + '/confirm', json={'revision': j['revision']}, headers=HEADERS).status_code == 409
    client.post(path + '/draft', json={'revision': switched['revision']}, headers=HEADERS)
    assert client.get('/api/session').json()['job']['text'] == 'Please bring me tea.'
    assert calls[-1]['personal_wording'] == []

def test_ask_first_and_clarification_not_force_enriched():
    brief = context.brief(context.snapshot(selection('home')), 'soup', [])
    assert brief['personal_wording'][0]['mode'] == 'ask'
    proposed = {'text': 'Please bring me soup.', 'question': 'Would you like tomato soup?', 'options': []}
    result, applied = context.finish(proposed, brief)
    assert result == proposed and not applied

def test_context_cannot_change_during_inference_or_cross_session(client, monkeypatch):
    async def delayed(*args):
        await asyncio.sleep(10)
    monkeypatch.setattr(providers, 'draft', delayed)
    j = client.post('/api/messages', json={'text': 'tea'}, headers=HEADERS).json()
    path = f"/api/messages/{j['id']}"
    running = client.post(path + '/draft', json={'revision': 1, 'context': selection().model_dump()}, headers=HEADERS).json()
    body = {'revision': running['revision'], 'context': selection('home').model_dump()}
    assert client.post(path + '/context', json=body, headers=HEADERS).status_code == 409
    cookies = dict(client.cookies)
    client.cookies.clear()
    client.get('/api/session')
    assert client.post(path + '/context', json=body, headers=HEADERS).status_code == 404
    client.cookies.clear()
    client.cookies.update(cookies)
    client.post(path + '/cancel', headers=HEADERS)

def test_ask_first_failure_is_reported_and_direct_context_change_drops_old_text(client, monkeypatch):
    async def invalid(*args):
        return {'text': 'Please bring me tomato soup.', 'question': '', 'options': []}, {}
    monkeypatch.setattr(providers, 'draft', invalid)
    j = client.post('/api/messages', json={'text': 'soup'}, headers=HEADERS).json()
    path = f"/api/messages/{j['id']}"
    client.post(path + '/draft', json={'revision': 1, 'context': selection('home').model_dump()}, headers=HEADERS)
    failed = client.get('/api/session').json()['job']
    assert failed['error']['code'] == 'context_draft'
    assert failed['text'] == 'soup'

    async def good(*args):
        return {'text': 'Please bring me tea.', 'question': '', 'options': []}, {}
    monkeypatch.setattr(providers, 'draft', good)
    j = client.post('/api/messages', json={'text': 'tea'}, headers=HEADERS).json()
    path = f"/api/messages/{j['id']}"
    client.post(path + '/draft', json={'revision': 1, 'context': selection().model_dump()}, headers=HEADERS)
    j = client.get('/api/session').json()['job']
    assert 'Lipton' in j['text']
    async def failed(*args):
        raise providers.ProviderFailure('rate_limit', 'Limit reached')
    monkeypatch.setattr(providers, 'draft', failed)
    client.post(path + '/draft', json={'revision': j['revision'], 'context': selection('home').model_dump()}, headers=HEADERS)
    j = client.get('/api/session').json()['job']
    assert j['text'] == 'tea' and j['context']['selection']['scenario'] == 'home'
    assert j['confirmed'] is None and j['context_trace'] is None

def test_personal_moment_requires_exact_cue_and_selected_setting():
    chosen = selection('cafe', moment_id='tea-stop')
    snap = context.snapshot(chosen)
    assert context.brief(snap, 'usual', [])['approved_message'] == 'Could I have Lipton green tea, please?'
    for text in ['not usual', 'usual black tea', 'not my usual', 'tea']:
        assert not context.brief(snap, text, [])['approved_message']
    assert not context.brief(snap, 'usual', [{'answer': 'no'}])['approved_message']
    assert not context.brief(context.snapshot(selection('cafe')), 'usual', [])['approved_message']
    with pytest.raises(Exception) as error:
        context.snapshot(selection('home', moment_id='tea-stop'))
    assert error.value.status_code == 422

def test_saved_moment_needs_no_cloud_and_keeps_confirmation_gate(client, monkeypatch):
    monkeypatch.delenv('GROQ_API_KEY')
    monkeypatch.setattr(module, 'calls_used', module.CALL_LIMIT)
    async def forbidden(*args):
        raise AssertionError('Saved message should not invoke a model')
    monkeypatch.setattr(providers, 'draft', forbidden)
    j = client.post('/api/messages', json={'text': 'usual'}, headers=HEADERS).json()
    path = f"/api/messages/{j['id']}"
    r = client.post(path + '/draft', json={'revision': 1, 'output_language': 'English', 'context': selection('cafe', moment_id='tea-stop').model_dump()}, headers=HEADERS)
    assert r.status_code == 200
    j = client.get('/api/session').json()['job']
    assert j['text'] == 'Could I have Lipton green tea, please?'
    assert j['confirmed'] is None and j['original'] == 'usual'
    assert j['metadata']['wording']['model'] == 'Saved personal message'
    assert j['context_trace']['moment_title'] == 'My usual tea stop'
    changed = client.post(path + '/context', json={'revision': j['revision'], 'context': selection('home').model_dump()}, headers=HEADERS).json()
    assert changed['text'] == 'usual' and changed['confirmed'] is None

def test_moment_translation_uses_saved_message_as_source(client, monkeypatch):
    async def translate(source, answers, output_language, brief):
        assert source == 'Could I have Lipton green tea, please?'
        assert output_language == 'Hindi/Hinglish'
        return {'text': 'Mujhe Lipton green tea chahiye, please.', 'question': '', 'options': []}, {'model': providers.TEXT_MODEL, 'elapsed_ms': 5}
    monkeypatch.setattr(providers, 'draft', translate)
    j = client.post('/api/messages', json={'text': 'usual'}, headers=HEADERS).json()
    client.post(f"/api/messages/{j['id']}/draft", json={'revision': 1, 'output_language': 'Hindi/Hinglish', 'context': selection('cafe', moment_id='tea-stop').model_dump()}, headers=HEADERS)
    j = client.get('/api/session').json()['job']
    assert j['text'] == 'Mujhe Lipton green tea chahiye, please.'
    assert module.calls_used == 1

def test_custom_life_profile_roundtrip_and_legacy_defaults(client):
    saved = client.post('/api/profiles', json={'label': 'My own life', 'about': 'I enjoy painting.', 'manner': 'warm', 'moments': [{'id': 'painting', 'title': 'At my easel', 'scenario': 'home', 'cue': 'brush', 'message': 'Please pass me my wide paintbrush.'}]}, headers=HEADERS).json()
    profile = context.get_profile(saved['id'])
    assert profile.about == 'I enjoy painting.' and profile.moments[0].cue == 'brush'
    old = context.Profile.model_validate({'label': 'Existing profile', 'rules': []})
    assert old.moments == [] and old.manner == 'neutral'
