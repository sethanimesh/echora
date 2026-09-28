"""Explicit storage, isolated retrieval and deletion-race contract checks."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from communication.backend import context, memory, memory_routes, retrieval
from communication.backend.profile_adapter import resolve_snapshot


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv('ECHORA_PROFILE_DB', str(tmp_path / 'profiles.sqlite3'))
    old = memory._invalidator
    memory.set_invalidator(None)
    yield
    memory.set_invalidator(old)


def profile(label='Speaker', **fields):
    return context.save_profile(context.Profile(label=label, language='English', **fields))


def snapshot(document, **fields):
    selection = context.Selection(profile_id=document['id'], profile_revision=document['revision'], scenario='home', **fields)
    return resolve_snapshot(selection.model_dump(), document)


def job(document, **values):
    return {'id': 'job-1', 'revision': 2, 'status': 'review', 'text': 'Please bring water.',
            'context': snapshot(document), 'language': 'en', 'output_language': 'original',
            'question': '', 'confirmed': None, 'auto_speak_revision': 2,
            'candidates': [{'id': 'm1', 'source_literals': ['water'], 'source_hypothesis_ids': ['h1']}],
            'selected_candidate_id': 'm1', 'evidence': {'recognizer': 'adapted', 'model': 'test-model'}, **values}


class Encoder:
    dimension = 4
    model_revision = 'deterministic-test-v1'

    def encode_chunks(self, text):
        values = np.zeros(4, dtype=np.float32)
        for i, word in enumerate(('water', 'tea', 'sugar', 'meena')):
            values[i] = word in text.casefold()
        return values.reshape(1, -1)


def beams(*texts):
    return [{'id': 'h' + str(i), 'literal_text': text} for i, text in enumerate(texts)]


def test_remember_is_explicit_exact_and_idempotent_without_speech_changes():
    p = profile()
    j = job(p)
    assert memory.list_memories(p['id'])['memories'] == []
    before = copy.deepcopy(j)
    saved = memory.remember(j, 0)
    again = memory.remember(j, 0)
    assert saved['created'] and not again['created']
    assert saved['memory_id'] == again['memory_id']
    assert j == before
    record = memory.list_memories(p['id'])['memories'][0]
    assert record['message'] == j['text']
    assert record['source']['selected_literal'] == 'water'
    assert record['source']['source_literals'] == ['water']
    assert record['source']['literal_verified'] is False
    assert 'audio' not in record and 'speech_text' not in record
    edited = job(p, revision=3, text='Please bring tea.')
    assert memory.remember(edited, 1)['memory_id'] != saved['memory_id']


@pytest.mark.parametrize('changes', [
    {'status': 'cancelled'}, {'status': 'drafting'}, {'question': 'Tea or water?'},
    {'selected_candidate_id': None, 'candidates': [{'id': 'a'}, {'id': 'b'}]},
])
def test_unresolved_or_unavailable_jobs_cannot_be_remembered(changes):
    p = profile()
    with pytest.raises(HTTPException) as exc:
        memory.remember(job(p, **changes), 0)
    assert exc.value.status_code == 409
    assert not memory.list_memories(p['id'])['memories']


def test_profile_edit_preserves_memories_and_rejects_stale_save():
    p = profile()
    memory.remember(job(p), 0)
    context.save_profile(context.Profile.model_validate({**p, 'label': 'Edited'}))
    assert len(memory.list_memories(p['id'])['memories']) == 1
    with pytest.raises(HTTPException) as exc:
        memory.remember(job(p, revision=3), 1)
    assert exc.value.status_code == 409


def test_delete_clear_and_profile_delete_are_revision_bound_and_isolated():
    first, second = profile('First'), profile('Second')
    a = memory.remember(job(first), 0)
    b = memory.remember(job(second), 0)
    with pytest.raises(HTTPException) as exc:
        memory.delete_memories(second['id'], 1, a['memory_id'])
    assert exc.value.status_code == 404
    memory.delete_memories(first['id'], 1, a['memory_id'])
    with pytest.raises(HTTPException) as exc:
        memory.remember(job(first), 0)  # an in-flight retry cannot resurrect deletion
    assert exc.value.status_code == 409
    assert memory.list_memories(second['id'])['memories'][0]['id'] == b['memory_id']
    clear = memory.delete_memories(first['id'], 2)
    assert clear['memory_revision'] == 3
    memory.delete_profile(second['id'], second['revision'], 1)
    with context.db() as conn:
        for table in ('remembered_messages', 'personal_sources', 'personal_embeddings', 'memory_state'):
            assert conn.execute('SELECT count(*) FROM ' + table + ' WHERE profile_id=?', (second['id'],)).fetchone()[0] == 0


def test_retrieval_uses_full_candidates_filters_scope_and_limits_hits():
    entries = [{'source_id': f'e{i}', 'kind': 'wording', 'text': 'water', 'wording': 'water',
                'language': 'en', 'scope': {'scenarios': ['home'], 'recipient': 'Meena'}, 'created_at': 0}
               for i in range(8)]
    entries += [{'source_id': 'wrong-setting', 'kind': 'wording', 'text': 'tea', 'wording': 'tea',
                 'language': 'en', 'scope': {'scenarios': ['care']}, 'created_at': 0}]
    ranked = retrieval.rank_entries(beams('water', 'tea'), entries, selection={'scenario': 'home', 'recipient': 'Meena'}, embedder=Encoder())
    assert len(ranked['candidates']['h0']['hits']) == 1
    assert not ranked['candidates']['h1']['eligible']
    assert ranked['candidates']['h0']['relevance'] <= 1
    other = retrieval.rank_entries(beams('water'), entries, selection={'scenario': 'home', 'recipient': 'Someone else'}, embedder=Encoder())
    assert not other['candidates']['h0']['hits']


def test_retrieval_is_per_profile_and_deletion_invalidates_dependencies():
    first, second = profile('First'), profile('Second')
    remembered = memory.remember(job(first), 0)
    result = retrieval.retrieve_candidates(snapshot(first), beams('water'), embedder=Encoder())
    assert result['status'] == 'ready' and result['candidates']['h0']['eligible']
    assert memory.dependencies_valid(result)
    assert retrieval.retrieve_candidates(snapshot(second), beams('water'), embedder=Encoder())['status'] == 'empty'
    source = memory.source_snapshot(snapshot(first))['entries'][0]
    memory.delete_memories(first['id'], 1, remembered['memory_id'])
    assert not memory.dependencies_valid(result)
    assert not memory.save_vectors(first['id'], source, Encoder.model_revision, Encoder().encode_chunks('water'))
    with context.db() as conn:
        assert conn.execute('SELECT count(*) FROM personal_embeddings').fetchone()[0] == 0


def test_profile_source_edit_invalidates_vectors_and_calls_lifecycle_callback():
    p = profile(rules=[context.Rule(anchor='tea', wording='green tea', scenarios=['home'])])
    result = retrieval.retrieve_candidates(snapshot(p), beams('tea'), embedder=Encoder())
    calls = []
    memory.set_invalidator(lambda *args: calls.append(args))
    changed = context.Profile.model_validate(p)
    changed.rules[0].wording = 'black tea'
    context.save_profile(changed)
    assert calls and calls[0][0] == p['id']
    assert not memory.dependencies_valid(result)
    with context.db() as conn:
        assert conn.execute('SELECT count(*) FROM personal_embeddings').fetchone()[0] == 0


def test_hinglish_and_unknown_language_are_not_semantically_guessed():
    p = profile()
    memory.remember(job(p, output_language='Hindi/Hinglish', text='Mujhe paani chahiye.'), 0)
    result = retrieval.retrieve_candidates(snapshot(p), beams('paani'), language='hi', embedder=Encoder())
    assert result['status'] == 'unsupported_language'
    assert memory.list_memories(p['id'])['memories'][0]['language'] == 'hi'
    original = {**p, 'language': 'original', 'rules': [{'anchor': 'tea', 'wording': 'green tea', 'scenarios': ['home'], 'mode': 'use'}]}
    assert memory.profile_entries(original)[0]['language'] == 'und'


def test_router_checks_job_revision_and_has_identical_native_contract():
    p = profile()
    state = SimpleNamespace(job=job(p))
    app = FastAPI()

    def current(s, job_id, revision=None):
        if s.job['id'] != job_id or s.job['revision'] != revision:
            raise HTTPException(409, 'Changed')
        return s.job

    for prefix in ('/api', '/api/v1/communication'):
        app.include_router(memory_routes.router(lambda _: state, current, prefix=prefix))
    client = TestClient(app)
    path = '/api/messages/job-1/remember'
    assert client.post(path, json={'revision': 1, 'memory_revision': 0}).status_code == 409
    saved = client.post(path, json={'revision': 2, 'memory_revision': 0}).json()
    native = client.get(f"/api/v1/communication/profiles/{p['id']}/memories").json()
    assert native['memory_revision'] == 1 and native['memories'][0]['id'] == saved['memory_id']
    assert state.job['confirmed'] is None
    deleted = client.post(f"/api/v1/communication/profiles/{p['id']}/memories/clear", json={'memory_revision': 1})
    assert deleted.status_code == 200


def test_recency_and_context_are_bounded_and_duplicates_do_not_accumulate():
    item = {'source_id': 'm', 'kind': 'remembered', 'language': 'en', 'text': 'water', 'wording': 'water', 'scope': {}, 'created_at': 1}
    old = retrieval.rank_entries(beams('water'), [item], embedder=Encoder(), now=1 + 300 * 86400)['candidates']['h0']
    month = retrieval.rank_entries(beams('water'), [item], embedder=Encoder(), now=1 + 30 * 86400)['candidates']['h0']
    recent = retrieval.rank_entries(beams('water'), [item] * 7, embedder=Encoder(), now=1, reference={'text': 'water'})['candidates']['h0']['relevance']
    assert old['relevance'] == 0 and not old['eligible']
    assert month['reliability'] == pytest.approx(.5)
    assert recent == pytest.approx(1.0)


@pytest.mark.parametrize('literal,stored', [
    ('water', 'I do not want water'), ('two teas', 'three teas'),
    ('water for Maya', 'water for Meena'), ('cold water', 'hot water'),
])
def test_contradictory_context_has_no_ranking_or_expansion_influence(literal, stored):
    entry = {'source_id': 'm', 'kind': 'remembered', 'language': 'en', 'text': stored,
             'wording': stored, 'scope': {}, 'created_at': 1}
    result = retrieval.rank_entries(beams(literal), [entry], now=1, embedder=Encoder(),
                                    selection={'protected_terms': ['Maya', 'Meena']})
    assert result['candidates']['h0'] == {'relevance': 0, 'reliability': 0, 'eligible': False, 'hits': []}


def test_retrieval_returns_at_most_five_distinct_sources():
    entries = [{'source_id': f'e{i}', 'kind': 'wording', 'text': f'water item {i}',
                'language': 'en', 'scope': {}, 'created_at': 0} for i in range(8)]
    # Distinct descriptive wording with no contradictory quantities.
    for item, suffix in zip(entries, ['please', 'now', 'glass', 'drink', 'jug', 'bottle', 'cup', 'some']):
        item['text'] = 'water ' + suffix
    result = retrieval.rank_entries(beams('water'), entries, embedder=Encoder())
    assert len(result['candidates']['h0']['hits']) == 5


def test_internally_conflicting_memories_do_not_vote_for_either_interpretation():
    entries = [{'source_id': f'm{i}', 'kind': 'remembered', 'language': 'en',
                'text': text, 'scope': {}, 'created_at': 1}
               for i, text in enumerate(['I want water', 'I do not want water'])]
    result = retrieval.rank_entries(beams('I want water', 'I do not want water'), entries,
                                    now=1, embedder=Encoder())
    assert result['status'] == 'empty'
    assert all(not candidate['eligible'] and not candidate['hits'] for candidate in result['candidates'].values())


def test_source_eligibility_and_provenance_survive_retrieval():
    from app.personal.profile import AudienceProfile, SpecializationRule
    p = profile(specializations=[SpecializationRule(id='usual-tea', anchor='tea', plain='tea', surface='green tea', kind='food', settings=['home'])],
                audiences=[AudienceProfile(id='carer', label='Meena', listener='familiar', known_detail_ids=['usual-tea'])])
    eligible = retrieval.retrieve_candidates(snapshot(p), beams('tea'), embedder=Encoder())
    hit = eligible['candidates']['h0']['hits'][0]
    assert hit['source_id'] == 'specialization/usual-tea'
    assert hit['wording'] == 'green tea' and hit['anchor'] == hit['plain'] == 'tea'
    assert hit['scope'] == {'settings': ['home']} and hit['approval'] == 'profile_edit'
    assert hit['source_revision'] == 1 and len(hit['content_hash']) == 64
    known = retrieval.retrieve_candidates(snapshot(p, audience_id='carer'), beams('tea'), embedder=Encoder())
    assert known['status'] == 'empty'
    outside = resolve_snapshot(context.Selection(profile_id=p['id'], profile_revision=1, scenario='outside').model_dump(), p)
    assert retrieval.retrieve_candidates(outside, beams('tea'), embedder=Encoder())['status'] == 'empty'


@pytest.mark.parametrize('clear_all', [False, True])
def test_deleted_displayed_revision_cannot_be_restored_with_fresh_memory_revision(clear_all):
    p = profile()
    displayed = job(p)
    saved = memory.remember(displayed, 0)
    memory.delete_memories(p['id'], 1, None if clear_all else saved['memory_id'])
    # A pending Remember can fetch this *new* revision after deletion has
    # committed. The displayed message authorization is still the old one.
    fresh = memory.list_memories(p['id'])['memory_revision']
    with pytest.raises(HTTPException) as exc:
        memory.remember(copy.deepcopy(displayed), fresh)
    assert exc.value.status_code == 409
    assert memory.list_memories(p['id'])['memories'] == []
    # An explicit edit creates a new displayed revision, even if the person
    # deliberately keeps the same words. That new approval may be remembered.
    reviewed = {**displayed, 'revision': displayed['revision'] + 1}
    repeated = memory.remember(reviewed, fresh)
    assert repeated['created'] and repeated['memory_id'] != saved['memory_id']
    assert not memory.remember(reviewed, fresh)['created']


def test_deleted_revision_tombstones_contain_no_wording_and_are_profile_scoped():
    first, second = profile('First'), profile('Second')
    old = memory.remember(job(first), 0)
    memory.delete_memories(first['id'], 1, old['memory_id'])
    # Another speaker can approve the same session/revision identifiers.
    assert memory.remember(job(second), 0)['created']
    with context.db() as conn:
        columns = [row[1] for row in conn.execute('PRAGMA table_info(deleted_memory_revisions)')]
        assert columns == ['profile_id', 'job_id', 'message_revision']
        assert conn.execute('SELECT * FROM deleted_memory_revisions').fetchall() == [(first['id'], 'job-1', 2)]
        assert conn.execute('SELECT count(*) FROM personal_sources WHERE profile_id=?', (first['id'],)).fetchone()[0] == 0
    memory.delete_profile(first['id'], first['revision'], 2)
    with context.db() as conn:
        assert conn.execute('SELECT count(*) FROM deleted_memory_revisions').fetchone()[0] == 0


@pytest.mark.parametrize('prefix', ['/api', '/api/v1/communication'])
def test_shared_memory_routes_reject_stale_save_even_after_refreshing_list(prefix):
    p = profile()
    displayed = job(p)
    original = copy.deepcopy(displayed)
    state = SimpleNamespace(job=displayed)
    app = FastAPI()

    def current(s, job_id, revision=None):
        if s.job['id'] != job_id or s.job['revision'] != revision:
            raise HTTPException(409, 'Changed')
        return s.job

    app.include_router(memory_routes.router(lambda _: state, current, prefix=prefix))
    client = TestClient(app)
    target = prefix + '/messages/job-1/remember'
    saved = client.post(target, json={'revision': displayed['revision'], 'memory_revision': 0}).json()
    deleted = client.post(f"{prefix}/profiles/{p['id']}/memories/{saved['memory_id']}/delete", json={'memory_revision': 1})
    assert deleted.status_code == 200
    refreshed = client.get(f"{prefix}/profiles/{p['id']}/memories").json()
    stale = client.post(target, json={'revision': displayed['revision'], 'memory_revision': refreshed['memory_revision']})
    assert stale.status_code == 409
    assert client.get(f"{prefix}/profiles/{p['id']}/memories").json()['memories'] == []
    assert state.job == original
