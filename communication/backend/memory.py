"""Explicitly remembered wording and rebuildable vectors in the profile database.

Nothing subscribes to speech confirmation: ``remember`` is called only by the
dedicated user action. A remembered message is approved wording, never a verified
literal transcript or an acoustic training label. All IDs are SQL parameters.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import time
from typing import Callable

from fastapi import HTTPException

MAX_MEMORIES = 1000
_invalidator: Callable[[str, set[str], bool], None] | None = None


def set_invalidator(callback):
    """Install a synchronous lifecycle callback, called only after commits.

    Arguments are (profile_id, changed_source_ids, profile_deleted). Clear calls
    it even for an empty store, allowing the host to forget session references.
    """
    global _invalidator
    _invalidator = callback


def invalidate(profile_id, source_ids, profile_deleted=False):
    if _invalidator:
        _invalidator(profile_id, set(source_ids), profile_deleted)


def ensure_schema(conn):
    """Additive schema; callers enable foreign keys before starting a transaction."""
    conn.execute('CREATE TABLE IF NOT EXISTS memory_state (profile_id TEXT PRIMARY KEY REFERENCES profiles(id) ON DELETE CASCADE, revision INTEGER NOT NULL DEFAULT 0)')
    conn.execute('''CREATE TABLE IF NOT EXISTS remembered_messages (
        id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
        job_id TEXT NOT NULL, message_revision INTEGER NOT NULL, body TEXT NOT NULL,
        UNIQUE(profile_id, job_id, message_revision))''')
    # Deleting a memory also revokes its displayed-message authorization. Keep
    # only opaque identifiers, never deleted wording, so a delayed Remember
    # cannot recreate it after fetching a newer memory revision.
    conn.execute('''CREATE TABLE IF NOT EXISTS deleted_memory_revisions (
        profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
        job_id TEXT NOT NULL, message_revision INTEGER NOT NULL,
        PRIMARY KEY(profile_id, job_id, message_revision))''')
    conn.execute('''CREATE TABLE IF NOT EXISTS personal_sources (
        profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
        source_id TEXT NOT NULL, source_revision INTEGER NOT NULL, content_hash TEXT NOT NULL,
        kind TEXT NOT NULL, body TEXT NOT NULL, PRIMARY KEY(profile_id, source_id))''')
    conn.execute('''CREATE TABLE IF NOT EXISTS personal_embeddings (
        profile_id TEXT NOT NULL, source_id TEXT NOT NULL, content_hash TEXT NOT NULL,
        encoder_revision TEXT NOT NULL, dimensions INTEGER NOT NULL, chunk_index INTEGER NOT NULL,
        vector BLOB NOT NULL, PRIMARY KEY(profile_id,source_id,encoder_revision,chunk_index),
        FOREIGN KEY(profile_id,source_id) REFERENCES personal_sources(profile_id,source_id) ON DELETE CASCADE)''')


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _short(value):
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def _require_profile(conn, profile_id, *, revision=None):
    row = conn.execute('SELECT revision,body FROM profiles WHERE id=?', (profile_id,)).fetchone()
    if not row:
        raise HTTPException(404, 'This personal profile is unavailable. Choose a personal profile before remembering a message.')
    if revision is not None and row[0] != revision:
        raise HTTPException(409, 'This profile changed. Reload its latest version before remembering this message.')
    conn.execute('INSERT OR IGNORE INTO memory_state(profile_id,revision) VALUES (?,0)', (profile_id,))
    return json.loads(row[1])


def _revision(conn, profile_id):
    row = conn.execute('SELECT revision FROM memory_state WHERE profile_id=?', (profile_id,)).fetchone()
    return row[0] if row else 0


def _expect_revision(conn, profile_id, expected):
    actual = _revision(conn, profile_id)
    if actual != expected:
        raise HTTPException(409, 'Remembered messages changed. Review the latest list before trying again.')
    return actual


def _advance(conn, profile_id):
    conn.execute('UPDATE memory_state SET revision=revision+1 WHERE profile_id=?', (profile_id,))
    return _revision(conn, profile_id)


def profile_entries(profile):
    """Pure derivation for runtime indexing and frozen evaluation fixtures.

    Unknown language is deliberately not inferred from Latin characters. The
    existing exact lexicon/rule paths continue to work for every profile language.
    """
    language = 'en' if profile.get('language') == 'English' else 'hi' if profile.get('language') == 'Hindi/Hinglish' else 'und'
    entries = []

    def add(source_id, kind, text, wording='', *, anchor='', plain='', scope=None, detail_id=''):
        entry = dict(source_id=source_id, kind=kind, language=language, text=text,
                     wording=wording, anchor=anchor, plain=plain, scope=scope or {},
                     detail_id=detail_id, created_at=0, approval='profile_edit')
        entry['content_hash'] = _hash(entry)
        entries.append(entry)

    for item in profile.get('lexicon', []):
        add('lexicon/' + item['id'], 'lexicon', ' | '.join(dict.fromkeys([item['word'], item.get('display', item['word']), *item.get('aliases', [])])),
            scope={'settings': item.get('settings', [])})
    for item in profile.get('specializations', []):
        add('specialization/' + item['id'], 'wording', item['surface'], item['surface'],
            anchor=item['anchor'], plain=item['plain'], scope={'settings': item.get('settings', [])}, detail_id=item['id'])
    for item in profile.get('rules', []):
        add('rule/' + _short(item['anchor'] + '\0' + '|'.join(sorted(item['scenarios']))), 'wording', item['wording'], item['wording'],
            anchor=item['anchor'], plain=item['anchor'], scope={'scenarios': item['scenarios'], 'mode': item.get('mode', 'use')})
    for item in profile.get('moments', []):
        add('moment/' + item['id'], 'wording', item['message'], item['message'], anchor=item['cue'], plain=item['cue'],
            scope={'scenarios': [item['scenario']], 'moment_id': item['id'], 'exact_cue': True})
    for item in profile.get('people', []):
        add('person/' + _short(item['name']), 'lexicon', item['name'])
    for term in profile.get('protected_terms', []):
        add('protected/' + _short(term), 'lexicon', term)
    return entries


def _upsert_source(conn, profile_id, entry):
    old = conn.execute('SELECT source_revision,content_hash FROM personal_sources WHERE profile_id=? AND source_id=?',
                       (profile_id, entry['source_id'])).fetchone()
    if old and old[1] == entry['content_hash']:
        return False
    revision = old[0] + 1 if old else 1
    body = {**entry, 'source_revision': revision}
    conn.execute('DELETE FROM personal_embeddings WHERE profile_id=? AND source_id=?', (profile_id, entry['source_id']))
    conn.execute('''INSERT INTO personal_sources VALUES (?,?,?,?,?,?)
        ON CONFLICT(profile_id,source_id) DO UPDATE SET source_revision=excluded.source_revision,
        content_hash=excluded.content_hash,kind=excluded.kind,body=excluded.body''',
        (profile_id, entry['source_id'], revision, entry['content_hash'], entry['kind'], json.dumps(body, ensure_ascii=False)))
    return True


def sync_profile_sources(conn, profile):
    """Synchronize explicit profile fields in the caller's profile-save transaction."""
    profile_id = profile['id']
    conn.execute('INSERT OR IGNORE INTO memory_state(profile_id,revision) VALUES (?,0)', (profile_id,))
    desired = profile_entries(profile)
    expected_ids = {entry['source_id'] for entry in desired}
    old_ids = {row[0] for row in conn.execute("SELECT source_id FROM personal_sources WHERE profile_id=? AND kind!='remembered'", (profile_id,))}
    changed = old_ids - expected_ids
    for source_id in changed:
        conn.execute('DELETE FROM personal_sources WHERE profile_id=? AND source_id=?', (profile_id, source_id))
    for entry in desired:
        if _upsert_source(conn, profile_id, entry):
            changed.add(entry['source_id'])
    return changed


def list_memories(profile_id):
    from . import context
    if any(profile.id == profile_id for profile in context.SAMPLES):
        return {'profile_id': profile_id, 'memory_revision': 0, 'memories': []}
    with context.db() as conn:
        _require_profile(conn, profile_id)
        records = [json.loads(row[0]) for row in conn.execute('SELECT body FROM remembered_messages WHERE profile_id=? ORDER BY rowid DESC', (profile_id,))]
        return {'profile_id': profile_id, 'memory_revision': _revision(conn, profile_id), 'memories': records}


def _message_language(job, profile):
    output = job.get('output_language', 'original')
    if output == 'English':
        return 'en'
    if output == 'Hindi/Hinglish' or profile.get('language') == 'Hindi/Hinglish':
        return 'hi'
    evidence = job.get('evidence') or {}
    if output == 'original' and (job.get('language') in {'en', 'English'} or evidence.get('recognizer') == 'adapted'):
        return 'en'
    return 'und'


def remember(job, expected_memory_revision):
    """Store exactly one explicit approval; no speech or conversation mutation."""
    from . import context
    snap = job.get('context') or {}
    profile = snap.get('profile') or {}
    if not profile.get('id') or profile.get('sample'):
        raise HTTPException(422, 'Choose a personal profile before remembering a message.')
    if job.get('retrieval') and not dependencies_valid(job['retrieval']):
        raise HTTPException(409, 'The personal context for this message changed. Review the latest message before remembering it.')
    unresolved = len(job.get('candidates', [])) > 1 and not job.get('selected_candidate_id')
    if job.get('question') or unresolved or job.get('status') not in {'review', 'confirmed'} or not job.get('text', '').strip():
        raise HTTPException(409, 'Finish choosing or editing this message before remembering it.')
    selection = snap.get('selection') or {}
    audience = snap.get('resolved_audience') or {}
    selected = next((item for item in job.get('candidates', []) if item['id'] == job.get('selected_candidate_id')), None)
    literals = list((selected or {}).get('source_literals') or [])
    unique_literals = list(dict.fromkeys(literals))
    selected_literal = unique_literals[0] if len(unique_literals) == 1 else ''
    scope = dict(scenario=selection.get('scenario', 'general'), core_context=snap.get('core_context', 'general'),
                 audience_id=selection.get('audience_id', ''), recipient=selection.get('recipient', ''),
                 listener=audience.get('listener', ''), place_id=selection.get('place_id', ''))
    record = {'id': secrets.token_hex(12), 'profile_id': profile['id'], 'profile_revision': profile['revision'],
              'message': job['text'], 'language': _message_language(job, profile), 'created_at': time.time(), 'scope': scope,
              'source': {'job_id': job['id'], 'message_revision': job['revision'], 'selected_literal': selected_literal,
                         'source_hypothesis_ids': list((selected or {}).get('source_hypothesis_ids', [])),
                         'source_literals': literals, 'literal_verified': False,
                         'recognizer': (job.get('evidence') or {}).get('recognizer', ''),
                         'model': (job.get('evidence') or {}).get('model', '')}}
    with context.db() as conn:
        conn.execute('BEGIN IMMEDIATE')
        _require_profile(conn, profile['id'], revision=profile['revision'])
        if job.get('retrieval') and not _dependencies_valid(conn, job['retrieval']):
            raise HTTPException(409, 'The personal context for this message changed. Review the latest message before remembering it.')
        if conn.execute('SELECT 1 FROM deleted_memory_revisions WHERE profile_id=? AND job_id=? AND message_revision=?',
                        (profile['id'], job['id'], job['revision'])).fetchone():
            raise HTTPException(409, 'This displayed message was removed from memory. Edit it or create a new message before remembering it again.')
        old = conn.execute('SELECT body FROM remembered_messages WHERE profile_id=? AND job_id=? AND message_revision=?',
                           (profile['id'], job['id'], job['revision'])).fetchone()
        if old:
            record = json.loads(old[0])
            if record['message'] != job['text']:
                raise HTTPException(409, 'This message revision already has different remembered wording.')
            return dict(memory_id=record['id'], memory_revision=_revision(conn, profile['id']), message_revision=job['revision'], created=False, memory=record)
        _expect_revision(conn, profile['id'], expected_memory_revision)
        if conn.execute('SELECT count(*) FROM remembered_messages WHERE profile_id=?', (profile['id'],)).fetchone()[0] >= MAX_MEMORIES:
            raise HTTPException(422, 'This profile has reached its remembered-message limit. Delete some messages before adding more.')
        conn.execute('INSERT INTO remembered_messages VALUES (?,?,?,?,?)', (record['id'], profile['id'], job['id'], job['revision'], json.dumps(record, ensure_ascii=False)))
        entry = dict(source_id='memory/' + record['id'], kind='remembered', language=record['language'], text=record['message'],
                     wording=record['message'], anchor='', plain='', scope=scope, created_at=record['created_at'], approval='remember_click')
        entry['content_hash'] = _hash(entry)
        _upsert_source(conn, profile['id'], entry)
        revision = _advance(conn, profile['id'])
    return dict(memory_id=record['id'], memory_revision=revision, message_revision=job['revision'], created=True, memory=record)


def delete_memories(profile_id, expected_revision, memory_id=None):
    from . import context
    with context.db() as conn:
        conn.execute('BEGIN IMMEDIATE')
        _require_profile(conn, profile_id)
        _expect_revision(conn, profile_id, expected_revision)
        if memory_id is None:
            ids = [row[0] for row in conn.execute('SELECT id FROM remembered_messages WHERE profile_id=?', (profile_id,))]
        else:
            ids = [row[0] for row in conn.execute('SELECT id FROM remembered_messages WHERE profile_id=? AND id=?', (profile_id, memory_id))]
            if not ids:
                raise HTTPException(404, 'This remembered message is unavailable.')
        for entry_id in ids:
            conn.execute('''INSERT OR IGNORE INTO deleted_memory_revisions
                SELECT profile_id,job_id,message_revision FROM remembered_messages WHERE profile_id=? AND id=?''',
                (profile_id, entry_id))
            conn.execute('DELETE FROM personal_sources WHERE profile_id=? AND source_id=?', (profile_id, 'memory/' + entry_id))
            conn.execute('DELETE FROM remembered_messages WHERE profile_id=? AND id=?', (profile_id, entry_id))
        revision = _advance(conn, profile_id)  # even an empty Clear defeats a pending stale save
    invalidate(profile_id, {'memory/' + entry_id for entry_id in ids})
    return {'profile_id': profile_id, 'memory_revision': revision, 'deleted_ids': ids}


def delete_profile(profile_id, profile_revision, memory_revision):
    from . import context
    if any(profile.id == profile_id for profile in context.SAMPLES):
        raise HTTPException(422, 'Sample profiles cannot be deleted.')
    with context.db() as conn:
        conn.execute('BEGIN IMMEDIATE')
        _require_profile(conn, profile_id, revision=profile_revision)
        _expect_revision(conn, profile_id, memory_revision)
        ids = {row[0] for row in conn.execute('SELECT source_id FROM personal_sources WHERE profile_id=?', (profile_id,))}
        conn.execute('DELETE FROM profiles WHERE id=?', (profile_id,))
    invalidate(profile_id, ids, True)
    return {'deleted': True, 'profile_id': profile_id}


def source_snapshot(snapshot):
    """Read eligible source versions from the live profile, never from another ID."""
    from . import context
    profile = (snapshot or {}).get('profile') or {}
    if not profile or profile.get('sample'):
        return {'profile_id': profile.get('id', ''), 'profile_revision': profile.get('revision', 0), 'memory_revision': 0, 'entries': []}
    with context.db() as conn:
        conn.execute('BEGIN IMMEDIATE')
        current = _require_profile(conn, profile['id'], revision=profile['revision'])
        sync_profile_sources(conn, current)  # imported databases also acquire a rebuildable index
        entries = [json.loads(row[0]) for row in conn.execute('SELECT body FROM personal_sources WHERE profile_id=? ORDER BY source_id', (profile['id'],))]
        return {'profile_id': profile['id'], 'profile_revision': profile['revision'], 'memory_revision': _revision(conn, profile['id']), 'entries': entries}


def dependencies_valid(result):
    """Fail closed if a consumed source changed/deleted while work was running."""
    from . import context
    profile_id = (result or {}).get('profile_id')
    if not profile_id:
        return not (result or {}).get('dependencies')
    if any(profile.id == profile_id and profile.revision == result.get('profile_revision') for profile in context.SAMPLES):
        return not result.get('dependencies')
    with context.db() as conn:
        return _dependencies_valid(conn, result)


def _dependencies_valid(conn, result):
    profile_id = result.get('profile_id')
    if not profile_id:
        return not result.get('dependencies')
    profile = conn.execute('SELECT revision FROM profiles WHERE id=?', (profile_id,)).fetchone()
    if not profile or (result.get('profile_revision') is not None and profile[0] != result['profile_revision']):
        return False
    for item in result.get('dependencies', []):
        row = conn.execute('SELECT source_revision,content_hash FROM personal_sources WHERE profile_id=? AND source_id=?', (profile_id, item['source_id'])).fetchone()
        if not row or row != (item['source_revision'], item['content_hash']):
            return False
    return True


def read_vectors(profile_id, source_id, content_hash, encoder_revision, dimensions):
    from . import context
    with context.db() as conn:
        rows = conn.execute('SELECT vector FROM personal_embeddings WHERE profile_id=? AND source_id=? AND content_hash=? AND encoder_revision=? AND dimensions=? ORDER BY chunk_index',
                            (profile_id, source_id, content_hash, encoder_revision, dimensions)).fetchall()
    return [row[0] for row in rows]


def save_vectors(profile_id, entry, encoder_revision, vectors):
    """Compare-and-store: a stale embedding task can never resurrect deleted text."""
    from . import context
    with context.db() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT content_hash FROM personal_sources WHERE profile_id=? AND source_id=?', (profile_id, entry['source_id'])).fetchone()
        if not row or row[0] != entry['content_hash']:
            return False
        conn.execute('DELETE FROM personal_embeddings WHERE profile_id=? AND source_id=? AND encoder_revision=?', (profile_id, entry['source_id'], encoder_revision))
        for i, vector in enumerate(vectors):
            conn.execute('INSERT INTO personal_embeddings VALUES (?,?,?,?,?,?,?)',
                         (profile_id, entry['source_id'], entry['content_hash'], encoder_revision, len(vector), i, vector.astype('<f4').tobytes()))
    return True
