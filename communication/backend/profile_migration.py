"""Non-destructive, idempotent import into the unified versioned profile store.

Run ``python -m communication.backend.profile_migration --help``. Dry run is
the default. Reports contain counts and IDs, never profile text or preferences.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone

from .context import Profile


def _identity(kind, source_id, namespace='default'):
    key = f'{kind}:{namespace}:{source_id}'
    prefix = {'a-demo': 'a-demo', 'a-profile': 'a-user', 'b-profile': 'b-user'}[kind]
    return key, prefix + '-' + hashlib.sha256(key.encode()).hexdigest()[:24]


def adapt_a(raw, *, kind='a-profile', namespace='default'):
    """Retain exact A lexicon/detail/audience IDs and every supported field."""
    from app.personal.profile import PersonaProfile
    original = PersonaProfile.model_validate(raw)
    return Profile(
        label=original.label, sample=False, about=original.blurb,
        style='concise' if original.style.brevity == 'short' else 'natural',
        communication_style=original.style, context_default=original.context_default,
        listener_by_setting=original.listener_by_setting, style_by_setting=original.style_by_setting,
        audiences=original.audiences, lexicon=original.lexicon, specializations=original.specializations,
        speaker_note=original.speaker_note, icon=original.icon,
        provenance={'source_project': 'echora', 'source_kind': kind, 'source_id': original.id,
                    'source_revision': 0, 'source_namespace': namespace,
                    'source_created_at': original.created_at, 'source_baseline': original.baseline},
    )


def adapt_b(raw, *, namespace='default'):
    original = Profile.model_validate(raw)
    return original.model_copy(update={'id': '', 'sample': False,
        'provenance': {'source_project': 'echora2', 'source_kind': 'b-profile',
                       'source_id': original.id, 'source_revision': original.revision,
                       'source_namespace': namespace,
                       'prior_provenance': original.provenance}})


def candidates(*, a_personas=None, a_profiles=(), b_database=None, namespace='default'):
    """Read only explicitly provided sources; never inspect personal directories."""
    result = []
    if a_personas:
        for line in Path(a_personas).read_text(encoding='utf-8').splitlines():
            if line.strip():
                raw = json.loads(line)
                result.append((adapt_a(raw, kind='a-demo', namespace=namespace), raw))
    for path in a_profiles:
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
        result.append((adapt_a(raw, namespace=namespace), raw))
    if b_database:
        uri = Path(b_database).resolve().as_uri() + '?mode=ro'
        with sqlite3.connect(uri, uri=True) as conn:
            for (body,) in conn.execute('SELECT body FROM profiles ORDER BY id'):
                raw = json.loads(body)
                result.append((adapt_b(raw, namespace=namespace), raw))
    return result


def import_profiles(items, destination, *, apply=False):
    """Import absent IDs only; re-import never overwrites subsequent user edits.

    Duplicate source IDs with different contents are reported as conflicts, not
    silently collapsed. Existing target IDs must have the same source identity.
    """
    destination = Path(destination)
    report = {'dry_run': not apply, 'imported': [], 'skipped': [], 'conflicts': []}
    existing = {}
    if destination.is_file():
        with sqlite3.connect(destination.resolve().as_uri() + '?mode=ro', uri=True) as conn:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='profiles'").fetchone():
                existing = {row[0]: json.loads(row[1]) for row in conn.execute('SELECT id, body FROM profiles')}
    prepared, seen = [], {}
    for profile, raw in items:
        provenance = dict(profile.provenance)
        key, target_id = _identity(provenance['source_kind'], provenance['source_id'], provenance['source_namespace'])
        fingerprint = hashlib.sha256(json.dumps(raw, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if target_id in seen:
            if seen[target_id] != fingerprint:
                report['conflicts'].append(target_id)
            continue
        seen[target_id] = fingerprint
        if target_id in existing:
            if existing[target_id].get('provenance', {}).get('source_key') == key:
                report['skipped'].append(target_id)
            else:
                report['conflicts'].append(target_id)
            continue
        provenance.update(source_key=key, source_hash=fingerprint,
                          imported_at=datetime.now(timezone.utc).isoformat())
        prepared.append(profile.model_copy(update={'id': target_id, 'revision': 1, 'provenance': provenance}))
    conflict_ids = set(report['conflicts'])
    prepared = [profile for profile in prepared if profile.id not in conflict_ids]
    if apply and prepared:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(destination) as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute('CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, body TEXT NOT NULL)')
            for profile in prepared:
                # A concurrent explicit edit or import must win over this run.
                cursor = conn.execute('INSERT OR IGNORE INTO profiles VALUES (?,?,?)',
                                      (profile.id, profile.revision, profile.model_dump_json()))
                report['imported' if cursor.rowcount else 'skipped'].append(profile.id)
    else:
        report['imported'] = [profile.id for profile in prepared]
    report['counts'] = {name: len(report[name]) for name in ('imported', 'skipped', 'conflicts')}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True, help='Unified destination SQLite path')
    parser.add_argument('--a-personas', help='Explicit path to A demonstration personas JSONL')
    parser.add_argument('--a-profile', action='append', default=[], help='Explicit A JSON profile path; repeat for more')
    parser.add_argument('--b-db', help='Original B SQLite path, opened read-only')
    parser.add_argument('--namespace', default='default', help='Stable source-store namespace; use a different value for distinct stores')
    parser.add_argument('--apply', action='store_true', help='Write new imports; default only reports proposed IDs')
    args = parser.parse_args()
    try:
        sources = candidates(a_personas=args.a_personas, a_profiles=args.a_profile,
                             b_database=args.b_db, namespace=args.namespace)
        report = import_profiles(sources, args.database, apply=args.apply)
    except (ValueError, OSError, sqlite3.Error) as error:
        # Pydantic exceptions embed original field values. Never print private
        # profile text on a failed import, even in command-line diagnostics.
        print(json.dumps({'error': type(error).__name__, 'message': 'Profile import could not be completed. Check source format and destination access.'}))
        raise SystemExit(2) from None
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
