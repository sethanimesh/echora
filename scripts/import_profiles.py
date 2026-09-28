#!/usr/bin/env python3
"""Import existing local profiles without changing either source store."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'backend'), str(ROOT)]
from communication.backend.profile_migration import candidates, import_profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--communication-project', type=Path, help='Optional original Echora 2.0 directory')
    parser.add_argument('--apply', action='store_true', help='Insert missing profiles; otherwise dry-run')
    args = parser.parse_args()
    existing = sorted((ROOT / 'data/personal').glob('*/profile.json'))
    other_db = args.communication_project / 'communication/data/profiles.sqlite3' if args.communication_project else None
    try:
        items = candidates(a_personas=ROOT / 'data/personas/personas.jsonl', a_profiles=existing,
                           b_database=other_db if other_db and other_db.is_file() else None,
                           namespace='unified-2026-09')
        # A live profile is explicitly a copy of the same source ID, so prefer
        # its edits when seeding the new store. Shipped baselines stay on disk.
        live_ids = {p.provenance['source_id'] for p,_ in items if p.provenance['source_kind']=='a-profile'}
        items = [(p,raw) for p,raw in items if not (p.provenance['source_kind']=='a-demo' and p.provenance['source_id'] in live_ids)]
        report = import_profiles(items, ROOT / 'data/personal/profiles.sqlite3', apply=args.apply)
        print(json.dumps({'dry_run': report['dry_run'], **report['counts']}))
        return 1 if report['conflicts'] else 0
    except Exception:
        print('Profile import did not finish. The original stores were not changed.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
