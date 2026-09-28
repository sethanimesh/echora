"""Resumable production-composition evaluation and blinded correction-review pack.

All inputs are frozen synthetic fixtures. This runner constructs profile snapshots
in memory; it never opens the personal database or stores a remembered message.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'backend'), str(ROOT)]
from app.config import load_settings
from app.messaging.groq_chain import GroqMessageChain
from app.messaging.fidelity import preserves_bounded_facts
from app.schemas import Hypothesis
from communication.backend import context, recognition, providers
from communication.backend.profile_adapter import resolve_snapshot
from research.benchmarks.metrics import normalize, score

HERE = Path(__file__).resolve().parent
CODE_PATHS = [Path(__file__), ROOT / 'backend/app/config.py', ROOT / 'backend/app/schemas.py',
              *[ROOT / 'backend/app/messaging' / name for name in ('groq_chain.py', 'fidelity.py', 'alignment.py')],
              *[ROOT / 'backend/app/personal' / name for name in ('profile.py', 'lexicon.py')],
              *[ROOT / 'communication/backend' / name for name in
                ('recognition.py', 'context.py', 'profile_adapter.py', 'wording_fidelity.py')],
              ROOT / 'research/benchmarks/metrics.py']


def file_hashes():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in CODE_PATHS}


def occurs(text, phrase, *, exact=False):
    return bool(re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', text,
                          flags=0 if exact else re.IGNORECASE))


def evaluate(case, wording, *, personal=False):
    details = [case['detail']['surface']] if personal and case.get('detail') else []
    protected = [word for word in case.get('protected_exact', []) if not occurs(wording, word, exact=True)]
    anchors = [word for word in case.get('intent_anchors', []) if not occurs(wording, word)]
    forbidden = [word for word in case.get('forbidden', []) if occurs(wording, word)]
    faithful = preserves_bounded_facts(case['literal'], wording, approved_details=details)
    # Ordered anchors are an intent condition, not a bag-of-words match.
    if case['id'] == 'order':
        normalized = normalize(wording)
        faithful = faithful and 'water' in normalized and 'medicine' in normalized and normalized.index('water') < normalized.index('medicine')
    references = case.get('personal_references', case['references']) if personal else case['references']
    edits = min((score(reference, wording).word_errors for reference in references), default=None) if wording else None
    return {'meaning_violation': bool(wording) and (not faithful or bool(protected or anchors or forbidden)),
            'wording_withheld': not bool(wording),
            'missing_protected': protected, 'missing_intent_anchors': anchors, 'forbidden_terms': forbidden,
            'minimum_reference_word_edits': edits,
            'effort_is_proxy': True}


def build_inputs(case, *, personal):
    """Use production profile adaptation without reading any profile store."""
    profile_id = 'synthetic-expansion-' + case['id']
    detail = case.get('detail')
    details = [{'id': 'fixture-detail/' + case['id'], **detail, 'kind': 'object',
                'settings': [case['context']]}] if personal and detail else []
    profile = context.Profile(id=profile_id, revision=1, label='Frozen synthetic evaluation speaker',
                              language='English', protected_terms=case.get('protected_exact', []),
                              specializations=details)
    selection = context.Selection(profile_id=profile_id, profile_revision=1,
        scenario='outside' if case['context'] == 'outdoors' else case['context']).model_dump()
    snapshot = resolve_snapshot(selection, profile.model_dump())
    hits = [{'source_id': 'fixture/' + case['id'], 'kind': 'wording' if detail else 'remembered',
             'text': case['distractor'], 'wording': case['distractor'], 'source_revision': 1,
             'anchor': (detail or {}).get('anchor', ''), 'plain': (detail or {}).get('plain', ''),
             'relevance': .9, 'scope': {'settings': [case['context']]},
             'provenance': 'Frozen synthetic fixture, not participant history'}] if personal else []
    raw = SimpleNamespace(hypotheses=[Hypothesis(id='h1', literal_text=case['literal'],
                                                 sequence_score=0, search_weight=1)])
    job = {'context': snapshot, 'source_text': case['literal'], 'answers': [],
           'output_language': 'English', 'output_script': 'auto',
           'ranking': {'route': 'verified', 'decision': 'selected', 'selected_hypothesis_id': 'h1',
                       'ordered_hypothesis_ids': ['h1']},
           'retrieval': {'status': 'fixture', 'candidates': {'h1': {'hits': hits}}}}
    return raw, job


class RecordedChain(GroqMessageChain):
    """Record synthetic provider replies without changing production decisions."""
    def __init__(self, settings):
        super().__init__(settings)
        self.provider_body = None
        self.provider_model = None
        self.last_result = None

    def _complete(self, payload, personalized=False):
        model, body = super()._complete(payload, personalized)
        self.provider_body, self.provider_model = body, model
        return model, body

    async def run(self, *args, **kwargs):
        self.last_result = await super().run(*args, **kwargs)
        return self.last_result


async def run_trial(case, variant, trial, settings, *, chain=None):
    personal = variant == 'retrieved'
    chain = chain or RecordedChain(settings)
    raw, job = build_inputs(case, personal=personal)
    runtime = SimpleNamespace(settings=settings, message_chain=chain)
    started = time.perf_counter()
    rejection = None
    try:
        candidates, result, _ = await recognition.compose(runtime, raw, job, None)
    except providers.ProviderFailure as exc:
        if exc.code != 'wording_fidelity':
            raise
        rejection = exc.code
        candidates, result = [], chain.last_result
    body = chain.provider_body
    provider_available = body is not None
    generated = [item for item in candidates if item.get('available')]
    wording = generated[0]['text'] if generated else ''
    if not provider_available:
        outcome = 'provider_failure'
    elif body.get('unclear'):
        outcome = 'model_abstained'
    elif not generated:
        outcome = 'safety_withheld'
    else:
        outcome = 'suggested'
    assessments = [{'wording': option.get('message', ''), 'reading': option.get('reading', ''),
                    **evaluate(case, option.get('message', ''), personal=personal)}
                   for option in (body or {}).get('options', [])]
    return {'case_id': case['id'], 'variant': variant, 'trial': trial, 'literal': case['literal'],
            'wording': wording, 'provider_available': provider_available, 'outcome': outcome,
            'rejection': rejection, 'reason': result.ranker.reason if result else '',
            'model': chain.provider_model, 'elapsed_seconds': round(time.perf_counter() - started, 3),
            'provider_options': assessments,
            'provider_meaning_violation': any(item['meaning_violation'] for item in assessments),
            'contextual_additions': generated[0].get('contextual_additions', []) if generated else [],
            **evaluate(case, wording, personal=personal)}


def write_reports(out, manifest, rows, cases):
    valid = [row for row in rows if row['provider_available']]
    summary = {'manifest': manifest, 'trials': len(rows), 'provider_failures': len(rows) - len(valid),
               'comparison_valid': len(valid) == len(rows) and len(rows) == manifest['expected_trials'],
               'limitations': ['Synthetic English fixtures are controlled tests, not participant histories or acoustic evaluation.',
                               'The selected literal is supplied to isolate expansion; no ASR accuracy claim follows.',
                               'Edit distances cover suggested messages only, and measure differences from approved examples, not actual user effort.',
                               'Withheld wording is reported separately and is not counted as a faithful completed suggestion.',
                               'The bounded grader shares checks with production; it cannot prove semantic fidelity. Human review is still required.',
                               'The two repeats measure observed variability; provider sampling is not seeded by the local shuffle seed.'],
               'configurations': {}}
    for variant in ('plain', 'retrieved'):
        group = [row for row in valid if row['variant'] == variant]
        edits = [row['minimum_reference_word_edits'] for row in group if row['minimum_reference_word_edits'] is not None]
        summary['configurations'][variant] = {'completed': len(group),
            'suggestions': sum(row['outcome'] == 'suggested' for row in group),
            'meaning_violations': sum(row['meaning_violation'] for row in group),
            'provider_meaning_violations': sum(row['provider_meaning_violation'] for row in group),
            'wording_withheld': sum(row['wording_withheld'] for row in group),
            'safety_withheld': sum(row['outcome'] == 'safety_withheld' for row in group),
            'model_abstained': sum(row['outcome'] == 'model_abstained' for row in group),
            'edit_proxy_observations': len(edits),
            'mean_minimum_word_edits': sum(edits) / len(edits) if edits else None}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    shuffled = list(valid)
    random.Random(20260928).shuffle(shuffled)
    case_map = {case['id']: case for case in cases}
    with (out / 'blinded_review.csv').open('w', newline='') as handle, (out / 'review_key.csv').open('w', newline='') as key_handle:
        writer, key = csv.writer(handle), csv.writer(key_handle)
        writer.writerow(['review_id', 'literal', 'approved_optional_detail', 'suggested_message', 'meaning_preserved', 'corrected_message', 'notes'])
        key.writerow(['review_id', 'case_id', 'variant', 'trial', 'outcome'])
        for number, row in enumerate(shuffled, 1):
            identifier = f'R{number:03d}'
            case = case_map[row['case_id']]
            # Give reviewers the declared detail needed to judge fidelity, while
            # hiding condition labels, automatic grades and provider identity.
            detail = case.get('detail', {}).get('surface', '') if row['variant'] == 'retrieved' else ''
            writer.writerow([identifier, row['literal'], detail, row['wording'], '', '', ''])
            key.writerow([identifier, row['case_id'], row['variant'], row['trial'], row['outcome']])
    (out / 'review_instructions.md').write_text(
        '# Blinded wording review\n\nReview `blinded_review.csv` before opening the key or summary. '
        'Condition names, automatic grades and provider identity are hidden. An approved optional detail is shown only when it is needed '
        'to assess meaning; this makes blinding partial for the two personalization fixtures.\n\n'
        'Judge whether each completed suggestion preserves the literal\'s names, negation, quantities, order, and communicative act. '
        'Only the listed approved optional detail may add meaning. Enter yes, no, or uncertain; for no, write a minimal faithful correction. '
        'An empty suggestion means wording was withheld: mark not applicable, and optionally write a useful faithful message. '
        'These are synthetic examples, not clinical observations. Reviewers have not yet rated this pack.\n')
    lines = ['# Expansion evaluation', '',
             f'Production `recognition.compose`: {len(rows)}/{manifest["expected_trials"]} trials; {summary["provider_failures"]} provider failures.', '',
             '| Condition | Suggestions | Displayed meaning violations | Provider meaning violations | Safety withheld | Model abstained | Mean word-edit proxy |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for variant, values in summary['configurations'].items():
        proxy = values['mean_minimum_word_edits']
        lines.append(f'| {variant} | {values["suggestions"]} | {values["meaning_violations"]} | {values["provider_meaning_violations"]} | {values["safety_withheld"]} | {values["model_abstained"]} | {proxy:.3f} |' if proxy is not None else f'| {variant} | 0 | 0 | {values["provider_meaning_violations"]} | {values["safety_withheld"]} | {values["model_abstained"]} | — |')
    lines.extend(['', *['- ' + limitation for limitation in summary['limitations']], '',
                  'The review pack is awaiting human ratings. V1 remains a historical chain-only run; its capitalization-sensitive anchor check produced false positives. V2 is a new run through the full composition boundary with corrected frozen expectations.'])
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    return summary


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/derived/verification/expansion-v2')
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--retry-failures', action='store_true')
    args = parser.parse_args()
    if args.repeat < 2:
        parser.error('Use at least two trials per condition.')
    settings = load_settings()
    if not settings.groq_configured:
        raise SystemExit('Configured Groq wording access is required; no provider was called.')
    fixtures = HERE / 'expansion_cases.json'
    cases = json.loads(fixtures.read_text())['cases']
    frozen_inputs = []
    for case in cases:
        for variant in ('plain', 'retrieved'):
            raw, job = build_inputs(case, personal=variant == 'retrieved')
            frozen_inputs.append({'case_id': case['id'], 'variant': variant,
                                  'hypotheses': [item.model_dump() for item in raw.hypotheses], 'job': job})
    frozen_input_bytes = (json.dumps(frozen_inputs, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode()
    hashes = file_hashes()
    manifest = {'version': 2, 'fixture_sha256': hashlib.sha256(fixtures.read_bytes()).hexdigest(),
                'code_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                'inputs_sha256': hashlib.sha256(frozen_input_bytes).hexdigest(),
                'code_files': hashes, 'models': list(settings.groq_models), 'repeat': args.repeat,
                'expected_trials': len(cases) * 2 * args.repeat, 'shuffle_seed': 20260928,
                'composition_entrypoint': 'communication.backend.recognition.compose',
                'settings': {name: getattr(settings, name) for name in
                             ('personal_enabled', 'personal_specializations', 'personal_anchor_share', 'personal_max_hints')},
                'evaluation': 'Separate expansion; synthetic frozen profiles; no acoustic model selection, personal database or TORGO labels.',
                'prior_run': 'expansion-v1 retained; anchor case matching corrected, declared proper names remain exact.'}
    if not settings.personal_enabled or not settings.personal_specializations:
        raise SystemExit('Personalization must be enabled for the declared comparison; no provider was called.')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise SystemExit('Evaluation identity changed; use a new output directory.')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'fixtures.json').write_bytes(fixtures.read_bytes())
    (args.output / 'frozen_inputs.json').write_bytes(frozen_input_bytes)
    trial_file = args.output / 'trials.jsonl'
    rows = [json.loads(line) for line in trial_file.read_text().splitlines()] if trial_file.exists() else []
    by_key = {(r['case_id'], r['variant'], r['trial']): r for r in rows}
    for case in cases:
        for variant in ('plain', 'retrieved'):
            for trial in range(args.repeat):
                key = (case['id'], variant, trial)
                if key in by_key and (by_key[key]['provider_available'] or not args.retry_failures):
                    continue
                if file_hashes() != hashes or hashlib.sha256(fixtures.read_bytes()).hexdigest() != manifest['fixture_sha256']:
                    raise SystemExit('Code or fixtures changed during the run; progress retained without mixing identities.')
                row = await run_trial(case, variant, trial, settings)
                by_key[key] = row
                temporary = trial_file.with_suffix('.tmp')
                temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in by_key.values()))
                temporary.replace(trial_file)
                write_reports(args.output, manifest, list(by_key.values()), cases)
                print(f'{len(by_key)}/{manifest["expected_trials"]} {case["id"]} {variant} {row["outcome"]}', flush=True)
    print(json.dumps(write_reports(args.output, manifest, list(by_key.values()), cases), indent=2))


if __name__ == '__main__':
    asyncio.run(main())
