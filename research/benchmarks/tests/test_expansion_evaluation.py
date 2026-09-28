"""Frozen expansion expectations and production-boundary evaluation checks."""
import asyncio
import json
from types import SimpleNamespace

from research.benchmarks.message_chain.evaluate_expansion import (
    HERE, RecordedChain, build_inputs, evaluate, run_trial, write_reports,
)
from communication.backend import context


def settings():
    return SimpleNamespace(groq_api_key=None, groq_configured=True, personal_enabled=True,
                           personal_specializations=True, personal_anchor_share=.75, personal_max_hints=8)


class FixtureChain(RecordedChain):
    def __init__(self, response):
        super().__init__(settings())
        self.client = object()
        self.response = response
        self.calls = 0

    def _complete(self, payload, personalized=False):
        self.calls += 1
        self.payload = payload
        if isinstance(self.response, Exception):
            raise self.response
        self.provider_body, self.provider_model = self.response, 'fixture'
        return 'fixture', self.response


def response(literal, message, *, unclear=False):
    return {'options': [{'reading': literal, 'message': message}], 'note': 'fixture', 'unclear': unclear}


def test_all_fixtures_build_without_opening_personal_database(monkeypatch):
    def forbidden_store_access():
        raise AssertionError('The synthetic benchmark must not open a personal database')
    monkeypatch.setattr(context, 'db', forbidden_store_access)
    for case in json.loads((HERE / 'expansion_cases.json').read_text())['cases']:
        for personal in (False, True):
            raw, job = build_inputs(case, personal=personal)
            assert raw.hypotheses[0].literal_text == case['literal']
            assert job['context']['profile']['id'].startswith('synthetic-expansion-')
            assert job['context']['profile']['protected_terms'] == case['protected_exact']
            if personal and case.get('detail'):
                detail = job['context']['profile']['specializations'][0]
                assert detail['settings'] == [case['context']]


def test_rejection_is_withheld_wording_and_has_no_completed_edit_proxy():
    case = {'id': 'test', 'literal': 'no tea', 'references': ['No tea, please.']}
    assert evaluate(case, '')['wording_withheld']
    assert evaluate(case, '')['minimum_reference_word_edits'] is None
    assert evaluate(case, 'I want tea.')['meaning_violation']
    assert not evaluate(case, 'No tea, please.')['meaning_violation']


def test_protected_names_exact_and_intent_anchors_case_insensitive():
    question = {'id': 'question', 'literal': 'where is the washroom', 'intent_anchors': ['where'],
                'references': ['Where is the washroom?']}
    assert not evaluate(question, 'Where is the washroom?')['meaning_violation']
    assert evaluate(question, 'I want to use the washroom.')['meaning_violation']
    named = {'id': 'name', 'literal': 'Maya water', 'protected_exact': ['Maya'],
             'references': ['Maya, please bring water.']}
    assert evaluate(named, 'Kiran, please bring water.')['meaning_violation']
    assert evaluate(named, 'maya, please bring water.')['meaning_violation']
    assert evaluate(named, 'Mayan water')['meaning_violation']
    assert not evaluate(named, 'Maya, please bring water.')['meaning_violation']


def test_order_is_not_a_bag_of_words():
    ordered = {'id': 'order', 'literal': 'first water then medicine',
               'references': ['First water, then medicine.']}
    assert evaluate(ordered, 'First medicine, then water.')['meaning_violation']


def test_production_composition_withholds_missing_declared_name():
    case = {'id': 'recipient', 'literal': 'Maya bring water', 'protected_exact': ['Maya'],
            'references': ['Maya, please bring water.'], 'context': 'home', 'distractor': 'Kiran bring water'}
    chain = FixtureChain(response(case['literal'], 'Please bring me water.'))
    result = asyncio.run(run_trial(case, 'plain', 0, settings(), chain=chain))
    assert chain.calls == 1
    assert chain.payload['communication_context']['allowed_hypothesis_ids'] == ['h1']
    assert result['outcome'] == 'safety_withheld'
    assert result['rejection'] == 'wording_fidelity'
    assert result['provider_meaning_violation']
    assert not result['meaning_violation']
    assert not result['wording']


def test_production_composition_applies_scoped_approved_detail():
    case = next(case for case in json.loads((HERE / 'expansion_cases.json').read_text())['cases']
                if case['id'] == 'approved-tea')
    chain = FixtureChain(response('tea', 'I want tea.'))
    result = asyncio.run(run_trial(case, 'retrieved', 0, settings(), chain=chain))
    assert result['outcome'] == 'suggested'
    assert 'Lipton green tea' in result['wording']
    assert result['contextual_additions'][0]['wording'] == 'Lipton green tea'
    assert not result['meaning_violation']


def test_provider_failure_is_separate_from_model_abstention_and_guard():
    case = {'id': 'test', 'literal': 'no tea', 'references': ['No tea, please.'], 'context': 'home', 'distractor': 'I want tea.'}
    failed = asyncio.run(run_trial(case, 'plain', 0, settings(), chain=FixtureChain(RuntimeError('failed'))))
    abstained = asyncio.run(run_trial(case, 'plain', 0, settings(), chain=FixtureChain(response('no tea', 'no tea', unclear=True))))
    unsafe = asyncio.run(run_trial(case, 'plain', 0, settings(), chain=FixtureChain(response('no tea', 'I want tea.'))))
    assert failed['outcome'] == 'provider_failure' and not failed['provider_available']
    assert abstained['outcome'] == 'model_abstained' and abstained['provider_available']
    assert unsafe['outcome'] == 'safety_withheld' and unsafe['provider_available']
    assert unsafe['provider_meaning_violation']


def test_report_excludes_provider_failures_and_withheld_edit_proxy(tmp_path):
    case = {'id': 'test', 'literal': 'no tea', 'references': ['No tea, please.'], 'context': 'home', 'distractor': 'I want tea.'}
    rows = [asyncio.run(run_trial(case, 'plain', 0, settings(), chain=FixtureChain(response('no tea', 'No tea, please.')))),
            asyncio.run(run_trial(case, 'plain', 1, settings(), chain=FixtureChain(response('no tea', 'I want tea.')))),
            asyncio.run(run_trial(case, 'retrieved', 0, settings(), chain=FixtureChain(RuntimeError('failed'))))]
    report = write_reports(tmp_path, {'expected_trials': 4}, rows, [case])
    assert report['provider_failures'] == 1
    assert not report['comparison_valid']
    plain = report['configurations']['plain']
    assert plain['completed'] == 2 and plain['edit_proxy_observations'] == 1
    assert plain['meaning_violations'] == 0 and plain['provider_meaning_violations'] == 1
    assert plain['safety_withheld'] == 1
    assert plain['mean_minimum_word_edits'] == 0
