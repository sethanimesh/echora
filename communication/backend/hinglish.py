"""Async cloud-only adapter around the vendored Echora 2.0 text pipeline."""
import asyncio
import json
import time
import re
from . import providers
from .hinglish_core.classify.lexical import LexicalClassifier
from .hinglish_core.classify.port import SpanQuery, SpanDecision, SentencePriorGuard
from .hinglish_core.classify.prompt import build_user_message, parse_response, response_schema, system_prompt
from .hinglish_core.core.model import Label, Source
from .hinglish_core.detect.segment import segment
from .hinglish_core.pipeline import Pipeline
from .hinglish_core.tts.normalize import HinglishTtsNormalizer

async def classify(text, queries):
    schema = response_schema(queries)
    schema['properties']['spans']['items']['properties']['id']['enum'] = [q.span_id for q in queries]
    listing = build_user_message(text, queries) + '\nReturn exactly ' + str(len(queries)) + ' entries, one for each of these ids: ' + ', '.join(str(q.span_id) for q in queries) + '. Do not label sentence words absent from the numbered list.'
    response = await providers.request('chat/completions', json={
        'model': providers.TEXT_MODEL, 'temperature': 0, 'reasoning_effort': 'low', 'max_completion_tokens': 2200,
        'messages': [{'role': 'system', 'content': system_prompt(queries)},
                     {'role': 'user', 'content': listing}],
        'response_format': {'type': 'json_schema', 'json_schema': {'name': 'hinglish_spans', 'strict': True, 'schema': schema}}})
    try:
        choice = response['choices'][0]
        if choice.get('finish_reason') != 'stop': raise ValueError('Incomplete classification')
        data = json.loads(choice['message']['content'])
        ids = [item['id'] for item in data['spans']]
        if len(ids) != len(queries) or set(ids) != {q.span_id for q in queries}:
            raise ValueError('Missing or duplicate spans')
        parsed = parse_response(data, queries)
        if len(parsed) != len(queries): raise ValueError('Invalid label')
        decisions = []
        for q in queries:
            label, confidence, index = parsed[q.span_id]
            if label == 'HI' and len(q.candidates) > 1 and index is None:
                raise ValueError('Missing word-sense selection')
            decisions.append(SpanDecision(q.span_id, Label(label), confidence, index))
        return decisions, response.get('usage')
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise providers.ProviderFailure('hinglish_response', 'The wording model did not return a complete Hinglish classification. Your message is unchanged; no other model was used.') from exc

class RecordedClassifier:
    last_used = providers.TEXT_MODEL
    def __init__(self, decisions): self.decisions = decisions
    def classify(self, sentence, spans): return self.decisions

class ProtectedLexical(LexicalClassifier):
    """Protect exact name/brand spans without treating every homonym as a name."""
    def __init__(self, protected):
        super().__init__()
        self.terms = [term for term in protected if term]

    def run(self, utterance):
        result = super().run(utterance)
        text = ''.join(part.text for part in result.segments)
        ranges = [(m.start(), m.end()) for term in self.terms
                  for m in re.finditer(r'(?<!\w)'+re.escape(term)+r'(?!\w)', text)]
        return result.with_segments([
            part.decided(Label.NAME, Source.ALLOWLIST) if part.text.strip() and any(a <= part.start and part.end <= b for a,b in ranges) else part
            for part in result.segments])


def lexical_pass(text, protected):
    lexical = ProtectedLexical(protected)
    utterance = lexical.run(segment(text))
    pending = [s for s in utterance.segments if s.label is Label.AMBIGUOUS]
    queries = [SpanQuery(i, s.text, s.start, s.end, s.candidates) for i, s in enumerate(pending)]
    return lexical, queries

def finish(text, lexical, decisions):
    guard = SentencePriorGuard(RecordedClassifier(decisions), lexical=lexical)
    result = Pipeline(lexical=lexical, classifier=guard).run(text)
    # Reuse the Archive's pronunciation normalizer while keeping recognized
    # names and code/URL spans outside its acronym/currency rewrite pass.
    normalizer = HinglishTtsNormalizer()
    spoken, buffer = [], []
    for part in result.utterance.segments:
        if part.label in {Label.NAME, Label.CODE_URL}:
            spoken.append(normalizer.normalize(''.join(buffer))); buffer.clear()
            spoken.append(part.rendered)
        else: buffer.append(part.rendered)
    spoken.append(normalizer.normalize(''.join(buffer)))
    return result, ''.join(spoken), guard.overrides

async def prepare(text, protected, charge):
    started = time.monotonic()
    lexical, queries = await asyncio.to_thread(lexical_pass, text, protected)
    if len(queries) > 48:
        raise providers.ProviderFailure('hinglish_length', 'This message has too many uncertain words to prepare in one request. Try a shorter message or use the original pronunciation.')
    decisions, usage = [], None
    if queries:
        charge()
        decisions, usage = await classify(text, queries)
    result, speech, overrides = await asyncio.to_thread(finish, text, lexical, decisions)
    return {'display_text': text, 'pronunciation_text': result.text, 'speech_text': speech,
            'changes': [{'original': s.text, 'pronunciation': s.output} for s in result.utterance.segments if s.output and s.output != s.text],
            'metadata': {'model': providers.TEXT_MODEL if queries else 'Echora 2.0 lexicon',
                         'elapsed_ms': round((time.monotonic() - started) * 1000),
                         'ambiguous_spans': len(queries), 'english_guard_overrides': overrides, 'usage': usage}}
