"""Bridge the unified message lifecycle to the adapted recognizer and grounded chain.

Imports of the model/audio stack are lazy: typed input and the explicit Whisper
option remain usable without loading the adapted model or its dependencies.
"""
import asyncio
import os
import time
import re
from types import SimpleNamespace

from . import context, providers, wording_fidelity


def status(runtime=None):
    adapted_ready = runtime is not None and getattr(runtime, 'asr', None) is not None
    settings = getattr(runtime, 'settings', None)
    verifier = getattr(runtime, 'verification', None)
    return {
        'default_backend': 'adapted' if runtime is not None else 'whisper',
        'verification': {'status': getattr(verifier, 'status', 'unavailable'),
                         'reason': getattr(verifier, 'reason', 'verification_unavailable'),
                         'scope': 'local-adapted-English'},
        'options': [
            {'id': 'adapted', 'label': 'Adapted speech recognition', 'ready': adapted_ready,
             'model': 'echora-qwen3-asr-command-v3',
             'backend': getattr(settings, 'backend', 'local'),
             'message': '' if adapted_ready else 'Adapted recognition is unavailable. Choose another recognizer explicitly.'},
            {'id': 'whisper', 'label': 'Whisper · cloud',
             'ready': bool(os.getenv('GROQ_API_KEY', '').strip()),
             'model': providers.ASR_MODEL},
        ],
    }


async def decode(data, filename, max_seconds):
    from app.audio import AudioValidationError, decode_audio
    try:
        return await asyncio.to_thread(decode_audio, data, filename, max_seconds)
    except AudioValidationError as exc:
        raise providers.ProviderFailure('audio_invalid', str(exc)) from exc


async def transcribe(runtime, audio, filename):
    if runtime is None or getattr(runtime, 'asr', None) is None:
        raise providers.ProviderFailure('adapted_unavailable', 'Adapted recognition is unavailable. Your recording was not sent to another recognizer.')
    started = time.monotonic()
    waveform = await decode(audio, filename, runtime.settings.max_audio_seconds)
    try:
        verifier = getattr(runtime, 'verification', None)
        acoustic_scores = None
        if verifier is not None and hasattr(runtime.asr, 'transcribe_verified'):
            raw, acoustic_scores = await runtime.asr.transcribe_verified(waveform, runtime.settings.beams, verifier)
        else:
            raw = await runtime.asr.transcribe(waveform, runtime.settings.beams)
    except Exception as exc:
        raise providers.ProviderFailure('adapted_failed', 'Adapted recognition could not finish. Retry or choose another recognizer explicitly.') from exc
    if not raw.hypotheses or not any(item.literal_text.strip() for item in raw.hypotheses):
        raise providers.ProviderFailure('no_speech', 'No words were recognized. Try again or type your message.')
    return raw, {'model': raw.model, 'elapsed_ms': round((time.monotonic() - started) * 1000),
                 'audio_seconds': raw.audio_quality.seconds, 'language': 'en',
                 'acoustic_scores': acoustic_scores}


async def rank(runtime, raw, job, acoustic_scores=None, reference=None):
    """Retrieve before fusion without changing any recognizer-owned evidence."""
    verifier = getattr(runtime, 'verification', None)
    if raw.backend != 'local' or verifier is None or runtime.settings.beams != 5:
        return {'route': 'legacy', 'status': 'unsupported', 'decision': 'ambiguous',
                'selected_hypothesis_id': None, 'ordered_hypothesis_ids': [h.id for h in raw.hypotheses],
                'scores': [], 'artifacts': {}, 'reason': 'Acoustic verification is unavailable on this recognition route.'}, None
    retrieval_result = None
    if getattr(runtime.settings, 'personal_enabled', True) and job.get('context'):
        from . import retrieval
        try:
            retrieval_result = await asyncio.to_thread(retrieval.retrieve_candidates,
                job['context'], raw.hypotheses, language='en', reference=reference)
        except Exception:
            retrieval_result = {'status': 'unavailable', 'candidates': {}, 'dependencies': []}
    relevance = {key: {'score': value.get('relevance', 0), 'reliability': value.get('reliability', 0),
                       'eligible': bool(value.get('eligible') and value.get('hits'))}
                 for key, value in (retrieval_result or {}).get('candidates', {}).items()}
    ranking = verifier.rank(raw, acoustic_scores, relevance)
    return {**ranking, 'route': 'verified'}, retrieval_result


def literal_candidates(raw, candidates):
    """Every literal remains selectable, including a selected beam's raw form."""
    result = list(candidates)
    for item in raw.hypotheses:
        if any(c['text'] == item.literal_text and item.id in c['source_hypothesis_ids'] for c in result):
            continue
        result.append({'id': f'raw-{item.id}', 'text': item.literal_text,
            'reading': item.literal_text, 'source_hypothesis_ids': [item.id],
            'source_literals': [item.literal_text], 'plain_text': None,
            'specializations': [], 'word_alternatives': {}, 'available': False,
            'literal_fallback': True, 'retrieved_sources': [], 'contextual_additions': []})
    return result


def _reading(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.casefold()))


def evidence(raw):
    return {'kind': 'audio', 'backend': raw.backend, 'recognizer': 'adapted', 'model': raw.model,
            'hypotheses': [item.model_dump() for item in raw.hypotheses],
            'audio_quality': raw.audio_quality.model_dump(),
            'beam_weights_are_calibrated_confidence': False}


def candidate(item, available=True):
    return {'id': item.message_id, 'text': item.corrected_text, 'reading': item.interpreted_intent,
            'source_hypothesis_ids': list(item.source_hypothesis_ids),
            'source_literals': list(item.source_literals), 'plain_text': item.plain_text,
            'specializations': [entry.model_dump() for entry in item.specializations],
            'word_alternatives': item.word_alternatives, 'available': available,
            'retrieved_sources': getattr(item, 'retrieved_sources', []),
            'contextual_additions': getattr(item, 'contextual_additions', [])}


async def compose(runtime, raw, job, reference, charge=None):
    resolved = context.recognition_context(job.get('context'), raw.hypotheses, runtime,
                                           text=job['source_text'], answers=job['answers'], reference=reference)
    selection = (job.get('context') or {}).get('selection') or {}
    fidelity_context = {
        'output_script': job.get('output_script', 'auto'),
        'protected_terms': wording_fidelity.terms(job.get('context')),
        'recipient': selection.get('recipient', ''),
        'conversation_reference': (reference or {}).get('text', ''),
    }
    fidelity = wording_fidelity.policy(' '.join(item.literal_text for item in raw.hypotheses),
                                      job['answers'], job['output_language'], fidelity_context)
    saved = context.brief(job.get('context'), job['source_text'], job['answers']).get('approved_message')
    profile = (job.get('context') or {}).get('profile') or {}
    ranking = job.get('ranking') or {}
    constrained = ranking.get('route') == 'verified'
    allowed = None
    if constrained:
        ordered = ranking.get('ordered_hypothesis_ids') or [h.id for h in raw.hypotheses]
        allowed = ([ranking['selected_hypothesis_id']] if ranking.get('decision') == 'selected'
                   and ranking.get('selected_hypothesis_id') else ordered[:3])
    retrieval_result = job.get('retrieval') or {}
    retrieved = {key: value.get('hits', []) for key, value in retrieval_result.get('candidates', {}).items()
                 if allowed is None or key in allowed}
    if (saved and not constrained and not reference and job.get('output_script', 'auto') == 'auto'
            and job['output_language'] == profile.get('language')
            and len({item.literal_text.strip().casefold() for item in raw.hypotheses}) == 1):
        # A selected exact cue is already user-authored wording, not an LLM
        # inference. Preserve the same zero-call route as typed personal moments.
        candidates = [{'id': 'saved-moment', 'text': saved, 'reading': job['source_text'],
                       'source_hypothesis_ids': [item.id for item in raw.hypotheses],
                       'source_literals': [item.literal_text for item in raw.hypotheses],
                       'plain_text': None, 'specializations': [], 'available': True,
                       'saved_message': True}]
        result = SimpleNamespace(warnings=[], ranking_seconds=0,
                                 ranker=SimpleNamespace(source='saved', assistant_model='Saved personal message'))
        return candidates, result, resolved
    if charge and getattr(runtime.settings, 'groq_configured', False): charge()
    result = await runtime.message_chain.run(
        raw.hypotheses, resolved['context'], brief=resolved['brief'],
        listener=resolved['listener'], register=resolved['register'],
        conversation_reference=reference,
        communication_context={'output_language': job['output_language'],
                               'output_script': job.get('output_script', 'auto'),
                               'fidelity': fidelity, 'scenario': selection.get('scenario', 'general'),
                               'recipient': selection.get('recipient', ''),
                               'situation': selection.get('situation', ''),
                               **({'allowed_hypothesis_ids': allowed,
                                   'allowed_readings': [h.literal_text for h in raw.hypotheses if h.id in allowed],
                                   'retrieved_context': retrieved}
                                  if constrained else {})},
    )
    available = result.ranker.source != 'unavailable'
    candidates = [candidate(item, available) for item in result.messages]
    if constrained:
        # Enforce the full-beam contract even when a custom provider ignores the
        # prompt. Mixed readings have no genuine acoustic compatibility score.
        candidates = [item for item in candidates if any(
            h.id in allowed and _reading(item['reading']) == _reading(h.literal_text)
            and h.id in item['source_hypothesis_ids'] for h in raw.hypotheses)]
    if available:
        for item in candidates:
            try:
                # The chain normalizes its reading for grounding. Name spelling
                # requirements must come from immutable recognizer literals.
                literal_readings = [h.literal_text for h in raw.hypotheses
                                    if h.id in item['source_hypothesis_ids']
                                    and _reading(h.literal_text) == _reading(item['reading'])]
                fidelity_reading = ' '.join(literal_readings) or item['reading']
                candidate_fidelity = wording_fidelity.policy(fidelity_reading, job['answers'],
                                                            job['output_language'], fidelity_context)
                wording_fidelity.validate({'text': item['text'], 'question': '', 'options': []}, candidate_fidelity)
            except ValueError as exc:
                raise providers.ProviderFailure('wording_fidelity', 'A suggestion changed a protected name or writing choice. Your original alternatives are kept.') from exc
    return candidates, result, resolved
