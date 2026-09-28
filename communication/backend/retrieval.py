"""English-only local semantic retrieval over one speaker's explicit context.

Retrieval scores are bounded relevance features, not acoustic confidence. The
selected model never receives audio, coordinates, or another speaker's entries.
The pure ``rank_entries`` interface is also used by frozen evaluation fixtures.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time

import numpy as np

from . import memory

ROOT = Path(__file__).resolve().parents[2] / 'models' / 'echora-minilm-l6-v2'
MAX_HITS = 5
HALF_LIFE_SECONDS = 30 * 86400
MAX_MEMORY_AGE_SECONDS = 180 * 86400
_embedder = None
_load_lock = threading.Lock()


class MiniLMEmbedder:
    """Pinned local MiniLM, CPU mean/L2 pooling, bounded overlapping chunks.

    ``encode_chunks`` returns every chunk; tokenizer truncation is never used.
    The protocol also permits an injected deterministic encoder in unit tests.
    """
    def __init__(self, root=ROOT):
        import torch
        from transformers import AutoModel, AutoTokenizer

        root = Path(root)
        manifest = json.loads((root / 'manifest.json').read_text())
        self.dimension = int(manifest['dimension'])
        self.max_tokens = int(manifest.get('max_tokens', 128))
        identity = hashlib.sha256((root / 'manifest.json').read_bytes())
        for path in sorted((root / 'encoder').rglob('*')):
            if path.is_file():
                identity.update(str(path.relative_to(root)).encode())
                with path.open('rb') as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                        identity.update(chunk)
        self.model_revision = manifest['foundation_revision'] + ':' + identity.hexdigest() + ':mean-l2:chunks-v1:' + str(self.max_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(str(root / 'encoder'), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(root / 'encoder'), local_files_only=True, use_safetensors=True).to('cpu').eval()
        if int(self.model.config.hidden_size) != self.dimension:
            raise ValueError('Embedding dimensions do not match the pinned manifest.')
        if self.tokenizer.cls_token_id is None or self.tokenizer.sep_token_id is None or self.tokenizer.num_special_tokens_to_add(pair=False) != 2:
            raise ValueError('The pinned MiniLM bundle requires BERT start/end tokens.')
        self._torch = torch
        self._lock = threading.Lock()

    def encode_chunks(self, text):
        torch = self._torch
        tokens = self.tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
        if not tokens:
            return np.zeros((1, self.dimension), dtype=np.float32)
        size = self.max_tokens - self.tokenizer.num_special_tokens_to_add(pair=False)
        if size < 2:
            raise ValueError('Embedding token limit is too small.')
        step = max(1, size - min(16, size // 4))
        chunks = [tokens[start:start + size] for start in range(0, len(tokens), step)]
        vectors = []
        with self._lock, torch.inference_mode():
            for chunk in chunks:
                # transformers 5 tokenizers no longer expose prepare_for_model.
                # Use the pinned BERT token layout without decoding/re-tokenizing
                # chunks (which would lose word-piece boundaries).
                token_ids = torch.tensor([[self.tokenizer.cls_token_id, *chunk, self.tokenizer.sep_token_id]], dtype=torch.long)
                encoded = {'input_ids': token_ids, 'attention_mask': torch.ones_like(token_ids)}
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded['attention_mask'].to(hidden.dtype).unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                vectors.append(torch.nn.functional.normalize(pooled, p=2, dim=1)[0].to(torch.float32).cpu().numpy())
        return np.asarray(vectors, dtype=np.float32)


def local_embedder():
    global _embedder
    if _embedder is None:
        with _load_lock:
            if _embedder is None:
                _embedder = MiniLMEmbedder()
    return _embedder


def _arrays(value, dimension):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not len(array) or array.shape[1] != dimension or not np.isfinite(array).all():
        raise ValueError('The embedding encoder returned invalid vectors.')
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def _query(embedder, text):
    chunks = _arrays(embedder.encode_chunks(text), embedder.dimension)
    vector = chunks.mean(axis=0)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _candidate(item):
    return (item['id'], item['literal_text']) if isinstance(item, dict) else (item.id, item.literal_text)


def _eligible(entry, selection):
    if entry.get('language') != 'en':
        return False
    scope = entry.get('scope') or {}
    if scope.get('settings') and selection.get('core_context') not in scope['settings']:
        return False
    if scope.get('scenarios') and selection.get('scenario') not in scope['scenarios']:
        return False
    for field in ('scenario', 'core_context', 'audience_id', 'listener', 'place_id'):
        if scope.get(field) and scope[field] != selection.get(field, ''):
            return False
    if scope.get('recipient') and scope['recipient'].casefold() != selection.get('recipient', '').casefold():
        return False
    if scope.get('moment_id') and scope['moment_id'] != selection.get('moment_id'):
        return False
    if entry.get('detail_id') and entry['detail_id'] in selection.get('known_detail_ids', []):
        return False
    return True


def _fresh(entry, now):
    # Expired wording stays manageable in the memory list, but cannot vote for
    # a new interpretation. Source-revision checks are separately transactional.
    if any(entry.get(flag) for flag in ('stale', 'contradictory', 'deleted')):
        return False
    if entry.get('kind') != 'remembered':
        return True
    created = float(entry.get('created_at', 0))
    return math.isfinite(created) and 0 <= now - created <= MAX_MEMORY_AGE_SECONDS


def _contradicts(literal, entry, selection):
    """Conservative lexical exclusions, not an unrestricted semantic proof."""
    from app.messaging.fidelity import quantities, words
    source, remembered = words(literal), words(entry['text'])
    negative = {'not', 'no', 'never', 'without', 'neither'}
    if bool(source & negative) != bool(remembered & negative):
        return True
    if quantities(source) != quantities(remembered):
        return True
    for opposites in ({'left', 'right'}, {'open', 'close'}, {'hot', 'cold'}, {'before', 'after'}):
        if source & opposites and remembered & opposites and source & opposites != remembered & opposites:
            return True
    names = selection.get('protected_terms', [])
    present = lambda text: {name.casefold() for name in names if name and
                           re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', text, re.IGNORECASE)}
    spoken_names, stored_names = present(literal), present(entry['text'])
    return bool(spoken_names and stored_names and spoken_names != stored_names)


def _exclude_conflicting_sources(entries):
    from app.messaging.fidelity import words
    negative = {'not', 'no', 'never', 'without', 'neither'}
    groups = {}
    keys = []
    for entry in entries:
        tokens = words(entry['text'])
        key = tuple(sorted(tokens - negative - {'do', 'does', 'did'}))
        keys.append(key)
        groups.setdefault(key, set()).add(bool(tokens & negative))
    return [entry for entry, key in zip(entries, keys, strict=True) if len(groups[key]) == 1]


def rank_entries(hypotheses, entries, *, selection=None, language='en', reference=None,
                 embedder=None, min_similarity=0.35, now=None, vector_loader=None):
    """Pure retrieval contract for runtime and offline evaluation.

    Returns ``{status, candidates: {id: {relevance, reliability, eligible, hits}}}``.
    Entries carry ``source_id,kind,language,text,wording,scope,created_at`` and
    version/hash metadata. ``reference`` is already bounded by conversation.py;
    this function never fetches or extends a conversation history itself.
    """
    if not 0 <= min_similarity <= 1:
        raise ValueError('The retrieval floor must be between zero and one.')
    selection = selection or {}
    candidates = {identifier: {'relevance': 0.0, 'reliability': 0.0, 'eligible': False, 'hits': []}
                  for identifier, _ in map(_candidate, hypotheses)}
    if language not in {'en', 'English'}:
        return {'status': 'unsupported_language', 'candidates': candidates}
    now = time.time() if now is None else now
    entries = [entry for entry in entries if _eligible(entry, selection) and _fresh(entry, now)]
    # Duplicate text cannot fill all five source slots or accumulate evidence.
    unique = {}
    for entry in entries:
        key = (' '.join(entry['text'].casefold().split()), json.dumps(entry.get('scope') or {}, sort_keys=True))
        if key not in unique or entry.get('created_at', 0) > unique[key].get('created_at', 0):
            unique[key] = entry
    entries = _exclude_conflicting_sources(list(unique.values()))
    if not entries:
        return {'status': 'empty', 'candidates': candidates}
    embedder = embedder or local_embedder()
    vectors = [(_arrays(vector_loader(entry), embedder.dimension) if vector_loader else
                _arrays(embedder.encode_chunks(entry['text']), embedder.dimension)) for entry in entries]
    reference_text = reference.get('text', '') if isinstance(reference, dict) else reference or ''
    context_text = reference_text or selection.get('situation', '')
    conversation_vector = _query(embedder, context_text) if context_text else None
    for identifier, literal in map(_candidate, hypotheses):
        query = _query(embedder, literal)
        hits = []
        for entry, matrix in zip(entries, vectors, strict=True):
            if _contradicts(literal, entry, selection):
                continue
            semantic = float(np.clip(np.max(matrix @ query), 0, 1))
            if semantic < min_similarity:
                continue
            conversational = float(np.clip(np.max(matrix @ conversation_vector), 0, 1)) if conversation_vector is not None else 0.0
            recency = 0.5 ** (max(0, now - float(entry.get('created_at', 0))) / HALF_LIFE_SECONDS) if entry['kind'] == 'remembered' else 1.0
            relevance = semantic * (0.8 + 0.1 * conversational + 0.1 * recency)
            hits.append({**entry, 'cosine': semantic, 'conversation_relevance': conversational,
                         'recency': recency, 'reliability': recency, 'relevance': relevance})
        hits.sort(key=lambda item: (-item['relevance'], item['source_id']))
        hits = hits[:MAX_HITS]
        candidates[identifier] = {'relevance': hits[0]['relevance'] if hits else 0.0,
                                  'reliability': hits[0]['reliability'] if hits else 0.0, 'eligible': bool(hits), 'hits': hits}
    return {'status': 'ready', 'candidates': candidates}


def retrieve_candidates(snapshot, hypotheses, *, language='en', reference=None, embedder=None, min_similarity=0.35):
    """Database-backed lookup with versioned provenance and stale-write protection.

    Failure is explicit in ``status`` and contributes zero context. Callers must
    check ``memory.dependencies_valid`` before publishing async results/audio.
    """
    profile = (snapshot or {}).get('profile') or {}
    selection = dict((snapshot or {}).get('selection') or {})
    selection['core_context'] = (snapshot or {}).get('core_context', selection.get('core_context', 'general'))
    selection['listener'] = ((snapshot or {}).get('resolved_audience') or {}).get('listener', '')
    audience = next((item for item in profile.get('audiences', []) if item['id'] == selection.get('audience_id')), {})
    selection['known_detail_ids'] = audience.get('known_detail_ids', [])
    selection['protected_terms'] = [person['name'] for person in profile.get('people', [])] + profile.get('protected_terms', [])
    if profile.get('language') == 'Hindi/Hinglish':
        language = 'hi'
    result = {'profile_id': profile.get('id', ''), 'profile_revision': profile.get('revision', 0),
              'memory_revision': 0, 'dependencies': []}
    empty = lambda status: {**result, **rank_entries(hypotheses, [], language=language), 'status': status}
    if not profile or profile.get('sample') or not selection.get('use_personal_wording', True):
        return empty('disabled')
    if language not in {'en', 'English'}:
        return empty('unsupported_language')
    try:
        source = memory.source_snapshot(snapshot)
        result.update({key: source[key] for key in ('profile_id', 'profile_revision', 'memory_revision')})
        entries = [entry for entry in source['entries'] if _eligible(entry, selection)]
        if not entries:
            return empty('empty')
        embedder = embedder or local_embedder()

        def vectors(entry):
            cached = memory.read_vectors(source['profile_id'], entry['source_id'], entry['content_hash'], embedder.model_revision, embedder.dimension)
            if cached:
                return np.vstack([np.frombuffer(chunk, dtype='<f4') for chunk in cached])
            encoded = _arrays(embedder.encode_chunks(entry['text']), embedder.dimension)
            if not memory.save_vectors(source['profile_id'], entry, embedder.model_revision, encoded):
                raise ValueError('A personal entry changed during retrieval.')
            return encoded

        ranked = rank_entries(hypotheses, entries, selection=selection, language=language, reference=reference,
                              embedder=embedder, min_similarity=min_similarity, vector_loader=vectors)
        dependencies = {hit['source_id']: {key: hit[key] for key in ('source_id', 'source_revision', 'content_hash')}
                        for candidate in ranked['candidates'].values() for hit in candidate['hits']}
        result.update(ranked, dependencies=list(dependencies.values()), encoder_revision=embedder.model_revision)
        if not memory.dependencies_valid(result):
            return empty('stale_context')
        return result
    except Exception as exc:
        # No personal text, vectors, filesystem details or provider credentials
        # belong in error telemetry. Existing exact context remains available.
        return empty('stale_context' if getattr(exc, 'status_code', 0) in {404, 409} else 'unavailable')
