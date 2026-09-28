"""One short-lived approved reference, never a transcript/history dump."""
import copy
import re
import time

TTL = 600

def scope(snapshot):
    selection = (snapshot or {}).get('selection') or {}
    profile = (snapshot or {}).get('profile') or {}
    return (profile.get('id', ''), profile.get('revision', 0), selection.get('scenario', 'general'),
            selection.get('recipient', '').casefold(), selection.get('listener', 'unspecified'),
            selection.get('situation', ''), selection.get('moment_id', ''), selection.get('use_personal_wording', True),
            selection.get('place_id', ''), selection.get('audience_id', ''), selection.get('declared_listener'),
            (snapshot or {}).get('core_context', selection.get('core_context')),
            ((snapshot or {}).get('resolved_audience') or {}).get('listener'),
            tuple(sorted((((snapshot or {}).get('resolved_audience') or {}).get('style') or {}).items())))


def remember(job):
    confirmed = job.get('confirmed')
    if not confirmed: return None
    return {'job_id': job['id'], 'text': confirmed['text'], 'scope': scope(confirmed.get('context')),
            'at': time.monotonic()}


def candidate(previous, memory):
    if not previous or not previous.get('confirmed') or previous.get('status') == 'cancelled': return None
    if not memory or memory['job_id'] != previous['id'] or time.monotonic()-memory['at'] > TTL: return None
    return copy.deepcopy(memory)


def followup(text):
    # Narrow, auditable triggers. A new named need or full sentence gets no memory.
    value = text.strip().rstrip('.!।').strip().casefold()
    patterns = [r'(?:without|with|less|more|no) (?:sugar|milk|ice|salt|cheeni|doodh)',
                r'(?:make (?:it|that)|change (?:it|that) to) (?:one|two|three|four|[1-4]|hot|cold|warm|large|small)',
                r'(?:the same|same again|that one|it too)',
                r'(?:bina|kam|zyada) (?:cheeni|chini|doodh|namak)',
                r'(?:cheeni|chini|doodh|namak) (?:nahi|nahin|mat)',
                r'(?:बिना|कम|ज़्यादा|ज्यादा) (?:चीनी|दूध|नमक)', r'(?:चीनी|दूध|नमक) (?:नहीं|मत)']
    return any(re.fullmatch(pattern, value) for pattern in patterns)


def reference(saved, text, snapshot, independent=False):
    if independent or not saved or not followup(text): return None
    if time.monotonic()-saved['at'] > TTL or saved['scope'] != scope(snapshot): return None
    return {'text': saved['text'], 'job_id': saved['job_id']}
