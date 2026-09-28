"""Explicit Fish Audio synthesis. No retries, model fallback or voice cloning."""
import asyncio
import os
import re
import time
import httpx
from dotenv import dotenv_values
from .providers import ProviderFailure

MODEL = 's2.1-pro-free'
ENDPOINT = 'https://api.fish.audio/v1/tts'
MAX_BYTES = 5 * 1024 * 1024
TONES = {
    'neutral': '[neutral]',
    'warm': '[warm and friendly]',
    'cheerful': '[cheerful]',
    'firm': '[firm and composed]',
}


def load_configuration(paths):
    # Nonempty process variables and the first nonempty supplied file value win.
    for path in paths:
        values = dotenv_values(path)
        for name in ('FISH_API_KEY', 'FISH_REFERENCE_ID'):
            value = (values.get(name) or '').strip()
            if not os.getenv(name, '').strip() and value:
                os.environ[name] = value


def configuration():
    key = os.getenv('FISH_API_KEY', '').strip()
    voice = os.getenv('FISH_REFERENCE_ID', '').strip()
    return key, voice


def status():
    key, voice = configuration()
    return {'configured': bool(key and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', voice)),
            'model': MODEL, 'key_present': bool(key), 'voice_present': bool(voice)}


def payload(confirmed, voice):
    text = confirmed['speech_text']
    # S2 treats brackets and speaker tokens as controls. Do not silently delete
    # literal words, or let text introduce unreviewed expression/speaker changes.
    if any(token in text for token in ('[', ']', '<|', '|>')):
        raise ProviderFailure('fish_control_text', 'This text contains Fish voice-control symbols. Use the device voice to read it as written, or edit the message first.')
    if not text.strip() or len(text.encode('utf-8')) > 12000:
        raise ProviderFailure('fish_text_limit', 'This message is too long for the voice trial. Shorten it or use the device voice.')
    delivery = confirmed['delivery']
    return {'text': TONES[delivery['tone']] + ' ' + text,
            'reference_id': voice, 'format': 'mp3', 'mp3_bitrate': 128,
            'normalize': False, 'latency': 'normal',
            'prosody': {'speed': delivery['rate'], 'volume': 0, 'normalize_loudness': True}}


async def synthesize(body):
    key, _ = configuration()
    started = time.monotonic()
    try:
        async with asyncio.timeout(60):
            async with httpx.AsyncClient(timeout=httpx.Timeout(45, connect=10), follow_redirects=False) as client:
                async with client.stream('POST', ENDPOINT, headers={
                    'Authorization': f'Bearer {key}', 'model': MODEL,
                    'Content-Type': 'application/json',
                }, json=body) as response:
                    if response.status_code != 200:
                        messages = {
                            401: 'Fish Audio rejected the API key.',
                            402: 'Fish Audio requires credits or access for this request. No paid model was substituted.',
                            403: 'Fish Audio denied access to this voice or model.',
                            404: 'Fish Audio could not find this voice or model.',
                            429: 'Fish Audio reached its usage limit. Try later; no automatic retry was made.',
                        }
                        raise ProviderFailure('fish_request_failed', messages.get(response.status_code,
                            'Fish Audio could not generate this message. No other model was tried.'))
                    if response.headers.get('content-type', '').split(';')[0] not in {'audio/mpeg', 'audio/mp3', 'application/octet-stream'}:
                        raise ProviderFailure('fish_invalid_audio', 'Fish Audio returned an unexpected response. Nothing was played.')
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > MAX_BYTES:
                            raise ProviderFailure('fish_audio_limit', 'The voice response exceeded the local size limit. Nothing was played.')
                    if len(data) < 4 or not (data[:3] == b'ID3' or (data[0] == 255 and data[1] & 224 == 224)):
                        raise ProviderFailure('fish_invalid_audio', 'Fish Audio did not return recognizable MP3 audio. Nothing was played.')
                    return bytes(data), {'model': MODEL, 'elapsed_ms': round((time.monotonic()-started)*1000),
                                         'input_bytes': len(body['text'].encode('utf-8'))}
    except (httpx.HTTPError, TimeoutError):
        raise ProviderFailure('fish_unavailable', 'Fish Audio could not be reached in time. Retry explicitly or choose the device voice.') from None
