#!/usr/bin/env python3
"""Check the running web proxy and message lifecycle, without inference or playback."""
import httpx

BASE_URL = 'http://127.0.0.1:3000'
HEADERS = {'X-Echora-Client': '1', 'Origin': BASE_URL}


def main():
    with httpx.Client(base_url=BASE_URL, timeout=30) as client:
        page = client.get('/')
        page.raise_for_status()
        assert 'Echora' in page.text, 'The web page did not render Echora.'
        health = client.get('/api/v1/health')
        health.raise_for_status()
        assert health.json()['model_ready'], 'The recognition runtime is not ready.'
        session = client.get('/api/session')
        session.raise_for_status()
        before = session.json()
        assert any(option['id'] == 'adapted' and option['ready']
                   for option in before['recognition']['options'])

        def post(path, body, expected=200):
            result = client.post(path, json=body, headers=HEADERS)
            assert result.status_code == expected, f'{path}: expected {expected}, got {result.status_code}'
            return result.json()

        job = post('/api/messages', {'text': 'This is a local integration check.'})
        assert job['evidence']['kind'] == 'text' and job['auto_speak_revision'] is None
        path = '/api/messages/' + job['id']
        confirmed = post(path + '/confirm', {'revision': job['revision'], 'pronunciation': 'original'})
        assert confirmed['confirmed']['speech_text'] == job['text']
        edited = post(path + '/edit', {'revision': confirmed['revision'], 'text': 'The revised local check.'})
        assert edited['confirmed'] is None and edited['auto_speak_revision'] is None
        post(path + '/confirm', {'revision': confirmed['revision'], 'pronunciation': 'original'}, expected=409)
        cancelled = post(path + '/cancel', {})
        assert cancelled['status'] == 'cancelled' and cancelled['confirmed'] is None
        post(path + '/confirm', {'revision': cancelled['revision'], 'pronunciation': 'original'}, expected=409)
        after = client.get('/api/session').json()
        assert after['calls_used'] == before['calls_used'], 'Unexpected wording/recognition call during smoke check.'
    print('PASS: rendered page, API proxy, model readiness, typed evidence, exact confirmation, stale revision rejection and cancellation. No inference or speech requested.')


if __name__ == '__main__':
    main()
