"""Delivery never changes meaning or bypasses message confirmation."""
import pytest
from test_app import client, create, post
from communication.backend import app as module


def test_delivery_is_separate_and_confirmation_tracks_changes(client):
    words = 'Meena, please do not bring 2 cups of tea.'
    j = create(client, words)
    first = post(client, j, 'confirm', revision=1,
                 delivery={'tone': 'warm', 'rate': 0.65}).json()
    assert first['original'] == first['text'] == first['confirmed']['speech_text'] == words
    assert first['confirmed']['delivery'] == {'tone': 'warm', 'rate': 0.65, 'source': 'user'}
    again = post(client, j, 'confirm', revision=1,
                 delivery={'tone': 'warm', 'rate': 0.65}).json()
    assert again['confirmed']['id'] == first['confirmed']['id']
    changed = post(client, j, 'confirm', revision=1,
                   delivery={'tone': 'firm', 'rate': 1.0}).json()
    assert changed['confirmed']['id'] != first['confirmed']['id']
    assert changed['revision'] == first['revision']
    assert changed['text'] == changed['confirmed']['speech_text'] == words
    assert changed['confirmed']['delivery']['tone'] == 'firm'
    # Emitted snapshots of earlier confirmations retain their original choice.
    events = next(iter(module.sessions.values())).events
    confirmations = [e['job']['confirmed'] for e in events if e['type'] == 'message_confirmed']
    assert [c['delivery']['tone'] for c in confirmations] == ['warm', 'firm']
    assert module.calls_used == 0


def test_edit_invalidates_delivery_confirmation(client):
    j = create(client)
    post(client, j, 'confirm', revision=1, delivery={'tone': 'cheerful', 'rate': 1.2})
    edited = post(client, j, 'edit', revision=1, text='Please bring water.').json()
    assert edited['confirmed'] is None
    assert post(client, j, 'confirm', revision=1, delivery={'tone': 'firm'}).status_code == 409


@pytest.mark.parametrize('delivery', [
    {'tone': 'sad'}, {'rate': 0.1}, {'rate': 2.0}, {'rate': 'fast'},
    {'rate': True}, {'tone': 'warm', 'source': 'camera'}, {'instructions': 'add crying'},
])
def test_invalid_or_inferred_delivery_is_rejected(client, delivery):
    j = create(client)
    assert post(client, j, 'confirm', revision=1, delivery=delivery).status_code == 422
    assert client.get('/api/session').json()['job']['confirmed'] is None


def test_pronunciation_is_preserved_when_only_delivery_changes(client):
    j = create(client, 'Mujhe paani chahiye.')
    saved = next(iter(module.sessions.values())).job
    saved['prepared_speech'] = {'revision': 1, 'display_text': j['text'], 'speech_text': 'मुझे पानी चाहिए।'}
    first = post(client, j, 'confirm', revision=1, delivery={'tone': 'warm'}).json()
    second = post(client, j, 'confirm', revision=1, delivery={'tone': 'firm'}).json()
    assert second['prepared_speech'] == first['prepared_speech']
    assert second['confirmed']['speech_text'] == 'मुझे पानी चाहिए।'
    assert second['text'] == 'Mujhe paani chahiye.'
    assert module.calls_used == 0
