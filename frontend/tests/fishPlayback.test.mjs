import test from 'node:test';
import assert from 'node:assert/strict';
import { FishPlayback } from '../components/echora/fishPlayback.ts';

const target = {id:'message', revision:1, confirmation_id:'approved'};
const callbacks = {started(){throw new Error('Must not play');}, ended(){}, failed(){}};

test('stopping during download discards late audio even when fetch ignores abort', async (t) => {
  let finish;
  const requests = [];
  t.mock.method(globalThis, 'fetch', (url) => {
    requests.push(url);
    if (url.endsWith('/stop')) return Promise.resolve(new Response('{}'));
    return new Promise(resolve => {finish = resolve;});
  });
  const player = new FishPlayback();
  const running = player.play(target, callbacks);
  player.stop();
  finish(new Response('ID3-audio', {headers:{'Content-Type':'audio/mpeg', 'X-Confirmation-ID':'approved'}}));
  await running;
  assert.equal(player.analyser, null);
  assert.equal(requests.filter(url => url.endsWith('/audio')).length, 1);
  assert.ok(requests.some(url => url.endsWith('/stop')));
});

test('mismatched confirmation cannot play and failure never retries', async (t) => {
  let calls = 0;
  t.mock.method(globalThis, 'fetch', async (url) => {
    if (url.endsWith('/stop')) return new Response('{}');
    calls++;
    return new Response('ID3-audio', {headers:{'Content-Type':'audio/mpeg','X-Confirmation-ID':'old-message'}});
  });
  await assert.rejects(new FishPlayback().play(target, callbacks), /did not match/);
  assert.equal(calls, 1);
});
