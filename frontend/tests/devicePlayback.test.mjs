import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DevicePlayback,
  deviceSpeechLanguage,
} from '../components/echora/devicePlayback.ts';

function browser(t) {
  const previousWindow = globalThis.window;
  const previousUtterance = globalThis.SpeechSynthesisUtterance;
  const spoken = [];
  globalThis.window = {
    speechSynthesis: { cancel() {}, speak(utterance) { spoken.push(utterance); } },
  };
  globalThis.SpeechSynthesisUtterance = class {
    constructor(text) { this.text = text; }
  };
  t.after(() => {
    globalThis.window = previousWindow;
    globalThis.SpeechSynthesisUtterance = previousUtterance;
  });
  return spoken;
}

test('late events from stopped or replaced device speech cannot update playback state', t => {
  const spoken = browser(t);
  const events = [];
  const callbacks = {
    started: () => events.push('start'),
    ended: () => events.push('end'),
    failed: () => events.push('failure'),
  };
  const player = new DevicePlayback();
  player.play('First message', 'en-IN', 1, callbacks);
  const old = { start: spoken[0].onstart, end: spoken[0].onend, error: spoken[0].onerror };
  player.stop();
  player.play('New message', 'hi-IN', 0.8, callbacks);
  old.start();
  old.end();
  old.error({ error: 'synthesis-failed' });
  assert.deepEqual(events, []);
  spoken[1].onstart();
  assert.deepEqual(events, ['start']);
  assert.equal(spoken[1].lang, 'hi-IN');
  assert.equal(spoken[1].rate, 0.8);
  spoken[1].onend();
  assert.deepEqual(events, ['start', 'end']);
});

test('current device speech failure is reported once', t => {
  const spoken = browser(t);
  let failures = 0;
  new DevicePlayback().play('Message', 'en-IN', 1, {
    started() {}, ended() {}, failed() { failures++; },
  });
  const fail = spoken[0].onerror;
  fail({ error: 'synthesis-failed' });
  fail({ error: 'synthesis-failed' });
  assert.equal(failures, 1);
});

test('device voice follows output language while original language and script remain usable', () => {
  assert.equal(deviceSpeechLanguage('Bring water', 'hi', 'English'), 'en-IN');
  assert.equal(deviceSpeechLanguage('Paani lao', 'en', 'Hindi/Hinglish'), 'hi-IN');
  assert.equal(deviceSpeechLanguage('Paani lao', 'hi', 'original'), 'hi-IN');
  assert.equal(deviceSpeechLanguage('पानी लाओ', 'auto', 'original'), 'hi-IN');
  assert.equal(deviceSpeechLanguage('Bring water', 'auto', 'original'), 'en-IN');
});
