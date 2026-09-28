import assert from 'node:assert/strict';
import test from 'node:test';
import { VoiceMotion } from '../components/echora/voiceMotion.ts';
const frame = (amplitude, bias = 0) =>
  Float32Array.from(
    { length: 1024 },
    (_, i) => bias + amplitude * Math.sin(i * 0.13),
  );

test('silence, DC bias and low room noise leave the globe still', () => {
  for (const audio of [frame(0), frame(0, 0.15), frame(0.006)]) {
    const motion = new VoiceMotion();
    for (let i = 0; i < 300; i++) assert.equal(motion.step(audio, 1 / 30), 0);
  }
});
test('brief microphone clicks do not trigger motion', () => {
  const motion = new VoiceMotion();
  assert.equal(motion.step(frame(0.2), 1 / 30), 0);
  for (let i = 0; i < 30; i++) assert.equal(motion.step(frame(0), 1 / 30), 0);
});
test('sustained input reacts gently and settles to a complete stop', () => {
  const motion = new VoiceMotion();
  for (let i = 0; i < 15; i++) motion.step(frame(0.08), 1 / 30);
  const active = motion.step(frame(0.08), 1 / 30);
  assert.ok(active > 0.1 && active < 0.6);
  const released = motion.step(frame(0), 1 / 30);
  assert.ok(released > 0 && released < active);
  for (let i = 0; i < 90; i++) motion.step(frame(0), 1 / 30);
  assert.equal(motion.step(frame(0), 1 / 30), 0);
});
test('ending recording clears motion and a new recording resets the floor', () => {
  const motion = new VoiceMotion();
  for (let i = 0; i < 30; i++) motion.step(frame(0.1), 1 / 30);
  for (let i = 0; i < 90; i++) motion.step(null, 1 / 30);
  assert.equal(motion.step(null, 1 / 30), 0);
  motion.reset();
  assert.equal(motion.step(frame(0.003), 1 / 30), 0);
});
