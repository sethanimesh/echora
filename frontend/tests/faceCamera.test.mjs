import test from 'node:test';
import assert from 'node:assert/strict';
import { FaceCamera } from '../components/echora/faceCamera.ts';
function resources() {
  const track = { readyState: 'live', stops: 0, stop() { this.readyState = 'ended'; this.stops++; } };
  const stream = { getTracks: () => [track], getVideoTracks: () => [track] };
  const video = { srcObject: null, play: async () => {}, readyState: 2, videoWidth: 640, videoHeight: 480 };
  return { track, stream, video };
}
function canvas(t) {
  const old = globalThis.document;
  globalThis.document = { createElement: () => ({ width: 0, height: 0, getContext: () => ({ drawImage() {} }), toBlob: (callback) => callback(new Blob(['jpeg'], { type: 'image/jpeg' })) }) };
  t.after(() => { globalThis.document = old; });
}
test('enabling preview requests video only; stop releases it', async () => {
  const { stream, track, video } = resources();
  let calls = 0;
  const camera = new FaceCamera(async constraints => { calls++; assert.equal(constraints.audio, false); return stream; });
  assert.equal(calls, 0);
  assert.equal(await camera.start(video), true);
  assert.equal(video.srcObject, stream);
  camera.stop();
  assert.equal(track.stops, 1); assert.equal(video.srcObject, null);
});
test('permission resolving after stop cannot turn the camera on', async () => {
  const { stream, track, video } = resources();
  let resolve;
  const camera = new FaceCamera(() => new Promise(r => { resolve = r; }));
  const pending = camera.start(video);
  camera.stop(); resolve(stream);
  assert.equal(await pending, false);
  assert.equal(track.stops, 1); assert.equal(video.srcObject, null);
});
test('capture returns exactly three JPEG snapshots and turns off the camera', async t => {
  canvas(t);
  const { stream, track, video } = resources();
  const camera = new FaceCamera(async () => stream);
  await camera.start(video);
  const frames = await camera.capture();
  assert.equal(frames.length, 3);
  assert.ok(frames.every(frame => frame.type === 'image/jpeg'));
  assert.equal(track.stops, 1); assert.equal(video.srcObject, null);
});
test('stop during capture discards the partial batch', async t => {
  canvas(t);
  const { stream, track, video } = resources();
  const camera = new FaceCamera(async () => stream);
  await camera.start(video);
  const pending = camera.capture();
  camera.stop();
  await assert.rejects(pending, /Camera stopped/);
  assert.equal(track.stops, 1);
});
