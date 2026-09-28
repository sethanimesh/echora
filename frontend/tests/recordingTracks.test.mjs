import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import ts from 'typescript';
import { RecordingFaces } from '../components/echora/recordingFaces.ts';

test('combined capture records only audio and releases both devices on stop', async t => {
  const names = ['navigator', 'window', 'MediaStream', 'MediaRecorder'];
  const originals = names.map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)]);
  t.after(() => originals.forEach(([name, descriptor]) => {
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else delete globalThis[name];
  }));
  const track = kind => ({kind, stopped: false, stop() { this.stopped = true; }});
  const microphone = track('audio'), camera = track('video');
  class Stream {
    constructor(tracks) { this.tracks = tracks; }
    getTracks() { return this.tracks; }
    getAudioTracks() { return this.tracks.filter(t => t.kind === 'audio'); }
    getVideoTracks() { return this.tracks.filter(t => t.kind === 'video'); }
  }
  let recorded, constraints;
  class Recorder {
    static isTypeSupported() { return true; }
    constructor(stream) { recorded = stream; this.state = 'inactive'; this.mimeType = 'audio/webm'; }
    start() { this.state = 'recording'; }
    stop() { this.state = 'inactive'; this.onstop?.(); }
  }
  const source = new Stream([microphone, camera]);
  Object.defineProperty(globalThis, 'navigator', {configurable: true, value: {mediaDevices: {
    async getUserMedia(value) { constraints = value; return source; },
  }}});
  globalThis.window = {MediaRecorder: Recorder};
  globalThis.MediaRecorder = Recorder;
  globalThis.MediaStream = Stream;
  const react = {useRef: value => ({current: value}), useState: value => [value, () => {}], useEffect() {}};
  const exports = {};
  const code = ts.transpileModule(readFileSync(new URL('../components/echora/useRecorder.ts', import.meta.url), 'utf8'), {
    compilerOptions: {module: ts.ModuleKind.CommonJS},
  }).outputText;
  // Execute only the local hook compiled above, with controlled browser/React substitutes.
  // oxlint-disable-next-line typescript/no-implied-eval
  new Function('require', 'exports', code)(id => id === 'react' ? react : {RecordingFaces}, exports);
  const hook = exports.useRecorder();
  t.after(() => hook.stop());
  const video = {srcObject: null, readyState: 0, async play() {}};
  await hook.start({includeFace: true, video});
  assert.ok(constraints.video && constraints.audio);
  assert.equal(video.srcObject, source);
  assert.deepEqual(recorded.getTracks(), [microphone]);
  assert.equal(camera.stopped, false);
  hook.stop();
  assert.equal(microphone.stopped, true);
  assert.equal(camera.stopped, true);
  assert.equal(video.srcObject, null);
});
