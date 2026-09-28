import test from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { loadVisionRuntime } from '../components/echora/accessibility/visionRuntime.ts';

test('ES-module worker startup avoids MediaPipe document-based loader', async () => {
  globalThis.self=globalThis;
  assert.equal(typeof globalThis.document,'undefined');
  assert.equal(typeof globalThis.importScripts,'undefined');
  const base=pathToFileURL(resolve('public/accessibility')).href;
  const original=await FilesetResolver.forVisionTasks(base+'/wasm',true);
  await assert.rejects(FaceLandmarker.createFromOptions(original,{canvas:{},baseOptions:{modelAssetPath:'unused'}}),/document is not defined/);
  const files=await loadVisionRuntime(base);
  assert.equal(files.wasmLoaderPath,'');
  assert.equal(typeof globalThis.ModuleFactory,'function');
  assert.equal(files.wasmBinaryPath,original.wasmBinaryPath);
  // Stop at the WASM factory boundary: this regression check does not emulate
  // WebGL, claim full camera inference, or need a user's camera.
  let reached=false;
  globalThis.ModuleFactory=async options=>{
    reached=true;
    assert.equal(options.locateFile('runtime.wasm'),files.wasmBinaryPath);
    throw new Error('WASM factory reached');
  };
  await assert.rejects(FaceLandmarker.createFromOptions(files,{canvas:{},baseOptions:{modelAssetPath:'unused'}}),/WASM factory reached/);
  assert.equal(reached,true);
  delete globalThis.ModuleFactory;
  delete globalThis.self;
});
