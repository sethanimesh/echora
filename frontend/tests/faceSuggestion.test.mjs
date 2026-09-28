import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { tones } from '../components/echora/delivery.ts';
import { FaceCamera } from '../components/echora/faceCamera.ts';
const require = createRequire(import.meta.url);
const output = ts.transpileModule(readFileSync(new URL('../components/echora/FaceSuggestion.tsx', import.meta.url), 'utf8'), {
  compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS },
}).outputText;
const exports = {};
new Function('require', 'exports', output)((id) => id === './delivery' ? { tones } : id === './faceCamera' ? { FaceCamera } : require(id), exports);
const base = { configured: true, disabled: false, cameraBlocked: false, onCheck: async () => {}, onStop() {}, onAccept() {} };
const render = props => renderToStaticMarkup(React.createElement(exports.default, { ...base, ...props }));
test('camera starts off, with explicit cloud disclosure and no automatic application', () => {
  let calls=0;
  const html=render({onCheck:()=>calls++,onAccept:()=>calls++});
  assert.equal(calls,0);
  assert.match(html,/hidden=""/);
  assert.match(html,/Enable camera preview/);
  assert.match(html,/sends three snapshots/);
  assert.doesNotMatch(html,/Use camera tone/);
});
test('conflicting voice and camera tones remain choices, without changing pace', () => {
  const html=render({voiceTone:'firm',suggestion:{id:'one',revision:1,state:'ready',visibility:'clear_face',cue:'smile',tone:'warm'}});
  assert.match(html,/Voice suggests Firm; camera suggests Warm/);
  assert.match(html,/Use camera tone/);
  assert.match(html,/Speaking pace and your words/);
});
test('unusable views offer no tone action; camera access conflict is explained', () => {
  const html=render({cameraBlocked:true,suggestion:{id:'one',revision:1,state:'ready',visibility:'multiple_faces',cue:'unclear',tone:null}});
  assert.match(html,/More than one face/);
  assert.match(html,/Stop webcam head or gaze control/);
  assert.doesNotMatch(html,/Use camera tone/);
});
