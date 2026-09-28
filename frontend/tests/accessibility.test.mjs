import test from 'node:test';
import assert from 'node:assert/strict';
import { defaults, normalizePreferences, DwellGate } from '../components/echora/accessibility/preferences.ts';
import { solve3, fitCalibration, mapPoint, faceFeature } from '../components/echora/accessibility/cameraMath.ts';

test('corrupt storage cannot activate hardware or create unsafe timing', () => {
  assert.deepEqual(normalizePreferences(null), defaults);
  const settings=normalizePreferences({textSize:'huge',dwellMs:-1,scanMs:Infinity,speechRate:8,contrast:'yes',mode:'gaze',camera:true});
  assert.equal(settings.dwellMs,800);assert.equal(settings.scanMs,defaults.scanMs);
  assert.equal(settings.speechRate,1.5);assert.equal(settings.contrast,false);
  assert.equal(settings.mode,undefined);assert.equal(settings.camera,undefined);
});
test('dwell requires full duration, fires once and rearms only on exit',()=>{
  const gate=new DwellGate();
  assert.equal(gate.update('speak',0,1400).activate,false);
  assert.equal(gate.update('speak',1399,1400).activate,false);
  assert.equal(gate.update('speak',1400,1400).activate,true);
  assert.equal(gate.update('speak',9000,1400).activate,false);
  gate.update(null,9100,1400);
  assert.equal(gate.update('speak',9200,1400).activate,false);
  assert.equal(gate.update('speak',10600,1400).activate,true);
});
test('moving target or losing tracking discards accumulated dwell',()=>{
  const gate=new DwellGate();gate.update('record',0,1000);
  assert.equal(gate.update('confirm',999,1000).progress,0);
  gate.update(null,1500,1000);
  assert.equal(gate.update('confirm',2100,1000).activate,false);
});
test('calibration learns translation, scale and cross-axis motion',()=>{
  const samples=[{x:0,y:0},{x:1,y:0},{x:0,y:1},{x:1,y:1},{x:.5,y:.5}].map(feature=>({feature,target:{x:feature.x*.7+feature.y*.1+.1,y:feature.y*.6-feature.x*.1+.2}}));
  const fit=fitCalibration(samples);assert.ok(fit);
  const p=mapPoint(fit,{x:.3,y:.8});
  assert.ok(Math.abs(p.x-.39)<1e-8);assert.ok(Math.abs(p.y-.65)<1e-8);
});
test('stationary or inconsistent calibration cannot enable selection',()=>{
  assert.equal(solve3([[0,0,0],[0,0,0],[0,0,0]],[0,0,0]),null);
  assert.equal(fitCalibration([{feature:{x:.5,y:.5},target:{x:0,y:0}},{feature:{x:.5,y:.5},target:{x:1,y:1}}]),null);
  const samples=Array.from({length:12},(_,i)=>({feature:{x:i%3,y:Math.floor(i/3)},target:{x:i%2,y:(i*3)%2}}));
  assert.equal(fitCalibration(samples),null);
});
test('missing eyes and blink frames produce no gaze pointer',()=>{
  assert.equal(faceFeature([],'gaze'),null);
  const points=Array.from({length:478},()=>({x:.5,y:.5}));
  assert.equal(faceFeature(points,'gaze'),null);
  assert.deepEqual(faceFeature(points,'head'),{x:.5,y:.5});
});
