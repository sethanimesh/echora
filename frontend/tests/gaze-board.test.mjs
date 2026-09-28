import test from 'node:test';
import assert from 'node:assert/strict';
import {hitTile,scoreTrial,TileDwell} from '../public/accessibility/webgazer/board-core.mjs';
const rects={water:{left:0,top:0,right:400,bottom:300},help:{left:430,top:0,right:830,bottom:300},yes:{left:0,top:330,right:400,bottom:630},no:{left:430,top:330,right:830,bottom:630}};
test('gaze must land inside a tile, with a dead zone around borders',()=>{
  assert.equal(hitTile({x:200,y:150},rects),'water');
  assert.equal(hitTile({x:600,y:500},rects),'no');
  assert.equal(hitTile({x:398,y:150},rects),null);
  assert.equal(hitTile({x:420,y:150},rects),null);
  assert.equal(hitTile({x:NaN,y:100},rects),null);
  assert.equal(hitTile(null,rects),null);
});
test('validation needs enough independent frames and 70 percent per tile',()=>{
  assert.equal(scoreTrial(Array(11).fill('water'),'water').passed,false);
  assert.equal(scoreTrial([...Array(14).fill('water'),...Array(6).fill(null)],'water').passed,true);
  assert.equal(scoreTrial([...Array(13).fill('water'),...Array(7).fill('help')],'water').passed,false);
});
test('unstable gaze cannot accumulate dwell across tiles or missing frames',()=>{
  const gate=new TileDwell();gate.update('yes',0,1500);gate.update('no',1200,1500);
  assert.equal(gate.update('yes',1800,1500).selected,null);
  gate.update(null,2500,1500);gate.update('yes',2600,1500);
  assert.equal(gate.update('yes',4100,1500).selected,'yes');
  assert.equal(gate.update('yes',9000,1500).selected,null);
  gate.reset();assert.equal(gate.update('yes',10000,1500).selected,null);
});

import { CalibrationSampler } from '../public/accessibility/webgazer/board-core.mjs';
import Regression from '../node_modules/webgazer/src/ridgeReg.mjs';
test('all nine calibration positions survive WebGazer’s actual 50-sample window',()=>{
  const reg=new Regression.RidgeReg();
  assert.equal(reg.screenXClicksArray.windowSize,50);
  const sampler=new CalibrationSampler();
  for(let target=0;target<9;target++){
    sampler.reset();
    for(let time=0;time<10000;time+=20){
      if(time===1700 || time===4300)sampler.pause();
      if(sampler.take(time))reg.screenXClicksArray.push([target]);
    }
    assert.equal(sampler.count,5);
  }
  assert.equal(reg.screenXClicksArray.length,45);
  for(let target=0;target<9;target++)assert.equal(reg.screenXClicksArray.data.filter(([x])=>x===target).length,5);
});
test('calibration waits after a lost face without duplicating earlier samples',()=>{
  const sampler=new CalibrationSampler();
  assert.equal(sampler.take(0),false);
  assert.equal(sampler.take(1000),true);
  assert.equal(sampler.take(1010),false);
  sampler.pause();
  assert.equal(sampler.take(2000),false);
  assert.equal(sampler.take(2999),false);
  assert.equal(sampler.take(3000),true);
  assert.equal(sampler.count,2);
});
