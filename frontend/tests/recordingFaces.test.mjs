import test from 'node:test';
import assert from 'node:assert/strict';
import { RecordingFaces } from '../components/echora/recordingFaces.ts';
import { combinedDelivery } from '../components/echora/deliveryMerge.ts';
function setup(t, delayed=false) {
  const old={document:globalThis.document,setInterval:globalThis.setInterval,clearInterval:globalThis.clearInterval};
  let tick, cleared=0;
  const pending=[];
  globalThis.setInterval=fn=>{tick=fn;return 1;};
  globalThis.clearInterval=()=>cleared++;
  globalThis.document={createElement:()=>({getContext:()=>({drawImage(){}}),toBlob:cb=>delayed?pending.push(cb):cb(new Blob(['snapshot']))})};
  t.after(()=>Object.assign(globalThis,old));
  return {tick:()=>tick(),pending,cleared:()=>cleared};
}
const video=()=>({readyState:2,videoWidth:640,videoHeight:480,currentTime:1});
const flush=()=>Promise.resolve();
test('snapshots are sampled only inside the recording window and a new recording starts empty',async t=>{
  const clock=setup(t), sampler=new RecordingFaces(), v=video();
  assert.deepEqual(sampler.stop(),[]);
  sampler.start(v);await flush();
  for(let i=0;i<5;i++){v.currentTime++;clock.tick();await flush();}
  assert.equal(sampler.stop().length,3);
  assert.deepEqual(sampler.stop(),[]);
  sampler.start(v);await flush();
  assert.deepEqual(sampler.stop(),[]); // short new recording cannot reuse old frames
  assert.ok(clock.cleared()>0);
});
test('a snapshot finishing after stop is discarded',async t=>{
  const clock=setup(t,true), sampler=new RecordingFaces();
  sampler.start(video());
  assert.equal(clock.pending.length,1);
  assert.deepEqual(sampler.stop(),[]);
  clock.pending[0](new Blob(['late']));await flush();
  assert.deepEqual(sampler.stop(),[]);
});
test('frozen preview cannot supply three copies as fresh snapshots',async t=>{
  const clock=setup(t), sampler=new RecordingFaces();
  sampler.start(video());await flush();
  for(let i=0;i<5;i++){clock.tick();await flush();}
  assert.deepEqual(sampler.stop(),[]);
});
const voice=(tone='neutral',rate=1)=>({state:'ready',audio_status:'single_speaker',tone,rate});
const face=(cue='smile')=>({state:'ready',visibility:'clear_face',cue});
test('a clear expression changes both final tone and pace without choosing a source',()=>{
  let result=combinedDelivery(voice(),face());
  assert.equal(result.tone,'warm');assert.equal(result.rate,.9);
  result=combinedDelivery(voice(),face('broad_smile'));
  assert.equal(result.tone,'cheerful');assert.equal(result.rate,1.2);
  result=combinedDelivery(voice('firm',1.2),face());
  assert.equal(result.tone,'warm');assert.equal(result.rate,1);
  assert.equal('conflict' in result,false);
});
test('face influence cannot jump multiple pace presets or flatten expressive voice with neutral face',()=>{
  assert.equal(combinedDelivery(voice('neutral',.65),face('broad_smile')).rate,.9);
  const result=combinedDelivery(voice('cheerful',1.2),face('neutral'));
  assert.equal(result.tone,'cheerful');assert.equal(result.rate,1.2);
});
test('unusable faces never affect tone or pace even if an inconsistent tone label is present',()=>{
  for(const invalid of [null,{state:'error',tone:'warm'},face('unclear'),
      {...face(),visibility:'no_face'}, {...face(),visibility:'multiple_faces'},
      {...face(),visibility:'obscured'}, {state:'ready',tone:'warm'}]) {
    const result=combinedDelivery(voice('firm',1.2),invalid);
    assert.equal(result.tone,'firm');assert.equal(result.rate,1.2);
  }
});
test('face can style pace using the current preference when voice pace is unavailable',()=>{
  let result=combinedDelivery({state:'error'},face('broad_smile'),.9);
  assert.equal(result.tone,'cheerful');assert.equal(result.rate,1);
  result=combinedDelivery({...voice('firm',1.2),audio_status:'overlapping'},face(),1);
  assert.equal(result.tone,'warm');assert.equal(result.rate,.9);
  result=combinedDelivery({state:'error'},face('neutral'),.65);
  assert.equal(result.rate,null);
});
test('wait for both checks and preserve current delivery when neither provides usable cues',()=>{
  const waiting=combinedDelivery(voice(),{...face(),state:'analyzing'});
  assert.equal(waiting.pending,true);assert.equal(waiting.available,false);
  assert.equal(waiting.tone,null);assert.equal(waiting.rate,null);
  const empty=combinedDelivery({state:'stopped'},face('unclear'));
  assert.equal(empty.available,false);assert.equal(empty.tone,null);assert.equal(empty.rate,null);
});
test('all combined outputs remain valid for the existing confirmation and speech controls',()=>{
  for(const tone of ['neutral','warm','cheerful','firm']) {
    for(const rate of [.65,.9,1,1.2]) {
      for(const cue of ['neutral','smile','broad_smile','unclear']) {
        const result=combinedDelivery(voice(tone,rate),face(cue));
        assert.ok(['neutral','warm','cheerful','firm'].includes(result.tone));
        assert.ok([.65,.9,1,1.2].includes(result.rate));
      }
    }
  }
});

test('pace matches the style chosen jointly from facial and voice cues',()=>{
  const lively=combinedDelivery(voice('cheerful',1),face('smile'));
  assert.equal(lively.tone,'cheerful');assert.equal(lively.rate,1.2);
  assert.match(lively.explanation,/smile and lively vocal intonation/);
  const emphasized=combinedDelivery(voice('firm',1),face('broad_smile'));
  assert.equal(emphasized.tone,'warm');assert.equal(emphasized.rate,.9);
  assert.match(emphasized.explanation,/softens the emphasized voice/);
  assert.match(emphasized.explanation,/gentler/);
  const matched=combinedDelivery(voice('cheerful',1.2),face('smile'));
  assert.equal(matched.rate,1.2);assert.match(matched.explanation,/already fits/);
});
