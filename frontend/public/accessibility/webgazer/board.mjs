import { choices, hitTile, scoreTrial, TileDwell, CalibrationSampler } from './board-core.mjs';
import { createGazeFollower } from './native-adapter.mjs';
const neural=new URLSearchParams(location.search).get('engine')==='gazefollower';
const $=id=>document.getElementById(id), wg=neural?createGazeFollower():window.webgazer;
if(neural){document.querySelector('header strong').textContent='Eye gaze · GazeFollower trial';$('hint').textContent='Processed on this computer · No saved frames or calibration';}
const ids=Object.keys(choices), tiles=Object.fromEntries(ids.map(id=>[id,document.querySelector(`[data-choice="${id}"]`)]));
const training=[[.15,.2],[.5,.2],[.85,.2],[.85,.5],[.5,.5],[.15,.5],[.15,.8],[.5,.8],[.85,.8]];
let phase='loading', step=0, started=0, samples=0, fresh=0, point=null, checkHits=[], scores=[], settled=false;
const sampler=new CalibrationSampler();
const dwell=new TileDwell(), milliseconds=Math.max(1000,Math.min(4000,Number(new URLSearchParams(location.search).get('dwell'))||1600));
const status=text=>{if($('status').textContent!==text)$('status').textContent=text;};
const rects=()=>Object.fromEntries(ids.map(id=>[id,tiles[id].getBoundingClientRect()]));
function clearTarget(){started=0;samples=0;sampler.pause();if(phase==='checking')checkHits=[];dwell.reset();point=null;$('pointer').hidden=true;for(const tile of Object.values(tiles))tile.classList.remove('active');}
function release(){
  fresh=0;clearTarget();
  try{wg?.clearGazeListener().removeMouseEventListeners().pause();}catch{}
  for(const video of document.querySelectorAll('video')) if(video.srcObject instanceof MediaStream)video.srcObject.getTracks().forEach(track=>track.stop());
  try{wg?.getTracker()?.detector?.dispose();}catch{}
  try{wg?.end();}catch{}
}
function close(){phase='stopped';release();parent.postMessage({type:'echora-gaze-stop'},location.origin);}
function fail(message){
  if(phase==='stopped'||phase==='failed')return;
  phase='failed';release();$('intro').hidden=true;$('dot').hidden=true;$('board').hidden=true;$('result').hidden=false;$('enable').hidden=true;
  $('result-title').textContent='Gaze controls could not continue';$('scores').textContent='';$('explanation').textContent=message;status('Selection is off.');
}
function calibration(){
  wg.getRegression().forEach(reg=>reg.init());
  sampler.reset();
  phase='training';step=0;scores=[];clearTarget();$('intro').hidden=true;$('result').hidden=true;$('board').hidden=true;$('dot').hidden=false;document.body.className='';showDot();
}
function showDot(){const [x,y]=training[step];$('dot').style.left=x*100+'%';$('dot').style.top=y*100+'%';status(`Look at the dot · ${step+1} of ${training.length}. Keep your head comfortably still.`);}
async function startCheck(){
  if(neural){phase='fitting';$('dot').hidden=true;status('Learning your gaze positions…');try{await wg.fit();}catch(error){fail(error.message);return;}if(phase==='stopped'||phase==='failed')return;}
  const retained=wg.getRegression()[0]?.screenXClicksArray?.length;
  if(retained!==45){fail(`Calibration stored ${retained ?? 0} of 45 samples. Please stop and report this error.`);return;}
  phase='checking';step=0;started=0;checkHits=[];scores=[];point=null;$('pointer').hidden=true;$('dot').hidden=true;$('board').hidden=false;document.body.className='testing';showCheck();
}
function showCheck(){for(const id of ids)tiles[id].classList.toggle('expected',id===ids[step]);status(`Accuracy check ${step+1} of 4 · Look at ${ids[step]}.`);}
function finishCheck(){
  const passed=scores.every(score=>score.passed);phase=passed?'practice':'insufficient';clearTarget();
  for(const tile of Object.values(tiles))tile.classList.remove('expected');
  $('result').hidden=false;$('enable').hidden=!passed;
  $('result-title').textContent=passed?'The large-choice check passed':'The gaze check needs another try';
  $('scores').textContent=scores.map((score,i)=>`${ids[i]}: ${score.percent}% (${score.correct}/${score.total})`).join(' · ');
  $('explanation').textContent=passed?'Check the pointer, then enable gaze choices. Hold over one tile to put it into your message for review.':'Selection stays off. These percentages show how often your estimated gaze landed inside each tile. We need at least 70% for every tile. You can recalibrate or stop; this is a tracker limitation, not a mistake you made.';
  status(passed?'Practice first. Nothing will be chosen until you enable gaze choices.':'Accuracy was insufficient. Selection is off.');
}
function enable(){if(phase!=='practice')return;phase='board';$('result').hidden=true;document.body.className='ready';clearTarget();status('Look at a tile and hold. Your choice will open for review.');}
function choose(id){
  if(phase!=='board'||!Object.hasOwn(choices,id))return;
  phase='stopped';release();parent.postMessage({type:'echora-gaze-choice',choice:id},location.origin);
}
function drawPointer(p,progress=0){
  const el=$('pointer');el.hidden=!p;if(!p)return;
  el.style.left=p.x+'px';el.style.top=p.y+'px';el.style.background=`conic-gradient(#315e8b ${progress*360}deg,transparent 0)`;
}
function gaze(data){
  const now=performance.now();
  if(document.hidden||!document.hasFocus()||now-fresh>450){clearTarget();return;}
  point=data&&Number.isFinite(data.x)&&Number.isFinite(data.y)?{x:data.x,y:data.y}:null;
  if(phase==='training'){
    showDot();
    if(sampler.take(now)){const [x,y]=training[step];wg.recordScreenPosition(x*innerWidth,y*innerHeight,'click');}
    $('dot').style.background=`conic-gradient(#315e8b ${sampler.count/5*360}deg,#dae6f1 0)`;
    if(sampler.complete){step++;sampler.reset();if(step===training.length)startCheck();else showDot();}
  }else if(phase==='checking'){
    showCheck();
    if(!started)started=now;
    if(now-started>1200)checkHits.push(hitTile(point,rects()));
    if(now-started>=4200&&checkHits.length>=12){scores.push(scoreTrial(checkHits,ids[step]));step++;started=0;checkHits=[];if(step===ids.length)finishCheck();else showCheck();}
  }else if(phase==='practice'){
    drawPointer(point);
    const r=$('enable').getBoundingClientRect();const target=point&&point.x>=r.left&&point.x<=r.right&&point.y>=r.top&&point.y<=r.bottom?'enable':null;
    const result=dwell.update(target,now,milliseconds);drawPointer(point,result.progress);if(result.selected)enable();
  }else if(phase==='board'){
    status('Look at a tile and hold. Your choice will open for review.');
    const id=hitTile(point,rects());
    for(const key of ids)tiles[key].classList.toggle('active',key===id);
    const result=dwell.update(id,now,milliseconds);drawPointer(point,result.progress);if(result.selected)choose(result.selected);
  }
}
$('start').onclick=calibration;$('enable').onclick=enable;$('stop').onclick=close;
$('retry').onclick=()=>{if(phase==='failed')location.reload();else calibration();};
for(const id of ids)tiles[id].onclick=()=>choose(id);
window.addEventListener('keydown',e=>{if(e.code==='Escape'){e.preventDefault();close();}});
window.addEventListener('pagehide',release);
document.addEventListener('visibilitychange',()=>{if(document.hidden)close();});
window.addEventListener('resize',()=>{if(!['loading','intro','failed','stopped'].includes(phase)){phase='insufficient';clearTarget();$('dot').hidden=true;$('result').hidden=false;$('enable').hidden=true;$('result-title').textContent='The screen size changed';$('scores').textContent='';$('explanation').textContent='Recalibrate for the new screen size before choosing.';}});
window.addEventListener('unhandledrejection',e=>fail(e.reason?.message||'The local tracker stopped unexpectedly.'));
window.addEventListener('error',e=>fail(e.message||'A local tracking error occurred.'));
const watchdog=setInterval(()=>{if(['training','checking','practice','board'].includes(phase)&&performance.now()-fresh>450){clearTarget();status('Gaze paused. Keep your face visible and return to the trial.');}},150);
window.addEventListener('pagehide',()=>clearInterval(watchdog));
const startup=setTimeout(()=>{if(!settled)fail('The camera or local models did not finish loading. Stop and retry camera access.');},45000);
try{
  if(!wg)throw new Error('WebGazer could not load. Refresh and try again.');
  wg.params.faceMeshSolutionPath=new URL('./mediapipe/face_mesh',location.href).href;
  wg.saveDataAcrossSessions(false).showPredictionPoints(false).showFaceOverlay(false).showFaceFeedbackBox(false).applyKalmanFilter(true).setRegression('ridge');
  // Timestamp actual eye-patch extraction, including the initial frames for
  // which no trained gaze prediction exists. Null predictions are not proof of
  // a missing face during calibration.
  const tracker=wg.getTracker(), extract=tracker.getEyePatches.bind(tracker);
  tracker.getEyePatches=async(...args)=>{const eyes=await extract(...args);fresh=eyes?performance.now():0;return eyes;};
  wg.setGazeListener(gaze);
  await wg.begin();settled=true;clearTimeout(startup);
  if(phase==='stopped'||phase==='failed'){release();}else{
    wg.removeMouseEventListeners();phase='intro';$('intro').hidden=false;status('Camera ready. Start when you are comfortable.');
  }
}catch(error){settled=true;clearTimeout(startup);fail(error instanceof Error?error.message:String(error));}
