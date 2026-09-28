// Runs the GazeFollower neural model on this computer, through the loopback proxy.
// The shared board API keeps calibration/validation/confirmation identical.
export function createGazeFollower(){
  let stream=null,video=null,timer=0,token='',running=false,listener=()=>{},latest=null;
  let features=[],labels=[];const pending=new Set();
  const canvas=document.createElement('canvas');canvas.width=640;canvas.height=480;
  const ctx=canvas.getContext('2d',{willReadFrequently:true});
  const tracker={getEyePatches:async()=>latest?.tracked?latest:null};
  const reg={init(){features=[];labels=[];},screenXClicksArray:{get length(){return features.length;}}};
  async function request(path,body){
    const abort=new AbortController();pending.add(abort);
    const timeout=setTimeout(()=>abort.abort(),15000);
    try{
      const response=await fetch('/gaze-api/'+path,{method:'POST',headers:{'Content-Type':'application/json','X-Echora-Client':'1','X-Gaze-Session':token},body:JSON.stringify(body),signal:abort.signal});
      const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'The local gaze service could not process this request.');return data;
    }finally{clearTimeout(timeout);pending.delete(abort);}
  }
  async function loop(){
    if(!running)return;
    try{
      if(document.hidden||!video||video.readyState<2){timer=setTimeout(loop,80);return;}
      ctx.drawImage(video,0,0,640,480);const captured=performance.now();
      latest=await request('frame',{jpeg:canvas.toDataURL('image/jpeg',.75).split(',')[1]});
      if(!running)return;
      if(performance.now()-captured>700)latest={tracked:false};
      await tracker.getEyePatches();
      listener(latest.tracked&&latest.point?{x:latest.point[0]*innerWidth,y:latest.point[1]*innerHeight}:null);
      if(running)timer=setTimeout(loop,80);
    }catch(error){if(running)window.dispatchEvent(new ErrorEvent('error',{message:error.message||'The local gaze service stopped.'}));}
  }
  const api={
    params:{},getTracker:()=>tracker,getRegression:()=>[reg],
    setGazeListener(fn){listener=fn;return api;},clearGazeListener(){listener=()=>{};return api;},
    recordScreenPosition(x,y){if(running&&latest?.tracked&&features.length<45){features.push([...latest.features]);labels.push([x/innerWidth,y/innerHeight]);}},
    async fit(){await request('calibrate',{features,labels});},
    async begin(){
      running=true;
      try{
        token=(await request('session',{})).token;
        stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},facingMode:'user'},audio:false});
        if(!running){stream.getTracks().forEach(t=>t.stop());return;}
        const container=document.createElement('div');container.id='webgazerVideoContainer';
        video=document.createElement('video');video.muted=true;video.playsInline=true;video.srcObject=stream;container.append(video);document.body.append(container);await video.play();
        if(running)void loop();
      }catch(error){api.end();throw new Error(`GazeFollower could not start: ${error.message||'Start the local gaze service.'}`);}
    },
    pause(){running=false;clearTimeout(timer);return api;},
    end(){
      running=false;clearTimeout(timer);for(const controller of pending)controller.abort();
      stream?.getTracks().forEach(t=>t.stop());document.getElementById('webgazerVideoContainer')?.remove();
      if(token)void fetch('/gaze-api/stop',{method:'POST',headers:{'Content-Type':'application/json','X-Echora-Client':'1','X-Gaze-Session':token},body:'{}',keepalive:true}).catch(()=>{});
      token='';latest=null;features=[];labels=[];return api;
    },
  };
  for(const method of ['saveDataAcrossSessions','showPredictionPoints','showFaceOverlay','showFaceFeedbackBox','applyKalmanFilter','setRegression','removeMouseEventListeners'])api[method]=()=>api;
  return api;
}
