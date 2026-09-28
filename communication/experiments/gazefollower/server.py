"""Local-only neural gaze trial. Camera JPEGs are processed in memory, never saved."""
import asyncio, base64, io, secrets, time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pool=ThreadPoolExecutor(max_workers=1)
model=aligner=None
state={'token':None,'touched':0,'calibration':None}

def init_models():
    global model,aligner
    from gazefollower.gaze_estimator import MGazeNetGazeEstimator
    from gazefollower.face_alignment import MediaPipeFaceAlignment
    model=MGazeNetGazeEstimator();aligner=MediaPipeFaceAlignment()

@asynccontextmanager
async def lifespan(app):
    await asyncio.get_running_loop().run_in_executor(pool,init_models)
    yield
    if aligner: aligner.release()
    if model: model.release()
    pool.shutdown(wait=False,cancel_futures=True)

app=FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None)
@app.middleware('http')
async def boundary(request:Request,call_next):
    host=request.headers.get('host','').split(':')[0]
    origin=request.headers.get('origin')
    if host not in {'127.0.0.1','localhost','testserver'} or (origin and origin not in {'http://127.0.0.1:3000','http://localhost:3000'}):
        return JSONResponse({'detail':'Local application requests only.'},403)
    if request.method!='GET':
        if request.headers.get('x-echora-client')!='1':return JSONResponse({'detail':'Missing application header.'},403)
        size=0;parts=[]
        async for chunk in request.stream():
            size+=len(chunk)
            if size>1_000_000:return JSONResponse({'detail':'Frame is too large.'},413)
            parts.append(chunk)
        request._body=b''.join(parts)
    response=await call_next(request);response.headers['Cache-Control']='no-store';return response

@app.get('/gaze-api/health')
async def health():return {'ready':model is not None,'engine':'GazeFollower 1.0.2','camera_opened_by_server':False}

def check(request):
    if not state['token'] or request.headers.get('x-gaze-session')!=state['token'] or time.monotonic()-state['touched']>120:
        raise HTTPException(409,'Gaze session ended. Restart the trial.')
    state['touched']=time.monotonic()

@app.post('/gaze-api/session')
async def new_session():
    state.update(token=secrets.token_urlsafe(24),touched=time.monotonic(),calibration=None)
    return {'token':state['token']}

@app.post('/gaze-api/stop')
async def stop(request:Request):
    check(request);state.update(token=None,calibration=None);return {'stopped':True}

def predict(jpeg,calibration):
    try:
        raw=base64.b64decode(jpeg,validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.format!='JPEG' or image.width>640 or image.height>480:raise ValueError()
        frame=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if frame is None:raise ValueError()
    except Exception:raise HTTPException(422,'Use a JPEG frame at most 640 × 480.')
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    start=time.perf_counter()
    face=aligner.detect(int(time.monotonic()*1000),rgb)
    result=model.detect(rgb,face)
    if not result.status:return {'tracked':False}
    features=np.asarray(result.features,dtype=np.float64)
    if len(features)!=258 or not np.isfinite(features).all():raise HTTPException(503,'Neural model returned invalid features.')
    point=calibration.predict(features.reshape(1,-1))[0].tolist() if calibration else None
    return {'tracked':True,'features':features.tolist(),'point':point,'elapsed_ms':round((time.perf_counter()-start)*1000,1)}

@app.post('/gaze-api/frame')
async def frame(request:Request):
    check(request);token=state['token'];body=await request.json()
    if not isinstance(body,dict) or not isinstance(body.get('jpeg'),str):raise HTTPException(422,'Missing frame.')
    result=await asyncio.get_running_loop().run_in_executor(pool,predict,body['jpeg'],state['calibration'])
    if token!=state['token']:raise HTTPException(409,'Gaze session changed.')
    return result

@app.post('/gaze-api/calibrate')
async def calibrate(request:Request):
    check(request);body=await request.json()
    try:
        features=np.asarray(body['features'],dtype=np.float64);labels=np.asarray(body['labels'],dtype=np.float64)
        if features.shape!=(45,258) or labels.shape!=(45,2) or not np.isfinite(features).all() or not np.isfinite(labels).all() or (labels<0).any() or (labels>1).any():raise ValueError()
        _,counts=np.unique(labels,axis=0,return_counts=True)
        if len(counts)!=9 or not (counts==5).all():raise ValueError()
    except Exception:raise HTTPException(422,'Calibration needs five samples at each of nine positions.')
    if np.max(np.std(features,axis=0))<1e-6:raise HTTPException(422,'The model features stayed constant. Selection is off.')
    state['calibration']=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(features,labels)
    return {'retained':45,'validation_required':True}
