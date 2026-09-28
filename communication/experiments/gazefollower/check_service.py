"""Smoke-check the running local service without webcam access."""
import base64,json
from pathlib import Path
import cv2,httpx,numpy as np
with httpx.Client(base_url='http://127.0.0.1:3000',timeout=20) as c:
    health=c.get('/gaze-api/health');health.raise_for_status()
    assert c.post('/gaze-api/session',json={}).status_code==403
    assert c.post('/gaze-api/session',json={},headers={'X-Echora-Client':'1','Origin':'https://example.invalid'}).status_code==403
    headers={'X-Echora-Client':'1','Origin':'http://127.0.0.1:3000'}
    r=c.post('/gaze-api/session',json={},headers=headers);r.raise_for_status()
    headers['X-Gaze-Session']=r.json()['token']
    _,jpeg=cv2.imencode('.jpg',np.zeros((480,640,3),dtype=np.uint8))
    r=c.post('/gaze-api/frame',json={'jpeg':base64.b64encode(jpeg).decode()},headers=headers);r.raise_for_status()
    assert r.json()=={'tracked':False},r.text
    assert c.post('/gaze-api/frame',json={'jpeg':'invalid'},headers=headers).status_code==422
    assert c.post('/gaze-api/calibrate',json={'features':[], 'labels':[]},headers=headers).status_code==422
    features=np.random.default_rng(7).normal(size=(45,258)).tolist()
    targets=[[x,y] for y in [.2,.5,.8] for x in [.15,.5,.85] for _ in range(5)]
    r=c.post('/gaze-api/calibrate',json={'features':features,'labels':targets},headers=headers);r.raise_for_status()
    assert r.json()['retained']==45
    c.post('/gaze-api/stop',json={},headers=headers).raise_for_status()
    assert c.post('/gaze-api/frame',json={'jpeg':'x'},headers=headers).status_code==409
result={'loopback_proxy':True,'request_header_required':True,'frontend_origin_accepted':True,'foreign_origin_rejected':True,'blank_image_no_prediction':True,'invalid_frame_rejected':True,'invalid_calibration_rejected':True,'fit_accepts_45_balanced_samples':True,'stop_invalidates_session':True,'camera_used':False,'accuracy_tested':False}
Path(__file__).with_name('service-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
