"""No camera access: checks native imports, model inference and no-face handling."""
import json
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
from gazefollower.gaze_estimator import MGazeNetGazeEstimator
from gazefollower.face_alignment import MediaPipeFaceAlignment
from gazefollower.misc import FaceInfo

result={'platform':platform.platform(),'python':platform.python_version(),'camera_used':False,'accuracy_tested':False}
model=MGazeNetGazeEstimator()
face=FaceInfo(status=True,can_gaze_estimation=True,img_w=640,img_h=480,
    face_rect=np.array([160,80,320,320]),left_rect=np.array([200,160,80,60]),right_rect=np.array([350,160,80,60]))
outputs=[];timings=[]
for seed in range(8):
    frame=np.random.default_rng(seed).integers(0,256,(480,640,3),dtype=np.uint8)
    start=perf_counter();output=model.detect(frame,face);timings.append((perf_counter()-start)*1000)
    assert output.status and np.isfinite(output.features).all(), 'Model returned invalid features'
    outputs.append(output.features)
result.update(feature_count=len(outputs[0]),median_inference_ms=round(float(np.median(timings[1:])),2),features_change_with_input=bool(np.max(np.abs(outputs[0]-outputs[1]))>1e-6))
assert result['features_change_with_input'],'Model outputs did not respond to input'
no_face=model.detect(np.zeros((480,640,3),dtype=np.uint8),FaceInfo())
assert not no_face.status
result['no_face_blocks_prediction']=True
model.release()
print(json.dumps(result,indent=2))
Path(__file__).with_name('runtime-result.json').write_text(json.dumps(result,indent=2)+'\n')
