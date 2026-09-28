import asyncio
import io
import threading
import pytest
np = pytest.importorskip('numpy')
Image = pytest.importorskip('PIL.Image')
from communication.backend import local_faces as local, providers


def jpeg(array=None):
    array = array if array is not None else np.random.default_rng(4).integers(50, 210, (320,480,3),dtype=np.uint8)
    out=io.BytesIO();Image.fromarray(array).save(out,format='JPEG');return out.getvalue()


def runtime(boxes):
    result=local.Runtime.__new__(local.Runtime)
    class Detector:
        def setInputSize(self,size): self.size=size
        def detect(self,image): return None, boxes
    result.detector=Detector()
    result.classify=lambda crop: ('Happiness',.8,.5)
    return result


def box(x=100,y=80,w=150,h=150): return np.array([x,y,w,h]+[0]*10+[.95])


@pytest.mark.parametrize('boxes,visibility',[(None,'no_face'),(np.array([box(),box(280)]),'multiple_faces'),(np.array([box(w=30)]),'obscured'),(np.array([box(x=-80)]),'obscured')])
def test_quality_rejections_do_not_classify(boxes,visibility):
    r=runtime(boxes)
    def forbidden(crop): raise AssertionError('Unusable image classified')
    r.classify=forbidden
    assert r.assess(jpeg())['visibility']==visibility


def test_uniform_dark_crop_is_rejected():
    r=runtime(np.array([box()]))
    assert r.assess(jpeg(np.zeros((320,480,3),dtype=np.uint8)))['visibility']=='obscured'


@pytest.mark.parametrize('label,score,margin,cue',[
    ('Happiness',.8,.5,'positive_expression'),('Neutral',.8,.5,'neutral'),
    ('Happiness',.6,.5,'unclear'),('Happiness',.8,.1,'unclear'),
    ('Anger',.9,.7,'unclear'),('Surprise',.9,.7,'unclear')])
def test_scores_gate_expression_without_treating_other_labels_as_delivery(label,score,margin,cue):
    r=runtime(np.array([box()]));r.classify=lambda crop:(label,score,margin)
    assert r.assess(jpeg())['cue']==cue


def test_bad_or_oversized_jpeg_rejected_before_detection():
    r=runtime(None)
    with pytest.raises(Exception): r.assess(b'not jpeg')
    with pytest.raises(ValueError): r.assess(jpeg(np.full((1300,1300,3),128,dtype=np.uint8)))


def test_three_frame_consistency_and_unusable_view_veto():
    f=lambda cue,visibility='clear_face':{'cue':cue,'visibility':visibility}
    assert local.aggregate([f('positive_expression')]*2+[f('unclear')])['tone']=='warm'
    assert local.aggregate([f('positive_expression'),f('neutral'),f('unclear')])['tone'] is None
    assert local.aggregate([f('positive_expression')]*2+[f('unclear','no_face')])['tone'] is None


def test_preprocessing_and_raw_classifier_output(monkeypatch):
    r=runtime(None);r.input_name='input';seen=[]
    class Classifier:
        def run(self,outputs,inputs):
            seen.append(inputs['input']);return [np.array([[0,0,0,0,5,0,0,0]],dtype=np.float32)]
    r.classifier=Classifier()
    label,score,margin=local.Runtime.classify(r,np.full((100,100,3),128,dtype=np.uint8))
    assert label=='Happiness' and score>.9 and margin>.8
    assert seen[0].shape==(1,3,224,224) and seen[0].dtype==np.float32
    np.testing.assert_allclose(seen[0][0,:,0,0],(128/255-np.array([.485,.456,.406]))/np.array([.229,.224,.225]),rtol=1e-5)


def test_inference_is_bounded_and_has_no_silent_fallback(monkeypatch):
    monkeypatch.setattr(local,'configured',lambda:False)
    with pytest.raises(providers.ProviderFailure,match='not installed'): asyncio.run(local.suggest([b'jpg']*3))
    monkeypatch.setattr(local,'configured',lambda:True)
    with pytest.raises(providers.ProviderFailure): asyncio.run(local.suggest([b'jpg']*2))
    assert local._lock.acquire(False)
    try:
        with pytest.raises(providers.ProviderFailure,match='busy'): local.infer([b'jpg']*3)
    finally: local._lock.release()
