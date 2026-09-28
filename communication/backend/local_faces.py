"""Pinned EmotiEffLib ONNX model, with YuNet crops; no camera or network access.

The upstream full classifier is run directly, using EmotiEffLib's RGB resize,
ImageNet normalization and eight-class order. No identification embeddings are
extracted or retained. Scores are uncalibrated model scores, not emotion certainty.
"""
import asyncio
from collections import Counter
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import threading
import time
from .providers import ProviderFailure

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'data' / 'face-models'
MANIFEST = ROOT / 'experiments' / 'facial-cues' / 'models.json'
MODEL = 'EmotiEffLib/enet_b0_8_best_afew'
LABELS = ('Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise')
MIN_SCORE = 0.65
MIN_MARGIN = 0.20
MAX_PIXELS = 1280 * 1280
_lock = threading.Lock()
_runtime = None
startup_error = None


def configured():
    return startup_error is None and all(importlib.util.find_spec(name) for name in ('cv2', 'numpy', 'onnxruntime', 'PIL')) and all(
        (MODEL_DIR / name).is_file() for name in ('enet_b0_8_best_afew.onnx', 'face_detection_yunet_2026may.onnx'))


def checked_models():
    manifest = json.loads(MANIFEST.read_text())
    for item in manifest['files']:
        if item['model']:
            data = (MODEL_DIR / item['name']).read_bytes()
            if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('Local model checksum mismatch')


class Runtime:
    def __init__(self):
        import cv2
        import onnxruntime as ort
        ort.disable_telemetry_events()
        checked_models()
        cv2.setNumThreads(1)
        self.detector = cv2.FaceDetectorYN.create(str(MODEL_DIR / 'face_detection_yunet_2026may.onnx'), '', (320, 320), 0.8, 0.3, 100)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.classifier = ort.InferenceSession(str(MODEL_DIR / 'enet_b0_8_best_afew.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
        self.input_name = self.classifier.get_inputs()[0].name

    def classify(self, rgb):
        import cv2
        import numpy as np
        # Same preprocessing as EmotiEffLibRecognizerOnnx._preprocess.
        values = cv2.resize(rgb, (224, 224)) / 255
        values = (values - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        batch = values.transpose(2, 0, 1).astype('float32')[None, ...]
        logits = self.classifier.run(None, {self.input_name: batch})[0].reshape(-1)
        if logits.shape != (8,) or not np.isfinite(logits).all():
            raise ValueError('Invalid classifier output')
        scores = np.exp(logits - logits.max())
        scores /= scores.sum()
        order = np.argsort(scores)
        return LABELS[int(order[-1])], float(scores[order[-1]]), float(scores[order[-1]] - scores[order[-2]])

    def assess(self, image):
        import cv2
        import numpy as np
        from PIL import Image
        with Image.open(io.BytesIO(image)) as decoded:
            if decoded.format != 'JPEG' or min(decoded.size) < 64 or decoded.width * decoded.height > MAX_PIXELS:
                raise ValueError('Unsupported snapshot dimensions')
            rgb = np.asarray(decoded.convert('RGB'))
        height, width = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.detector.setInputSize((width, height))
        _, found = self.detector.detect(bgr)
        if found is None or len(found) == 0:
            return {'visibility': 'no_face', 'cue': 'unclear'}
        if len(found) != 1:
            return {'visibility': 'multiple_faces', 'cue': 'unclear'}
        box = found[0]
        if not np.isfinite(box).all():
            return {'visibility': 'obscured', 'cue': 'unclear'}
        x, y, w, h = map(float, box[:4])
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(width, int(x + w)), min(height, int(y + h))
        if min(x1-x0, y1-y0) < 80 or (x1-x0)*(y1-y0) < .88*w*h:
            return {'visibility': 'obscured', 'cue': 'unclear'}
        crop = rgb[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        if not 25 <= gray.mean() <= 235 or gray.std() < 12 or cv2.Laplacian(gray, cv2.CV_64F).var() < 18:
            return {'visibility': 'obscured', 'cue': 'unclear'}
        label, score, margin = self.classify(crop)
        usable = score >= MIN_SCORE and margin >= MIN_MARGIN
        cue = {'Happiness': 'positive_expression', 'Neutral': 'neutral'}.get(label, 'unclear') if usable else 'unclear'
        return {'visibility': 'clear_face', 'cue': cue, 'label': label,
                'score': round(score, 4), 'margin': round(margin, 4)}


def aggregate(frames):
    unsuitable = next((frame for frame in frames if frame['visibility'] != 'clear_face'), None)
    if unsuitable:
        return {'visibility': unsuitable['visibility'], 'cue': 'unclear', 'tone': None, 'frames': frames}
    cue, count = Counter(frame['cue'] for frame in frames).most_common(1)[0]
    if count < 2 or cue == 'unclear': cue = 'unclear'
    return {'visibility': 'clear_face', 'cue': cue,
            'tone': {'positive_expression': 'warm', 'neutral': 'neutral'}.get(cue), 'frames': frames}


def infer(frames):
    global _runtime
    if not _lock.acquire(blocking=False):
        raise ProviderFailure('face_local_busy', 'The local facial model is busy. No other model was substituted.')
    try:
        started = time.monotonic()
        if _runtime is None: _runtime = Runtime()
        result = aggregate([_runtime.assess(frame) for frame in frames])
        return {**result, 'model': MODEL, 'elapsed_ms': round((time.monotonic()-started)*1000), 'policy': 'local-face-1'}
    finally:
        _lock.release()


async def suggest(frames):
    if len(frames) != 3 or any(not frame or len(frame) > 256*1024 for frame in frames):
        raise ProviderFailure('face_images', 'Use three bounded JPEG snapshots for local facial cues.')
    if not configured():
        raise ProviderFailure('face_local_config', 'The local facial model is not installed. Your voice analysis still works.')
    try:
        return await asyncio.wait_for(asyncio.to_thread(infer, tuple(frames)), timeout=15)
    except ProviderFailure:
        raise
    except TimeoutError as exc:
        raise ProviderFailure('face_local_timeout', 'The local facial check took too long. No fallback was used.') from exc
    except Exception as exc:
        raise ProviderFailure('face_local_failed', 'The local facial model could not process these snapshots. No fallback was used.') from exc


def warmup():
    """Load verified weights before a test recording, without opening a camera."""
    global _runtime
    with _lock:
        if _runtime is None: _runtime = Runtime()
