import { FaceLandmarker } from '@mediapipe/tasks-vision';
import { loadVisionRuntime } from './visionRuntime';
let detector: FaceLandmarker | null = null;
self.onmessage = async (event: MessageEvent) => {
  const data = event.data;
  try {
    if (data.type === 'init') {
      const files = await loadVisionRuntime(data.base);
      if (typeof OffscreenCanvas === 'undefined') {
        throw new Error(
          'This browser does not support the offscreen canvas needed for camera controls.',
        );
      }
      const canvas = new OffscreenCanvas(640, 480);
      if (!canvas.getContext('webgl2')) {
        throw new Error(
          'WebGL 2 is unavailable in the camera worker. Enable graphics acceleration or try Chrome.',
        );
      }
      detector = await FaceLandmarker.createFromOptions(files, {
        baseOptions: {
          modelAssetPath: data.base + '/face_landmarker.task',
          delegate: 'CPU',
        },
        canvas,
        runningMode: 'VIDEO',
        numFaces: 2,
        minFaceDetectionConfidence: 0.7,
        minFacePresenceConfidence: 0.7,
        minTrackingConfidence: 0.7,
      });
      self.postMessage({ type: 'ready' });
    } else if (data.type === 'frame') {
      try {
        const result = detector?.detectForVideo(data.bitmap, data.time);
        self.postMessage({
          type: 'result',
          time: data.time,
          points:
            result?.faceLandmarks.length === 1 ? result.faceLandmarks[0] : [],
        });
      } finally {
        data.bitmap.close();
      }
    }
  } catch (error) {
    self.postMessage({
      type: 'error',
      stage: data.type === 'init' ? 'startup' : 'frame',
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
