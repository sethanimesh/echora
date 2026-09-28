// Camera lifecycle is separate from inference. This class never uploads anything.
export class FaceCamera {
  private generation = 0;
  private stream: MediaStream | null = null;
  private video: HTMLVideoElement | null = null;
  private getMedia: (
    constraints: MediaStreamConstraints,
  ) => Promise<MediaStream>;
  constructor(
    getMedia: (constraints: MediaStreamConstraints) => Promise<MediaStream> = (
      constraints,
    ) => navigator.mediaDevices.getUserMedia(constraints),
  ) {
    this.getMedia = getMedia;
  }

  stop() {
    this.generation += 1;
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    if (this.video) this.video.srcObject = null;
    this.video = null;
  }

  async start(video: HTMLVideoElement) {
    this.stop();
    const generation = this.generation;
    const stream = await this.getMedia({
      audio: false,
      video: {
        facingMode: 'user',
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
    });
    if (generation !== this.generation) {
      stream.getTracks().forEach((track) => track.stop());
      return false;
    }
    this.stream = stream;
    this.video = video;
    video.srcObject = stream;
    try {
      await video.play();
    } catch (error) {
      if (generation === this.generation) this.stop();
      throw error;
    }
    return generation === this.generation;
  }

  async capture(): Promise<Blob[]> {
    const generation = this.generation;
    const frames: Blob[] = [];
    for (let index = 0; index < 3; index++) {
      if (index) await new Promise((resolve) => setTimeout(resolve, 450));
      const video = this.video;
      if (
        generation !== this.generation ||
        !video ||
        !this.stream
          ?.getVideoTracks()
          .some((track) => track.readyState === 'live')
      )
        throw new Error('Camera stopped. No snapshots were sent.');
      if (video.readyState < 2 || !video.videoWidth || !video.videoHeight)
        throw new Error(
          'The camera is not ready. Wait for the preview and try again.',
        );
      const canvas = document.createElement('canvas');
      canvas.width = Math.min(480, video.videoWidth);
      canvas.height = Math.round(
        (canvas.width * video.videoHeight) / video.videoWidth,
      );
      if (canvas.height > 640) {
        canvas.height = 640;
        canvas.width = Math.round((640 * video.videoWidth) / video.videoHeight);
      }
      const context = canvas.getContext('2d');
      if (!context)
        throw new Error('Camera capture is unavailable in this browser.');
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, 'image/jpeg', 0.8),
      );
      if (generation !== this.generation)
        throw new Error('Camera stopped. No snapshots were sent.');
      if (!blob || blob.size > 256 * 1024)
        throw new Error('The snapshot could not be prepared. Try again.');
      frames.push(blob);
    }
    this.stop();
    return frames;
  }
}
