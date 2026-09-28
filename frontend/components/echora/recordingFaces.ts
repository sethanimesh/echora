// Samples only within the audio recording window; never requests hardware or uploads.
export class RecordingFaces {
  private generation = 0;
  private frames: Blob[] = [];
  private timer: ReturnType<typeof setInterval> | null = null;
  private pending = false;
  private lastTime = -1;
  start(video: HTMLVideoElement) {
    this.stop();
    this.frames = [];
    this.lastTime = -1;
    const generation = this.generation;
    const sample = async () => {
      if (
        this.pending ||
        generation !== this.generation ||
        video.readyState < 2 ||
        !video.videoWidth ||
        video.currentTime === this.lastTime
      )
        return;
      this.lastTime = video.currentTime;
      this.pending = true;
      try {
        const canvas = document.createElement('canvas');
        const scale = Math.min(
          480 / video.videoWidth,
          640 / video.videoHeight,
          1,
        );
        canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
        canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
        const context = canvas.getContext('2d');
        if (!context) return;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise<Blob | null>((resolve) =>
          canvas.toBlob(resolve, 'image/jpeg', 0.8),
        );
        if (generation === this.generation && blob && blob.size <= 256 * 1024)
          this.frames = [...this.frames.slice(-2), blob];
      } finally {
        if (generation === this.generation) this.pending = false;
      }
    };
    this.pending = false;
    this.timer = setInterval(() => {
      void sample();
    }, 300);
    void sample();
  }
  stop(): Blob[] {
    this.generation++;
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.pending = false;
    const result = this.frames.length === 3 ? [...this.frames] : [];
    this.frames = [];
    return result;
  }
}
