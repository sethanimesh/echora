/** Visual-only level gate. Never filters recordings or decides when speech ends. */
export class VoiceMotion {
  private floor = 0.004;
  private aboveThreshold = 0;
  private open = false;
  private level = 0;

  reset() {
    this.floor = 0.004;
    this.aboveThreshold = 0;
    this.open = false;
    this.level = 0;
  }

  step(samples: Float32Array | null, dt: number): number {
    dt = Math.max(0, Math.min(dt, 0.1));
    let target = 0;
    if (samples?.length) {
      // Remove DC bias so a stationary microphone offset cannot animate the orb.
      let mean = 0;
      for (const sample of samples) mean += sample;
      mean /= samples.length;
      let square = 0;
      for (const sample of samples) square += (sample - mean) ** 2;
      const rms = Math.sqrt(square / samples.length);
      const threshold = Math.max(0.012, this.floor * 3);
      if (!this.open && rms < threshold) {
        const seconds = rms < this.floor ? 0.2 : 2;
        this.floor += (rms - this.floor) * (1 - Math.exp(-dt / seconds));
      }
      if (rms > threshold) this.aboveThreshold += dt;
      else this.aboveThreshold = 0;
      if (this.aboveThreshold >= 0.08) this.open = true;
      if (rms < threshold * 0.65) this.open = false;
      if (this.open)
        target = Math.min(1, Math.max(0, (rms - threshold * 0.65) / 0.12));
    } else {
      this.open = false;
      this.aboveThreshold = 0;
    }
    const seconds = target > this.level ? 0.16 : 0.28;
    this.level += (target - this.level) * (1 - Math.exp(-dt / seconds));
    if (target === 0 && this.level < 0.002) this.level = 0;
    return this.level;
  }
}
