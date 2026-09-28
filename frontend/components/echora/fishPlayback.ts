// Owns one generation/download/playback; stopped or superseded audio cannot play.
export class FishPlayback {
  private generation = 0;
  private request: AbortController | null = null;
  private target: {
    id: string;
    revision: number;
    confirmation_id: string;
  } | null = null;
  private audio: HTMLAudioElement | null = null;
  private url: string | null = null;
  private context: AudioContext | null = null;
  analyser: AnalyserNode | null = null;

  stop() {
    this.generation++;
    this.request?.abort();
    this.request = null;
    if (this.target) {
      const { id, ...body } = this.target;
      void fetch(`/api/messages/${id}/audio/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Echora-Client': '1' },
        body: JSON.stringify(body),
        keepalive: true,
      }).catch(() => {});
      this.target = null;
    }
    if (this.audio) {
      this.audio.onplaying = this.audio.onended = this.audio.onerror = null;
      this.audio.pause();
      this.audio.removeAttribute('src');
      this.audio.load();
      this.audio = null;
    }
    if (this.url) URL.revokeObjectURL(this.url);
    this.url = null;
    if (this.context) void this.context.close().catch(() => {});
    this.context = null;
    this.analyser = null;
  }

  async play(
    target: { id: string; revision: number; confirmation_id: string },
    callbacks: {
      started: () => void;
      ended: () => void;
      failed: (message: string) => void;
    },
  ) {
    this.stop();
    const generation = this.generation;
    this.target = target;
    this.request = new AbortController();
    try {
      const { id, ...body } = target;
      const response = await fetch(`/api/messages/${id}/audio`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Echora-Client': '1' },
        body: JSON.stringify(body),
        signal: this.request.signal,
      });
      if (!response.ok) {
        const result = (await response.json().catch(() => ({}))) as {
          detail?: unknown;
        } | null;
        throw new Error(
          typeof result?.detail === 'string'
            ? result.detail
            : 'Fish Audio could not prepare this message.',
        );
      }
      if (
        response.headers.get('X-Confirmation-ID') !== target.confirmation_id ||
        !response.headers.get('Content-Type')?.startsWith('audio/mpeg')
      ) {
        throw new Error(
          'The voice response did not match your confirmed message. Nothing was played.',
        );
      }
      const blob = await response.blob();
      if (generation !== this.generation) return;
      this.request = null;
      this.target = null;
      this.url = URL.createObjectURL(blob);
      const audio = new Audio(this.url);
      this.audio = audio;
      // Meter failure must not prevent accessible audio playback.
      try {
        this.context = new AudioContext();
        await this.context.resume();
        if (generation !== this.generation) return;
        this.analyser = this.context.createAnalyser();
        this.analyser.fftSize = 1024;
        this.context.createMediaElementSource(audio).connect(this.analyser);
        this.analyser.connect(this.context.destination);
      } catch {
        this.analyser = null;
      }
      if (generation !== this.generation) return;
      audio.onplaying = () => {
        if (generation === this.generation) callbacks.started();
      };
      audio.onended = () => {
        if (generation === this.generation) {
          this.stop();
          callbacks.ended();
        }
      };
      audio.onerror = () => {
        if (generation === this.generation) {
          this.stop();
          callbacks.failed(
            'The generated voice could not play. Choose device voice or try playback again.',
          );
        }
      };
      await audio.play();
    } catch (error) {
      if (generation !== this.generation) return;
      this.stop();
      throw new Error(
        error instanceof DOMException && error.name === 'NotAllowedError'
          ? 'Your browser needs another click to play audio. Press Speak again; the prepared voice will be reused.'
          : error instanceof Error
            ? error.message
            : 'Fish Audio playback failed.',
      );
    }
  }
}
