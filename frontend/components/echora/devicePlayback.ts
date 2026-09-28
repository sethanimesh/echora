export function deviceSpeechLanguage(
  text: string,
  inputLanguage: string,
  outputLanguage: string,
) {
  if (/[\u0900-\u097f]/.test(text)) return 'hi-IN';
  if (outputLanguage === 'Hindi/Hinglish') return 'hi-IN';
  if (outputLanguage === 'English') return 'en-IN';
  return inputLanguage === 'hi' ? 'hi-IN' : 'en-IN';
}

/** Only the current utterance may update playback state. */
export class DevicePlayback {
  private utterance: SpeechSynthesisUtterance | null = null;

  stop() {
    if (this.utterance) {
      this.utterance.onstart = null;
      this.utterance.onend = null;
      this.utterance.onerror = null;
      this.utterance = null;
    }
    window.speechSynthesis?.cancel();
  }

  play(
    text: string,
    language: string,
    rate: number,
    callbacks: {
      started: () => void;
      ended: () => void;
      failed: () => void;
    },
  ) {
    this.stop();
    const utterance = new SpeechSynthesisUtterance(text);
    this.utterance = utterance;
    utterance.lang = language;
    utterance.rate = rate;
    utterance.onstart = () => {
      if (this.utterance === utterance) callbacks.started();
    };
    utterance.onend = () => {
      if (this.utterance !== utterance) return;
      this.utterance = null;
      callbacks.ended();
    };
    utterance.onerror = (event) => {
      if (this.utterance !== utterance) return;
      this.utterance = null;
      if (event.error === 'interrupted' || event.error === 'canceled')
        callbacks.ended();
      else callbacks.failed();
    };
    window.speechSynthesis.speak(utterance);
  }
}
