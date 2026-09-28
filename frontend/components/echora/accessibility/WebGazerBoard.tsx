'use client';
import { useEffect, useRef } from 'react';
const messages: Record<string, string> = {
  water: 'Please bring me water.',
  help: 'I need help, please.',
  yes: 'Yes.',
  no: 'No.',
};
export default function WebGazerBoard({
  engine = 'webgazer',
  dwellMs,
  onChoose,
  onClose,
}: {
  engine?: 'webgazer' | 'gazefollower';
  dwellMs: number;
  onChoose: (text: string) => void;
  onClose: () => void;
}) {
  const frame = useRef<HTMLIFrameElement>(null);
  useEffect(() => {
    const receive = (event: MessageEvent) => {
      if (
        event.origin !== location.origin ||
        event.source !== frame.current?.contentWindow
      )
        return;
      if (event.data?.type === 'echora-gaze-stop') onClose();
      if (
        event.data?.type === 'echora-gaze-choice' &&
        typeof event.data.choice === 'string' &&
        Object.hasOwn(messages, event.data.choice)
      )
        onChoose(messages[event.data.choice]);
    };
    window.addEventListener('message', receive);
    return () => window.removeEventListener('message', receive);
  }, [onChoose, onClose]);
  return (
    <section className="webgazer-trial" aria-label="Webcam gaze choice trial">
      <iframe
        ref={frame}
        title={`${engine} calibration and large message choices`}
        src={`/accessibility/webgazer/index.html?dwell=${dwellMs}&engine=${engine}`}
        allow="camera"
      />
      <button className="webgazer-exit" onClick={onClose}>
        Exit gaze trial
      </button>
    </section>
  );
}
