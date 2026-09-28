'use client';
import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import {
  Accessibility,
  Mic,
  Square,
  Upload,
  Volume2,
  Sparkles,
  Droplets,
  Hand,
  Heart,
  MessageCircle,
  ArrowUpRight,
  Pause,
  Play,
  SlidersHorizontal,
  Check,
  ChevronDown,
  Download,
  X,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import Globe, { type GlobeState } from '@/components/echora/Globe';
import { useRecorder } from '@/components/echora/useRecorder';
import {
  api,
  type Job,
  type RecognitionBackend,
  type RecognitionOption,
} from '@/components/echora/types';
import { AutoSpeakGate } from '@/components/echora/autoSpeakGate';
import { confirmationStillCurrent, serverRevokedPlayback } from '@/components/echora/serverPlayback';
import { startupRead } from '@/components/echora/startupRead';
import PlacesPanel from '@/components/echora/PlacesPanel';
import ContextPanel from '@/components/echora/ContextPanel';
import MemoriesPanel from '@/components/echora/MemoriesPanel';
import { rememberMessage } from '@/components/echora/memories';
import {
  emptyContext,
  scenarios,
  type ContextSelection,
  type Profile,
  type Scenario,
} from '@/components/echora/contextTypes';

import AccessPanel, {
  type InputMode,
} from '@/components/echora/accessibility/AccessPanel';
import AccessInput, {
  type PointerSample,
} from '@/components/echora/accessibility/AccessInput';
import AccessKeyboard, {
  AccessSelect,
} from '@/components/echora/accessibility/AccessKeyboard';
import CameraAccess from '@/components/echora/accessibility/CameraAccess';
import WebGazerBoard from '@/components/echora/accessibility/WebGazerBoard';
import { useAccessPreferences } from '@/components/echora/accessibility/useAccessPreferences';
import CombinedDelivery from '@/components/echora/CombinedDelivery';
import { combinedDelivery } from '@/components/echora/deliveryMerge';
import SpeechDelivery from '@/components/echora/SpeechDelivery';
import { FishPlayback } from '@/components/echora/fishPlayback';
import {
  DevicePlayback,
  deviceSpeechLanguage,
} from '@/components/echora/devicePlayback';
import {
  deliveryMatches,
  deliveryFromSuggestion,
  type Tone,
} from '@/components/echora/delivery';

const phrases = [
  { Icon: Droplets, label: 'Water', text: 'Please bring me water.' },
  { Icon: Hand, label: 'Help', text: 'I need help, please.' },
  { Icon: Heart, label: 'Company', text: 'Please sit with me.' },
  { Icon: MessageCircle, label: 'Talk', text: 'I would like to talk.' },
];
const testPhrases = [
  'Meena, please bring me water.',
  'Please do not close the window.',
  'My left leg hurts.',
  'I would like some company.',
  'Mujhe paani chahiye.',
];
type TestResult = {
  expected: string;
  transcript: string;
  judgment: string;
  metadata: Job['metadata'];
  timestamp: string;
};
export default function Home() {
  const access = useAccessPreferences();
  const [accessOpen, setAccessOpen] = useState(false);
  const [inputMode, setInputMode] = useState<InputMode>('off');
  const [tone, setTone] = useState<Tone>('neutral');
  const [voice, setVoice] = useState<'device' | 'fish'>('device');
  const [faceReady, setFaceReady] = useState(false);
  const [faceMode, setFaceMode] = useState('gemini');
  const [faceMessage, setFaceMessage] = useState('');
  const [includeFace, setIncludeFace] = useState(false);
  const recordingVideo = useRef<HTMLVideoElement>(null);
  const [fishReady, setFishReady] = useState(false);
  const [preparingVoice, setPreparingVoice] = useState(false);
  const fishPlayer = useRef<FishPlayback | null>(null);
  const devicePlayer = useRef(new DevicePlayback());
  const playbackAnalyser = useRef<AnalyserNode | null>(null);
  const [keyboardField, setKeyboardField] = useState<
    HTMLInputElement | HTMLTextAreaElement | null
  >(null);
  const cameraPointer = useRef<PointerSample>(null);
  function stopAssistance() {
    cameraPointer.current = null;
    setInputMode('off');
  }
  function changeInputMode(mode: InputMode) {
    if (['head', 'gaze', 'neural-gaze'].includes(mode)) {
      if (recorder.recording || recorder.starting) recorder.stop();
      setIncludeFace(false);
    }
    cameraPointer.current = null;
    setInputMode(mode);
    if (mode === 'head' || mode === 'gaze' || mode === 'neural-gaze')
      setAccessOpen(false);
  }

  const autoSpeak = useRef(new AutoSpeakGate());
  const [recognition, setRecognition] = useState<RecognitionBackend>('adapted');
  const [recognizers, setRecognizers] = useState<RecognitionOption[]>([]);
  const recordingContext = useRef<{
    context: ContextSelection;
    recognition: RecognitionBackend;
    language: string;
    outputLanguage: string;
    outputScript: string;
  } | null>(null);
  const [placeManualOverride, setPlaceManualOverride] = useState(false);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [memoryRefresh, setMemoryRefresh] = useState(0);
  const [context, setContext] = useState<ContextSelection>(emptyContext);
  const [contextReady, setContextReady] = useState(false);
  const [outputScript, setOutputScript] = useState<
    'auto' | 'latin' | 'devanagari'
  >('auto');
  const [job, setJob] = useState<Job | null>(null),
    [text, setText] = useState(''),
    [error, setError] = useState('');
  const [connected, setConnected] = useState(false),
    [configured, setConfigured] = useState(false),
    [pending, setPending] = useState(false);
  const [speaking, setSpeaking] = useState(false),
    [speechSupported, setSpeechSupported] = useState(false),
    [paused, setPaused] = useState(false);
  const [language, setLanguage] = useState('auto'),
    [outputLanguage, setOutputLanguage] = useState('original'),
    [settings, setSettings] = useState(false);
  const [testMode, setTestMode] = useState(false),
    [testIndex, setTestIndex] = useState(0),
    [results, setResults] = useState<TestResult[]>([]);
  const [testExpected, setTestExpected] = useState('');
  const [answer, setAnswer] = useState(''),
    [notice, setNotice] = useState(''),
    [canUpload, setCanUpload] = useState(true);
  const playbackGeneration = useRef(0);
  const playbackJob = useRef<string | null>(null);
  const active = useRef<Job | null>(null),
    lock = useRef(false),
    sequence = useRef(-1);
  const fileInput = useRef<HTMLInputElement>(null),
    editor = useRef<HTMLTextAreaElement>(null);
  const recorder = useRecorder();
  const processing =
    job?.status === 'transcribing' ||
    job?.status === 'drafting' ||
    job?.status === 'preparing' ||
    job?.status === 'synthesizing' ||
    job?.status === 'suggesting' ||
    job?.status === 'checking_face' ||
    job?.status === 'analyzing_delivery';
  const busy = pending || processing || recorder.recording || recorder.starting;
  const selectedProfile = profiles.find((profile) => profile.id === context.profile_id);
  function accept(next: Job | null, live = false, eventType = '') {
    autoSpeak.current.observe(next, live);
    if (next && active.current) {
      if (next.created_at < active.current.created_at) return;
      if (
        next.id === active.current.id &&
        next.revision < active.current.revision
      )
        return;
    }
    if (serverRevokedPlayback(active.current, next, eventType)) {
      if (playbackJob.current) stopPlayback();
      if (eventType === 'personal_context_invalidated' || next?.status === 'cancelled' ||
          (next?.ranking?.status === 'stale_context' && next.revision > (active.current?.revision ?? -1)))
        autoSpeak.current.cancel();
    }
    if (
      next &&
      active.current &&
      next.id === active.current.id &&
      next.revision === active.current.revision &&
      next.status === active.current.status &&
      next.auto_speak_revision === active.current.auto_speak_revision &&
      next.confirmed?.id === active.current.confirmed?.id &&
      next.delivery_suggestion?.id === active.current.delivery_suggestion?.id &&
      next.delivery_suggestion?.state ===
        active.current.delivery_suggestion?.state &&
      next.face_suggestion?.id === active.current.face_suggestion?.id &&
      next.face_suggestion?.state === active.current.face_suggestion?.state
    )
      return;
    active.current = next;
    setJob(next);
    setText(next?.text ?? '');
    setAnswer('');
    if (next?.context) setContext(next.context.selection);
  }
  useEffect(() => {
    let source: EventSource | undefined,
      alive = true;
    const liveGate = autoSpeak.current;
    const devicePlayback = devicePlayer.current;
    const bootstrap = new AbortController();

    startupRead<{
      job: Job | null;
      sequence: number;
      configured: boolean;
      recognition?: {
        options: RecognitionOption[];
        default_backend: RecognitionBackend;
      };
      fish?: { configured: boolean };
      delivery_cues?: { configured: boolean };
      face_cues?: { configured: boolean; mode?: string; message?: string };
    }>('/session', bootstrap.signal)
      .then((data) => {
        if (!alive) return;
        setSpeechSupported('speechSynthesis' in window);
        setConfigured(data.configured);
        setRecognizers(data.recognition?.options ?? []);
        setRecognition(data.recognition?.default_backend ?? 'whisper');
        setFishReady(data.fish?.configured ?? false);
        setFaceReady(data.face_cues?.configured ?? false);
        setFaceMode(data.face_cues?.mode ?? 'gemini');
        setFaceMessage(data.face_cues?.message ?? '');
        accept(data.job);
        if (data.job) {
          setOutputLanguage(data.job.output_language);
          setOutputScript(data.job.output_script ?? 'auto');
        }
        void startupRead<Profile[]>('/profiles', bootstrap.signal)
          .then((items) => {
            if (alive) {
              setProfiles(items);
              setContextReady(true);
            }
          })
          .catch(() => {
            if (alive)
              setError('Your profiles could not be loaded. Refresh to try again.');
          });
        sequence.current = -1;
        source = new EventSource('/api/events');
        source.onopen = () => setConnected(true);
        source.onerror = () => {
          setConnected(false);
          autoSpeak.current.cancel();
        };
        source.addEventListener('state', (event) => {
          try {
            const data = JSON.parse((event as MessageEvent).data);
            if (data.type === 'snapshot' || data.id > sequence.current) {
              sequence.current = data.id;
              accept(data.job, data.type !== 'snapshot', data.type);
            }
          } catch {
            setError(
              'A status update could not be read. Refresh to reconnect.',
            );
          }
        });
      })
      .catch(() => {
        if (alive)
          setError('Echora could not connect. Please refresh in a moment.');
      });
    return () => {
      alive = false;
      bootstrap.abort();
      liveGate.cancel();
      source?.close();
      playbackGeneration.current += 1;
      devicePlayback.stop();
      fishPlayer.current?.stop();
    };
  }, []);
  async function action(work: () => Promise<void>) {
    if (lock.current) return;
    lock.current = true;
    setPending(true);
    setError('');
    setNotice('');
    try {
      await work();
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : 'Something went wrong. Please try again.',
      );
    } finally {
      lock.current = false;
      setPending(false);
    }
  }
  function stopSpeech() {
    autoSpeak.current.cancel();
    stopPlayback();
  }
  function stopPlayback() {
    playbackGeneration.current += 1;
    playbackJob.current = null;
    devicePlayer.current.stop();
    fishPlayer.current?.stop();
    playbackAnalyser.current = null;
    setPreparingVoice(false);
    setSpeaking(false);
  }
  async function applyContext(
    next: ContextSelection,
    selectedProfile?: Profile,
  ) {
    stopSpeech();
    let old = active.current;
    if (old) {
      if (text.trim() && text.trim() !== old.text) old = await stage(text);
      accept(
        await api<Job>(`/messages/${old.id}/context`, {
          revision: old.revision,
          context: next,
        }),
      );
    }
    setContext(next);
    if (selectedProfile) setOutputLanguage(selectedProfile.language);
    else if (!next.profile_id && context.profile_id)
      setOutputLanguage('original');
    setNotice(
      'Context updated. Use “Help me phrase it” to create a fresh draft.',
    );
  }
  function changeContext(next: ContextSelection) {
    void action(() =>
      applyContext(
        next,
        next.profile_id !== context.profile_id
          ? profiles.find((p) => p.id === next.profile_id)
          : undefined,
      ),
    );
  }
  function saveProfile(profile: Profile) {
    void action(async () => {
      const saved = await api<Profile>('/profiles', profile);
      setProfiles((items) => [
        ...items.filter((p) => p.id !== saved.id),
        saved,
      ]);
      await applyContext(
        {
          ...context,
          profile_id: saved.id,
          profile_revision: saved.revision,
          moment_id: '',
          recipient: '',
          listener: 'unspecified',
          audience_id: '',
        },
        saved,
      );
      setNotice(
        'Profile saved on this computer. Your next draft will use these preferences.',
      );
    });
  }
  async function stage(value: string, resolve = false, selection = false) {
    const old = active.current;
    const next =
      old &&
      old.status !== 'cancelled' &&
      !(old.confirmed && value.trim() !== old.text)
        ? value.trim() !== old.text || resolve
          ? await api<Job>(`/messages/${old.id}/edit`, {
              revision: old.revision,
              text: value,
              selection,
            })
          : old
        : await api<Job>('/messages', { text: value });
    accept(next);
    return next;
  }
  function choose(value: string, selection = false) {
    stopSpeech();
    void action(async () => {
      const token = selection ? autoSpeak.current.arm() : null;
      const before = active.current?.revision ?? -1;
      const next = await stage(value, true, selection);
      if (token !== null) {
        autoSpeak.current.bind(token, next.id, before);
        autoSpeak.current.observe(next, true);
      }
      setText(value);
      editor.current?.focus();
    });
  }
  function chooseCandidate(candidateId: string) {
    stopSpeech();
    void action(async () => {
      const old = active.current;
      if (!old) return;
      const token = autoSpeak.current.arm();
      const next = await api<Job>(`/messages/${old.id}/choose`, {
        revision: old.revision,
        candidate_id: candidateId,
      });
      autoSpeak.current.bind(token, next.id, old.revision);
      accept(next, true);
    });
  }
  function chooseWordsAndSpeak(value: string) {
    stopSpeech();
    void action(async () => {
      const generation = playbackGeneration.current;
      const next = await stage(value, true);
      if (generation === playbackGeneration.current) await speakJob(next);
    });
  }
  function rememberCurrentMessage() {
    if (!selectedProfile || selectedProfile.sample) return;
    stopSpeech();
    void action(async () => {
      let saved = await stage(text);
      if (!saved.context) {
        saved = await api<Job>(`/messages/${saved.id}/context`, {
          revision: saved.revision, context,
        });
        accept(saved);
      }
      const profileId = saved.context?.selection.profile_id;
      if (!profileId || profileId !== selectedProfile.id)
        throw new Error('Choose the profile for this message before remembering it.');
      const result = await rememberMessage(saved, profileId);
      setMemoryRefresh((value) => value + 1);
      setNotice(result.created ? 'Message remembered for this profile.' : 'This message is already remembered.');
    });
  }
  async function refreshAfterMemoryChange() {
    stopSpeech();
    const state = await api<{ job: Job | null }>('/session');
    accept(state.job);
  }
  async function profileDeleted() {
    await refreshAfterMemoryChange();
    setProfiles(await api<Profile[]>('/profiles'));
    setContext({ ...emptyContext, scenario: context.scenario });
    setOutputLanguage('original');
    setNotice('Profile and its remembered messages deleted.');
  }
  async function upload(
    blob: Blob,
    name: string,
    frames: Blob[] = [],
    recorded = false,
  ) {
    await action(async () => {
      stopSpeech();
      const token = autoSpeak.current.arm();
      const frozen =
        recorded && recordingContext.current
          ? recordingContext.current
          : {
              context: structuredClone(context),
              recognition,
              language,
              outputLanguage,
              outputScript,
            };
      setTestExpected(testMode ? testPhrases[testIndex] : '');
      const body = new FormData();
      body.append('file', blob, name);
      body.append('language', frozen.language);
      body.append('recognition_backend', frozen.recognition);
      body.append('context', JSON.stringify(frozen.context));
      body.append('output_language', frozen.outputLanguage);
      body.append('script', frozen.outputScript);
      body.append('suggest_delivery', 'true');
      frames.forEach((frame, index) =>
        body.append('frames', frame, `snapshot-${index}.jpg`),
      );
      const next = await api<Job>('/audio', body);
      autoSpeak.current.bind(token, next.id);
      accept(next, true);
      setCanUpload(false);
    });
  }
  function stopSuggestion() {
    autoSpeak.current.cancel();
    void action(async () => {
      const current = active.current;
      if (!current?.delivery_suggestion) return;
      accept(
        await api<Job>(`/messages/${current.id}/delivery-suggestion/stop`, {
          revision: current.revision,
          suggestion_id: current.delivery_suggestion.id,
        }),
      );
    });
  }
  function stopFaceSuggestion() {
    autoSpeak.current.cancel();
    void action(async () => {
      const current = active.current;
      if (!current?.face_suggestion) return;
      accept(
        await api<Job>(`/messages/${current.id}/face-suggestion/stop`, {
          revision: current.revision,
          suggestion_id: current.face_suggestion.id,
        }),
      );
    });
  }
  function stopRecordingAnalysis() {
    autoSpeak.current.cancel();
    void action(async () => {
      const current = active.current;
      if (!current?.analysis_id) return;
      accept(
        await api<Job>(`/messages/${current.id}/recording-analysis/stop`, {
          revision: current.revision,
          analysis_id: current.analysis_id,
        }),
      );
    });
  }
  function improve() {
    void action(async () => {
      stopSpeech();
      const generation = playbackGeneration.current;
      const saved = await stage(text);
      if (generation !== playbackGeneration.current) return;
      const token = autoSpeak.current.arm();
      autoSpeak.current.bind(token, saved.id, saved.revision);
      accept(
        await api<Job>(`/messages/${saved.id}/draft`, {
          revision: saved.revision,
          output_language: outputLanguage,
          independent: saved.independent ?? false,
          script: outputScript,
          context,
        }),
        true,
      );
    });
  }
  function clarify() {
    void action(async () => {
      if (!job) return;
      stopSpeech();
      const token = autoSpeak.current.arm();
      autoSpeak.current.bind(token, job.id, job.revision);
      accept(
        await api<Job>(`/messages/${job.id}/draft`, {
          revision: job.revision,
          output_language: outputLanguage,
          context,
          answer,
          independent: job.independent ?? false,
          script: outputScript,
        }),
        true,
      );
    });
  }
  async function speakJob(saved: Job, asWritten = false, manual = false) {
    const generation = playbackGeneration.current;
    playbackJob.current = saved.id;
    const confirmed = await api<Job>(`/messages/${saved.id}/confirm`, {
      revision: saved.revision,
      output_language: manual ? outputLanguage : saved.output_language,
      context: manual ? context : (saved.context?.selection ?? context),
      pronunciation: asWritten ? 'original' : 'auto',
      delivery: { tone, rate: access.preferences.speechRate },
    });
    if (generation !== playbackGeneration.current || !confirmationStillCurrent(active.current, confirmed)) return;
    accept(confirmed);
    if (!confirmed.confirmed) return;
    if (
      !deliveryMatches(
        confirmed.confirmed.delivery,
        tone,
        access.preferences.speechRate,
      )
    ) {
      throw new Error(
        'Speech settings could not be confirmed. Nothing was spoken. Refresh and try again.',
      );
    }
    if (voice === 'fish') {
      setPreparingVoice(true);
      setNotice('Preparing your expressive voice…');
      if (!fishPlayer.current) fishPlayer.current = new FishPlayback();
      try {
        await fishPlayer.current.play(
          {
            id: confirmed.id,
            revision: confirmed.revision,
            confirmation_id: confirmed.confirmed.id,
          },
          {
            started: () => {
              playbackAnalyser.current = fishPlayer.current?.analyser ?? null;
              setPreparingVoice(false);
              setSpeaking(true);
              setNotice('Speaking with Fish Audio.');
            },
            ended: () => {
              playbackAnalyser.current = null;
              setSpeaking(false);
              setNotice('Finished speaking.');
            },
            failed: (message) => {
              playbackAnalyser.current = null;
              setPreparingVoice(false);
              setSpeaking(false);
              setError(message);
            },
          },
        );
      } finally {
        setPreparingVoice(false);
      }
      return;
    }
    if (!speechSupported) {
      setNotice(
        'Message confirmed. Speech playback is unavailable in this browser.',
      );
      return;
    }
    stopSpeech();
    playbackJob.current = confirmed.id;
    const speechText =
      confirmed.confirmed.speech_text ?? confirmed.confirmed.text;
    devicePlayer.current.play(
      speechText,
      deviceSpeechLanguage(
        speechText,
        manual ? language : saved.language,
        confirmed.output_language,
      ),
      confirmed.confirmed.delivery!.rate,
      {
        started: () => {
          setSpeaking(true);
          setNotice('Speaking your confirmed message.');
        },
        ended: () => {
          setSpeaking(false);
          setNotice('Finished speaking.');
        },
        failed: () => {
          setSpeaking(false);
          setError(
            'Your message is confirmed, but the browser could not play it. Try speaking it again.',
          );
        },
      },
    );
  }
  function confirmAndSpeak(asWritten = false) {
    stopSpeech();
    const generation = playbackGeneration.current;
    void action(async () => {
      const saved = await stage(text);
      if (generation !== playbackGeneration.current) return;
      await speakJob(saved, asWritten, true);
    });
  }
  useEffect(() => {
    if (pending || processing || lock.current) return;
    const next = active.current;
    if (autoSpeak.current.consume(next) && next)
      void action(() => speakJob(next));
  });
  function cancel() {
    stopSpeech();
    const current = active.current;
    if (!current) return;
    void api<Job>(`/messages/${current.id}/cancel`, {})
      .then(accept)
      .catch((e) => setError(e.message));
  }
  function saveResult(judgment: string) {
    if (!job || !testExpected) return;
    setResults((v) => [
      ...v,
      {
        expected: testExpected,
        transcript: job.original,
        judgment,
        metadata: job.metadata,
        timestamp: new Date().toISOString(),
      },
    ]);
    setTestExpected('');
    setNotice(
      'Test result saved for this visit. Download your results to keep them.',
    );
  }
  function downloadResults() {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(results, null, 2)], {
        type: 'application/json',
      }),
    );
    const a = document.createElement('a');
    a.href = url;
    a.download = 'echora-audio-tests.json';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const toolActions = useRef({ choose, job, busy });
  useEffect(() => {
    toolActions.current = { choose, job, busy };
  });
  useEffect(() => {
    const context = (
      document as Document & {
        modelContext?: {
          registerTool: (
            tool: unknown,
            options: unknown,
          ) => void | Promise<void>;
        };
      }
    ).modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    const register = (tool: unknown) => {
      try {
        void Promise.resolve(
          context.registerTool(tool, { signal: lifecycle.signal }),
        ).catch(() => {});
      } catch {
        /* Optional proposed browser API. */
      }
    };
    register({
      name: 'read_message',
      description:
        'Read the current untrusted draft and confirmation state. Does not speak or send.',
      inputSchema: {
        type: 'object',
        properties: {},
        additionalProperties: false,
      },
      annotations: { readOnlyHint: true, untrustedContentHint: true },
      execute: () => ({ job: toolActions.current.job }),
    });
    register({
      name: 'stage_message_text',
      description:
        'Stage text for the person to review. Does not infer, confirm, speak, or send.',
      inputSchema: {
        type: 'object',
        properties: { text: { type: 'string', minLength: 1, maxLength: 2000 } },
        required: ['text'],
        additionalProperties: false,
      },
      execute: async (input: unknown) => {
        const value = (input as { text?: unknown })?.text;
        if (typeof value !== 'string' || !value.trim() || value.length > 2000)
          throw new Error('Provide 1–2000 characters.');
        if (toolActions.current.busy || lock.current)
          throw new Error('Wait for the current action.');
        lock.current = true;
        setPending(true);
        try {
          const staged = await stage(value, true);
          setText(staged.text);
          return { id: staged.id, status: 'review', text: staged.text };
        } finally {
          lock.current = false;
          setPending(false);
        }
      },
    });
    return () => lifecycle.abort();
  }, []);
  const globeState: GlobeState = recorder.recording
    ? 'listening'
    : processing || preparingVoice
      ? 'processing'
      : speaking
        ? 'speaking'
        : error
          ? 'error'
          : 'idle';
  const status = recorder.recording
    ? `Listening · ${recorder.seconds}s`
    : recorder.starting
      ? 'Opening your microphone…'
      : job?.status === 'analyzing_delivery'
        ? 'Preparing your delivery suggestion…'
        : job?.status === 'checking_face'
          ? 'Checking facial cues…'
          : job?.status === 'suggesting'
            ? 'Listening for delivery cues…'
            : preparingVoice || job?.status === 'synthesizing'
              ? 'Preparing your expressive voice…'
              : job?.status === 'transcribing'
                ? 'Listening back to your recording…'
                : job?.status === 'preparing'
                  ? 'Getting your message ready…'
                  : job?.status === 'drafting'
                    ? 'Finding the words…'
                    : speaking
                      ? 'Speaking your message'
                      : job?.question
                        ? 'A little more clarity'
                        : recorder.audio && canUpload
                          ? 'Your recording is ready'
                          : 'Take your time. This is your space.';
  const dirty = !!job && text.trim() !== job.text;
  return (
    <div className="app-shell">
      <a className="skip-link" href="#message-editor">
        Skip to your message
      </a>
      <AccessPanel
        open={accessOpen}
        onClose={() => setAccessOpen(false)}
        preferences={access.preferences}
        update={access.update}
        mode={inputMode}
        setMode={changeInputMode}
        storageError={access.storageError}
        reset={access.reset}
      />
      <AccessInput
        mode={
          inputMode === 'gaze' || inputMode === 'neural-gaze'
            ? 'off'
            : inputMode
        }
        preferences={access.preferences}
        cameraPointer={cameraPointer}
        stop={stopAssistance}
        editField={setKeyboardField}
      />
      {keyboardField && (
        <AccessKeyboard
          field={keyboardField}
          onClose={() => setKeyboardField(null)}
        />
      )}
      <AccessSelect />
      {inputMode === 'head' && (
        <CameraAccess
          mode={inputMode}
          pointer={cameraPointer}
          stop={stopAssistance}
        />
      )}

      {(inputMode === 'gaze' || inputMode === 'neural-gaze') && (
        <WebGazerBoard
          key={inputMode}
          engine={inputMode === 'neural-gaze' ? 'gazefollower' : 'webgazer'}
          dwellMs={access.preferences.dwellMs}
          onClose={stopAssistance}
          onChoose={(value) => {
            stopAssistance();
            choose(value);
          }}
        />
      )}

      <header>
        <Link href="/" className="brand">
          echora<span>.</span>
        </Link>
        <div className="header-actions">
          <button
            className="access-open"
            onClick={() => setAccessOpen(true)}
            aria-haspopup="dialog"
          >
            <Accessibility size={19} /> Accessibility
          </button>
          <span className={`connection ${connected ? '' : 'offline'}`}>
            <i />
            {connected ? 'Connected' : 'Connecting…'}
          </span>
          <Button
            variant="ghost"
            className="icon-button"
            aria-label="Open preferences"
            aria-expanded={settings}
            onClick={() => setSettings(!settings)}
          >
            <SlidersHorizontal size={19} />
          </Button>
        </div>
      </header>
      {settings && (
        <section className="settings-panel" aria-label="Preferences">
          <label>
            Spoken language
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              disabled={busy}
            >
              <option value="auto">Detect automatically</option>
              <option value="en">English</option>
              <option value="hi">Hindi</option>
            </select>
          </label>
          <label>
            Message language
            <select
              value={outputLanguage}
              onChange={(e) => setOutputLanguage(e.target.value)}
              disabled={busy}
            >
              <option value="original">Keep my language</option>
              <option value="English">English</option>
              <option value="Hindi/Hinglish">Hindi / Hinglish</option>
            </select>
          </label>
          {outputLanguage === 'Hindi/Hinglish' && (
            <label>
              Hindi writing
              <select
                value={outputScript}
                onChange={(e) =>
                  setOutputScript(e.target.value as typeof outputScript)
                }
                disabled={busy}
              >
                <option value="auto">Match my input</option>
                <option value="latin">Hinglish · ABC</option>
                <option value="devanagari">Hindi · देवनागरी</option>
              </select>
            </label>
          )}
          <label className="check-label">
            <input
              type="checkbox"
              checked={paused}
              onChange={(e) => setPaused(e.target.checked)}
            />{' '}
            Pause globe animation
          </label>
          <span className="settings-note">
            Choose device voice or Fish Audio below your message.
          </span>
        </section>
      )}
      <main>
        <section className="voice-stage" aria-label="Record your message">
          <p className="eyebrow">YOUR VOICE. YOUR WORDS.</p>
          <h1>What would you like to say?</h1>
          <Globe
            state={globeState}
            analyser={recorder.analyser}
            playbackAnalyser={playbackAnalyser}
            paused={paused || access.preferences.reduceMotion}
          />
          <output
            className="voice-status"
            aria-live={recorder.recording ? 'off' : 'polite'}
            aria-atomic="true"
          >
            {status}
          </output>
          <div className="recording-camera-option">
            {faceMessage && <output>{faceMessage}</output>}
            <label>
              <input
                type="checkbox"
                checked={includeFace}
                disabled={
                  busy ||
                  speaking ||
                  !faceReady ||
                  ['head', 'gaze', 'neural-gaze'].includes(inputMode)
                }
                onChange={(event) => {
                  setIncludeFace(event.target.checked);
                  if (!event.target.checked) recorder.discardFaces();
                }}
              />{' '}
              Include facial cues while I speak
            </label>
            <p>
              {includeFace
                ? faceMode === 'compare'
                  ? 'Facial model trial: three snapshots are checked on this computer and by Gemini. Voice delivery cues use Gemini. No video file is recorded.'
                  : faceMode === 'local'
                    ? 'Three facial snapshots are checked on this computer only. Voice delivery cues use Gemini. No video file is recorded.'
                    : 'The camera records no video file. Three snapshots taken while you speak are submitted with your audio to Gemini for delivery cues.'
                : 'Voice only. Enable facial cues to capture snapshots during the same recording.'}
            </p>
            {['head', 'gaze', 'neural-gaze'].includes(inputMode) && (
              <p>
                Stop webcam head or gaze control before including facial cues.
              </p>
            )}
            <video
              ref={recordingVideo}
              muted
              autoPlay
              playsInline
              className="face-camera-preview"
              hidden={
                !includeFace || (!recorder.recording && !recorder.starting)
              }
              aria-label="Camera preview during your message"
            />
            {recorder.cameraNote && (
              <p aria-live="polite">{recorder.cameraNote}</p>
            )}
          </div>
          <div className="voice-controls">
            <Button
              className={`record-button ${recorder.recording ? 'recording' : ''}`}
              disabled={pending || processing || recorder.starting || speaking}
              onClick={() => {
                if (recorder.recording) {
                  recorder.stop();
                  setCanUpload(true);
                } else {
                  stopSpeech();
                  recordingContext.current = {
                    context: structuredClone(context),
                    recognition,
                    language,
                    outputLanguage,
                    outputScript,
                  };
                  void recorder.start({
                    includeFace,
                    video: recordingVideo.current,
                  });
                  setCanUpload(true);
                }
              }}
            >
              {recorder.recording ? (
                <Square size={16} fill="currentColor" />
              ) : (
                <Mic size={18} />
              )}{' '}
              {recorder.recording
                ? 'Finish recording'
                : recorder.starting
                  ? 'Opening microphone…'
                  : 'Start speaking'}
            </Button>
            <Button
              variant="ghost"
              className="quiet-button"
              disabled={busy || speaking}
              onClick={() => editor.current?.focus()}
            >
              Type instead <ArrowUpRight size={15} />
            </Button>
            <Button
              variant="ghost"
              className="icon-button"
              onClick={() => setPaused(!paused)}
              aria-label={
                paused ? 'Resume globe animation' : 'Pause globe animation'
              }
            >
              {paused ? <Play size={16} /> : <Pause size={16} />}
            </Button>
          </div>
          <label className="recognition-picker">
            Recognizer
            <select
              value={recognition}
              disabled={busy || speaking}
              onChange={(e) =>
                setRecognition(e.target.value as RecognitionBackend)
              }
            >
              {recognizers.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.label}
                  {r.ready ? '' : ' · unavailable'}
                </option>
              ))}
            </select>
            <small>
              {recognizers.find((r) => r.id === recognition)?.message ||
                (recognition === 'adapted'
                  ? 'Adapted Qwen recognizer with literal alternatives.'
                  : 'Cloud Whisper returns one transcript.')}
            </small>
          </label>
          <p className="recording-note">
            Up to 60 seconds. Pauses are welcome.
          </p>
          {recorder.url && (
            <div className="recording-review">
              {/* User-created audio has no transcript before submission; the original transcript is shown below when available. */}
              {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
              <audio
                controls
                src={recorder.url}
                aria-label="Your original recording"
              />
              <Button
                className="quiet-button"
                disabled={
                  busy ||
                  !recognizers.find((r) => r.id === recognition)?.ready ||
                  (!canUpload && !job?.error)
                }
                onClick={() => {
                  if (recorder.audio)
                    void upload(
                      recorder.audio,
                      recorder.audio.type.includes('mp4')
                        ? 'recording.m4a'
                        : 'recording.webm',
                      recorder.frames,
                      true,
                    );
                }}
              >
                {recognition === 'adapted'
                  ? 'Find my message'
                  : 'Transcribe recording'}{' '}
                <ArrowUpRight size={16} />
              </Button>
            </div>
          )}
          <div className="upload-line">
            <button disabled={busy} onClick={() => fileInput.current?.click()}>
              <Upload size={14} /> Upload an audio file
            </button>
            <span>
              {recognition === 'adapted'
                ? 'Adapted recognition keeps literal alternatives. Wording uses Groq; optional delivery cues use Gemini.'
                : 'Groq transcribes your recording. Optional delivery cues use Gemini.'}
            </span>
            <input
              ref={fileInput}
              type="file"
              accept="audio/*,.webm,.m4a,.mp4"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) {
                  recorder.clear();
                  void upload(f, f.name);
                }
                e.target.value = '';
              }}
            />
          </div>
        </section>
        <PlacesPanel
          selection={context}
          disabled={busy || speaking}
          idle={
            !busy &&
            !speaking &&
            !text.trim() &&
            !recorder.audio &&
            (!job || job.status === 'cancelled')
          }
          onChange={changeContext}
          manualOverride={placeManualOverride}
          onResume={() => setPlaceManualOverride(false)}
        />
        <ContextPanel
          key={`${context.profile_id}:${context.profile_revision}:${context.situation}`}
          profiles={profiles}
          selection={context}
          disabled={busy || speaking}
          ready={contextReady}
          onChange={(next) => {
            setPlaceManualOverride(true);
            changeContext(next);
          }}
          onSave={saveProfile}
          onTry={(value) => choose(value)}
        />
        {selectedProfile && <MemoriesPanel
          key={selectedProfile.id}
          profile={selectedProfile}
          refresh={memoryRefresh}
          disabled={busy || speaking}
          onChanged={refreshAfterMemoryChange}
          onDeleted={profileDeleted}
        />}
        {(error || recorder.error || job?.error) && (
          <div role="alert" className="error-banner">
            {error || recorder.error || job?.error?.message}
          </div>
        )}
        {!configured && connected && (
          <div className="error-banner">
            Wording and cloud transcription need a Groq key. You can still type
            and speak; adapted recognition availability is shown above.
          </div>
        )}
        <section className="composition" aria-label="Review your message">
          <details className="conversation-context">
            <summary>
              {job?.conversation && !dirty
                ? 'Following your previous message'
                : 'Conversation context'}
            </summary>
            {job?.conversation && !dirty ? (
              <blockquote>{job.conversation.text}</blockquote>
            ) : (
              <p>
                Only clear follow-ups can use your last approved message, for up
                to ten minutes in the same setting. New requests stay
                independent.
              </p>
            )}
            {job && (
              <div className="delivery-options">
                <button
                  type="button"
                  disabled={busy || speaking}
                  onClick={() =>
                    void action(async () => {
                      const saved = await stage(text);
                      accept(
                        await api<Job>(`/messages/${saved.id}/conversation`, {
                          revision: saved.revision,
                          action: 'separate',
                        }),
                      );
                      setNotice(
                        'This message is separate. Help me phrase it will use only its own words and selected profile context.',
                      );
                    })
                  }
                >
                  Keep this message separate
                </button>
                <button
                  type="button"
                  disabled={busy || speaking}
                  onClick={() =>
                    void action(async () => {
                      const saved = await stage(text);
                      accept(
                        await api<Job>(`/messages/${saved.id}/conversation`, {
                          revision: saved.revision,
                          action: 'forget',
                        }),
                      );
                      setNotice(
                        'Conversation memory cleared. Your original words are kept.',
                      );
                    })
                  }
                >
                  Forget conversation
                </button>
              </div>
            )}
          </details>
          <div className="compose-heading">
            <span className="eyebrow">
              {job?.confirmed && !dirty
                ? 'YOUR SPOKEN MESSAGE'
                : 'YOUR MESSAGE'}
            </span>
            {job?.confirmed &&
            !dirty &&
            deliveryMatches(
              job.confirmed.delivery,
              tone,
              access.preferences.speechRate,
            ) ? (
              <span className="confirmed-label">
                <Check size={14} /> Ready to speak
              </span>
            ) : (
              <span>
                A completed recommendation speaks automatically. You can choose
                another message.
              </span>
            )}
          </div>
          {job?.original && (
            <details className="evidence">
              <summary>
                {job.modality === 'audio'
                  ? 'What the recognizer heard'
                  : 'Original words'}
                {job.evidence?.model ? ` · ${job.evidence.model}` : ''}
              </summary>
              {job.evidence?.hypotheses?.length ? (
                <ol>
                  {job.evidence.hypotheses.map((hypothesis) => (
                    <li key={hypothesis.id}>
                      <p>{hypothesis.literal_text}</p>
                      {hypothesis.search_weight !== undefined && (
                        <small>
                          Relative search weight{' '}
                          {(hypothesis.search_weight * 100).toFixed(1)}% · not
                          confidence
                        </small>
                      )}
                    </li>
                  ))}
                </ol>
              ) : (
                <p>{job.original}</p>
              )}
            </details>
          )}
          {job?.ranking && <details className="speech-details">
            <summary>How the transcript was selected</summary>
            <p>{job.ranking.reason}</p>
            <p>{job.ranking.status === 'unavailable' || job.ranking.route === 'legacy'
              ? 'Audio verification was not used for this recording.'
              : job.ranking.decision === 'ambiguous'
                ? 'Competing interpretations need your choice.'
                : 'A transcript was recommended. You can still choose another reading.'}</p>
            {job.decision_source === 'user' && <p>Your explicit choice is used for this message.</p>}
          </details>}
          {!!job?.candidates?.length && !dirty && job.status !== 'cancelled' && (
            <div
              className="message-candidates"
              aria-label="Message suggestions"
            >
              {job.candidates.length > 1 && (
                  <p>
                    You can choose any alternative. Your choice speaks immediately.
                  </p>
                )}
              {job.candidates.map((candidate) => (
                <div className="candidate-card" key={candidate.id}>
                  <Button
                    variant="outline"
                    disabled={busy || speaking || !!job.options.length}
                    onClick={() => chooseCandidate(candidate.id)}
                  >
                    {candidate.text}
                  </Button>
                  {candidate.available === false && (
                    <small>
                      Literal words only. Choose to speak these exact words.
                    </small>
                  )}
                  <details>
                    <summary>Words and personal details</summary>
                    <p>Reading: {candidate.reading}</p>
                    {candidate.source_literals.map((literal, i) => (
                      <p key={i}>{literal}</p>
                    ))}
                    {candidate.specializations?.map((detail, i) => (
                      <p key={i}>
                        {detail.anchor} → {detail.surface}
                      </p>
                    ))}
                    {candidate.contextual_additions?.map((detail, i) => (
                      <p key={`addition-${i}`}>Added from your saved wording: {detail.wording || detail.surface || detail.text}</p>
                    ))}
                    {candidate.retrieved_sources?.map((source, i) => (
                      <p key={`source-${i}`}>Saved reference considered: {source.label || source.text || source.message || source.source_id || source.id || source.entry_id}</p>
                    ))}
                  </details>
                  {candidate.plain_text &&
                    candidate.plain_text !== candidate.text && (
                      <button
                        className="context-link"
                        disabled={busy || speaking}
                        onClick={() => chooseWordsAndSpeak(candidate.plain_text!)}
                      >
                        Use plain wording
                      </button>
                    )}
                </div>
              ))}
            </div>
          )}
          <Textarea
            ref={editor}
            id="message-editor"
            aria-label="Your message"
            maxLength={2000}
            placeholder="A word, a thought, anything you want to say…"
            value={text}
            disabled={processing || pending || recorder.recording || speaking}
            onChange={(e) => {
              autoSpeak.current.cancel();
              setText(e.target.value);
              setNotice('');
            }}
          />
          {(access.preferences.keyboard || inputMode !== 'off') && (
            <button
              className="access-keyboard-open"
              disabled={busy || speaking}
              onClick={() => setKeyboardField(editor.current)}
            >
              Open keyboard & phrases
            </button>
          )}
          {job?.context_trace && !dirty && (
            <div className="context-trace">
              <p>
                Draft context:{' '}
                {[
                  job.context_trace.profile_label,
                  job.context_trace.moment_title,
                  scenarios[job.context_trace.scenario as Scenario],
                  job.context_trace.recipient,
                  job.context_trace.listener === 'new'
                    ? 'New listener'
                    : job.context_trace.listener === 'familiar'
                      ? 'Familiar listener'
                      : '',
                ]
                  .filter(Boolean)
                  .join(' · ')}
              </p>
              {job.context_trace.situation && (
                <p>Situation: {job.context_trace.situation}</p>
              )}
              {job.context_trace.applied_details.map((detail, i) => (
                <p key={i}>
                  From your saved wording: {detail.anchor} →{' '}
                  <strong>{detail.wording}</strong>
                </p>
              ))}
              <button
                className="context-link"
                disabled={busy || speaking}
                onClick={() => chooseWordsAndSpeak(job.source_text)}
              >
                Use my words without enrichment
              </button>
            </div>
          )}
          {job?.prepared_speech && !dirty && !processing && (
            <details className="speech-details">
              <summary>Speech details</summary>
              <p>
                Hindi pronunciation is prepared automatically, with English
                words preserved.
              </p>
              <p className="pronunciation-preview" lang="hi">
                {job.prepared_speech.speech_text}
              </p>
            </details>
          )}
          {job?.error?.stage === 'pronunciation' && !dirty && (
            <p className="speech-fallback">
              Pronunciation could not be prepared. You can try Speak again, or{' '}
              <button
                className="context-link"
                disabled={busy || speaking}
                onClick={() => confirmAndSpeak(true)}
              >
                speak as written
              </button>
              .
            </p>
          )}
          {job?.question && (
            <div className="clarification">
              <p>{job.question}</p>
              <div className="options">
                {job.options.map((option) => (
                  <Button
                    key={option}
                    variant="outline"
                    className="option"
                    disabled={busy}
                    onClick={() => choose(option, true)}
                  >
                    {option}
                  </Button>
                ))}
              </div>
              <label className="answer-label">
                Your answer
                <input
                  value={answer}
                  onChange={(e) => setAnswer(e.target.value)}
                  placeholder="Tell us a little more…"
                  maxLength={300}
                  disabled={busy}
                />
              </label>
              <div className="clarify-actions">
                <Button
                  className="quiet-button"
                  disabled={busy || !answer.trim()}
                  onClick={clarify}
                >
                  Use this answer
                </Button>
                <Button
                  variant="ghost"
                  className="quiet-button"
                  disabled={busy}
                  onClick={() => choose(job.original)}
                >
                  Use my original words
                </Button>
              </div>
            </div>
          )}
          {text.trim() && (
            <SpeechDelivery
              tone={tone}
              voice={voice}
              fishReady={fishReady}
              onVoice={(next) => {
                stopSpeech();
                setVoice(next);
                setError('');
                setNotice('Voice selected. Speak when ready.');
              }}
              rate={access.preferences.speechRate}
              disabled={busy || speaking}
              onTone={(next) => {
                stopSpeech();
                setTone(next);
                setNotice('Tone selected. Speak to use this delivery choice.');
              }}
              onRate={(speechRate) => {
                stopSpeech();
                access.update({ speechRate });
                setNotice('Speaking pace updated. Speak to hear it.');
              }}
            >
              <CombinedDelivery
                key={`${job?.id}:${job?.revision}:${job?.delivery_suggestion?.id}:${job?.face_suggestion?.id}`}
                voice={
                  !dirty && job?.delivery_suggestion?.revision === job?.revision
                    ? job?.delivery_suggestion
                    : null
                }
                face={
                  !dirty && job?.face_suggestion?.revision === job?.revision
                    ? job?.face_suggestion
                    : null
                }
                disabled={busy || speaking}
                currentRate={access.preferences.speechRate}
                onAccept={() => {
                  const current = active.current;
                  if (!current || dirty || busy) return;
                  const voice =
                    current.delivery_suggestion?.revision === current.revision
                      ? current.delivery_suggestion
                      : null;
                  const face =
                    current.face_suggestion?.revision === current.revision
                      ? current.face_suggestion
                      : null;
                  const combined = combinedDelivery(
                    voice,
                    face,
                    access.preferences.speechRate,
                  );
                  if (!combined.available) return;
                  const next = deliveryFromSuggestion(
                    {
                      state: 'ready',
                      tone: combined.tone,
                      rate: combined.rate,
                    },
                    tone,
                    access.preferences.speechRate,
                  );
                  if (!next) return;
                  stopSpeech();
                  setTone(next.tone);
                  access.update({ speechRate: next.rate });
                  setNotice('Delivery selected. Speak when ready.');
                }}
              />
            </SpeechDelivery>
          )}
          <div className="compose-actions">
            <Button
              variant="ghost"
              className="quiet-button"
              disabled={busy || speaking || !text.trim() || !selectedProfile || selectedProfile.sample || !!(job?.question && !dirty)}
              onClick={rememberCurrentMessage}
              title={selectedProfile?.sample ? 'Save your own profile copy before remembering messages.' : !selectedProfile ? 'Choose a profile to remember this message.' : 'Save these exact words for this profile without speaking.'}
            >Remember</Button>
            <Button
              variant="ghost"
              className="quiet-button"
              disabled={busy || !text.trim() || !configured || speaking}
              onClick={improve}
            >
              <Sparkles size={16} /> Help me phrase it
            </Button>
            <div className="speak-actions">
              {preparingVoice || job?.status === 'synthesizing' ? (
                <Button
                  className="quiet-button"
                  onClick={() => {
                    stopSpeech();
                    setNotice('Voice preparation stopped.');
                  }}
                >
                  <Square size={16} /> Stop voice preparation
                </Button>
              ) : job?.status === 'analyzing_delivery' ? (
                <Button
                  className="quiet-button"
                  onClick={stopRecordingAnalysis}
                >
                  <X size={16} /> Stop delivery analysis
                </Button>
              ) : job?.status === 'checking_face' ? (
                <Button className="quiet-button" onClick={stopFaceSuggestion}>
                  <X size={16} /> Stop camera analysis
                </Button>
              ) : job?.status === 'suggesting' ? (
                <Button className="quiet-button" onClick={stopSuggestion}>
                  <X size={16} /> Stop voice analysis
                </Button>
              ) : processing || pending ? (
                <Button className="quiet-button" onClick={cancel}>
                  <X size={16} /> Cancel
                </Button>
              ) : speaking ? (
                <Button
                  className="speak-button"
                  onClick={() => {
                    stopSpeech();
                    setNotice('Speech stopped.');
                  }}
                >
                  <Square size={16} /> Stop speaking
                </Button>
              ) : (
                <Button
                  className="speak-button"
                  disabled={busy || !text.trim() || !!(job?.question && !dirty)}
                  onClick={() => confirmAndSpeak()}
                >
                  <Volume2 size={17} />
                  {job?.confirmed &&
                  !dirty &&
                  deliveryMatches(
                    job.confirmed.delivery,
                    tone,
                    access.preferences.speechRate,
                  )
                    ? 'Speak again'
                    : 'Speak'}
                </Button>
              )}
            </div>
          </div>
          <output
            className="message-notice"
            aria-live="polite"
            aria-atomic="true"
          >
            {notice ||
              (processing
                ? 'Your original words stay unchanged while processing.'
                : 'Speaking plays on this device. Caregiver delivery comes later.')}
          </output>
        </section>
        <section className="quick-phrases" aria-label="Quick phrases">
          {phrases.map(({ Icon, label, text }) => (
            <button
              key={label}
              disabled={busy || speaking}
              onClick={() => choose(text)}
            >
              <Icon size={18} />
              <span>{label}</span>
            </button>
          ))}
        </section>
        <section className="test-section">
          <button
            className="test-toggle"
            aria-expanded={testMode}
            onClick={() => setTestMode(!testMode)}
          >
            <span>Let’s test your voice</span>
            <ChevronDown size={16} className={testMode ? 'rotated' : ''} />
          </button>
          {testMode && (
            <div className="test-content">
              <p>
                Record one phrase naturally, then select “Transcribe & suggest
                delivery”. Compare the original transcript before improving the
                wording.
              </p>
              <label>
                Try this phrase
                <select
                  disabled={busy}
                  value={testIndex}
                  onChange={(e) => setTestIndex(Number(e.target.value))}
                >
                  {testPhrases.map((p, i) => (
                    <option key={p} value={i}>
                      {i + 1}. {p}
                    </option>
                  ))}
                </select>
              </label>
              <p className="test-prompt">“{testPhrases[testIndex]}”</p>
              <p className="test-note">
                Say the words in your usual voice. No need to imitate a speech
                difficulty. The expected phrase is never sent to the model.
              </p>
              {job?.modality === 'audio' && job.original && testExpected && (
                <div className="test-score">
                  <span>Did it preserve what you said?</span>
                  <Button
                    className="quiet-button"
                    disabled={busy}
                    onClick={() => saveResult('Meaning preserved')}
                  >
                    <Check size={15} /> Yes
                  </Button>
                  <Button
                    className="quiet-button"
                    disabled={busy}
                    onClick={() =>
                      saveResult('Meaning changed or words missing')
                    }
                  >
                    Needs correction
                  </Button>
                </div>
              )}
              {Object.entries(job?.metadata ?? {}).map(([stage, meta]) => (
                <p className="model-detail" key={stage}>
                  {stage === 'transcription'
                    ? 'Transcription'
                    : stage === 'pronunciation'
                      ? 'Pronunciation'
                      : 'Wording'}{' '}
                  · {meta.model} · {(meta.elapsed_ms / 1000).toFixed(2)}s
                </p>
              ))}
              {results.length > 0 && (
                <Button
                  variant="ghost"
                  className="quiet-button"
                  onClick={downloadResults}
                >
                  <Download size={16} /> Download {results.length} test{' '}
                  {results.length === 1 ? 'result' : 'results'}
                </Button>
              )}
            </div>
          )}
        </section>
      </main>
      <footer>
        <span>A space to express yourself.</span>
        <a
          href="https://github.com/brunosimon/organic-sphere"
          target="_blank"
          rel="noreferrer"
        >
          Sphere inspired by Bruno Simon <ArrowUpRight size={12} />
        </a>
      </footer>
    </div>
  );
}
