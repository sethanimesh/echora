import { StatusBar } from "expo-status-bar";
import {
  AudioModule,
  RecordingPresets,
  setAudioModeAsync,
  useAudioPlayer,
  useAudioPlayerStatus,
  useAudioRecorder,
  useAudioRecorderState,
} from "expo-audio";
import * as Clipboard from "expo-clipboard";
import * as DocumentPicker from "expo-document-picker";
import { File, Paths } from "expo-file-system";
import * as Location from "expo-location";
import * as Speech from "expo-speech";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  AppState,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from "react-native";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";

import {
  API_URL,
  cancelActiveMessage,
  chooseMessage,
  clearMemories,
  deleteMemory,
  deleteProfile,
  editMessage,
  readHealth,
  readPersonas,
  readPlaces,
  readProfile,
  readMemories,
  refreshActiveMessage,
  rememberActiveMessage,
  replaceProfile,
  resolveAudience,
  savePlaces,
  synthesizeSpeech,
  transcribeAudio,
  watchMessageInvalidation,
} from "./src/api";
import { Orb } from "./src/components/Orb";
import { Sheet } from "./src/components/Sheet";
import { isTagged, nearestPlace } from "./src/location";
import { colors } from "./src/theme";
import type {
  AudienceProfile,
  CommunicationStyle,
  CommunicationContext,
  Health,
  Listener,
  LexiconEntry,
  MessageCandidate,
  MemoryList,
  PersonaSummary,
  PersonaProfile,
  Place,
  PlaceSettings,
  ResolvedAudience,
  SpecializationRule,
  SpeechAudio,
  Transcription,
} from "./src/types";

type Phase = "idle" | "recording" | "working" | "ready" | "error";
type Stage = "idle" | "recording" | "working" | "choosing" | "composing" | "error";
type Overlay = "evidence" | "about" | "persona" | "settings" | null;

const contexts: { value: CommunicationContext; label: string; hint: string }[] = [
  { value: "general", label: "General", hint: "No setting assumptions" },
  { value: "home", label: "At home", hint: "Everyday needs and household help" },
  { value: "care", label: "Care", hint: "Symptoms, comfort, and assistance" },
  { value: "outdoors", label: "Outdoors", hint: "Travel, safety, and nearby places" },
];

const LISTENER_DEFAULTS: Record<CommunicationContext, Listener> = {
  general: "familiar",
  home: "familiar",
  care: "familiar",
  outdoors: "unfamiliar",
};

const MAX_SECONDS = 45;
const IDLE_CHIPS = 5;
const RECORDING_OPTIONS = {
  // Keep the native format Expo validates on each iOS release. The API already
  // mixes and resamples every accepted format to 16 kHz mono in audio.py, so
  // forcing the phone's AAC encoder to that rate only creates a failure point.
  ...RecordingPresets.HIGH_QUALITY,
  isMeteringEnabled: true,
};

async function waitForActiveApp(timeoutMs = 2_000) {
  if (AppState.currentState === "active") return;
  await new Promise<void>((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      subscription.remove();
      resolve();
    };
    const subscription = AppState.addEventListener("change", (state) => {
      if (state === "active") finish();
    });
    const timer = setTimeout(finish, timeoutMs);
  });
}

function behaviourLabel(context: CommunicationContext) {
  return contexts.find((item) => item.value === context)?.label || context;
}

function listenerLabel(context: CommunicationContext) {
  return LISTENER_DEFAULTS[context] === "unfamiliar"
    ? "people who do not know me"
    : "people who know me";
}

function formatBackend(value?: string) {
  if (value === "local") return "Local Mac";
  if (value === "pod") return "RunPod Pod";
  if (value === "runpod") return "RunPod Serverless";
  return "Connecting";
}

function groupedWeight(result: Transcription, candidate: MessageCandidate) {
  return candidate.source_hypothesis_ids.reduce(
    (total, id) => total + (result.hypotheses.find((item) => item.id === id)?.search_weight || 0),
    0,
  );
}

function audioLevel(metering?: number) {
  if (typeof metering !== "number" || !Number.isFinite(metering)) return 0.15;
  // Expo reports dBFS: -160 is silence and 0 is full scale.
  return Math.min(1, Math.max(0.08, (metering + 55) / 55));
}

function decodeBase64(value: string) {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const clean = value.replace(/[^A-Za-z0-9+/=]/g, "");
  const padding = clean.endsWith("==") ? 2 : clean.endsWith("=") ? 1 : 0;
  const bytes = new Uint8Array(Math.max(0, (clean.length * 3) / 4 - padding));
  let output = 0;
  for (let index = 0; index < clean.length; index += 4) {
    const a = alphabet.indexOf(clean[index] || "A");
    const b = alphabet.indexOf(clean[index + 1] || "A");
    const c = clean[index + 2] === "=" ? 0 : alphabet.indexOf(clean[index + 2] || "A");
    const d = clean[index + 3] === "=" ? 0 : alphabet.indexOf(clean[index + 3] || "A");
    const block = (a << 18) | (b << 12) | (c << 6) | d;
    if (output < bytes.length) bytes[output++] = (block >> 16) & 255;
    if (output < bytes.length) bytes[output++] = (block >> 8) & 255;
    if (output < bytes.length) bytes[output++] = block & 255;
  }
  return bytes;
}

function AppContent() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<Transcription | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [message, setMessage] = useState("");
  const [spokenText, setSpokenText] = useState("");
  const [speaking, setSpeaking] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState("");
  const [context, setContext] = useState<CommunicationContext>("general");
  // Null means the place abstains. It must stay null all the way to the API so
  // the selected profile still gets a chance to decide the listener.
  const [listener, setListener] = useState<Listener | null>(null);
  const [place, setPlace] = useState("");
  const [placeSettings, setPlaceSettings] = useState<PlaceSettings | null>(null);
  const [detected, setDetected] = useState<{ id: string; label: string; meters: number } | null>(null);
  const [personas, setPersonas] = useState<PersonaSummary[]>([]);
  const [persona, setPersona] = useState("");
  const [profile, setProfile] = useState<PersonaProfile | null>(null);
  const [audienceId, setAudienceId] = useState("");
  const [resolvedAudience, setResolvedAudience] = useState<ResolvedAudience | null>(null);
  const [overlay, setOverlay] = useState<Overlay>(null);
  const [remembering, setRemembering] = useState(false);
  const [memoryNotice, setMemoryNotice] = useState('');
  const [memoryRefresh, setMemoryRefresh] = useState(0);

  const recorder = useAudioRecorder(RECORDING_OPTIONS);
  const recorderState = useAudioRecorderState(recorder, 100);
  const player = useAudioPlayer(null, { updateInterval: 100 });
  const playerStatus = useAudioPlayerStatus(player);
  const manualPlaceRef = useRef(false);
  const finishingRef = useRef(false);
  const cancelledRef = useRef(false);
  const speechFileRef = useRef<File | null>(null);
  const speechGenerationRef = useRef(0);
  const flowGenerationRef = useRef(0);
  const requestContextRef = useRef({
    context: "general" as CommunicationContext,
    persona: "",
    listener: null as Listener | null,
    audience: "",
    place: "",
  });

  const stage: Stage = phase === "ready" ? (selectedId ? "composing" : "choosing") : phase;
  const elapsed = Math.floor(recorderState.durationMillis / 1000);
  const level = audioLevel(recorderState.metering);
  const activePersona = useMemo(
    () => personas.find((item) => item.id === persona) || null,
    [personas, persona],
  );
  const selected = useMemo(
    () => result?.messages.find((candidate) => candidate.message_id === selectedId) || null,
    [result, selectedId],
  );
  const edited = message.trim() !== spokenText;
  const modelReady = Boolean(health?.model_ready) && !healthError;
  const visibleAudiences = useMemo(
    () =>
      (profile?.audiences || []).filter((item) => {
        const unscoped = item.visible_in_places.length === 0 && item.visible_in_settings.length === 0;
        return unscoped || item.visible_in_places.includes(place) || item.visible_in_settings.includes(context);
      }),
    [context, place, profile],
  );

  const stopSpeaking = useCallback(() => {
    speechGenerationRef.current += 1;
    player.pause();
    void Speech.stop();
    setSpeaking(false);
  }, [player]);

  const deviceSpeak = useCallback((text: string, generation: number) => {
    setSpeaking(true);
    // Wait for the previous utterance to leave the native queue before adding
    // this one, otherwise a late stop can cancel the fallback it enabled.
    void Speech.stop().finally(() => {
      if (generation !== speechGenerationRef.current) return;
      Speech.speak(text, {
        rate: 0.92,
        onDone: () => { if (generation === speechGenerationRef.current) setSpeaking(false); },
        onStopped: () => { if (generation === speechGenerationRef.current) setSpeaking(false); },
        onError: () => { if (generation === speechGenerationRef.current) setSpeaking(false); },
      });
    });
  }, []);

  const playClip = useCallback(
    async (clip: SpeechAudio, generation: number) => {
      await setAudioModeAsync({
        allowsRecording: false,
        playsInSilentMode: true,
      });
      if (generation !== speechGenerationRef.current) return;
      const previous = speechFileRef.current;
      if (previous?.exists) previous.delete();
      const suffix = clip.media_type.includes("wav") ? "wav" : "audio";
      const file = new File(Paths.cache, `echora-speech-${Date.now()}.${suffix}`);
      file.write(decodeBase64(clip.audio_base64));
      speechFileRef.current = file;
      player.replace(file.uri);
      player.play();
    },
    [player],
  );

  const speak = useCallback(
    async (text: string, _clip?: SpeechAudio | null) => {
      const spoken = text.trim();
      if (!spoken) return;
      stopSpeaking();
      const generation = speechGenerationRef.current;
      setSpeaking(true);
      try {
        // The same server decision used by the web client authorizes this exact
        // revision and records its ten-minute reference before any voice plays.
        const authorized = await synthesizeSpeech(spoken);
        if (generation !== speechGenerationRef.current) return;
        setSpokenText(spoken);
        if (!authorized.speech) {
          deviceSpeak(authorized.speech_text, generation);
          return;
        }
        try {
          await playClip(authorized.speech, generation);
        } catch {
          if (generation === speechGenerationRef.current) deviceSpeak(authorized.speech_text, generation);
        }
      } catch (caught) {
        // A stale/cancelled decision cannot fall through to an unbound voice.
        if (generation === speechGenerationRef.current) {
          setSpeaking(false);
          Alert.alert("Message not spoken", caught instanceof Error ? caught.message : "This message could not be authorized for speech.");
        }
      }
    },
    [deviceSpeak, playClip, stopSpeaking],
  );

  useEffect(() => {
    if (playerStatus.didJustFinish) setSpeaking(false);
  }, [playerStatus.didJustFinish]);

  useEffect(() => {
    if (!result?.request_id || phase !== 'ready') return;
    return watchMessageInvalidation((next) => {
      stopSpeaking();
      flowGenerationRef.current += 1;
      setResult(next);
      setSelectedId(next?.recommended_message_id ?? null);
      setMessage(next?.messages.find((item) => item.message_id === next.recommended_message_id)?.corrected_text || '');
      setMemoryNotice('Personal context changed. Review the current words before speaking.');
      if (!next) setPhase('idle');
    }, stopSpeaking);
  }, [result?.request_id, phase, stopSpeaking]);

  useEffect(() => {
    let alive = true;
    const check = async () => {
      try {
        const next = await readHealth();
        if (alive) {
          setHealth(next);
          setHealthError(false);
        }
      } catch {
        if (alive) setHealthError(true);
      }
    };
    void check();
    const timer = setInterval(check, 15_000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    readPersonas().then(setPersonas).catch(() => setPersonas([]));
    readPlaces().then(setPlaceSettings).catch(() => setPlaceSettings(null));
  }, []);

  useEffect(() => {
    let cancelled = false;
    setAudienceId("");
    if (!persona) {
      setProfile(null);
      return;
    }
    readProfile(persona)
      .then((next) => {
        if (!cancelled) setProfile(next);
      })
      .catch(() => {
        if (!cancelled) setProfile(null);
      });
    return () => {
      cancelled = true;
    };
  }, [persona]);

  useEffect(() => {
    let cancelled = false;
    setResolvedAudience(null);
    resolveAudience({
      context,
      persona,
      audience: audienceId,
      declared_listener: listener,
      place_id: place,
    })
      .then((next) => {
        if (!cancelled) setResolvedAudience(next);
      })
      .catch(() => {
        if (!cancelled) setResolvedAudience(null);
      });
    return () => {
      cancelled = true;
    };
  }, [audienceId, context, listener, persona, place]);

  useEffect(() => {
    if (audienceId && !visibleAudiences.some((item) => item.id === audienceId)) {
      setAudienceId("");
    }
  }, [audienceId, visibleAudiences]);

  useEffect(() => {
    if (phase !== "working") return;
    const timer = setInterval(() => setProgress((value) => Math.min(3, value + 1)), 1800);
    return () => clearInterval(timer);
  }, [phase]);

  useEffect(() => {
    if (stage !== "idle" || !placeSettings?.auto_detect || manualPlaceRef.current) return;
    if (!placeSettings.places.some(isTagged)) return;
    let cancelled = false;
    const detect = async () => {
      try {
        const permission = await Location.requestForegroundPermissionsAsync();
        if (!permission.granted) return;
        const position = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.High });
        if (cancelled || manualPlaceRef.current) return;
        const found = nearestPlace(
          placeSettings.places,
          position.coords.latitude,
          position.coords.longitude,
          position.coords.accuracy ?? 0,
        );
        if (!found) {
          setDetected(null);
          return;
        }
        setDetected({ id: found.place.id, label: found.place.label, meters: Math.round(found.meters) });
        setPlace(found.place.id);
        setContext(found.place.context);
        setListener(found.place.listener);
        setAudienceId("");
      } catch {
        if (!cancelled) setDetected(null);
      }
    };
    void detect();
    return () => {
      cancelled = true;
    };
  }, [placeSettings, stage]);

  useEffect(() => {
    if (phase === "recording" && elapsed >= MAX_SECONDS && !finishingRef.current) {
      void finishRecording();
    }
  }, [elapsed, phase]);

  useEffect(
    () => () => {
      speechGenerationRef.current += 1;
      flowGenerationRef.current += 1;
      void cancelActiveMessage();
      player.pause();
      void Speech.stop();
      if (speechFileRef.current?.exists) speechFileRef.current.delete();
    },
    [player],
  );

  async function submitAudio(uri: string, filename: string) {
    const generation = ++flowGenerationRef.current;
    setPhase("working");
    setProgress(0);
    setError("");
    setSpokenText("");
    try {
      const frozen = requestContextRef.current;
      const next = await transcribeAudio(
        uri,
        filename,
        frozen.context,
        frozen.persona,
        frozen.listener,
        frozen.audience,
        frozen.place,
      );
      if (generation !== flowGenerationRef.current) return;
      setResult(next);
      const initial = next.ranker.decision === "selected" ? next.recommended_message_id : null;
      const initialMessage = next.messages.find((candidate) => candidate.message_id === initial);
      setSelectedId(initial);
      setMessage(initialMessage?.corrected_text || "");
      setPhase("ready");
      // Recommendation and permission to play speech are separate server decisions.
      if (initialMessage && next.auto_speak) {
        void speak(initialMessage.corrected_text, next.speech);
      }
    } catch (caught) {
      if (generation !== flowGenerationRef.current) return;
      setError(caught instanceof Error ? caught.message : "Something went wrong while transcribing");
      setPhase("error");
    }
  }

  async function startRecording() {
    flowGenerationRef.current += 1;
    void cancelActiveMessage();
    setError("");
    setResult(null);
    setSelectedId(null);
    setSpokenText("");
    setCopied(false);
    stopSpeaking();
    requestContextRef.current = { context, persona, listener, audience: audienceId, place };
    try {
      const permission = await AudioModule.requestRecordingPermissionsAsync();
      if (!permission.granted) {
        setError("Microphone access was denied. You can still choose an audio file.");
        setPhase("error");
        return;
      }
      // The first permission sheet briefly makes an Expo experience inactive.
      // Activating AVAudioSession during that handoff can make iOS refuse the
      // recorder even though permission was just granted.
      await waitForActiveApp();
      await setAudioModeAsync({
        allowsRecording: true,
        playsInSilentMode: true,
      });
      cancelledRef.current = false;
      finishingRef.current = false;
      await recorder.prepareToRecordAsync();
      recorder.record();
      setPhase("recording");
    } catch (caught) {
      const detail = caught instanceof Error ? caught.message : String(caught);
      console.error("Echora could not start the iOS recorder", caught);
      setError(
        detail
          ? `The microphone could not start. iPhone reported: ${detail}`
          : "The microphone could not start. You can still choose an audio file.",
      );
      setPhase("error");
    }
  }

  async function finishRecording() {
    if (finishingRef.current) return;
    finishingRef.current = true;
    try {
      await recorder.stop();
      await setAudioModeAsync({
        allowsRecording: false,
        playsInSilentMode: true,
      });
      const uri = recorder.uri || recorderState.url;
      if (!cancelledRef.current && uri) await submitAudio(uri, "recording.m4a");
      else if (!cancelledRef.current) throw new Error("The recording file was not available");
    } catch (caught) {
      if (!cancelledRef.current) {
        setError(caught instanceof Error ? caught.message : "The recording could not be finished");
        setPhase("error");
      }
    } finally {
      finishingRef.current = false;
    }
  }

  async function cancelRecording() {
    cancelledRef.current = true;
    if (!finishingRef.current) {
      finishingRef.current = true;
      await recorder.stop().catch(() => undefined);
      finishingRef.current = false;
    }
    setPhase("idle");
  }

  async function chooseAudioFile() {
    stopSpeaking();
    flowGenerationRef.current += 1;
    void cancelActiveMessage();
    requestContextRef.current = { context, persona, listener, audience: audienceId, place };
    try {
      const picked = await DocumentPicker.getDocumentAsync({
        type: ["audio/*", "video/mp4"],
        multiple: false,
        copyToCacheDirectory: true,
      });
      if (!picked.canceled && picked.assets[0]) {
        const asset = picked.assets[0];
        await submitAudio(asset.uri, asset.name || "recording.m4a");
      }
    } catch {
      setError("The audio file could not be opened.");
      setPhase("error");
    }
  }

  async function choose(candidate: MessageCandidate) {
    if (!result) return;
    stopSpeaking();
    const generation = ++flowGenerationRef.current;
    try {
      const next = await chooseMessage(candidate.message_id);
      if (generation !== flowGenerationRef.current) return;
      const selected = next.messages.find((item) => item.message_id === next.recommended_message_id)
        || next.messages.find((item) => item.message_id === candidate.message_id);
      setResult(next);
      setSelectedId(selected?.message_id || null);
      setMessage(selected?.corrected_text || '');
      setCopied(false);
      if (selected && next.auto_speak) void speak(selected.corrected_text);
    } catch (caught) {
      if (generation === flowGenerationRef.current) setError(caught instanceof Error ? caught.message : "This alternative is no longer available.");
    }
  }

  function revert(candidate: MessageCandidate) {
    if (!candidate.plain_text) return;
    setMessage(candidate.plain_text);
    void speak(candidate.plain_text);
  }

  function pickPlace(next: string) {
    manualPlaceRef.current = true;
    setDetected(null);
    setPlace(next);
    const chosen = placeSettings?.places.find((item) => item.id === next);
    setContext(chosen?.context || "general");
    setListener(chosen?.listener ?? null);
    setAudienceId("");
  }

  function reset() {
    flowGenerationRef.current += 1;
    void cancelActiveMessage();
    stopSpeaking();
    setResult(null);
    setSelectedId(null);
    setMessage("");
    setSpokenText("");
    setSpeaking(false);
    setCopied(false);
    setError("");
    setOverlay(null);
    setPhase("idle");
  }

  function pickPersona(next: string) {
    setPersona(next);
    setAudienceId("");
    const picked = personas.find((item) => item.id === next);
    if (!picked || !next || manualPlaceRef.current || detected) return;
    const match = placeSettings?.places.find((item) => item.context === picked.context_default);
    setPlace(match?.id || "");
    setContext(picked.context_default);
    // Only carry what the matching place declared; the server applies the
    // profile's listener-by-setting itself.
    setListener(match?.listener ?? null);
  }

  async function copyMessage() {
    if (!message.trim()) return;
    await Clipboard.setStringAsync(message.trim());
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }

  async function remember() {
    if (remembering || !message.trim()) return;
    const generation = flowGenerationRef.current;
    setRemembering(true);
    setMemoryNotice('');
    try {
      const saved = await rememberActiveMessage(message);
      if (generation !== flowGenerationRef.current) return;
      setMemoryNotice(saved.created ? 'Message remembered for this profile.' : 'This message is already remembered.');
      setMemoryRefresh((value) => value + 1);
    } catch (caught) {
      if (generation === flowGenerationRef.current)
        setMemoryNotice(caught instanceof Error ? caught.message : 'This message could not be remembered.');
    } finally { setRemembering(false); }
  }

  async function memoryChanged() {
    stopSpeaking();
    const generation = ++flowGenerationRef.current;
    const next = await refreshActiveMessage();
    if (generation !== flowGenerationRef.current) return;
    setMemoryNotice('');
    if (!next) {
      setResult(null);
      setSelectedId(null);
      setMessage('');
      setPhase('idle');
    } else {
      setResult(next);
      const id = next.recommended_message_id;
      setSelectedId(id);
      setMessage(next.messages.find((item) => item.message_id === id)?.corrected_text || '');
    }
  }
  async function profileDeleted() {
    await memoryChanged();
    setProfile(null);
    setPersona('');
    setPersonas(await readPersonas());
  }

  const progressLabel = [
    "Listening closely to the recording",
    "Finding literal speech candidates",
    "Comparing the available evidence",
    "Preparing clear wording",
  ][progress];

  const liveText =
    stage === "recording"
      ? "Listening"
      : stage === "working"
        ? "Working on your message"
        : stage === "choosing"
          ? `${result?.messages.length || 0} options ready. Which one sounds right?`
          : stage === "composing"
            ? speaking
              ? "Speaking your message"
              : edited
                ? "Message edited and ready to speak"
                : "Message spoken"
            : stage === "error"
              ? "Something went wrong"
              : "Ready to record";

  return (
    <SafeAreaView style={styles.safe} edges={["top", "bottom"]}>
      <StatusBar style="dark" />
      <View style={styles.shell}>
        <Header
          health={health}
          healthError={healthError}
          activePersona={activePersona}
          hasResult={Boolean(result)}
          onEvidence={() => setOverlay("evidence")}
          onReset={reset}
          onPersona={() => setOverlay("persona")}
          onSettings={() => setOverlay("settings")}
          onAbout={() => setOverlay("about")}
          showPersona={personas.length > 0}
        />

        <View style={styles.stage}>
          {stage === "idle" && (
            <IdleStage
              modelReady={modelReady && Boolean(resolvedAudience)}
              detail={
                modelReady && !resolvedAudience
                  ? "Resolving who you are speaking to"
                  : health?.detail
              }
              place={place}
              places={placeSettings?.places || []}
              detected={detected}
              audiences={visibleAudiences}
              audienceId={audienceId}
              resolvedAudience={resolvedAudience}
              onPickPlace={pickPlace}
              onPickAudience={setAudienceId}
              onSettings={() => setOverlay("settings")}
              onRecord={() => void startRecording()}
              onPickFile={() => void chooseAudioFile()}
            />
          )}
          {stage === "recording" && (
            <RecordingStage
              elapsed={elapsed}
              level={level}
              onStop={() => void finishRecording()}
              onCancel={() => void cancelRecording()}
            />
          )}
          {stage === "working" && <WorkingStage label={progressLabel} progress={progress} />}
          {stage === "choosing" && result && <ChooseStage result={result} onChoose={choose} />}
          {stage === "composing" && result && selected && (
            <ComposeStage
              result={result}
              selectedId={selectedId}
              message={message}
              speaking={speaking}
              edited={edited}
              copied={copied}
              remembering={remembering}
              memoryNotice={memoryNotice}
              canRemember={Boolean(result.persona && profile && !profile.baseline && result.persona === profile.id)}
              onRemember={() => void remember()}
              onChoose={choose}
              onCompare={() => {
                stopSpeaking();
                setSelectedId(null);
                setSpokenText("");
              }}
              onEdit={(value) => {
                stopSpeaking();
                setMessage(value);
                setMemoryNotice('');
                void editMessage(value).catch(() => setError("This edit could not be saved. Try speaking again."));
              }}
              onCopy={() => void copyMessage()}
              onSpeak={() => void speak(message)}
              onEvidence={() => setOverlay("evidence")}
              onRevert={revert}
            />
          )}
          {stage === "error" && <ErrorStage message={error} onRetry={reset} />}
        </View>

        <Text accessibilityLiveRegion="polite" style={styles.srOnly}>{liveText}</Text>
        <View style={styles.footer}>
          <Text>Your chosen message is spoken as soon as it is ready.</Text>
          <Text>A follow-up stays for 10 minutes. Remember saves only the messages you choose.</Text>
        </View>

        <Sheet open={overlay === "evidence"} title="Literal ASR evidence" onClose={() => setOverlay(null)}>
          {result && <EvidenceBody result={result} />}
        </Sheet>
        <Sheet open={overlay === "persona"} title="Who is speaking?" onClose={() => setOverlay(null)}>
          <PersonaBody
            personas={personas}
            persona={persona}
            profile={profile}
            places={placeSettings?.places || []}
            memoryRefresh={memoryRefresh}
            onMemoriesChanged={memoryChanged}
            onProfileDeleted={profileDeleted}
            onMemoryMutation={stopSpeaking}
            onPick={pickPersona}
            onProfileSaved={(next) => {
              setProfile(next);
              setPersona(next.id);
              void readPersonas().then(setPersonas);

            }}
          />
        </Sheet>
        <Sheet open={overlay === "settings"} title="Places" onClose={() => setOverlay(null)}>
          <SettingsBody
            settings={placeSettings}
            onSaved={(next) => {
              setPlaceSettings(next);
              const active = next.places.find((item) => item.id === place);
              if (place && !active) {
                setPlace("");
                setContext("general");
                setListener(null);
                setAudienceId("");
              } else if (active) {
                setContext(active.context);
                setListener(active.listener);
              }
              if (!next.auto_detect) setDetected(null);
            }}
          />
        </Sheet>
        <Sheet open={overlay === "about"} title="About Echora" onClose={() => setOverlay(null)}>
          <AboutBody />
        </Sheet>
      </View>
    </SafeAreaView>
  );
}

export default function App() {
  return (
    <SafeAreaProvider>
      <AppContent />
    </SafeAreaProvider>
  );
}

function Header({
  health,
  healthError,
  activePersona,
  hasResult,
  showPersona,
  onEvidence,
  onReset,
  onPersona,
  onSettings,
  onAbout,
}: {
  health: Health | null;
  healthError: boolean;
  activePersona: PersonaSummary | null;
  hasResult: boolean;
  showPersona: boolean;
  onEvidence: () => void;
  onReset: () => void;
  onPersona: () => void;
  onSettings: () => void;
  onAbout: () => void;
}) {
  const ready = Boolean(health?.model_ready) && !healthError;
  return (
    <View style={styles.header}>
      <View style={styles.brandRow}>
        <View style={styles.brandMark}><View /><View /><View /></View>
        <Text style={styles.brand}>Echora</Text>
        <View accessibilityRole="text" style={styles.status}>
          <View style={[styles.statusDot, ready && styles.statusDotReady]} />
          <Text numberOfLines={1} style={styles.statusText}>
            {healthError ? "Backend offline" : `${formatBackend(health?.asr_backend)} · ${ready ? "Ready" : "Starting"}`}
          </Text>
        </View>
      </View>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.headerActions}>
        {hasResult && <HeaderButton label="Evidence" onPress={onEvidence} />}
        {hasResult && <HeaderButton label="Start over" onPress={onReset} strong />}
        {showPersona && <HeaderButton label={activePersona?.label.split(",")[0] || "No profile"} onPress={onPersona} />}
        <HeaderButton label="Places" onPress={onSettings} />
        <HeaderButton label="About" onPress={onAbout} />
      </ScrollView>
    </View>
  );
}

function HeaderButton({ label, onPress, strong = false }: { label: string; onPress: () => void; strong?: boolean }) {
  return (
    <Pressable accessibilityRole="button" onPress={onPress} style={({ pressed }) => [styles.headerButton, strong && styles.headerButtonStrong, pressed && styles.pressed]}>
      <Text numberOfLines={1} style={[styles.headerButtonText, strong && styles.headerButtonStrongText]}>{label}</Text>
    </Pressable>
  );
}

function IdleStage({
  modelReady,
  detail,
  place,
  places,
  detected,
  audiences,
  audienceId,
  resolvedAudience,
  onPickPlace,
  onPickAudience,
  onSettings,
  onRecord,
  onPickFile,
}: {
  modelReady: boolean;
  detail?: string;
  place: string;
  places: Place[];
  detected: { id: string; label: string; meters: number } | null;
  audiences: AudienceProfile[];
  audienceId: string;
  resolvedAudience: ResolvedAudience | null;
  onPickPlace: (id: string) => void;
  onPickAudience: (id: string) => void;
  onSettings: () => void;
  onRecord: () => void;
  onPickFile: () => void;
}) {
  const selected = places.filter((item) => item.id === place);
  const kept = [...selected, ...places.filter((item) => item.id !== place)].slice(0, IDLE_CHIPS);
  const shown = places.filter((item) => kept.includes(item));
  const hidden = Math.max(0, places.length - shown.length);
  return (
    <ScrollView contentContainerStyle={styles.centerStage} showsVerticalScrollIndicator={false}>
      <View style={styles.micHalo}>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Start recording"
          accessibilityHint="Starts listening to your speech"
          disabled={!modelReady}
          onPress={onRecord}
          style={({ pressed }) => [styles.micButton, !modelReady && styles.disabled, pressed && styles.micPressed]}
        >
          <Text style={styles.micGlyph}>●</Text>
          <View style={styles.micStem} />
        </Pressable>
      </View>
      <Text accessibilityRole="header" style={styles.stageTitle}>{modelReady ? "Tap to speak" : detail || "Waiting for the speech model"}</Text>
      <Text style={styles.stageHint}>Take your time. Pauses are welcome.</Text>

      <Text style={styles.legend}>Where are you speaking?</Text>
      <View style={styles.chipWrap}>
        <Chip label="General" selected={place === ""} onPress={() => onPickPlace("")} />
        {shown.map((item) => (
          <Chip key={item.id} label={item.label} selected={place === item.id} onPress={() => onPickPlace(item.id)} />
        ))}
        {hidden > 0 && <Chip label={`${hidden} more…`} onPress={onSettings} />}
      </View>
      {detected && (
        <Text accessibilityLiveRegion="polite" style={styles.detected}>
          Detected <Text style={styles.bold}>{detected.label}</Text> · {detected.meters} m away. Tap another if that is wrong.
        </Text>
      )}
      <Text style={styles.legend}>Who are you speaking to?</Text>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.audienceRow}
      >
        <AudienceButton
          audience={null}
          label="Usual here"
          selected={audienceId === ""}
          onPress={() => onPickAudience("")}
        />
        {audiences.map((item) => (
          <AudienceButton
            key={item.id}
            audience={item}
            label={item.label}
            selected={audienceId === item.id}
            onPress={() => onPickAudience(item.id)}
          />
        ))}
      </ScrollView>
      {resolvedAudience && (
        <View style={styles.resolvedCard}>
          <Text style={styles.resolvedTitle}>Speaking to {resolvedAudience.audience_label}</Text>
          <Text style={styles.resolvedText}>
            {resolvedAudience.listener === "familiar" ? "Knows you" : "Does not know you"}
            {" · "}
            {styleLabel(resolvedAudience)}
          </Text>
        </View>
      )}
      <LinkButton label="Choose an audio file" onPress={onPickFile} />
    </ScrollView>
  );
}

const AUDIENCE_COLORS = [colors.sageLight, colors.coralLight, colors.paper, "#E8E2F4", "#E2EEF4"];

function audienceGlyph(audience: AudienceProfile | null) {
  if (!audience) return "◎";
  if (audience.icon.trim()) return audience.icon.trim().slice(0, 2);
  if (audience.kind === "role") return "✦";
  if (audience.kind === "generic") return "○";
  return audience.label
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((word) => word[0]?.toUpperCase())
    .join("") || "○";
}

function audienceColor(audience: AudienceProfile | null) {
  if (!audience) return colors.sageLight;
  const total = [...audience.id].reduce((sum, value) => sum + value.charCodeAt(0), 0);
  return AUDIENCE_COLORS[total % AUDIENCE_COLORS.length];
}

function styleLabel(audience: ResolvedAudience) {
  const parts: string[] = [audience.style.brevity];
  if (audience.style.courtesy === "please") parts.push("polite");
  parts.push(audience.style.formality);
  return parts.join(" · ");
}

function AudienceButton({
  audience,
  label,
  selected,
  onPress,
}: {
  audience: AudienceProfile | null;
  label: string;
  selected: boolean;
  onPress: () => void;
}) {
  return (
    <Pressable
      accessibilityRole="radio"
      accessibilityLabel={label}
      accessibilityState={{ checked: selected }}
      onPress={onPress}
      style={({ pressed }) => [styles.audienceButton, selected && styles.audienceButtonSelected, pressed && styles.pressed]}
    >
      <View style={[styles.audienceAvatar, { backgroundColor: audienceColor(audience) }, selected && styles.audienceAvatarSelected]}>
        <Text style={[styles.audienceGlyph, selected && styles.audienceGlyphSelected]}>{audienceGlyph(audience)}</Text>
      </View>
      <Text numberOfLines={2} style={[styles.audienceLabel, selected && styles.audienceLabelSelected]}>{label}</Text>
    </Pressable>
  );
}

function RecordingStage({ elapsed, level, onStop, onCancel }: { elapsed: number; level: number; onStop: () => void; onCancel: () => void }) {
  const waveform = Array.from({ length: 21 }, (_, index) => {
    const variation = 0.38 + 0.62 * Math.abs(Math.sin(index * 1.7 + elapsed * 0.35));
    return Math.max(7, 52 * level * variation);
  });
  return (
    <View style={styles.centerStage}>
      <Orb state="listening" level={level} />
      <Text style={styles.recordingLabel}><Text style={styles.recordingDot}>●</Text> Listening · {elapsed}s</Text>
      <View accessibilityLabel="Live microphone level" style={styles.waveform}>
        {waveform.map((height, index) => <View key={index} style={[styles.waveBar, { height }]} />)}
      </View>
      <PrimaryButton label="■  Finish speaking" onPress={onStop} />
      <LinkButton label="Cancel recording" onPress={onCancel} />
    </View>
  );
}

function WorkingStage({ label, progress }: { label: string; progress: number }) {
  return (
    <View accessibilityRole="progressbar" style={styles.centerStage}>
      <ActivityIndicator size="large" color={colors.green} />
      <Text accessibilityRole="header" style={styles.stageTitle}>{label}</Text>
      <Text style={styles.stageHint}>Your literal transcript stays separate from wording suggestions.</Text>
      <View style={styles.progressTrack}><View style={[styles.progressFill, { width: `${30 + progress * 22}%` }]} /></View>
    </View>
  );
}

function ChooseStage({ result, onChoose }: { result: Transcription; onChoose: (candidate: MessageCandidate) => void }) {
  return (
    <ScrollView contentContainerStyle={styles.scrollStage} showsVerticalScrollIndicator={false}>
      <Text accessibilityRole="header" style={styles.heading}>Which one sounds right?</Text>
      <Text style={styles.stageHint}>{result.ranker.reason}</Text>
      {result.warnings.filter((warning) => warning !== result.ranker.reason).map((warning) => (
        <Text key={warning} style={styles.warning}>{warning}</Text>
      ))}
      <View style={styles.cardList}>
        {result.messages.map((candidate, index) => (
          <Pressable
            key={candidate.message_id}
            accessibilityRole="button"
            accessibilityLabel={`Option ${index + 1}: ${candidate.corrected_text}`}
            accessibilityHint="Chooses and immediately speaks this message"
            onPress={() => onChoose(candidate)}
            style={({ pressed }) => [styles.candidate, pressed && styles.cardPressed]}
          >
            <View style={styles.candidateTop}>
              <Text style={styles.optionLabel}>Option {index + 1}</Text>
              <Text style={styles.weight}>{Math.round(groupedWeight(result, candidate) * 100)}% grouped search weight</Text>
            </View>
            <Text style={styles.candidateMessage}>{candidate.corrected_text}</Text>
            {candidate.repair_status === 'unavailable' && <Text style={styles.sheetNote}>Literal words only. Choose to speak these exact words.</Text>}
            <Text style={styles.literalLabel}>ALL TRANSCRIPTIONS</Text>
            <Text style={styles.literalText}>{candidate.source_literals.map((text) => `“${text}”`).join(" · ")}</Text>
            <Text style={styles.chooseLabel}>Choose this →</Text>
          </Pressable>
        ))}
      </View>
    </ScrollView>
  );
}

function ComposeStage({
  result,
  selectedId,
  message,
  speaking,
  edited,
  copied,
  remembering,
  memoryNotice,
  canRemember,
  onRemember,
  onChoose,
  onCompare,
  onEdit,
  onCopy,
  onSpeak,
  onEvidence,
  onRevert,
}: {
  result: Transcription;
  selectedId: string | null;
  message: string;
  speaking: boolean;
  edited: boolean;
  copied: boolean;
  remembering: boolean;
  memoryNotice: string;
  canRemember: boolean;
  onRemember: () => void;
  onChoose: (candidate: MessageCandidate) => void;
  onCompare: () => void;
  onEdit: (text: string) => void;
  onCopy: () => void;
  onSpeak: () => void;
  onEvidence: () => void;
  onRevert: (candidate: MessageCandidate) => void;
}) {
  const current = result.messages.find((candidate) => candidate.message_id === selectedId) || null;
  const details = current?.specializations || [];
  return (
    <KeyboardAvoidingView style={styles.fill} behavior={Platform.OS === "ios" ? "padding" : undefined} keyboardVerticalOffset={12}>
      <ScrollView contentContainerStyle={styles.composeStage} keyboardShouldPersistTaps="handled" showsVerticalScrollIndicator={false}>
        {speaking && <Orb state="talking" size={150} level={0.55} />}
        {result.messages.length > 1 && (
          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.pills}>
            {result.messages.map((candidate) => (
              <Chip
                key={candidate.message_id}
                label={candidate.corrected_text}
                selected={candidate.message_id === selectedId}
                onPress={() => onChoose(candidate)}
              />
            ))}
          </ScrollView>
        )}
        <Text accessibilityRole="header" style={styles.heading}>Your message</Text>
        {result.conversation_reference && <Text style={styles.sheetNote}>Following your previous message: {result.conversation_reference}</Text>}
        {result.question && <Text style={styles.sheetNote}>{result.question}</Text>}
        <TextInput
          accessibilityLabel="Message to communicate"
          multiline
          value={message}
          onChangeText={onEdit}
          style={styles.messageInput}
          textAlignVertical="top"
        />
        {details.length > 0 && current?.plain_text && (
          <View style={styles.detailNote}>
            <Text style={styles.detailText}>◆  Using your wording for <Text style={styles.bold}>{details.map((item) => item.anchor).join(", ")}</Text></Text>
            <LinkButton label="Say it plainly instead" onPress={() => onRevert(current)} />
          </View>
        )}
        {message.trim() === current?.corrected_text.trim() && !!current.contextual_additions?.length && <View style={styles.detailNote}>
          {current.contextual_additions.map((addition, index) => <Text key={index} style={styles.detailText}>Added from your saved wording: {addition.wording || addition.surface || addition.text}</Text>)}
          <Text style={styles.sheetNote}>You can edit these details in your message.</Text>
          {current.plain_text && <LinkButton label="Say it without additions" onPress={() => onRevert(current)} />}
        </View>}
        <Text style={styles.evidenceCaption}>Based on literal evidence · <Text onPress={onEvidence} style={styles.inlineLink}>see all transcriptions</Text></Text>
        <View style={styles.speakPanel}>
          <View style={styles.speakStatus}>
            <Text style={styles.speakIcon}>{speaking ? "♪" : edited ? "✎" : "✓"}</Text>
            <View style={styles.fill}>
              <Text style={styles.speakTitle}>{speaking ? "Speaking" : edited ? "Ready to speak" : "Spoken aloud"}</Text>
              <Text style={styles.speakHint}>{edited ? "Speak when you are ready" : "Edit the wording and say it again"}</Text>
            </View>
          </View>
          <SecondaryButton label={remembering ? 'Remembering…' : 'Remember'} onPress={onRemember} disabled={!canRemember || !message.trim() || speaking || remembering} />
          {!canRemember && <Text style={styles.sheetNote}>Choose your own profile to remember messages.</Text>}
          {!!memoryNotice && <Text accessibilityLiveRegion="polite" style={styles.sheetNote}>{memoryNotice}</Text>}
          <View style={styles.actionRow}>
            <SecondaryButton label={copied ? "Copied" : "Copy"} onPress={onCopy} disabled={!message.trim()} />
            <PrimaryButton label={speaking ? "Speaking…" : edited ? "Speak this" : "Speak again"} onPress={onSpeak} disabled={!message.trim() || speaking} compact />
          </View>
        </View>
        {result.messages.length > 1 && <LinkButton label="Compare all options" onPress={onCompare} />}
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

function ErrorStage({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <View style={styles.centerStage}>
      <View accessibilityRole="alert" style={styles.errorBanner}>
        <Text style={styles.errorTitle}>We could not finish that.</Text>
        <Text style={styles.errorText}>{message}</Text>
      </View>
      <PrimaryButton label="Try again" onPress={onRetry} />
    </View>
  );
}

function EvidenceBody({ result }: { result: Transcription }) {
  const trace = result.ranker.personalization;
  return (
    <View style={styles.sheetStack}>
      {result.ranking && <>
        <SheetHeading text="Transcript selection" />
        <Text style={styles.sheetNote}>{result.ranking.reason}</Text>
        <Text style={styles.sheetNote}>{result.ranking.status === 'unavailable' || result.ranking.route === 'legacy' ? 'Audio verification was not used for this recording.' : result.ranking.decision === 'ambiguous' ? 'The audio left competing interpretations.' : 'A transcript was recommended. Alternatives remain available.'}</Text>
        {result.decision_source === 'user' && <Text style={styles.sheetNote}>Your explicit choice is used for this message.</Text>}
      </>}
      {result.hypotheses.map((item) => (
        <View key={item.id} style={styles.evidenceRow}>
          <Text style={styles.evidenceText}>{item.literal_text}</Text>
          {typeof item.search_weight === 'number' && <Text style={styles.codeText}>{(item.search_weight * 100).toFixed(1)}% search weight{typeof item.sequence_score === 'number' ? ` · score ${item.sequence_score.toFixed(3)}` : ''}</Text>}
        </View>
      ))}
      <Text style={styles.sheetNote}>Search weights compare only these beam hypotheses. They are not calibrated confidence.</Text>
      <SheetHeading text="What this message was written for" />
      <InfoPair label="Setting" value={behaviourLabel(result.context)} />
      <InfoPair label="Speaking to" value={result.audience.audience_label} />
      <InfoPair
        label="Listener"
        value={result.listener === "familiar" ? "Someone who knows the speaker — a need is stated to them." : "Someone who does not know the speaker — a need is asked of them, and kept short."}
      />
      <InfoPair label="Speaking style" value={styleLabel(result.audience)} />
      {trace && (
        <>
          <SheetHeading text={`What ${trace.profile_label.split(",")[0]}’s profile contributed`} />
          <InfoPair label="Known words the recognizer produced" value={trace.lexicon_hints.join(" · ") || "None in this utterance"} />
          <InfoPair
            label="Personal details"
            value={trace.specializations_offered === 0 ? "None offered — either scoped out here, or its anchor was not settled in the beams." : `${trace.specializations_offered} offered · ${trace.specializations_applied} applied in code`}
          />
        </>
      )}
      <SheetHeading text="How each option was formed" />
      {result.messages.map((candidate, index) => (
        <View key={candidate.message_id} style={styles.evidenceOption}>
          <Text style={styles.bold}>Option {index + 1} · {candidate.corrected_text}</Text>
          <InfoPair label="Interpreted meaning" value={candidate.interpreted_intent} />
          {Object.entries(candidate.word_alternatives || {}).map(([word, options]) => (
            <Text key={word} style={styles.sheetNote}><Text style={styles.bold}>{word}</Text> — also heard as {options.join(" · ")}</Text>
          ))}
          <Text style={styles.repairNote}>{candidate.repair_note}</Text>
          {candidate.contextual_additions?.map((addition, i) => <Text key={`addition-${i}`} style={styles.sheetNote}>Added from saved wording: {addition.wording || addition.surface || addition.text}</Text>)}
          {candidate.retrieved_sources?.map((source, i) => <Text key={`source-${i}`} style={styles.sheetNote}>Saved reference considered: {source.text || source.message || source.label || source.source_id || source.id}</Text>)}
        </View>
      ))}
    </View>
  );
}

function PersonaBody({
  personas,
  persona,
  profile,
  places,
  memoryRefresh,
  onMemoriesChanged,
  onProfileDeleted,
  onMemoryMutation,
  onPick,
  onProfileSaved,
}: {
  personas: PersonaSummary[];
  persona: string;
  profile: PersonaProfile | null;
  places: Place[];
  memoryRefresh: number;
  onMemoriesChanged: () => Promise<void>;
  onProfileDeleted: () => Promise<void>;
  onMemoryMutation: () => void;
  onPick: (id: string) => void;
  onProfileSaved: (profile: PersonaProfile) => void;
}) {
  const [showGallery, setShowGallery] = useState(persona !== "user");

  if (persona && !showGallery) {
    return (
      <View style={styles.sheetStack}>
        <View style={styles.profileEditorBanner}>
          <View style={[styles.personaMark, styles.personaMarkSelected]}><Text style={[styles.personaGlyph, styles.personaGlyphSelected]}>◎</Text></View>
          <View style={styles.fill}><Text style={styles.personaLabel}>Editing your profile</Text><Text style={styles.personaBlurb}>These choices stay private to this Echora installation.</Text></View>
        </View>
        <SecondaryButton label="Choose another profile" onPress={() => setShowGallery(true)} />
        {profile ? <ProfileQuestionnaire profile={profile} places={places} onSaved={onProfileSaved} /> : <Text style={styles.sheetNote}>Loading your profile…</Text>}
        {profile && <MemoriesBody key={profile.id} profile={profile} refresh={memoryRefresh} onChanged={onMemoriesChanged} onDeleted={onProfileDeleted} onMutation={onMemoryMutation} />}
      </View>
    );
  }

  return (
    <View style={styles.sheetStack}>
      <Text style={styles.sheetNote}>A profile helps choose among words the recognizer produced and can apply details the speaker explicitly declared. It never replaces the literal transcript.</Text>
      <PersonaCard selected={persona === ""} label="No profile" blurb="Nothing personal is used. This is how Echora behaves for a stranger." onPress={() => onPick("")} />
      {personas.map((item) => (
        <PersonaCard
          key={item.id}
          selected={persona === item.id}
          label={item.label}
          blurb={`${item.blurb}\n${item.lexicon_size} known words · ${item.specialization_size} details`}
          onPress={() => {
            onPick(item.id);
            setShowGallery(false);
          }}
        />
      ))}
    </View>
  );
}

function MemoriesBody({ profile, refresh, onChanged, onDeleted, onMutation }: {
  profile: PersonaProfile; refresh: number; onChanged: () => Promise<void>; onDeleted: () => Promise<void>; onMutation: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<MemoryList | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [confirm, setConfirm] = useState<'clear' | 'profile' | null>(null);
  useEffect(() => {
    if (!open) return;
    let current = true;
    setData(null);
    setError('');
    readMemories(profile.id).then((value) => { if (current) setData(value); })
      .catch((reason) => { if (current) setError(reason instanceof Error ? reason.message : 'Saved messages could not be loaded.'); });
    return () => { current = false; };
  }, [profile.id, refresh, open]);

  async function remove(kind: 'entry' | 'clear' | 'profile', id?: string) {
    if (!data || busy) return;
    onMutation();
    setBusy(true);
    setError('');
    try {
      if (kind === 'profile') {
        await deleteProfile(profile.id, profile.revision || 0, data.memory_revision);
        await onDeleted();
        return;
      }
      const changed = kind === 'clear'
        ? await clearMemories(profile.id, data.memory_revision)
        : await deleteMemory(profile.id, id!, data.memory_revision);
      setData({ ...data, memory_revision: changed.memory_revision,
        memories: data.memories.filter((item) => !changed.deleted_ids.includes(item.id)) });
      setConfirm(null);
      await onChanged();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'This change could not be saved.');
      try { setData(await readMemories(profile.id)); } catch { /* Keep the error visible. */ }
    } finally { setBusy(false); }
  }

  return <View style={styles.sheetStack}>
    <SecondaryButton label={open ? 'Close remembered messages' : 'Remembered messages'} disabled={busy} onPress={() => { setOpen(!open); setConfirm(null); }} />
    {open && <>
      <Text style={styles.sheetNote}>Only messages you choose to Remember are saved here. They can help with future wording in this profile.</Text>
      {!!error && <Text accessibilityRole="alert" style={styles.warning}>{error}</Text>}
      {!data && !error && <Text style={styles.sheetNote}>Loading saved messages…</Text>}
      {data && <>
        {!data.memories.length && <Text style={styles.sheetNote}>No remembered messages.</Text>}
        {data.memories.map((memory) => <View key={memory.id} style={styles.evidenceRow}>
          <Text style={styles.evidenceText}>{memory.message}</Text>
          <Text style={styles.sheetNote}>{[memory.scope.scenario, memory.scope.recipient, new Date(memory.created_at * 1000).toLocaleDateString()].filter(Boolean).join(' · ')}</Text>
          <SecondaryButton label="Forget" disabled={busy} onPress={() => void remove('entry', memory.id)} />
        </View>)}
        <SecondaryButton label="Forget all saved messages" disabled={busy || !data.memories.length} onPress={() => setConfirm('clear')} />
        {!profile.baseline && <SecondaryButton label="Delete this profile" danger disabled={busy} onPress={() => setConfirm('profile')} />}
        {confirm && <View style={styles.detailNote}>
          <Text style={styles.sheetNote}>{confirm === 'profile' ? `Delete ${profile.label} and its saved messages?` : 'Forget every saved message in this profile?'}</Text>
          <SecondaryButton label={confirm === 'profile' ? 'Delete profile and messages' : 'Forget all'} danger disabled={busy} onPress={() => void remove(confirm)} />
          <SecondaryButton label="Keep them" disabled={busy} onPress={() => setConfirm(null)} />
        </View>}
      </>}
    </>}
  </View>;
}

function PersonaCard({ selected, label, blurb, onPress }: { selected: boolean; label: string; blurb: string; onPress: () => void }) {
  return (
    <Pressable accessibilityRole="radio" accessibilityState={{ checked: selected }} onPress={onPress} style={({ pressed }) => [styles.personaCard, selected && styles.personaCardSelected, pressed && styles.pressed]}>
      <View style={[styles.personaMark, selected && styles.personaMarkSelected]}><Text style={[styles.personaGlyph, selected && styles.personaGlyphSelected]}>◎</Text></View>
      <View style={styles.fill}>
        <Text style={styles.personaLabel}>{label}</Text>
        <Text style={styles.personaBlurb}>{blurb}</Text>
      </View>
    </Pressable>
  );
}

function SettingsBody({ settings, onSaved }: { settings: PlaceSettings | null; onSaved: (settings: PlaceSettings) => void }) {
  const [status, setStatus] = useState("");
  const [label, setLabel] = useState("");
  const [behaviour, setBehaviour] = useState<CommunicationContext>("home");
  const [company, setCompany] = useState<Listener | null>(null);
  const [busy, setBusy] = useState(false);

  if (!settings) return <Text style={styles.sheetNote}>Places are unavailable, so the setting stays yours to choose by hand.</Text>;
  const doc = settings;
  const tagged = doc.places.filter(isTagged).length;

  async function persist(next: PlaceSettings, note: string) {
    setBusy(true);
    setStatus("Saving…");
    try {
      onSaved(await savePlaces(next));
      setStatus(note);
    } catch {
      setStatus("That could not be saved.");
    } finally {
      setBusy(false);
    }
  }

  async function tag(target: Place) {
    setStatus("Finding where you are…");
    try {
      const permission = await Location.requestForegroundPermissionsAsync();
      if (!permission.granted) {
        setStatus("Location permission was refused, so places stay yours to choose by hand.");
        return;
      }
      const position = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.High });
      await persist(
        {
          ...doc,
          places: doc.places.map((item) => item.id === target.id ? { ...item, latitude: position.coords.latitude, longitude: position.coords.longitude, tagged_at: null } : item),
        },
        `${target.label} is now tagged here.`,
      );
    } catch {
      setStatus("Your location could not be found just now.");
    }
  }

  function setPlaceListener(target: Place, next: Listener | null) {
    void persist(
      { ...doc, places: doc.places.map((item) => item.id === target.id ? { ...item, listener: next } : item) },
      next === "familiar" ? `Messages for ${target.label} will be said to someone who knows you.` : next === "unfamiliar" ? `Messages for ${target.label} will be asked of someone who does not know you.` : `${target.label} will follow whoever is speaking, then ${listenerLabel(target.context)}.`,
    );
  }

  function addPlace() {
    const trimmed = label.trim();
    if (!trimmed) return;
    setLabel("");
    void persist(
      {
        ...doc,
        places: [...doc.places, { id: "", label: trimmed, context: behaviour, listener: company, builtin: false, latitude: null, longitude: null, radius_m: 150, tagged_at: null }],
      },
      `${trimmed} was added. Tag it while you are there.`,
    );
  }

  return (
    <View style={styles.sheetStack}>
      <Text style={styles.sheetNote}>A place borrows one of the four settings and may say whether people there know you. “Leave it to me” is a real third choice: it lets the speaker’s profile decide. Coordinates are kept by your local Echora service and matched on this phone; only the borrowed setting reaches a transcription request.</Text>
      <View style={styles.toggleRow}>
        <Switch
          accessibilityLabel="Choose the setting from where I am"
          disabled={busy}
          value={doc.auto_detect}
          onValueChange={(value) => void persist({ ...doc, auto_detect: value }, value ? (tagged ? "Echora will choose the setting from where you are, and say so." : "Turned on. Tag a place while you are there.") : "Turned off. The setting stays yours to choose.")}
          trackColor={{ false: colors.line, true: colors.sage }}
          thumbColor={colors.paper}
        />
        <View style={styles.fill}>
          <Text style={styles.toggleTitle}>Choose the setting from where I am</Text>
          <Text style={styles.sheetNote}>{tagged ? `${tagged} of ${doc.places.length} places are tagged.` : "Nothing is tagged yet."}</Text>
        </View>
      </View>

      {doc.places.map((item) => (
        <View key={item.id} style={styles.placeCard}>
          <Text style={styles.placeTitle}>{item.label}</Text>
          <Text style={styles.sheetNote}>Behaves as {behaviourLabel(item.context)} · {item.listener === "familiar" ? "people here know me" : item.listener === "unfamiliar" ? "people here do not know me" : `whoever is speaking decides, then ${listenerLabel(item.context)}`} · {isTagged(item) ? `tagged within ${item.radius_m} m` : "not tagged"}</Text>
          <Text style={styles.fieldLabel}>Who is usually here?</Text>
          <ListenerChoices value={item.listener} disabled={busy} onChange={(next) => setPlaceListener(item, next)} />
          <View style={styles.actionRowWrap}>
            <SecondaryButton label={isTagged(item) ? "Retag here" : "Tag here"} onPress={() => void tag(item)} disabled={busy} />
            {isTagged(item) && <SecondaryButton label="Clear tag" onPress={() => void persist({ ...doc, places: doc.places.map((place) => place.id === item.id ? { ...place, latitude: null, longitude: null, tagged_at: null } : place) }, `The location for ${item.label} was removed.`)} disabled={busy} />}
            {!item.builtin && <SecondaryButton label="Remove" onPress={() => void persist({ ...doc, places: doc.places.filter((place) => place.id !== item.id) }, `${item.label} was removed.`)} disabled={busy} danger />}
          </View>
        </View>
      ))}

      <View style={styles.addCard}>
        <SheetHeading text="Add a place" />
        <TextInput accessibilityLabel="Place name" placeholder="Shopping centre" placeholderTextColor={colors.muted} value={label} onChangeText={setLabel} style={styles.field} />
        <Text style={styles.fieldLabel}>Behaves as</Text>
        <View style={styles.chipWrap}>{contexts.map((item) => <Chip key={item.value} label={item.label} selected={behaviour === item.value} onPress={() => setBehaviour(item.value)} />)}</View>
        <Text style={styles.fieldLabel}>Who is usually here?</Text>
        <ListenerChoices value={company} onChange={setCompany} />
        <PrimaryButton label="Add place" onPress={addPlace} disabled={busy || !label.trim()} compact />
      </View>
      {status ? <Text accessibilityLiveRegion="polite" style={styles.sheetNote}>{status}</Text> : null}
    </View>
  );
}

function ListenerChoices({ value, onChange, disabled = false }: { value: Listener | null; onChange: (value: Listener | null) => void; disabled?: boolean }) {
  return (
    <View accessibilityRole="radiogroup" style={styles.choiceStack}>
      <Choice label="Leave it to whoever is speaking" selected={value === null} disabled={disabled} onPress={() => onChange(null)} />
      <Choice label="People here know me" selected={value === "familiar"} disabled={disabled} onPress={() => onChange("familiar")} />
      <Choice label="People here do not know me" selected={value === "unfamiliar"} disabled={disabled} onPress={() => onChange("unfamiliar")} />
    </View>
  );
}

function AudienceListenerChoices({ value, onChange }: { value: Listener; onChange: (value: Listener) => void }) {
  return (
    <View accessibilityRole="radiogroup" style={styles.choiceStack}>
      <Choice label="They know me and my routines" selected={value === "familiar"} disabled={false} onPress={() => onChange("familiar")} />
      <Choice label="They do not know me yet" selected={value === "unfamiliar"} disabled={false} onPress={() => onChange("unfamiliar")} />
    </View>
  );
}

function Choice({ label, selected, disabled, onPress }: { label: string; selected: boolean; disabled: boolean; onPress: () => void }) {
  return (
    <Pressable accessibilityRole="radio" accessibilityState={{ checked: selected, disabled }} disabled={disabled} onPress={onPress} style={styles.choice}>
      <View style={[styles.radio, selected && styles.radioSelected]}>{selected && <View style={styles.radioDot} />}</View>
      <Text style={styles.choiceText}>{label}</Text>
    </Pressable>
  );
}

const STYLE_PRESETS: { id: string; label: string; example: string; value: CommunicationStyle }[] = [
  { id: "direct", label: "Short and direct", example: "I need water.", value: { brevity: "short", courtesy: "plain", formality: "neutral" } },
  { id: "natural", label: "Natural", example: "I need some water.", value: { brevity: "natural", courtesy: "plain", formality: "neutral" } },
  { id: "warm", label: "Warm and informal", example: "I need some water, please.", value: { brevity: "natural", courtesy: "please", formality: "informal" } },
  { id: "formal", label: "Polite and complete", example: "I need water, please.", value: { brevity: "complete", courtesy: "please", formality: "formal" } },
];

function sameStyle(left: CommunicationStyle | undefined, right: CommunicationStyle) {
  return Boolean(left) && left?.brevity === right.brevity && left.courtesy === right.courtesy && left.formality === right.formality;
}

function styleName(style: CommunicationStyle) {
  return STYLE_PRESETS.find((item) => sameStyle(style, item.value))?.label || `${style.brevity}, ${style.courtesy}, ${style.formality}`;
}

function slug(value: string) {
  return value.toLowerCase().trim().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "item";
}

function toggleValue<T>(values: T[], value: T) {
  return values.includes(value) ? values.filter((item) => item !== value) : [...values, value];
}

function StyleChoices({ value, onChange, allowDefault = false }: { value: CommunicationStyle | null; onChange: (value: CommunicationStyle | null) => void; allowDefault?: boolean }) {
  return (
    <View accessibilityRole="radiogroup" style={styles.styleChoiceGrid}>
      {allowDefault && (
        <Pressable accessibilityRole="radio" accessibilityState={{ checked: value === null }} onPress={() => onChange(null)} style={({ pressed }) => [styles.styleChoice, value === null && styles.styleChoiceSelected, pressed && styles.pressed]}>
          <Text style={[styles.styleChoiceTitle, value === null && styles.styleChoiceTitleSelected]}>Use the profile choice</Text>
          <Text style={styles.styleChoiceExample}>Keep the style selected in the previous step.</Text>
        </Pressable>
      )}
      {STYLE_PRESETS.map((item) => (
        <Pressable key={item.id} accessibilityRole="radio" accessibilityState={{ checked: sameStyle(value || undefined, item.value) }} onPress={() => onChange(item.value)} style={({ pressed }) => [styles.styleChoice, sameStyle(value || undefined, item.value) && styles.styleChoiceSelected, pressed && styles.pressed]}>
          <View style={styles.styleChoiceTopline}>
            <Text style={[styles.styleChoiceTitle, sameStyle(value || undefined, item.value) && styles.styleChoiceTitleSelected]}>{item.label}</Text>
            {sameStyle(value || undefined, item.value) && <Text style={styles.styleChoiceCheck}>✓</Text>}
          </View>
          <Text style={styles.styleChoiceExample}>“{item.example}”</Text>
        </Pressable>
      ))}
    </View>
  );
}

function ProfileQuestionnaire({ profile, places, onSaved }: { profile: PersonaProfile; places: Place[]; onSaved: (profile: PersonaProfile) => void }) {
  const [draft, setDraft] = useState(profile);
  const [step, setStep] = useState(0);
  const [status, setStatus] = useState("");
  const [editingAudience, setEditingAudience] = useState("");
  const [audienceLabel, setAudienceLabel] = useState("");
  const [relationship, setRelationship] = useState("");
  const [audienceKind, setAudienceKind] = useState<AudienceProfile["kind"]>("person");
  const [audienceListener, setAudienceListener] = useState<Listener>("familiar");
  const [audienceStyle, setAudienceStyle] = useState<CommunicationStyle>(STYLE_PRESETS[1].value);
  const [audienceSettings, setAudienceSettings] = useState<CommunicationContext[]>([]);
  const [audiencePlaces, setAudiencePlaces] = useState<string[]>([]);
  const [audienceOverrides, setAudienceOverrides] = useState<Partial<Record<CommunicationContext, CommunicationStyle>>>({});
  const [wordDisplay, setWordDisplay] = useState("");
  const [wordKind, setWordKind] = useState<LexiconEntry["kind"]>("object");
  const [wordSettings, setWordSettings] = useState<CommunicationContext[]>([]);
  const [plain, setPlain] = useState("");
  const [surface, setSurface] = useState("");
  const [detailSettings, setDetailSettings] = useState<CommunicationContext[]>([]);
  const [knownBy, setKnownBy] = useState<string[]>([]);

  useEffect(() => setDraft(profile), [profile]);

  function resetAudienceEditor() {
    setEditingAudience("");
    setAudienceLabel("");
    setRelationship("");
    setAudienceKind("person");
    setAudienceListener("familiar");
    setAudienceStyle(STYLE_PRESETS[1].value);
    setAudienceSettings([]);
    setAudiencePlaces([]);
    setAudienceOverrides({});
  }

  function editAudience(item: AudienceProfile) {
    setEditingAudience(item.id);
    setAudienceLabel(item.label);
    setRelationship(item.relationship);
    setAudienceKind(item.kind);
    setAudienceListener(item.listener);
    setAudienceStyle(item.style);
    setAudienceSettings(item.visible_in_settings);
    setAudiencePlaces(item.visible_in_places);
    setAudienceOverrides(item.style_by_setting);
  }

  function saveAudience() {
    const label = audienceLabel.trim();
    if (!label) return;
    const id = editingAudience || slug(label);
    const next: AudienceProfile = {
      id,
      label,
      kind: audienceKind,
      relationship: relationship.trim(),
      icon: "",
      listener: audienceListener,
      style: audienceStyle,
      style_by_setting: audienceOverrides,
      visible_in_settings: audienceSettings,
      visible_in_places: audiencePlaces,
      known_detail_ids: draft.audiences.find((item) => item.id === id)?.known_detail_ids || [],
    };
    const personEntry: LexiconEntry = {
      id: `user/person/${id}`,
      word: label.toLowerCase().split(/\s+/).at(-1) || label.toLowerCase(),
      aliases: [],
      display: label,
      kind: "person",
      note: relationship.trim(),
      settings: audienceSettings,
    };
    setDraft((current) => ({
      ...current,
      audiences: [...current.audiences.filter((item) => item.id !== id), next],
      lexicon: audienceKind === "person"
        ? [...current.lexicon.filter((item) => item.id !== personEntry.id), personEntry]
        : current.lexicon.filter((item) => item.id !== personEntry.id),
    }));
    resetAudienceEditor();
  }

  function removeAudience(id: string) {
    setDraft((current) => ({
      ...current,
      audiences: current.audiences.filter((item) => item.id !== id),
      lexicon: current.lexicon.filter((item) => item.id !== `user/person/${id}`),
    }));
    if (editingAudience === id) resetAudienceEditor();
  }

  function addWord() {
    const display = wordDisplay.trim();
    if (!display) return;
    const word = display.toLowerCase().split(/\s+/).at(-1) || display.toLowerCase();
    const item: LexiconEntry = { id: `user/${wordKind}/${slug(display)}`, word, aliases: [], display, kind: wordKind, note: "", settings: wordSettings };
    setDraft((current) => ({ ...current, lexicon: [...current.lexicon.filter((entry) => entry.id !== item.id), item] }));
    setWordDisplay("");
    setWordSettings([]);
  }

  function addDetail() {
    const ordinary = plain.trim();
    const detailed = surface.trim();
    if (!ordinary || !detailed || !detailed.toLowerCase().includes(ordinary.toLowerCase())) {
      setStatus("The fuller wording must contain the ordinary wording exactly.");
      return;
    }
    const anchor = ordinary.toLowerCase().split(/\s+/).at(-1) || ordinary.toLowerCase();
    const id = `user/detail/${slug(anchor)}`;
    const item: SpecializationRule = { id, anchor, plain: ordinary, surface: detailed, kind: "brand", note: "", settings: detailSettings };
    setDraft((current) => ({
      ...current,
      specializations: [...current.specializations.filter((detail) => detail.id !== id), item],
      audiences: current.audiences.map((audience) => ({
        ...audience,
        known_detail_ids: knownBy.includes(audience.id)
          ? [...new Set([...audience.known_detail_ids, id])]
          : audience.known_detail_ids.filter((detailId) => detailId !== id),
      })),
    }));
    setPlain("");
    setSurface("");
    setDetailSettings([]);
    setKnownBy([]);
    setStatus("");
  }

  async function save() {
    setStatus("Saving the reviewed profile…");
    try {
      const { id: _id, icon: _icon, baseline: _baseline, speaker_note: _note, created_at: _created, ...input } = draft;
      const response = await replaceProfile(input, profile.id, profile.revision || 0);
      setDraft(response.profile);
      onSaved(response.profile);
      setStatus(response.refused.length ? `Saved, but ${response.refused.length} invalid detail was not kept.` : "Profile saved. The speaking screen now uses these people and styles.");
    } catch (caught) {
      setStatus(caught instanceof Error ? caught.message : "That profile could not be saved.");
    }
  }

  const stepTitles = ["Your usual voice", "Settings", "People", "Words and details", "Review"];
  const stepCopy = [
    "Start with the voice that should feel like you most of the time.",
    "Fine-tune who is usually present and how your voice changes by setting.",
    "Create the people and roles you want to choose before speaking.",
    "Add only the words and exact personal details that genuinely help.",
    "Check every decision before anything becomes active.",
  ];
  return (
    <View style={styles.onboarding}>
      <View style={styles.questionnaireHead}>
        <View style={styles.stepTopline}><Text style={styles.stepCount}>Personal setup</Text><Text style={styles.stepCount}>Step {step + 1} of {stepTitles.length}</Text></View>
        <View accessibilityRole="progressbar" accessibilityValue={{ min: 1, max: stepTitles.length, now: step + 1 }} style={styles.stepRail}>
          {stepTitles.map((title, index) => <View key={title} style={[styles.stepSegment, index <= step && styles.stepSegmentComplete]} />)}
        </View>
        <Text accessibilityRole="header" style={styles.questionnaireTitle}>{stepTitles[step]}</Text>
        <Text style={styles.questionnaireCopy}>{stepCopy[step]}</Text>
      </View>

      <View style={styles.questionPanel}>

      {step === 0 && (
        <View style={styles.questionStack}>
          <Text style={styles.fieldLabel}>What should this profile be called?</Text>
          <TextInput accessibilityLabel="Profile name" value={draft.label} onChangeText={(label) => setDraft((current) => ({ ...current, label }))} style={styles.field} />
          <Text style={styles.fieldLabel}>Where do you usually begin?</Text>
          <View style={styles.chipWrap}>{contexts.map((item) => <Chip key={item.value} label={item.label} selected={draft.context_default === item.value} onPress={() => setDraft((current) => ({ ...current, context_default: item.value }))} />)}</View>
          <Text style={styles.fieldLabel}>How should Echora usually speak for you?</Text>
          <StyleChoices value={draft.style} onChange={(style) => style && setDraft((current) => ({ ...current, style }))} />
        </View>
      )}

      {step === 1 && (
        <View style={styles.questionStack}>
          <Text style={styles.sheetNote}>These are profile fallbacks. A named place that explicitly declares its listener remains stronger.</Text>
          {contexts.map((item) => (
            <View key={item.value} style={styles.placeCard}>
              <Text style={styles.placeTitle}>{item.label}</Text>
              <Text style={styles.fieldLabel}>Who is usual here?</Text>
              <ListenerChoices value={draft.listener_by_setting[item.value] || null} onChange={(value) => setDraft((current) => {
                const listener_by_setting = { ...current.listener_by_setting };
                if (value) listener_by_setting[item.value] = value;
                else delete listener_by_setting[item.value];
                return { ...current, listener_by_setting };
              })} />
              <Text style={styles.fieldLabel}>Should your style change here?</Text>
              <StyleChoices allowDefault value={draft.style_by_setting[item.value] || null} onChange={(value) => setDraft((current) => {
                const style_by_setting = { ...current.style_by_setting };
                if (value) style_by_setting[item.value] = value;
                else delete style_by_setting[item.value];
                return { ...current, style_by_setting };
              })} />
            </View>
          ))}
        </View>
      )}

      {step === 2 && (
        <View style={styles.questionStack}>
          <Text style={styles.sheetNote}>These become the listener icons on the speaking screen. A place only decides which icons are offered; you still choose the person.</Text>
          {draft.audiences.map((item) => (
            <View key={item.id} style={styles.audienceSummary}>
              <View style={[styles.audienceAvatar, { backgroundColor: audienceColor(item) }]}><Text style={styles.audienceGlyph}>{audienceGlyph(item)}</Text></View>
              <View style={styles.fill}><Text style={styles.bold}>{item.label}</Text><Text style={styles.sheetNote}>{item.listener} · {styleName(item.style)}</Text></View>
              <SecondaryButton label="Edit" onPress={() => editAudience(item)} />
              <SecondaryButton label="Remove" danger onPress={() => removeAudience(item.id)} />
            </View>
          ))}
          <View style={styles.addCard}>
            <Text style={styles.fieldLabel}>{editingAudience ? "Edit this listener" : "Add a person or recurring role"}</Text>
            <TextInput accessibilityLabel="Listener name" placeholder="Priya or New care staff" placeholderTextColor={colors.muted} value={audienceLabel} onChangeText={setAudienceLabel} style={styles.field} />
            <TextInput accessibilityLabel="Relationship" placeholder="Daughter, carer, colleague" placeholderTextColor={colors.muted} value={relationship} onChangeText={setRelationship} style={styles.field} />
            <View style={styles.chipWrap}>{(["person", "role", "generic"] as const).map((kind) => <Chip key={kind} label={kind} selected={audienceKind === kind} onPress={() => setAudienceKind(kind)} />)}</View>
            <Text style={styles.fieldLabel}>Do they know you and your routines?</Text>
            <AudienceListenerChoices value={audienceListener} onChange={setAudienceListener} />
            <Text style={styles.fieldLabel}>How should Echora speak to them?</Text>
            <StyleChoices value={audienceStyle} onChange={(value) => value && setAudienceStyle(value)} />
            <Text style={styles.fieldLabel}>Show this icon in these settings</Text>
            <View style={styles.chipWrap}>{contexts.map((item) => <Chip key={item.value} label={item.label} selected={audienceSettings.includes(item.value)} onPress={() => setAudienceSettings((current) => toggleValue(current, item.value))} />)}</View>
            {places.length > 0 && <><Text style={styles.fieldLabel}>Or at these named places</Text><View style={styles.chipWrap}>{places.map((item) => <Chip key={item.id} label={item.label} selected={audiencePlaces.includes(item.id)} onPress={() => setAudiencePlaces((current) => toggleValue(current, item.id))} />)}</View></>}
            {audienceSettings.map((context) => <View key={context} style={styles.overrideCard}><Text style={styles.fieldLabel}>Different style in {behaviourLabel(context)}?</Text><StyleChoices allowDefault value={audienceOverrides[context] || null} onChange={(value) => setAudienceOverrides((current) => {
              const next = { ...current };
              if (value) next[context] = value;
              else delete next[context];
              return next;
            })} /></View>)}
            <View style={styles.actionRowWrap}><PrimaryButton label={editingAudience ? "Update listener" : "Add listener"} onPress={saveAudience} disabled={!audienceLabel.trim()} compact />{editingAudience && <SecondaryButton label="Cancel" onPress={resetAudienceEditor} />}</View>
          </View>
        </View>
      )}

      {step === 3 && (
        <View style={styles.questionStack}>
          <SheetHeading text="Important words" />
          {draft.lexicon.map((item) => <View key={item.id} style={styles.listRow}><View style={styles.fill}><Text style={styles.bold}>{item.display}</Text><Text style={styles.sheetNote}>{item.kind}{item.settings.length ? ` · ${item.settings.join(", ")}` : " · everywhere"}</Text></View><SecondaryButton label="Remove" danger onPress={() => setDraft((current) => ({ ...current, lexicon: current.lexicon.filter((entry) => entry.id !== item.id) }))} /></View>)}
          <View style={styles.addCard}>
            <TextInput accessibilityLabel="Important word" placeholder="walker, physio, Akash" placeholderTextColor={colors.muted} value={wordDisplay} onChangeText={setWordDisplay} style={styles.field} />
            <View style={styles.chipWrap}>{(["object", "food", "routine", "place", "brand"] as const).map((kind) => <Chip key={kind} label={kind} selected={wordKind === kind} onPress={() => setWordKind(kind)} />)}</View>
            <View style={styles.chipWrap}>{contexts.map((item) => <Chip key={item.value} label={item.label} selected={wordSettings.includes(item.value)} onPress={() => setWordSettings((current) => toggleValue(current, item.value))} />)}</View>
            <PrimaryButton label="Add word" onPress={addWord} disabled={!wordDisplay.trim()} compact />
          </View>
          <SheetHeading text="Exact personal details" />
          {draft.specializations.map((item) => <View key={item.id} style={styles.listRow}><View style={styles.fill}><Text style={styles.bold}>{item.plain} → {item.surface}</Text><Text style={styles.sheetNote}>{item.settings.length ? item.settings.join(", ") : "Everywhere"}</Text></View><SecondaryButton label="Remove" danger onPress={() => setDraft((current) => ({ ...current, specializations: current.specializations.filter((detail) => detail.id !== item.id), audiences: current.audiences.map((audience) => ({ ...audience, known_detail_ids: audience.known_detail_ids.filter((id) => id !== item.id) })) }))} /></View>)}
          <View style={styles.addCard}>
            <TextInput accessibilityLabel="Ordinary wording" placeholder="my tea" placeholderTextColor={colors.muted} value={plain} onChangeText={setPlain} style={styles.field} />
            <TextInput accessibilityLabel="Full approved wording" placeholder="my Lipton tea with milk" placeholderTextColor={colors.muted} value={surface} onChangeText={setSurface} style={styles.field} />
            <Text style={styles.fieldLabel}>Where is this detail useful?</Text>
            <View style={styles.chipWrap}>{contexts.map((item) => <Chip key={item.value} label={item.label} selected={detailSettings.includes(item.value)} onPress={() => setDetailSettings((current) => toggleValue(current, item.value))} />)}</View>
            {draft.audiences.length > 0 && <><Text style={styles.fieldLabel}>Who already knows the ordinary meaning?</Text><View style={styles.chipWrap}>{draft.audiences.map((item) => <Chip key={item.id} label={item.label} selected={knownBy.includes(item.id)} onPress={() => setKnownBy((current) => toggleValue(current, item.id))} />)}</View></>}
            <PrimaryButton label="Add detail" onPress={addDetail} disabled={!plain.trim() || !surface.trim()} compact />
          </View>
        </View>
      )}

      {step === 4 && (
        <View style={styles.questionStack}>
          <Text style={styles.sheetNote}>This is the exact profile Echora will use. Named people are never selected from location alone.</Text>
          <InfoPair label="Profile" value={draft.label} />
          <InfoPair label="Usual style" value={styleName(draft.style)} />
          {contexts.map((item) => <InfoPair key={item.value} label={item.label} value={`${draft.listener_by_setting[item.value] || listenerLabel(item.value)} · ${styleName(draft.style_by_setting[item.value] || draft.style)}`} />)}
          <SheetHeading text="Listener icons" />
          {draft.audiences.length ? draft.audiences.map((item) => <InfoPair key={item.id} label={item.label} value={`${item.listener} · ${styleName(item.style)} · ${item.visible_in_settings.length ? item.visible_in_settings.join(", ") : "all settings"}`} />) : <Text style={styles.sheetNote}>No named listeners. “Usual here” will always be available.</Text>}
          <InfoPair label="Known words" value={`${draft.lexicon.length}`} />
          <InfoPair label="Approved details" value={`${draft.specializations.length}`} />
          <PrimaryButton label="Approve and save profile" onPress={() => void save()} compact />
        </View>
      )}
      </View>

      <View style={styles.questionNav}>
        {step > 0 && <SecondaryButton label="Back" onPress={() => setStep((current) => current - 1)} />}
        {step < stepTitles.length - 1 && <PrimaryButton label="Continue" onPress={() => setStep((current) => current + 1)} compact />}
      </View>
      {status ? <Text accessibilityLiveRegion="polite" style={styles.sheetNote}>{status}</Text> : null}
    </View>
  );
}

function AboutBody() {
  const promises = [
    ["01", "Your words stay visible", "The literal transcript is never replaced by a cleaned-up message."],
    ["02", "Uncertainty stays honest", "When the evidence is unclear, Echora asks you instead of pretending."],
    ["03", "You are in control", "A message you choose is spoken straight away, and stays editable so you can say it again."],
  ];
  return (
    <View style={styles.sheetStack}>
      {promises.map(([number, title, body]) => (
        <View key={number} style={styles.promise}>
          <Text style={styles.promiseNumber}>{number}</Text>
          <View style={styles.fill}><Text style={styles.promiseTitle}>{title}</Text><Text style={styles.sheetNote}>{body}</Text></View>
        </View>
      ))}
      <Text style={styles.sheetNote}>The last approved message can support a follow-up for 10 minutes. Only messages you explicitly Remember are saved for future retrieval. Manage or delete them in your profile. Profiles and places stay in the local Echora service; selected cloud features send their required inputs to the provider.</Text>
      <Text selectable style={styles.apiNote}>API · {API_URL}</Text>
    </View>
  );
}

function FormArea({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <View style={styles.formGroup}>
      <Text style={styles.fieldLabel}>{label}</Text>
      <TextInput accessibilityLabel={label} multiline value={value} onChangeText={onChange} style={[styles.field, styles.area]} />
    </View>
  );
}

function InfoPair({ label, value }: { label: string; value: string }) {
  return <View style={styles.infoPair}><Text style={styles.infoLabel}>{label}</Text><Text style={styles.infoValue}>{value}</Text></View>;
}

function SheetHeading({ text }: { text: string }) {
  return <Text accessibilityRole="header" style={styles.sheetHeading}>{text}</Text>;
}

function Chip({ label, selected = false, onPress }: { label: string; selected?: boolean; onPress: () => void }) {
  return (
    <Pressable accessibilityRole="radio" accessibilityState={{ checked: selected }} onPress={onPress} style={({ pressed }) => [styles.chip, selected && styles.chipSelected, pressed && styles.pressed]}>
      <Text numberOfLines={1} style={[styles.chipText, selected && styles.chipSelectedText]}>{label}</Text>
    </Pressable>
  );
}

function PrimaryButton({ label, onPress, disabled = false, compact = false }: { label: string; onPress: () => void; disabled?: boolean; compact?: boolean }) {
  return (
    <Pressable accessibilityRole="button" accessibilityState={{ disabled }} disabled={disabled} onPress={onPress} style={({ pressed }) => [styles.primaryButton, compact && styles.compactButton, disabled && styles.disabled, pressed && styles.pressed]}>
      <Text style={styles.primaryButtonText}>{label}</Text>
    </Pressable>
  );
}

function SecondaryButton({ label, onPress, disabled = false, danger = false }: { label: string; onPress: () => void; disabled?: boolean; danger?: boolean }) {
  return (
    <Pressable accessibilityRole="button" accessibilityState={{ disabled }} disabled={disabled} onPress={onPress} style={({ pressed }) => [styles.secondaryButton, danger && styles.dangerButton, disabled && styles.disabled, pressed && styles.pressed]}>
      <Text style={[styles.secondaryButtonText, danger && styles.dangerText]}>{label}</Text>
    </Pressable>
  );
}

function LinkButton({ label, onPress }: { label: string; onPress: () => void }) {
  return <Pressable accessibilityRole="button" hitSlop={8} onPress={onPress} style={({ pressed }) => [styles.linkButton, pressed && styles.pressed]}><Text style={styles.linkText}>{label}</Text></Pressable>;
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.cream },
  shell: { flex: 1, backgroundColor: colors.cream },
  fill: { flex: 1 },
  header: { borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: colors.line, paddingHorizontal: 16, paddingBottom: 8, gap: 7 },
  brandRow: { minHeight: 44, flexDirection: "row", alignItems: "center", gap: 9 },
  brandMark: { width: 25, height: 24, flexDirection: "row", alignItems: "flex-end", gap: 2 },
  brand: { color: colors.ink, fontSize: 22, fontWeight: "800", letterSpacing: -0.4 },
  status: { marginLeft: "auto", maxWidth: "55%", flexDirection: "row", alignItems: "center", gap: 6 },
  statusDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: colors.coral },
  statusDotReady: { backgroundColor: colors.sage },
  statusText: { color: colors.muted, fontSize: 12, fontWeight: "600" },
  headerActions: { gap: 7 },
  headerButton: { height: 36, justifyContent: "center", paddingHorizontal: 13, borderRadius: 18, borderWidth: 1, borderColor: colors.line, backgroundColor: colors.paper },
  headerButtonStrong: { backgroundColor: colors.green, borderColor: colors.green },
  headerButtonText: { maxWidth: 120, color: colors.ink, fontSize: 13, fontWeight: "700" },
  headerButtonStrongText: { color: colors.white },
  stage: { flex: 1, minHeight: 0 },
  centerStage: { flexGrow: 1, alignItems: "center", justifyContent: "center", paddingHorizontal: 22, paddingVertical: 20, gap: 14 },
  scrollStage: { paddingHorizontal: 18, paddingVertical: 22, gap: 14 },
  composeStage: { alignItems: "center", paddingHorizontal: 18, paddingVertical: 18, gap: 14 },
  micHalo: { width: 138, height: 138, borderRadius: 69, alignItems: "center", justifyContent: "center", backgroundColor: colors.sageLight },
  micButton: { width: 102, height: 102, borderRadius: 51, alignItems: "center", justifyContent: "center", backgroundColor: colors.green, shadowColor: colors.greenDark, shadowOpacity: 0.22, shadowRadius: 14, shadowOffset: { width: 0, height: 7 } },
  micPressed: { transform: [{ scale: 0.97 }] },
  micGlyph: { color: colors.white, fontSize: 30, lineHeight: 32 },
  micStem: { width: 22, height: 7, borderRadius: 4, borderTopWidth: 3, borderColor: colors.white, marginTop: -3 },
  stageTitle: { color: colors.ink, fontSize: 24, fontWeight: "700", textAlign: "center" },
  heading: { color: colors.ink, fontSize: 27, fontWeight: "800", letterSpacing: -0.5, textAlign: "center" },
  stageHint: { color: colors.muted, fontSize: 15, lineHeight: 21, textAlign: "center" },
  legend: { color: colors.ink, fontSize: 13, fontWeight: "800", textTransform: "uppercase", letterSpacing: 0.8, marginTop: 8 },
  chipWrap: { flexDirection: "row", flexWrap: "wrap", alignItems: "center", justifyContent: "center", gap: 8 },
  chip: { minHeight: 42, maxWidth: 210, justifyContent: "center", paddingHorizontal: 15, borderRadius: 21, borderWidth: 1, borderColor: colors.line, backgroundColor: colors.paper },
  chipSelected: { backgroundColor: colors.green, borderColor: colors.green },
  chipText: { color: colors.ink, fontSize: 14, fontWeight: "600" },
  chipSelectedText: { color: colors.white },
  detected: { maxWidth: 440, color: colors.muted, lineHeight: 20, textAlign: "center" },
  audienceRow: { alignItems: "flex-start", justifyContent: "center", gap: 9, paddingHorizontal: 4 },
  audienceButton: { width: 82, minHeight: 98, alignItems: "center", gap: 6, borderRadius: 16, paddingVertical: 7, paddingHorizontal: 4, borderWidth: 1, borderColor: "transparent" },
  audienceButtonSelected: { borderColor: colors.green, backgroundColor: colors.paper },
  audienceAvatar: { width: 54, height: 54, borderRadius: 27, alignItems: "center", justifyContent: "center", borderWidth: 1, borderColor: colors.line },
  audienceAvatarSelected: { backgroundColor: colors.green, borderColor: colors.green },
  audienceGlyph: { color: colors.greenDark, fontSize: 18, fontWeight: "900" },
  audienceGlyphSelected: { color: colors.white },
  audienceLabel: { color: colors.ink, fontSize: 12, lineHeight: 15, fontWeight: "700", textAlign: "center" },
  audienceLabelSelected: { color: colors.greenDark },
  resolvedCard: { width: "100%", maxWidth: 430, borderRadius: 13, paddingHorizontal: 14, paddingVertical: 10, alignItems: "center", backgroundColor: colors.sageLight, gap: 2 },
  resolvedTitle: { color: colors.ink, fontSize: 15, fontWeight: "800" },
  resolvedText: { color: colors.muted, fontSize: 13, textTransform: "capitalize" },
  bold: { fontWeight: "800", color: colors.ink },
  recordingLabel: { color: colors.ink, fontSize: 17, fontWeight: "700" },
  recordingDot: { color: colors.coral },
  waveform: { height: 64, flexDirection: "row", alignItems: "center", gap: 4 },
  waveBar: { width: 4, minHeight: 7, maxHeight: 58, borderRadius: 3, backgroundColor: colors.green },
  progressTrack: { width: "82%", maxWidth: 390, height: 7, borderRadius: 4, backgroundColor: colors.line, overflow: "hidden" },
  progressFill: { height: 7, backgroundColor: colors.green, borderRadius: 4 },
  warning: { alignSelf: "stretch", color: colors.danger, backgroundColor: colors.coralLight, padding: 10, borderRadius: 10, textAlign: "center" },
  cardList: { gap: 12 },
  candidate: { borderWidth: 1, borderColor: colors.line, borderRadius: 20, padding: 18, backgroundColor: colors.paper, gap: 10, shadowColor: colors.ink, shadowOpacity: 0.05, shadowRadius: 10, shadowOffset: { width: 0, height: 4 } },
  cardPressed: { borderColor: colors.green, transform: [{ scale: 0.992 }] },
  candidateTop: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", gap: 8 },
  optionLabel: { color: colors.green, fontWeight: "800" },
  weight: { color: colors.muted, fontSize: 11, flexShrink: 1, textAlign: "right" },
  candidateMessage: { color: colors.ink, fontSize: 22, lineHeight: 29, fontWeight: "600" },
  literalLabel: { color: colors.muted, fontSize: 10, fontWeight: "800", letterSpacing: 1 },
  literalText: { color: colors.muted, fontSize: 14, lineHeight: 20 },
  chooseLabel: { color: colors.green, fontWeight: "800", marginTop: 2 },
  pills: { alignItems: "center", gap: 8, paddingHorizontal: 2 },
  messageInput: { alignSelf: "stretch", minHeight: 116, maxHeight: 220, borderRadius: 18, borderWidth: 1, borderColor: colors.line, backgroundColor: colors.paper, color: colors.ink, fontSize: 24, lineHeight: 31, padding: 17 },
  detailNote: { alignSelf: "stretch", padding: 13, borderRadius: 13, backgroundColor: colors.sageLight, gap: 6 },
  detailText: { color: colors.ink, lineHeight: 20 },
  evidenceCaption: { color: colors.muted, fontSize: 13 },
  inlineLink: { color: colors.green, fontWeight: "800", textDecorationLine: "underline" },
  speakPanel: { alignSelf: "stretch", borderRadius: 18, padding: 15, backgroundColor: colors.paper, borderWidth: 1, borderColor: colors.line, gap: 13 },
  speakStatus: { flexDirection: "row", alignItems: "center", gap: 11 },
  speakIcon: { width: 34, color: colors.green, fontSize: 25, fontWeight: "700", textAlign: "center" },
  speakTitle: { color: colors.ink, fontWeight: "800", fontSize: 16 },
  speakHint: { color: colors.muted, fontSize: 12, marginTop: 2 },
  actionRow: { flexDirection: "row", alignItems: "center", justifyContent: "flex-end", gap: 9 },
  actionRowWrap: { flexDirection: "row", flexWrap: "wrap", alignItems: "center", gap: 8 },
  errorBanner: { width: "100%", maxWidth: 440, borderLeftWidth: 5, borderLeftColor: colors.danger, backgroundColor: colors.coralLight, borderRadius: 14, padding: 18, gap: 7 },
  errorTitle: { color: colors.danger, fontSize: 19, fontWeight: "800" },
  errorText: { color: colors.ink, fontSize: 15, lineHeight: 22 },
  footer: { minHeight: 44, borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: colors.line, paddingHorizontal: 14, paddingVertical: 7, alignItems: "center" },
  srOnly: { position: "absolute", width: 1, height: 1, opacity: 0 },
  pressed: { opacity: 0.68 },
  disabled: { opacity: 0.42 },
  primaryButton: { minWidth: 210, minHeight: 54, borderRadius: 27, alignItems: "center", justifyContent: "center", paddingHorizontal: 22, backgroundColor: colors.green },
  compactButton: { minWidth: 120, minHeight: 46, borderRadius: 23 },
  primaryButtonText: { color: colors.white, fontSize: 16, fontWeight: "800" },
  secondaryButton: { minHeight: 44, borderRadius: 22, alignItems: "center", justifyContent: "center", paddingHorizontal: 16, borderWidth: 1, borderColor: colors.green, backgroundColor: colors.paper },
  secondaryButtonText: { color: colors.green, fontWeight: "800" },
  dangerButton: { borderColor: colors.danger },
  dangerText: { color: colors.danger },
  linkButton: { minHeight: 44, justifyContent: "center", paddingHorizontal: 8 },
  linkText: { color: colors.green, fontSize: 15, fontWeight: "800", textDecorationLine: "underline" },
  sheetStack: { gap: 14 },
  sheetNote: { color: colors.muted, fontSize: 14, lineHeight: 20 },
  sheetHeading: { color: colors.ink, fontSize: 18, fontWeight: "800", marginTop: 6 },
  evidenceRow: { borderRadius: 12, padding: 13, backgroundColor: colors.sageLight, gap: 5 },
  evidenceText: { color: colors.ink, fontSize: 17, fontWeight: "600" },
  codeText: { color: colors.muted, fontFamily: Platform.select({ ios: "Menlo", default: "monospace" }), fontSize: 11 },
  infoPair: { gap: 4 },
  infoLabel: { color: colors.muted, fontSize: 11, fontWeight: "800", letterSpacing: 0.7, textTransform: "uppercase" },
  infoValue: { color: colors.ink, fontSize: 15, lineHeight: 21 },
  evidenceOption: { borderWidth: 1, borderColor: colors.line, borderRadius: 15, padding: 14, gap: 11 },
  repairNote: { color: colors.muted, fontStyle: "italic", fontSize: 13 },
  personaCard: { minHeight: 78, flexDirection: "row", alignItems: "center", gap: 12, borderWidth: 1, borderColor: colors.line, borderRadius: 16, padding: 13, backgroundColor: colors.paper },
  personaCardSelected: { borderColor: colors.green, backgroundColor: colors.sageLight },
  personaMark: { width: 45, height: 45, borderRadius: 23, alignItems: "center", justifyContent: "center", backgroundColor: colors.cream },
  personaMarkSelected: { backgroundColor: colors.green },
  personaGlyph: { color: colors.green, fontSize: 25, fontWeight: "800" },
  personaGlyphSelected: { color: colors.white },
  profileEditorBanner: { flexDirection: "row", alignItems: "center", gap: 12, padding: 13, borderWidth: 1, borderColor: colors.line, borderRadius: 16, backgroundColor: colors.sageLight },
  personaLabel: { color: colors.ink, fontSize: 16, fontWeight: "800" },
  personaBlurb: { color: colors.muted, fontSize: 13, lineHeight: 18, marginTop: 3 },
  toggleRow: { flexDirection: "row", alignItems: "center", gap: 12, borderRadius: 15, padding: 13, backgroundColor: colors.sageLight },
  toggleTitle: { color: colors.ink, fontWeight: "800", fontSize: 15 },
  placeCard: { borderWidth: 1, borderColor: colors.line, borderRadius: 17, padding: 14, gap: 10, backgroundColor: colors.paper },
  placeTitle: { color: colors.ink, fontSize: 18, fontWeight: "800" },
  fieldLabel: { color: colors.ink, fontSize: 13, fontWeight: "800", marginTop: 3 },
  choiceStack: { gap: 4 },
  choice: { minHeight: 42, flexDirection: "row", alignItems: "center", gap: 9 },
  radio: { width: 22, height: 22, borderRadius: 11, borderWidth: 2, borderColor: colors.line, alignItems: "center", justifyContent: "center" },
  radioSelected: { borderColor: colors.green },
  radioDot: { width: 10, height: 10, borderRadius: 5, backgroundColor: colors.green },
  choiceText: { color: colors.ink, flex: 1 },
  addCard: { borderRadius: 17, padding: 14, gap: 11, backgroundColor: colors.sageLight },
  field: { minHeight: 48, borderWidth: 1, borderColor: colors.line, borderRadius: 11, paddingHorizontal: 12, paddingVertical: 10, backgroundColor: colors.paper, color: colors.ink, fontSize: 15 },
  area: { minHeight: 74, textAlignVertical: "top" },
  onboarding: { borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: colors.line, paddingTop: 18, gap: 15 },
  questionnaireHead: { gap: 8, paddingHorizontal: 2, paddingBottom: 2 },
  stepTopline: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", gap: 12 },
  stepCount: { color: colors.green, fontSize: 10, fontWeight: "900", letterSpacing: 1.1, textTransform: "uppercase" },
  stepRail: { height: 5, flexDirection: "row", gap: 5, marginVertical: 3 },
  stepSegment: { flex: 1, height: 5, borderRadius: 3, backgroundColor: colors.line },
  stepSegmentComplete: { backgroundColor: colors.green },
  questionnaireTitle: { color: colors.ink, fontSize: 28, lineHeight: 33, fontWeight: "800", letterSpacing: -0.6 },
  questionnaireCopy: { maxWidth: 480, color: colors.muted, fontSize: 14, lineHeight: 20 },
  questionPanel: { borderWidth: 1, borderColor: colors.line, borderRadius: 20, padding: 15, backgroundColor: colors.paper, shadowColor: colors.ink, shadowOpacity: 0.045, shadowRadius: 18, shadowOffset: { width: 0, height: 8 } },
  questionStack: { gap: 14 },
  questionNav: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", gap: 10, paddingTop: 3 },
  styleChoiceGrid: { gap: 8 },
  styleChoice: { minHeight: 70, justifyContent: "center", gap: 5, paddingHorizontal: 14, paddingVertical: 11, borderWidth: 1, borderColor: colors.line, borderRadius: 14, backgroundColor: colors.cream },
  styleChoiceSelected: { borderColor: colors.green, backgroundColor: colors.sageLight, shadowColor: colors.greenDark, shadowOpacity: 0.07, shadowRadius: 8, shadowOffset: { width: 0, height: 3 } },
  styleChoiceTopline: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", gap: 8 },
  styleChoiceTitle: { color: colors.ink, fontSize: 15, fontWeight: "800" },
  styleChoiceTitleSelected: { color: colors.greenDark },
  styleChoiceExample: { color: colors.muted, fontSize: 13, lineHeight: 18 },
  styleChoiceCheck: { width: 24, height: 24, borderRadius: 12, color: colors.white, backgroundColor: colors.green, textAlign: "center", lineHeight: 24, fontWeight: "900", overflow: "hidden" },
  audienceSummary: { flexDirection: "row", flexWrap: "wrap", alignItems: "center", gap: 9, borderWidth: 1, borderColor: colors.line, borderRadius: 15, padding: 10, backgroundColor: colors.paper },
  overrideCard: { gap: 7, borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: colors.line, paddingTop: 9 },
  listRow: { flexDirection: "row", alignItems: "center", gap: 10, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: colors.line, paddingVertical: 9 },
  formGroup: { gap: 6 },
  promise: { flexDirection: "row", gap: 13, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: colors.line, paddingBottom: 14 },
  promiseNumber: { color: colors.coral, fontSize: 15, fontWeight: "800" },
  promiseTitle: { color: colors.ink, fontSize: 17, fontWeight: "800", marginBottom: 4 },
  apiNote: { color: colors.muted, fontFamily: Platform.select({ ios: "Menlo", default: "monospace" }), fontSize: 11 },
});
