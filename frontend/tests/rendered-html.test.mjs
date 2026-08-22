import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

test("the communication flow preserves literal evidence and speaks the chosen message", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  assert.match(page, /Based on literal evidence/);
  // A chosen message is spoken without a confirmation step, so the old gate must
  // stay gone -- but choosing must still be what triggers speech, never the raw
  // ambiguous screen.
  assert.doesNotMatch(page, /Confirm this message|setConfirmed/);
  assert.match(page, /api\/v1\/speech/);
  assert.match(page, /function choose\([\s\S]*?void speak\(candidate\.corrected_text\)/);
  // Raw beams reach the screen as a single "ambiguous" message when the chain is
  // unavailable; auto-speech must follow the ranker, not the option count.
  assert.match(page, /next\.ranker\.decision === "selected"/);
  assert.match(page, /speechSynthesis/);
  assert.match(page, /MediaRecorder/);
  assert.match(page, /search weight/);
  assert.match(page, /Where are you speaking/);
  assert.match(page, /Interpreted meaning/);
  assert.match(page, /form\.append\("context", context\)/);
});

test("the starter preview and persistence dependencies are absent", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  const packageJson = await readFile(new URL("package.json", root), "utf8");
  assert.doesNotMatch(page, /_sites-preview|react-loading-skeleton/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton|drizzle/);
});

test("the page exposes accessible recording and result controls", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  assert.match(page, /aria-label="Start recording"/);
  assert.match(page, /aria-live="polite"/);
  assert.match(page, /role="alert"/);
  assert.match(page, /aria-pressed/);
});

test("recording and recovery states are implemented", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  assert.match(page, /NotAllowedError/);
  assert.match(page, /Choose an audio file/);
  assert.match(page, /Which one sounds right/);
  assert.match(page, /Here’s the clearest match/);
  // Groq audio is an enhancement: every failure path still reaches the browser voice.
  assert.match(page, /catch \{\s*await browserSpeak\(spoken\);/);
  assert.match(page, /window\.speechSynthesis/);
  assert.match(page, /navigator\.clipboard/);
  assert.match(page, /Speak again|Speak this/);
});

test("the shell is locked to a single screen", async () => {
  const css = await readFile(new URL("app/globals.css", root), "utf8");
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  assert.match(css, /100dvh/);
  assert.match(css, /html, body \{[^}]*overflow: hidden/);
  assert.match(css, /grid-template-rows: auto minmax\(0, 1fr\) auto/);
  assert.doesNotMatch(page, /setBars\(/);
});

test("places resolve to a built-in setting and never become one themselves", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  // The four settings stay a closed set. A place borrows one; it never adds one,
  // so the message chain can never be handed a setting it does not know.
  assert.match(page, /type CommunicationContext = "general" \| "home" \| "care" \| "outdoors";/);
  assert.match(page, /context: CommunicationContext;/);
  // The transcription request is untouched: still the setting, never a place id.
  assert.match(page, /form\.append\("context", context\)/);
  assert.doesNotMatch(page, /form\.append\("place"/);
  assert.match(page, /const nextContext: CommunicationContext = chosen \? chosen\.context : "general";/);
});

test("who is listening travels beside the setting, and is its own closed set", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  // Two values, orthogonal to the four settings. A place says which one applies
  // where; it still never becomes a setting itself.
  assert.match(page, /type Listener = "familiar" \| "unfamiliar";/);
  assert.doesNotMatch(page, /form\.append\("listener", place/);
  // A place that declares nothing sends nothing. Filling in the setting's own
  // default here would look identical to a declaration and would silently
  // outrank the speaker's profile, which is the middle rung of the chain.
  assert.match(page, /form\.append\("listener", listener \?\? ""\)/);
  assert.match(page, /setListener\(chosen\?\.listener \?\? null\)/);
  assert.doesNotMatch(page, /setListener\([^)]*defaultListener/);
  // The chain itself lives on the server, in one expression. The browser only
  // ever reports what the place said.
  assert.match(page, /const \[listener, setListener\] = useState<Listener \| null>\(null\)/);
  assert.doesNotMatch(page, /setListener\([\s\S]{0,80}listener_by_setting/);
  // Three states, not two, because "leave it to whoever is speaking" is the
  // only one that lets a profile be heard -- so it has to be choosable.
  assert.match(page, /function setPlaceListener\(target: Place, next: Listener \| null\)/);
  assert.match(page, /<option value="">Leave it to whoever is speaking<\/option>/);
  // Offered on the built-ins too: the shipped Outdoors has to be able to keep
  // the familiar register for someone who never goes out alone.
  assert.doesNotMatch(page, /item\.builtin && .{0,40}setPlaceListener/);
});

test("an accepted message is stamped with what the server resolved, not local state", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  // The two differ exactly when the profile decided the listener. A record filed
  // under the wrong stance is retrieved later as an example of a shape this
  // speaker never used.
  assert.match(page, /context: from\.context,/);
  assert.match(page, /listener: from\.listener,/);
  // Raw ASR shown because the assistant was unavailable is evidence, not a
  // message. Storing it teaches the profile the recognizer's typos.
  assert.match(page, /if \(candidate\.repair_status === "unavailable"\) return;/);
});

test("location is resolved in the browser and never blocks speaking", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  assert.match(page, /navigator\.geolocation/);
  // Coordinates are matched here, so they never leave the machine -- only the
  // borrowed setting is sent, through the field that already existed.
  assert.match(page, /function nearestPlace\(/);
  assert.match(page, /function metersBetween\(/);
  assert.doesNotMatch(page, /api\/v1\/places\/resolve/);
  // Detection only ever runs on the idle screen, so a reading cannot land
  // mid-utterance, and a refused permission just leaves the chips in charge.
  assert.match(page, /if \(stage !== "idle"\) return;/);
  assert.match(page, /PERMISSION_DENIED/);
});

test("a detected setting is shown, and a tap outranks it", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  assert.match(page, /Detected <b>\{detected\.label\}<\/b>/);
  assert.match(page, /manualPlaceRef/);
  assert.match(page, /function pickPlace\([\s\S]*?manualPlaceRef\.current = true/);
});

test("the places panel cannot leave the speaker without the shipped places", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  const css = await readFile(new URL("app/globals.css", root), "utf8");
  assert.match(page, /api\/v1\/places/);
  // Remove is offered only on a place the speaker added.
  assert.match(page, /\{!item\.builtin && \([\s\S]*?Remove/);
  // The idle row is capped because that screen cannot scroll; the rest lives in
  // the sheet, which can.
  assert.match(page, /const IDLE_CHIPS = 5;/);
  assert.match(page, /\{hidden\} more…/);
  assert.match(css, /\.chip-more \{/);
});

test("the header cannot be clipped by its own actions", async () => {
  const css = await readFile(new URL("app/globals.css", root), "utf8");
  // The shell's implicit column is auto-sized, so the bar's min-content -- a row of
  // nowrap buttons that grows by two whenever there is a result -- could force it
  // wider than the shell and take the stage and footer with it, clipped in silence
  // by `overflow: hidden`. Both axes are pinned for the same reason.
  assert.match(css, /grid-template-columns: minmax\(0, 1fr\)/);
  // Height is the cheaper thing to give: the stage row is minmax(0, 1fr) and already
  // drops its own optional prose, so the bar wraps rather than hiding an action.
  assert.match(css, /\.bar \{[^}]*flex-wrap: wrap/);
  assert.match(css, /\.bar \{[^}]*min-height: var\(--bar-h\)/);
  // A profile name is arbitrary text and must not be able to widen the bar.
  assert.match(css, /\.persona-action span \{[^}]*text-overflow: ellipsis/);
});

test("the orb is decoration and never touches recognition or speech", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  const css = await readFile(new URL("app/globals.css", root), "utf8");

  // The bars stayed hand-rolled: levels go straight to the DOM, and the orb takes
  // one scalar from the same loop rather than opening a second AudioContext.
  assert.match(page, /style\.setProperty\("--l"/);
  assert.match(page, /inputVolumeRef\.current = /);
  assert.equal(page.match(/new AudioContext\(\)/g).length, 1);
  assert.equal(page.match(/await navigator\.mediaDevices\.getUserMedia/g).length, 1);

  // The tuning that makes difficult speech recognisable is not a style choice.
  assert.match(page, /noiseSuppression: false/);
  assert.match(page, /autoGainControl: false/);

  // Speech is never routed through an AudioContext to be visualised: a suspended
  // context would take the voice with it, and silence leaves the speaker unheard.
  assert.doesNotMatch(page, /createMediaElementSource/);

  // Skipped outright under reduced motion, which also means three.js is never
  // fetched; and it is lazy, so it stays out of the first paint either way.
  assert.match(page, /const orbReady = useHydrated\(\) && !reducedMotion;/);
  assert.match(page, /lazy\(\(\) => import\("@\/components\/ui\/orb"\)/);
  assert.match(page, /prefers-reduced-motion/);

  // It is the centrepiece of exactly two stages, in flow so the stage lays out
  // around it, and bounded against the container so a short window shrinks it
  // rather than pushing anything off a screen that cannot scroll.
  assert.match(css, /\.orb-slot \{[^}]*--orb-size: min\(calc\(var\(--mic\)/);
  assert.match(css, /\.orb-slot \{[^}]*42cqh/);
  assert.match(css, /\.orb-slot \{[^}]*pointer-events: none/);
  // Blurred in proportion to itself: a fixed blur erases it once --mic clamps
  // down on a short window, which is exactly how it went missing before.
  assert.match(css, /filter: blur\(calc\(var\(--orb-size\)/);
  assert.equal(page.match(/<StageOrb/g).length, 2);
  assert.match(page, /<StageOrb state="listening"/);
  assert.match(page, /<StageOrb state="talking"/);
});

test("the orb is self-contained and never reaches the network", async () => {
  const orb = await readFile(new URL("components/ui/orb.tsx", root), "utf8");
  // Upstream suspends on a texture fetched from a Google CDN. In a local-first tool
  // that is a hard network dependency, a request this app does not want to make,
  // and -- because useTexture suspends -- a silent way for the orb to never appear.
  assert.doesNotMatch(orb, /useTexture\(/);
  assert.doesNotMatch(orb, /new THREE\.TextureLoader\(/);
  assert.doesNotMatch(orb, /@react-three\/drei/);
  assert.doesNotMatch(orb, /load\("https:/);
  assert.match(orb, /new THREE\.DataTexture/);
  // The seven lobe centres collapse onto four angles, so an opaque white base shows
  // through the gaps as a pie chart. Transparent, the gaps are simply the page.
  assert.match(orb, /vec4 color = vec4\(1\.0, 1\.0, 1\.0, 0\.0\);/);
});

test("the saved-state promise matches what is actually stored", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  // Places persist on this machine now, so the old blanket claim would overstate it.
  assert.doesNotMatch(page, /No history is saved after refresh/);
  assert.match(page, /No conversation is saved after refresh/);
  assert.match(page, /stay on\s*\n?\s*this machine|stay on this machine/);
});
