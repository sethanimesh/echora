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
  assert.match(page, /setContext\(chosen \? chosen\.context : "general"\)/);
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

test("the saved-state promise matches what is actually stored", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  // Places persist on this machine now, so the old blanket claim would overstate it.
  assert.doesNotMatch(page, /No history is saved after refresh/);
  assert.match(page, /No conversation is saved after refresh/);
  assert.match(page, /stay on\s*\n?\s*this machine|stay on this machine/);
});
