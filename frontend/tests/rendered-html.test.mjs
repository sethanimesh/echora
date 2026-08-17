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
