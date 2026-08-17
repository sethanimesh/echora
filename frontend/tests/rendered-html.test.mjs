import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

test("the communication flow preserves literal evidence and confirmation", async () => {
  const page = await readFile(new URL("app/page.tsx", root), "utf8");
  assert.match(page, /Based on literal evidence/);
  assert.match(page, /Confirm this message/);
  assert.match(page, /disabled=!confirmed|if \(!confirmed\)/);
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
  assert.match(page, /setConfirmed\(false\)/);
  assert.match(page, /window\.speechSynthesis/);
  assert.match(page, /navigator\.clipboard/);
});
