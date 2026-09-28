import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

test("the native client preserves the server's listener precedence", async () => {
  const app = await readFile(new URL("App.tsx", root), "utf8");
  const api = await readFile(new URL("src/api.ts", root), "utf8");

  assert.match(app, /const \[listener, setListener\] = useState<Listener \| null>\(null\)/);
  assert.match(app, /setListener\(chosen\?\.listener \?\? null\)/);
  assert.match(api, /declared_listener: listener/);
  assert.match(api, /audience_id: audience/);
  assert.match(api, /\/audience\/resolve/);
  assert.doesNotMatch(app, /setListener\(LISTENER_DEFAULTS/);
});

test("the idle screen resolves and freezes a dynamically selected audience", async () => {
  const app = await readFile(new URL("App.tsx", root), "utf8");

  assert.match(app, /Who are you speaking to\?/);
  assert.match(app, /function AudienceButton/);
  assert.match(app, /profile\?\.audiences/);
  assert.match(app, /setAudienceId\(""\)/);
  assert.match(app, /requestContextRef\.current = \{ context, persona, listener, audience: audienceId, place \}/);
  assert.match(app, /const frozen = requestContextRef\.current/);
});

test("the reviewed questionnaire persists setting and person-specific styles", async () => {
  const app = await readFile(new URL("App.tsx", root), "utf8");
  const api = await readFile(new URL("src/api.ts", root), "utf8");

  assert.match(app, /Step \{step \+ 1\} of \{stepTitles\.length\}/);
  assert.match(app, /style_by_setting/);
  assert.match(app, /visible_in_places/);
  assert.match(app, /Approve and save profile/);
  assert.match(api, /"PUT", \{revision, profile\}/);
  assert.match(api, /readProfile/);
});

test("only a reviewed unambiguous result speaks on arrival", async () => {
  const app = await readFile(new URL("App.tsx", root), "utf8");

  assert.match(app, /initialMessage && next\.auto_speak/);
  assert.match(app, /void speak\(initialMessage\.corrected_text, next\.speech\)/);
  assert.match(app, /function choose\([\s\S]*?next\.auto_speak[\s\S]*?void speak\(selected\.corrected_text\)/);
  assert.match(app, /const initial = next\.ranker\.decision === "selected" \? next\.recommended_message_id : null/);
  assert.doesNotMatch(app, /Confirm this message|confirmation/);
});

test("speech has an on-device fallback", async () => {
  const app = await readFile(new URL("App.tsx", root), "utf8");

  assert.match(app, /const authorized = await synthesizeSpeech\(spoken\)/);
  assert.match(app, /catch \{[\s\S]*?deviceSpeak\(authorized\.speech_text, generation\)/);
  assert.match(app, /Speech\.speak/);
});

test("native long-term memory requires the explicit Remember action", async () => {
  const api = await readFile(new URL("src/api.ts", root), "utf8");
  const app = await readFile(new URL("App.tsx", root), "utf8");
  assert.doesNotMatch(api, /api\/v1\/accepted|rememberAccepted/);
  assert.doesNotMatch(app, /rememberAccepted|history_size|examples_used/);
  assert.match(app, /onRemember=\{\(\) => void remember\(\)\}/);
  assert.match(api, /function rememberActiveMessage/);
});

test("place coordinates are matched locally and never sent with a transcription", async () => {
  const api = await readFile(new URL("src/api.ts", root), "utf8");
  const location = await readFile(new URL("src/location.ts", root), "utf8");

  assert.match(location, /export function nearestPlace/);
  assert.match(api, /core_context: context/);
  assert.doesNotMatch(api, /form\.append\("latitude"|form\.append\("longitude"|form\.append\("place"/);
});

test("literal evidence remains visible and search weight is not called confidence", async () => {
  const app = await readFile(new URL("App.tsx", root), "utf8");

  assert.match(app, /Literal ASR evidence/);
  assert.match(app, /Search weights compare only these beam hypotheses\. They are not calibrated confidence\./);
  assert.match(app, /candidate\.source_literals/);
});
