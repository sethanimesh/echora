import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import test from "node:test";
import ts from "typescript";

const source = await readFile(new URL("../src/api.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, {compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022}}).outputText;

function fixture(beforeResponse = async () => undefined, overrides = {}) {
  const calls = [];
  class FormData { fields = {}; append(key, value) { this.fields[key] = value; } }
  let job = {id: "job-one", revision: 3, status: "review", original: "tea", source_text: "tea", text: "tea", question: "Choose your message", options: [], candidates: [
    {id: "m1", text: "Please bring tea.", source_hypothesis_ids: ["h1"], source_literals: ["tea"]},
    {id: "m2", text: "Please bring coffee.", source_hypothesis_ids: ["h2"], source_literals: ["coffee"]}],
    evidence: {backend: "local", model: "adapted", hypotheses: [{id: "h1", literal_text: "tea", search_weight: .6}, {id: "h2", literal_text: "coffee", search_weight: .4}]}};
  job = {...job, ...overrides};
  const fakeFetch = async (url, options = {}) => {
    const body = options.body instanceof FormData ? options.body.fields : options.body ? JSON.parse(options.body) : null;
    calls.push({url, body, headers: options.headers});
    let response;
    if (url.endsWith("/session")) response = {session_token: "opaque-token"};
    else {
      assert.equal(options.headers["X-Echora-Session"], "opaque-token");
      if (url.endsWith("/audio")) response = job;
      else if (url.endsWith("/choose")) {
        assert.equal(body.revision, job.revision);
        const selected = job.candidates.find(candidate => candidate.id === body.candidate_id);
        job = {...job, text: selected.text, question: "", error: null, selected_candidate_id: selected.id, decision_source: "user",
          revision: job.revision + 1, auto_speak_revision: job.revision + 1};
        response = job;
      } else if (url.endsWith("/edit")) {
        assert.equal(body.revision, job.revision);
        const authored = {id: `edited-${job.revision + 1}`, text: body.text, reading: body.text,
          source_hypothesis_ids: [], source_literals: [], available: true, based_on_user_edit: true};
        const literals = job.evidence.hypotheses.map(item => ({id: `raw-${item.id}`, text: item.literal_text,
          reading: item.literal_text, source_hypothesis_ids: [item.id], source_literals: [item.literal_text], available: false}));
        job = {...job, text: body.text, revision: job.revision + 1, confirmed: null,
          question: "", options: [], candidates: [authored, ...literals], selected_candidate_id: authored.id, error: null,
          decision_source: "user", auto_speak_revision: body.selection ? job.revision + 1 : null};
        response = job;
      } else if (url.endsWith("/confirm")) {
        assert.equal(body.revision, job.revision);
        job = {...job, status: "confirmed", confirmed: {id: `decision-${job.revision}`, speech_text: job.text}};
        response = job;
      } else if (url.endsWith("/speech")) {
        assert.equal(body.revision, job.revision);
        assert.equal(body.confirmation_id, job.confirmed.id);
        response = {job, speech: null, speech_text: job.confirmed.speech_text};
      } else if (url.endsWith("/cancel")) {
        job = {...job, status: "cancelled", revision: job.revision + 1, confirmed: null};
        response = job;
      } else if (url.endsWith("/memories")) {
        response = {profile_id: job.context?.selection.profile_id || "person", memory_revision: 7, memories: []};
      } else if (url.endsWith("/remember")) {
        assert.equal(body.revision, job.revision);
        assert.equal(body.memory_revision, 7);
        response = {memory_id: "saved-one", memory_revision: 8, message_revision: job.revision, created: true,
          memory: {id: "saved-one", message: job.text, scope: {scenario: "home"}, created_at: 1790589600}};
      } else if (url.includes("/memories/") && (url.endsWith("/delete") || url.endsWith("/clear"))) {
        response = {profile_id: "person", memory_revision: body.memory_revision + 1, deleted_ids: ["saved-one"]};
      } else if (url.endsWith("/delete")) {
        response = {deleted: true, profile_id: "person"};
      } else if (url.endsWith("/state")) response = {job};
      else throw new Error(`Unexpected route: ${url}`);
    }
    await beforeResponse(url);
    return new Response(JSON.stringify(response), {status: 200});
  };
  const exports = {};
  vm.runInNewContext(compiled, {exports, require: name => {
    if (name === "expo/fetch") return {fetch: fakeFetch};
    if (name === "expo-file-system") return {File: class { constructor(uri) { this.uri = uri; } }};
    if (name === "react-native") return {NativeModules: {SourceCode: {scriptURL: "http://192.0.2.20:8081/index.bundle"}}};
    throw new Error(name);
  }, FormData, Response, URL, setTimeout, clearTimeout, process: {env: {}}});
  return {api: exports, calls, changeServerJob: patch => { job = {...job, ...patch}; }};
}

test("native observer rejects queued speech after another session deletes its context", async () => {
  let releaseSpeech;
  let speechStarted;
  const started = new Promise(resolve => { speechStarted = resolve; });
  const held = new Promise(resolve => { releaseSpeech = resolve; });
  const {api, changeServerJob} = fixture(async url => {
    if (url.endsWith('/speech')) { speechStarted(); await held; }
  });
  await api.transcribeAudio('recording.m4a', 'recording.m4a', 'home', '', null, '');
  await api.chooseMessage('m1');
  const speech = api.synthesizeSpeech('Please bring tea.');
  await started;
  changeServerJob({revision: 5, status: 'review', confirmed: null, text: 'tea',
    question: 'Personal context changed.', auto_speak_revision: null,
    ranking: {status: 'stale_context', route: 'verified', decision: 'ambiguous'}});
  let invalidated;
  const notice = new Promise(resolve => { invalidated = resolve; });
  const stop = api.watchMessageInvalidation(invalidated, () => assert.fail('State was available'));
  const next = await notice;
  stop();
  assert.equal(next.auto_speak, false);
  assert.equal(next.needs_user_choice, true);
  releaseSpeech();
  await assert.rejects(speech, /no longer active/);
});

test("native observer leaves ordinary confirmation updates authorized", async () => {
  const {api, calls} = fixture();
  await api.transcribeAudio('recording.m4a', 'recording.m4a', 'home', '', null, '');
  await api.chooseMessage('m1');
  const voice = await api.synthesizeSpeech('Please bring tea.');
  let invalidations = 0;
  const stop = api.watchMessageInvalidation(() => { invalidations++; }, () => assert.fail('State was available'));
  await new Promise(resolve => setTimeout(resolve, 10));
  stop();
  assert.equal(invalidations, 0);
  assert.equal(voice.speech_text, 'Please bring tea.');
  assert.ok(calls.some(call => call.url.endsWith('/state')));
});

test("missing verification stays silent with one suggestion even when no question was returned", async () => {
  const {api} = fixture(undefined, {question: '', error: null, text: 'Please bring tea.',
    candidates: [{id: 'm1', text: 'Please bring tea.', source_hypothesis_ids: ['h1'], source_literals: ['tea']}],
    selected_candidate_id: 'm1', auto_speak_revision: 3,
    ranking: {status: 'unavailable', route: 'verified', decision: 'ambiguous'}});
  const result = await api.transcribeAudio('recording.m4a', 'recording.m4a', 'home', '', null, '');
  assert.equal(result.auto_speak, false);
  assert.equal(result.needs_user_choice, true);
  assert.equal(result.recommended_message_id, null);
});

test("native choices authorize the same selected revision and retain all evidence", async () => {
  const {api, calls} = fixture();
  assert.equal(api.API_URL, "http://192.0.2.20:8000");
  const result = await api.transcribeAudio("recording.m4a", "recording.m4a", "outdoors", "", null, "", "station");
  assert.equal(result.hypotheses.length, 2);
  const selection = JSON.parse(calls.find(call => call.url.endsWith("/audio")).body.selection);
  assert.equal(selection.declared_listener, null);
  assert.equal(selection.place_id, "station");
  assert.equal(selection.scenario, "outside");
  await api.chooseMessage("m2");
  const voice = await api.synthesizeSpeech("Please bring coffee.");
  assert.equal(voice.speech_text, "Please bring coffee.");
  assert.equal(voice.speech, null);
  assert.equal(calls.find(call => call.url.endsWith("/confirm")).body.revision, 4);
});

test("a raw fallback stays silent on arrival and its explicit choice speaks the exact selected revision", async () => {
  const {api, calls} = fixture(undefined, {question: "", error: {message: "Wording is unavailable."},
    candidates: [{id: "raw-h1", text: "tea", available: false, source_hypothesis_ids: ["h1"], source_literals: ["tea"]}]});
  const arrival = await api.transcribeAudio("recording.m4a", "recording.m4a", "home", "", null, "");
  assert.equal(arrival.ranker.decision, "ambiguous");
  assert.equal(arrival.recommended_message_id, null);
  assert.equal(calls.filter(call => call.url.endsWith("/confirm") || call.url.endsWith("/speech")).length, 0);
  const chosen = await api.chooseMessage("raw-h1");
  assert.equal(chosen.ranker.decision, "selected");
  assert.equal(chosen.recommended_message_id, "raw-h1");
  const selected = chosen.messages.find(candidate => candidate.message_id === "raw-h1");
  assert.equal(selected.repair_status, "unavailable");
  const voice = await api.synthesizeSpeech(selected.corrected_text);
  assert.equal(voice.speech_text, "tea");
  assert.equal(calls.find(call => call.url.endsWith("/confirm")).body.revision, chosen.job_revision);
  assert.equal(calls.filter(call => call.url.endsWith("/edit")).length, 0);
});

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return {promise, resolve};
}

test("cancelling an upload rejects its late result and cancels the created server job", async () => {
  const arrived = deferred(), released = deferred();
  const {api, calls} = fixture(async url => {
    if (url.endsWith("/audio")) { arrived.resolve(); await released.promise; }
  });
  const recording = api.transcribeAudio("recording.m4a", "recording.m4a", "home", "", null, "");
  await arrived.promise;
  await api.cancelActiveMessage();
  released.resolve();
  await assert.rejects(recording, /Recording cancelled/);
  assert.equal(calls.filter(call => call.url.endsWith("/cancel")).length, 1);
  await assert.rejects(api.synthesizeSpeech("tea"), /no longer active/);
});

test("reset discards a late voice response while preserving the approved reference on the server", async () => {
  const arrived = deferred(), released = deferred();
  const {api, calls} = fixture(async url => {
    if (url.endsWith("/speech")) { arrived.resolve(); await released.promise; }
  });
  await api.transcribeAudio("recording.m4a", "recording.m4a", "home", "", null, "");
  await api.chooseMessage("m1");
  const speech = api.synthesizeSpeech("Please bring tea.");
  await arrived.promise;
  await api.cancelActiveMessage();
  released.resolve();
  await assert.rejects(speech, /no longer active/);
  assert.equal(calls.filter(call => call.url.endsWith("/cancel")).length, 0);
  await assert.rejects(api.synthesizeSpeech("Please bring tea."), /no longer active/);
});

test("a late voice response cannot overwrite a newer edit revision", async () => {
  const arrived = deferred(), released = deferred();
  const {api} = fixture(async url => {
    if (url.endsWith("/speech")) { arrived.resolve(); await released.promise; }
  });
  await api.transcribeAudio("recording.m4a", "recording.m4a", "home", "", null, "");
  await api.chooseMessage("m1");
  const speech = api.synthesizeSpeech("Please bring tea.");
  await arrived.promise;
  await api.editMessage("Please bring warm tea.");
  released.resolve();
  await assert.rejects(speech, /no longer active/);
  const current = await api.synthesizeSpeech("Please bring warm tea.");
  assert.equal(current.speech_text, "Please bring warm tea.");
});

test("queued native edits invalidate prior authorization before the next voice", async () => {
  const {api, calls} = fixture();
  await api.transcribeAudio("recording.m4a", "recording.m4a", "home", "", null, "");
  await api.chooseMessage("m1");
  await api.synthesizeSpeech("Please bring tea.");
  const first = api.editMessage("Please bring warm tea.");
  const second = api.editMessage("Please bring warm tea without sugar.");
  await Promise.all([first, second]);
  const voice = await api.synthesizeSpeech("Please bring warm tea without sugar.");
  assert.equal(voice.speech_text, "Please bring warm tea without sugar.");
  const edits = calls.filter(call => call.url.endsWith("/edit"));
  assert.deepEqual(edits.map(call => call.body.revision), [4, 5]);
  const confirms = calls.filter(call => call.url.endsWith("/confirm"));
  assert.deepEqual(confirms.map(call => call.body.revision), [4, 6]);
});


const fiveChoices = () => Array.from({length: 5}, (_, index) => ({
  id: `raw-h${index + 1}`, text: `literal ${index + 1}`, available: false,
  source_hypothesis_ids: [`h${index + 1}`], source_literals: [`literal ${index + 1}`],
}));

test("resolved recommendation is independent of autoplay and all five literal choices survive", async () => {
  const literals = fiveChoices();
  const chosen = {id: "generated-h4", text: "Please bring water.", available: true,
    source_hypothesis_ids: ["h4"], source_literals: ["literal 4"],
    retrieved_sources: [{source_id: "memory-1", kind: "message", text: "Remembered wording"}],
    contextual_additions: [{source_id: "memory-1", anchor: "water", wording: "cold water"}]};
  const {api, calls} = fixture(undefined, {question: "", text: chosen.text, selected_candidate_id: chosen.id,
    auto_speak_revision: null, candidates: [chosen, ...literals],
    ranking: {status: "ready", decision: "selected", selected_hypothesis_id: "h4", reason: "Matched audio"}});
  const result = await api.transcribeAudio("recording.m4a", "recording.m4a", "home", "", null, "");
  assert.equal(result.ranker.decision, "selected");
  assert.equal(result.recommended_message_id, "generated-h4");
  assert.equal(result.auto_speak, false);
  assert.equal(result.messages.length, 6);
  assert.equal(result.ranking.selected_hypothesis_id, "h4");
  assert.equal(result.messages[0].contextual_additions[0].wording, "cold water");
  assert.equal(result.messages[0].retrieved_sources[0].source_id, "memory-1");
  const explicit = await api.chooseMessage("raw-h5");
  assert.equal(explicit.auto_speak, true);
  const voice = await api.synthesizeSpeech("literal 5");
  assert.equal(voice.speech_text, "literal 5");
  assert.equal(calls.filter(call => call.url.endsWith("/draft") || call.url.endsWith("/edit")).length, 0);
});

test("ambiguous machine ranking is not resolved by a single completion or a speech marker", async () => {
  const {api} = fixture(undefined, {question: "", candidates: [{id: "m1", text: "Please bring tea."}],
    selected_candidate_id: "m1", auto_speak_revision: 3,
    ranking: {status: "ready", decision: "ambiguous", selected_hypothesis_id: null, reason: "Two interpretations remain"}});
  const result = await api.transcribeAudio("recording.m4a", "recording.m4a", "home", "", null, "");
  assert.equal(result.auto_speak, false);
  assert.equal(result.needs_user_choice, true);
  assert.equal(result.recommended_message_id, null);
  const explicit = await api.chooseMessage("m1");
  assert.equal(explicit.ranking.decision, "ambiguous");
  assert.equal(explicit.decision_source, "user");
  assert.equal(explicit.auto_speak, true);
});

test("unsupported verification preserves the legacy route's server-authorized playback", async () => {
  const {api} = fixture(undefined, {question: "", candidates: [{id: "m1", text: "Please bring tea."}],
    selected_candidate_id: "m1", auto_speak_revision: 3,
    ranking: {status: "unsupported", route: "legacy", decision: "ambiguous", selected_hypothesis_id: null, reason: "Remote acoustic-only evidence"}});
  const result = await api.transcribeAudio("recording.m4a", "recording.m4a", "home", "", null, "");
  assert.equal(result.auto_speak, true);
  assert.equal(result.ranker.decision, "selected");
  assert.equal(result.ranking.status, "unsupported");
});

test("personal clarification options take precedence over generated candidates", async () => {
  const {api, calls} = fixture(undefined, {question: "Add your usual detail?", options: ["Tea with milk", "Tea"],
    ranking: {status: "ready", decision: "ambiguous", selected_hypothesis_id: null, reason: "Choose a detail"}});
  const result = await api.transcribeAudio("recording.m4a", "recording.m4a", "home", "", null, "");
  assert.deepEqual(Array.from(result.messages, item => item.message_id), ["option-0", "option-1"]);
  assert.equal(result.auto_speak, false);
  const selected = await api.chooseMessage("option-1");
  assert.equal(selected.auto_speak, true);
  assert.equal(selected.recommended_message_id, "edited-4");
  assert.equal(selected.messages[0].corrected_text, "Tea");
  assert.equal(calls.find(call => call.url.endsWith("/edit")).body.selection, true);
  assert.equal(calls.filter(call => call.url.endsWith("/draft")).length, 0);
});

test("Remember explicitly saves the current edited revision without confirming or speaking", async () => {
  const {api, calls} = fixture(undefined, {context: {selection: {profile_id: "person"}}, question: ""});
  await api.transcribeAudio("recording.m4a", "recording.m4a", "home", "", null, "");
  assert.equal(calls.filter(call => call.url.endsWith("/remember")).length, 0);
  await api.chooseMessage("m1");
  const edits = api.editMessage("Please bring tea without sugar.");
  const save = api.rememberActiveMessage("Please bring tea without sugar.");
  await edits;
  const result = await save;
  assert.equal(result.memory.message, "Please bring tea without sugar.");
  assert.equal(result.message_revision, 5);
  assert.equal(calls.filter(call => call.url.endsWith("/remember")).length, 1);
  assert.equal(calls.filter(call => call.url.endsWith("/confirm") || call.url.endsWith("/speech")).length, 0);
});

test("a newer edit while loading memory revision prevents remembering the old message", async () => {
  const arrived = deferred(), released = deferred();
  const {api, calls} = fixture(async url => {
    if (url.endsWith("/memories")) { arrived.resolve(); await released.promise; }
  }, {context: {selection: {profile_id: "person"}}, question: ""});
  await api.transcribeAudio("recording.m4a", "recording.m4a", "home", "", null, "");
  const save = api.rememberActiveMessage("tea");
  await arrived.promise;
  await api.editMessage("coffee");
  released.resolve();
  await assert.rejects(save, /no longer active/);
  assert.equal(calls.filter(call => call.url.endsWith("/remember")).length, 0);
});

test("memory deletion uses explicit profile and memory revisions and does not speak", async () => {
  const {api, calls} = fixture();
  await api.deleteMemory("profile/one", "memory/one", 4);
  await api.clearMemories("profile/one", 5);
  await api.deleteProfile("profile/one", 9, 6);
  assert.equal(calls.find(call => call.url.includes("/memories/memory%2Fone/")).body.memory_revision, 4);
  assert.equal(calls.find(call => call.url.endsWith("/memories/clear")).body.memory_revision, 5);
  const removal = calls.find(call => call.url.endsWith("/profiles/profile%2Fone/delete"));
  assert.deepEqual(removal.body, {profile_revision: 9, memory_revision: 6});
  assert.equal(calls.filter(call => call.url.endsWith("/confirm") || call.url.endsWith("/speech")).length, 0);
});

test("refresh after managing memories preserves a current manual edit without replaying it", async () => {
  const {api, calls} = fixture();
  await api.transcribeAudio("recording.m4a", "recording.m4a", "home", "", null, "");
  await api.chooseMessage("m1");
  await api.editMessage("Please bring tea without milk.");
  const refreshed = await api.refreshActiveMessage();
  assert.equal(refreshed.auto_speak, false);
  assert.equal(refreshed.recommended_message_id, "edited-5");
  assert.equal(refreshed.messages[0].corrected_text, "Please bring tea without milk.");
  assert.equal(refreshed.decision_source, "user");
  assert.equal(calls.filter(call => call.url.endsWith("/confirm") || call.url.endsWith("/speech")).length, 0);
});

test("native manual editing keeps authored words silent and the fifth literal remains an exact speaking choice", async () => {
  const hypotheses = ['tea', 'toast', 'soup', 'water', 'coffee'].map((literal_text, index) => ({id: `h${index + 1}`, literal_text}));
  const {api, calls} = fixture(undefined, {context: {selection: {profile_id: 'person'}},
    evidence: {backend: 'local', model: 'adapted', hypotheses}});
  await api.transcribeAudio('recording.m4a', 'recording.m4a', 'home', '', null, '');
  await api.editMessage('Please call Maya.');
  const edited = await api.refreshActiveMessage();
  assert.equal(edited.auto_speak, false);
  assert.equal(edited.needs_user_choice, false);
  assert.equal(edited.recommended_message_id, 'edited-4');
  assert.equal(edited.messages[0].corrected_text, 'Please call Maya.');
  assert.deepEqual(Array.from(edited.messages[0].source_hypothesis_ids), []);
  assert.deepEqual(Array.from(edited.messages.slice(1), item => item.message_id), ['raw-h1', 'raw-h2', 'raw-h3', 'raw-h4', 'raw-h5']);
  await api.rememberActiveMessage('Please call Maya.');
  assert.equal(calls.filter(call => call.url.endsWith('/confirm') || call.url.endsWith('/speech') || call.url.endsWith('/draft')).length, 0);
  const chosen = await api.chooseMessage('raw-h5');
  assert.equal(chosen.auto_speak, true);
  assert.equal(chosen.messages.find(item => item.message_id === chosen.recommended_message_id).corrected_text, 'coffee');
  assert.equal(calls.filter(call => call.url.endsWith('/audio')).length, 1);
});
