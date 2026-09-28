import test from 'node:test';
import assert from 'node:assert/strict';
import { AutoSpeakGate } from '../components/echora/autoSpeakGate.ts';
const result = (extra = {}) => ({
  id: 'new',
  revision: 4,
  auto_speak_revision: 4,
  status: 'review',
  text: 'Please bring water.',
  question: '',
  error: null,
  ...extra,
});
test('restored or reconnect snapshots cannot authorize speech', () => {
  const g = new AutoSpeakGate();
  const j = result();
  g.observe(j, false);
  assert.equal(g.consume(j), false);
  const t = g.arm();
  g.bind(t, j.id);
  g.observe(j, false);
  assert.equal(g.consume(j), false);
  g.observe(j, true);
  g.cancel();
  assert.equal(g.consume(j), false);
});
test('a live result is spoken exactly once, including duplicate events', () => {
  const g = new AutoSpeakGate();
  const t = g.arm();
  g.bind(t, 'new');
  const j = result();
  g.observe(j, true);
  assert.equal(g.consume(j), true);
  g.observe(j, true);
  assert.equal(g.consume(j), false);
});
test('stop before HTTP binding rejects late response and events', () => {
  const g = new AutoSpeakGate();
  const t = g.arm();
  g.cancel();
  g.bind(t, 'new');
  g.observe(result(), true);
  assert.equal(g.consume(result()), false);
});
test('old job, old revision and raw fallback stay silent', () => {
  const g = new AutoSpeakGate();
  const t = g.arm();
  g.bind(t, 'new', 3);
  for (const j of [
    result({ id: 'old' }),
    result({ revision: 3, auto_speak_revision: 3 }),
    result({ auto_speak_revision: null }),
  ]) {
    g.observe(j, true);
    assert.equal(g.consume(j), false);
  }
});
test('clarification, errors and ongoing analysis cannot be spoken', () => {
  const g = new AutoSpeakGate();
  const t = g.arm();
  g.bind(t, 'new');
  for (const j of [
    result({ question: 'Tea or water?' }),
    result({ error: { message: 'Unavailable' } }),
    result({ status: 'analyzing_delivery' }),
  ]) {
    g.observe(j, true);
    assert.equal(g.consume(j), false);
  }
  const j = result();
  g.observe(j, true);
  assert.equal(g.consume(j), true);
});
test('events arriving before HTTP binding are retained only for that live operation', () => {
  const g = new AutoSpeakGate();
  const t = g.arm();
  const j = result();
  g.observe(j, true);
  assert.equal(g.consume(j), false);
  g.bind(t, 'new');
  assert.equal(g.consume(j), true);
});
test('choosing raw words speaks once after the service explicitly authorizes that revision', () => {
  const g = new AutoSpeakGate();
  const token = g.arm();
  g.bind(token, 'new', 4);
  const chosen = result({
    revision: 5,
    auto_speak_revision: 5,
    text: 'water',
    candidates: [{ text: 'water', available: false }],
    warning: { code: 'raw_fallback', message: 'Literal words only.' },
  });
  g.observe(chosen, true);
  assert.equal(g.consume(chosen), true);
  assert.equal(g.consume(chosen), false);
});

test('unresolved verification survives a single completion and an accidental speech marker', () => {
  const gate = new AutoSpeakGate();
  const token = gate.arm();
  gate.bind(token, 'new');
  const ambiguous = result({ranking: {status: 'ready', decision: 'ambiguous'}, candidates: [{id: 'm1'}]});
  gate.observe(ambiguous, true);
  assert.equal(gate.consume(ambiguous), false);
  const explicit = {...ambiguous, revision: 5, auto_speak_revision: 5, decision_source: 'user'};
  gate.observe(explicit, true);
  assert.equal(gate.consume(explicit), true);
});

test('five alternatives do not block a resolved authorized recommendation; ranking alone cannot play it', () => {
  const gate = new AutoSpeakGate();
  const token = gate.arm();
  gate.bind(token, 'new');
  const job = result({ranking: {status: 'ready', decision: 'selected'},
    candidates: Array.from({length: 5}, (_, i) => ({id: `candidate-${i}`})),
    selected_candidate_id: 'candidate-3', auto_speak_revision: null});
  gate.observe(job, true);
  assert.equal(gate.consume(job), false);
  const authorized = {...job, auto_speak_revision: 4};
  gate.observe(authorized, true);
  assert.equal(gate.consume(authorized), true);
});

test('an unsupported verification route preserves explicit legacy speech authorization', () => {
  const gate = new AutoSpeakGate();
  const token = gate.arm();
  gate.bind(token, 'new');
  const job = result({ranking: {status: 'unsupported', route: 'legacy', decision: 'ambiguous'}});
  gate.observe(job, true);
  assert.equal(gate.consume(job), true);
});

test('missing verification cannot speak one suggestion even if its question is lost', () => {
  const gate = new AutoSpeakGate();
  const token = gate.arm();
  gate.bind(token, 'new');
  const job = result({ranking: {status: 'unavailable', route: 'verified', decision: 'ambiguous'},
    candidates: [{id: 'm1'}], selected_candidate_id: 'm1', question: '', error: null});
  gate.observe(job, true);
  assert.equal(gate.consume(job), false);
});
