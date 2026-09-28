import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import ts from 'typescript';
import { renderToStaticMarkup } from 'react-dom/server';
import { tones, speechPaces, deliveryFromSuggestion } from '../components/echora/delivery.ts';

const require = createRequire(import.meta.url);
const output = ts.transpileModule(readFileSync(new URL('../components/echora/VoiceSuggestion.tsx', import.meta.url), 'utf8'), {
  compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS },
}).outputText;
const exports = {};
new Function('require', 'exports', output)((id) => id === './delivery' ? { tones, speechPaces } : require(id), exports);
const Component = exports.default;
function buttons(node) {
  if (!node || typeof node !== 'object') return [];
  if (Array.isArray(node)) return node.flatMap(buttons);
  return [...(node.type === 'button' ? [node] : []), ...buttons(node.props?.children)];
}
const base = { configured: true, available: true, disabled: false, dismissed: false,
  onSuggest() {}, onStop() {}, onAccept() {}, onDismiss() {} };

test('a returned cue cannot automatically accept a tone or start playback', () => {
  const selected = [];
  let requests = 0;
  const suggestion = { id: 'one', revision: 2, state: 'ready', tone: 'warm', cue: 'soft', rate: 1.2 };
  const tree = Component({ ...base, suggestion,
    onSuggest: () => requests++, onAccept: () => selected.push(deliveryFromSuggestion(suggestion, 'neutral', .9)) });
  assert.deepEqual(selected, []);
  assert.equal(requests, 0);
  buttons(tree).find(button => button.props.children === 'Use suggested delivery').props.onClick();
  assert.deepEqual(selected, [{ tone: 'warm', rate: 1.2 }]);
  assert.match(renderToStaticMarkup(tree), /Faster.*1.2/);
  assert.equal(requests, 0);
  assert.match(renderToStaticMarkup(tree), /sends the recording to/);
});

test('unclear and dismissed suggestions offer no acceptance control', () => {
  for (const props of [
    { suggestion: { state: 'ready', tone: null, cue: 'unclear' } },
    { suggestion: { state: 'ready', tone: 'warm', cue: 'soft' }, dismissed: true },
  ]) {
    assert.doesNotMatch(renderToStaticMarkup(Component({ ...base, ...props })), />Use /);
  }
});

test('missing recording disables analysis; running analysis stays stoppable', () => {
  assert.equal(buttons(Component({ ...base, available: false }))[0].props.disabled, true);
  assert.equal(buttons(Component({ ...base, configured: false }))[0].props.disabled, true);
  let stops = 0;
  const controls = buttons(Component({ ...base, disabled: true, suggestion: { state: 'analyzing' }, onStop: () => stops++ }));
  assert.equal(controls[0].props.disabled, undefined);
  controls[0].props.onClick();
  assert.equal(stops, 1);
});


test('tone and pace apply together while independent unclear fields preserve the current choice', () => {
  assert.deepEqual(deliveryFromSuggestion({ state: 'ready', tone: 'warm', rate: .65 }, 'firm', 1.2), { tone: 'warm', rate: .65 });
  assert.deepEqual(deliveryFromSuggestion({ state: 'ready', tone: null, rate: 1.2 }, 'firm', .9), { tone: 'firm', rate: 1.2 });
  assert.deepEqual(deliveryFromSuggestion({ state: 'ready', tone: 'warm', rate: null }, 'firm', 1.2), { tone: 'warm', rate: 1.2 });
  for (const suggestion of [
    { state: 'ready', tone: null, rate: null },
    { state: 'analyzing', tone: 'warm', rate: 1 },
    { state: 'stopped', tone: 'warm', rate: 1 },
    { state: 'ready', tone: 'angry', rate: 1 },
    { state: 'ready', tone: 'warm', rate: 9 },
  ]) assert.equal(deliveryFromSuggestion(suggestion, 'neutral', .9), null);
  const tree = Component({ ...base, suggestion: { state: 'ready', tone: null, rate: .65, cue: 'unclear' } });
  assert.ok(buttons(tree).some(button => button.props.children === 'Use suggested delivery'));
});
