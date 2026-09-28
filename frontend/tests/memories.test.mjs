import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import test from 'node:test';
import ts from 'typescript';

const compile = async (file) => ts.transpileModule(await readFile(new URL(file, import.meta.url), 'utf8'), {
  compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022},
}).outputText;
const [apiCode, memoryCode] = await Promise.all([
  compile('../components/echora/types.ts'), compile('../components/echora/memories.ts'),
]);

function fixture(conflict = false) {
  const calls = [];
  const exports = {};
  vm.runInNewContext(apiCode, {exports, FormData, fetch: async (url, options) => {
    calls.push({url, ...options, body: options.body && JSON.parse(options.body)});
    if (url.endsWith('/memories')) return new Response(JSON.stringify({profile_id: 'p', memory_revision: 8, memories: []}));
    if (conflict) return new Response(JSON.stringify({detail: 'Saved messages changed. Review the current list.'}), {status: 409});
    return new Response(JSON.stringify({memory_id: 'm', memory_revision: 9, message_revision: 4, created: true}));
  }});
  const memory = {};
  vm.runInNewContext(memoryCode, {exports: memory, require: (name) => {
    assert.equal(name, './types'); return exports;
  }});
  return {memory, calls};
}

test('Remember sends the reviewed message revision and fresh memory revision without speech authorization', async () => {
  const {memory, calls} = fixture();
  assert.equal(calls.length, 0);
  const saved = await memory.rememberMessage({id: 'job-one', revision: 4}, 'profile/one');
  assert.equal(saved.memory_id, 'm');
  assert.deepEqual(calls.map((call) => call.url), ['/api/profiles/profile%2Fone/memories', '/api/messages/job-one/remember']);
  assert.deepEqual(calls[1].body, {revision: 4, memory_revision: 8});
  assert.equal(calls[1].headers['X-Echora-Client'], '1');
});

test('a concurrent memory change is shown without automatically retrying Remember', async () => {
  const {memory, calls} = fixture(true);
  await assert.rejects(memory.rememberMessage({id: 'job-one', revision: 4}, 'p'), /Saved messages changed/);
  assert.equal(calls.filter((call) => call.url.endsWith('/remember')).length, 1);
});

test('deletions name the chosen profile and carry both required revisions', async () => {
  const {memory, calls} = fixture();
  await memory.deleteMemory('profile/one', 'memory/one', 8);
  await memory.clearMemories('profile/one', 9);
  await memory.deleteProfile('profile/one', 3, 10);
  assert.equal(calls[0].url, '/api/profiles/profile%2Fone/memories/memory%2Fone/delete');
  assert.deepEqual(calls[0].body, {memory_revision: 8});
  assert.deepEqual(calls[1].body, {memory_revision: 9});
  assert.deepEqual(calls[2].body, {profile_revision: 3, memory_revision: 10});
});
