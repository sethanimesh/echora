import test from 'node:test';
import assert from 'node:assert/strict';
import { startupRead } from '../components/echora/startupRead.ts';

test('startup reads recover from an unavailable service and incomplete proxy responses', async t => {
  const requests = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    requests.push({ url, ...options });
    if (requests.length === 1) throw new TypeError('Failed to fetch');
    if (requests.length === 2) return new Response('starting', { status: 503 });
    if (requests.length === 3) return new Response('<html>proxy starting</html>');
    return Response.json({ job: null, configured: true });
  });
  const session = await startupRead('/session', new AbortController().signal, {
    delays: [0, 0, 0],
  });
  assert.deepEqual(session, { job: null, configured: true });
  assert.equal(requests.length, 4);
  assert.ok(requests.every(request => request.url === '/api/session' && request.method === 'GET'));
  assert.ok(requests.every(request => request.cache === 'no-store' && request.signal));
});

test('initial saved places recover after service startup using only GET requests', async t => {
  const requests = [];
  const settings = {
    auto_detect: false,
    places: [{ id: 'home', label: 'Home', context: 'home', listener: null }],
  };
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    requests.push({ url, ...options });
    if (requests.length === 1) return new Response('starting', { status: 503 });
    return Response.json(settings);
  });
  assert.deepEqual(
    await startupRead('/v1/places', new AbortController().signal, { delays: [0] }),
    settings,
  );
  assert.equal(requests.length, 2);
  assert.ok(requests.every(request => request.url === '/api/v1/places' && request.method === 'GET' && !request.body));
});

test('persistent startup failures exhaust a bounded retry budget without exposing response details', async t => {
  let requests = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    requests++;
    return new Response('Internal connection error: private path', { status: 502 });
  });
  await assert.rejects(
    startupRead('/profiles', new AbortController().signal, { delays: [0, 0] }),
    { message: 'Echora could not connect. Please refresh in a moment.' },
  );
  assert.equal(requests, 3);
});

test('a permanent request rejection does not retry', async t => {
  let requests = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    requests++;
    return new Response('not allowed', { status: 403 });
  });
  await assert.rejects(startupRead('/session', new AbortController().signal));
  assert.equal(requests, 1);
});

test('unmount cancels an active bootstrap request and never retries it', async t => {
  let requests = 0;
  let requestSignal;
  t.mock.method(globalThis, 'fetch', (_url, options) => {
    requests++;
    requestSignal = options.signal;
    return new Promise((_resolve, reject) => {
      options.signal.addEventListener('abort', () => reject(options.signal.reason), { once: true });
    });
  });
  const controller = new AbortController();
  const loading = startupRead('/session', controller.signal);
  controller.abort();
  await assert.rejects(loading, { name: 'AbortError' });
  assert.equal(requestSignal.aborted, true);
  assert.equal(requests, 1);
});

test('unmount cancels a pending retry delay', async t => {
  let requests = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    requests++;
    throw new TypeError('Failed to fetch');
  });
  const controller = new AbortController();
  const loading = startupRead('/profiles', controller.signal, { delays: [10000] });
  await new Promise(resolve => setImmediate(resolve));
  controller.abort();
  await assert.rejects(loading, { name: 'AbortError' });
  assert.equal(requests, 1);
});

test('an unresponsive startup request times out and uses only the allowed attempts', async t => {
  let requests = 0;
  t.mock.method(globalThis, 'fetch', (_url, options) => {
    requests++;
    return new Promise((_resolve, reject) => {
      options.signal.addEventListener('abort', () => reject(options.signal.reason), { once: true });
    });
  });
  await assert.rejects(
    startupRead('/session', new AbortController().signal, {
      delays: [0], requestTimeoutMs: 5,
    }),
    { message: 'Echora could not connect. Please refresh in a moment.' },
  );
  assert.equal(requests, 2);
});
