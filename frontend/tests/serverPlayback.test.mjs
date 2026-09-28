import assert from 'node:assert/strict';
import test from 'node:test';
import { confirmationStillCurrent, serverRevokedPlayback } from '../components/echora/serverPlayback.ts';

const reviewed = { id: 'message', revision: 4, status: 'review' };
test('a cross-session deletion revokes queued speech before its confirmation arrives', async () => {
  let generation = 0;
  let current = reviewed;
  let finish;
  let played = false;
  const confirmation = new Promise(resolve => { finish = resolve; });
  const pendingGeneration = generation;
  const play = (async () => {
    const authorized = await confirmation;
    if (pendingGeneration === generation && confirmationStillCurrent(current, authorized)) played = true;
  })();
  const invalidated = {...reviewed, revision: 5, ranking: {status: 'stale_context'}};
  if (serverRevokedPlayback(current, invalidated, 'personal_context_invalidated')) generation++;
  current = invalidated;
  finish({...reviewed, status: 'confirmed', confirmed: {id: 'old-authorization'}});
  await play;
  assert.equal(played, false);
});

test('server updates revoke active audio without cancelling normal confirmation or synthesis', () => {
  const confirmed = {...reviewed, status: 'confirmed', confirmed: {id: 'voice'}};
  assert.equal(serverRevokedPlayback(reviewed, confirmed, 'message_confirmed'), false);
  assert.equal(serverRevokedPlayback(confirmed, {...confirmed, status: 'synthesizing'}), false);
  assert.equal(serverRevokedPlayback(confirmed, {...confirmed, confirmed: {id: 'new-delivery'}}, 'message_confirmed'), false);
  assert.equal(serverRevokedPlayback(confirmed, {...reviewed, revision: 5, confirmed: null}), true);
  assert.equal(serverRevokedPlayback(confirmed, {...reviewed, revision: 3}), false);
  assert.equal(serverRevokedPlayback(confirmed, null), true);
  assert.equal(confirmationStillCurrent(reviewed, confirmed), true);
  assert.equal(confirmationStillCurrent({...reviewed, revision: 5}, confirmed), false);
  assert.equal(confirmationStillCurrent({...reviewed, status: 'cancelled'}, confirmed), false);
});

test('pronunciation preparation and explicit recovery can still authorize the latest words', () => {
  assert.equal(serverRevokedPlayback(reviewed, {...reviewed, revision: 5, status: 'preparing'}), false);
  assert.equal(confirmationStillCurrent({...reviewed, revision: 6, ranking: {status: 'stale_context'}},
    {...reviewed, revision: 6, status: 'confirmed', confirmed: {id: 'revised-words'}}), true);
});
