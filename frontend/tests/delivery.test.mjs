import test from 'node:test';
import assert from 'node:assert/strict';
import { deliveryMatches } from '../components/echora/delivery.ts';

test('replay requires the same user-approved tone and pace', () => {
  const confirmed = {tone: 'warm', rate: 0.9, source: 'user'};
  assert.equal(deliveryMatches(confirmed, 'warm', 0.9), true);
  assert.equal(deliveryMatches(confirmed, 'firm', 0.9), false);
  assert.equal(deliveryMatches(confirmed, 'warm', 1.2), false);
  assert.equal(deliveryMatches(undefined, 'neutral', 0.9), false);
  assert.equal(deliveryMatches({...confirmed, source: 'camera'}, 'warm', 0.9), false);
});
