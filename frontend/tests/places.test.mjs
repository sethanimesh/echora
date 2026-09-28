import test from 'node:test';
import assert from 'node:assert/strict';
import { matchPlace, placeSelection } from '../components/echora/places.ts';
const place = {
  id: 'home',
  label: 'Home',
  context: 'home',
  listener: null,
  builtin: true,
  latitude: 12,
  longitude: 77,
  radius_m: 100,
  tagged_at: null,
};
test('place preserves profile and explicit audience while carrying nullable declaration', () => {
  const result = placeSelection(
    { profile_id: 'p', audience_id: 'daughter' },
    place,
  );
  assert.equal(result.audience_id, 'daughter');
  assert.equal(result.declared_listener, null);
  assert.equal(result.profile_id, 'p');
  assert.equal(result.core_context, 'home');
});
test('ambiguous, inaccurate and unmatched locations abstain', () => {
  assert.equal(matchPlace([place], 12, 77, 15)?.id, 'home');
  assert.equal(
    matchPlace([place, { ...place, id: 'other' }], 12, 77, 15),
    null,
  );
  assert.equal(matchPlace([place], 12, 77, 300), null);
  assert.equal(matchPlace([place], 13, 78, 15), null);
});
