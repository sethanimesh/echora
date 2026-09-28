type PlaybackJob = {
  id: string;
  revision: number;
  status: string;
  confirmed?: { id: string } | null;
  ranking?: { status: string } | null;
};

/** A server invalidation also revokes audio already queued in this browser. */
export function serverRevokedPlayback(previous: PlaybackJob | null, next: PlaybackJob | null, eventType = '') {
  if (!previous) return false;
  if (next && next.id === previous.id && next.revision < previous.revision) return false;
  return !next || next.id !== previous.id || next.status === 'cancelled' ||
    eventType === 'personal_context_invalidated' ||
    (next.revision > previous.revision && next.ranking?.status === 'stale_context') ||
    Boolean(previous.confirmed && !next.confirmed);
}

/** A delayed confirmation response cannot replace newer visible server state. */
export function confirmationStillCurrent(current: PlaybackJob | null, confirmed: PlaybackJob) {
  return Boolean(current && current.id === confirmed.id && current.revision <= confirmed.revision &&
    current.status !== 'cancelled');
}
