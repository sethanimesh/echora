'use client';
import { useEffect, useState } from 'react';
import type { Profile } from './contextTypes';
import { clearMemories, deleteMemory, deleteProfile, readMemories, type MemoryList } from './memories';

export default function MemoriesPanel({ profile, refresh, disabled, onChanged, onDeleted }: {
  profile: Profile;
  refresh: number;
  disabled: boolean;
  onChanged: () => Promise<void>;
  onDeleted: () => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<MemoryList | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [confirm, setConfirm] = useState<'clear' | 'profile' | null>(null);
  useEffect(() => {
    if (!open) return;
    let current = true;
    setData(null);
    setError('');
    readMemories(profile.id).then((value) => { if (current) setData(value); })
      .catch((reason) => { if (current) setError(reason instanceof Error ? reason.message : 'Saved messages could not be loaded.'); });
    return () => { current = false; };
  }, [profile.id, refresh, open]);

  async function remove(kind: 'entry' | 'clear' | 'profile', id?: string) {
    if (!data || busy || disabled) return;
    setBusy(true);
    setError('');
    try {
      if (kind === 'profile') {
        await deleteProfile(profile.id, profile.revision, data.memory_revision);
        await onDeleted();
        return;
      }
      const changed = kind === 'clear'
        ? await clearMemories(profile.id, data.memory_revision)
        : await deleteMemory(profile.id, id!, data.memory_revision);
      setData({ ...data, memory_revision: changed.memory_revision,
        memories: data.memories.filter((item) => !changed.deleted_ids.includes(item.id)) });
      setConfirm(null);
      await onChanged();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'This change could not be saved.');
      // Refresh the displayed list after a conflict, but never repeat a deletion.
      try { setData(await readMemories(profile.id)); } catch { /* Keep the error visible. */ }
    } finally { setBusy(false); }
  }

  return <section className="context-panel" aria-label="Remembered messages">
    <button className="context-link" aria-expanded={open} disabled={disabled || busy}
      onClick={() => { setOpen(!open); setConfirm(null); }}>Remembered messages · {profile.label}</button>
    {open && <div className="context-extra">
      <p>Only messages you choose to Remember are saved here. They can help with future wording in this profile.</p>
      {error && <p role="alert">{error}</p>}
      {!data && !error && <p role="status">Loading saved messages…</p>}
      {data && <>
        {!data.memories.length && <p>No remembered messages.</p>}
        <ul className="memory-list">
          {data.memories.map((memory) => <li key={memory.id}>
            <p>{memory.message}</p>
            <small>{[memory.scope.scenario, memory.scope.recipient,
              new Date(memory.created_at * 1000).toLocaleDateString()].filter(Boolean).join(' · ')}</small>{' '}
            <button className="context-link" disabled={disabled || busy}
              onClick={() => void remove('entry', memory.id)} aria-label={`Forget saved message: ${memory.message}`}>Forget</button>
          </li>)}
        </ul>
        <div className="clarify-actions">
          <button className="context-link" disabled={disabled || busy || !data.memories.length} onClick={() => setConfirm('clear')}>Forget all saved messages</button>
          {!profile.sample && <button className="context-link" disabled={disabled || busy} onClick={() => setConfirm('profile')}>Delete this profile</button>}
        </div>
        {confirm && <div role="group" aria-label="Confirm removal">
          <p>{confirm === 'profile' ? `Delete ${profile.label} and its saved messages?` : 'Forget every saved message in this profile?'}</p>
          <button className="context-save" disabled={disabled || busy} onClick={() => void remove(confirm)}>{busy ? 'Removing…' : confirm === 'profile' ? 'Delete profile and messages' : 'Forget all'}</button>{' '}
          <button className="context-link" disabled={busy} onClick={() => setConfirm(null)}>Keep them</button>
        </div>}
      </>}
    </div>}
  </section>;
}
