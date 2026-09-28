import { api } from './types';

export type SavedMemory = {
  id: string;
  message: string;
  language: string;
  created_at: number;
  scope: { scenario?: string; core_context?: string; recipient?: string; audience_id?: string; listener?: string };
  profile_revision: number;
};
export type MemoryList = { profile_id: string; memory_revision: number; memories: SavedMemory[] };
export type RememberResult = { memory_id: string; memory_revision: number; message_revision: number; created: boolean; memory: SavedMemory };
type MemoryChange = { profile_id: string; memory_revision: number; deleted_ids: string[] };

export const readMemories = (profileId: string) =>
  api<MemoryList>(`/profiles/${encodeURIComponent(profileId)}/memories`);

export async function rememberMessage(job: { id: string; revision: number }, profileId: string) {
  const memories = await readMemories(profileId);
  return api<RememberResult>(`/messages/${encodeURIComponent(job.id)}/remember`, {
    revision: job.revision,
    memory_revision: memories.memory_revision,
  });
}

export const deleteMemory = (profileId: string, memoryId: string, revision: number) =>
  api<MemoryChange>(`/profiles/${encodeURIComponent(profileId)}/memories/${encodeURIComponent(memoryId)}/delete`, { memory_revision: revision });

export const clearMemories = (profileId: string, revision: number) =>
  api<MemoryChange>(`/profiles/${encodeURIComponent(profileId)}/memories/clear`, { memory_revision: revision });

export const deleteProfile = (profileId: string, profileRevision: number, memoryRevision: number) =>
  api<{ deleted: boolean; profile_id: string }>(`/profiles/${encodeURIComponent(profileId)}/delete`, {
    profile_revision: profileRevision,
    memory_revision: memoryRevision,
  });
