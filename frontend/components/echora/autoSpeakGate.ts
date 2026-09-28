/** A live user operation authorizes one generated result. Restored state never does. */
export class AutoSpeakGate {
  private serial = 0;
  private operation: {
    token: number;
    jobId: string | null;
    afterRevision: number;
  } | null = null;
  private live = new Set<string>();
  private spoken = new Set<string>();
  arm() {
    const token = ++this.serial;
    this.operation = { token, jobId: null, afterRevision: -1 };
    this.live.clear();
    return token;
  }
  bind(token: number, jobId: string, afterRevision = -1) {
    if (this.operation?.token !== token) return;
    this.operation.jobId = jobId;
    this.operation.afterRevision = afterRevision;
  }
  observe(job: { id: string; revision: number } | null, live: boolean) {
    if (job && live && this.operation)
      this.live.add(`${job.id}:${job.revision}`);
  }
  cancel() {
    this.serial++;
    this.operation = null;
    this.live.clear();
  }
  consume(
    job: {
      id: string;
      revision: number;
      auto_speak_revision?: number | null;
      ranking?: { status: string; route?: string; decision: 'selected' | 'ambiguous' } | null;
      decision_source?: string;
      status: string;
      question?: string;
      error?: unknown;
      text: string;
    } | null,
  ) {
    const request = this.operation;
    if (
      !job ||
      !request ||
      request.jobId !== job.id ||
      job.revision <= request.afterRevision
    )
      return false;
    const key = `${job.id}:${job.revision}`;
    if (
      !this.live.has(key) ||
      this.spoken.has(key) ||
      job.auto_speak_revision !== job.revision ||
      (job.ranking?.decision === 'ambiguous' &&
        job.ranking.route !== 'legacy' &&
        job.decision_source !== 'user') ||
      job.question ||
      job.error ||
      !job.text.trim() ||
      !['review', 'confirmed'].includes(job.status)
    )
      return false;
    this.spoken.add(key);
    this.operation = null;
    this.live.clear();
    return true;
  }
}
