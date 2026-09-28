const RETRY_DELAYS = [500, 1000, 2000, 4000, 8000, 8000, 8000, 8000];
const UNAVAILABLE = 'Echora could not connect. Please refresh in a moment.';

class ReadFailure extends Error {
  retryable: boolean;
  constructor(retryable: boolean) {
    super(UNAVAILABLE);
    this.retryable = retryable;
  }
}

function waitForRetry(delay: number, signal: AbortSignal) {
  signal.throwIfAborted();
  return new Promise<void>((resolve, reject) => {
    const cancelled = () => {
      clearTimeout(timer);
      reject(signal.reason);
    };
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', cancelled);
      resolve();
    }, delay);
    signal.addEventListener('abort', cancelled, { once: true });
  });
}

/** Startup reads may precede the model service. Never use this for mutations. */
export async function startupRead<T>(
  path: '/session' | '/profiles' | '/v1/places',
  signal: AbortSignal,
  policy: { delays?: readonly number[]; requestTimeoutMs?: number } = {},
): Promise<T> {
  const delays = policy.delays ?? RETRY_DELAYS;
  for (let attempt = 0; ; attempt++) {
    signal.throwIfAborted();
    const request = new AbortController();
    const cancelled = () => request.abort(signal.reason);
    signal.addEventListener('abort', cancelled, { once: true });
    const timer = setTimeout(
      () => request.abort(),
      policy.requestTimeoutMs ?? 5000,
    );
    let failure: unknown;
    try {
      const response = await fetch('/api' + path, {
        method: 'GET',
        signal: request.signal,
        cache: 'no-store',
      });
      if (!response.ok)
        throw new ReadFailure(
          response.status >= 500 || [408, 425, 429].includes(response.status),
        );
      const data = (await response.json()) as T;
      signal.throwIfAborted();
      return data;
    } catch (error) {
      signal.throwIfAborted();
      failure = error;
    } finally {
      clearTimeout(timer);
      signal.removeEventListener('abort', cancelled);
    }
    if (
      attempt >= delays.length ||
      (failure instanceof ReadFailure && !failure.retryable)
    )
      throw new Error(UNAVAILABLE);
    await waitForRetry(delays[attempt], signal);
  }
}
