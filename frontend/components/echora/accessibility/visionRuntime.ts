import { FilesetResolver } from '@mediapipe/tasks-vision';

// MediaPipe 1.0.1's automatic script loader uses document when importScripts is
// absent (including some module-worker environments). Load the ES module
// explicitly, then let MediaPipe instantiate its factory without a script tag.
export async function loadVisionRuntime(base: string) {
  const files = await FilesetResolver.forVisionTasks(base + '/wasm', true);
  const loader = await import(/* @vite-ignore */ files.wasmLoaderPath);
  const runtime = globalThis as typeof globalThis & { ModuleFactory?: unknown };
  if (typeof loader.default !== 'function') {
    throw new Error(
      'The local vision runtime did not export its model loader.',
    );
  }
  runtime.ModuleFactory = loader.default;
  return { ...files, wasmLoaderPath: '' };
}
