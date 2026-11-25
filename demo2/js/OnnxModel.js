// ONNX Runtime Web loader & priority-serialized session manager for browser-only SAM inference.
// Modeled on LICHEN/js/modules/img2img/OnnxModel.js:
// - Loads ort.all.min.js once and sets an absolute wasmPaths URL (needed for blob: proxy workers).
// - Prefers WebGPU when available, falling back to WASM when WebGPU is unavailable or fails.
// - Serializes all session.run() calls across models (WebGPU allows one active run at a time),
//   with priority scheduling so lightweight mask decoder runs jump ahead of encoder frames.

const ORT_DIR = new URL('../node_modules/onnxruntime-web/dist/', import.meta.url).href;

let ortReady = null;
let defaultBackend = null; // 'webgpu' or 'wasm' once probed
let loading = Promise.resolve();
const models = new Map(); // `${url}::${ep}` -> Promise<OnnxModel>
const byteCache = new Map(); // url -> Promise<Uint8Array>

// Priority run queue (higher priority runs first; serialized so WebGPU never overlaps runs)
const runQueue = [];
let runActive = false;

function drainRunQueue() {
  if (runActive || runQueue.length === 0) return;
  runActive = true;
  // Pick highest-priority pending job (stable FIFO within same priority)
  let bestIdx = 0;
  for (let i = 1; i < runQueue.length; i++) {
    if (runQueue[i].priority > runQueue[bestIdx].priority) bestIdx = i;
  }
  const job = runQueue.splice(bestIdx, 1)[0];
  job.fn().then(
    (res) => {
      runActive = false;
      job.resolve(res);
      drainRunQueue();
    },
    (err) => {
      runActive = false;
      job.reject(err);
      drainRunQueue();
    }
  );
}

function enqueueRun(fn, priority = 0) {
  return new Promise((resolve, reject) => {
    runQueue.push({ fn, priority, resolve, reject });
    drainRunQueue();
  });
}

function loadOrt() {
  ortReady ??= new Promise((resolve, reject) => {
    if (window.ort) {
      window.ort.env.wasm.wasmPaths = ORT_DIR;
      window.ort.env.logLevel = 'error';
      resolve(window.ort);
      return;
    }
    const script = document.createElement('script');
    script.src = ORT_DIR + 'ort.all.min.js';
    script.onload = () => {
      window.ort.env.wasm.wasmPaths = ORT_DIR;
      window.ort.env.logLevel = 'error';
      resolve(window.ort);
    };
    script.onerror = () => reject(new Error(`Could not load ${script.src}`));
    document.head.appendChild(script);
  });
  return ortReady;
}

async function hasWebGPU() {
  if (!navigator.gpu) return false;
  try {
    return !!(await navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}

function fetchModelBytes(url, onProgress) {
  if (!byteCache.has(url)) {
    const p = (async () => {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText} for ${url}`);
      }
      const total = Number(response.headers.get('content-length')) || 0;
      if (!response.body || !onProgress) {
        const buf = new Uint8Array(await response.arrayBuffer());
        if (onProgress) onProgress(buf.byteLength, buf.byteLength);
        return buf;
      }
      const reader = response.body.getReader();
      const chunks = [];
      let loaded = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        onProgress(loaded, total);
      }
      const out = new Uint8Array(loaded);
      let offset = 0;
      for (const chunk of chunks) {
        out.set(chunk, offset);
        offset += chunk.byteLength;
      }
      return out;
    })();
    byteCache.set(url, p);
    p.catch(() => byteCache.delete(url));
  }
  return byteCache.get(url);
}

async function start(url, { options = {}, onProgress = null, forceBackend = null } = {}) {
  const ort = await loadOrt();
  const bytes = await fetchModelBytes(url, onProgress);

  const tryWebGPU =
    forceBackend === 'webgpu' ||
    (forceBackend === null && (defaultBackend === 'webgpu' || (defaultBackend === null && (await hasWebGPU()))));

  if (tryWebGPU) {
    try {
      const session = await ort.InferenceSession.create(bytes, {
        ...options,
        executionProviders: ['webgpu'],
      });
      if (defaultBackend === null) defaultBackend = 'webgpu';
      return new OnnxModel(ort, session, 'webgpu', bytes.byteLength);
    } catch (e) {
      if (forceBackend === 'webgpu') throw e;
      console.warn(`ONNX Runtime: WebGPU failed for ${url}, falling back to WASM:`, e);
    }
  }

  // Only enable proxy worker if no main-thread WebGPU session has started
  if (defaultBackend === null) {
    ort.env.wasm.proxy = true;
    defaultBackend = 'wasm';
  }
  const session = await ort.InferenceSession.create(bytes, {
    ...options,
    executionProviders: ['wasm'],
  });
  return new OnnxModel(ort, session, 'wasm', bytes.byteLength);
}

export function loadOnnxModel(url, opts = {}) {
  const key = `${url}::${opts.forceBackend || 'auto'}`;
  if (!models.has(key)) {
    const model = loading.then(() => start(url, opts));
    loading = model.catch(() => {});
    models.set(key, model);
    model.catch(() => models.delete(key));
  }
  return models.get(key);
}

export class Backoff {
  constructor(first = 1000, most = 30000) {
    this.first = first;
    this.most = most;
    this.reset();
  }

  reset() {
    this.wait = 0;
    this.until = 0;
  }

  fail() {
    this.wait = this.wait ? Math.min(this.wait * 2, this.most) : this.first;
    this.until = performance.now() + this.wait;
    return this.wait;
  }

  get ready() {
    return performance.now() >= this.until;
  }
}

class OnnxModel {
  constructor(ort, session, ep, byteLength) {
    this.ort = ort;
    this.session = session;
    this.backend = ep;
    this.byteLength = byteLength;
  }

  run(feeds, { priority = 0 } = {}) {
    return enqueueRun(() => this.session.run(feeds), priority);
  }
}
