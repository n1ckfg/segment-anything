import { loadOnnxModel, Backoff } from './OnnxModel.js';

const SAM_DECODER_SIZE = 1024;
const PIXEL_MEAN = [123.675, 116.28, 103.53];
const PIXEL_STD = [58.395, 57.12, 57.375];

const ENCODER_URL = new URL('../mobile_sam_encoder_fp16.onnx?v=5', import.meta.url).href;
const DECODER_FILENAME = 'mobile_sam_decoder_fp16.onnx';
const DECODER_URL = new URL('../mobile_sam_decoder_fp16.onnx?v=5', import.meta.url).href;

// DOM elements
const statusDot = document.getElementById('statusDot');
const statusBadge = document.getElementById('statusBadge');
const resSelect = document.getElementById('resSelect');
const stage = document.getElementById('stage');
const video = document.getElementById('webcam');
const frozenCanvas = document.getElementById('frozenCanvas');
const maskCanvas = document.getElementById('maskCanvas');
const overlayCanvas = document.getElementById('overlayCanvas');
const emptyState = document.getElementById('emptyState');

const toggleCamBtn = document.getElementById('toggleCamBtn');
const modeBtn = document.getElementById('modeBtn');
const snapBtn = document.getElementById('snapBtn');
const clearBtn = document.getElementById('clearBtn');
const hoverToggle = document.getElementById('hoverToggle');
const mirrorToggle = document.getElementById('mirrorToggle');
const opacitySlider = document.getElementById('opacitySlider');

const metricModel = document.getElementById('metricModel');
const metricBackend = document.getElementById('metricBackend');
const metricEncode = document.getElementById('metricEncode');
const metricDecode = document.getElementById('metricDecode');
const metricIou = document.getElementById('metricIou');

const preprocessCanvas = document.createElement('canvas');
const preprocessCtx = preprocessCanvas.getContext('2d', { willReadFrequently: true });
const maskBufCanvas = document.createElement('canvas');
const maskBufCtx = maskBufCanvas.getContext('2d');
const frozenCtx = frozenCanvas.getContext('2d');
const maskCtx = maskCanvas.getContext('2d');
const overlayCtx = overlayCanvas.getContext('2d');

// State
let encoderModel = null;
let encoderPromise = null;
let decoderModel = null;

let stream = null;
let liveMode = true;
let encodeRunning = false;
let pendingEncode = null;
let encodeBackoff = new Backoff();

let decodeRunning = false;
let pendingDecode = false;

let currentEmbedding = null; // Float32Array (1 * 256 * 64 * 64)
let frameInfo = null; // { width, height, samScale }
let loggedFirstRun = false;

let pinnedPoints = [];
let hoverPoint = null;

// ---- Utilities ----

function setStatus(state, text) {
  statusDot.className = `dot ${state}`;
  statusBadge.textContent = text;
}

function formatMB(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getPreprocessShape(oldH, oldW, longSide) {
  const scale = longSide / Math.max(oldH, oldW);
  return {
    newH: Math.floor(oldH * scale + 0.5),
    newW: Math.floor(oldW * scale + 0.5),
    scale,
  };
}

function getSelectedEncSize() {
  return parseInt(resSelect.value, 10) || 640;
}

function syncCanvasSizes(w, h) {
  if (!w || !h) return;
  if (maskCanvas.width !== w || maskCanvas.height !== h) {
    maskCanvas.width = w;
    maskCanvas.height = h;
    overlayCanvas.width = w;
    overlayCanvas.height = h;
    frozenCanvas.width = w;
    frozenCanvas.height = h;
  }
}

// ---- Model Loading ----

async function loadDecoder() {
  metricModel.textContent = `Loading ${DECODER_FILENAME}…`;
  const t0 = performance.now();
  try {
    const model = await loadOnnxModel(DECODER_URL);
    const loadMs = performance.now() - t0;
    decoderModel = model;
    metricModel.textContent = `${DECODER_FILENAME} (${loadMs.toFixed(0)} ms)`;
    metricBackend.textContent =
      model.backend === 'webgpu'
        ? 'WebGPU'
        : `WASM (${self.crossOriginIsolated ? 'Threaded' : '1T'})`;
    scheduleDecode();
  } catch (err) {
    console.error(`Failed to load ${DECODER_FILENAME}:`, err);
    metricModel.textContent = `Error: ${DECODER_FILENAME}`;
  }
}

function ensureEncoderLoaded() {
  if (encoderModel) return Promise.resolve(encoderModel);
  if (encoderPromise) return encoderPromise;

  setStatus('loading', 'Downloading MobileSAM FP16 encoder…');
  const t0 = performance.now();
  encoderPromise = loadOnnxModel(ENCODER_URL, {
    onProgress: (loaded, total) => {
      if (total > 0) {
        const pct = Math.min(100, Math.round((loaded / total) * 100));
        setStatus('loading', `Downloading encoder ${pct}% (${formatMB(loaded)} / ${formatMB(total)})`);
      } else {
        setStatus('loading', `Downloading encoder (${formatMB(loaded)})…`);
      }
    },
  })
    .then((model) => {
      encoderModel = model;
      const sec = ((performance.now() - t0) / 1000).toFixed(1);
      setStatus('ready', `Encoder ready (${model.backend.toUpperCase()}, ${sec}s)`);
      metricBackend.textContent =
        model.backend === 'webgpu'
          ? 'WebGPU'
          : `WASM (${self.crossOriginIsolated ? 'Threaded' : '1T'})`;
      if (stream && liveMode) queueWebcamFrame(false);
      else if (stream && !currentEmbedding) queueWebcamFrame(true);
      return model;
    })
    .catch((err) => {
      encoderPromise = null;
      console.error('Encoder load failed:', err);
      setStatus('error', `Encoder failed: ${err.message}`);
      throw err;
    });
  return encoderPromise;
}

// ---- Webcam & Latest-Frame In-Browser Encoder ----

function waitForVideoReady() {
  return new Promise((resolve) => {
    if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
      resolve();
      return;
    }
    const onReady = () => {
      if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
        video.removeEventListener('loadeddata', onReady);
        video.removeEventListener('canplay', onReady);
        video.removeEventListener('resize', onReady);
        resolve();
      }
    };
    video.addEventListener('loadeddata', onReady);
    video.addEventListener('canplay', onReady);
    video.addEventListener('resize', onReady);
  });
}

async function startWebcam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: 'user' },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    await waitForVideoReady();

    emptyState.classList.add('hidden');
    toggleCamBtn.textContent = 'Stop Webcam';
    toggleCamBtn.classList.remove('primary');
    toggleCamBtn.classList.add('danger');
    modeBtn.disabled = false;
    snapBtn.disabled = false;
    clearBtn.disabled = false;

    syncMirror();
    syncCanvasSizes(video.videoWidth, video.videoHeight);

    if (encoderModel) {
      queueWebcamFrame(!liveMode);
    } else {
      ensureEncoderLoaded().catch(() => {});
    }
  } catch (err) {
    console.error('Webcam error:', err);
    alert(`Could not access webcam: ${err.message}`);
  }
}

function stopWebcam() {
  pendingEncode = null;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  frozenCanvas.classList.add('hidden');
  emptyState.classList.remove('hidden');
  clearMask();
  pinnedPoints = [];
  hoverPoint = null;
  drawOverlay();

  toggleCamBtn.textContent = 'Start Webcam';
  toggleCamBtn.classList.add('primary');
  toggleCamBtn.classList.remove('danger');
  modeBtn.disabled = true;
  snapBtn.disabled = true;
  clearBtn.disabled = true;
}

function grabCurrentFrame(freezeFrame) {
  if (!stream || video.readyState < 2) return null;
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return null;
  syncCanvasSizes(vw, vh);

  if (freezeFrame) {
    frozenCtx.drawImage(video, 0, 0, vw, vh);
    frozenCanvas.classList.remove('hidden');
  } else {
    frozenCanvas.classList.add('hidden');
  }

  const encSize = getSelectedEncSize();
  const { newH, newW } = getPreprocessShape(vh, vw, encSize);
  if (preprocessCanvas.width !== newW || preprocessCanvas.height !== newH) {
    preprocessCanvas.width = newW;
    preprocessCanvas.height = newH;
  }
  preprocessCtx.drawImage(video, 0, 0, newW, newH);
  const rgba = preprocessCtx.getImageData(0, 0, newW, newH).data;
  return { rgba, newW, newH, encSize, origW: vw, origH: vh };
}

function queueWebcamFrame(freezeFrame) {
  if (!encoderModel || !stream || !encodeBackoff.ready) return;
  const frame = grabCurrentFrame(freezeFrame);
  if (!frame) {
    // Video frame not ready yet — retry on next animation frame
    requestAnimationFrame(() => queueWebcamFrame(freezeFrame));
    return;
  }
  if (encodeRunning) {
    pendingEncode = frame;
  } else {
    runEncode(frame);
  }
}

function hasNonFinite(arr) {
  const step = Math.max(1, Math.floor(arr.length / 256));
  for (let i = 0; i < arr.length; i += step) {
    if (!Number.isFinite(arr[i])) return true;
  }
  return false;
}

async function runEncode(frame) {
  encodeRunning = true;
  const model = encoderModel;
  const { rgba, newW, newH, encSize } = frame;

  const plane = encSize * encSize;
  const data = new Float32Array(3 * plane);
  const invStd0 = 1 / PIXEL_STD[0];
  const invStd1 = 1 / PIXEL_STD[1];
  const invStd2 = 1 / PIXEL_STD[2];

  for (let y = 0; y < newH; y++) {
    const srcRow = y * newW * 4;
    const dstRow = y * encSize;
    for (let x = 0; x < newW; x++) {
      const s = srcRow + x * 4;
      const d = dstRow + x;
      data[d] = (rgba[s] - PIXEL_MEAN[0]) * invStd0;
      data[d + plane] = (rgba[s + 1] - PIXEL_MEAN[1]) * invStd1;
      data[d + 2 * plane] = (rgba[s + 2] - PIXEL_MEAN[2]) * invStd2;
    }
  }

  const t0 = performance.now();
  try {
    const inputTensor = new model.ort.Tensor('float32', data, [1, 3, encSize, encSize]);
    const out = await model.run({ input_image: inputTensor }, { priority: 0 });
    const emb = new Float32Array(out.image_embeddings.data);
    if (hasNonFinite(emb)) {
      throw new Error(`Encoder produced NaN/Inf on ${model.backend}`);
    }
    if (!loggedFirstRun) {
      let emin = Infinity, emax = -Infinity;
      for (let i = 0; i < emb.length; i += 64) {
        if (emb[i] < emin) emin = emb[i];
        if (emb[i] > emax) emax = emb[i];
      }
      console.log(`[SAM] First encode (${model.backend}, ${encSize}px): min=${emin.toFixed(3)}, max=${emax.toFixed(3)}, mask=${newW}x${newH}`);
    }
    currentEmbedding = emb;
    // Decoder operates in 1024 frame; output mask at (newH, newW) for fast readback & draw
    frameInfo = { maskW: newW, maskH: newH, encSize };
    encodeBackoff.reset();

    const encodeMs = performance.now() - t0;
    const fps = encodeMs > 0 ? (1000 / encodeMs).toFixed(1) : '—';
    metricEncode.textContent = `${encodeMs.toFixed(0)} ms (${fps} FPS, ${encSize}px)`;
    setStatus('ready', `Live (${newW}×${newH}) • Hover or click video to segment`);
    scheduleDecode();
  } catch (err) {
    console.error(`Encoder run failed on ${model.backend}:`, err);
    if (model.backend === 'webgpu') {
      setStatus('loading', 'WebGPU encoder issue; switching to WASM…');
      try {
        encoderModel = await loadOnnxModel(ENCODER_URL, { forceBackend: 'wasm' });
        setStatus('ready', 'Encoder ready (WASM fallback)');
        metricBackend.textContent = `WASM (${self.crossOriginIsolated ? 'Threaded' : '1T'})`;
        encodeBackoff.reset();
      } catch (wasmErr) {
        console.error('WASM encoder fallback failed:', wasmErr);
        setStatus('error', `Encoder failed: ${wasmErr.message}`);
        encodeBackoff.fail();
      }
    } else {
      encodeBackoff.fail();
      metricEncode.textContent = 'error (retrying)';
    }
  }

  encodeRunning = false;

  if (!stream) return;

  if (pendingEncode) {
    const next = pendingEncode;
    pendingEncode = null;
    setTimeout(() => runEncode(next), 50);
  } else if (liveMode) {
    setTimeout(() => {
      if (liveMode && stream && !encodeRunning) {
        queueWebcamFrame(false);
      }
    }, 80);
  }
}

// ---- High-Priority In-Browser ONNX Mask Decoder ----

function getActivePoints() {
  const pts = pinnedPoints.slice();
  if (hoverToggle.checked && hoverPoint) {
    pts.push(hoverPoint);
  }
  return pts;
}

function scheduleDecode() {
  if (!decoderModel || !currentEmbedding || !frameInfo) return;
  if (decodeRunning) {
    pendingDecode = true;
    return;
  }
  requestAnimationFrame(runDecode);
}

async function runDecode() {
  if (!decoderModel || !currentEmbedding || !frameInfo) return;
  const pts = getActivePoints();
  if (pts.length === 0) {
    clearMask();
    drawOverlay();
    return;
  }

  decodeRunning = true;
  pendingDecode = false;

  const model = decoderModel;
  const ort = model.ort;

  try {
    const n = pts.length;
    const coords = new Float32Array(2 * (n + 1));
    const labels = new Float32Array(n + 1);

    const coordScale = SAM_DECODER_SIZE / frameInfo.encSize;
    for (let i = 0; i < n; i++) {
      coords[2 * i] = pts[i].x * frameInfo.maskW * coordScale;
      coords[2 * i + 1] = pts[i].y * frameInfo.maskH * coordScale;
      labels[i] = pts[i].label;
    }
    coords[2 * n] = 0.0;
    coords[2 * n + 1] = 0.0;
    labels[n] = -1.0;

    const feeds = {
      image_embeddings: new ort.Tensor('float32', new Float32Array(currentEmbedding), [1, 256, 64, 64]),
      point_coords: new ort.Tensor('float32', coords, [1, n + 1, 2]),
      point_labels: new ort.Tensor('float32', labels, [1, n + 1]),
      mask_input: new ort.Tensor('float32', new Float32Array(256 * 256), [1, 1, 256, 256]),
      has_mask_input: new ort.Tensor('float32', new Float32Array([0]), [1]),
      orig_im_size: new ort.Tensor('float32', new Float32Array([frameInfo.maskH, frameInfo.maskW]), [2]),
    };

    const t0 = performance.now();
    const results = await model.run(feeds, { priority: 1 });
    const decodeMs = performance.now() - t0;

    const maskTensor = results[model.session.outputNames[0]];
    const iouTensor = results[model.session.outputNames[1]];

    if (hasNonFinite(maskTensor.data)) {
      console.error(`Decoder produced NaN/Inf on ${model.backend}; switching decoder to WASM`);
      if (model.backend === 'webgpu') {
        decoderModel = await loadOnnxModel(DECODER_URL, { forceBackend: 'wasm' });
      }
      return;
    }

    const fgPixels = renderMask(maskTensor.data, maskTensor.dims[3], maskTensor.dims[2]);
    if (!loggedFirstRun) {
      loggedFirstRun = true;
      let lmin = Infinity, lmax = -Infinity;
      const d = maskTensor.data;
      for (let i = 0; i < d.length; i += 64) {
        if (d[i] < lmin) lmin = d[i];
        if (d[i] > lmax) lmax = d[i];
      }
      console.log(`[SAM] First decode (${model.backend}): logits=[${lmin.toFixed(2)}, ${lmax.toFixed(2)}], fgPixels=${fgPixels}, iou=${iouTensor?.data?.[0]}`);
    }
    drawOverlay();

    const fps = decodeMs > 0 ? (1000 / decodeMs).toFixed(1) : '—';
    metricDecode.textContent = `${decodeMs.toFixed(1)} ms (${fps} FPS, ${model.backend})`;
    if (iouTensor && iouTensor.data && iouTensor.data.length > 0) {
      metricIou.textContent = `${Number(iouTensor.data[0]).toFixed(3)} (${fgPixels.toLocaleString()} px)`;
    }
  } catch (err) {
    console.error(`Decode error (${DECODER_FILENAME}):`, err);
  } finally {
    decodeRunning = false;
    if (pendingDecode) {
      pendingDecode = false;
      scheduleDecode();
    }
  }
}

// ---- Rendering & Pointer Mapping ----

function renderMask(logits, w, h) {
  if (maskBufCanvas.width !== w || maskBufCanvas.height !== h) {
    maskBufCanvas.width = w;
    maskBufCanvas.height = h;
  }
  const imgData = maskBufCtx.createImageData(w, h);
  const data = imgData.data;
  const alpha = Math.round(parseFloat(opacitySlider.value) * 255);
  const [r, g, b] = [30, 144, 255];

  let fgPixels = 0;
  for (let i = 0, p = 0; i < logits.length; i++, p += 4) {
    if (logits[i] > 0.0) {
      data[p] = r;
      data[p + 1] = g;
      data[p + 2] = b;
      data[p + 3] = alpha;
      fgPixels++;
    }
  }
  maskBufCtx.putImageData(imgData, 0, 0);
  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  maskCtx.drawImage(maskBufCanvas, 0, 0, maskCanvas.width, maskCanvas.height);
  return fgPixels;
}

function clearMask() {
  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
}

function drawOverlay() {
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  const w = overlayCanvas.width;
  const h = overlayCanvas.height;

  for (const pt of pinnedPoints) {
    overlayCtx.beginPath();
    overlayCtx.arc(pt.x * w, pt.y * h, 7, 0, Math.PI * 2);
    overlayCtx.fillStyle = pt.label === 1 ? '#22c55e' : '#ef4444';
    overlayCtx.fill();
    overlayCtx.lineWidth = 2;
    overlayCtx.strokeStyle = '#ffffff';
    overlayCtx.stroke();
  }

  if (hoverToggle.checked && hoverPoint) {
    overlayCtx.beginPath();
    overlayCtx.arc(hoverPoint.x * w, hoverPoint.y * h, 5, 0, Math.PI * 2);
    overlayCtx.fillStyle = 'rgba(59, 130, 246, 0.85)';
    overlayCtx.fill();
    overlayCtx.lineWidth = 1.5;
    overlayCtx.strokeStyle = '#ffffff';
    overlayCtx.stroke();
  }
}

function eventToNormalizedCoords(e) {
  const vw = maskCanvas.width || video.videoWidth;
  const vh = maskCanvas.height || video.videoHeight;
  if (!vw || !vh) return null;

  const rect = stage.getBoundingClientRect();
  const stageAspect = rect.width / rect.height;
  const imgAspect = vw / vh;

  let drawW, drawH, offsetX, offsetY;
  if (imgAspect > stageAspect) {
    drawW = rect.width;
    drawH = rect.width / imgAspect;
    offsetX = 0;
    offsetY = (rect.height - drawH) / 2;
  } else {
    drawH = rect.height;
    drawW = rect.height * imgAspect;
    offsetX = (rect.width - drawW) / 2;
    offsetY = 0;
  }

  let nx = (e.clientX - rect.left - offsetX) / drawW;
  let ny = (e.clientY - rect.top - offsetY) / drawH;
  if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return null;

  if (mirrorToggle.checked) {
    nx = 1 - nx;
  }
  return { x: nx, y: ny };
}

function syncMirror() {
  stage.classList.toggle('mirrored', mirrorToggle.checked);
}

// ---- Event Listeners ----

resSelect.addEventListener('change', () => {
  pendingEncode = null;
  if (stream && encoderModel) {
    queueWebcamFrame(!liveMode);
  }
});
mirrorToggle.addEventListener('change', syncMirror);
opacitySlider.addEventListener('input', scheduleDecode);

toggleCamBtn.addEventListener('click', () => {
  if (stream) stopWebcam();
  else startWebcam();
});

modeBtn.addEventListener('click', () => {
  liveMode = !liveMode;
  modeBtn.textContent = liveMode ? 'Mode: Live Auto-Encode' : 'Mode: Freeze Frame';
  if (liveMode) {
    frozenCanvas.classList.add('hidden');
    queueWebcamFrame(false);
  } else {
    pendingEncode = null;
    queueWebcamFrame(true);
  }
});

snapBtn.addEventListener('click', () => queueWebcamFrame(!liveMode));

clearBtn.addEventListener('click', () => {
  pinnedPoints = [];
  clearMask();
  drawOverlay();
  scheduleDecode();
});

stage.addEventListener('pointermove', (e) => {
  if (!stream || !hoverToggle.checked) return;
  const coords = eventToNormalizedCoords(e);
  hoverPoint = coords ? { x: coords.x, y: coords.y, label: 1 } : null;
  drawOverlay();
  scheduleDecode();
});

stage.addEventListener('pointerleave', () => {
  hoverPoint = null;
  drawOverlay();
  if (pinnedPoints.length === 0) clearMask();
  else scheduleDecode();
});

stage.addEventListener('click', (e) => {
  if (!stream) return;
  const coords = eventToNormalizedCoords(e);
  if (!coords) return;
  pinnedPoints.push({ x: coords.x, y: coords.y, label: 1 });
  drawOverlay();
  scheduleDecode();
});

stage.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  if (!stream) return;
  const coords = eventToNormalizedCoords(e);
  if (!coords) return;
  pinnedPoints.push({ x: coords.x, y: coords.y, label: 0 });
  drawOverlay();
  scheduleDecode();
});

// ---- Init ----
syncMirror();
loadDecoder().then(() => ensureEncoderLoaded().catch(() => {}));
