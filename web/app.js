import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.wasm.bundle.min.mjs";

ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";
ort.env.wasm.numThreads = 1;

const DEPTH_MODEL_URL =
  "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model_quantized.onnx";
const DEPTH_INPUT_SIZE = 518;
const SIZE_MULTIPLE = 14;
const IMAGE_MEAN = [0.485, 0.456, 0.406];
const IMAGE_STD = [0.229, 0.224, 0.225];
// Large images are slower in WebAssembly than they are worth here.
const MAX_SIDE = 2048;

const el = (id) => document.getElementById(id);
const ui = {
  stage: el("stage"),
  dropzone: el("dropzone"),
  file: el("file"),
  pick: el("pick"),
  sample: el("sample"),
  source: el("source"),
  sourceLabel: el("source-label"),
  sourceMeta: el("source-meta"),
  output: el("output"),
  outputMeta: el("output-meta"),
  controls: el("controls"),
  method: el("method"),
  strength: el("strength"),
  strengthValue: el("strength-value"),
  range: el("range"),
  invert: el("invert"),
  estimate: el("estimate"),
  download: el("download"),
  reset: el("reset"),
  status: el("status"),
};

const conversionSessions = new Map();
let depthSession = null;
let state = { depth: null, photo: null };

const status = (text, busy = false) => {
  ui.status.textContent = text;
  ui.status.dataset.busy = String(busy);
};

async function conversionSession(name) {
  if (!conversionSessions.has(name)) {
    status(`Loading ${name}.onnx`, true);
    conversionSessions.set(
      name,
      await ort.InferenceSession.create(`models/${name}.onnx`),
    );
  }
  return conversionSessions.get(name);
}

/** Draw a bitmap into a canvas, capped at MAX_SIDE, and read its pixels. */
function pixelsOf(bitmap) {
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0, width, height);
  return context.getImageData(0, 0, width, height);
}

/** Luminance of an image, which is the depth for a grayscale depth map. */
function toDepth(image) {
  const values = new Float32Array(image.width * image.height);
  for (let i = 0; i < values.length; i += 1) {
    const p = i * 4;
    values[i] =
      0.299 * image.data[p] + 0.587 * image.data[p + 1] + 0.114 * image.data[p + 2];
  }
  return { data: values, width: image.width, height: image.height };
}

/** How colourful an image is, as a hint that it is a photo, not a depth map. */
function colourfulness(image) {
  let total = 0;
  const step = 4 * 37; // a prime stride, so the sample is spread out
  let count = 0;
  for (let p = 0; p < image.data.length; p += step) {
    const r = image.data[p];
    const g = image.data[p + 1];
    const b = image.data[p + 2];
    const high = Math.max(r, g, b);
    total += high === 0 ? 0 : (high - Math.min(r, g, b)) / high;
    count += 1;
  }
  return count ? total / count : 0;
}

/** The JavaScript twin of depth2normal.rescale_depth. */
function rescale(depth, mode, invert) {
  const values = Float32Array.from(depth.data);
  let low = Infinity;
  let high = -Infinity;
  for (const value of values) {
    if (value < low) low = value;
    if (value > high) high = value;
  }
  if (invert) {
    for (let i = 0; i < values.length; i += 1) values[i] = high - values[i];
    [low, high] = [0, high - low];
  }
  if (mode === "raw") return values;

  let scale = 1;
  let offset = 0;
  if (mode === "minmax") {
    const span = high - low;
    scale = span ? 255 / span : 0;
    offset = low;
  } else if (high <= 1) {
    scale = 255;
  } else if (high > 255) {
    scale = high <= 65535 ? 255 / 65535 : 255 / (high - low);
    offset = high <= 65535 ? 0 : low;
  }
  if (scale !== 1 || offset !== 0) {
    for (let i = 0; i < values.length; i += 1) values[i] = (values[i] - offset) * scale;
  }
  return values;
}

function paint(canvas, imageData) {
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.getContext("2d").putImageData(imageData, 0, 0);
}

function paintDepth(values, width, height) {
  const image = new ImageData(width, height);
  for (let i = 0; i < values.length; i += 1) {
    const level = Math.max(0, Math.min(255, values[i]));
    image.data.set([level, level, level, 255], i * 4);
  }
  paint(ui.source, image);
}

async function render() {
  if (!state.depth) return;
  const { width, height } = state.depth;
  const values = rescale(state.depth, ui.range.value, ui.invert.checked);
  paintDepth(values, width, height);

  const session = await conversionSession(ui.method.value);
  const feeds = {
    depth: new ort.Tensor("float32", values, [1, 1, height, width]),
    strength: new ort.Tensor(
      "float32",
      new Float32Array([Number(ui.strength.value)]),
      [],
    ),
  };

  const started = performance.now();
  const results = await session.run(feeds);
  const elapsed = Math.round(performance.now() - started);

  const rgb = results[session.outputNames[0]].data;
  const image = new ImageData(width, height);
  for (let i = 0; i < width * height; i += 1) {
    image.data.set([rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2], 255], i * 4);
  }
  paint(ui.output, image);
  ui.sourceMeta.textContent = `${width} \u00d7 ${height}`;
  ui.outputMeta.textContent = `${elapsed} ms`;
  status("");
}

async function loadDepthSession() {
  if (depthSession) return depthSession;

  status("Downloading the depth model (27 MB, once)", true);
  const response = await fetch(DEPTH_MODEL_URL);
  if (!response.ok) throw new Error(`model download failed: ${response.status}`);

  const total = Number(response.headers.get("Content-Length")) || 27_300_000;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    status(
      `Downloading the depth model: ${Math.round((100 * received) / total)}%`,
      true,
    );
  }

  const bytes = new Uint8Array(received);
  let at = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, at);
    at += chunk.length;
  }
  status("Starting the depth model", true);
  depthSession = await ort.InferenceSession.create(bytes);
  return depthSession;
}

/** Aspect kept, both sides a multiple of 14, as the model's config asks. */
function estimationSize(height, width) {
  const scaleHeight = DEPTH_INPUT_SIZE / height;
  const scaleWidth = DEPTH_INPUT_SIZE / width;
  const scale =
    Math.abs(1 - scaleWidth) < Math.abs(1 - scaleHeight) ? scaleWidth : scaleHeight;
  const snap = (value) =>
    Math.max(SIZE_MULTIPLE, Math.round((value * scale) / SIZE_MULTIPLE) * SIZE_MULTIPLE);
  return [snap(height), snap(width)];
}

async function estimate() {
  if (!state.photo) return;
  ui.estimate.disabled = true;
  try {
    const session = await loadDepthSession();
    const [height, width] = estimationSize(state.photo.height, state.photo.width);

    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext("2d", { willReadFrequently: true });
    context.drawImage(state.photo, 0, 0, width, height);
    const image = context.getImageData(0, 0, width, height);

    const input = new Float32Array(3 * height * width);
    const plane = height * width;
    for (let i = 0; i < plane; i += 1) {
      for (let c = 0; c < 3; c += 1) {
        input[c * plane + i] =
          (image.data[i * 4 + c] / 255 - IMAGE_MEAN[c]) / IMAGE_STD[c];
      }
    }

    status("Estimating depth", true);
    const started = performance.now();
    const results = await session.run({
      pixel_values: new ort.Tensor("float32", input, [1, 3, height, width]),
    });
    const predicted = results[session.outputNames[0]].data;
    const elapsed = Math.round(performance.now() - started);

    let low = Infinity;
    let high = -Infinity;
    for (const value of predicted) {
      if (value < low) low = value;
      if (value > high) high = value;
    }
    const span = high - low || 1;
    const grey = new ImageData(width, height);
    for (let i = 0; i < plane; i += 1) {
      const level = ((predicted[i] - low) / span) * 255;
      grey.data.set([level, level, level, 255], i * 4);
    }

    const scaled = new OffscreenCanvas(state.photo.width, state.photo.height);
    const target = scaled.getContext("2d", { willReadFrequently: true });
    const source = new OffscreenCanvas(width, height);
    source.getContext("2d").putImageData(grey, 0, 0);
    target.drawImage(source, 0, 0, scaled.width, scaled.height);

    state.depth = toDepth(pixelsOf(scaled));
    ui.sourceLabel.textContent = "Depth, estimated";
    await render();
    status(`Depth estimated in ${elapsed} ms`);
  } catch (error) {
    status(`Depth estimation failed: ${error.message}`);
  } finally {
    ui.estimate.disabled = false;
  }
}

async function open(source) {
  const bitmap = await createImageBitmap(source);
  const pixels = pixelsOf(bitmap);
  state = { depth: toDepth(pixels), photo: bitmap };
  ui.stage.classList.remove("empty");
  ui.controls.hidden = false;
  ui.sourceLabel.textContent = "Depth";
  await render();
  if (colourfulness(pixels) > 0.15) {
    status("This looks like a photo. Estimate its depth map for a real result.");
  }
}

async function openFile(file) {
  if (!file) return;
  try {
    await open(file);
  } catch (error) {
    status(`Could not open that file: ${error.message}`);
  }
}

ui.pick.addEventListener("click", () => ui.file.click());
ui.file.addEventListener("change", () => openFile(ui.file.files[0]));
ui.sample.addEventListener("click", async () => {
  const response = await fetch("sample-depth.png");
  await open(await response.blob());
});

ui.dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  ui.dropzone.classList.add("over");
});
ui.dropzone.addEventListener("dragleave", () =>
  ui.dropzone.classList.remove("over"),
);
document.addEventListener("drop", (event) => {
  event.preventDefault();
  ui.dropzone.classList.remove("over");
  openFile(event.dataTransfer?.files?.[0]);
});
document.addEventListener("dragover", (event) => event.preventDefault());

ui.method.addEventListener("change", render);
ui.range.addEventListener("change", render);
ui.invert.addEventListener("change", render);
ui.strength.addEventListener("input", () => {
  ui.strengthValue.textContent = Number(ui.strength.value).toFixed(1);
  render();
});
ui.estimate.addEventListener("click", estimate);
ui.download.addEventListener("click", () => {
  ui.output.toBlob((blob) => {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "normal_map.png";
    link.click();
    URL.revokeObjectURL(link.href);
  });
});
ui.reset.addEventListener("click", () => {
  state = { depth: null, photo: null };
  ui.stage.classList.add("empty");
  ui.controls.hidden = true;
  ui.file.value = "";
  ui.sourceMeta.textContent = "";
  ui.outputMeta.textContent = "";
  status("");
});
