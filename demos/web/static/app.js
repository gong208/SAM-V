const fileInput = document.getElementById("fileInput");
const frameList = document.getElementById("frameList");
const resultList = document.getElementById("resultList");
const canvas = document.getElementById("promptCanvas");
const emptyState = document.getElementById("emptyState");
const statusText = document.getElementById("status");
const runButton = document.getElementById("runButton");
const tipsButton = document.getElementById("tipsButton");
const tipsModal = document.getElementById("tipsModal");
const closeTipsButton = document.getElementById("closeTipsButton");
const tipsOriginalImages = Array.from(document.querySelectorAll("[data-tips-original]"));
const tipsOverlayImages = Array.from(document.querySelectorAll("[data-tips-overlay]"));
const undoButton = document.getElementById("undoButton");
const clearButton = document.getElementById("clearButton");
const ctx = canvas.getContext("2d");

let frames = [];
let selectedFrame = 0;
let prompts = [];
let drawRect = { x: 0, y: 0, width: 0, height: 0 };
const tipsPreviewOriginalUrls = [
  "/static/tips/frame_000_original.JPG",
  "/static/tips/frame_001_original.JPG",
  "/static/tips/frame_002_original.JPG",
  "/static/tips/frame_003_original.JPG",
];
const tipsPreviewOverlayUrls = [
  "/static/tips/frame_000_overlay.png",
  "/static/tips/frame_001_overlay.png",
  "/static/tips/frame_002_overlay.png",
  "/static/tips/frame_003_overlay.png",
];

function setStatus(text) {
  statusText.textContent = text;
}

function updateButtons() {
  runButton.disabled = frames.length === 0 || prompts.length === 0;
  undoButton.disabled = prompts.length === 0;
  clearButton.disabled = prompts.length === 0;
}

function updateTipsPreview() {
  tipsOriginalImages.forEach((image, index) => {
    image.src = tipsPreviewOriginalUrls[index];
  });
  tipsOverlayImages.forEach((image, index) => {
    image.src = tipsPreviewOverlayUrls[index];
  });
}

function openTipsModal() {
  updateTipsPreview();
  tipsModal.classList.remove("hidden");
  closeTipsButton.focus();
}

function closeTipsModal() {
  tipsModal.classList.add("hidden");
}

function fitCanvasToPanel(image) {
  const wrap = canvas.parentElement.getBoundingClientRect();
  const maxW = Math.max(240, wrap.width - 32);
  const maxH = Math.max(240, wrap.height - 32);
  const scale = Math.min(maxW / image.naturalWidth, maxH / image.naturalHeight, 1);
  canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
  canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
  drawRect = { x: 0, y: 0, width: canvas.width, height: canvas.height };
}

function resizeAndDrawPromptCanvas() {
  if (!frames.length) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    emptyState.classList.remove("hidden");
    return;
  }

  emptyState.classList.add("hidden");
  const frame = frames[selectedFrame];
  fitCanvasToPanel(frame.image);
  drawPromptCanvas();
}

function drawPromptCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!frames.length) {
    emptyState.classList.remove("hidden");
    return;
  }

  emptyState.classList.add("hidden");
  const frame = frames[selectedFrame];
  ctx.drawImage(frame.image, 0, 0, canvas.width, canvas.height);

  const scaleX = canvas.width / frame.image.naturalWidth;
  const scaleY = canvas.height / frame.image.naturalHeight;
  prompts
    .filter((point) => point.frame_index === selectedFrame)
    .forEach((point) => {
      const x = point.x * scaleX;
      const y = point.y * scaleY;
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, Math.PI * 2);
      ctx.fillStyle = "#ffdc28";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#141414";
      ctx.stroke();
    });
}

function renderFrameList() {
  frameList.innerHTML = "";
  frames.forEach((frame, index) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `frame-item ${index === selectedFrame ? "active" : ""}`;
    item.addEventListener("click", () => {
      selectedFrame = index;
      renderFrameList();
      resizeAndDrawPromptCanvas();
    });

    const img = document.createElement("img");
    img.src = frame.url;
    img.alt = frame.file.name;
    const meta = document.createElement("div");
    const count = prompts.filter((point) => point.frame_index === index).length;
    meta.className = "frame-meta";
    meta.textContent = `${index + 1}. ${frame.file.name} - ${count} pts`;

    item.append(img, meta);
    frameList.appendChild(item);
  });
}

function renderResults(results) {
  resultList.innerHTML = "";
  results.frames.forEach((frame) => {
    const item = document.createElement("div");
    item.className = "result-item";

    const link = document.createElement("a");
    link.href = frame.overlay_url;
    link.target = "_blank";
    link.rel = "noreferrer";
    const img = document.createElement("img");
    img.src = `${frame.overlay_url}?t=${Date.now()}`;
    img.alt = `${frame.filename} overlay`;
    link.appendChild(img);

    const meta = document.createElement("div");
    meta.className = "result-meta";
    meta.textContent = `${frame.index + 1}. ${frame.filename} - ${frame.foreground_pixels} px`;
    item.append(link, meta);
    resultList.appendChild(item);
  });
}

function clearObjectUrls() {
  frames.forEach((frame) => URL.revokeObjectURL(frame.url));
}

fileInput.addEventListener("change", async () => {
  clearObjectUrls();
  frames = [];
  prompts = [];
  selectedFrame = 0;
  resultList.innerHTML = "";

  const files = Array.from(fileInput.files || []);
  for (const file of files) {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.src = url;
    await image.decode();
    frames.push({ file, url, image });
  }

  renderFrameList();
  resizeAndDrawPromptCanvas();
  updateButtons();
  setStatus(frames.length ? `${frames.length} frame(s) loaded` : "Idle");
});

canvas.addEventListener("click", (event) => {
  if (!frames.length) return;
  const rect = canvas.getBoundingClientRect();
  const frame = frames[selectedFrame];
  const xCanvas = event.clientX - rect.left - drawRect.x;
  const yCanvas = event.clientY - rect.top - drawRect.y;
  const x = xCanvas * (frame.image.naturalWidth / drawRect.width);
  const y = yCanvas * (frame.image.naturalHeight / drawRect.height);
  if (x < 0 || y < 0 || x >= frame.image.naturalWidth || y >= frame.image.naturalHeight) return;
  prompts.push({ frame_index: selectedFrame, x, y, label: 1 });
  renderFrameList();
  drawPromptCanvas();
  updateButtons();
});

undoButton.addEventListener("click", () => {
  prompts.pop();
  renderFrameList();
  drawPromptCanvas();
  updateButtons();
});
clearButton.addEventListener("click", () => {
  prompts = prompts.filter((point) => point.frame_index !== selectedFrame);
  renderFrameList();
  drawPromptCanvas();
  updateButtons();
});

runButton.addEventListener("click", async () => {
  if (!frames.length || !prompts.length) return;
  runButton.disabled = true;
  setStatus("Running");

  const form = new FormData();
  frames.forEach((frame) => form.append("images", frame.file, frame.file.name));
  form.append("prompts", JSON.stringify(prompts));

  try {
    const response = await fetch("/api/infer", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Inference failed");
    renderResults(payload);
    const seconds = payload.runtime_seconds.toFixed(2);
    setStatus(`Done in ${seconds}s`);
  } catch (error) {
    setStatus(error.message);
  } finally {
    updateButtons();
  }
});

tipsButton.addEventListener("click", openTipsModal);
closeTipsButton.addEventListener("click", closeTipsModal);
tipsModal.addEventListener("click", (event) => {
  if (event.target === tipsModal) closeTipsModal();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !tipsModal.classList.contains("hidden")) {
    closeTipsModal();
  }
});

window.addEventListener("resize", resizeAndDrawPromptCanvas);
updateButtons();
