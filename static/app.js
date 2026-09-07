const $ = (selector) => document.querySelector(selector);
const recordButton = $("#recordButton");
const stopButton = $("#stopButton");
const languageSelect = $("#language");
const modelSelect = $("#finalModel");
const vocabularyInput = $("#vocabulary");
const speakerCount = $("#speakerCount");
const selfCorrect = $("#selfCorrect");
const knownSpeakers = $("#knownSpeakers");
const prepPanel = $("#prepPanel");
const timeDisplay = $("#time");
const recordState = $("#recordState");
const statusPill = $("#statusPill");
const transcript = $("#transcript");
const reviewList = $("#reviewList");
const speakerList = $("#speakerList");
const qualityBar = $("#quality");
const tabs = $("#tabs");
const tabTranscript = $("#tabTranscript");
const tabReview = $("#tabReview");
const tabSpeakers = $("#tabSpeakers");
const reviewCount = $("#reviewCount");
const player = $("#player");
const finalizeStatus = $("#finalizeStatus");
const finalizeStage = $("#finalizeStage");
const resultActions = $("#resultActions");
const exportButton = $("#exportButton");
const transcriptButton = $("#transcriptButton");
const bundleButton = $("#bundleButton");
const footerState = $("#footerState");
const backlog = $("#backlog");
const hostWarnings = $("#hostWarnings");
const flagButton = $("#flagButton");
const jumpLive = $("#jumpLive");
const legend = $("#legend");
const signalWarning = $("#signalWarning");
const recoverPanel = $("#recoverPanel");
const recoverButton = $("#recoverButton");
const canvas = $("#waveform");
const canvasContext = canvas.getContext("2d");

const CHUNK_SECONDS = 6;
const LOW_CONFIDENCE = 0.55;
const FLAG_LABELS = {
  "very-low-confidence": "very low confidence",
  "low-confidence": "low confidence",
  "weak-segment": "weak audio",
  "maybe-silence": "possible non-speech",
  "repetitive-segment": "repetition",
  disputed: "models disagree",
  vocabulary: "vocabulary fix",
  "re-decoded": "re-transcribed",
  confirmed: "re-checked",
  edited: "corrected by hand",
  accepted: "second pass accepted",
};

const SEGMENT_FLAGS = new Set(["disputed", "re-decoded", "vocabulary", "edited", "accepted"]);
const SILENCE_LEVEL = 0.004;
const SILENCE_SECONDS = 60;

let sessionId = localStorage.getItem("threadmark-session");
let stream = null;
let audioContext = null;
let sourceNode = null;
let captureNode = null;
let analyser = null;
let silentGain = null;
let animationFrame = null;
let timer = null;
let pollTimer = null;
let startedAt = null;
let chunkIndex = 0;
let recording = false;
let pcmParts = [];
let pcmLength = 0;
let uploadQueue = [];
let uploadActive = false;
let uploadWaiters = [];
let renderedChunks = new Set();
let latest = null;
let stickToBottom = true;
let quietSince = null;
let wordSpans = [];
let reviewFilter = "all";
let activeTab = "transcript";

function setStatus(label, mode = "") {
  statusPill.className = `status-pill ${mode}`.trim();
  statusPill.querySelector("b").textContent = label;
}

function clock(seconds) {
  const value = Math.max(0, Math.floor(seconds));
  return [Math.floor(value / 3600), Math.floor((value % 3600) / 60), value % 60]
    .map((part) => String(part).padStart(2, "0")).join(":");
}

function elapsed() {
  return startedAt ? (Date.now() - startedAt) / 1000 : 0;
}

function drawWaveform() {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    canvasContext.setTransform(ratio, 0, 0, ratio, 0, 0);
  }
  canvasContext.clearRect(0, 0, width, height);
  if (analyser) {
    const values = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteTimeDomainData(values);
    canvasContext.beginPath();
    canvasContext.strokeStyle = recording ? "#b8ff4f" : "#3b463e";
    canvasContext.lineWidth = 2;
    values.forEach((value, index) => {
      const x = (index / (values.length - 1)) * width;
      const y = (value / 255) * height;
      index ? canvasContext.lineTo(x, y) : canvasContext.moveTo(x, y);
    });
    canvasContext.stroke();
    if (recording) checkSilence(values);
  }
  animationFrame = requestAnimationFrame(drawWaveform);
}

function checkSilence(values) {
  let sum = 0;
  values.forEach((value) => { const offset = (value - 128) / 128; sum += offset * offset; });
  const level = Math.sqrt(sum / values.length);
  if (level > SILENCE_LEVEL) {
    quietSince = null;
    signalWarning.classList.add("hidden");
    return;
  }
  quietSince ??= Date.now();
  const quiet = (Date.now() - quietSince) / 1000;
  if (quiet > SILENCE_SECONDS) {
    signalWarning.textContent = `No sound for ${Math.floor(quiet / 60)} min — check the microphone.`;
    signalWarning.classList.remove("hidden");
  }
}

function showError(message) {
  recordState.textContent = message;
  recordState.classList.add("error");
  setStatus("Needs attention");
  footerState.textContent = "Error";
}

function renderWarnings(host) {
  const warnings = host?.warnings ?? [];
  hostWarnings.classList.toggle("hidden", !warnings.length);
  hostWarnings.replaceChildren(...warnings.map((text) => {
    const row = document.createElement("div");
    row.textContent = text;
    return row;
  }));
}

// --------------------------------------------------------------------------- //
// Live capture
// --------------------------------------------------------------------------- //

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset, text) => [...text].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
  write(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); write(8, "WAVE");
  write(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  write(36, "data"); view.setUint32(40, samples.length * 2, true);
  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(44 + index * 2, sample < 0 ? sample * 32768 : sample * 32767, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function takeSamples(count) {
  const output = new Float32Array(count);
  let written = 0;
  while (written < count && pcmParts.length) {
    const head = pcmParts[0];
    const amount = Math.min(head.length, count - written);
    output.set(head.subarray(0, amount), written);
    written += amount;
    pcmLength -= amount;
    pcmParts[0] = head.subarray(amount);
    if (!pcmParts[0].length) pcmParts.shift();
  }
  return output;
}

function enqueueChunk(samples) {
  if (!samples.length) return;
  const index = chunkIndex++;
  const offset = index * CHUNK_SECONDS;
  uploadQueue.push({ blob: encodeWav(samples, audioContext.sampleRate), index, offset });
  pumpUploads();
}

function receiveSamples(samples) {
  pcmParts.push(samples);
  pcmLength += samples.length;
  const target = Math.round(audioContext.sampleRate * CHUNK_SECONDS);
  while (pcmLength >= target) enqueueChunk(takeSamples(target));
}

async function pumpUploads() {
  if (uploadActive) return;
  uploadActive = true;
  while (uploadQueue.length) {
    const item = uploadQueue[0];
    const form = new FormData();
    form.append("audio", item.blob, `chunk-${String(item.index).padStart(6, "0")}.wav`);
    form.append("index", String(item.index));
    form.append("offset", String(item.offset));
    try {
      const response = await fetch(`/api/sessions/${sessionId}/chunks`, { method: "POST", body: form });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Audio upload failed");
      uploadQueue.shift();
    } catch (error) {
      showError(`${error.message}. Retrying…`);
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  }
  uploadActive = false;
  uploadWaiters.splice(0).forEach((resolve) => resolve());
}

function waitForUploads() {
  if (!uploadActive && !uploadQueue.length) return Promise.resolve();
  return new Promise((resolve) => uploadWaiters.push(resolve));
}

// --------------------------------------------------------------------------- //
// Rendering
// --------------------------------------------------------------------------- //

function appendLiveUtterance(item) {
  $("#emptyState")?.remove();
  const row = document.createElement("article");
  row.className = "utterance provisional";
  row.innerHTML = `<time></time><div class="utterance-body"><strong></strong><p></p></div>`;
  row.querySelector("time").textContent = clock(item.offset ?? 0);
  const uncertain = item.confidence !== null && item.confidence < LOW_CONFIDENCE;
  row.querySelector("strong").textContent = uncertain ? "Live transcript · low confidence" : "Live transcript";
  if (uncertain) row.classList.add("uncertain");
  row.querySelector("p").textContent = item.text;
  transcript.appendChild(row);
  if (stickToBottom) transcript.scrollTop = transcript.scrollHeight;
  else jumpLive.classList.remove("hidden");
}

function seek(seconds) {
  if (!player.src) return;
  player.classList.remove("hidden");
  player.currentTime = Math.max(0, seconds - 0.4);
  player.play().catch(() => {});
}

function wordNode(word) {
  const flags = word.flags ?? [];
  const node = document.createElement("span");
  node.className = "word";
  node.textContent = `${word.text} `;
  node.dataset.start = String(word.start);
  node.dataset.end = String(word.end);
  node.addEventListener("click", () => seek(word.start));
  wordSpans.push(node);
  if (flags.includes("disputed")) {
    node.classList.add("disputed");
    node.title = `Second pass heard: ${word.alternative ?? "something different"}`;
  } else if (word.probability < 0.35) {
    node.classList.add("very-low");
    node.title = `Confidence ${Math.round(word.probability * 100)}%`;
  } else if (word.probability < LOW_CONFIDENCE) {
    node.classList.add("low");
    node.title = `Confidence ${Math.round(word.probability * 100)}%`;
  }
  if (flags.includes("vocabulary")) {
    node.classList.add("fixed");
    node.title = `Corrected to meeting vocabulary from "${word.corrected_from ?? ""}"`;
  }
  return node;
}

function renderSegment(group, index) {
  const row = document.createElement("article");
  row.className = "utterance final";
  row.dataset.segment = String(index);
  if (group.confidence < LOW_CONFIDENCE) row.classList.add("uncertain");

  const time = document.createElement("time");
  time.textContent = clock(group.start);
  time.className = "seek";
  time.title = "Play this passage";
  time.addEventListener("click", () => seek(group.start));

  const body = document.createElement("div");
  body.className = "utterance-body";
  const heading = document.createElement("strong");
  heading.textContent = group.speaker;
  const marks = group.edited
    ? [(group.flags ?? []).includes("accepted") ? "accepted" : "edited"]
    : (group.flags ?? []).filter((flag) => SEGMENT_FLAGS.has(flag));
  if (marks.length) {
    const badge = document.createElement("em");
    badge.textContent = marks.map((flag) => FLAG_LABELS[flag]).join(" · ");
    heading.append(" ", badge);
  }

  const paragraph = document.createElement("p");
  if (group.edited || !group.words?.length) {
    // A corrected passage has no per-word timing left, but it should still be
    // possible to hear it.
    paragraph.textContent = group.text;
    paragraph.className = "seekable";
    paragraph.title = "Play this passage";
    paragraph.addEventListener("click", () => seek(group.start));
  } else {
    group.words.forEach((word) => paragraph.appendChild(wordNode(word)));
  }

  const edit = document.createElement("button");
  edit.className = "inline-edit";
  edit.textContent = "Correct";
  edit.addEventListener("click", () => startEdit(row, group, index, paragraph));

  body.append(heading, paragraph, edit);
  row.append(time, body);
  return row;
}

function startEdit(row, group, index, paragraph) {
  if (row.querySelector("textarea")) return;
  const area = document.createElement("textarea");
  area.value = group.text;
  area.rows = Math.max(2, Math.ceil(group.text.length / 70));
  const save = document.createElement("button");
  save.className = "inline-edit primary";
  save.textContent = "Save correction";
  const cancel = document.createElement("button");
  cancel.className = "inline-edit";
  cancel.textContent = "Cancel";
  const controls = document.createElement("div");
  controls.className = "edit-controls";
  controls.append(save, cancel);
  paragraph.replaceWith(area);
  row.querySelector(".inline-edit").replaceWith(controls);

  cancel.addEventListener("click", () => renderResult(latest));
  save.addEventListener("click", async () => {
    save.disabled = true;
    try {
      const response = await fetch(`/api/sessions/${sessionId}/segments/${index}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: area.value }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not save the correction");
      latest.result[index] = data.segment;
      latest.review = data.review;
      renderResult(latest);
    } catch (error) {
      showError(error.message);
    }
  });
}

function renderQuality(data) {
  const quality = data.quality ?? {};
  if (!Object.keys(quality).length) return;
  const gaps = quality.gaps?.length ?? 0;
  const stats = [
    ["Mean confidence", `${Math.round((quality.mean_confidence ?? 0) * 100)}%`],
    ["Low-confidence words", `${quality.low_confidence_words ?? 0} of ${quality.total_words ?? 0}`],
    ["Re-checked passages", `${quality.refined ?? 0} · ${quality.replaced ?? 0} replaced · ${quality.disputed ?? 0} disputed`],
    ["Audio repaired", gaps ? `${quality.repaired_seconds}s across ${gaps} gap(s)` : "none needed"],
    ["Vocabulary fixes", String(quality.vocabulary_corrections?.length ?? 0)],
  ];
  qualityBar.replaceChildren(...stats.map(([label, value]) => {
    const cell = document.createElement("div");
    cell.innerHTML = `<span></span><b></b>`;
    cell.querySelector("span").textContent = label;
    cell.querySelector("b").textContent = value;
    return cell;
  }));
  qualityBar.classList.remove("hidden");
}

function renderReview(data) {
  const items = data.review ?? [];
  reviewCount.textContent = String(items.length);
  const kinds = [...new Set(items.flatMap((item) => item.kinds))].sort();
  const filters = document.createElement("div");
  filters.className = "filters";
  [["all", `All (${items.length})`], ...kinds.map((kind) => {
    const count = items.filter((item) => item.kinds.includes(kind)).length;
    const label = items.find((item) => item.kinds.includes(kind))
      .labels[items.find((item) => item.kinds.includes(kind)).kinds.indexOf(kind)];
    return [kind, `${label} (${count})`];
  })].forEach(([kind, label]) => {
    const chip = document.createElement("button");
    chip.className = `chip${reviewFilter === kind ? " active" : ""}`;
    chip.textContent = label;
    chip.addEventListener("click", () => { reviewFilter = kind; renderReview(latest); });
    filters.appendChild(chip);
  });

  const visible = items.filter((item) => reviewFilter === "all" || item.kinds.includes(reviewFilter));
  const rows = visible.map((item) => {
    const row = document.createElement("article");
    row.className = "review-item";
    const time = document.createElement("time");
    time.className = "seek";
    time.textContent = clock(item.start);
    time.addEventListener("click", () => seek(item.start));
    const body = document.createElement("div");
    const heading = document.createElement("strong");
    heading.textContent = `${item.speaker} — ${item.labels.join(" · ")}`;
    const text = document.createElement("p");
    text.textContent = item.text;
    body.append(heading, text);

    if (item.alternative) {
      const other = document.createElement("p");
      other.className = "alternative";
      other.textContent = `Second pass heard: ${item.alternative}`;
      body.appendChild(other);
      body.appendChild(action("Use this", "primary", () =>
        post(`/api/sessions/${sessionId}/segments/${item.segment}/accept`)));
    }
    if (item.segment !== null && item.segment !== undefined) {
      const jump = document.createElement("button");
      jump.className = "inline-edit";
      jump.textContent = "Open in transcript";
      jump.addEventListener("click", () => {
        selectTab("transcript");
        const target = transcript.querySelector(`[data-segment="${item.segment}"]`);
        target?.scrollIntoView({ block: "center", behavior: "smooth" });
        target?.classList.add("highlight");
        window.setTimeout(() => target?.classList.remove("highlight"), 2000);
      });
      body.appendChild(jump);
    }
    body.appendChild(action("Looks fine", "", () =>
      post(`/api/sessions/${sessionId}/review/${encodeURIComponent(item.id)}/resolve`)));
    row.append(time, body);
    return row;
  });
  reviewList.replaceChildren(filters, ...(rows.length
    ? rows
    : [emptyNote(items.length ? "No items match this filter." : "Nothing is left to review.")]));
}

function action(label, variant, run) {
  const button = document.createElement("button");
  button.className = `inline-edit ${variant}`.trim();
  button.textContent = label;
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      await run();
      await refreshComplete();
    } catch (error) {
      showError(error.message);
      button.disabled = false;
    }
  });
  return button;
}

async function post(url, body) {
  const response = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "That did not work");
  return data;
}

function emptyNote(text) {
  const note = document.createElement("p");
  note.className = "note";
  note.textContent = text;
  return note;
}

function renderSpeakers(data) {
  const speakers = data.speakers ?? [];
  if (!speakers.length) {
    speakerList.replaceChildren(emptyNote("No speakers were identified."));
    return;
  }
  const rows = speakers.map((speaker) => {
    const row = document.createElement("article");
    row.className = "speaker-item";
    const heading = document.createElement("strong");
    heading.textContent = speaker.label;
    const meta = document.createElement("span");
    const similarity = `${Math.round(speaker.similarity * 100)}% similar`;
    const described = {
      named: `named “${speaker.name}” in this meeting`,
      confident: `matched “${speaker.suggested}” (${similarity})`,
      possible: `possibly “${speaker.suggested}” (${similarity})`,
    }[speaker.certainty] ?? "no stored voice match";
    meta.textContent = `${described} · ${speaker.seconds}s of speech`;

    const input = document.createElement("input");
    input.type = "text";
    input.value = speaker.name === speaker.label ? (speaker.suggested ?? "") : speaker.name;
    input.placeholder = "Name this speaker";
    input.setAttribute("list", "known-speaker-names");

    const remember = document.createElement("label");
    remember.className = "toggle";
    remember.innerHTML = `<input type="checkbox" checked /><span>Remember this voice</span>`;

    const save = document.createElement("button");
    save.className = "inline-edit primary";
    save.textContent = "Apply";
    save.addEventListener("click", async () => {
      save.disabled = true;
      try {
        const response = await fetch(`/api/sessions/${sessionId}/speakers`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            label: speaker.label, name: input.value,
            remember: remember.querySelector("input").checked,
          }),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || "Could not rename that speaker");
        await refreshComplete();
      } catch (error) {
        showError(error.message);
      } finally {
        save.disabled = false;
      }
    });

    const controls = document.createElement("div");
    controls.className = "speaker-controls";
    controls.append(input, remember, save);
    const body = document.createElement("div");
    body.append(heading, meta, controls);
    row.append(body);
    return row;
  });
  const datalist = document.createElement("datalist");
  datalist.id = "known-speaker-names";
  (data.known_speakers ?? []).forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    datalist.appendChild(option);
  });
  speakerList.replaceChildren(datalist, ...rows);
}

function selectTab(name) {
  activeTab = name;
  const panels = { transcript, review: reviewList, speakers: speakerList };
  const buttons = { transcript: tabTranscript, review: tabReview, speakers: tabSpeakers };
  Object.entries(panels).forEach(([key, panel]) => panel.classList.toggle("hidden", key !== name));
  Object.entries(buttons).forEach(([key, button]) => {
    button.classList.toggle("active", key === name);
    button.setAttribute("aria-selected", String(key === name));
  });
}

function followPlayback() {
  const at = player.currentTime;
  let current = null;
  for (const node of wordSpans) {
    const playing = at >= Number(node.dataset.start) && at < Number(node.dataset.end);
    node.classList.toggle("playing", playing);
    if (playing) current = node;
  }
  if (current && !isVisible(current)) current.scrollIntoView({ block: "center" });
}

function isVisible(node) {
  const box = node.getBoundingClientRect();
  const frame = transcript.getBoundingClientRect();
  return box.top >= frame.top && box.bottom <= frame.bottom;
}

function renderResult(data) {
  latest = data;
  wordSpans = [];
  transcript.replaceChildren(...(data.result ?? []).map(renderSegment));
  legend.classList.remove("hidden");
  renderQuality(data);
  renderReview(data);
  renderSpeakers(data);
  tabs.classList.remove("hidden");
  selectTab(activeTab);
  if (data.audio_url) {
    if (!player.src) player.src = data.audio_url;
    player.classList.remove("hidden");
  }
  exportButton.href = data.export_url;
  transcriptButton.href = data.transcript_url;
  bundleButton.href = data.bundle_url;
  resultActions.classList.remove("hidden");
}

async function refreshComplete() {
  const data = await readStatus();
  if (data?.status === "complete") renderResult(data);
}

// --------------------------------------------------------------------------- //
// Session flow
// --------------------------------------------------------------------------- //

async function readStatus() {
  if (!sessionId) return null;
  const response = await fetch(`/api/sessions/${sessionId}`);
  if (response.status === 404) {
    localStorage.removeItem("threadmark-session");
    sessionId = null;
    return null;
  }
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Could not read session status");
  return data;
}

function renderLiveStatus(data) {
  data.chunks.filter((item) => item.status === "ready" && item.text).forEach((item) => {
    if (!renderedChunks.has(item.index)) {
      renderedChunks.add(item.index);
      appendLiveUtterance(item);
    }
  });
  renderWarnings(data.host);
  backlog.textContent = data.pending_chunks
    ? `${data.pending_chunks} segment${data.pending_chunks === 1 ? "" : "s"} in transcription queue`
    : data.accepted_chunks ? "Live transcript caught up" : "Listening for speech";
  if (recording) recordState.textContent = `Recording continuously · ${data.accepted_chunks} segments safely stored`;
}

async function pollLive() {
  window.clearTimeout(pollTimer);
  try {
    const data = await readStatus();
    if (data) renderLiveStatus(data);
  } catch (error) {
    backlog.textContent = "Server reconnecting… audio capture continues";
  }
  if (recording) pollTimer = window.setTimeout(pollLive, 1200);
}

function watchMicrophone() {
  stream?.getAudioTracks().forEach((track) => {
    track.addEventListener("ended", () => {
      if (recording) showError("The microphone was disconnected. Stop and finalize to keep what was recorded.");
    });
    track.addEventListener("mute", () => {
      if (recording) recordState.textContent = "Microphone is muted — audio is silent right now";
    });
    track.addEventListener("unmute", () => {
      if (recording) recordState.classList.remove("error");
    });
  });
}

async function startRecording() {
  if (recording) return;
  recordState.classList.remove("error");
  recordState.textContent = "Requesting microphone access…";
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }, video: false,
    });
    const response = await fetch("/api/sessions", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        language: languageSelect.value, final_model: modelSelect.value,
        vocabulary: vocabularyInput.value, num_speakers: Number(speakerCount.value) || 0,
        self_correct: selfCorrect.checked,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not start a session");
    sessionId = data.session_id;
    localStorage.setItem("threadmark-session", sessionId);
    localStorage.setItem("threadmark-vocabulary", vocabularyInput.value);
    startedAt = Date.now(); chunkIndex = 0; recording = true; pcmParts = []; pcmLength = 0;
    uploadQueue = []; renderedChunks = new Set(); transcript.replaceChildren();
    finalizeStatus.classList.add("hidden"); resultActions.classList.add("hidden");
    recoverPanel.classList.add("hidden"); tabs.classList.add("hidden");
    legend.classList.add("hidden"); jumpLive.classList.add("hidden");
    qualityBar.classList.add("hidden"); player.classList.add("hidden"); player.removeAttribute("src");
    prepPanel.open = false;

    audioContext = new AudioContext({ sampleRate: 16000, latencyHint: "interactive" });
    await audioContext.audioWorklet.addModule("/static/pcm-worklet.js");
    sourceNode = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser(); analyser.fftSize = 512;
    captureNode = new AudioWorkletNode(audioContext, "pcm-capture");
    silentGain = audioContext.createGain(); silentGain.gain.value = 0;
    captureNode.port.onmessage = (event) => receiveSamples(event.data);
    sourceNode.connect(analyser); sourceNode.connect(captureNode); captureNode.connect(silentGain).connect(audioContext.destination);
    watchMicrophone();

    timer = window.setInterval(() => { timeDisplay.textContent = clock(elapsed()); }, 250);
    recordButton.classList.add("active"); recordButton.disabled = true;
    recordButton.setAttribute("aria-label", "Recording in progress");
    stopButton.disabled = false; flagButton.disabled = false;
    stickToBottom = true; quietSince = null;
    [languageSelect, modelSelect, vocabularyInput, speakerCount, selfCorrect]
      .forEach((control) => { control.disabled = true; });
    recordState.textContent = "Recording continuously · first words arrive shortly";
    footerState.textContent = "Microphone active"; setStatus("Recording", "recording"); pollLive();
  } catch (error) {
    stream?.getTracks().forEach((track) => track.stop());
    showError(error.message || "Microphone access failed");
  }
}

function releaseControls() {
  recordButton.disabled = false;
  [languageSelect, modelSelect, vocabularyInput, speakerCount, selfCorrect]
    .forEach((control) => { control.disabled = false; });
}

async function stopCapture() {
  recording = false;
  window.clearInterval(timer); window.clearTimeout(pollTimer);
  captureNode?.port.postMessage("stop");
  await new Promise((resolve) => setTimeout(resolve, 80));
  if (pcmLength) enqueueChunk(takeSamples(pcmLength));
  stream?.getTracks().forEach((track) => track.stop());
  sourceNode?.disconnect(); captureNode?.disconnect(); silentGain?.disconnect();
  await audioContext?.close(); analyser = null;
  await waitForUploads();
}

async function beginFinalization() {
  finalizeStatus.classList.remove("hidden");
  const response = await fetch(`/api/sessions/${sessionId}/finish`, { method: "POST" });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Could not start finalization");
  pollFinalization();
}

async function stopRecording() {
  if (!recording) return;
  recordButton.classList.remove("active");
  stopButton.disabled = true; flagButton.disabled = true;
  signalWarning.classList.add("hidden");
  recordState.textContent = "Saving the final audio segment…"; setStatus("Finishing");
  try {
    await stopCapture();
    footerState.textContent = "Microphone off";
    recordState.textContent = "Recording safely stored · finalizing locally";
    await beginFinalization();
  } catch (error) {
    showError(error.message);
  }
}

async function pollFinalization() {
  try {
    const data = await readStatus();
    if (!data) throw new Error("Recording session was not found");
    finalizeStage.textContent = data.stage;
    renderWarnings(data.host);
    backlog.textContent = data.pending_chunks ? `${data.pending_chunks} live segments remaining` : "All audio safely stored";
    setStatus(data.status === "complete" ? "Ready" : "Finalizing");
    if (data.status === "complete") {
      renderResult(data);
      finalizeStatus.classList.add("hidden");
      recordState.textContent = data.review?.length
        ? `Ready · ${data.review.length} passage${data.review.length === 1 ? "" : "s"} flagged for review`
        : "Final transcript and speaker labels are ready";
      releaseControls();
      footerState.textContent = "Final transcript ready";
      localStorage.removeItem("threadmark-session");
      return;
    }
    if (data.status === "error") throw new Error(data.error || "Finalization failed");
    window.setTimeout(pollFinalization, 2500);
  } catch (error) {
    finalizeStatus.classList.add("hidden");
    releaseControls();
    showError(error.message);
  }
}

async function restoreSession() {
  vocabularyInput.value = localStorage.getItem("threadmark-vocabulary") ?? "";
  try {
    const response = await fetch("/api/speakers");
    const data = await response.json();
    if (data.names?.length) {
      knownSpeakers.textContent = `Known voices: ${data.names.join(", ")}`;
    }
  } catch (error) {
    knownSpeakers.textContent = "";
  }
  try {
    const health = await (await fetch("/api/health")).json();
    renderWarnings(health);
  } catch (error) { /* the recorder still works without a health report */ }

  if (!sessionId) return;
  try {
    const data = await readStatus();
    if (!data) return;
    if (["queued", "finalizing"].includes(data.status)) {
      finalizeStatus.classList.remove("hidden");
      recordButton.disabled = true;
      [languageSelect, modelSelect, vocabularyInput, speakerCount, selfCorrect]
        .forEach((control) => { control.disabled = true; });
      pollFinalization();
      return;
    }
    if (data.status === "complete") {
      renderResult(data);
      setStatus("Ready");
      localStorage.removeItem("threadmark-session");
      return;
    }
    if (["recording", "interrupted", "error"].includes(data.status) && data.accepted_chunks) {
      renderLiveStatus(data); recoverPanel.classList.remove("hidden"); setStatus("Recovered");
      recordState.textContent = "A safely stored recording was recovered";
    }
  } catch (error) {
    showError(error.message);
  }
}

transcript.addEventListener("scroll", () => {
  const atBottom = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 40;
  stickToBottom = atBottom;
  if (atBottom) jumpLive.classList.add("hidden");
});
jumpLive.addEventListener("click", () => {
  stickToBottom = true;
  jumpLive.classList.add("hidden");
  transcript.scrollTop = transcript.scrollHeight;
});
player.addEventListener("timeupdate", followPlayback);
player.addEventListener("pause", () => wordSpans.forEach((node) => node.classList.remove("playing")));
flagButton.addEventListener("click", async () => {
  const at = elapsed();
  flagButton.disabled = true;
  try {
    await post(`/api/sessions/${sessionId}/flags`, { offset: at });
    flagButton.lastChild.textContent = ` Flagged ${clock(at)}`;
    window.setTimeout(() => { flagButton.lastChild.textContent = " Flag moment"; }, 2500);
  } catch (error) {
    showError(error.message);
  } finally {
    flagButton.disabled = !recording;
  }
});
recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);
tabTranscript.addEventListener("click", () => selectTab("transcript"));
tabReview.addEventListener("click", () => selectTab("review"));
tabSpeakers.addEventListener("click", () => selectTab("speakers"));
recoverButton.addEventListener("click", async () => {
  recoverPanel.classList.add("hidden"); recordButton.disabled = true;
  try { await beginFinalization(); } catch (error) { showError(error.message); }
});
window.addEventListener("beforeunload", (event) => {
  if (recording) { event.preventDefault(); event.returnValue = ""; }
});
drawWaveform();
restoreSession();
