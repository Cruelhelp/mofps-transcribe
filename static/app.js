const $ = (id) => document.getElementById(id);
const transcript = $("transcript");
const statusText = $("status");
const spinner = $("spinner");
let socket;
let audioContext;
let mediaStream;
let analyser;
let animationFrame;
let liveRunning = false;
let worklet;
let lastMeterUpdate = 0;
let stopResolver;
let serverReady = false;

function setStatus(text, busy = false) {
  statusText.textContent = text;
  spinner.hidden = !busy;
}

document.querySelectorAll(".mode").forEach((button) => {
  button.addEventListener("click", async () => {
    if (button.dataset.mode === "upload" && liveRunning) await stopLive();
    document.querySelectorAll(".mode").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    $("upload-panel").hidden = button.dataset.mode !== "upload";
    $("live-panel").hidden = button.dataset.mode !== "live";
  });
});

$("audio-file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  $("upload-transcribe").disabled = !file;
  if (file) {
    $("audio-preview").src = URL.createObjectURL(file);
    $("audio-preview").hidden = false;
  }
});

$("upload-transcribe").addEventListener("click", async () => {
  const file = $("audio-file").files[0];
  if (!file) return;
  const data = new FormData();
  data.append("file", file);
  data.append("quality", $("upload-quality").value);
  data.append("glossary", $("glossary").value);
  setStatus("Uploading and transcribing.", true);
  $("upload-transcribe").disabled = true;
  try {
    const response = await fetch("/api/transcribe", { method: "POST", body: data });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Transcription failed");
    transcript.value = result.text;
    setStatus("Transcription complete");
  } catch (error) {
    setStatus(error.message);
  } finally {
    $("upload-transcribe").disabled = false;
    spinner.hidden = true;
  }
});

function estimatePitch(data, rate) {
  let rms = 0;
  for (const value of data) rms += value * value;
  rms = Math.sqrt(rms / data.length);
  $("level-fill").style.width = `${Math.min(100, rms * 450)}%`;
  if (rms < 0.015) return "No clear pitch";

  let bestOffset = -1;
  let bestCorrelation = 0;
  for (let offset = Math.floor(rate / 350); offset <= Math.floor(rate / 80); offset++) {
    let correlation = 0;
    for (let i = 0; i < data.length - offset; i++) correlation += data[i] * data[i + offset];
    if (correlation > bestCorrelation) {
      bestCorrelation = correlation;
      bestOffset = offset;
    }
  }
  return bestOffset > 0 ? `${Math.round(rate / bestOffset)} Hz` : "No clear pitch";
}

function updateMeter() {
  if (!analyser || !liveRunning) return;
  const now = performance.now();
  if (now - lastMeterUpdate > 125) {
    const data = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(data);
    $("pitch-value").textContent = estimatePitch(data, audioContext.sampleRate);
    lastMeterUpdate = now;
  }
  animationFrame = requestAnimationFrame(updateMeter);
}

async function startLive() {
  $("live-toggle").disabled = true;
  setStatus("Requesting microphone access...", true);
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  audioContext = new AudioContext();
  await audioContext.resume();
  await audioContext.audioWorklet.addModule("/static/pcm-worklet.js");
  const source = audioContext.createMediaStreamSource(mediaStream);
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;
  source.connect(analyser);

  worklet = new AudioWorkletNode(audioContext, "pcm-processor");
  source.connect(worklet);
  const silentOutput = audioContext.createGain();
  silentOutput.gain.value = 0;
  worklet.connect(silentOutput);
  silentOutput.connect(audioContext.destination);
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${scheme}://${location.host}/ws/live`);
  socket.binaryType = "arraybuffer";
  const openTimeout = setTimeout(() => {
    if (socket.readyState !== WebSocket.OPEN) socket.close();
  }, 10000);
  socket.onopen = () => {
    clearTimeout(openTimeout);
    socket.send(JSON.stringify({
      quality: $("live-quality").value,
      chunk_seconds: Number($("chunk-seconds").value),
      glossary: $("glossary").value,
    }));
  };
  socket.onmessage = (event) => {
    let message;
    try {
      message = JSON.parse(event.data);
    } catch {
      return;
    }
    if (message.type === "transcript") transcript.value = message.text;
    if (message.type === "ready") serverReady = true;
    if (message.type === "ready_to_stop" && stopResolver) stopResolver();
    const busy = message.type === "status" && (
      message.message === "Transcribing..." || message.message.startsWith("Loading ")
    );
    setStatus(message.message || "Listening...", busy);
  };
  socket.onerror = () => setStatus("Live connection failed. Please restart live mode.");
  socket.onclose = () => liveRunning && setStatus("Live connection closed. Please restart live mode.");
  worklet.port.onmessage = (event) => {
    if (serverReady && socket.readyState === WebSocket.OPEN && socket.bufferedAmount < 512 * 1024) {
      socket.send(event.data);
    }
  };
  liveRunning = true;
  $("live-toggle").disabled = false;
  $("live-toggle").textContent = "Stop live transcription";
  updateMeter();
}

async function stopLive() {
  if (!liveRunning && !mediaStream && !audioContext) return;
  liveRunning = false;
  cancelAnimationFrame(animationFrame);
  if (worklet) worklet.port.postMessage("flush");
  if (socket && socket.readyState === WebSocket.OPEN) {
    await new Promise((resolve) => setTimeout(resolve, 120));
    const finished = new Promise((resolve) => {
      stopResolver = resolve;
      setTimeout(resolve, 12000);
    });
    socket.send(JSON.stringify({ type: "stop" }));
    setStatus("Finishing the last audio...", true);
    await finished;
  }
  if (socket) socket.close();
  if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
  if (audioContext) await audioContext.close();
  socket = null;
  mediaStream = null;
  audioContext = null;
  analyser = null;
  worklet = null;
  stopResolver = null;
  serverReady = false;
  $("level-fill").style.width = "0";
  $("pitch-value").textContent = "Waiting";
  $("live-toggle").textContent = "Start live transcription";
  $("live-toggle").disabled = false;
  setStatus("Live transcription stopped");
}

$("live-toggle").addEventListener("click", async () => {
  try {
    if (liveRunning) await stopLive();
    else await startLive();
  } catch (error) {
    setStatus(error.message || "Could not start microphone.");
    await stopLive();
    $("live-toggle").disabled = false;
  }
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden && liveRunning) setStatus("Live transcription is still running in the background.");
});

$("copy").addEventListener("click", async () => {
  await navigator.clipboard.writeText(transcript.value);
  setStatus("Transcript copied");
});

$("download").addEventListener("click", () => {
  const url = URL.createObjectURL(new Blob([transcript.value], { type: "text/plain" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "transcript.txt";
  link.click();
  URL.revokeObjectURL(url);
});

$("clear").addEventListener("click", () => {
  transcript.value = "";
  setStatus("Transcript cleared");
});

$("refine").addEventListener("click", async () => {
  if (!transcript.value.trim()) return;
  $("refine").disabled = true;
  setStatus("Polishing transcript...", true);
  try {
    const response = await fetch("/api/refine", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: transcript.value, glossary: $("glossary").value }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Could not polish transcript");
    transcript.value = result.text;
    setStatus("Transcript polished");
  } catch (error) {
    setStatus(error.message);
  } finally {
    $("refine").disabled = false;
    spinner.hidden = true;
  }
});





