/* =========================================================
   audioRecorder.js
   ---------------------------------------------------------
   Browser-side audio recording utility for Agri-Connect AI

   RESPONSIBILITIES:
   - Capture microphone audio
   - Produce WAV Blob (16kHz mono)
   - Provide live waveform data
   - Press-and-hold friendly
   - No UI logic
   - No API calls

   DESIGN:
   - Deterministic
   - Fail-fast
   - Backend-compatible
   ========================================================= */

(function () {
  "use strict";

  const TARGET_SAMPLE_RATE = 16000;

  let mediaRecorder = null;
  let audioChunks = [];
  let mediaStream = null;

  // 🔥 Web Audio nodes (for waveform)
  let audioContext = null;
  let analyser = null;
  let sourceNode = null;
  let rafId = null;

  /* =======================================================
     PUBLIC API
     ======================================================= */

  async function startRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") return;

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        }
      });

      // 🔥 SINGLE AudioContext + Analyser
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;

      sourceNode = audioContext.createMediaStreamSource(mediaStream);
      sourceNode.connect(analyser);

      // expose analyser for voiceChat.js
      window._voiceAnalyser = analyser;

      const options = getSupportedRecorderOptions();
      mediaRecorder = new MediaRecorder(mediaStream, options);

      audioChunks = [];

      mediaRecorder.ondataavailable = e => {
        if (e.data?.size) audioChunks.push(e.data);
      };

      mediaRecorder.start();

    } catch (err) {
      cleanup();
      throw err;
    }
  }


  async function stopRecording() {
    return new Promise((resolve, reject) => {
      if (!mediaRecorder || mediaRecorder.state !== "recording") {
        reject(new Error("Recorder is not active"));
        return;
      }

      mediaRecorder.onstop = async () => {
        try {
          stopWaveformMonitoring();
          const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
          const wavBlob = await convertToWav(blob);
          cleanup();
          resolve(wavBlob);
        } catch (err) {
          cleanup();
          reject(err);
        }
      };

      mediaRecorder.stop();
    });
  }

  /* =======================================================
     WAVEFORM (REAL-TIME)
     ======================================================= */

  function setupWaveformMonitoring(stream) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;

    sourceNode = audioContext.createMediaStreamSource(stream);
    sourceNode.connect(analyser);

    startWaveformLoop();
  }

  function startWaveformLoop() {
    const buffer = new Uint8Array(analyser.frequencyBinCount);

    function tick() {
      analyser.getByteTimeDomainData(buffer);

      // Normalize amplitude (0–1)
      let sum = 0;
      for (let i = 0; i < buffer.length; i++) {
        const v = (buffer[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / buffer.length);

      // 🔥 Emit waveform data
      if (typeof window.AudioRecorder.onWaveform === "function") {
        window.AudioRecorder.onWaveform(rms);
      }

      rafId = requestAnimationFrame(tick);
    }

    tick();
  }

  function stopWaveformMonitoring() {
    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }

    if (sourceNode) {
      sourceNode.disconnect();
      sourceNode = null;
    }

    if (analyser) {
      analyser.disconnect();
      analyser = null;
    }

    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }
  }

  /* =======================================================
     INTERNAL HELPERS
     ======================================================= */

  function cleanup() {
    audioChunks = [];

    stopWaveformMonitoring();

    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
      mediaStream = null;
    }

    mediaRecorder = null;
  }

  function getSupportedRecorderOptions() {
    const candidates = [
      { mimeType: "audio/webm;codecs=opus" },
      { mimeType: "audio/webm" },
      { mimeType: "audio/ogg;codecs=opus" },
      { mimeType: "audio/ogg" },
    ];

    for (const opt of candidates) {
      if (window.MediaRecorder.isTypeSupported(opt.mimeType)) {
        return opt;
      }
    }
    return {};
  }

  async function convertToWav(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    const channelData = audioBuffer.getChannelData(0);
    const sampleRate = audioBuffer.sampleRate;

    const wavBuffer = encodeWAV(
      channelData,
      sampleRate,
      TARGET_SAMPLE_RATE
    );

    return new Blob([wavBuffer], { type: "audio/wav" });
  }

  function encodeWAV(samples, srcSampleRate, targetSampleRate) {
    const resampled = resample(samples, srcSampleRate, targetSampleRate);
    const buffer = new ArrayBuffer(44 + resampled.length * 2);
    const view = new DataView(buffer);

    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + resampled.length * 2, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, targetSampleRate, true);
    view.setUint32(28, targetSampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, "data");
    view.setUint32(40, resampled.length * 2, true);

    floatTo16BitPCM(view, 44, resampled);
    return buffer;
  }

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  function floatTo16BitPCM(view, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, input[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
  }

  function resample(data, srcRate, targetRate) {
    if (srcRate === targetRate) return data;

    const ratio = srcRate / targetRate;
    const newLength = Math.round(data.length / ratio);
    const result = new Float32Array(newLength);

    let offsetResult = 0;
    let offsetBuffer = 0;

    while (offsetResult < result.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
      let accum = 0, count = 0;

      for (let i = offsetBuffer; i < nextOffsetBuffer && i < data.length; i++) {
        accum += data[i];
        count++;
      }

      result[offsetResult] = accum / count;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }

    return result;
  }

  /* =======================================================
     EXPORT
     ======================================================= */

  window.AudioRecorder = {
    start: startRecording,
    stop: stopRecording,

    // 🔥 waveform hook (UI subscribes)
    onWaveform: null
  };

})();
