/* =========================================================
   voiceChat.js
   ---------------------------------------------------------
   GPT-style Voice Interaction for Agri-Connect AI
   ========================================================= */

(function () {
  "use strict";

  // Get backend IP from localStorage or default to localhost
  const BACKEND_IP = localStorage.getItem("backend_ip") || "127.0.0.1";
  const API_BASE_URL = backend
    ? backend
    : location.hostname === "localhost"
    ? "http://127.0.0.1:5000"
    : "https://onie-unchristened-ernesto.ngrok-free.dev";

  let isRecording = false;
  let recordStartTime = 0;
  let waveformTimer = null;
  let processingBubble = null;

  const voiceOverlay = document.getElementById("voiceOverlay");
  const waveformEl = document.getElementById("waveform");

  /* =======================================================
     OVERLAY + WAVEFORM
     ======================================================= */

  function showOverlay() {
    if (!voiceOverlay) return;
    voiceOverlay.classList.remove("hidden");
    startWaveform();
  }

  function hideOverlay() {
    if (!voiceOverlay) return;
    voiceOverlay.classList.add("hidden");
    stopWaveform();
  }

  function startWaveform() {
    if (!waveformEl || !window._voiceAnalyser) return;

    waveformEl.innerHTML = "";
    for (let i = 0; i < 12; i++) {
      waveformEl.appendChild(document.createElement("span"));
    }

    const bars = waveformEl.querySelectorAll("span");
    const analyser = window._voiceAnalyser;
    const data = new Uint8Array(analyser.frequencyBinCount);

    function animate() {
      if (!isRecording) return;

      analyser.getByteFrequencyData(data);

      const energy = data.slice(0, 10).reduce((a, b) => a + b, 0) / 10;
      const level = energy / 255;

      bars.forEach(bar => {
        bar.style.height = `${6 + level * 22}px`;
      });

      requestAnimationFrame(animate);
    }

    animate();
  }

  function stopWaveform() {
    clearInterval(waveformTimer);
    waveformTimer = null;
  }

  /* =======================================================
     START RECORDING
     ======================================================= */

  async function startVoiceCapture() {
    if (isRecording) return;

    try {
      isRecording = true;
      recordStartTime = Date.now();

      await AudioRecorder.start();
      showOverlay();
    } catch (err) {
      isRecording = false;
      console.error(err);
      addMsg("❌ Microphone access denied.", "bot");
    }
  }

  /* =======================================================
     STOP RECORDING
     ======================================================= */

  async function stopVoiceCapture() {
    if (!isRecording) return;
    isRecording = false;

    const duration = (Date.now() - recordStartTime) / 1000;
    if (duration < 2) {
      hideOverlay();
      addMsg("❌ Hold the mic for at least 2 seconds.", "bot");
      return;
    }

    try {
      const wavBlob = await AudioRecorder.stop(); // 🔥 stop FIRST
      hideOverlay();

      processingBubble = addMsg("⏳ Processing voice…", "bot", false);
      await sendVoiceToBackend(wavBlob);

    } catch (err) {
      hideOverlay();
      console.error(err);
      addMsg("❌ Voice recording failed.", "bot");
    }
  }


  /* =======================================================
     SEND AUDIO → BACKEND
     ======================================================= */

  async function sendVoiceToBackend(wavBlob) {
    const formData = new FormData();
    formData.append("file", wavBlob, "voice.wav");

    let response;
    try {
      response = await fetch(API_VOICE_URL, {
        method: "POST",
        body: formData
      });
    } catch {
      removeProcessing();
      addMsg("❌ Voice service unreachable.", "bot");
      return;
    }

    removeProcessing();

    let data;
    try {
      data = await response.json();
    } catch {
      addMsg("❌ Invalid server response.", "bot");
      return;
    }

    if (data.status === "stt_success" && data.text) {
      showTranscriptionEditor(data.text);
      return;
    }

    addMsg(data.message || "❌ Speech not clear.", "bot");
  }

  function removeProcessing() {
    if (processingBubble) {
      processingBubble.remove();
      processingBubble = null;
    }
  }

  /* =======================================================
     TRANSCRIPTION EDITOR
     ======================================================= */

  function showTranscriptionEditor(text) {
    const wrapper = document.createElement("div");
    wrapper.className = "msg-wrapper bot";

    wrapper.innerHTML = `
      <div class="msg bot" style="border:2px dashed #43a047;background:#f6fff6;">
        <div style="font-weight:600;margin-bottom:6px;">📝 Recognized speech</div>
        <textarea style="width:100%;min-height:70px;border-radius:10px;padding:8px;">
${text}</textarea>
        <div style="margin-top:8px;display:flex;gap:8px;">
          <button class="voice-confirm">Send</button>
          <button class="voice-cancel">Cancel</button>
        </div>
      </div>
    `;

    chatEl.appendChild(wrapper);
    chatEl.scrollTop = chatEl.scrollHeight;

    wrapper.querySelector(".voice-confirm").onclick = () => {
      const t = wrapper.querySelector("textarea").value.trim();
      if (!t) {
        addMsg("❌ Empty message.", "bot");
        return;
      }
      wrapper.remove();
      input.value = t;
      send(); // 🔥 reuse EXISTING send()
    };

    wrapper.querySelector(".voice-cancel").onclick = () => wrapper.remove();
  }

  /* =======================================================
     EXPORT
     ======================================================= */

  window.VoiceChat = {
    start: startVoiceCapture,
    stop: stopVoiceCapture
  };

})();
