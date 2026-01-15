/* =====================================
   Agri-Connect AI — Frontend App Logic
   ===================================== */

// Get backend IP from localStorage or default to localhost on port 5000
const API_BASE_URL = backend
  ? backend
  : location.hostname === "localhost"
  ? "http://127.0.0.1:5000"
  : "https://onie-unchristened-ernesto.ngrok-free.dev";

/* ---------- DOM Elements ---------- */
const chatBox = document.getElementById("chat");
const inputEl = document.getElementById("question");
const metaEl = document.getElementById("meta");
const askBtn =
  document.getElementById("askBtn") ||
  document.querySelector("button");

/* ---------- State ---------- */
let isRequestInFlight = false;

/* ---------- Utilities ---------- */
function scrollToBottom() {
  chatBox.scrollTop = chatBox.scrollHeight;
}

function clearEmptyState() {
  const empty = document.getElementById("empty");
  if (empty) empty.remove();
}

function createMessage(content, type, extraClass = "") {
  clearEmptyState();

  const msg = document.createElement("div");
  msg.className = `message ${type} ${extraClass}`.trim();
  msg.innerHTML = content;
  chatBox.appendChild(msg);
  scrollToBottom();
  return msg;
}

/* ---------- API Call ---------- */
async function sendQuestion(question) {
  const res = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question })
  });

  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`);
  }

  return res.json();
}

/* ---------- Main Handler ---------- */
async function askQuestion() {
  const question = inputEl.value.trim();
  if (!question || isRequestInFlight) return;

  isRequestInFlight = true;
  if (askBtn) askBtn.disabled = true;

  // User message
  createMessage(question, "user");
  inputEl.value = "";
  metaEl.textContent = "";

  // Bot loading message
  const loadingMsg = createMessage("⏳ Thinking…", "bot", "loading");

  try {
    const data = await sendQuestion(question);

    // ---------- No Answer ----------
    if (data.status === "no_answer") {
      loadingMsg.classList.remove("loading");
      loadingMsg.classList.add("warning");
      loadingMsg.innerHTML = `⚠️ ${data.message}`;

      if (data.suggestion) {
        metaEl.textContent = data.suggestion;
      }
      return;
    }

    // ---------- Normal Answer ----------
    loadingMsg.classList.remove("loading");

    const answerHTML = marked.parse(data.answer || "");
    loadingMsg.innerHTML = answerHTML;

    // Optional typing effect (if chat.js is loaded)
    if (window.ChatUI?.typeMessage) {
      window.ChatUI.typeMessage(loadingMsg, answerHTML);
    }

    // ---------- Meta Info ----------
    const confidence =
      typeof data.confidence === "number"
        ? data.confidence
        : null;

    const source = data.diagnostics?.fallback
      ? "LLM Fallback"
      : "RAG";

    const category = data.diagnostics?.category || "general";

    metaEl.innerHTML = `
      <strong>Source:</strong> ${source} |
      <strong>Category:</strong> ${category}
    `;

    // Visual confidence bar (if effects.js is loaded)
    if (confidence !== null && window.Effects?.confidenceBar) {
      window.Effects.confidenceBar(metaEl, confidence);
    }

  } catch (err) {
    loadingMsg.classList.remove("loading");
    loadingMsg.classList.add("warning");
    loadingMsg.innerHTML = "❌ Unable to connect to backend.";
    console.error(err);
  } finally {
    isRequestInFlight = false;
    if (askBtn) askBtn.disabled = false;
  }
}

/* ---------- Event Listeners ---------- */
if (askBtn) {
  askBtn.addEventListener("click", askQuestion);
}

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    askQuestion();
  }
});

/* ---------- Optional Enhancements ---------- */
window.addEventListener("load", () => {
  // Button press feedback (effects.js)
  if (window.Effects?.pressEffect && askBtn) {
    window.Effects.pressEffect(askBtn);
  }

  // Suggested questions (chat.js)
  if (window.ChatUI?.suggestions) {
    window.ChatUI.suggestions(chatBox, (text) => {
      inputEl.value = text;
      inputEl.focus();
    });
  }
});
