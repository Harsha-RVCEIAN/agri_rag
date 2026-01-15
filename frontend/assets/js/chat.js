/* =========================================================
   chat.js — Core Chat Logic (Text + Voice Compatible)
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

  // ---------- STATE ----------
  window.chats = [];
  window.activeId = null;

  const chatEl = document.getElementById("chat");
  const inputEl = document.getElementById("input");

  /* =======================================================
     INIT
     ======================================================= */

  function init() {
    createNewChat();
  }

  /* =======================================================
     CHAT MANAGEMENT
     ======================================================= */

  function createNewChat() {
    const chat = {
      id: crypto.randomUUID(),
      messages: []
    };
    chats.push(chat);
    activeId = chat.id;
    renderChat();
    save();
  }

  function getActiveChat() {
    return chats.find(c => c.id === activeId);
  }

  /* =======================================================
     MESSAGE RENDERING
     ======================================================= */

  window.addMsg = function (text, role) {
    const chat = getActiveChat();
    if (!chat) return;

    const msg = { role, text };
    chat.messages.push(msg);

    const wrapper = document.createElement("div");
    wrapper.className = `msg-wrapper ${role}`;

    const bubble = document.createElement("div");
    bubble.className = `msg ${role}`;
    bubble.textContent = text;

    wrapper.appendChild(bubble);
    chatEl.appendChild(wrapper);

    chatEl.scrollTop = chatEl.scrollHeight;
    save();
  };

  function addBotTyping() {
    const wrapper = document.createElement("div");
    wrapper.className = "msg-wrapper bot";

    const bubble = document.createElement("div");
    bubble.className = "msg bot";
    bubble.textContent = "Typing…";

    wrapper.appendChild(bubble);
    chatEl.appendChild(wrapper);
    chatEl.scrollTop = chatEl.scrollHeight;

    return bubble;
  }

  function removeLastBot() {
    const chat = getActiveChat();
    if (!chat || !chat.messages.length) return;

    const last = chat.messages[chat.messages.length - 1];
    if (last.role === "bot") {
      chat.messages.pop();
      chatEl.lastChild?.remove();
      save();
    }
  }

  /* =======================================================
     TEXT SEND (REUSED BY VOICE)
     ======================================================= */

  window.sendTextMessage = async function (text) {
    if (!text || !text.trim()) return;

    const typingBubble = addBotTyping();

    let response;
    try {
      response = await fetch(API_CHAT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text })
      });
    } catch {
      removeLastBot();
      addMsg("❌ Server unreachable.", "bot");
      return;
    }

    removeLastBot();

    let data;
    try {
      data = await response.json();
    } catch {
      addMsg("❌ Invalid server response.", "bot");
      return;
    }

    if (data.status === "answer") {
      const wrapper = document.createElement("div");
      wrapper.className = "msg-wrapper bot";

      const bubble = document.createElement("div");
      bubble.className = "msg bot";

      wrapper.appendChild(bubble);
      chatEl.appendChild(wrapper);

      ChatUI.typeMessage(bubble, data.answer);
      save();
      return;
    }

    addMsg(data.message || "No answer available.", "bot");
  };

  /* =======================================================
     INPUT HANDLER
     ======================================================= */

  window.handleSend = function () {
    const text = inputEl.value.trim();
    if (!text) return;

    inputEl.value = "";
    addMsg(text, "user");
    sendTextMessage(text);
  };

  /* =======================================================
     RENDER
     ======================================================= */

  function renderChat() {
    chatEl.innerHTML = "";
    const chat = getActiveChat();
    if (!chat) return;

    chat.messages.forEach(m => addMsg(m.text, m.role));
  }

  function save() {
    localStorage.setItem("agri_chats", JSON.stringify(chats));
  }

  init();
})();
