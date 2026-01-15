/* =====================================
   Agri-Connect AI — Visual Effects
   ===================================== */

(function () {

  /* ---------- Fade In Utility ---------- */
  function fadeIn(element, duration = 200) {
    element.style.opacity = 0;
    element.style.transition = `opacity ${duration}ms ease`;
    requestAnimationFrame(() => {
      element.style.opacity = 1;
    });
  }

  /* ---------- Confidence Bar ---------- */
  function renderConfidenceBar(container, confidence) {
    if (typeof confidence !== "number") return;

    const wrapper = document.createElement("div");
    wrapper.style.marginTop = "8px";

    const label = document.createElement("div");
    label.textContent = `Confidence: ${(confidence * 100).toFixed(0)}%`;
    label.style.fontSize = "12px";
    label.style.marginBottom = "4px";
    label.style.color = "var(--text-muted)";

    const barBg = document.createElement("div");
    barBg.style.height = "8px";
    barBg.style.borderRadius = "999px";
    barBg.style.background = "#e5e7eb";
    barBg.style.overflow = "hidden";

    const bar = document.createElement("div");
    bar.style.height = "100%";
    bar.style.width = "0%";
    bar.style.borderRadius = "999px";
    bar.style.background =
      confidence >= 0.6
        ? "linear-gradient(90deg, #2e7d32, #66bb6a)"
        : "linear-gradient(90deg, #f59e0b, #fbbf24)";
    bar.style.transition = "width 600ms ease";

    barBg.appendChild(bar);
    wrapper.appendChild(label);
    wrapper.appendChild(barBg);
    container.appendChild(wrapper);

    requestAnimationFrame(() => {
      bar.style.width = `${Math.min(confidence * 100, 100)}%`;
    });
  }

  /* ---------- Button Click Feedback ---------- */
  function pressEffect(button) {
    if (!button) return;
    button.addEventListener("click", () => {
      button.style.transform = "scale(0.96)";
      setTimeout(() => {
        button.style.transform = "";
      }, 120);
    });
  }

  /* ---------- Public API ---------- */
  window.Effects = {
    fadeIn,
    confidenceBar: renderConfidenceBar,
    pressEffect
  };

})();
