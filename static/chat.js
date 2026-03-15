(function () {
  const root = document.querySelector("[data-chat-root]");
  if (!root) return;

  const messagesEl = root.querySelector("[data-messages]");
  const inputEl = root.querySelector("[data-input]");
  const sendBtn = root.querySelector("[data-send]");
  const statusEl = root.querySelector("[data-status]");
  const charCountEl = root.querySelector("[data-char-count]");
  const prompts = root.querySelectorAll("[data-prompt]");
  const endpoint = root.dataset.endpoint || "/chat";
  const sessionKey = root.dataset.sessionKey || "vishal_portfolio_chat_session";

  inputEl.addEventListener("input", () => {
    autoResize();
    updateMeta();
  });

  inputEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !sendBtn.disabled) {
      event.preventDefault();
      sendMessage();
    }
  });

  prompts.forEach((button) => {
    button.addEventListener("click", () => {
      inputEl.value = button.dataset.prompt || "";
      autoResize();
      updateMeta();
      inputEl.focus();
    });
  });

  function autoResize() {
    inputEl.style.height = "auto";
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, 180)}px`;
  }

  function updateMeta() {
    charCountEl.textContent = `${inputEl.value.trim().length} chars`;
  }

  function setLoading(isLoading) {
    sendBtn.disabled = isLoading;
    inputEl.disabled = isLoading;
    sendBtn.textContent = isLoading ? "Sending..." : "Send message";
    statusEl.textContent = isLoading ? "Thinking through the portfolio..." : "Ready for the next question";
  }

  function appendMessage(role, content, options) {
    const message = document.createElement("div");
    message.className = `message ${role}`;
    if (options && options.html) {
      message.innerHTML = content;
    } else {
      message.innerHTML = formatText(content);
    }
    messagesEl.appendChild(message);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return message;
  }

  function formatText(text) {
    return escapeHtml(text)
      .split(/\n{2,}/)
      .map((paragraph) => `<p>${paragraph.replace(/\n/g, "<br />")}</p>`)
      .join("");
  }

  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text) return;

    appendMessage("user", text);
    inputEl.value = "";
    autoResize();
    updateMeta();
    setLoading(true);
    const loadingMessage = appendMessage("bot", '<span class="typing">Composing answer</span>', { html: true });

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          session_id: localStorage.getItem(sessionKey),
        }),
      });

      loadingMessage.remove();

      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `Server error ${response.status}`);
      }

      const data = await response.json();
      if (data.session_id) {
        localStorage.setItem(sessionKey, data.session_id);
      }
      appendMessage("bot", data.answer || "No answer returned.");
      statusEl.textContent = "Answer delivered";
    } catch (error) {
      if (loadingMessage.isConnected) {
        loadingMessage.remove();
      }
      appendMessage("bot", `Error: ${error.message}`);
      statusEl.textContent = "Connection issue";
    } finally {
      setLoading(false);
      inputEl.focus();
    }
  }

  function escapeHtml(value) {
    const div = document.createElement("div");
    div.textContent = value;
    return div.innerHTML;
  }

  root.querySelector("[data-send]").addEventListener("click", sendMessage);
  autoResize();
  updateMeta();
})();
