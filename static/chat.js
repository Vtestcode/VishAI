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
  const streamEndpoint = root.dataset.streamEndpoint || "/chat/stream";
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

  function updateMessage(message, content, options) {
    if (options && options.html) {
      message.innerHTML = content;
    } else {
      message.innerHTML = formatText(content);
    }
    messagesEl.scrollTop = messagesEl.scrollHeight;
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
    const botMessage = appendMessage("bot", '<span class="typing">Searching the knowledge base</span>', { html: true });

    try {
      await sendStreamingMessage(text, botMessage);
    } catch (error) {
      try {
        await sendJsonMessage(text, botMessage);
      } catch (fallbackError) {
        appendMessage("bot", `Error: ${fallbackError.message}`);
        botMessage.remove();
        statusEl.textContent = "Connection issue";
      }
    } finally {
      setLoading(false);
      inputEl.focus();
    }
  }

  async function sendStreamingMessage(text, botMessage) {
    const response = await fetch(streamEndpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        session_id: localStorage.getItem(sessionKey),
      }),
    });

    if (!response.ok || !response.body) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `Server error ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffered = "";
    let answer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffered += decoder.decode(value, { stream: true });
      const lines = buffered.split("\n");
      buffered = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "session" && event.session_id) {
          localStorage.setItem(sessionKey, event.session_id);
        } else if (event.type === "status") {
          statusEl.textContent = event.message || "Working...";
          if (!answer) {
            updateMessage(botMessage, `<span class="typing">${escapeHtml(event.message || "Working")}</span>`, { html: true });
          }
        } else if (event.type === "token") {
          answer += event.text || "";
          updateMessage(botMessage, answer || " ");
          statusEl.textContent = "Answering...";
        } else if (event.type === "replace") {
          answer = event.text || "";
          updateMessage(botMessage, answer || "No answer returned.");
        } else if (event.type === "done") {
          if (event.session_id) {
            localStorage.setItem(sessionKey, event.session_id);
          }
          answer = event.answer || answer;
          updateMessage(botMessage, answer || "No answer returned.");
          statusEl.textContent = "Answer delivered";
        } else if (event.type === "error") {
          throw new Error(event.message || "Streaming failed");
        }
      }
    }

    if (!answer) {
      throw new Error("No answer returned.");
    }
  }

  async function sendJsonMessage(text, botMessage) {
    const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          session_id: localStorage.getItem(sessionKey),
        }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `Server error ${response.status}`);
    }

    const data = await response.json();
    if (data.session_id) {
      localStorage.setItem(sessionKey, data.session_id);
    }
    updateMessage(botMessage, data.answer || "No answer returned.");
    statusEl.textContent = "Answer delivered";
  }

  function escapeHtml(value) {
    const div = document.createElement("div");
    div.textContent = value;
    return div.innerHTML;
  }

  root.querySelector("[data-send]").addEventListener("click", sendMessage);
  autoResize();
  updateMeta();
  root.classList.add("is-ready");
})();
