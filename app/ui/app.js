const conversation = document.querySelector("#conversation");
const welcome = document.querySelector("#welcome");
const form = document.querySelector("#chatForm");
const messageInput = document.querySelector("#messageInput");
const employeeInput = document.querySelector("#employeeId");
const sendButton = document.querySelector("#sendButton");
const newChatButton = document.querySelector("#newChatButton");

let sessionId = newSessionId();

function newSessionId() {
  return globalThis.crypto?.randomUUID?.() || `helpdesk-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function formatAnswer(text) {
  const safe = escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
  return safe.split(/\n{2,}/).map(block => {
    const lines = block.split("\n");
    const listType = lines.every(line => /^\s*\d+[.)]\s+/.test(line)) ? "ol"
      : lines.every(line => /^\s*[-*]\s+/.test(line)) ? "ul" : null;
    if (listType) {
      const items = lines.map(line => `<li>${line.replace(/^\s*(?:\d+[.)]|[-*])\s+/, "")}</li>`).join("");
      return `<${listType}>${items}</${listType}>`;
    }
    return `<p>${lines.join("<br>")}</p>`;
  }).join("");
}

function addUserMessage(text) {
  const row = document.createElement("div");
  row.className = "message-row user";
  const bubble = document.createElement("div");
  bubble.className = "message-content";
  bubble.textContent = text;
  row.append(bubble);
  conversation.append(row);
  scrollToLatest();
}

function addAssistantMessage({ answer, sources = [], confidence = "high", ticketId = null, loading = false, error = false }) {
  const row = document.createElement("div");
  row.className = `message-row assistant${error ? " error" : ""}`;
  const mark = document.createElement("div");
  mark.className = "assistant-mark";
  mark.setAttribute("aria-hidden", "true");
  mark.innerHTML = '<svg viewBox="0 0 24 24" fill="none"><path d="M5 7.5A2.5 2.5 0 0 1 7.5 5h9A2.5 2.5 0 0 1 19 7.5v6a2.5 2.5 0 0 1-2.5 2.5H11l-4 3v-3h-.5A2.5 2.5 0 0 1 4 13.5v-6Z" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/></svg>';

  const content = document.createElement("div");
  content.className = "message-content";

  if (loading) {
    content.innerHTML = '<span class="loading"><i class="spinner" aria-hidden="true"></i>Checking the policy documents…</span>';
    row.append(mark, content);
    conversation.append(row);
    scrollToLatest();
    return { row, content };
  }

  if (confidence === "partial" || confidence === "low") {
    const badge = document.createElement("div");
    badge.className = `answer-badge${confidence === "low" ? " low" : ""}`;
    badge.textContent = confidence === "partial" ? "Partly supported by the documents" : "Limited information found";
    content.append(badge);
  }

  const answerText = document.createElement("div");
  answerText.className = "answer-text";
  answerText.innerHTML = formatAnswer(answer);
  content.append(answerText);

  if (ticketId) {
    const ticket = document.createElement("div");
    ticket.className = "ticket-result";
    ticket.textContent = `Ticket created · ${ticketId}`;
    content.append(ticket);
  }

  if (sources.length) {
    const details = document.createElement("details");
    details.className = "sources";
    const summary = document.createElement("summary");
    summary.textContent = `Sources · ${sources.length}`;
    const list = document.createElement("ul");
    list.className = "source-list";
    for (const source of sources) {
      const item = document.createElement("li");
      item.textContent = source;
      list.append(item);
    }
    details.append(summary, list);
    content.append(details);
  }

  row.append(mark, content);
  conversation.append(row);
  scrollToLatest();
  return { row, content };
}

function scrollToLatest() {
  requestAnimationFrame(() => window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" }));
}

function resizeComposer() {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 150)}px`;
}

async function submitMessage(text = messageInput.value) {
  const message = text.trim();
  if (!message || sendButton.disabled) return;

  welcome.hidden = true;
  addUserMessage(message);
  messageInput.value = "";
  resizeComposer();
  sendButton.disabled = true;
  const pending = addAssistantMessage({ loading: true });

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        employee_id: employeeInput.value.trim() || null,
        session_id: sessionId,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      if (response.status === 503 && String(payload.detail || "").includes("GEMINI_API_KEY")) {
        throw new Error("The assistant needs a Gemini key. Add it to the project's .env file and restart the server.");
      }
      throw new Error("I couldn't complete that just now. Please try again.");
    }

    const answer = String(payload.answer || "I couldn't find this in the provided documents.")
      .replace(/\n\s*Sources:\s*[\s\S]*$/i, "")
      .replace(payload.ticket_id ? new RegExp(`\\s*Ticket created:\\s*${payload.ticket_id}\\.?`, "gi") : /$^/, "")
      .trim();
    const finalMessage = addAssistantMessage({
      answer,
      sources: payload.sources || [],
      confidence: payload.confidence_flag || "high",
      ticketId: payload.ticket_id,
    });
    pending.row.remove();
    return finalMessage;
  } catch (error) {
    pending.row.remove();
    addAssistantMessage({ answer: error.message || "Something went wrong. Please try again.", error: true });
  } finally {
    sendButton.disabled = false;
    messageInput.focus();
  }
}

form.addEventListener("submit", event => {
  event.preventDefault();
  submitMessage();
});

messageInput.addEventListener("input", resizeComposer);
messageInput.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach(button => {
  button.addEventListener("click", () => submitMessage(button.dataset.prompt));
});

newChatButton.addEventListener("click", () => {
  sessionId = newSessionId();
  conversation.replaceChildren();
  welcome.hidden = false;
  messageInput.value = "";
  resizeComposer();
  messageInput.focus();
});
