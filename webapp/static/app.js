mermaid.initialize({ startOnLoad: false, theme: "dark" });

const thread = document.getElementById("thread");
const form = document.getElementById("composer-form");
const messageEl = document.getElementById("message");
const imageInput = document.getElementById("image-input");
const attachBtn = document.getElementById("attach-btn");
const attachmentRow = document.getElementById("attachment-row");
const attachmentName = document.getElementById("attachment-name");
const removeAttachmentBtn = document.getElementById("remove-attachment");
const sendBtn = document.getElementById("send-btn");

let pendingImage = null;
let renderCounter = 0;

attachBtn.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;
  pendingImage = file;
  attachmentName.textContent = `📎 ${file.name}`;
  attachmentRow.classList.remove("hidden");
});

removeAttachmentBtn.addEventListener("click", () => {
  pendingImage = null;
  imageInput.value = "";
  attachmentRow.classList.add("hidden");
});

function scrollToBottom() {
  thread.scrollTop = thread.scrollHeight;
}

function addUserMessage(text, imageFile) {
  const wrap = document.createElement("div");
  wrap.className = "msg user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text || "(image only)";
  if (imageFile) {
    const img = document.createElement("img");
    img.className = "attachment-preview";
    img.src = URL.createObjectURL(imageFile);
    bubble.appendChild(img);
  }
  wrap.appendChild(bubble);
  thread.appendChild(wrap);
  scrollToBottom();
}

function addAssistantPlaceholder() {
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  const bubble = document.createElement("div");
  bubble.className = "bubble thinking";
  bubble.textContent = "Thinking…";
  wrap.appendChild(bubble);
  thread.appendChild(wrap);
  scrollToBottom();
  return bubble;
}

function badge(text) {
  const span = document.createElement("span");
  span.className = "badge";
  span.textContent = text;
  return span;
}

async function renderMermaidInto(container, code) {
  const id = `flowmind-diagram-${renderCounter++}`;
  try {
    const { svg } = await mermaid.render(id, code);
    container.innerHTML = svg;
  } catch (err) {
    container.textContent = `(couldn't render the diagram: ${err.message || err})`;
  }
}

async function fillAssistantResult(bubble, data) {
  bubble.classList.remove("thinking");
  bubble.textContent = "";

  const metaRow = document.createElement("div");
  metaRow.className = "meta-row";
  metaRow.appendChild(badge(`intent: ${data.intent}`));
  metaRow.appendChild(badge(data.branch));
  if (data.source) metaRow.appendChild(badge(`source: ${data.source}`));
  if (data.verdict) metaRow.appendChild(badge(`self-check: ${data.verdict}`));
  if (data.revisions) metaRow.appendChild(badge(`revisions: ${data.revisions}`));
  bubble.appendChild(metaRow);

  const answer = document.createElement("div");
  answer.className = "answer";
  answer.textContent = data.answer;
  bubble.appendChild(answer);

  if (data.mermaid) {
    const diagramWrap = document.createElement("div");
    diagramWrap.className = "diagram-wrap";
    bubble.appendChild(diagramWrap);
    renderMermaidInto(diagramWrap, data.mermaid);

    const details = document.createElement("details");
    details.className = "mermaid-source";
    const summary = document.createElement("summary");
    summary.textContent = "Mermaid source";
    const pre = document.createElement("pre");
    pre.textContent = data.mermaid;
    details.appendChild(summary);
    details.appendChild(pre);
    bubble.appendChild(details);
  }
  scrollToBottom();
}

function fillAssistantError(bubble, message) {
  bubble.classList.remove("thinking");
  bubble.classList.add("error");
  bubble.textContent = message;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageEl.value.trim();
  if (!message && !pendingImage) return;

  addUserMessage(message, pendingImage);
  const assistantBubble = addAssistantPlaceholder();

  const formData = new FormData();
  formData.append("message", message);
  if (pendingImage) formData.append("image", pendingImage);

  messageEl.value = "";
  pendingImage = null;
  imageInput.value = "";
  attachmentRow.classList.add("hidden");
  sendBtn.disabled = true;

  try {
    const resp = await fetch("/analyze", { method: "POST", body: formData });
    const data = await resp.json();
    if (!resp.ok || data.error) {
      fillAssistantError(assistantBubble, data.error || "Something went wrong.");
    } else {
      await fillAssistantResult(assistantBubble, data);
    }
  } catch (err) {
    fillAssistantError(assistantBubble, `Network error: ${err.message || err}`);
  } finally {
    sendBtn.disabled = false;
  }
});

messageEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
