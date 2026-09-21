const state = {
  sessionId: crypto.randomUUID().replaceAll("-", ""),
  selected: null,
  runId: null,
  pollTimer: null,
  busy: false,
};

const messages = document.querySelector("#messages");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const dialog = document.querySelector("#run-dialog");
const confirmButton = document.querySelector("#confirm-run");
const runState = document.querySelector("#run-state");
const runOutput = document.querySelector("#run-output");
const statusPill = document.querySelector("#status-pill");
const cancelButton = document.querySelector("#cancel-run");

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "请求失败");
  return data;
}

function appendMessage(role, text, proposal = null) {
  const article = document.createElement("article");
  article.className = `message ${role}-message`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "assistant" ? "Q" : "你";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  bubble.append(paragraph);

  if (proposal) {
    const card = document.createElement("div");
    card.className = "proposal-card";
    const label = document.createElement("small");
    label.textContent = "需要你的确认";
    const title = document.createElement("strong");
    title.textContent = proposal.action.title;
    const description = document.createElement("span");
    description.textContent = proposal.action.description;
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "查看命令并确认";
    button.addEventListener("click", () => openProposal(proposal));
    card.append(label, title, description, button);
    bubble.append(card);
  }

  article.append(avatar, bubble);
  messages.append(article);
  messages.scrollTop = messages.scrollHeight;
  return article;
}

function appendThinking() {
  const item = appendMessage("assistant", "正在理解你的问题…");
  item.classList.add("thinking");
  return item;
}

function setBusy(value) {
  state.busy = value;
  input.disabled = value;
  sendButton.disabled = value;
  sendButton.querySelector("span:first-child").textContent = value ? "思考中" : "发送";
}

async function sendMessage(text) {
  const message = text.trim();
  if (!message || state.busy) return;
  appendMessage("user", message);
  input.value = "";
  document.querySelector("#suggestions").hidden = true;
  const thinking = appendThinking();
  setBusy(true);
  try {
    const result = await request("/api/assistant/chat", {
      method: "POST",
      body: JSON.stringify({ session_id: state.sessionId, message }),
    });
    thinking.remove();
    appendMessage("assistant", result.message, result.type === "proposal" ? result.proposal : null);
  } catch (error) {
    thinking.remove();
    appendMessage("assistant", `无法完成请求：${error.message}`);
  } finally {
    setBusy(false);
    input.focus();
  }
}

function openProposal(preview) {
  state.selected = { ...preview.action, confirmationToken: preview.confirmation_token };
  state.runId = null;
  clearInterval(state.pollTimer);
  document.querySelector("#dialog-category").textContent = state.selected.category;
  document.querySelector("#dialog-title").textContent = state.selected.title;
  document.querySelector("#dialog-description").textContent = state.selected.description;
  document.querySelector("#dialog-duration").textContent = state.selected.duration;
  document.querySelector("#dialog-command").textContent = state.selected.command;
  runState.hidden = true;
  runOutput.textContent = "";
  cancelButton.hidden = true;
  cancelButton.disabled = false;
  cancelButton.textContent = "停止任务";
  confirmButton.hidden = false;
  confirmButton.disabled = false;
  confirmButton.textContent = "确认并运行";
  dialog.showModal();
}

async function startRun() {
  confirmButton.disabled = true;
  confirmButton.textContent = "正在启动…";
  try {
    const run = await request("/api/runs", {
      method: "POST",
      body: JSON.stringify({ action_id: state.selected.id, confirmation_token: state.selected.confirmationToken }),
    });
    state.runId = run.id;
    runState.hidden = false;
    confirmButton.hidden = true;
    updateRun(run);
    state.pollTimer = setInterval(pollRun, 800);
  } catch (error) {
    confirmButton.disabled = false;
    confirmButton.textContent = "确认并运行";
    showRunError(error.message);
  }
}

async function pollRun() {
  if (!state.runId) return;
  try {
    const run = await request(`/api/runs/${state.runId}`);
    updateRun(run);
    if (["completed", "failed"].includes(run.status)) clearInterval(state.pollTimer);
  } catch (error) {
    clearInterval(state.pollTimer);
    showRunError(error.message);
  }
}

function updateRun(run) {
  const labels = { queued: "等待启动", running: "运行中", completed: "已完成", failed: "运行失败" };
  statusPill.textContent = labels[run.status] || run.status;
  statusPill.className = `status-pill ${run.status}`;
  runOutput.textContent = run.output || "任务已启动，正在等待输出…";
  runOutput.scrollTop = runOutput.scrollHeight;
  cancelButton.hidden = run.status !== "running";
}

function showRunError(message) {
  runState.hidden = false;
  statusPill.textContent = "发生错误";
  statusPill.className = "status-pill failed";
  runOutput.textContent = message;
}

document.querySelector("#chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(input.value);
});
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    document.querySelector("#chat-form").requestSubmit();
  }
});
document.querySelectorAll("[data-query]").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.dataset.query));
});
document.querySelector("#reset-chat").addEventListener("click", async () => {
  try {
    await request("/api/assistant/reset", { method: "POST", body: JSON.stringify({ session_id: state.sessionId }) });
  } catch (_error) {
    // A local reset still gives the user a clean conversation if the request fails.
  }
  state.sessionId = crypto.randomUUID().replaceAll("-", "");
  messages.replaceChildren();
  appendMessage("assistant", "新对话已开始。告诉我你想查询的数据或要完成的研究任务。");
  document.querySelector("#suggestions").hidden = false;
  input.focus();
});

confirmButton.addEventListener("click", startRun);
document.querySelector("#close-dialog").addEventListener("click", () => dialog.close());
document.querySelector("#dismiss-dialog").addEventListener("click", () => dialog.close());
dialog.addEventListener("close", () => clearInterval(state.pollTimer));
cancelButton.addEventListener("click", async () => {
  cancelButton.disabled = true;
  try {
    await request(`/api/runs/${state.runId}/cancel`, { method: "POST", body: "{}" });
    cancelButton.textContent = "正在停止…";
  } catch (error) {
    showRunError(error.message);
  }
});

request("/api/assistant/status")
  .then((status) => {
    const badge = document.querySelector("#model-badge");
    const hint = document.querySelector("#configuration-hint");
    badge.textContent = status.configured ? "GPT-5.6 Luna · 已连接" : "GPT-5.6 Luna · 待配置";
    badge.classList.toggle("ready", status.configured);
    if (!status.configured) {
      hint.hidden = false;
      hint.textContent = status.error || "请将新的 OpenRouter Key 保存到 ~/.config/qlib/openrouter.key，并设置权限为 600。";
    }
  })
  .catch(() => { document.querySelector("#model-badge").textContent = "模型状态未知"; });
