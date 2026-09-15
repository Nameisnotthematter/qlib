const state = { actions: [], selected: null, runId: null, pollTimer: null };

const grid = document.querySelector("#workflow-grid");
const template = document.querySelector("#workflow-template");
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

function renderActions(recommended = []) {
  grid.replaceChildren();
  for (const action of state.actions) {
    const card = template.content.firstElementChild.cloneNode(true);
    card.dataset.actionId = action.id;
    if (recommended.includes(action.id)) card.classList.add("recommended");
    card.querySelector(".category").textContent = action.category;
    card.querySelector(".duration").textContent = action.duration;
    card.querySelector("h3").textContent = action.title;
    card.querySelector(".description").textContent = action.description;
    card.querySelector(".outcome").textContent = action.outcome;
    card.querySelector(".card-action").addEventListener("click", () => openAction(action));
    grid.append(card);
  }
}

async function openAction(action) {
  let preview;
  try {
    preview = await request(`/api/actions/${action.id}/preview`, { method: "POST", body: "{}" });
  } catch (error) {
    const answer = document.querySelector("#assistant-answer");
    answer.hidden = false;
    answer.textContent = error.message;
    return;
  }
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
  confirmButton.hidden = false;
  confirmButton.disabled = false;
  dialog.showModal();
}

async function startRun() {
  confirmButton.disabled = true;
  confirmButton.textContent = "正在启动…";
  try {
    const run = await request("/api/runs", {
      method: "POST",
      body: JSON.stringify({
        action_id: state.selected.id,
        confirmation_token: state.selected.confirmationToken,
      }),
    });
    state.runId = run.id;
    runState.hidden = false;
    confirmButton.hidden = true;
    updateRun(run);
    state.pollTimer = setInterval(pollRun, 800);
  } catch (error) {
    confirmButton.disabled = false;
    confirmButton.textContent = "确认并运行";
    showInlineError(error.message);
  }
}

async function pollRun() {
  if (!state.runId) return;
  try {
    const run = await request(`/api/runs/${state.runId}`);
    updateRun(run);
    if (["completed", "failed"].includes(run.status)) {
      clearInterval(state.pollTimer);
    }
  } catch (error) {
    clearInterval(state.pollTimer);
    showInlineError(error.message);
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

function showInlineError(message) {
  runState.hidden = false;
  statusPill.textContent = "发生错误";
  statusPill.className = "status-pill failed";
  runOutput.textContent = message;
}

async function ask(query) {
  const answer = document.querySelector("#assistant-answer");
  answer.hidden = false;
  answer.textContent = "正在查找合适的工作流…";
  try {
    const result = await request("/api/recommend", {
      method: "POST",
      body: JSON.stringify({ query }),
    });
    answer.textContent = result.answer;
    renderActions(result.action_ids);
    grid.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    answer.textContent = error.message;
  }
}

document.querySelector("#ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  ask(document.querySelector("#query").value);
});

document.querySelectorAll("[data-query]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelector("#query").value = button.dataset.query;
    ask(button.dataset.query);
  });
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
    showInlineError(error.message);
  }
});

request("/api/catalog")
  .then(({ actions }) => {
    state.actions = actions;
    document.querySelector("#catalog-count").textContent = `${actions.length} 个工作流`;
    renderActions();
  })
  .catch((error) => {
    grid.textContent = `无法加载工作流：${error.message}`;
  });
