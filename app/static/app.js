// Chat page for the support assistant. Plain JavaScript, no build step.
// All text from the server is inserted with textContent (never innerHTML),
// so answers cannot inject HTML into the page.

import { icon } from "/static/icons.js";

const SUGGESTIONS = [
  { title: "Account", items: ["I forgot my password", "How do I change my email address?"] },
  { title: "Billing", items: ["Can I pay by bank transfer?", "How do I cancel my subscription?"] },
  { title: "Integrations and data", items: ["Do you have a Slack integration?", "Is my data encrypted?"] },
  {
    title: "See how it decides",
    demo: true,
    items: ["I want to stop paying", "Can I pay with PayPal?", "What is the weather in Berlin?"],
  },
];

const ROUTE = {
  answer: { label: "Answered from a help article", icon: "circle-check" },
  clarify: { label: "Needs one more detail", icon: "circle-help" },
  decline: { label: "Outside what I can help with", icon: "circle-slash" },
};

const DECIDED_BY = {
  llm: "The AI model read the articles and decided. Safety rules checked its output.",
  guardrail: "A safety rule decided before the AI model was asked.",
  rules: "Similarity score rules decided (no AI model).",
};

const SUPPORT_EMAIL = "support@test.com";
const FEEDBACK_PROMPT = {
  answer: "Was this helpful?",
  clarify: "Was this question helpful?",
  decline: "Was this the right reply?",
};

// ---- state -------------------------------------------------------------------

const state = {
  mode: "llm",
  minRelevance: 0.55,
  turns: [], // { question, previous, mode, response, rating, element }
  selected: -1,
  pendingClarify: null, // the vague question, sent along with the customer's reply
  busy: false,
  conversations: [], // saved in this browser only: { id, title, created, updated, turns, pendingClarify }
  currentId: null,
  storageOk: true,
};

// Conversations are kept in the browser (localStorage), never on the server:
// customer messages stay on the customer's device, and the API stays stateless.
const STORE_KEY = "support-assistant.conversations.v1";
const MAX_CONVERSATIONS = 50;

// ---- elements -------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const welcomeEl = $("welcome");
const inputEl = $("input");
const sendEl = $("send");
const traceEl = $("trace");
const traceBodyEl = $("trace-body");
const scrimEl = $("scrim");

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function withIcon(node, name, text, size = 16) {
  node.innerHTML = icon(name, size); // icons are our own static SVG strings
  if (text) node.append(document.createTextNode(text));
  return node;
}

// ---- setup ----------------------------------------------------------------------

function renderSuggestions() {
  const box = $("suggestions");
  for (const group of SUGGESTIONS) {
    const section = el("div", "suggestion-group" + (group.demo ? " demo" : ""));
    section.append(el("h3", null, group.title));
    const list = el("ul");
    for (const text of group.items) {
      const li = el("li");
      const button = el("button", "suggestion", text);
      button.type = "button";
      button.addEventListener("click", () => send(text));
      li.append(button);
      list.append(li);
    }
    section.append(list);
    box.append(section);
  }
}

async function checkHealth() {
  const status = $("status");
  const text = status.querySelector(".status-text");
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error();
    const data = await res.json();
    state.minRelevance = data.min_relevance ?? state.minRelevance;
    if (data.llm === "ok" || data.llm === "unknown") {
      status.className = "status ok";
      text.textContent = `${data.model} ready, ${data.cards} articles`;
    } else {
      // The assistant still works without the model: rules decide and answers show the article text.
      status.className = "status error";
      text.textContent = data.llm === "model_missing"
        ? `Model ${data.model} not downloaded: simple mode`
        : "AI model offline: simple mode";
      status.title = "Answers come straight from the help articles until the AI model is available.";
    }
  } catch {
    status.className = "status error";
    text.textContent = "Server not reachable";
  }
}

function setupTheme() {
  const button = $("theme-toggle");
  // Dark is the default (like the MML site); light is available for anyone who prefers it.
  const isDark = () => document.documentElement.dataset.theme !== "light";
  const paint = () => {
    button.innerHTML = icon(isDark() ? "sun" : "moon", 17);
    button.setAttribute("aria-label", isDark() ? "Switch to light mode" : "Switch to dark mode");
  };
  button.addEventListener("click", () => {
    const next = isDark() ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch {}
    paint();
  });
  paint();
}

function setupModeSwitch() {
  const buttons = document.querySelectorAll(".mode-switch button");
  for (const button of buttons) {
    button.addEventListener("click", () => {
      state.mode = button.dataset.mode;
      buttons.forEach((b) => b.setAttribute("aria-checked", String(b === button)));
    });
  }
}

function setupComposer() {
  sendEl.innerHTML = icon("send-horizontal", 18);
  const resize = () => {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
  };
  inputEl.addEventListener("input", resize);
  inputEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
  });
  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    const text = inputEl.value.trim();
    if (!text || state.busy) return;
    inputEl.value = "";
    resize();
    send(text, state.pendingClarify);
  });
}

function setPendingClarify(text) {
  state.pendingClarify = text;
  const box = $("reply-to");
  box.hidden = !text;
  $("reply-to-text").textContent = text || "";
}

function setupReplyTo() {
  $("reply-to-clear").innerHTML = icon("x", 16);
  $("reply-to-clear").addEventListener("click", () => {
    setPendingClarify(null);
    inputEl.focus();
  });
}

function setupTracePanel() {
  $("trace-close").innerHTML = icon("x", 18);
  $("trace-close").addEventListener("click", closeTrace);
  scrimEl.addEventListener("click", closeTrace);
  document.addEventListener("keydown", (e) => e.key === "Escape" && closeTrace());
}

function openTrace(index) {
  state.selected = index;
  state.turns.forEach((t, i) => t.element && t.element.classList.toggle("selected", i === index));
  renderTrace(state.turns[index]);
  traceEl.classList.add("open");
  traceEl.setAttribute("aria-hidden", "false");
  scrimEl.hidden = false;
  // Keep keyboard and screen reader focus inside the open panel.
  $("main").inert = true;
  $("topbar").inert = true;
  $("sidebar").inert = true;
  $("trace-close").focus();
}

function closeTrace() {
  if (!traceEl.classList.contains("open")) return;
  const turn = state.turns[state.selected];
  traceEl.classList.remove("open");
  traceEl.setAttribute("aria-hidden", "true");
  scrimEl.hidden = true;
  $("main").inert = false;
  $("topbar").inert = false;
  $("sidebar").inert = false;
  state.turns.forEach((t) => t.element && t.element.classList.remove("selected"));
  // Return focus to the button that opened the panel.
  turn?.element?.querySelector(".trace-link")?.focus();
}

function setupNewChat() {
  const button = $("new-chat");
  button.innerHTML = icon("square-pen", 16);
  button.append(el("span", "label", "New conversation"));
  button.addEventListener("click", () => {
    if (state.busy) return;
    startNewConversation();
    closeSidebarOnMobile();
    inputEl.focus();
  });
}

function startNewConversation() {
  closeTrace();
  state.currentId = null;
  state.turns = [];
  state.selected = -1;
  setPendingClarify(null);
  messagesEl.replaceChildren(welcomeEl);
  welcomeEl.hidden = false;
  inputEl.value = "";
  renderSidebar();
}

// ---- saved conversations ---------------------------------------------------------------

function loadConversations() {
  try {
    const list = JSON.parse(localStorage.getItem(STORE_KEY) || "[]");
    return Array.isArray(list) ? list.filter((c) => c && c.id && Array.isArray(c.turns)) : [];
  } catch {
    return [];
  }
}

function saveConversations() {
  try {
    // The DOM element of each turn is not saved.
    const json = JSON.stringify(state.conversations, (key, value) => (key === "element" ? undefined : value));
    localStorage.setItem(STORE_KEY, json);
    state.storageOk = true;
  } catch {
    state.storageOk = false; // private window, blocked storage, or full storage: the chat still works
  }
  renderStorageNote();
}

function currentConversation() {
  return state.conversations.find((c) => c.id === state.currentId) || null;
}

function newId() {
  return (crypto.randomUUID && crypto.randomUUID()) || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function recordTurn(turn) {
  let conversation = currentConversation();
  const now = new Date().toISOString();
  if (!conversation) {
    conversation = { id: newId(), title: turn.question.slice(0, 80), created: now, updated: now, turns: [], pendingClarify: null };
    state.conversations.unshift(conversation);
    state.currentId = conversation.id;
    state.turns = conversation.turns;
  }
  conversation.turns.push(turn);
  conversation.updated = now;
  conversation.pendingClarify = state.pendingClarify;
  // Most recent first, and keep the list a reasonable size.
  state.conversations = [conversation, ...state.conversations.filter((c) => c !== conversation)].slice(0, MAX_CONVERSATIONS);
  saveConversations();
  renderSidebar();
}

function openConversation(id) {
  if (state.busy) return;
  const conversation = state.conversations.find((c) => c.id === id);
  if (!conversation) return;
  closeTrace();
  state.currentId = id;
  state.turns = conversation.turns;
  state.selected = -1;
  messagesEl.replaceChildren(welcomeEl);
  welcomeEl.hidden = true;
  conversation.turns.forEach((turn, index) => {
    messagesEl.append(userRow(turn.question), renderBotMessage(turn, index));
  });
  setPendingClarify(conversation.pendingClarify || null);
  renderSidebar();
  closeSidebarOnMobile();
  scrollDown();
  inputEl.focus();
}

function deleteConversation(id) {
  const conversation = state.conversations.find((c) => c.id === id);
  if (!conversation || state.busy) return;
  if (!window.confirm(`Delete the conversation "${conversation.title}"? This cannot be undone.`)) return;
  state.conversations = state.conversations.filter((c) => c.id !== id);
  saveConversations();
  if (state.currentId === id) startNewConversation();
  else renderSidebar();
}

function deleteAllConversations() {
  if (!state.conversations.length || state.busy) return;
  if (!window.confirm(`Delete all ${state.conversations.length} saved conversations? This cannot be undone.`)) return;
  state.conversations = [];
  saveConversations();
  startNewConversation();
}

// Export as JSON: the question, the reply, and the decision behind it, so an exported
// conversation can be read, shared with support, or turned into test questions.
function exportJson(conversations, filename) {
  const data = {
    exported_at: new Date().toISOString(),
    source: "Help center assistant",
    conversations: conversations.map((c) => ({
      id: c.id,
      title: c.title,
      created: c.created,
      updated: c.updated,
      turns: c.turns.map((t) => ({
        question: t.question,
        previous_message: t.previous || null,
        mode: t.mode,
        action: t.response.action,
        reply: t.response.message,
        reason: t.response.reason,
        decided_by: t.response.decided_by,
        source_card: t.response.source ? t.response.source.id : null,
        options: t.response.options.map((o) => o.card_id),
        search_query: t.response.search_query || null,
        matches: t.response.matches.map((m) => ({ card_id: m.card_id, score: m.score, rerank_score: m.rerank_score ?? null })),
        rating: t.rating || null,
      })),
    })),
  };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = el("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function slug(text) {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40) || "conversation";
}

function formatWhen(iso) {
  const date = new Date(iso);
  const sameDay = date.toDateString() === new Date().toDateString();
  return sameDay
    ? date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : date.toLocaleDateString([], { day: "numeric", month: "short" });
}

function renderSidebar() {
  const list = $("conv-list");
  list.replaceChildren();
  for (const conversation of state.conversations) {
    const current = conversation.id === state.currentId;
    const item = el("li", "conv-item" + (current ? " current" : ""));

    const open = el("button", "conv-open");
    open.type = "button";
    if (current) open.setAttribute("aria-current", "page");
    open.append(el("span", "conv-title", conversation.title));
    const count = conversation.turns.length;
    open.append(el("span", "conv-meta", `${count} ${count === 1 ? "reply" : "replies"} · ${formatWhen(conversation.updated)}`));
    open.addEventListener("click", () => openConversation(conversation.id));

    const actions = el("div", "conv-actions");
    const exportButton = el("button", "mini-btn");
    exportButton.type = "button";
    exportButton.innerHTML = icon("download", 15);
    exportButton.setAttribute("aria-label", `Export "${conversation.title}" as JSON`);
    exportButton.title = "Export as JSON";
    exportButton.addEventListener("click", () =>
      exportJson([conversation], `conversation-${slug(conversation.title)}.json`)
    );
    const deleteButton = el("button", "mini-btn danger");
    deleteButton.type = "button";
    deleteButton.innerHTML = icon("trash-2", 15);
    deleteButton.setAttribute("aria-label", `Delete "${conversation.title}"`);
    deleteButton.title = "Delete";
    deleteButton.addEventListener("click", () => deleteConversation(conversation.id));
    actions.append(exportButton, deleteButton);

    item.append(open, actions);
    list.append(item);
  }
  const empty = state.conversations.length === 0;
  $("conv-empty").hidden = !empty;
  $("export-all").disabled = empty;
  $("delete-all").disabled = empty;
}

function renderStorageNote() {
  const note = $("storage-note");
  note.innerHTML = icon("info", 14);
  note.append(
    document.createTextNode(
      state.storageOk
        ? "Saved only in this browser, on this device."
        : "This browser does not allow saving, so conversations are not kept."
    )
  );
}

function setupSidebar() {
  state.conversations = loadConversations();
  const exportAll = $("export-all");
  exportAll.innerHTML = icon("download", 15);
  exportAll.append(document.createTextNode("Export all"));
  exportAll.addEventListener("click", () =>
    exportJson(state.conversations, `conversations-${new Date().toISOString().slice(0, 10)}.json`)
  );
  const deleteAll = $("delete-all");
  deleteAll.innerHTML = icon("trash-2", 15);
  deleteAll.append(document.createTextNode("Delete all"));
  deleteAll.addEventListener("click", deleteAllConversations);

  // One button: collapses the sidebar on wide screens, opens it as a drawer on small ones.
  const toggle = $("sidebar-toggle");
  toggle.innerHTML = icon("panel-left", 18);
  const small = window.matchMedia("(max-width: 900px)");
  let collapsed = false;
  try { collapsed = localStorage.getItem("sidebar") === "hidden"; } catch {}
  const paint = () => {
    const visible = small.matches ? $("sidebar").classList.contains("open") : !collapsed;
    $("app").classList.toggle("sidebar-hidden", !small.matches && collapsed);
    toggle.setAttribute("aria-expanded", String(visible));
    toggle.setAttribute("aria-label", visible ? "Hide conversations" : "Show conversations");
  };
  toggle.addEventListener("click", () => {
    if (small.matches) {
      $("sidebar").classList.contains("open") ? closeSidebarOnMobile() : openSidebarOnMobile();
    } else {
      collapsed = !collapsed;
      try { localStorage.setItem("sidebar", collapsed ? "hidden" : "shown"); } catch {}
    }
    paint();
  });
  small.addEventListener("change", () => { closeSidebarOnMobile(); paint(); });
  $("sidebar-scrim").addEventListener("click", () => { closeSidebarOnMobile(); paint(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && $("sidebar").classList.contains("open")) { closeSidebarOnMobile(); paint(); toggle.focus(); }
  });
  state.paintSidebarToggle = paint;
  paint();
  renderSidebar();
  renderStorageNote();
}

function openSidebarOnMobile() {
  $("sidebar").classList.add("open");
  $("sidebar-scrim").hidden = false;
  $("main").inert = true;
  $("new-chat").focus();
}

function closeSidebarOnMobile() {
  if (!$("sidebar").classList.contains("open")) return;
  $("sidebar").classList.remove("open");
  $("sidebar-scrim").hidden = true;
  $("main").inert = false;
  state.paintSidebarToggle && state.paintSidebarToggle();
}

// ---- sending -----------------------------------------------------------------------

async function send(question, previous = null) {
  if (state.busy) return;
  state.busy = true;
  sendEl.disabled = true;
  $("new-chat").disabled = true;
  welcomeEl.hidden = true;

  messagesEl.append(userRow(question));

  const thinking = el("div", "thinking");
  thinking.append(el("span", "thinking-bar"), el("span", null, "Searching the help articles"));
  messagesEl.append(thinking);
  scrollDown();

  const mode = state.mode;
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question, previous_message: previous, mode }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const response = await res.json();

    // After a clarifying question, the customer's next message is read together with it.
    setPendingClarify(response.action === "clarify" ? (previous ? `${previous} ${question}` : question) : null);

    const turn = { question, previous, mode, response, rating: null };
    recordTurn(turn); // adds the turn to the current conversation (state.turns) and saves it
    thinking.replaceWith(renderBotMessage(turn, state.turns.length - 1));
    // An earlier clarifying question has been answered: its option buttons are no longer needed.
    disableAnsweredOptions();
  } catch {
    thinking.replaceWith(
      el("p", "error-note", "The assistant is not reachable. Check that the server and Ollama are running, then send your message again.")
    );
  } finally {
    state.busy = false;
    sendEl.disabled = false;
    $("new-chat").disabled = false;
    scrollDown();
    inputEl.focus();
  }
}

function userRow(question) {
  const row = el("div", "msg-user");
  row.append(el("p", null, question));
  return row;
}

function disableAnsweredOptions() {
  state.turns.slice(0, -1).forEach((t) => t.element?.querySelectorAll(".option").forEach((b) => (b.disabled = true)));
}

function scrollDown() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ---- rendering a reply ----------------------------------------------------------------

function renderBotMessage(turn, index) {
  const r = turn.response;
  const box = el("article", `msg-bot ${r.action}`);
  turn.element = box;

  const head = el("div", "msg-head");
  head.append(withIcon(el("span", "badge"), ROUTE[r.action].icon, ROUTE[r.action].label, 15));
  const traceLink = el("button", "trace-link", "How I decided");
  traceLink.type = "button";
  traceLink.addEventListener("click", () => openTrace(index));
  head.append(traceLink);
  box.append(head);

  box.append(el("p", "msg-text", r.message));

  if (r.action === "clarify" && r.options.length) {
    const options = el("div", "options");
    const answered = index < state.turns.length - 1; // the customer already replied
    for (const option of r.options) {
      const button = el("button", "option", option.label);
      button.type = "button";
      button.disabled = answered;
      button.addEventListener("click", () => {
        options.querySelectorAll("button").forEach((b) => (b.disabled = true));
        send(option.label, state.pendingClarify || turn.question);
      });
      options.append(button);
    }
    box.append(options);
  }

  if (r.source) {
    const details = el("details", "source");
    const summary = el("summary");
    summary.innerHTML = icon("file-text", 16);
    summary.append(document.createTextNode("Source: "));
    summary.append(el("strong", null, r.source.question));
    const chev = el("span", "chev");
    chev.innerHTML = icon("chevron-down", 16);
    summary.append(chev);
    details.append(summary, el("p", "source-text", r.source.answer));
    box.append(details);
  }

  const actions = el("div", "msg-actions");
  // Feedback on every kind of reply: a thumbs-down on a decline or a clarifying
  // question shows a wrong decision, not only a wrong answer.
  actions.append(el("span", null, turn.rating ? "Thanks, your feedback was saved." : FEEDBACK_PROMPT[r.action]));
  actions.append(feedbackButton("up", turn), feedbackButton("down", turn));
  if (r.action === "decline" || (r.action === "answer" && r.confidence < 0.6)) {
    const mail = withIcon(el("a", "human-btn"), "mail", "Email support", 16);
    const subject = encodeURIComponent("Question from the help center");
    const body = encodeURIComponent(`My question: ${turn.question}`);
    mail.href = `mailto:${SUPPORT_EMAIL}?subject=${subject}&body=${body}`;
    actions.append(mail);
  }
  if (actions.childElementCount) box.append(actions);
  return box;
}

function feedbackButton(rating, turn) {
  const button = el("button", "feedback-btn");
  button.type = "button";
  button.innerHTML = icon(rating === "up" ? "thumbs-up" : "thumbs-down", 16);
  button.setAttribute("aria-label", rating === "up" ? "Helpful" : "Not helpful");
  button.setAttribute("aria-pressed", String(turn.rating === rating));
  button.disabled = Boolean(turn.rating); // a saved conversation remembers its rating
  button.addEventListener("click", async () => {
    const row = button.parentElement;
    row.querySelectorAll(".feedback-btn").forEach((b) => (b.disabled = true));
    button.setAttribute("aria-pressed", "true");
    const r = turn.response;
    try {
      const res = await fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          response_id: r.id,
          rating,
          question: turn.question,
          previous_message: turn.previous,
          answer: r.message,
          // The decision behind the reply, so feedback can become a test case or training example.
          action: r.action,
          source_card: r.source ? r.source.id : null,
          options: r.options.map((o) => o.card_id),
          decided_by: r.decided_by,
          reason: r.reason,
          search_query: r.search_query,
          matches: r.matches.map((m) => ({ card_id: m.card_id, score: m.score })),
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      row.firstChild.textContent = "Thanks, your feedback was saved.";
      turn.rating = rating;
      saveConversations();
    } catch {
      row.firstChild.textContent = "Feedback could not be saved.";
    }
  });
  return button;
}

// ---- the trace panel ---------------------------------------------------------------------

// Bars do not start at 0: real similarity scores sit between about 0.3 and 0.9,
// so a zoomed scale keeps differences (and the cut-off line) visible.
function makeScale(scores) {
  const low = Math.min(0.3, state.minRelevance - 0.1, ...scores) - 0.02;
  const high = Math.max(0.9, ...scores);
  return (score) => Math.max(0, Math.min(100, ((score - low) / (high - low)) * 100));
}

function renderTrace(turn) {
  const r = turn.response;
  traceBodyEl.replaceChildren();

  const asked = el("p", "trace-question");
  asked.append("You asked ", el("q", null, turn.question));
  if (turn.previous) asked.append(" as a reply to ", el("q", null, turn.previous));
  asked.append(turn.mode === "rules" ? " (rules only mode)." : ".");
  traceBodyEl.append(asked);

  const steps = el("ol", "trace-steps");

  // Step 1: retrieval with the relevance cut-off
  const chosen = new Set([r.source && r.source.id, ...r.options.map((o) => o.card_id)].filter(Boolean));
  const search = el("li", "trace-step");
  search.append(el("h3", null, "Searched the help articles"));
  if (r.search_query) {
    const rewrite = el("p", "rewrite-note");
    rewrite.append("The first search found nothing relevant, so I searched again for ", el("q", null, r.search_query), ".");
    search.append(rewrite);
  }
  const list = el("ul", "scores");
  const toPercent = makeScale(r.matches.map((m) => m.score));
  for (const m of r.matches) {
    const row = el("li", "score-row" + (chosen.has(m.card_id) ? " chosen" : ""));
    row.style.setProperty("--route", `var(--${r.action})`);
    const label = el("span", "label");
    label.append(el("b", null, m.card_id), document.createTextNode(m.question));
    label.title = m.question;
    const bar = el("span", "bar");
    const fill = el("span", "fill");
    fill.style.width = toPercent(m.score) + "%";
    const cutoff = el("span", "cutoff");
    cutoff.style.left = toPercent(state.minRelevance) + "%";
    bar.append(fill, cutoff);
    row.append(label, el("span", "value", m.score.toFixed(2)), bar);
    if (m.out_of_scope) {
      row.append(el("span", "note", "Out-of-scope example. Never shown to the AI as an answer."));
    }
    list.append(row);
  }
  search.append(list);
  search.append(el("p", "cutoff-legend", `Relevance cut-off ${state.minRelevance.toFixed(2)}: below it, no article counts as relevant.`));
  steps.append(search);

  // Step 2: who decided, and why
  const decide = el("li", "trace-step");
  decide.append(el("h3", null, "Decided what to do"));
  decide.append(el("p", null, DECIDED_BY[r.decided_by]));
  decide.append(el("p", "reason", r.reason));
  steps.append(decide);

  // Step 3: the outcome
  const final = el("li", `trace-step final trace-final ${r.action}`);
  final.append(el("h3", null, "Result"));
  const pill = withIcon(el("span", "route-pill"), ROUTE[r.action].icon, outcomeText(r), 15);
  final.append(pill);
  steps.append(final);

  traceBodyEl.append(steps);
}

function outcomeText(r) {
  if (r.action === "answer") return `Answered from ${r.source.id}`;
  if (r.action === "clarify") return `Asked to choose: ${r.options.map((o) => o.card_id).join(" or ")}`;
  return "Declined politely";
}

// ---- start --------------------------------------------------------------------------------

renderSuggestions();
setupSidebar();
setupTheme();
setupModeSwitch();
setupComposer();
setupTracePanel();
setupReplyTo();
setupNewChat();
checkHealth();
inputEl.focus();
