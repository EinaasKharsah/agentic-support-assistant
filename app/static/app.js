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

// ---- state -------------------------------------------------------------------

const state = {
  mode: "llm",
  minRelevance: 0.55,
  turns: [], // { question, previous, mode, response, element }
  selected: -1,
  pendingClarify: null, // the vague question, sent along with the customer's reply
  busy: false,
};

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
  state.turns.forEach((t) => t.element && t.element.classList.remove("selected"));
  // Return focus to the button that opened the panel.
  turn?.element?.querySelector(".trace-link")?.focus();
}

function setupNewChat() {
  const button = $("new-chat");
  button.innerHTML = icon("rotate-ccw", 15);
  button.append(el("span", "label", "New conversation"));
  button.addEventListener("click", () => {
    if (state.busy) return;
    closeTrace();
    state.turns = [];
    state.selected = -1;
    setPendingClarify(null);
    messagesEl.replaceChildren(welcomeEl);
    welcomeEl.hidden = false;
    inputEl.value = "";
    inputEl.focus();
  });
}

// ---- sending -----------------------------------------------------------------------

async function send(question, previous = null) {
  if (state.busy) return;
  state.busy = true;
  sendEl.disabled = true;
  $("new-chat").disabled = true;
  welcomeEl.hidden = true;

  const userRow = el("div", "msg-user");
  userRow.append(el("p", null, question));
  messagesEl.append(userRow);

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

    const turn = { question, previous, mode, response };
    state.turns.push(turn);
    thinking.replaceWith(renderBotMessage(turn, state.turns.length - 1));
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
    for (const option of r.options) {
      const button = el("button", "option", option.label);
      button.type = "button";
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
  if (r.action === "answer") {
    actions.append(el("span", null, "Was this helpful?"));
    const up = feedbackButton("up", turn);
    const down = feedbackButton("down", turn);
    actions.append(up, down);
  }
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
  button.setAttribute("aria-pressed", "false");
  button.addEventListener("click", async () => {
    const row = button.parentElement;
    row.querySelectorAll(".feedback-btn").forEach((b) => (b.disabled = true));
    button.setAttribute("aria-pressed", "true");
    try {
      await fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          response_id: turn.response.id,
          rating,
          question: turn.question,
          answer: turn.response.message,
        }),
      });
      row.firstChild.textContent = "Thanks, your feedback was saved.";
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
setupTheme();
setupModeSwitch();
setupComposer();
setupTracePanel();
setupReplyTo();
setupNewChat();
checkHealth();
inputEl.focus();
