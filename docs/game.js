// The page is the toy. Python, in worker.js, still owns the rules and the agent.

const COORDS = [
  [0, 0], [0, 3], [0, 6], [1, 1], [1, 3], [1, 5],
  [2, 2], [2, 3], [2, 4], [3, 0], [3, 1], [3, 2],
  [3, 4], [3, 5], [3, 6], [4, 2], [4, 3], [4, 4],
  [5, 1], [5, 3], [5, 5], [6, 0], [6, 3], [6, 6],
];

const MILLS = [
  [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
  [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24],
  [1, 10, 22], [4, 11, 19], [7, 12, 16], [2, 5, 8],
  [17, 20, 23], [9, 13, 18], [6, 14, 21], [3, 15, 24],
  [1, 4, 7], [3, 6, 9], [16, 19, 22], [18, 21, 24],
];

const board = document.querySelector("#board");
const hint = document.querySelector("#hint");
const status = document.querySelector("#status");
const errorBox = document.querySelector("#error");
const log = document.querySelector("#log");
const gate = document.querySelector("#gate");
const table = document.querySelector("#table");
const again = document.querySelector("#again");
const cards = document.querySelectorAll(".card");

const worker = new Worker("worker.js");
const pending = new Map();
let requestId = 0;
let pythonReady = false;
let state = null;
let busy = false;
let selected = null;
let captureAt = null;
let humanStarts = true;
let clock = null;
let landed = null;

const FILES = [
  "python/agent.py",
  "python/web_game.py",
  "python/famnit_gym/__init__.py",
  "python/famnit_gym/envs/__init__.py",
  "python/famnit_gym/envs/mill/__init__.py",
  "python/famnit_gym/envs/mill/mill_model.py",
];

worker.onmessage = (event) => {
  const message = event.data;
  if (message.type === "ready") {
    pythonReady = true;
    document.body.classList.add("ready");
    cards.forEach((card) => { card.disabled = false; });
    return;
  }
  if (message.type === "init-error") {
    showError(message.error);
    document.querySelector("#warmup").textContent = "No se pudo abrir el juego.";
    return;
  }
  if (message.type === "result") {
    const waiter = pending.get(message.id);
    pending.delete(message.id);
    if (!waiter) return;
    if (message.ok) waiter.resolve(message.result);
    else waiter.reject(new Error(message.error));
  }
};

boot();

cards.forEach((card) => {
  card.addEventListener("click", () => begin(card));
});

document.querySelectorAll(".who-btn").forEach((button) => {
  button.addEventListener("click", () => {
    humanStarts = button.dataset.start === "human";
    document.querySelectorAll(".who-btn").forEach((item) => {
      item.classList.toggle("is-on", item === button);
    });
  });
});

again.addEventListener("click", () => {
  if (busy) return;
  table.hidden = true;
  gate.hidden = false;
  gate.classList.remove("leaving");
  status.textContent = "Te toca";
});

async function boot() {
  try {
    const files = {};
    for (const path of FILES) {
      const response = await fetch(path, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`No pude leer ${path} (${response.status}).`);
      }
      files[`/game/${path.slice("python/".length)}`] = await response.text();
    }
    worker.postMessage({ type: "init", files });
  } catch (error) {
    showError(error.message);
    document.querySelector("#warmup").textContent = "Abre la página desde el servidor del proyecto.";
  }
}

function callPython(name, payload) {
  const id = ++requestId;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ type: "call", id, name, payload });
  });
}

async function begin(card) {
  if (!pythonReady || busy) return;
  cards.forEach((item) => item.classList.remove("is-chosen"));
  card.classList.add("is-chosen");
  gate.classList.add("leaving");
  await wait(280);
  gate.hidden = true;
  table.hidden = false;
  await startGame(card.dataset.difficulty);
}

async function startGame(difficulty) {
  clearError();
  selected = null;
  captureAt = null;
  landed = null;
  log.replaceChildren();
  setBusy(true, "Preparando…");
  try {
    state = JSON.parse(await callPython("new_game", {
      human: humanStarts ? 1 : 2,
      difficulty,
    }));
    draw();
    describeTurn();
    if (state.turn === state.ai) await requestAiMove();
  } catch (error) {
    showError(error.message);
    status.textContent = "No pudo empezar.";
    table.hidden = true;
    gate.hidden = false;
    gate.classList.remove("leaving");
  } finally {
    setBusy(false);
  }
}

async function requestAiMove() {
  setBusy(true, "Pensando…");
  try {
    state = JSON.parse(await callPython("play_ai", {}));
    if (state.move) {
      landed = state.move[1];
      addLog("ai", state.move);
    }
    selected = null;
    captureAt = null;
    draw();
    describeTurn();
  } catch (error) {
    showError(error.message);
    status.textContent = "Molino se lió.";
  } finally {
    setBusy(false);
  }
}

function onPoint(position) {
  if (!state || busy || state.over || state.turn !== state.human) return;
  const moves = state.legal;

  if (captureAt) {
    const move = moves.find((item) => item[0] === captureAt[0] && item[1] === captureAt[1] && item[2] === position);
    if (move) submitMove(move);
    return;
  }

  const placing = moves.length > 0 && moves.every((item) => item[0] === 0);
  if (placing) {
    chooseDestination(0, position, moves.filter((item) => item[1] === position));
    return;
  }

  if (selected === null) {
    if (moves.some((item) => item[0] === position)) {
      selected = position;
      draw();
      hint.textContent = "Ahora el destino.";
    }
    return;
  }

  if (position === selected) {
    selected = null;
    draw();
    describeTurn();
    return;
  }

  const options = moves.filter((item) => item[0] === selected && item[1] === position);
  if (options.length === 0 && moves.some((item) => item[0] === position)) {
    selected = position;
    draw();
    return;
  }
  chooseDestination(selected, position, options);
}

function chooseDestination(src, dst, options) {
  if (options.length === 0) return;
  if (options.length === 1) {
    submitMove(options[0]);
    return;
  }
  captureAt = [src, dst];
  draw();
  hint.textContent = "Molino cerrado. Quítale una.";
}

async function submitMove(move) {
  setBusy(true, "Vale…");
  try {
    state = JSON.parse(await callPython("play_human", { move }));
    landed = move[1];
    addLog("you", move);
    selected = null;
    captureAt = null;
    draw();
    describeTurn();
    if (!state.over && state.turn === state.ai) await requestAiMove();
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
}

function describeTurn() {
  if (!state) return;
  paintHand("#you-hand", seat(state.human).holding, "you");
  paintHand("#ai-hand", seat(state.ai).holding, "ai");

  if (state.over) {
    status.textContent = state.winner === state.human ? "¡Ganaste!" : "Gana Molino.";
    hint.textContent = "Otra, cuando quieras.";
    return;
  }
  if (state.turn === state.human) {
    status.textContent = "Te toca";
    const placing = state.legal.every((item) => item[0] === 0);
    hint.textContent = placing ? "Pon una ficha en un punto libre." : "Elige una ficha tuya.";
    return;
  }
  status.textContent = "Turno de Molino";
  hint.textContent = "";
}

function seat(player) {
  return { holding: state.holding[player - 1] };
}

function paintHand(selector, count, kind) {
  const holder = document.querySelector(selector);
  holder.replaceChildren();
  for (let index = 0; index < count; index += 1) {
    const pip = document.createElement("i");
    pip.className = `pip ${kind}`;
    holder.appendChild(pip);
  }
}

function draw() {
  const highlights = highlightSets();
  const svg = [];
  for (const [a, b] of edges()) {
    const [x1, y1] = point(a);
    const [x2, y2] = point(b);
    svg.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#1c140f" stroke-width="1.7" stroke-linecap="round"></line>`);
  }

  for (let position = 1; position <= 24; position += 1) {
    const owner = state ? state.board[position - 1] : 0;
    const [x, y] = point(position);
    const classes = ["point"];
    if (!highlights.playable.has(position)) classes.push("idle");
    if (selected === position) {
      svg.push(`<circle cx="${x}" cy="${y}" r="4.15" fill="none" stroke="#1c140f" stroke-width="0.4"></circle>`);
    } else if (highlights.playable.has(position) && owner === 0) {
      const color = highlights.captures.has(position) ? "#d01212" : "#fff8e6";
      svg.push(`<circle cx="${x}" cy="${y}" r="2.35" fill="none" stroke="${color}" stroke-width="0.45"></circle>`);
    }
    if (owner === 0) {
      svg.push(`<circle cx="${x}" cy="${y}" r="1.25" fill="#1c140f"></circle>`);
    } else {
      svg.push(disc(x, y, owner === state.human ? "you" : "ai", position === landed));
    }
    svg.push(`<circle cx="${x}" cy="${y}" r="4.5" fill="transparent" class="${classes.join(" ")}" data-position="${position}"></circle>`);
  }

  board.innerHTML = svg.join("");
  for (const node of board.querySelectorAll(".point:not(.idle)")) {
    node.addEventListener("click", () => onPoint(Number(node.dataset.position)));
  }
}

function disc(x, y, kind, fresh) {
  const outer = kind === "you" ? "#7f9f00" : "#800040";
  const inner = kind === "you" ? "#efe200" : "#d01212";
  const radius = fresh ? 3.7 : 3.35;
  return `<circle cx="${x}" cy="${y}" r="${radius}" fill="${outer}"></circle><circle cx="${x}" cy="${y}" r="${radius * 0.62}" fill="${inner}"></circle>`;
}

function highlightSets() {
  const playable = new Set();
  const captures = new Set();
  if (!state || busy || state.over || state.turn !== state.human) {
    return { playable, captures };
  }
  if (captureAt) {
    for (const move of state.legal) {
      if (move[0] === captureAt[0] && move[1] === captureAt[1]) captures.add(move[2]);
    }
    captures.forEach((position) => playable.add(position));
    return { playable, captures };
  }
  const placing = state.legal.length > 0 && state.legal.every((item) => item[0] === 0);
  if (placing || selected !== null) {
    const source = placing ? 0 : selected;
    for (const move of state.legal) {
      if (move[0] === source) playable.add(move[1]);
    }
    if (selected !== null) playable.add(selected);
  } else {
    for (const move of state.legal) playable.add(move[0]);
  }
  return { playable, captures };
}

function edges() {
  const seen = new Set();
  const lines = [];
  for (const mill of (state && state.mills) || MILLS) {
    for (let index = 0; index < mill.length - 1; index += 1) {
      const a = mill[index];
      const b = mill[index + 1];
      const key = a < b ? `${a}-${b}` : `${b}-${a}`;
      if (seen.has(key)) continue;
      seen.add(key);
      lines.push([a, b]);
    }
  }
  return lines;
}

function point(position) {
  const [row, column] = COORDS[position - 1];
  return [10 + (column / 6) * 80, 10 + (row / 6) * 80];
}

function formatMove(move) {
  const [src, , take] = move;
  if (src === 0) return take ? "Nueva ficha · captura" : "Nueva ficha";
  return take ? "Mueve · captura" : "Mueve una ficha";
}

function addLog(who, move) {
  const item = document.createElement("li");
  item.className = who;
  item.innerHTML = `<i class="pip ${who}"></i><span></span>`;
  item.querySelector("span").textContent = formatMove(move);
  log.prepend(item);
}

function setBusy(value, message) {
  busy = value;
  again.disabled = value;
  cards.forEach((card) => { card.disabled = value || !pythonReady; });
  if (message) status.textContent = message;
  if (!table.hidden) draw();
  window.clearInterval(clock);
  if (!value) return;
  const started = performance.now();
  clock = window.setInterval(() => {
    const seconds = Math.floor((performance.now() - started) / 1000);
    if (seconds > 0 && message) status.textContent = `${message.replace("…", "")} ${seconds}s`;
  }, 500);
}

function showError(text) {
  errorBox.hidden = false;
  errorBox.textContent = text;
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

function wait(ms) {
  return new Promise((resolve) => { window.setTimeout(resolve, ms); });
}
