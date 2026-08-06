/* bird_count web UI — drives train.py / test.py from the browser. */

const S = {
  page: 'train',      // 'train' | 'test' | 'annotations'
  kind: 'train',      // entrypoint key currently configured
  entrypoints: [],    // [{key, label, page, blurb, script}]
  schemas: {},        // kind -> argparse spec
  values: {},         // kind -> {dest: value}
  runId: null,        // run currently displayed
  cursor: 0,          // log cursor for the displayed run
  detail: null,       // last /log payload
  timer: null,
  sort: { key: 'err', dir: -1 },
  galleryFor: null,   // run id whose gallery is already loaded
  missing: [],        // required fields still empty
  listTimer: null,    // background refresh of the run list
  runs: [],           // one cached list; the sidebar filters it per workspace
  activeId: null,     // the one globally active process, even on another tab
  runByPage: {},      // last run viewed in each operation (no shared panel)
  stateVersion: null, // structured metrics/result version for incremental polls
  pollBusy: false,
};

const labelOf = (key) => S.entrypoints.find((e) => e.key === key)?.label ?? key;

const $ = (sel) => document.querySelector(sel);
const el = (tag, props = {}, ...kids) => {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    // `dataset` and attribute-style keys must not go through plain assignment:
    // element.dataset is a read-only accessor, and "data-x"/"role" are attributes.
    if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key.includes('-') || key === 'role') node.setAttribute(key, value);
    else node[key] = value;
  }
  for (const kid of kids.flat()) node.append(kid?.nodeType ? kid : document.createTextNode(kid));
  return node;
};

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* keep statusText */ }
    throw new Error(detail);
  }
  return res.json();
}

function toast(message) {
  const node = $('#toast');
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { node.hidden = true; }, 6000);
}

/* ---------------- configuration form ---------------- */

const STORE = (kind) => `birdcount.webui.values.${kind}`;

// Fields that get a "pick one" dropdown next to their text input. Each entry
// names the endpoint to enumerate from and the placeholder to show.
const PICKERS = {
  // Evaluation only ever wants a run's best model, i.e. the `best.pth` each run
  // directory holds. --resume takes the unfiltered list, since resuming means
  // picking up the periodic <epoch>_ckpt.tar that carries the optimizer state.
  ckpt: { url: '/api/checkpoints?root=../ckpts&match=best.pth', placeholder: 'browse checkpoints…' },
  resume: { url: '/api/checkpoints?root=../ckpts', placeholder: 'browse checkpoints…' },
  project_id: { url: '/api/label-studio/projects', placeholder: 'pick a project…' },
};

const pickerCache = new Map();
const PICKER_CACHE_MS = 60_000;

function defaultsOf(schema) {
  const out = {};
  for (const group of schema.groups) {
    for (const opt of group.options) {
      out[opt.dest] = opt.kind === 'bool' ? !!opt.default
        : opt.default === null || opt.default === undefined ? ''
        : Array.isArray(opt.default) ? opt.default.join(' ')
        : String(opt.default);
    }
  }
  return out;
}

async function loadSchema(kind) {
  if (!S.schemas[kind]) {
    S.schemas[kind] = await api(`/api/schema/${kind}`);
    const stored = localStorage.getItem(STORE(kind));
    S.values[kind] = { ...defaultsOf(S.schemas[kind]), ...(stored ? JSON.parse(stored) : {}) };
  }
  return S.schemas[kind];
}

function renderToolPicker() {
  const tools = S.entrypoints.filter((e) => e.page === S.page);
  const picker = $('#tool-picker');
  // Train and Test are pages of one tool each; only multi-tool pages need a list.
  picker.hidden = tools.length < 2;
  picker.replaceChildren();
  if (picker.hidden) return;
  for (const tool of tools) {
    const button = el('button', {
      className: tool.key === S.kind ? 'tool is-active' : 'tool',
      title: tool.script,
    }, tool.label);
    button.addEventListener('click', () => selectTool(tool.key));
    picker.append(button);
  }
}

function renderForm() {
  const schema = S.schemas[S.kind];
  const values = S.values[S.kind];
  const defaults = defaultsOf(schema);

  const blurb = $('#blurb');
  blurb.hidden = !schema.blurb;
  blurb.replaceChildren(schema.blurb || '', el('code', {}, schema.script));

  const form = $('#form');
  form.replaceChildren();
  schema.groups.forEach((group, index) => {
    const body = el('div');
    for (const opt of group.options) {
      body.append(renderField(opt, values, defaults));
    }
    form.append(el('details', { open: index < 2 }, el('summary', {}, group.title), body));
  });
  updateCommand();
}

function renderField(opt, values, defaults) {
  const wrap = el('div', { className: 'field' });
  const id = `f-${opt.dest}`;
  const setValue = (value) => {
    values[opt.dest] = value;
    localStorage.setItem(STORE(S.kind), JSON.stringify(values));
    wrap.classList.toggle('changed', String(value) !== String(defaults[opt.dest]));
    updateCommand();
  };

  let input;
  if (opt.kind === 'bool') {
    wrap.classList.add('is-bool');
    input = el('input', { type: 'checkbox', id, checked: !!values[opt.dest] });
    input.addEventListener('change', () => setValue(input.checked));
    wrap.append(input, el('div', {}, el('label', { htmlFor: id }, opt.flag),
      opt.help ? el('span', { className: 'hint' }, opt.help) : ''));
    wrap.classList.toggle('changed', !!values[opt.dest] !== !!defaults[opt.dest]);
    return wrap;
  }

  if (opt.multi) {
    // One value per line, so paths containing spaces survive the round trip.
    input = el('textarea', { id, rows: 3, value: values[opt.dest] ?? '', placeholder: 'one per line' });
  } else if (opt.kind === 'choice') {
    input = el('select', { id }, ...opt.choices.map((c) => el('option', { value: c, selected: values[opt.dest] === c }, c)));
  } else if (opt.kind === 'int' || opt.kind === 'float') {
    input = el('input', {
      type: 'number', id, value: values[opt.dest] ?? '',
      step: opt.kind === 'int' ? '1' : 'any',
    });
  } else {
    input = el('input', { type: 'text', id, value: values[opt.dest] ?? '' });
  }
  input.addEventListener('input', () => setValue(input.value));
  input.addEventListener('change', () => setValue(input.value));

  const label = el('label', { htmlFor: id }, opt.flag);
  if (opt.required) label.append(el('span', { className: 'req' }, 'required'));
  wrap.append(label);
  const source = PICKERS[opt.dest];
  if (source) {
    const picker = el('select', {}, el('option', { value: '' }, source.placeholder));
    picker.addEventListener('change', () => {
      if (!picker.value) return;
      input.value = picker.value;
      setValue(picker.value);
      picker.value = '';
    });
    loadPickerOptions(source, picker);
    wrap.append(el('div', { className: 'with-picker' }, input, picker));
  } else {
    wrap.append(input);
  }
  if (opt.help) wrap.append(el('span', { className: 'hint' }, opt.help));
  wrap.classList.toggle('changed', String(values[opt.dest] ?? '') !== String(defaults[opt.dest] ?? ''));
  return wrap;
}

async function loadPickerOptions(source, picker) {
  let data;
  try {
    const cached = pickerCache.get(source.url);
    if (cached && Date.now() - cached.at < PICKER_CACHE_MS) data = cached.data;
    else {
      data = await api(source.url);
      pickerCache.set(source.url, { at: Date.now(), data });
    }
  } catch (err) {
    picker.firstChild.textContent = `unavailable (${err.message})`;
    return;
  }
  if (data.error) {
    picker.firstChild.textContent = data.error;
    return;
  }
  for (const item of data.items) {
    // Checkpoints report path/size, projects report value/detail.
    const value = item.value ?? item.path;
    const detail = item.detail ?? (item.size_mb != null ? `${item.size_mb} MB` : '');
    picker.append(el('option', { value }, detail ? `${item.label}  (${detail})` : item.label));
  }
  if (!data.items.length) picker.firstChild.textContent = 'nothing found';
}

// Mirrors schema.build_argv on the server: options first, positionals last.
function updateCommand() {
  const schema = S.schemas[S.kind];
  const values = S.values[S.kind];
  const parts = ['python', '-u', schema.script];
  const positionals = [];
  const missing = [];

  for (const group of schema.groups) {
    for (const opt of group.options) {
      const value = values[opt.dest];
      if (opt.kind === 'bool') { if (value) parts.push(opt.flag); continue; }
      const items = String(value ?? '').split(opt.multi ? '\n' : ' ').map((s) => s.trim()).filter(Boolean);
      if (!items.length) { if (opt.required) missing.push(opt.flag); continue; }
      const quoted = items.map((i) => (i.includes(' ') ? `"${i}"` : i));
      if (opt.positional) positionals.push(...quoted);
      else parts.push(opt.flag, ...quoted);
    }
  }

  $('#cmd-preview').textContent = [...parts, ...positionals].join(' ');
  S.missing = missing;
  syncStartButton();
}

function syncStartButton() {
  const running = !!S.activeId;
  const start = $('#btn-start');
  start.disabled = running || S.missing.length > 0;
  start.title = running ? 'another operation is already running'
    : S.missing.length ? `fill in: ${S.missing.join(', ')}` : '';
  $('#btn-stop').disabled = !running;
}

/* ---------------- runs ---------------- */

async function start() {
  try {
    const run = await api('/api/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind: S.kind, values: S.values[S.kind] }),
    });
    S.activeId = run.id;
    S.runs = [run, ...S.runs.filter((item) => item.id !== run.id)];
    selectRun(run.id);
    renderRunList();
  } catch (err) {
    toast(err.message);
  }
}

async function stop() {
  if (!S.activeId) return;
  try { await api(`/api/runs/${S.activeId}/stop`, { method: 'POST' }); } catch (err) { toast(err.message); }
}

function showEmptyState(show) {
  $('#empty-state').hidden = !show;
  for (const sel of ['#chart-panel', '.panel-log']) $(sel).hidden = show;
  if (show) $('#result-panel').hidden = true;
  $('#workspace-run').hidden = show;
}

function selectRun(runId) {
  if (!runId || runId === S.runId) return;
  showEmptyState(false);
  S.runId = runId;
  const summary = S.runs.find((run) => run.id === runId);
  const page = summary ? pageOfKind(summary.kind) : S.page;
  S.runByPage[page === 'annotations' ? summary?.kind ?? S.kind : page] = runId;
  S.cursor = 0;
  S.detail = null;
  S.stateVersion = null;
  S.galleryFor = null;
  $('#log').replaceChildren();
  $('#result-panel').hidden = true;
  $('#gallery').replaceChildren();
  $('#workspace-run').hidden = false;
  $('#workspace-run-name').textContent = summary ? `${labelOf(summary.kind)} · ${formatRunTime(summary.started_at)}` : runId;
  clearInterval(S.timer);
  poll();
  S.timer = setInterval(poll, 900);
  renderRunList();
}

async function poll() {
  if (!S.runId || S.pollBusy) return;
  const runId = S.runId;
  S.pollBusy = true;
  let data;
  try {
    const version = S.stateVersion == null ? '' : `&state_version=${S.stateVersion}`;
    const tail = S.cursor === 0 ? '&tail=2000' : '';
    data = await api(`/api/runs/${runId}/log?cursor=${S.cursor}${tail}${version}`);
  } catch (err) {
    clearInterval(S.timer);
    toast(err.message);
    return;
  } finally {
    S.pollBusy = false;
  }
  if (runId !== S.runId) return;
  const structuredChanged = Object.hasOwn(data, 'metrics');
  if (!structuredChanged && S.detail) {
    data.metrics = S.detail.metrics;
    data.result = S.detail.result;
  }
  S.cursor = data.cursor;
  S.stateVersion = data.state_version;
  S.detail = data;
  appendLog(data.lines);

  updateGlobalStatus();
  $('#log-path').textContent = `${data.log_path}  ·  ${fmtDuration(data.elapsed)}`;
  syncStartButton();

  // A rendering slip must never stop the poll loop — the log is the payload.
  try {
    if (structuredChanged) {
      drawChart(data);
      if (data.kind === 'test') renderEvaluation(data);
      else if (data.kind === 'density_regions') renderRegions(data);
    }
  } catch (err) {
    toast(`render error: ${err.message}`);
  }

  if (data.status !== 'running') {
    clearInterval(S.timer);
    S.timer = null;
    if (S.activeId === data.id) S.activeId = null;
    refreshRunList();
    if (RESULT_TABLES[data.kind] && S.galleryFor !== S.runId) loadGallery(S.runId);
  }
}

function appendLog(lines) {
  if (!lines.length) return;
  const log = $('#log');
  const follow = $('#follow').checked;
  const fragment = document.createDocumentFragment();
  for (const line of lines) {
    let cls = '';
    if (/Traceback|Error|error:|FAILED|exit code [1-9]/.test(line)) cls = 'l-err';
    else if (/saved best/.test(line)) cls = 'l-best';
    else if (/ Val \(/.test(line)) cls = 'l-val';
    else if (/^\[webui\]|^\$ /.test(line)) cls = 'l-meta';
    fragment.append(el('span', { className: cls }, line + '\n'));
  }
  log.append(fragment);
  while (log.childElementCount > 5000) log.firstElementChild.remove();
  if (follow) log.scrollTop = log.scrollHeight;
}

const pageOfKind = (kind) => S.entrypoints.find((entry) => entry.key === kind)?.page ?? kind;
const contextKey = (page = S.page) => page === 'annotations' ? S.kind : page;
const runsForPage = (page = S.page) => S.runs.filter((run) =>
  pageOfKind(run.kind) === page && (page !== 'annotations' || run.kind === S.kind));
const formatRunTime = (stamp) => new Date(stamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

function updateGlobalStatus() {
  const active = S.runs.find((run) => run.id === S.activeId);
  const status = active ? 'running' : S.detail?.status ?? 'idle';
  $('#status-pill').textContent = active ? `${labelOf(active.kind)} running` : status;
  $('#status-pill').dataset.status = status;
}

function renderRunList() {
  const list = $('#run-list');
  list.replaceChildren();
  const visible = runsForPage();
  $('#runs-title').textContent = `${labelOf(S.page === 'annotations' ? S.kind : S.page)} runs`;
  $('#runs-count').textContent = String(visible.length);
  for (const run of visible) {
    const item = el('li', { className: run.id === S.runId ? 'sel' : '', tabIndex: 0, role: 'button', title: run.command },
      el('div', { className: 'r-top' },
        el('span', { className: 'r-kind' }, labelOf(run.kind)),
        el('span', { className: 'r-dot', 'data-status': run.status })),
      el('div', { className: 'r-sub' }, formatRunTime(run.started_at)),
      el('div', { className: 'r-sub' }, `${run.status} · ${fmtDuration(run.elapsed)}`));
    item.addEventListener('click', () => selectRun(run.id));
    item.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectRun(run.id); } });
    list.append(item);
  }
  if (!visible.length) list.append(el('div', { className: 'empty' }, `No ${S.page} runs yet.`));
  // Nothing to clear when the only entry is the job currently running.
  $('#btn-clear').disabled = !S.runs.some((run) => run.status !== 'running');
  updateGlobalStatus();
  syncStartButton();
}

async function refreshRunList(snapshot) {
  let data = snapshot;
  if (!data) {
    try { data = await api('/api/runs'); } catch { return; }
  }
  S.runs = data.runs;
  S.activeId = data.active;
  renderRunList();
}

// Shuts down the server this page is talking to, which frees its port.
async function quitServer() {
  const extra = [];
  if (S.detail?.status === 'running') extra.push('the run in progress');
  if (!confirm(`Shut down the web UI server and release port ${location.port || 80}?`
    + (extra.length ? `\n\nThis also stops ${extra.join(' and ')}, plus any Label Studio or tunnel started from here.` : ''))) return;

  let result;
  try {
    result = await api('/api/shutdown', { method: 'POST' });
  } catch (err) {
    return toast(err.message);
  }

  // The server has ~0.4s left. Stop every poller now so the page settles on a
  // clear "it is gone" instead of a stream of failed requests.
  clearInterval(S.timer);
  clearInterval(S.listTimer);
  clearTimeout(refreshLabelStudio._t);
  S.timer = S.listTimer = null;
  for (const view of Object.values(SERVICES)) clearTimeout(view.timer);

  const controls = ['#btn-start', '#btn-stop', '#btn-clear', '#btn-quit',
    '#ls-start', '#ls-stop', '#ng-start', '#ng-stop', '#btn-reset'];
  for (const id of controls) $(id).disabled = true;
  $('#status-pill').hidden = false;
  $('#status-pill').textContent = 'server stopped';
  $('#status-pill').dataset.status = 'stopped';

  const also = [result.run && 'the active run', ...(result.services || [])].filter(Boolean);
  toast(`server stopped — port released${also.length ? ` · also stopped: ${also.join(', ')}` : ''}`);
}

// Wipes the history — the logs go too, so a restart cannot bring them back.
async function clearRuns() {
  const keeping = S.detail?.status === 'running' ? '\n\nThe running job is kept.' : '';
  if (!confirm(`Delete all finished runs and their log files?${keeping}`)) return;

  let result;
  try {
    result = await api('/api/runs/clear', { method: 'POST' });
  } catch (err) {
    return toast(err.message);
  }
  toast(result.removed ? `cleared ${result.removed} run${result.removed === 1 ? '' : 's'}` : 'nothing to clear');
  if (result.failed?.length) toast(`kept ${result.failed.length} run(s): log file still in use`);

  if (result.kept) {          // a live run survived — show it instead of a blank page
    selectRun(result.kept);
    return;
  }
  clearInterval(S.timer);
  S.timer = null;
  S.runId = null;
  S.cursor = 0;
  S.detail = null;
  S.galleryFor = null;
  $('#log').replaceChildren();
  $('#log-path').textContent = '';
  $('#gallery').replaceChildren();
  $('#status-pill').textContent = 'idle';
  $('#status-pill').dataset.status = 'idle';
  showEmptyState(true);
  syncStartButton();
  await refreshRunList();
}

function fmtDuration(seconds) {
  if (!seconds && seconds !== 0) return '—';
  const s = Math.floor(seconds);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

/* ---------------- chart ---------------- */

const SERIES = {
  train: [
    { id: 'train_mae', label: 'train MAE', color: '#4d8dff', on: true, from: (m) => m.train.map((r) => [r.epoch, r.mae]) },
    { id: 'val_mae', label: 'val MAE', color: '#3fd07f', on: true, from: (m) => m.val.map((r) => [r.epoch, r.mae]) },
    { id: 'train_mse', label: 'train MSE', color: '#b18aff', on: false, from: (m) => m.train.map((r) => [r.epoch, r.mse]) },
    { id: 'val_mse', label: 'val MSE', color: '#e8b13a', on: false, from: (m) => m.val.map((r) => [r.epoch, r.mse]) },
    { id: 'loss', label: 'loss', color: '#ff6a63', on: false, from: (m) => m.train.map((r) => [r.epoch, r.loss]) },
  ],
  test: [
    { id: 'scatter', label: 'GT vs Pred', color: '#4d8dff', on: true, scatter: true,
      from: (_m, result) => result.images.map((r) => [r.gt, r.pred]) },
  ],
};

// Chart chrome follows the stylesheet's tokens, so the canvas can never drift
// from the page around it.
const token = (name, fallback) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

const toggleState = {};

function drawChart(data) {
  const defs = SERIES[data.kind];
  // Annotation tools have no metrics to plot — drop the panel entirely.
  $('#chart-panel').hidden = !defs;
  if (!defs) return;

  const series = defs
    .map((def) => ({ ...def, on: toggleState[def.id] ?? def.on, points: def.from(data.metrics, data.result).filter((p) => p[1] != null) }))
    .filter((s) => s.points.length);

  renderToggles(defs, series);

  const canvas = $('#chart');
  const visible = series.filter((s) => s.on);
  const hasData = visible.some((s) => s.points.length);
  $('#chart-empty').hidden = hasData;
  canvas.hidden = !hasData;
  if (!hasData) return;

  const dpr = window.devicePixelRatio || 1;
  // Guard against a degenerate width while the panel is still laying out.
  const width = Math.max(canvas.clientWidth || canvas.parentElement.clientWidth, 280);
  const height = 260;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const pad = { l: 52, r: 14, t: 12, b: 26 };
  const plotW = width - pad.l - pad.r;
  const plotH = height - pad.t - pad.b;

  const xs = visible.flatMap((s) => s.points.map((p) => p[0]));
  const ys = visible.flatMap((s) => s.points.map((p) => p[1]));
  let [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  let [y0, y1] = [Math.min(...ys, 0), Math.max(...ys)];
  if (data.kind === 'test') { x0 = 0; y0 = 0; x1 = y1 = Math.max(x1, y1) * 1.05; }
  if (x1 === x0) x1 = x0 + 1;
  if (y1 === y0) y1 = y0 + 1;

  const X = (v) => pad.l + ((v - x0) / (x1 - x0)) * plotW;
  const Y = (v) => pad.t + plotH - ((v - y0) / (y1 - y0)) * plotH;

  // grid + axis labels
  ctx.font = '10px ui-monospace, monospace';
  ctx.strokeStyle = token('--chart-grid', '#1a212b');
  ctx.fillStyle = token('--chart-tick', '#7a8798');
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (plotH * i) / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(width - pad.r, y); ctx.stroke();
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(fmtTick(y1 - ((y1 - y0) * i) / 4), pad.l - 8, y);
  }
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (let i = 0; i <= 4; i++) {
    const value = x0 + ((x1 - x0) * i) / 4;
    ctx.fillText(fmtTick(value), X(value), height - pad.b + 6);
  }

  if (data.kind === 'test') { // identity line: perfect prediction
    ctx.strokeStyle = token('--chart-guide', '#313c4b');
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(X(x0), Y(x0)); ctx.lineTo(X(x1), Y(x1)); ctx.stroke();
    ctx.setLineDash([]);
  }

  for (const s of visible) {
    ctx.strokeStyle = s.color;
    ctx.fillStyle = s.color;
    if (s.scatter) {
      for (const [px, py] of s.points) {
        ctx.beginPath(); ctx.arc(X(px), Y(py), 3, 0, Math.PI * 2); ctx.fill();
      }
      continue;
    }
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    s.points.forEach(([px, py], i) => (i ? ctx.lineTo(X(px), Y(py)) : ctx.moveTo(X(px), Y(py))));
    ctx.stroke();
    const last = s.points[s.points.length - 1];
    ctx.beginPath(); ctx.arc(X(last[0]), Y(last[1]), 2.5, 0, Math.PI * 2); ctx.fill();
  }
}

function fmtTick(v) {
  if (Math.abs(v) >= 1000) return v.toExponential(1);
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
}

function renderToggles(defs, series) {
  const host = $('#chart-toggles');
  const signature = defs.map((d) => d.id).join(',');
  if (host.dataset.signature !== signature) {
    host.replaceChildren();
    for (const def of defs) {
      const button = el('button', { dataset: { id: def.id } },
        el('span', { className: 'swatch', style: `background:${def.color}` }), def.label);
      button.addEventListener('click', () => {
        toggleState[def.id] = !(toggleState[def.id] ?? def.on);
        if (S.detail) drawChart(S.detail);
      });
      host.append(button);
    }
    host.dataset.signature = signature;
  }
  for (const button of host.children) {
    const match = series.find((s) => s.id === button.dataset.id);
    button.classList.toggle('on', !!match?.on);
    button.disabled = !match;
    button.style.opacity = match ? '' : '.35';
  }
}

/* ---------------- per-image results ---------------- */

// Two tools fill the result panel with a row per image, and they report
// different numbers: test.py a GT/prediction error, density_regions.py a count
// broken into regions. The columns (and the default sort) come from the kind.
const RESULT_TABLES = {
  test: {
    sort: 'err',
    columns: [
      { key: 'name', label: 'Image' },
      { key: 'gt', label: 'GT' },
      { key: 'pred', label: 'Pred' },
      { key: 'err', label: 'Err' },
      { key: 'rel', label: 'Err %' },
    ],
  },
  density_regions: {
    sort: 'total',
    columns: [
      { key: 'name', label: 'Image' },
      { key: 'total', label: 'Chickens' },
      { key: 'regions', label: 'Regions' },
      { key: 'residual', label: 'Unassigned' },
    ],
  },
};

/** Rebuild the header when the shape changes, and keep S.sort pointing at a
 *  column that still exists. */
function renderResultHead(kind) {
  const head = $('#result-table thead');
  if (head.dataset.kind === kind) return;
  const table = RESULT_TABLES[kind];
  head.replaceChildren(el('tr', {}, ...table.columns.map(
    (column) => el('th', { dataset: { sort: column.key } }, column.label))));
  head.dataset.kind = kind;
  S.sort = { key: table.sort, dir: -1 };
}

/** Show which column the rows are ordered by, and which way. */
function markSortedColumn() {
  for (const th of $('#result-table thead').querySelectorAll('th')) {
    const active = th.dataset.sort === S.sort.key;
    th.classList.toggle('sorted', active);
    th.classList.toggle('asc', active && S.sort.dir > 0);
  }
}

/** Rows sorted by S.sort; `err` sorts on magnitude, so the worst come first
 *  whichever side of zero they are on. */
function sortedRows(images) {
  markSortedColumn();
  const { key, dir } = S.sort;
  return [...images].sort((a, b) => {
    const av = key === 'err' ? Math.abs(a.err ?? 0) : a[key];
    const bv = key === 'err' ? Math.abs(b.err ?? 0) : b[key];
    return (typeof av === 'string' ? av.localeCompare(bv) : av - bv) * dir;
  });
}

function renderCards(entries) {
  const cards = $('#result-cards');
  cards.replaceChildren();
  for (const [key, value] of entries) {
    if (value === undefined) continue;
    cards.append(el('div', { className: 'card' }, el('div', { className: 'k' }, key), el('div', { className: 'v' }, value)));
  }
}

function renderEvaluation(data) {
  const result = data.result;
  if (!result.images.length && !Object.keys(result.technical).length) return;
  $('#result-panel').hidden = false;
  renderResultHead('test');

  const keys = ['MAE', 'RMSE', 'MAPE', 'Bias (signed)', 'R^2', 'N images'];
  const entries = keys.map((key) => [key, result.technical[key]]);
  const within = result.exhibition['Within +/- N% of GT'];
  if (within) entries.push(['within N% of GT', within]);
  renderCards(entries);

  const body = $('#result-table tbody');
  body.replaceChildren();
  for (const row of sortedRows(result.images)) {
    const large = Math.abs(row.err ?? 0) > Math.max(5, 0.1 * (row.gt ?? 0));
    body.append(el('tr', {},
      el('td', { title: row.name }, row.name),
      el('td', {}, fmtNum(row.gt)),
      el('td', {}, fmtNum(row.pred)),
      el('td', { className: large ? 'bad' : '' }, fmtNum(row.err, true)),
      el('td', { className: large ? 'bad' : '' }, row.rel ?? '—')));
  }
}

function renderRegions(data) {
  const result = data.result;
  if (!result.images.length && !Object.keys(result.technical).length) return;
  $('#result-panel').hidden = false;
  renderResultHead('density_regions');

  renderCards(Object.entries(result.technical));

  const body = $('#result-table tbody');
  body.replaceChildren();
  for (const row of sortedRows(result.images)) {
    // Unassigned mass is the density that no region claimed. A little is
    // normal; a large share means the regions are not describing the frame.
    const leaking = (row.residual ?? 0) > Math.max(2, 0.1 * (row.total ?? 0));
    body.append(el('tr', {},
      el('td', { title: row.name }, row.name),
      el('td', {}, fmtNum(row.total)),
      el('td', {}, String(row.regions ?? '—')),
      el('td', { className: leaking ? 'bad' : '' }, fmtNum(row.residual))));
  }
}

const fmtNum = (v, signed = false) =>
  v == null ? '—' : (signed && v > 0 ? '+' : '') + v.toFixed(1);

// An evaluation overlay is captioned with its error, a region overlay with its
// counts. The item carries whichever the run reported, so branch on that
// rather than on the kind — the gallery is the same widget either way.
const isRegionItem = (item) => item.err == null && item.regions != null;

const galleryCaption = (item) => (isRegionItem(item)
  ? `${fmtNum(item.total)} in ${item.regions} regions`
  : `${fmtNum(item.gt)} → ${fmtNum(item.pred)}  (${item.rel ?? '—'})`);

const lightboxCaption = (item) => (isRegionItem(item)
  ? `${item.name} — ${fmtNum(item.total)} chickens · ${item.regions} regions · ${fmtNum(item.residual)} unassigned`
  : `${item.name} — GT ${fmtNum(item.gt)} · Pred ${fmtNum(item.pred)} · Err ${fmtNum(item.err, true)}`);

async function loadGallery(runId) {
  let data;
  try { data = await api(`/api/runs/${runId}/gallery`); } catch { return; }
  if (runId !== S.runId) return;
  S.galleryFor = runId;
  const gallery = $('#gallery');
  gallery.replaceChildren();
  const order = data.items.some(isRegionItem) ? 'busiest first' : 'worst first';
  $('#gallery-note').textContent = data.dir ? `${data.items.length} overlays · ${order} · ${data.dir}` : 'none written';
  for (const item of data.items) {
    const figure = el('figure', {},
      el('img', { src: `/api/file?path=${encodeURIComponent(item.path)}`, loading: 'lazy', alt: item.name }),
      el('figcaption', {}, galleryCaption(item)));
    figure.addEventListener('click', () => {
      $('#lightbox-img').src = `/api/file?path=${encodeURIComponent(item.path)}`;
      $('#lightbox-cap').textContent = lightboxCaption(item);
      $('#lightbox').hidden = false;
    });
    gallery.append(figure);
  }
}

/* ---------------- wiring ---------------- */

const LAST_TOOL = (page) => `birdcount.webui.tool.${page}`;

async function selectTool(kind) {
  const changed = kind !== S.kind;
  S.kind = kind;
  localStorage.setItem(LAST_TOOL(S.page), kind);
  renderToolPicker();
  if (changed) {
    clearDisplayedRun();
    $('#form').replaceChildren(el('div', { className: 'form-loading' }, 'Loading configuration…'));
  }
  try {
    await loadSchema(kind);
    renderForm();
    if (S.entrypoints.length) {
      renderRunList();
      activatePageRun(S.page);
    }
  } catch (err) {
    $('#blurb').hidden = true;
    $('#form').replaceChildren(el('div', { className: 'empty' }, err.message));
  }
}

const PAGES = () => [...$('#tabs').children].map((tab) => tab.dataset.page);

const PAGE_META = {
  train: ['Model workspace', 'Training', 'Configure a model run and follow metrics as they arrive.'],
  test: ['Evaluation workspace', 'Testing', 'Inspect checkpoint accuracy, per-image error, and density overlays.'],
  annotations: ['Data preparation', 'Annotations', 'Prepare, convert, validate, and organize annotation data.'],
  data: ['Connected service', 'Label Studio', 'Manage the annotation service and its optional public tunnel.'],
};

function updatePageMeta(page) {
  const [kicker, title, description] = PAGE_META[page];
  $('#page-kicker').textContent = kicker;
  $('#page-title').textContent = title;
  $('#page-description').textContent = description;
  document.title = `${title} · bird_count`;
}

function clearDisplayedRun() {
  clearInterval(S.timer);
  S.timer = null;
  S.runId = null;
  S.cursor = 0;
  S.detail = null;
  S.stateVersion = null;
  S.galleryFor = null;
  $('#log').replaceChildren();
  $('#log-path').textContent = '';
  $('#gallery').replaceChildren();
  showEmptyState(true);
  updateGlobalStatus();
}

function activatePageRun(page) {
  const visible = runsForPage(page);
  const currentBelongsHere = S.runId && visible.some((run) => run.id === S.runId);
  if (currentBelongsHere) {
    if (!S.timer) {
      poll();
      if (S.detail?.status === 'running' || S.detail == null) S.timer = setInterval(poll, 900);
    }
    return;
  }
  const preferred = visible.find((run) => run.id === S.activeId)
    ?? visible.find((run) => run.id === S.runByPage[contextKey(page)])
    ?? visible[0];
  if (preferred) selectRun(preferred.id);
  else clearDisplayedRun();
}

async function switchPage(page) {
  S.page = page;
  updatePageMeta(page);
  // Keep the tab in the URL so a reload (or a bookmark) comes back to it.
  if (location.hash.slice(1) !== page) history.replaceState(null, '', `#${page}`);
  for (const tab of $('#tabs').children) tab.classList.toggle('is-active', tab.dataset.page === page);

  // The Data page is not a runnable script: swap the whole run view out and
  // hide the configuration column with it.
  const isData = page === 'data';
  document.body.classList.toggle('page-data', isData);
  $('#data-view').hidden = !isData;
  $('#run-view').hidden = isData;
  $('.col-config').hidden = isData;
  $('.col-runs').hidden = isData;
  $('#btn-start').hidden = isData;
  $('#btn-stop').hidden = isData;
  $('#status-pill').hidden = isData;
  if (isData) {
    clearInterval(S.timer);
    S.timer = null;
    renderRunList();
    return refreshLabelStudio();
  }
  clearTimeout(refreshLabelStudio._t);  // stop probing a panel nobody is looking at

  const tools = S.entrypoints.filter((e) => e.page === page);
  if (!tools.length) return;
  const remembered = localStorage.getItem(LAST_TOOL(page));
  await selectTool(tools.some((t) => t.key === remembered) ? remembered : tools[0].key);
  renderRunList();
  activatePageRun(page);
}

/* ---------------- data page: Label Studio ---------------- */

const LS_PUBLIC_KEY = 'birdcount.webui.ls.public';

const LS_POLL_MS = 5000;   // keep the panel honest while the Data tab is open
const LS_RETRY_MS = 2000;  // the web UI server is down: come back sooner

// One timer drives every re-probe, so they can never stack up.
function scheduleLabelStudioRefresh(delay) {
  clearTimeout(refreshLabelStudio._t);
  if (S.page !== 'data') return;  // nothing to keep fresh off-screen
  refreshLabelStudio._t = setTimeout(() => refreshLabelStudio({ quiet: true }), delay);
}

async function refreshLabelStudio({ quiet = false } = {}) {
  const wantPublic = $('#ls-public').checked;
  const pill = $('#ls-pill');
  if (!quiet) {  // the background re-probe must not flicker the pill
    pill.textContent = 'checking';
    pill.dataset.status = 'idle';
  }

  let info;
  try {
    info = await api(`/api/label-studio?public=${wantPublic}`);
  } catch (err) {
    // The web UI server itself is unreachable — stopped, crashed, or restarting
    // on an edit. Nothing here can be acted on and nothing is known about Label
    // Studio, so settle every control explicitly: leaving them however some
    // half-finished action left them is what makes the panel look frozen.
    pill.textContent = 'no server';
    pill.dataset.status = 'failed';
    $('#ls-detail').textContent = `cannot reach the web UI (${err.message})`;
    for (const id of ['#ls-start', '#ls-stop', '#ng-start', '#ng-stop']) {
      $(id).disabled = true;
      $(id).title = 'the web UI server is not answering';
    }
    $('#ls-open').classList.add('is-disabled');
    $('#ng-inspector').classList.add('is-disabled');
    scheduleLabelStudioRefresh(LS_RETRY_MS);  // heals as soon as it answers again
    return;
  }

  $('#ls-public').disabled = !info.public_url;
  $('#ls-url').textContent = info.url || 'no URL configured';
  $('#ls-open').href = info.url || '#';
  $('#ls-open').classList.toggle('is-disabled', !info.reachable);
  $('#ls-target').textContent = wantPublic
    ? 'Label Studio still runs locally; the tunnel forwards this domain to it.'
    : '';

  // "Reachable" is the truth the user cares about; the child process state only
  // says whether *we* are the ones running it.
  const managed = info.service.started && info.service.state === 'running';
  pill.textContent = info.reachable ? 'running' : managed ? 'starting' : 'not reachable';
  pill.dataset.status = info.reachable ? 'done' : managed ? 'idle' : 'failed';
  $('#ls-detail').textContent = managed ? `${info.detail} · started from this UI` : info.detail;

  // Already answering — whether we started it or the user did from a shell —
  // means starting again would only fail on a taken port. Stop stays ours-only:
  // we can kill the process we own, not one someone else launched.
  $('#ls-start').disabled = managed || info.reachable;
  $('#ls-start').title = !managed && info.reachable ? 'already running (not started from this UI)' : '';
  // Stop works on whatever is serving the port, not only on our own child.
  $('#ls-stop').disabled = !(managed || info.reachable);
  $('#ls-stop').title = !managed && info.reachable ? 'stops the Label Studio serving this port' : '';
  $('#ls-hint').hidden = info.reachable || managed;
  $('#ls-cmd').textContent = info.start_command;

  // ngrok panel
  const tunnel = info.services.ngrok;
  const tunnelManaged = tunnel.started && tunnel.state === 'running';
  const tunnelUp = tunnelManaged || info.ngrok_alive; // an agent we do not own still counts
  $('#ng-cmd').textContent = info.ngrok_command || 'no public domain configured';
  $('#ng-start').disabled = tunnelUp || !info.ngrok_command;
  $('#ng-start').title = tunnelUp && !tunnelManaged ? 'a tunnel is already running (not started from this UI)' : '';
  $('#ng-stop').disabled = !tunnelUp;
  $('#ng-stop').title = tunnelUp && !tunnelManaged ? 'stops the running ngrok agent' : '';
  $('#ng-pill').textContent = tunnelUp ? 'running' : tunnel.started ? tunnel.state : 'stopped';
  $('#ng-pill').dataset.status = tunnelUp ? 'done' : tunnel.started ? 'failed' : 'idle';
  $('#ng-detail').textContent = tunnelManaged
    ? fmtDuration(tunnel.elapsed)
    : tunnelUp ? 'started outside this UI' : '';
  $('#ng-inspector').href = info.ngrok_inspector;
  $('#ng-inspector').classList.toggle('is-disabled', !tunnelUp);
  $('#ls-ngrok').hidden = !wantPublic || tunnelUp;

  for (const name of Object.keys(SERVICES)) {
    if (info.services[name].started) pollServiceLog(name);
  }

  // Re-probe for as long as the panel is on screen. A server stopped or started
  // from a shell must not leave a stale "running" sitting here, and our own
  // process needs a few probes before its port begins to answer.
  scheduleLabelStudioRefresh(managed && !info.reachable ? 3000 : LS_POLL_MS);
}

const SERVICES = {
  label_studio: { log: '#ls-log', path: '#ls-log-path', cursor: 0, timer: null, busy: false },
  ngrok: { log: '#ng-log', path: '#ng-log-path', cursor: 0, timer: null, busy: false },
};

async function pollServiceLog(name) {
  const view = SERVICES[name];
  // This runs on its own timer *and* on every panel refresh; two overlapping
  // fetches would read the same cursor and append the same lines twice.
  if (view.busy) return;
  view.busy = true;
  let data;
  try {
    data = await api(`/api/services/${name}/log?cursor=${view.cursor}`);
  } catch {
    return;
  } finally {
    view.busy = false;
  }
  view.cursor = data.cursor;
  const log = $(view.log);
  for (const line of data.lines) log.append(el('span', {}, line + '\n'));
  if (data.lines.length) log.scrollTop = log.scrollHeight;
  $(view.path).textContent = data.log_path ?? '';

  clearTimeout(view.timer);
  if (data.state === 'running' && S.page === 'data') {
    view.timer = setTimeout(() => pollServiceLog(name), 1200);
  }
}

async function serviceAction(name, action) {
  const buttons = document.querySelectorAll(`#${name === 'ngrok' ? 'ng' : 'ls'}-start, #${name === 'ngrok' ? 'ng' : 'ls'}-stop`);
  buttons.forEach((b) => { b.disabled = true; });
  if (action === 'start') {
    $(SERVICES[name].log).replaceChildren();
    SERVICES[name].cursor = 0;
  }
  try {
    const result = await api(`/api/services/${name}/${action}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ public: $('#ls-public').checked }),
    });
    if (action === 'stop') reportStop(result);
  } catch (err) {
    toast(err.message);
  } finally {
    // The refresh is what re-enables the buttons greyed out above, so it has to
    // run on every path out of here — including a failure.
    await refreshLabelStudio();
  }
}

// Killing a process we did not start deserves an explicit word about what died.
function reportStop(result) {
  if (result.stopped === 'external' && result.killed?.length) {
    toast(`stopped ${result.killed.length} external process (pid ${result.killed.map((p) => p.pid).join(', ')})`);
  } else if (result.refused?.length) {
    toast(`left alone — the port is held by something else: ${result.refused[0].command.slice(0, 90)}`);
  } else if (result.stopped === 'nothing') {
    toast('nothing to stop');
  }
}

$('#ls-start').addEventListener('click', () => serviceAction('label_studio', 'start'));
$('#ls-stop').addEventListener('click', () => serviceAction('label_studio', 'stop'));
$('#ng-start').addEventListener('click', () => serviceAction('ngrok', 'start'));
$('#ng-stop').addEventListener('click', () => serviceAction('ngrok', 'stop'));
$('#ls-public').addEventListener('change', () => {
  localStorage.setItem(LS_PUBLIC_KEY, $('#ls-public').checked ? '1' : '');
  refreshLabelStudio();
});
$('#ls-recheck').addEventListener('click', () => refreshLabelStudio());
$('#ls-copy').addEventListener('click', async () => {
  await navigator.clipboard.writeText($('#ls-url').textContent);
  toast('link copied');
});

$('#tabs').addEventListener('click', (e) => {
  const tab = e.target.closest('.tab');
  if (tab) switchPage(tab.dataset.page);
});
$('#btn-start').addEventListener('click', start);
$('#btn-stop').addEventListener('click', stop);
$('#btn-reset').addEventListener('click', () => {
  S.values[S.kind] = defaultsOf(S.schemas[S.kind]);
  localStorage.removeItem(STORE(S.kind));
  renderForm();
});
$('#btn-clear').addEventListener('click', clearRuns);
$('#btn-quit').addEventListener('click', quitServer);
$('#btn-copy').addEventListener('click', async () => {
  await navigator.clipboard.writeText($('#cmd-preview').textContent);
  toast('command copied');
});
$('#lightbox').addEventListener('click', () => { $('#lightbox').hidden = true; });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') $('#lightbox').hidden = true; });

// Adjustable split between evaluation output and the live log. Pointer drag is
// the primary control; arrow keys keep the separator keyboard-accessible.
const resultLogResizer = $('#result-log-resizer');
const resultPanel = $('#result-panel');
const runView = $('#run-view');
const RESULT_PANE_KEY = 'bird-count-result-pane-height';
const savedResultPaneHeight = Number(localStorage.getItem(RESULT_PANE_KEY));
if (Number.isFinite(savedResultPaneHeight) && savedResultPaneHeight >= 150) {
  runView.style.setProperty('--result-pane-height', `${savedResultPaneHeight}px`);
}

function setResultPaneHeight(height) {
  const runRect = runView.getBoundingClientRect();
  const panelTop = resultPanel.getBoundingClientRect().top;
  const maxHeight = Math.max(150, runRect.bottom - panelTop - 136);
  const next = Math.round(Math.min(Math.max(height, 150), maxHeight));
  runView.style.setProperty('--result-pane-height', `${next}px`);
  localStorage.setItem(RESULT_PANE_KEY, String(next));
  const percent = Math.round((next / Math.max(1, runRect.height)) * 100);
  resultLogResizer.setAttribute('aria-valuenow', String(percent));
}

resultLogResizer.addEventListener('pointerdown', (e) => {
  if (e.button !== 0) return;
  e.preventDefault();
  const startY = e.clientY;
  const startHeight = resultPanel.getBoundingClientRect().height;
  resultLogResizer.classList.add('is-dragging');
  resultLogResizer.setPointerCapture(e.pointerId);
  const move = (event) => setResultPaneHeight(startHeight + event.clientY - startY);
  const stop = () => {
    resultLogResizer.classList.remove('is-dragging');
    resultLogResizer.removeEventListener('pointermove', move);
    resultLogResizer.removeEventListener('pointerup', stop);
    resultLogResizer.removeEventListener('pointercancel', stop);
  };
  resultLogResizer.addEventListener('pointermove', move);
  resultLogResizer.addEventListener('pointerup', stop);
  resultLogResizer.addEventListener('pointercancel', stop);
});
resultLogResizer.addEventListener('keydown', (e) => {
  if (!['ArrowUp', 'ArrowDown', 'Home'].includes(e.key)) return;
  e.preventDefault();
  if (e.key === 'Home') {
    runView.style.removeProperty('--result-pane-height');
    localStorage.removeItem(RESULT_PANE_KEY);
    return;
  }
  const delta = e.key === 'ArrowUp' ? -24 : 24;
  setResultPaneHeight(resultPanel.getBoundingClientRect().height + delta);
});
resultLogResizer.addEventListener('dblclick', () => {
  runView.style.removeProperty('--result-pane-height');
  localStorage.removeItem(RESULT_PANE_KEY);
});
// Delegated: the header row is rebuilt whenever the result shape changes, so
// listeners bound to individual <th> elements would not survive.
$('#result-table thead').addEventListener('click', (e) => {
  const key = e.target.closest('th')?.dataset.sort;
  if (!key) return;
  S.sort = { key, dir: S.sort.key === key ? -S.sort.dir : -1 };
  if (!S.detail) return;
  if (S.detail.kind === 'density_regions') renderRegions(S.detail);
  else renderEvaluation(S.detail);
});
window.addEventListener('resize', () => { if (S.detail) drawChart(S.detail); });
window.addEventListener('error', (e) => toast(`UI error: ${e.message}`));
// Most of this UI is async, so a thrown error surfaces here, not as 'error'.
window.addEventListener('unhandledrejection', (e) => toast(`UI error: ${e.reason?.message ?? e.reason}`));

(async function init() {
  try {
    const [entrypointData, runData] = await Promise.all([api('/api/entrypoints'), api('/api/runs')]);
    S.entrypoints = entrypointData.entrypoints;
    S.runs = runData.runs;
    S.activeId = runData.active;
  } catch (err) {
    toast(`cannot reach the server: ${err.message}`);
    document.body.classList.remove('is-booting');
    return;
  }
  $('#ls-public').checked = !!localStorage.getItem(LS_PUBLIC_KEY);
  const wanted = location.hash.slice(1);
  await switchPage(PAGES().includes(wanted) ? wanted : 'train');
  document.body.classList.remove('is-booting');
  S.listTimer = setInterval(() => { if (!S.timer) refreshRunList(); }, 5000);
})();
