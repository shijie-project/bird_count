/* bird_count web UI — drives train.py / test.py from the browser. */

const S = {
  page: 'train',      // 'train' | 'test' | 'annotations' | 'label_studio'
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
  galleryData: null,  // full overlay metadata for client-side name filtering
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
  // Recordings live on the machine running this server, so listing them beats
  // pushing gigabytes back through the browser. Uploading stays available.
  video: { url: '/api/videos?root=../data', placeholder: 'browse videos…' },
};

// A browser cannot put the real path of a selected local file into a command.
// Upload it to the WebUI server and store the returned project-relative path
// in the normal argparse-backed field instead.
const UPLOAD_FIELDS = {
  ls_import_annotations: {
    src: { accept: '.json,application/json', label: 'choose JSON…', url: '/api/uploads/json', type: 'application/json' },
  },
  density_to_ls_labels: {
    src: { accept: '.json,application/json', label: 'choose regions.json…', url: '/api/uploads/json', type: 'application/json' },
  },
  video_density: {
    video: { accept: 'video/*,.mkv,.ts', label: 'upload…', url: '/api/uploads/video' },
  },
};

/** POST one file to an upload endpoint, reporting progress.
 *  XHR rather than fetch: a recording takes long enough that a button reading
 *  "uploading…" with no number looks like a hang. */
function uploadFile(file, config, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('POST', `${config.url}?filename=${encodeURIComponent(file.name)}`);
    if (config.type) request.setRequestHeader('Content-Type', config.type);
    request.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    });
    request.addEventListener('load', () => {
      let payload = {};
      try { payload = JSON.parse(request.responseText); } catch { /* keep the status text */ }
      if (request.status >= 200 && request.status < 300) resolve(payload);
      else reject(new Error(payload.detail ?? request.statusText ?? `HTTP ${request.status}`));
    });
    request.addEventListener('error', () => reject(new Error('upload failed')));
    request.addEventListener('abort', () => reject(new Error('upload cancelled')));
    request.send(file);
  });
}

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
  const upload = UPLOAD_FIELDS[S.kind]?.[opt.dest];
  const source = PICKERS[opt.dest];
  const controls = [];
  if (source) {
    const picker = el('select', {}, el('option', { value: '' }, source.placeholder));
    picker.addEventListener('change', () => {
      if (!picker.value) return;
      input.value = picker.value;
      setValue(picker.value);
      picker.value = '';
    });
    loadPickerOptions(source, picker);
    controls.push(picker);
  }
  if (upload) {
    const chooser = el('input', { type: 'file', accept: upload.accept, hidden: true });
    const button = el('button', { type: 'button', className: 'btn btn-tiny' }, upload.label);
    button.addEventListener('click', () => chooser.click());
    chooser.addEventListener('change', async () => {
      const file = chooser.files?.[0];
      if (!file) return;
      button.disabled = true;
      button.textContent = 'uploading…';
      try {
        const uploaded = await uploadFile(file, upload, (done) => {
          button.textContent = `uploading ${Math.round(done * 100)}%`;
        });
        input.value = uploaded.path;
        setValue(uploaded.path);
        button.textContent = file.name;
      } catch (err) {
        button.textContent = upload.label;
        toast(err.message);
      } finally {
        button.disabled = false;
        chooser.value = '';
      }
    });
    controls.push(button, chooser);
  }
  if (controls.length) {
    // Three controls do not fit one row at this column width, so a field that
    // offers both a local listing and an upload puts the text input on its own.
    const className = source && upload ? 'with-picker is-triple' : 'with-picker';
    wrap.append(el('div', { className }, input, ...controls));
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
  if (show) {
    $('#result-panel').hidden = true;
    $('#timeline-panel').hidden = true;
  }
  $('#workspace-run').hidden = show;
}

function selectRun(runId) {
  if (!runId || runId === S.runId) return;
  showEmptyState(false);
  S.runId = runId;
  const summary = S.runs.find((run) => run.id === runId);
  const page = summary ? pageOfKind(summary.kind) : S.page;
  S.runByPage[contextKey(page)] = runId;
  S.cursor = 0;
  S.detail = null;
  S.stateVersion = null;
  clearGalleryView();
  resetTimelineView();
  $('#log').replaceChildren();
  $('#result-panel').hidden = true;
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
      else if (data.kind === 'regional_density_error') renderRegionalErrors(data);
      else if (data.kind === 'video_density') renderTimeline(data);
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
const isMultiToolPage = (page = S.page) => S.entrypoints.filter((entry) => entry.page === page).length > 1;
const contextKey = (page = S.page) => isMultiToolPage(page) ? S.kind : page;
const runsForPage = (page = S.page) => S.runs.filter((run) =>
  pageOfKind(run.kind) === page && (!isMultiToolPage(page) || run.kind === S.kind));
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
  $('#runs-title').textContent = `${labelOf(isMultiToolPage() ? S.kind : S.page)} runs`;
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

function stopClientPolling() {
  // The server has ~0.4s left. Stop every poller now so the page does not fill
  // with failed requests while it is down.
  clearInterval(S.timer);
  clearInterval(S.listTimer);
  clearTimeout(refreshLabelStudio._t);
  S.timer = S.listTimer = null;
  for (const view of Object.values(SERVICES)) clearTimeout(view.timer);
}

function disableServerControls() {
  const controls = ['#btn-start', '#btn-stop', '#btn-clear', '#btn-restart', '#btn-quit',
    '#ls-start', '#ls-stop', '#ng-start', '#ng-stop', '#btn-reset'];
  for (const id of controls) $(id).disabled = true;
}

// Shuts down the server this page is talking to, which frees its port.
async function quitServer() {
  const extra = [];
  if (S.activeId) extra.push('the run in progress');
  if (!confirm(`Shut down the web UI server and release port ${location.port || 80}?`
    + (extra.length ? `\n\nThis also stops ${extra.join(' and ')}, plus any Label Studio or tunnel started from here.` : ''))) return;

  let result;
  try {
    result = await api('/api/shutdown', { method: 'POST' });
  } catch (err) {
    return toast(err.message);
  }

  stopClientPolling();
  disableServerControls();
  $('#status-pill').hidden = false;
  $('#status-pill').textContent = 'server stopped';
  $('#status-pill').dataset.status = 'stopped';

  const also = [result.run && 'the active run', ...(result.services || [])].filter(Boolean);
  toast(`server stopped — port released${also.length ? ` · also stopped: ${also.join(', ')}` : ''}`);
}

const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function restartServer() {
  const warning = S.activeId
    ? '\n\nThis stops the run in progress, plus any Label Studio or tunnel started from here.'
    : '\n\nAny Label Studio or tunnel started from here will also stop.';
  if (!confirm(`Restart the web UI server on port ${location.port || 80}?${warning}`)) return;

  let result;
  try {
    result = await api('/api/restart', { method: 'POST' });
  } catch (err) {
    return toast(err.message);
  }

  stopClientPolling();
  disableServerControls();
  $('#status-pill').hidden = false;
  $('#status-pill').textContent = 'restarting';
  $('#status-pill').dataset.status = 'running';
  toast('server restarting — this page will reconnect automatically');

  // A successful response from the old process is not enough: wait until the
  // health endpoint reports a different instance id, then reload this tab so
  // it also picks up newly edited static assets.
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    await delay(250);
    try {
      const response = await fetch(`/api/health?_=${Date.now()}`, { cache: 'no-store' });
      if (!response.ok) continue;
      const health = await response.json();
      if (health.instance !== result.instance) {
        window.location.reload();
        return;
      }
    } catch { /* the short offline window is expected */ }
  }

  $('#status-pill').textContent = 'restart failed';
  $('#status-pill').dataset.status = 'failed';
  toast('server did not come back within 60 seconds — check the launch terminal');
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
  clearGalleryView();
  resetTimelineView();
  $('#log').replaceChildren();
  $('#log-path').textContent = '';
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
    { id: 'train_mae', label: 'train MAE', color: '#2563eb', on: true, from: (m) => m.train.map((r) => [r.epoch, r.mae]) },
    { id: 'val_mae', label: 'val MAE', color: '#16834d', on: true, from: (m) => m.val.map((r) => [r.epoch, r.mae]) },
    { id: 'train_mse', label: 'train MSE', color: '#7c3aed', on: false, from: (m) => m.train.map((r) => [r.epoch, r.mse]) },
    { id: 'val_mse', label: 'val MSE', color: '#a56408', on: false, from: (m) => m.val.map((r) => [r.epoch, r.mse]) },
    { id: 'loss', label: 'loss', color: '#c93632', on: false, from: (m) => m.train.map((r) => [r.epoch, r.loss]) },
  ],
  test: [
    { id: 'scatter', label: 'GT vs Pred', color: '#2563eb', on: true, scatter: true,
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
  ctx.strokeStyle = token('--chart-grid', '#e5eaf1');
  ctx.fillStyle = token('--chart-tick', '#64748b');
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
    ctx.strokeStyle = token('--chart-guide', '#b8c3d1');
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

/* ---------------- video density timeline ---------------- */

// Two readings of the same density map, never drawn on one axis: "birds in the
// frame" and "birds inside one small patch" differ by an order of magnitude,
// and overlaying them flattens the one that carries the pile-up signal.
const TIMELINE_METRICS = {
  peak: {
    label: 'Max local flock count',
    color: '#c2410c',
    digits: 2,
    unit: (meta) => `birds in one ${meta?.window_px ?? 64}×${meta?.window_px ?? 64} px patch`,
  },
  count: {
    label: 'Global flock count',
    color: '#2563eb',
    digits: 0,
    unit: () => 'birds in frame',
  },
};

const TIMELINE_METRIC_KEY = 'birdcount.webui.timeline.metric';
const TIMELINE_SMOOTH_KEY = 'birdcount.webui.timeline.smooth';
// Tick spacings that read as time rather than as arbitrary numbers of seconds.
const TIME_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400, 21600];
const TIMELINE_HEIGHT = 300;

const TL = {
  data: null,        // {meta, summary, samples, artifacts} from the run
  metric: localStorage.getItem(TIMELINE_METRIC_KEY) ?? 'peak',
  smooth: localStorage.getItem(TIMELINE_SMOOTH_KEY) !== '0',
  hover: null,       // index of the sample under the pointer
  points: [],        // plotted positions, kept for hit-testing
  peaksKey: '',      // artifact signature, so thumbnails are not rebuilt on every poll
  frame: null,       // pending animation frame for a redraw
};

const pad2 = (value) => String(value).padStart(2, '0');

const timelineStart = (meta) => {
  const parsed = meta?.start_time ? new Date(meta.start_time.replace(' ', 'T')) : null;
  return parsed && !Number.isNaN(parsed.getTime()) ? parsed : null;
};

/** Label for a moment in the video: wall clock when the run was told when the
 *  recording started, elapsed time otherwise. */
function timelineClock(meta, seconds, withSeconds = true) {
  const start = timelineStart(meta);
  if (start) {
    const at = new Date(start.getTime() + seconds * 1000);
    return `${pad2(at.getHours())}:${pad2(at.getMinutes())}${withSeconds ? `:${pad2(at.getSeconds())}` : ''}`;
  }
  const total = Math.max(0, Math.round(seconds));
  const minutes = pad2(Math.floor((total % 3600) / 60));
  return `${Math.floor(total / 3600)}:${minutes}${withSeconds ? `:${pad2(total % 60)}` : ''}`;
}

/** Centered rolling mean; the window shrinks at the edges so the smoothed line
 *  spans the whole axis instead of stopping short of it. */
function rollingMean(values, size) {
  if (!(size > 1) || values.length < 3) return values.slice();
  const half = Math.floor(size / 2);
  return values.map((_, index) => {
    const from = Math.max(0, index - half);
    const to = Math.min(values.length, index + half + 1);
    let total = 0;
    for (let i = from; i < to; i++) total += values[i];
    return total / (to - from);
  });
}

function withAlpha(hex, alpha) {
  const value = hex.replace('#', '');
  const full = value.length === 3 ? value.split('').map((c) => c + c).join('') : value;
  const number = Number.parseInt(full, 16);
  return `rgba(${(number >> 16) & 255}, ${(number >> 8) & 255}, ${number & 255}, ${alpha})`;
}

/** A grid step that lands on 1 / 2 / 2.5 / 5 × 10^k, so tick labels stay round. */
function niceStep(range, target = 4) {
  const raw = Math.max(range, 1e-9) / target;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / magnitude;
  const step = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10;
  return step * magnitude;
}

function timelineStats(samples) {
  const stats = {};
  for (const key of ['peak', 'count']) {
    let max = -Infinity;
    let best = 0;
    let total = 0;
    samples.forEach((sample, index) => {
      const value = sample[key] ?? 0;
      total += value;
      if (value > max) { max = value; best = index; }
    });
    stats[key] = { max, mean: total / samples.length, at: samples[best] };
  }
  return stats;
}

/** The series this run reports (its --report flag), and the one to draw.
 *  A run that reported only one of them must not leave the page showing an
 *  empty chart because the other one is what the viewer looked at last. */
const reportedMetrics = () => {
  const report = TL.data?.meta?.report ?? 'both';
  return report === 'both' ? Object.keys(TIMELINE_METRICS) : [report];
};

const activeMetric = () => {
  const available = reportedMetrics();
  return available.includes(TL.metric) ? TL.metric : available[0];
};

function resetTimelineView() {
  TL.data = null;
  TL.hover = null;
  TL.points = [];
  TL.peaksKey = '';
  $('#timeline-panel').hidden = true;
  $('#timeline-peaks').replaceChildren();
  $('#timeline-cards').replaceChildren();
  $('#timeline-tip').hidden = true;
  for (const id of ['#timeline-figure', '#timeline-csv']) $(id).disabled = true;
}

function renderMetricSwitch() {
  const host = $('#timeline-metrics');
  if (!host.childElementCount) {
    for (const [key, metric] of Object.entries(TIMELINE_METRICS)) {
      const button = el('button', { type: 'button', dataset: { metric: key } },
        el('span', { className: 'swatch', style: `background:${metric.color}` }), metric.label);
      button.addEventListener('click', () => {
        TL.metric = key;
        localStorage.setItem(TIMELINE_METRIC_KEY, key);
        renderMetricSwitch();
        drawTimeline();
      });
      host.append(button);
    }
  }
  const available = reportedMetrics();
  const shown = activeMetric();
  for (const button of host.children) {
    button.hidden = !available.includes(button.dataset.metric);
    button.classList.toggle('on', button.dataset.metric === shown);
  }
  // One series reported: the switch has nothing to switch between, so the
  // panel head says which one it is instead of offering a dead control.
  host.classList.toggle('is-single', available.length < 2);
}

function renderTimelineCards(timeline) {
  const { meta, samples } = timeline;
  const stats = timelineStats(samples);
  const span = `${timelineClock(meta, samples[0].t)} – ${timelineClock(meta, samples[samples.length - 1].t)}`;
  // Cards follow the run's --report: asking for the whole-frame density only
  // and still being given three local-density numbers is just noise.
  const available = reportedMetrics();
  const entries = [];
  if (available.includes('peak')) {
    entries.push(
      ['Peak density', stats.peak.max.toFixed(2)],
      ['Peak at', timelineClock(meta, stats.peak.at.t)],
      ['Mean density', stats.peak.mean.toFixed(2)]);
  }
  if (available.includes('count')) {
    entries.push(['Peak count', stats.count.max.toFixed(0)]);
    if (!available.includes('peak')) entries.push(['Peak at', timelineClock(meta, stats.count.at.t)]);
    entries.push(['Mean count', stats.count.mean.toFixed(0)]);
  }
  entries.push(
    ['Samples', `${samples.length}${meta?.interval ? ` · ${meta.interval}s` : ''}`],
    [timelineStart(meta) ? 'Clock span' : 'Elapsed', span]);
  const cards = $('#timeline-cards');
  cards.replaceChildren();
  for (const [key, value] of entries) {
    cards.append(el('div', { className: 'card' }, el('div', { className: 'k' }, key), el('div', { className: 'v' }, value)));
  }
}

function renderTimelinePeaks(timeline) {
  const frames = (timeline.artifacts ?? []).filter((artifact) => artifact.kind === 'frame');
  const signature = frames.map((frame) => frame.path).join('|');
  if (signature === TL.peaksKey) return;
  TL.peaksKey = signature;

  const head = $('#timeline-peaks-head');
  const gallery = $('#timeline-peaks');
  head.hidden = gallery.hidden = !frames.length;
  gallery.replaceChildren();
  if (!frames.length) return;

  $('#timeline-peaks-note').textContent = `${frames.length} density overlays at the busiest moments`;
  for (const frame of frames) {
    const source = `/api/file?path=${encodeURIComponent(frame.path)}`;
    const caption = `${frame.clock} · peak ${Number(frame.peak).toFixed(1)} · count ${Number(frame.count).toFixed(0)}`;
    const figure = el('figure', {},
      el('img', { src: source, loading: 'lazy', alt: caption }),
      el('figcaption', {},
        el('span', { className: 'gallery-name' }, frame.clock),
        el('span', { className: 'gallery-stats' }, caption)));
    figure.addEventListener('click', () => {
      $('#lightbox-img').src = source;
      $('#lightbox-cap').textContent = `${timeline.meta?.name ?? 'video'} — ${caption}`;
      $('#lightbox').hidden = false;
    });
    gallery.append(figure);
  }
}

/** Path of the frame saved for one sample, when the run saved any.
 *  Mirrors `frame_filename()` in webui/ops/video_density_timeline.py — the two
 *  spellings have to stay in step, which is why neither invents anything: the
 *  sample index and its whole second are both already in the payload. */
function framePath(sample) {
  const dir = TL.data?.meta?.frames_dir;
  if (!dir || !sample) return null;
  const index = String(sample.i).padStart(5, '0');
  const seconds = String(Math.floor(sample.t)).padStart(6, '0');
  return `${dir}/frame_${index}_${seconds}s.jpg`;
}

function openFrame(sample) {
  const path = framePath(sample);
  if (!path) return;
  const meta = TL.data?.meta ?? {};
  $('#lightbox-img').src = `/api/file?path=${encodeURIComponent(path)}`;
  $('#lightbox-cap').textContent =
    `${meta.name ?? 'video'} — ${timelineClock(meta, sample.t)} · `
    + `count ${(sample.count ?? 0).toFixed(0)} · peak ${(sample.peak ?? 0).toFixed(2)}`;
  $('#lightbox').hidden = false;
}

const timelineArtifact = (kind) => (TL.data?.artifacts ?? []).find((artifact) => artifact.kind === kind);

function renderTimeline(data) {
  const timeline = data.result?.timeline;
  if (!timeline) return;
  TL.data = timeline;
  $('#timeline-panel').hidden = false;
  renderMetricSwitch();
  $('#timeline-smooth').checked = TL.smooth;
  $('#timeline-figure').disabled = !timelineArtifact('chart');
  $('#timeline-csv').disabled = !timelineArtifact('csv');

  const { meta, samples } = timeline;
  const running = data.status === 'running';
  const canvas = $('#timeline-chart');
  if (!samples.length) {
    $('#timeline-sub').textContent = running
      ? 'Reading the video — the chart fills in as frames are sampled.'
      : 'No samples were produced. Check the log below.';
    canvas.hidden = true;
    return;
  }
  canvas.hidden = false;
  const source = meta?.name ? `${meta.name} · ${meta.infer_width}×${meta.infer_height}` : 'video';
  const saved = meta?.frames_saved
    ? ` · ${meta.frames_saved} ${meta.frames_style ?? ''} frames saved — click a point to open one`
    : '';
  $('#timeline-sub').textContent =
    `${source} · ${samples.length} samples${meta?.interval ? ` every ${meta.interval}s` : ''}` +
    `${meta?.window ? ` · window ${meta.window}×${meta.window} cells (${meta.window_px}px)` : ''}` +
    `${saved}${running ? ' · sampling…' : ''}`;

  renderTimelineCards(timeline);
  renderTimelinePeaks(timeline);
  drawTimeline();
}

function requestTimelineDraw() {
  if (TL.frame) return;
  TL.frame = requestAnimationFrame(() => { TL.frame = null; drawTimeline(); });
}

function drawTimeline() {
  const timeline = TL.data;
  const samples = timeline?.samples ?? [];
  const canvas = $('#timeline-chart');
  if (!samples.length || $('#timeline-panel').hidden || canvas.hidden) return;

  const meta = timeline.meta ?? {};
  const key = activeMetric();
  const metric = TIMELINE_METRICS[key] ?? TIMELINE_METRICS.peak;
  const values = samples.map((sample) => sample[key] ?? 0);
  const smoothed = rollingMean(values, TL.smooth ? Math.max(Number(meta.smooth) || 5, 3) : 0);
  // The threshold belongs to one series only (a count of 150 in the frame is
  // not the same event as 150 inside one patch), so it is drawn only on it.
  const threshold = (meta.threshold_metric ?? 'peak') === key ? Number(meta.threshold) || 0 : 0;

  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(canvas.parentElement.clientWidth - 16, 280);
  const height = TIMELINE_HEIGHT;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  // Frames saved for every sample make the chart itself clickable.
  canvas.style.cursor = meta.frames_dir ? 'pointer' : 'default';
  // Painted rather than left transparent, so "Save chart" produces a picture
  // that is readable outside this page too.
  ctx.fillStyle = token('--surface', '#ffffff');
  ctx.fillRect(0, 0, width, height);

  const pad = { l: 58, r: 16, t: 24, b: 32 };
  const plotW = width - pad.l - pad.r;
  const plotH = height - pad.t - pad.b;

  const t0 = samples[0].t;
  const t1 = Math.max(samples[samples.length - 1].t, t0 + 1);
  const highest = Math.max(...values, threshold, 0.001);
  const step = niceStep(highest * 1.1);
  const yMax = Math.max(Math.ceil((highest * 1.08) / step) * step, step);

  const X = (t) => pad.l + ((t - t0) / (t1 - t0)) * plotW;
  const Y = (v) => pad.t + plotH - (v / yMax) * plotH;

  ctx.font = '10px ui-monospace, monospace';
  ctx.lineWidth = 1;

  // horizontal grid + value ticks
  ctx.strokeStyle = token('--chart-grid', '#e5eaf1');
  ctx.fillStyle = token('--chart-tick', '#64748b');
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let value = 0; value <= yMax + 1e-9; value += step) {
    const y = Y(value);
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(width - pad.r, y);
    ctx.stroke();
    ctx.fillText(step >= 1 ? String(Math.round(value)) : value.toFixed(2), pad.l - 9, y);
  }

  // time ticks, aligned to the clock so labels land on round minutes
  const span = t1 - t0;
  const timeStep = TIME_STEPS.find((candidate) => span / candidate <= 7) ?? Math.ceil(span / 7);
  const start = timelineStart(meta);
  const offset = start ? start.getHours() * 3600 + start.getMinutes() * 60 + start.getSeconds() : 0;
  const first = Math.ceil((t0 + offset) / timeStep) * timeStep - offset;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let t = first; t <= t1; t += timeStep) {
    if (t < t0) continue;
    ctx.strokeStyle = token('--chart-grid', '#e5eaf1');
    ctx.beginPath();
    ctx.moveTo(X(t), pad.t);
    ctx.lineTo(X(t), pad.t + plotH);
    ctx.stroke();
    ctx.fillStyle = token('--chart-tick', '#64748b');
    ctx.fillText(timelineClock(meta, t, timeStep < 60), X(t), height - pad.b + 7);
  }

  // alert threshold: a line plus a wash over everything above it
  if (threshold > 0 && threshold < yMax) {
    ctx.fillStyle = withAlpha('#c93632', 0.06);
    ctx.fillRect(pad.l, pad.t, plotW, Y(threshold) - pad.t);
    ctx.strokeStyle = withAlpha('#c93632', 0.75);
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.l, Y(threshold));
    ctx.lineTo(width - pad.r, Y(threshold));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = withAlpha('#c93632', 0.9);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`threshold ${threshold}`, pad.l + 6, Y(threshold) - 3);
  }

  const points = samples.map((sample, index) => ({ x: X(sample.t), y: Y(values[index]), sample, index }));
  const line = smoothed.map((value, index) => ({ x: points[index].x, y: Y(value) }));
  TL.points = points;

  // area under the curve, fading out downwards
  const gradient = ctx.createLinearGradient(0, pad.t, 0, pad.t + plotH);
  gradient.addColorStop(0, withAlpha(metric.color, 0.3));
  gradient.addColorStop(1, withAlpha(metric.color, 0.02));
  ctx.fillStyle = gradient;
  ctx.beginPath();
  tracePath(ctx, TL.smooth ? line : points);
  ctx.lineTo(points[points.length - 1].x, pad.t + plotH);
  ctx.lineTo(points[0].x, pad.t + plotH);
  ctx.closePath();
  ctx.fill();

  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  if (TL.smooth) {  // the raw series stays visible underneath the smoothed one
    ctx.strokeStyle = withAlpha(metric.color, 0.32);
    ctx.lineWidth = 1;
    ctx.beginPath();
    tracePath(ctx, points);
    ctx.stroke();
  }
  ctx.strokeStyle = metric.color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  tracePath(ctx, TL.smooth ? line : points);
  ctx.stroke();

  // the maximum, named: it is the number the eye is looking for
  const peakIndex = values.indexOf(Math.max(...values));
  const peak = points[peakIndex];
  ctx.fillStyle = metric.color;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(peak.x, peak.y, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.font = '600 10.5px ui-monospace, monospace';
  ctx.fillStyle = metric.color;
  ctx.textBaseline = 'bottom';
  ctx.textAlign = peak.x > pad.l + plotW * 0.8 ? 'right' : peak.x < pad.l + plotW * 0.2 ? 'left' : 'center';
  ctx.fillText(`${values[peakIndex].toFixed(metric.digits)} @ ${timelineClock(meta, peak.sample.t)}`,
    peak.x, Math.max(peak.y - 9, pad.t + 10));

  // axis titles
  ctx.font = '10px ui-monospace, monospace';
  ctx.fillStyle = token('--chart-tick', '#64748b');
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(`${metric.label} — ${metric.unit(meta)}`, pad.l, 6);
  ctx.textAlign = 'right';
  ctx.fillText(start ? 'clock time' : 'elapsed', width - pad.r, height - 12);

  drawTimelineHover(ctx, pad, plotH, metric, meta, smoothed);
}

/** Straight segments through `points`, rounded at the joins. Midpoint
 *  quadratics smooth the corners without the overshoot a spline would add. */
function tracePath(ctx, points) {
  if (!points.length) return;
  ctx.moveTo(points[0].x, points[0].y);
  if (points.length < 3) {
    for (const point of points.slice(1)) ctx.lineTo(point.x, point.y);
    return;
  }
  for (let i = 1; i < points.length - 1; i++) {
    const midX = (points[i].x + points[i + 1].x) / 2;
    const midY = (points[i].y + points[i + 1].y) / 2;
    ctx.quadraticCurveTo(points[i].x, points[i].y, midX, midY);
  }
  const last = points[points.length - 1];
  ctx.lineTo(last.x, last.y);
}

/** Crosshair, marker and tooltip for the sample under the pointer. The tooltip
 *  shows both metrics, so hovering answers "and what was the count then?". */
function drawTimelineHover(ctx, pad, plotH, metric, meta, smoothed) {
  const tip = $('#timeline-tip');
  const index = TL.hover;
  const point = index == null ? null : TL.points[index];
  if (!point) {
    tip.hidden = true;
    return;
  }
  ctx.strokeStyle = token('--chart-guide', '#b8c3d1');
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(point.x, pad.t);
  ctx.lineTo(point.x, pad.t + plotH);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = metric.color;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(point.x, point.y, 3.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  const sample = point.sample;
  const canvas = $('#timeline-chart');
  const rows = reportedMetrics().map((name) => el('div', { className: 't-row' },
    el('span', { className: 'swatch', style: `background:${TIMELINE_METRICS[name].color}` }), name,
    el('b', {}, (sample[name] ?? 0).toFixed(TIMELINE_METRICS[name].digits))));
  tip.replaceChildren(
    el('div', { className: 't-time' }, timelineClock(meta, sample.t)),
    ...rows,
    ...(TL.smooth ? [el('div', { className: 't-row' },
      el('span', { className: 'swatch', style: 'background:#94a3b8' }), 'smoothed',
      el('b', {}, smoothed[index].toFixed(metric.digits)))] : []),
    ...(framePath(sample) ? [el('div', { className: 't-hint' }, 'click to open this frame')] : []),
  );
  tip.hidden = false;
  const half = tip.offsetWidth / 2;
  const left = canvas.offsetLeft + point.x;
  const limit = canvas.offsetLeft + canvas.clientWidth - half - 4;
  tip.style.left = `${Math.min(Math.max(left, canvas.offsetLeft + half + 4), limit)}px`;
  tip.style.top = `${canvas.offsetTop + Math.max(point.y - 14, 40)}px`;
}

function downloadUrl(url, filename) {
  const link = el('a', { href: url, download: filename });
  document.body.append(link);
  link.click();
  link.remove();
}

$('#timeline-chart').addEventListener('pointermove', (event) => {
  if (!TL.points.length) return;
  const x = event.clientX - event.currentTarget.getBoundingClientRect().left;
  let best = 0;
  let bestDistance = Infinity;
  for (const point of TL.points) {
    const distance = Math.abs(point.x - x);
    if (distance < bestDistance) { bestDistance = distance; best = point.index; }
  }
  if (best !== TL.hover) { TL.hover = best; requestTimelineDraw(); }
});
$('#timeline-chart').addEventListener('click', () => {
  if (TL.hover == null) return;
  openFrame(TL.points[TL.hover]?.sample);
});
$('#timeline-chart').addEventListener('pointerleave', () => {
  if (TL.hover == null) return;
  TL.hover = null;
  requestTimelineDraw();
});
$('#timeline-smooth').addEventListener('change', (event) => {
  TL.smooth = event.currentTarget.checked;
  localStorage.setItem(TIMELINE_SMOOTH_KEY, TL.smooth ? '1' : '0');
  drawTimeline();
});
$('#timeline-figure').addEventListener('click', () => {
  const chart = timelineArtifact('chart');
  if (!chart) return;
  $('#lightbox-img').src = `/api/file?path=${encodeURIComponent(chart.path)}`;
  $('#lightbox-cap').textContent = chart.path;
  $('#lightbox').hidden = false;
});
$('#timeline-csv').addEventListener('click', () => {
  const csv = timelineArtifact('csv');
  if (!csv) return;
  const name = `${TL.data?.meta?.name ?? 'video'}-timeline.csv`.replace(/[^\w.-]+/g, '_');
  downloadUrl(`/api/file?path=${encodeURIComponent(csv.path)}`, name);
});
$('#timeline-png').addEventListener('click', () => {
  const canvas = $('#timeline-chart');
  if (canvas.hidden || !TL.points.length) return toast('nothing to save yet');
  const name = `${TL.data?.meta?.name ?? 'video'}-${activeMetric()}.png`.replace(/[^\w.-]+/g, '_');
  downloadUrl(canvas.toDataURL('image/png'), name);
});

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
    sort: 'err',
    columns: [
      { key: 'name', label: 'Image / blob' },
      { key: 'gt', label: 'GT' },
      { key: 'pred', label: 'Pred' },
      { key: 'err', label: 'Err' },
      { key: 'rel', label: 'Err %' },
    ],
  },
  regional_density_error: {
    sort: 'err',
    columns: [
      { key: 'name', label: 'Image / region' },
      { key: 'gt', label: 'GT' },
      { key: 'pred', label: 'Pred' },
      { key: 'err', label: 'Err' },
      { key: 'rel', label: 'Err %' },
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

/** Rows sorted by S.sort; error columns sort on magnitude, so the worst come
 *  first whichever side of zero they are on. Relative errors arrive formatted
 *  for display (for example "+12.3%"), so convert them back to numbers instead
 *  of sorting those strings lexicographically. */
function sortedRows(images) {
  markSortedColumn();
  const { key, dir } = S.sort;
  return [...images].sort((a, b) => {
    const value = (row) => {
      if (key === 'err') return Math.abs(row.err ?? 0);
      if (key === 'rel') {
        const relative = Number.parseFloat(row.rel);
        return Number.isFinite(relative) ? Math.abs(relative) : null;
      }
      return row[key];
    };
    const av = value(a);
    const bv = value(b);
    // A zero-GT row has relative error "n/a". It is not comparable with a
    // percentage and should stay below the numeric rows in either direction.
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
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
  if (!result.regions.length && !Object.keys(result.technical).length) return;
  $('#result-panel').hidden = false;
  renderResultHead('density_regions');

  renderCards(Object.entries(result.technical));

  const body = $('#result-table tbody');
  body.replaceChildren();
  for (const row of sortedRows(result.regions)) {
    body.append(el('tr', {},
      el('td', { title: row.name }, row.name),
      el('td', {}, fmtNum(row.gt)),
      el('td', {}, fmtNum(row.pred)),
      el('td', { className: Math.abs(row.err ?? 0) > 1 ? 'bad' : '' }, fmtNum(row.err, true)),
      el('td', {}, row.rel ?? '—')));
  }
}

function renderRegionalErrors(data) {
  const result = data.result;
  if (!result.regions.length && !Object.keys(result.technical).length) return;
  $('#result-panel').hidden = false;
  renderResultHead('regional_density_error');
  renderCards(Object.entries(result.technical));

  const body = $('#result-table tbody');
  body.replaceChildren();
  for (const row of sortedRows(result.regions)) {
    body.append(el('tr', {},
      el('td', { title: row.name }, row.name),
      el('td', {}, fmtNum(row.gt)),
      el('td', {}, fmtNum(row.pred)),
      el('td', { className: Math.abs(row.err ?? 0) > 1 ? 'bad' : '' }, fmtNum(row.err, true)),
      el('td', {}, row.rel ?? '—')));
  }
}

const fmtNum = (v, signed = false) =>
  v == null ? '—' : (signed && v > 0 ? '+' : '') + v.toFixed(1);

// An evaluation overlay is captioned with its error, a region overlay with its
// counts. The item carries whichever the run reported, so branch on that
// rather than on the kind — the gallery is the same widget either way.
const isRegionItem = (item) => item.err == null && item.regions != null;
const isRegionalErrorItem = (item) => item.region_mae != null;
const isBlobErrorItem = (item) => item.blob_mae != null;

const galleryCaption = (item) => (isBlobErrorItem(item)
  ? `blob MAE ${fmtNum(item.blob_mae)} · worst ${fmtNum(item.worst_blob)} · ${item.gt_outside} GT outside`
  : isRegionalErrorItem(item)
  ? `region MAE ${fmtNum(item.region_mae)} · worst ${fmtNum(item.worst_region)}`
  : isRegionItem(item)
  ? `${fmtNum(item.total)} in ${item.regions} regions`
  : `${fmtNum(item.gt)} → ${fmtNum(item.pred)}  (${item.rel ?? '—'})`);

const lightboxCaption = (item) => (isBlobErrorItem(item)
  ? `${item.name} — blob MAE ${fmtNum(item.blob_mae)} · worst ${fmtNum(item.worst_blob)} · ${item.gt_outside} GT outside blobs`
  : isRegionalErrorItem(item)
  ? `${item.name} — region MAE ${fmtNum(item.region_mae)} · worst region ${fmtNum(item.worst_region)}`
  : isRegionItem(item)
  ? `${item.name} — ${fmtNum(item.total)} chickens · ${item.regions} regions · ${fmtNum(item.residual)} unassigned`
  : `${item.name} — GT ${fmtNum(item.gt)} · Pred ${fmtNum(item.pred)} · Err ${fmtNum(item.err, true)}`);

// Keep enough metadata locally that searching does not refetch the gallery on
// every keystroke. Only the first 500 matches are rendered, while the larger
// fetch lets a name search reach beyond the old 500-overlay gallery cutoff.
const GALLERY_FETCH_LIMIT = 5000;
const GALLERY_RENDER_LIMIT = 500;

function clearGalleryView() {
  S.galleryFor = null;
  S.galleryData = null;
  $('#gallery').replaceChildren();
  $('#gallery-note').textContent = '';
  $('#gallery-note').title = '';
  $('#gallery-search').disabled = true;
  $('#gallery-search-clear').hidden = !$('#gallery-search').value;
}

function gallerySearchTerms(raw) {
  let query = raw.trim().toLocaleLowerCase().replaceAll('\\', '/').split('/').pop();
  // Accept an original image path or an emitted overlay filename as well as
  // the bare image name shown in the evaluation table.
  query = query
    .replace(/_(?:density|regions|regional_error)\.png$/i, '')
    .replace(/\.(?:png|jpe?g|bmp|tiff?|webp)$/i, '');
  return query.split(/\s+/).filter(Boolean);
}

function renderGallery() {
  const data = S.galleryData;
  if (!data) return;

  const gallery = $('#gallery');
  const search = $('#gallery-search');
  const clear = $('#gallery-search-clear');
  const terms = gallerySearchTerms(search.value);
  const matches = terms.length
    ? data.items.filter((item) => {
      const name = String(item.name ?? '').toLocaleLowerCase();
      return terms.every((term) => name.includes(term));
    })
    : data.items;
  const shown = matches.slice(0, GALLERY_RENDER_LIMIT);

  clear.hidden = !search.value;
  gallery.replaceChildren();
  const order = data.order === 'name'
    ? 'name order'
    : data.items.some(isRegionItem) ? 'busiest first' : 'worst first';
  const shownCount = shown.length < matches.length ? `${shown.length} of ${matches.length}` : String(matches.length);
  const countText = terms.length ? `${shownCount} matches` : `${shownCount} overlays`;
  $('#gallery-note').textContent = data.dir ? `${countText} · ${order} · ${data.dir}` : 'none written';
  $('#gallery-note').title = data.dir || '';

  if (!shown.length) {
    const message = data.items.length && terms.length
      ? `No overlays match "${search.value.trim()}".`
      : 'No density overlays were written for this run.';
    gallery.append(el('div', { className: 'gallery-empty' }, message));
    return;
  }

  for (const item of shown) {
    const figure = el('figure', {},
      el('img', { src: `/api/file?path=${encodeURIComponent(item.path)}`, loading: 'lazy', alt: item.name }),
      el('figcaption', {},
        el('span', { className: 'gallery-name', title: item.name }, item.name),
        el('span', { className: 'gallery-stats' }, galleryCaption(item))));
    figure.addEventListener('click', () => {
      $('#lightbox-img').src = `/api/file?path=${encodeURIComponent(item.path)}`;
      $('#lightbox-cap').textContent = lightboxCaption(item);
      $('#lightbox').hidden = false;
    });
    gallery.append(figure);
  }
}

async function loadGallery(runId) {
  let data;
  try { data = await api(`/api/runs/${runId}/gallery?limit=${GALLERY_FETCH_LIMIT}`); } catch { return; }
  if (runId !== S.runId) return;
  S.galleryFor = runId;
  S.galleryData = data;
  $('#gallery-search').disabled = !data.dir || !data.items.length;
  renderGallery();
}

/* ---------------- wiring ---------------- */

const LAST_TOOL = (page) => `birdcount.webui.tool.${page}`;

async function selectTool(kind) {
  const changed = kind !== S.kind;
  S.kind = kind;
  updatePageMeta(S.page, kind);
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
  video: ['Video analysis', 'Video', 'Run the model over a recording and see how crowding changes over time.'],
  annotations: ['Data preparation', 'Annotations', 'Prepare, convert, validate, and organize annotation data.'],
  label_studio: ['Annotation workspace', 'Label Studio', 'Manage the service, projects, and Label Studio operations in one place.'],
};

function updatePageMeta(page, kind = null) {
  const [kicker, fallbackTitle, description] = PAGE_META[page];
  const tool = kind && S.entrypoints.find((entry) => entry.key === kind && entry.page === page);
  const title = tool?.label ?? fallbackTitle;
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
  clearGalleryView();
  resetTimelineView();
  $('#log').replaceChildren();
  $('#log-path').textContent = '';
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

  const isLabelStudio = page === 'label_studio';
  $('#data-view').hidden = !isLabelStudio;
  $('#run-view').hidden = false;
  $('.col-config').hidden = false;
  $('.col-runs').hidden = false;
  $('#btn-start').hidden = false;
  $('#btn-stop').hidden = false;
  $('#status-pill').hidden = false;
  if (isLabelStudio) refreshLabelStudio();
  else clearTimeout(refreshLabelStudio._t);  // stop probing a panel nobody is looking at

  const tools = S.entrypoints.filter((e) => e.page === page);
  if (!tools.length) return;
  const remembered = localStorage.getItem(LAST_TOOL(page));
  await selectTool(tools.some((t) => t.key === remembered) ? remembered : tools[0].key);
  renderRunList();
  activatePageRun(page);
}

/* ---------------- Label Studio page ---------------- */

const LS_PUBLIC_KEY = 'birdcount.webui.ls.public';

const LS_POLL_MS = 5000;   // keep the panel honest while the Label Studio tab is open
const LS_RETRY_MS = 2000;  // the web UI server is down: come back sooner

// One timer drives every re-probe, so they can never stack up.
function scheduleLabelStudioRefresh(delay) {
  clearTimeout(refreshLabelStudio._t);
  if (S.page !== 'label_studio') return;  // nothing to keep fresh off-screen
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
  if (data.state === 'running' && S.page === 'label_studio') {
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
$('#btn-restart').addEventListener('click', restartServer);
$('#btn-quit').addEventListener('click', quitServer);
$('#btn-copy').addEventListener('click', async () => {
  await navigator.clipboard.writeText($('#cmd-preview').textContent);
  toast('command copied');
});
$('#gallery-search').addEventListener('input', () => renderGallery());
$('#gallery-search').addEventListener('keydown', (e) => {
  if (e.key !== 'Escape' || !e.currentTarget.value) return;
  e.stopPropagation();
  e.currentTarget.value = '';
  renderGallery();
});
$('#gallery-search-clear').addEventListener('click', () => {
  const search = $('#gallery-search');
  search.value = '';
  renderGallery();
  search.focus();
});
$('#lightbox').addEventListener('click', () => { $('#lightbox').hidden = true; });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') $('#lightbox').hidden = true; });

// The two service cards fold independently, so starting Label Studio never
// leaves its controls clipped inside a fixed-height viewport. ngrok starts
// collapsed because it is optional; both choices persist across reloads.
const SERVICE_PANELS = {
  labelStudio: { panel: $('#ls-service-panel'), button: $('#ls-collapse'), defaultCollapsed: false },
  ngrok: { panel: $('#ng-service-panel'), button: $('#ng-collapse'), defaultCollapsed: true },
};

function servicePanelKey(name) { return `bird-count-service-panel-${name}-collapsed`; }

function setServicePanelCollapsed(name, collapsed) {
  const view = SERVICE_PANELS[name];
  view.panel.classList.toggle('is-collapsed', collapsed);
  view.button.textContent = collapsed ? 'Expand' : 'Collapse';
  view.button.setAttribute('aria-expanded', String(!collapsed));
  localStorage.setItem(servicePanelKey(name), collapsed ? '1' : '0');
}

for (const [name, view] of Object.entries(SERVICE_PANELS)) {
  const saved = localStorage.getItem(servicePanelKey(name));
  setServicePanelCollapsed(name, saved === null ? view.defaultCollapsed : saved === '1');
  view.button.addEventListener('click', () => {
    setServicePanelCollapsed(name, !view.panel.classList.contains('is-collapsed'));
  });
}

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
  else if (S.detail.kind === 'regional_density_error') renderRegionalErrors(S.detail);
  else renderEvaluation(S.detail);
});
window.addEventListener('resize', () => {
  if (!S.detail) return;
  drawChart(S.detail);
  drawTimeline();
});
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
  const requested = location.hash.slice(1);
  const wanted = requested === 'data' ? 'label_studio' : requested;
  await switchPage(PAGES().includes(wanted) ? wanted : 'train');
  document.body.classList.remove('is-booting');
  S.listTimer = setInterval(() => { if (!S.timer) refreshRunList(); }, 5000);
})();
