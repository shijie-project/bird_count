// The page: tabs, the tool form, starting and stopping jobs, the run list, the
// selected run's view and its log. Everything is polled; nothing is pushed.

import { DatasetBrowser } from './data.js';
import { forgetPickers, ToolForm } from './form.js';
import { LabelStudioPanel } from './labelstudio.js';
import { VIEWS } from './views.js';
import { $, api, debounce, duration, el, fmt, pct, store, toast, when } from './util.js';

const PAGES = {
  train: { title: 'Train', kicker: 'Model', text: 'Train the density model or sweep its settings; curves come from each run\'s metrics.jsonl.' },
  eval: { title: 'Evaluate', kicker: 'Model', text: 'Score a checkpoint on a split exactly as the live runtime counts.' },
  video: { title: 'Video', kicker: 'Analysis', text: 'The flock count and the busiest patch over the length of a recording.' },
  data: { title: 'Data', kicker: 'Dataset', text: 'Browse the dataset, and run the file pipeline from Label Studio exports to train / val.' },
  label_studio: { title: 'Label Studio', kicker: 'Annotation', text: 'Run the annotation server and keep its project in order.' },
};
const LIST_FAST_MS = 2000;
const LIST_SLOW_MS = 8000;
const DETAIL_MS = 1500;
const LOG_MS = 1000;
const LOG_KEEP = 400_000;

const S = {
  tools: [],
  page: null,
  tool: null,
  specs: new Map(),
  items: [],
  selected: null,   // {type: 'run', ref} | {type: 'job', id}
  view: null,
  viewRef: null,
  offset: 0,
  log: { key: null, offset: -1 },
  timers: {},
  instance: null,
};

const form = new ToolForm($('#form'), debounce((values, key) => preview(values, key), 200));
let browser = null;
let labelStudio = null;

// --- tools and the form ------------------------------------------------------------------------

async function spec(key) {
  if (!S.specs.has(key)) S.specs.set(key, await api(`/api/tools/${key}`));
  return S.specs.get(key);
}

function pageTools(page) { return S.tools.filter((t) => t.page === page); }

async function selectTool(key, preset = null) {
  S.tool = S.tools.find((t) => t.key === key) ?? pageTools(S.page)[0];
  if (!S.tool) return;
  store.set(`tool.${S.page}`, S.tool.key);
  for (const b of $('#tool-picker').children) b.classList.toggle('is-active', b.dataset.key === S.tool.key);
  $('#blurb').textContent = S.tool.blurb;
  $('#form').replaceChildren(el('div', { className: 'form-loading' }, 'Reading the tool\'s options…'));
  try {
    form.render(await spec(S.tool.key), preset);
  } catch (err) {
    $('#form').replaceChildren(el('div', { className: 'form-loading' }, err.message));
  }
}

async function preview(values, key) {
  if (!key || key !== S.tool?.key) return;  // a late edit of the form just replaced
  try {
    const p = await api('/api/preview', { method: 'POST', body: { tool: key, values } });
    $('#cmd-preview').textContent = p.error ?? p.command;
    $('#cmd').classList.toggle('is-error', Boolean(p.error));
  } catch (err) {
    $('#cmd-preview').textContent = err.message;
    $('#cmd').classList.add('is-error');
  }
}

async function start() {
  if (!S.tool) return;
  $('#btn-start').disabled = true;
  try {
    const job = await api('/api/jobs', { method: 'POST', body: { tool: S.tool.key, values: form.values() } });
    toast(`started ${job.id}`, 'info');
    forgetPickers();
    await refreshList();
    select(job.ref ? { type: 'run', ref: job.ref, pending: job.id } : { type: 'job', id: job.id });
  } catch (err) {
    toast(err.message);
  } finally {
    $('#btn-start').disabled = false;
  }
}

async function stop() {
  const job = selectedJob();
  if (!job) return;
  try {
    await api(`/api/jobs/${job.id}/stop`, { method: 'POST' });
    toast('asked to stop; it saves what it has first', 'info');
  } catch (err) { toast(err.message); }
}

// --- pages ------------------------------------------------------------------------------------------

async function switchPage(page, { tool = null, preset = null } = {}) {
  if (S.page !== page) {
    S.page = page;
    store.set('page', page);
    for (const t of $('#tabs').children) t.classList.toggle('is-active', t.dataset.page === page);
    const meta = PAGES[page];
    $('#page-title').textContent = meta.title;
    $('#page-kicker').textContent = meta.kicker;
    $('#page-description').textContent = meta.text;
    const tools = pageTools(page);
    $('#tool-picker').replaceChildren(...tools.map((t) => el('button', { className: 'tool', dataset: { key: t.key }, onclick: () => selectTool(t.key) }, t.label)));
    $('#tool-picker').hidden = tools.length < 2;
    $('#data-view').hidden = page !== 'data';
    $('#ls-view').hidden = page !== 'label_studio';
    if (page === 'data' && !browser) { browser = new DatasetBrowser($('#data-view')); await browser.init(); }
    if (page === 'label_studio') {
      if (!labelStudio) labelStudio = new LabelStudioPanel($('#ls-view'));
      labelStudio.activate();
    } else {
      labelStudio?.deactivate();
    }
    $('#runs-title').textContent = { train: 'Training runs', eval: 'Evaluations', video: 'Video runs' }[page] ?? 'Jobs';
    S.items = [];
    clearSelection();
    await selectTool(tool ?? store.get(`tool.${page}`, tools[0]?.key), preset);
    await refreshList();
    const remembered = store.get(`selected.${page}`, null);
    const item = (remembered && S.items.find((i) => sameItem(i, remembered))) || S.items[0];
    if (item) select(itemKey(item));
  } else if (tool) {
    await selectTool(tool, preset);
  }
}

// --- the run list -------------------------------------------------------------------------------------

const itemKey = (item) => (item.type === 'run' ? { type: 'run', ref: item.ref } : { type: 'job', id: item.id });
const sameItem = (item, key) => key && item.type === key.type && (item.type === 'run' ? item.ref === key.ref : item.id === key.id);

function itemSub(item) {
  if (item.type === 'job') return `${when(item.started)} · ${item.status}${item.finished ? ` · ${duration(item.finished - item.started)}` : ''}`;
  const parts = [when(item.started)];
  if (item.kind === 'train') {
    parts.push(`ep ${item.epochs?.[0] ?? 0}/${item.epochs?.[1] ?? '?'}`);
    if (item.best?.counting_accuracy != null) parts.push(`acc ${pct(item.best.counting_accuracy)}`);
    else if (item.best?.mae != null) parts.push(`MAE ${fmt(item.best.mae)}`);
  } else if (item.kind === 'sweep') {
    parts.push(`${item.runs} runs`);
    if (item.best) parts.push(`best acc ${pct(item.best.counting_accuracy)}`);
  } else if (item.kind === 'eval') {
    parts.push(item.split ?? '');
    if (item.counting_accuracy != null) parts.push(`acc ${pct(item.counting_accuracy)}`);
  } else if (item.kind === 'video' && item.max_count != null) {
    parts.push(`max ${fmt(item.max_count, 0)} · patch ${fmt(item.max_peak, 1)}`);
  }
  return parts.filter(Boolean).join(' · ');
}

function renderList() {
  const list = $('#run-list');
  $('#runs-count').textContent = S.items.length;
  $('#empty-state').hidden = S.items.length > 0 || S.page === 'data' || S.page === 'label_studio';
  list.replaceChildren(...S.items.map((item) => el('li', {
    className: sameItem(item, S.selected) ? 'sel' : '',
    onclick: () => select(itemKey(item)),
    title: item.type === 'run' ? item.ref : item.command,
  },
  el('div', { className: 'r-top' }, el('span', { className: 'r-kind' }, item.type === 'job' ? item.tool.replace('labeling-', '') : item.name),
    el('span', { className: 'r-dot', dataset: { status: item.status } })),
  el('div', { className: 'r-sub' }, itemSub(item)))));
}

async function refreshList() {
  clearTimeout(S.timers.list);
  try {
    const data = await api(`/api/runs?page=${S.page}`);
    S.items = data.items;
    S.running = data.running;
    renderList();
    updateHeader();
  } catch { /* panel restarting: keep the last list */ }
  S.timers.list = setTimeout(refreshList, S.running?.length ? LIST_FAST_MS : LIST_SLOW_MS);
}

function selectedItem() { return S.items.find((i) => sameItem(i, S.selected)); }

function selectedJob() {
  const item = selectedItem();
  if (!item) return S.selected?.pending ? { id: S.selected.pending, status: 'running' } : null;
  const job = item.type === 'job' ? item : item.job;
  return job?.status === 'running' ? job : null;
}

function updateHeader() {
  const item = selectedItem();
  const status = item?.status ?? (S.selected ? 'running' : 'idle');
  const pill = $('#status-pill');
  pill.dataset.status = status;
  pill.textContent = status;
  $('#btn-stop').disabled = !selectedJob();
  $('#workspace-run').hidden = !item;
  $('#workspace-run-name').textContent = item ? (item.type === 'run' ? item.ref : item.id) : '';
  for (const li of $('#run-list').children) li.classList.remove('sel');
  const index = S.items.findIndex((i) => sameItem(i, S.selected));
  $('#run-list').children[index]?.classList.add('sel');
}

// --- the selected run -------------------------------------------------------------------------------

function clearSelection() {
  S.selected = null;
  S.view = null;
  S.viewRef = null;
  clearTimeout(S.timers.detail);
  clearTimeout(S.timers.log);
  $('#run-view').replaceChildren();
  $('.panel-log').hidden = true;
  $('#log').textContent = '';
  $('#log-path').textContent = '';
  S.log = { key: null, offset: -1 };
}

function select(key) {
  clearSelection();
  S.selected = key;
  $('.panel-log').hidden = false;
  store.set(`selected.${S.page}`, key);
  updateHeader();
  if (key.type === 'run') pollDetail(); else pollLog();
}

async function pollDetail() {
  clearTimeout(S.timers.detail);
  const key = S.selected;
  if (key?.type !== 'run') return;
  let d;
  try {
    d = await api(`/api/run?ref=${encodeURIComponent(key.ref)}&offset=${S.viewRef === key.ref ? S.offset : 0}`);
  } catch (err) {
    if (key.pending) { S.timers.detail = setTimeout(pollDetail, DETAIL_MS); pollLog(); return; }  // folder not made yet
    $('#run-view').replaceChildren(el('div', { className: 'empty' }, err.message));
    return;
  }
  if (S.selected !== key) return;
  if (S.viewRef !== key.ref) {
    $('#run-view').replaceChildren();
    S.view = new VIEWS[d.kind]($('#run-view'), CTX);
    S.viewRef = key.ref;
  }
  S.offset = d.offset ?? 0;
  S.view.update(d);
  S.detailJob = d.job;
  if (S.log.key === null) pollLog();
  if (d.status === 'running') S.timers.detail = setTimeout(pollDetail, DETAIL_MS);
}

function logSource() {
  const key = S.selected;
  if (!key) return null;
  if (key.type === 'job') return `job=${encodeURIComponent(key.id)}`;
  const job = S.detailJob ?? (key.pending ? { id: key.pending } : null);
  if (job) return `job=${encodeURIComponent(job.id)}`;
  return `ref=${encodeURIComponent(key.ref)}`;
}

async function pollLog() {
  clearTimeout(S.timers.log);
  const source = logSource();
  if (!source) return;
  if (S.log.key !== source) { S.log = { key: source, offset: -1 }; $('#log').textContent = ''; }
  try {
    const data = await api(`/api/log?${source}&offset=${S.log.offset}`);
    if (S.log.key !== source) return;
    if (data.offset < S.log.offset) $('#log').textContent = '';
    S.log.offset = data.offset;
    if (data.missing && !$('#log').textContent) $('#log-path').textContent = 'no log for this run';
    else $('#log-path').textContent = source.startsWith('job') ? 'console of the job' : 'train.log';
    if (data.text) appendLog(data.text);
  } catch { /* retry */ }
  const item = selectedItem();
  const live = S.selected?.pending && !item ? true : (item?.status === 'running' || (item?.job?.status === 'running'));
  if (live) S.timers.log = setTimeout(pollLog, LOG_MS);
}

function appendLog(text) {
  const pre = $('#log');
  const follow = $('#follow').checked;
  let all = pre.textContent + text;
  if (all.length > LOG_KEEP) all = all.slice(-LOG_KEEP);
  pre.textContent = all;
  if (follow) pre.scrollTop = pre.scrollHeight;
}

// --- moves between pages, offered by the views -----------------------------------------------------

const CTX = {
  openRun: async (ref) => { await switchPage('train'); select({ type: 'run', ref }); },
  evaluate: (ckpt) => switchPage('eval', { tool: 'eval', preset: { ckpt } }),
  resume: (path) => switchPage('train', { tool: 'train', preset: { resume: path } }),
  prelabel: (blobs) => switchPage('label_studio', { tool: 'labeling-prelabel', preset: { blobs } }),
  browse: async (split, ref) => { await switchPage('data'); await browser.open(split, ref); },
};

// --- the panel itself ---------------------------------------------------------------------------------

async function lifecycle(restart) {
  const verb = restart ? 'restart' : 'shut down';
  if (S.running?.length && !confirmInline(`${S.running.length} job(s) running. They will be stopped (saving their state). ${verb} anyway?`)) return;
  try {
    await api(restart ? '/api/restart' : '/api/shutdown', { method: 'POST' });
  } catch (err) { toast(err.message); return; }
  document.body.classList.add('is-down');
  $('#down-text').textContent = restart ? 'Restarting…' : 'The panel has stopped. Close this tab, or start it again with python -m birdcount.training.webui';
  if (!restart) return;
  const old = S.instance;
  for (let i = 0; i < 120; i++) {
    await new Promise((r) => setTimeout(r, 500));
    try {
      const h = await api('/api/health');
      if (h.instance !== old) { location.reload(); return; }
    } catch { /* still down */ }
  }
  $('#down-text').textContent = 'The panel did not come back. Start it again from a shell.';
}

/** A confirmation without window.confirm (blocking dialogs freeze automation and look out of place). */
function confirmInline(message) {
  const node = $('#confirm');
  if (node.dataset.armed === message) { node.hidden = true; node.dataset.armed = ''; return true; }
  node.dataset.armed = message;
  node.textContent = `${message} Click again to confirm.`;
  node.hidden = false;
  setTimeout(() => { node.hidden = true; node.dataset.armed = ''; }, 6000);
  return false;
}

async function clearHistory() {
  if (!confirmInline('Forget finished jobs and delete their console logs? Run folders stay.')) return;
  try {
    const r = await api('/api/jobs/clear', { method: 'POST' });
    toast(`removed ${r.removed} job record(s)`, 'info');
    refreshList();
  } catch (err) { toast(err.message); }
}

async function boot() {
  try {
    const [tools, health] = await Promise.all([api('/api/tools'), api('/api/health')]);
    S.tools = tools.tools;
    S.instance = health.instance;
    $('#btn-restart').disabled = !health.restartable;
  } catch (err) {
    $('#boot-state').textContent = `Cannot reach the panel: ${err.message}`;
    return;
  }
  $('#tabs').replaceChildren(...Object.entries(PAGES).map(([key, p]) => el('button', { className: 'tab', dataset: { page: key }, onclick: () => switchPage(key) }, p.title)));
  $('#btn-start').addEventListener('click', start);
  $('#btn-stop').addEventListener('click', stop);
  $('#btn-reset').addEventListener('click', () => form.reset());
  $('#btn-copy').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText($('#cmd-preview').textContent); toast('command copied', 'info'); } catch { toast('could not copy'); }
  });
  $('#btn-clear').addEventListener('click', clearHistory);
  $('#btn-restart').addEventListener('click', () => lifecycle(true));
  $('#btn-quit').addEventListener('click', () => lifecycle(false));
  $('#lightbox').addEventListener('click', () => { $('#lightbox').hidden = true; });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') $('#lightbox').hidden = true; });
  document.body.classList.remove('is-booting');
  const page = store.get('page', 'train');
  await switchPage(PAGES[page] ? page : 'train');
}

boot();
