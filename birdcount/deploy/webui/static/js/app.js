// The page: routes, the header's runtime line, and polling. Views render from
// what is loaded here; nothing is pushed by the server.
//
//   #/                         the farm: every camera now, and the day's lanes
//   #/cameras?day=D            every camera's day as a table
//   #/camera/MAC?day=D         one camera's day
//   #/check?day=D&camera=&state=&at=   checking alarms: real pile-up or not
//   #/ratings?day=D               the vision model's crowding levels, every camera
//   #/thresholds               the calibration's suggested thresholds

import { renderCamera } from './camera.js';
import { renderCheck } from './check.js';
import { renderCameras } from './cameras.js';
import { renderFarm } from './farm.js';
import { renderRatings } from './ratings.js';
import { renderThresholds } from './thresholds.js';
import { $, ago, api, el } from './util.js';

const STATUS_MS = 2000;
const DAY_MS = 15000;

const S = { farm: null, status: null, days: [], dayCache: new Map(), route: null, timers: [] };

function parseRoute() {
  const [path, query = ''] = location.hash.replace(/^#/, '').split('?');
  const parts = path.split('/').filter(Boolean);
  return { page: parts[0] || 'farm', arg: parts[1] || null, q: Object.fromEntries(new URLSearchParams(query)) };
}

export function go(page, arg = null, q = {}) {
  const query = new URLSearchParams(Object.entries(q).filter(([, v]) => v !== undefined && v !== null && v !== ''));
  location.hash = `#/${page === 'farm' ? '' : page}${arg ? `/${arg}` : ''}${query.size ? `?${query}` : ''}`;
}

export async function dayData(day, { fresh = false } = {}) {
  if (!day) return null;
  const hit = S.dayCache.get(day);
  if (hit && !fresh && Date.now() - hit.at < DAY_MS) return hit.data;
  const data = await api(`/api/day/${day}`);
  S.dayCache.set(day, { at: Date.now(), data });
  return data;
}

function renderRuntime() {
  const st = S.status || { state: 'never' };
  const box = $('#runtime');
  const cams = st.cameras || [];
  const online = cams.filter((c) => c.online).length;
  const what = {
    running: 'Runtime running', stopped: 'Runtime stopped', gone: 'Runtime not responding',
    never: 'Runtime has not run here yet', disabled: 'Live status is off (STATUS_PATH)', unreadable: 'Runtime running',
  }[st.state];
  const parts = [el('span', {}, el('span', { className: 'dot', dataset: { state: st.state } }), el('b', { textContent: what }))];
  if (st.state === 'running') {
    parts.push(el('span', {}, el('b', { textContent: `${online} of ${cams.length}` }), ' cameras with a picture'));
    parts.push(el('span', {}, 'SMS ', el('b', { textContent: st.runtime?.sms ?? '—' })));
    parts.push(el('span', {}, 'Updated ', el('b', { textContent: ago(st.age) }), ' ago'));
  } else if (st.written) {
    parts.push(el('span', {}, 'Last update ', el('b', { textContent: ago(st.age) }), ' ago'));
  }
  box.replaceChildren(...parts);
}

async function pollStatus() {
  try {
    S.status = await api('/api/status');
  } catch {
    S.status = { state: 'unreadable' };
  }
  renderRuntime();
  if (S.route?.page === 'farm') render({ quiet: true });
}

async function updateTodo(day) {
  const data = await dayData(day);
  const left = data ? Object.values(data.cameras).reduce((n, c) => n + (c.unchecked?.length || 0), 0) : 0;
  $('#todo').textContent = left ? String(left) : '';
}

async function render({ quiet = false } = {}) {
  const route = parseRoute();
  S.route = route;
  for (const a of document.querySelectorAll('#nav a')) {
    const page = route.page === 'camera' ? 'cameras' : route.page;
    if (a.dataset.page === page) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  }
  const day = route.q.day || S.days[0] || null;
  const ctx = { farm: S.farm, status: S.status, days: S.days, day, q: route.q, go, dayData, quiet };
  const view = $('#view');
  try {
    if (route.page === 'farm') await renderFarm(view, ctx);
    else if (route.page === 'cameras') await renderCameras(view, ctx);
    else if (route.page === 'camera') await renderCamera(view, { ...ctx, mac: route.arg });
    else if (route.page === 'check') await renderCheck(view, ctx);
    else if (route.page === 'ratings') await renderRatings(view, ctx);
    else if (route.page === 'thresholds') await renderThresholds(view, ctx);
    else view.replaceChildren(el('div', { className: 'empty' }, el('h2', { textContent: 'No such page' })));
  } catch (err) {
    if (!quiet) view.replaceChildren(el('div', { className: 'empty' }, el('h2', { textContent: 'Could not load this page' }), el('p', { textContent: err.message })));
  }
  updateTodo(day).catch(() => {});
}

async function boot() {
  S.farm = await api('/api/farm');
  S.status = S.farm.status;
  S.days = S.farm.days;
  renderRuntime();
  window.addEventListener('hashchange', () => render());
  let resize = 0;
  window.addEventListener('resize', () => {  // canvases redraw at the new width; the checking page is CSS only
    if (S.route?.page === 'check') return;
    clearTimeout(resize);
    resize = setTimeout(() => render({ quiet: true }), 150);
  });
  await render();
  setInterval(pollStatus, STATUS_MS);
  setInterval(async () => {  // new days appear while the runtime runs past midnight
    try { S.days = (await api('/api/farm')).days; } catch { /* the next round */ }
  }, 60000);
}

boot().catch((err) => {
  $('#view').replaceChildren(el('div', { className: 'empty' }, el('h2', { textContent: 'The backend is not answering' }), el('p', { textContent: err.message })));
});
