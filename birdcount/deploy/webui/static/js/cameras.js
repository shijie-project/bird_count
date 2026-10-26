// Every camera's day as one table, sortable; a row opens that camera.

import { daySelect } from './farm.js';
import { el, short, span, zoneTitle } from './util.js';

const COLUMNS = [
  ['camera', 'Camera', (r) => r.mac],
  ['group', 'Zone', (r) => r.group],
  ['threshold', 'Threshold', (r) => r.threshold],
  ['now', 'Last count', (r) => r.d.count_now ?? -1],
  ['max', 'Highest', (r) => r.d.count_max ?? -1],
  ['mean', 'Average', (r) => r.d.count_mean ?? -1],
  ['above', 'Above threshold', (r) => r.d.above_threshold_seconds ?? 0],
  ['alarms', 'Alarms', (r) => r.alarms],
  ['alarm_time', 'In alarm', (r) => r.d.alarm_seconds ?? 0],
  ['level', 'Highest level', (r) => r.d.max_level ?? 0],
  ['labels', 'Labelled', (r) => (r.d.queued ? r.d.queue_done / r.d.queued : -1)],
];
let sortKey = 'alarm_time';
let sortDesc = true;

export async function renderCameras(view, ctx) {
  const day = await ctx.dayData(ctx.day);
  if (!day) {
    view.replaceChildren(el('div', { className: 'empty' }, el('h2', { textContent: 'No count log yet' })));
    return;
  }
  const rows = ctx.farm.cameras.map((c) => ({
    ...c, d: day.cameras[c.mac] || {}, threshold: day.cameras[c.mac]?.threshold ?? c.threshold,
    alarms: day.actions.filter((a) => a.mac === c.mac && a.type === 'level1').length,
  }));
  const col = COLUMNS.find((c) => c[0] === sortKey);
  rows.sort((a, b) => (col[2](a) > col[2](b) ? 1 : col[2](a) < col[2](b) ? -1 : 0) * (sortDesc ? -1 : 1));
  const head = el('tr', {}, COLUMNS.map(([key, label], i) => {
    const th = el('th', { textContent: label, className: i < 2 ? 'l' : '', scope: 'col', tabIndex: 0 });
    if (key === sortKey) th.setAttribute('aria-sort', sortDesc ? 'descending' : 'ascending');
    th.onclick = th.onkeydown = (e) => {
      if (e.type === 'keydown' && e.key !== 'Enter') return;
      sortDesc = key === sortKey ? !sortDesc : true; sortKey = key; renderCameras(view, ctx);
    };
    return th;
  }));
  const body = rows.map((r) => {
    const d = r.d;
    const tr = el('tr', { tabIndex: 0 },
      el('td', {}, el('b', { textContent: short(r.mac) }), el('span', { className: 'muted', textContent: `  ${r.mac}` })),
      el('td', { className: 'l', textContent: zoneTitle(r.group) }),
      el('td', { textContent: r.threshold }),
      el('td', { textContent: d.seconds ? Math.round(d.count_now) : '—' }),
      el('td', { textContent: d.seconds ? Math.round(d.count_max) : '—' }),
      el('td', { textContent: d.seconds ? Math.round(d.count_mean) : '—' }),
      el('td', { textContent: d.seconds ? span(d.above_threshold_seconds) : '—' }),
      el('td', { textContent: r.alarms }),
      el('td', { textContent: d.alarm_seconds ? span(d.alarm_seconds) : '—' }),
      el('td', {}, el('span', { className: 'lv', dataset: { level: String(d.max_level || 0) }, textContent: d.max_level ? `Level ${d.max_level}` : 'none' })),
      el('td', {}, d.queued ? el('span', { className: 'progress-cell' }, `${d.queue_done} of ${d.queued}`, el('span', { className: 'minibar' }, el('i', { style: `width:${(d.queue_done / d.queued) * 100}%` }))) : '—'));
    const open = () => ctx.go('camera', r.mac, { day: ctx.q.day });
    tr.onclick = open;
    tr.onkeydown = (e) => { if (e.key === 'Enter') open(); };
    return tr;
  });
  view.replaceChildren(
    el('div', { className: 'page-head' }, el('h1', { textContent: 'Cameras' }), el('span', { className: 'muted', textContent: `${rows.length} cameras covered by the alarm, ${day.day}` }), daySelect(ctx, 'cameras')),
    el('div', { className: 'panel table-wrap' }, el('table', {}, el('thead', {}, head), el('tbody', {}, body))));
}
