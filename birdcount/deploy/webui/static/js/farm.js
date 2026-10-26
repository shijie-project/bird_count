// The farm: what every camera shows now, placed on the shed's floor plan, and
// the day's lanes underneath.

import { drawLanes } from './chart.js';
import { datePicker, pickerOpen } from './datepick.js';
import { ago, clock, el, levelName, short, span, zoneTitle } from './util.js';

const GAUGE_TOP = 1.4;  // the gauge's full width is this multiple of the threshold

// What a tile shows. Today: the live status while the runtime runs, else the day's
// last row. An earlier day: its peak (highest 1-s mean) and the highest level it reached.
function cameraNow(cam, ctx, day) {
  const st = ctx.status;
  if (!isLatest(ctx)) {
    const d = day?.cameras[cam.mac];
    if (!d?.seconds) return { level: 'none', count: null, note: 'no record that day' };
    return { level: d.max_level, count: d.peak, note: `peak at ${clock(d.peak_at)}`, threshold: d.threshold };  // that day's threshold
  }
  const live = st?.state === 'running' ? (st.cameras || []).find((c) => c.mac === cam.mac) : null;
  if (live) {
    if (!live.online) return { level: 'off', count: null, note: live.captured ? `no picture for ${ago(st.written - live.captured)}` : 'no picture yet' };
    return { level: live.level, count: live.count, note: live.level ? `for ${span(st.written - live.alarm_since)}` : '' };
  }
  if (st?.state === 'running') return { level: 'none', count: null, note: 'not in this run' };
  const d = day?.cameras[cam.mac];
  if (!d?.seconds) return { level: 'none', count: null, note: 'no record today' };
  return { level: d.level_now, count: d.count_now, note: 'last recorded' };
}

function tile(cam, now, ctx) {
  const state = now.level === 'off' ? 'Offline' : now.level === 'none' ? '—' : levelName(now.level);
  const threshold = now.threshold ?? cam.threshold;
  const pct = now.count == null ? 0 : Math.min(100, (now.count / (threshold * GAUGE_TOP)) * 100);
  return el('a', { className: 'tile', href: `#/camera/${cam.mac}${ctx.q.day ? `?day=${ctx.q.day}` : ''}`, dataset: { level: String(now.level) }, title: `${cam.camera_id}: threshold ${threshold}` },
    el('div', { className: 'top' }, el('span', { textContent: short(cam.mac) }), el('span', { className: 'state', textContent: state })),
    el('div', { className: 'count' }, now.count == null ? '—' : String(Math.round(now.count)), el('small', { textContent: `/ ${threshold}` })),
    el('div', { className: 'gauge' }, el('i', { style: `width:${pct}%` }), el('b', { style: `left:${100 / GAUGE_TOP}%` })),
    el('div', { className: 'since', textContent: now.note }));
}

function block(z, cams, ctx, day, cls) {
  const devices = z.kind === 'zone' ? [z.speakers && `${z.speakers} speaker${z.speakers > 1 ? 's' : ''}`, z.smart_plugs && `${z.smart_plugs} plug${z.smart_plugs > 1 ? 's' : ''}`].filter(Boolean).join(', ') : '';
  const tiles = el('div', { className: 'tiles', style: `--cols:${Math.max(1, Math.min(5, cams.length || 1))}` }, cams.map((c) => tile(c, cameraNow(c, ctx, day), ctx)));
  const missing = z.kind === 'zone' && z.cameras_listed > cams.length
    ? el('div', { className: 'missing', textContent: `${z.cameras_listed - cams.length} camera${z.cameras_listed - cams.length > 1 ? 's' : ''} without a MAC in topology.yaml` })
    : null;
  return el('section', { className: `block ${cls}` }, el('h3', {}, zoneTitle(z.name), el('span', { className: 'devices', textContent: devices })), cams.length ? tiles : null, missing);
}

function barn(ctx, day) {
  const byMac = new Map(ctx.farm.cameras.map((c) => [c.mac, c]));
  const zones = ctx.farm.zones;
  const placed = zones.filter((z) => z.kind === 'zone' && z.macs.length);
  const groups = zones.filter((z) => z.kind === 'group');
  if (!placed.length) {  // topology.yaml has no MACs yet: group by recorder, in a plain grid
    return el('div', { className: 'barn groups' }, groups.map((z) => block(z, z.macs.map((m) => byMac.get(m)), ctx, day, '')));
  }
  const planned = zones.filter((z) => z.kind === 'zone' && z.side && (z.row === 'front' || z.row === 'back'));
  const outer = zones.filter((z) => z.kind === 'zone' && !planned.includes(z));
  const box = el('div', { className: `barn${outer.length ? ' with-outer' : ''}` });
  box.append(el('div', { className: 'edge', textContent: 'Front of the shed' }));
  for (const z of planned) box.append(block(z, z.macs.map((m) => byMac.get(m)), ctx, day, `${z.side} ${z.row}`));
  if (outer.length) box.append(el('div', { className: 'block outer' }, outer.map((z) => block(z, z.macs.map((m) => byMac.get(m)), ctx, day, ''))));
  box.append(el('div', { className: 'edge back', style: 'grid-row:4', textContent: 'Back of the shed' }));
  for (const z of groups) box.append(block(z, z.macs.map((m) => byMac.get(m)), ctx, day, ''));
  return box;
}

export function laneRows(ctx, day) {
  const byMac = new Map(ctx.farm.cameras.map((c) => [c.mac, c]));
  const rows = [];
  for (const z of ctx.farm.zones) {
    const label = z.kind === 'group' ? z.name.replace(/^axis/i, 'A') : `Z${(z.name.match(/\d+/) || [''])[0]}`;
    for (const mac of z.macs) {
      const d = day.cameras[mac] || {};
      rows.push({ key: mac, group: z.name, zone: label, label: short(mac), threshold: d.threshold ?? byMac.get(mac).threshold, periods: d.periods || [], snaps: d.unchecked || [] });  // alarms still to check
    }
  }
  return rows;
}

const isLatest = (ctx) => !ctx.day || ctx.day === ctx.days[0];

export async function renderFarm(view, ctx) {
  if (ctx.quiet && pickerOpen()) return;  // the 2-s refresh waits while the calendar is open
  const day = ctx.day ? await ctx.dayData(ctx.day) : null;
  const st = ctx.status;
  const latest = isLatest(ctx);
  const running = st?.state === 'running' && latest;
  const cams = ctx.farm.cameras;
  const alarming = running
    ? (st.cameras || []).filter((c) => c.level > 0).length
    : cams.filter((c) => (day?.cameras[c.mac]?.level_now || 0) > 0).length;
  const alarmed = cams.filter((c) => (day?.cameras[c.mac]?.max_level || 0) > 0).length;
  const actions = day?.actions || [];
  const alarmSec = day ? Object.values(day.cameras).reduce((n, c) => n + (c.alarm_seconds || 0), 0) : 0;
  const todo = day ? Object.values(day.cameras).reduce((n, c) => n + (c.unchecked?.length || 0), 0) : 0;

  const head = el('div', { className: 'page-head' },
    el('h1', {}, !latest
      ? (alarmed ? [el('span', { className: 'n', textContent: alarmed }), ` camera${alarmed > 1 ? 's' : ''} alarmed that day`] : [el('span', { className: 'n ok', textContent: cams.length }), ' cameras, no alarm that day'])
      : running
        ? (alarming ? [el('span', { className: 'n', textContent: alarming }), ` camera${alarming > 1 ? 's' : ''} alarming`] : [el('span', { className: 'n ok', textContent: cams.length }), ' cameras, none alarming'])
        : 'The runtime is not running'),
    el('div', { className: 'stats' },
      el('span', {}, el('b', { textContent: actions.filter((a) => a.type === 'level1').length }), 'alarms'),
      el('span', {}, el('b', { textContent: actions.filter((a) => a.type === 'level2' || a.type === 'level3').length }), 'escalated'),
      el('span', {}, el('b', { textContent: Math.round(alarmSec / 60) }), 'camera-minutes in alarm'),
      el('span', {}, el('a', { href: `#/check${ctx.q.day ? `?day=${ctx.q.day}` : ''}` }, el('b', { textContent: todo }), 'alarms to check'))),
    daySelect(ctx, 'farm'));

  const notices = [];
  if (!latest) notices.push(el('div', { className: 'notice' }, el('b', { textContent: `${ctx.day}: ` }), "each tile shows the camera's peak that day (its highest 1-second mean count) and the highest alarm level it reached. Pick the latest day for the live view."));
  else if (!running && st?.written) notices.push(el('div', { className: 'notice' }, 'Tiles show the last values the count log recorded. Start the runtime (', el('code', { textContent: 'python -m birdcount.deploy' }), ') to see them live.'));
  if (!ctx.farm.zones.some((z) => z.kind === 'zone' && z.macs.length)) {
    notices.push(el('div', { className: 'notice' }, el('b', { textContent: 'Cameras are grouped by recorder, not by zone: ' }), 'topology.yaml has no camera MACs yet. On the farm network run ', el('code', { textContent: 'python -m birdcount.deploy.tools.discover_ids --write' }), ' and the floor plan fills in.'));
  }

  const parts = [head, ...notices, barn(ctx, day)];
  if (day?.span) {
    const canvas = el('canvas', { className: 'clickable', role: 'img', 'aria-label': 'Alarm periods of every camera over the day' });
    parts.push(el('div', { className: 'panel' },
      el('div', { className: 'panel-head' }, el('h2', { textContent: ctx.day === ctx.days[0] ? 'Today' : ctx.day }), el('span', { className: 'muted', textContent: `alarm periods by camera, ${ctx.day}` }),
        el('div', { className: 'legend' }, el('span', {}, el('i', { className: 'l1' }), 'Level 1'), el('span', {}, el('i', { className: 'l2' }), 'Level 2'), el('span', {}, el('i', { className: 'l3' }), 'Level 3'), el('span', {}, el('i', { className: 'ring' }), 'alarm to check'))),
      canvas));
    view.replaceChildren(...parts);
    const [t0, t1] = day.span;
    const hit = drawLanes(canvas, laneRows(ctx, day), t0, t1 + 1);
    canvas.onclick = (e) => { const r = hit(e); if (r) ctx.go('camera', r.key, { day: ctx.q.day }); };
  } else {
    parts.push(el('div', { className: 'empty', style: 'margin-top:16px' }, el('h2', { textContent: 'No count log yet' }), el('p', { textContent: 'Turn the count logger on (ENABLE_COUNT_LOGGER=1) and the day appears here as the runtime runs.' })));
    view.replaceChildren(...parts);
  }
}

export function daySelect(ctx, page, extra = {}) {
  if (!ctx.days.length) return null;
  const go = (day) => ctx.go(page, extra.arg || null, { ...ctx.q, day: day === ctx.days[0] ? '' : day });
  return el('div', { className: 'day-pick' }, 'Day', datePicker(ctx.days, ctx.day || ctx.days[0], go));
}
