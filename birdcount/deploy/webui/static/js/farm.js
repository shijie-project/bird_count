// The farm: what every camera shows now, placed on the shed's floor plan, and
// the day's lanes underneath.

import { drawLanes } from './chart.js';
import { ago, clock, el, levelName, short, span, zoneTitle } from './util.js';

const GAUGE_TOP = 1.4;  // the gauge's full width is this multiple of the threshold
const DRAG_PX = 6;  // a press that moves less than this is a click (open the camera), not a zoom
const MIN_ZOOM_SEC = 120;

// The part of the day the lanes show once zoomed in, kept across the page's
// re-renders every 2 s; none of them happens while a zoom is being dragged.
let zoom = null;  // {day, t0, t1}
let dragging = false;

// What a tile shows: the live status while the runtime runs, else the day's last row.
function cameraNow(cam, ctx, day) {
  const st = ctx.status;
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
  const pct = now.count == null ? 0 : Math.min(100, (now.count / (cam.threshold * GAUGE_TOP)) * 100);
  return el('a', { className: 'tile', href: `#/camera/${cam.mac}${ctx.q.day ? `?day=${ctx.q.day}` : ''}`, dataset: { level: String(now.level) }, title: `${cam.camera_id}: threshold ${cam.threshold}` },
    el('div', { className: 'top' }, el('span', { textContent: short(cam.mac) }), el('span', { className: 'state', textContent: state })),
    el('div', { className: 'count' }, now.count == null ? '—' : String(Math.round(now.count)), el('small', { textContent: `/ ${cam.threshold}` })),
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
      rows.push({ key: mac, group: z.name, zone: label, label: short(mac), threshold: d.threshold ?? byMac.get(mac).threshold, periods: d.periods || [], snaps: d.to_label || [] });
    }
  }
  return rows;
}

export async function renderFarm(view, ctx) {
  if (ctx.quiet && dragging) return;
  const day = ctx.day ? await ctx.dayData(ctx.day) : null;
  const st = ctx.status;
  const running = st?.state === 'running';
  const cams = ctx.farm.cameras;
  const alarming = running
    ? (st.cameras || []).filter((c) => c.level > 0).length
    : cams.filter((c) => (day?.cameras[c.mac]?.level_now || 0) > 0).length;
  const actions = day?.actions || [];
  const alarmSec = day ? Object.values(day.cameras).reduce((n, c) => n + (c.alarm_seconds || 0), 0) : 0;
  const todo = day ? Object.values(day.cameras).reduce((n, c) => n + (c.to_label?.length || 0), 0) : 0;

  const head = el('div', { className: 'page-head' },
    el('h1', {}, running
      ? (alarming ? [el('span', { className: 'n', textContent: alarming }), ` camera${alarming > 1 ? 's' : ''} alarming`] : [el('span', { className: 'n ok', textContent: cams.length }), ' cameras, none alarming'])
      : 'The runtime is not running'),
    el('div', { className: 'stats' },
      el('span', {}, el('b', { textContent: actions.filter((a) => a.type === 'level1').length }), 'alarms'),
      el('span', {}, el('b', { textContent: actions.filter((a) => a.type === 'level2' || a.type === 'level3').length }), 'escalated'),
      el('span', {}, el('b', { textContent: Math.round(alarmSec / 60) }), 'camera-minutes in alarm'),
      el('span', {}, el('a', { href: `#/label${ctx.q.day ? `?day=${ctx.q.day}` : ''}` }, el('b', { textContent: todo }), 'snapshots to label'))),
    daySelect(ctx, 'farm'));

  const notices = [];
  if (!running && st?.written) notices.push(el('div', { className: 'notice' }, 'Tiles show the last values the count log recorded. Start the runtime (', el('code', { textContent: 'python -m birdcount.deploy' }), ') to see them live.'));
  if (!ctx.farm.zones.some((z) => z.kind === 'zone' && z.macs.length)) {
    notices.push(el('div', { className: 'notice' }, el('b', { textContent: 'Cameras are grouped by recorder, not by zone: ' }), 'topology.yaml has no camera MACs yet. On the farm network run ', el('code', { textContent: 'python -m birdcount.deploy.tools.discover_ids --write' }), ' and the floor plan fills in.'));
  }

  const parts = [head, ...notices, barn(ctx, day)];
  if (day?.span) {
    const z = zoom?.day === ctx.day ? zoom : null;
    const t0 = z ? Math.max(day.span[0], z.t0) : day.span[0];
    const t1 = z ? Math.min(day.span[1] + 1, z.t1) : day.span[1] + 1;
    const again = () => renderFarm(view, { ...ctx, quiet: true });
    const canvas = el('canvas', { className: 'clickable lanes-canvas', role: 'img', 'aria-label': 'Alarm periods of every camera over the day; drag across to zoom in' });
    const band = el('div', { className: 'lane-band', hidden: true });
    parts.push(el('div', { className: 'panel' },
      el('div', { className: 'panel-head' }, el('h2', { textContent: ctx.day === ctx.days[0] ? 'Today' : ctx.day }),
        el('span', { className: 'muted', textContent: z ? `${clock(t0)} to ${clock(t1)}, ${ctx.day}` : `alarm periods by camera, ${ctx.day}; drag across the lanes to zoom in` }),
        z ? el('button', { className: 'chip', textContent: 'Show the whole day', onclick: () => { zoom = null; again(); } }) : null,
        el('div', { className: 'legend' }, el('span', {}, el('i', { className: 'l1' }), 'Level 1'), el('span', {}, el('i', { className: 'l2' }), 'Level 2'), el('span', {}, el('i', { className: 'l3' }), 'Level 3'), el('span', {}, el('i', { className: 'tick' }), 'snapshot to label'))),
      el('div', { className: 'lanes' }, canvas, band)));
    view.replaceChildren(...parts);
    const lanes = drawLanes(canvas, laneRows(ctx, day), t0, t1);
    let press = null;  // {x, t} while the button is down
    const clamp = (clientX) => Math.min(Math.max(clientX - canvas.getBoundingClientRect().left, lanes.left), lanes.right);
    const end = () => { press = null; dragging = false; band.hidden = true; };
    canvas.onpointerdown = (e) => {
      if (e.button !== 0) return;
      press = { x: e.clientX, t: lanes.timeAt(e) };
      dragging = true;
      canvas.setPointerCapture(e.pointerId);
    };
    canvas.onpointermove = (e) => {
      if (!press) return;
      const a = clamp(press.x), b = clamp(e.clientX);
      band.hidden = Math.abs(e.clientX - press.x) < DRAG_PX;
      band.style.left = `${Math.min(a, b)}px`;
      band.style.width = `${Math.abs(b - a)}px`;
    };
    canvas.onpointerup = (e) => {
      if (!press) return;
      const moved = Math.abs(e.clientX - press.x) >= DRAG_PX;
      const ta = press.t, tb = lanes.timeAt(e);
      end();
      if (moved) {
        if (Math.abs(tb - ta) >= MIN_ZOOM_SEC) { zoom = { day: ctx.day, t0: Math.min(ta, tb), t1: Math.max(ta, tb) }; again(); }
        return;
      }
      const r = lanes.row(e);
      if (r) ctx.go('camera', r.key, { day: ctx.q.day });
    };
    canvas.onpointercancel = end;
  } else {
    parts.push(el('div', { className: 'empty', style: 'margin-top:16px' }, el('h2', { textContent: 'No count log yet' }), el('p', { textContent: 'Turn the count logger on (ENABLE_COUNT_LOGGER=1) and the day appears here as the runtime runs.' })));
    view.replaceChildren(...parts);
  }
}

export function daySelect(ctx, page, extra = {}) {
  if (!ctx.days.length) return null;
  const sel = el('select', { 'aria-label': 'Day' }, ctx.days.map((d, i) => el('option', { value: d, textContent: i === 0 ? `${d} (latest)` : d, selected: d === ctx.day })));
  sel.onchange = () => ctx.go(page, extra.arg || null, { ...ctx.q, day: sel.value === ctx.days[0] ? '' : sel.value });
  return el('label', { className: 'day-pick' }, 'Day', sel);
}
