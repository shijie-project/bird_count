// One camera's day: the count against its threshold, its alarm events with the
// pictures sent, and its snapshots with the model's level on each.

import { attachReadout, drawCurve } from './chart.js';
import { daySelect } from './farm.js';
import { VERDICT_TEXT } from './check.js';
import { pictureGallery } from './gallery.js';
import { ACTION_NAMES, alarmUrl, api, clock, el, lightbox, short, span, zoneTitle } from './util.js';

const DRAG_PX = 6;  // a press that moves less than this is not a zoom
const MIN_ZOOM_SEC = 120;

// The stretch of the day the count curve shows once zoomed in, for one camera and day.
let zoom = null;  // {day, mac, t0, t1}
let keyHandler = null;

// The cameras in the farm page's order (zone by zone, or recorder by recorder), for stepping through them.
function cameraOrder(farm) {
  const order = new Set(farm.zones.flatMap((z) => z.macs));
  for (const c of farm.cameras) order.add(c.mac);
  return [...order];
}

export async function renderCamera(view, ctx) {
  const z = zoom?.day === ctx.day && zoom?.mac === ctx.mac.toUpperCase() ? zoom : null;
  let data = await api(`/api/day/${ctx.day}/camera/${ctx.mac}${z ? `?from=${Math.floor(z.t0)}&to=${Math.ceil(z.t1)}` : ''}`);
  if (z && (data.curve?.t.length ?? 0) < 2) {  // nothing recorded in that stretch: the whole day
    zoom = null;
    data = await api(`/api/day/${ctx.day}/camera/${ctx.mac}`);
  }
  const zoomed = zoom !== null && data.curve?.t.length >= 2;
  const cam = data.camera;
  const s = data.summary;
  const readout = el('span', { className: 'muted' });
  const canvas = el('canvas', { role: 'img', 'aria-label': `Count over the day against threshold ${data.threshold}; drag across to zoom in` });
  const band = el('div', { className: 'zoom-band', hidden: true });
  const again = () => renderCamera(view, ctx);

  const events = data.actions.length
    ? el('ul', { className: 'events' }, data.actions.map((a) => el('li', {},
      el('span', { textContent: clock(a.t, true) }),
      el('div', {}, el('div', { className: 'what', dataset: { a: a.type }, textContent: ACTION_NAMES[a.type] || a.type }),
        el('div', { className: 'sub', textContent: `${Math.round(a.count)} birds${a.rating ? `, rated ${a.rating.level} ${ctx.farm.levels[a.rating.level].name}` : ''}` }),
        a.type === 'level1' ? el('a', { className: 'verdict-link', dataset: { v: a.verdict || '' }, href: `#/check?camera=${cam.mac}&state=all&at=${encodeURIComponent(a.event_id)}${ctx.q.day ? `&day=${ctx.q.day}` : ''}`, textContent: a.verdict ? VERDICT_TEXT[a.verdict] : 'Check it' }) : null),
      a.image ? el('button', { className: 'thumb', onclick: () => lightbox(alarmUrl(a.image), `${ACTION_NAMES[a.type]} at ${clock(a.t, true)}, ${Math.round(a.count)} birds`) }, el('img', { src: alarmUrl(a.image), alt: `${ACTION_NAMES[a.type]} picture`, loading: 'lazy' })) : el('span'))))
    : el('p', { className: 'muted', textContent: 'No alarm on this day.' });

  const shots = data.snapshots.length
    ? pictureGallery(ctx, { day: data.day, mac: cam.mac, items: data.snapshots })
    : el('p', { className: 'muted', textContent: 'No snapshots kept on this day.' });

  const order = cameraOrder(ctx.farm);
  const at = order.indexOf(cam.mac);
  const step = (d) => order[(at + d + order.length) % order.length];  // wraps: after the last, the first
  const goTo = (d) => ctx.go('camera', step(d), { day: ctx.q.day });
  if (keyHandler) document.removeEventListener('keydown', keyHandler);
  keyHandler = (e) => {
    if (!document.body.contains(camNav) || e.ctrlKey || e.metaKey || e.altKey) return;  // another page now
    if (e.target.closest('select, input, .calendar') || document.querySelector('dialog[open]')) return;
    if (e.key === 'ArrowLeft') { e.preventDefault(); goTo(-1); } else if (e.key === 'ArrowRight') { e.preventDefault(); goTo(1); }
  };
  document.addEventListener('keydown', keyHandler);
  const camNav = el('div', { className: 'cam-nav' },
    el('button', { className: 'chip', title: 'Previous camera (←)', onclick: () => goTo(-1) }, `‹ ${short(step(-1))}`),
    el('span', { className: 'muted', textContent: `${at + 1} of ${order.length}` }),
    el('button', { className: 'chip', title: 'Next camera (→)', onclick: () => goTo(1) }, `${short(step(1))} ›`));

  view.replaceChildren(
    el('div', { className: 'cam-top' },
      el('a', { className: 'back-link', href: `#/cameras${ctx.q.day ? `?day=${ctx.q.day}` : ''}`, textContent: 'All cameras' }), camNav),
    el('div', { className: 'cam-head' },
      el('h1', { textContent: short(cam.mac) }),
      el('div', { className: 'facts' },
        el('span', {}, zoneTitle(cam.group)),
        el('span', {}, 'MAC ', el('b', { textContent: cam.mac })),
        el('span', {}, 'Threshold ', el('b', { textContent: data.threshold }), `, clears below ${data.clear_level}`),
        s.seconds ? el('span', {}, 'Highest ', el('b', { textContent: Math.round(s.count_max) }), ', in alarm ', el('b', { textContent: s.alarm_seconds ? span(s.alarm_seconds) : 'never' })) : null),
      daySelect(ctx, 'camera', { arg: cam.mac })),
    el('div', { className: 'panel' },
      el('div', { className: 'panel-head' }, el('h2', { textContent: 'Count' }),
        el('span', { className: 'muted', textContent: zoomed ? `${clock(data.curve.t[0])} to ${clock(data.curve.t.at(-1))}` : 'drag across the curve to zoom in' }),
        zoomed ? el('button', { className: 'chip', textContent: 'Show the whole day', onclick: () => { zoom = null; again(); } }) : null,
        readout),
      s.seconds ? el('div', { className: 'zoomable' }, canvas, band) : el('p', { className: 'muted', textContent: 'No count recorded for this camera on this day.' })),
    el('div', { className: 'cam-grid' },
      el('div', { className: 'panel' }, el('div', { className: 'panel-head' }, el('h2', { textContent: 'Snapshots' }), el('span', { className: 'muted', textContent: "the model's level on each (hover for its reason), outside the mask darkened as it saw it; click one to see it large" })), shots),
      el('div', { className: 'panel' }, el('div', { className: 'panel-head' }, el('h2', { textContent: 'Alarm events' })), events)));
  if (s.seconds) {
    const [c0, c1] = [data.curve.t[0], data.curve.t.at(-1) + data.curve.step];
    const shown = data.actions.filter((a) => a.t >= c0 && a.t <= c1);  // a zoomed curve marks its own stretch's alarms only
    const geom = drawCurve(canvas, data.curve, data.threshold, data.clear_level, shown);
    attachReadout(canvas, data.curve, geom, readout);
    const right = () => canvas.clientWidth - geom.padR;
    const px = (e) => Math.min(Math.max(e.clientX - canvas.getBoundingClientRect().left, geom.padL), right());
    const timeAt = (e) => geom.t0 + ((px(e) - geom.padL) / (right() - geom.padL)) * (geom.t1 - geom.t0);
    let press = null;
    canvas.onpointerdown = (e) => {
      if (e.button !== 0) return;
      press = { x: e.clientX, px: px(e), t: timeAt(e) };
      canvas.setPointerCapture(e.pointerId);
    };
    canvas.onpointermove = (e) => {
      if (!press) return;
      const b = px(e);
      band.hidden = Math.abs(e.clientX - press.x) < DRAG_PX;
      band.style.left = `${Math.min(press.px, b)}px`;
      band.style.width = `${Math.abs(b - press.px)}px`;
    };
    canvas.onpointerup = (e) => {
      if (!press) return;
      const [ta, tb, moved] = [press.t, timeAt(e), Math.abs(e.clientX - press.x) >= DRAG_PX];
      press = null; band.hidden = true;
      if (moved && Math.abs(tb - ta) >= MIN_ZOOM_SEC) {
        zoom = { day: ctx.day, mac: cam.mac, t0: Math.min(ta, tb), t1: Math.max(ta, tb) };
        again();
      }
    };
    canvas.onpointercancel = () => { press = null; band.hidden = true; };
  }
}
