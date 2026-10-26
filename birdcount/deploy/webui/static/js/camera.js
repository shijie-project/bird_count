// One camera's day: the count against its threshold, its alarm events with the
// pictures sent, and the snapshots kept for labelling.

import { attachReadout, drawCurve } from './chart.js';
import { daySelect } from './farm.js';
import { ACTION_NAMES, alarmUrl, api, clock, el, lightbox, short, snapshotUrl, span, zoneTitle } from './util.js';

const LABEL_TEXT = { pileup: 'pile-up', normal: 'normal', unsure: 'not sure' };

export async function renderCamera(view, ctx) {
  const data = await api(`/api/day/${ctx.day}/camera/${ctx.mac}`);
  const cam = data.camera;
  const s = data.summary;
  const readout = el('span', { className: 'muted' });
  const canvas = el('canvas', { role: 'img', 'aria-label': `Count over the day against threshold ${data.threshold}` });

  const events = data.actions.length
    ? el('ul', { className: 'events' }, data.actions.map((a) => el('li', {},
      el('span', { textContent: clock(a.t, true) }),
      el('div', {}, el('div', { className: 'what', dataset: { a: a.type }, textContent: ACTION_NAMES[a.type] || a.type }),
        el('div', { className: 'sub', textContent: `${Math.round(a.count)} birds` })),
      a.image ? el('button', { className: 'thumb', onclick: () => lightbox(alarmUrl(a.image), `${ACTION_NAMES[a.type]} at ${clock(a.t, true)}, ${Math.round(a.count)} birds`) }, el('img', { src: alarmUrl(a.image), alt: `${ACTION_NAMES[a.type]} picture`, loading: 'lazy' })) : el('span'))))
    : el('p', { className: 'muted', textContent: 'No alarm on this day.' });

  const shots = data.snapshots.length
    ? el('div', { className: 'strip' }, data.snapshots.map((sn) => el('button', {
      className: 'shot', dataset: { label: sn.label || '' },
      onclick: () => ctx.go('label', null, { day: ctx.q.day, camera: cam.mac, kind: 'high,bin', state: 'all', at: sn.path }),
    }, el('img', { src: snapshotUrl(data.day, sn.path), alt: '', loading: 'lazy' }),
    el('span', {}, `${clock(sn.t)}  `, el('b', { textContent: sn.count }), ` birds${sn.level ? `, Level ${sn.level}` : ''}  ${sn.label ? LABEL_TEXT[sn.label] : ''}`))))
    : el('p', { className: 'muted', textContent: 'No snapshots kept on this day.' });

  view.replaceChildren(
    el('a', { className: 'back-link', href: `#/cameras${ctx.q.day ? `?day=${ctx.q.day}` : ''}`, textContent: 'All cameras' }),
    el('div', { className: 'cam-head' },
      el('h1', { textContent: short(cam.mac) }),
      el('div', { className: 'facts' },
        el('span', {}, zoneTitle(cam.group)),
        el('span', {}, 'MAC ', el('b', { textContent: cam.mac })),
        el('span', {}, 'Threshold ', el('b', { textContent: data.threshold }), `, clears below ${data.clear_level}`),
        s.seconds ? el('span', {}, 'Highest ', el('b', { textContent: Math.round(s.count_max) }), ', in alarm ', el('b', { textContent: s.alarm_seconds ? span(s.alarm_seconds) : 'never' })) : null),
      daySelect(ctx, 'camera', { arg: cam.mac })),
    el('div', { className: 'panel' }, el('div', { className: 'panel-head' }, el('h2', { textContent: 'Count' }), readout), s.seconds ? canvas : el('p', { className: 'muted', textContent: 'No count recorded for this camera on this day.' })),
    el('div', { className: 'cam-grid' },
      el('div', { className: 'panel' }, el('div', { className: 'panel-head' }, el('h2', { textContent: 'Snapshots' }), el('span', { className: 'muted', textContent: 'outlined once labelled; click one to label it' })), shots),
      el('div', { className: 'panel' }, el('div', { className: 'panel-head' }, el('h2', { textContent: 'Alarm events' })), events)));
  if (s.seconds) {
    const geom = drawCurve(canvas, data.curve, data.threshold, data.clear_level, data.actions);
    attachReadout(canvas, data.curve, geom, readout);
  }
}
