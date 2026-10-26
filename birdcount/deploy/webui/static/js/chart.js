// The two canvases: the day lanes (every camera's alarm periods on one time
// axis) and one camera's count curve against its threshold.

import { clock, css, minuteTicks, setupCanvas } from './util.js';

const LANE_H = 22;
const LANE_TOP = 40;
const LANE_LEFT = 168;

// rows: [{key, zone, label, threshold, periods: [[t0, t1, level]], snaps: [t], group}]
// Shows t0..t1 only (a zoomed part of the day clips to it). Alarm periods are
// bars on the lane; snapshots still to label are ticks under it, so a busy
// hour's many snapshots never hide its alarms.
export function drawLanes(canvas, rows, t0, t1) {
  const h = LANE_TOP + rows.length * LANE_H + 6;
  const { g, w } = setupCanvas(canvas, h);
  const right = w - 10;
  const x = (t) => LANE_LEFT + ((t - t0) / Math.max(1, t1 - t0)) * (right - LANE_LEFT);
  const ink = css('--ink'), ink2 = css('--ink-2'), line = css('--line');
  g.fillStyle = ink2;
  g.fillText('Zone', 0, 30); g.fillText('Camera', 50, 30); g.fillText('Thr.', 122, 30);
  g.strokeStyle = line;
  for (const t of minuteTicks(t0, t1)) {
    g.fillText(clock(t), x(t) - 16, 30);
    g.beginPath(); g.moveTo(x(t) + 0.5, LANE_TOP - 4); g.lineTo(x(t) + 0.5, h); g.stroke();
  }
  let group = null;
  rows.forEach((r, i) => {
    const y = LANE_TOP + i * LANE_H;
    if (r.group !== group) {
      if (group !== null) { g.strokeStyle = ink2; g.beginPath(); g.moveTo(0, y + 0.5); g.lineTo(w, y + 0.5); g.stroke(); }
      group = r.group;
    }
    g.fillStyle = ink2; g.fillText(r.zone, 0, y + 14);
    g.fillStyle = ink; g.fillText(r.label, 50, y + 14);
    g.fillStyle = ink2; g.fillText(String(r.threshold), 122, y + 14);
    g.fillStyle = line; g.fillRect(LANE_LEFT, y + 9, right - LANE_LEFT, 2);
    for (const [a, b, lv] of r.periods) {
      if (b < t0 || a > t1) continue;
      const xa = Math.max(LANE_LEFT, x(a)), xb = Math.min(right, x(b));
      g.fillStyle = css(`--l${lv}`);
      g.fillRect(xa, y + 3, Math.max(2, xb - xa), 14);
    }
    g.fillStyle = ink2;
    for (const t of r.snaps) if (t >= t0 && t <= t1) g.fillRect(Math.round(x(t)), y + 18, 1, 4);
  });
  return {
    row(event) {  // which row an event landed on
      const rect = canvas.getBoundingClientRect();
      return rows[Math.floor((event.clientY - rect.top - LANE_TOP) / LANE_H)] || null;
    },
    timeAt(event) {  // the time under the pointer, clamped to what is shown
      const px = event.clientX - canvas.getBoundingClientRect().left;
      return t0 + (Math.min(Math.max(px, LANE_LEFT), right) - LANE_LEFT) / (right - LANE_LEFT) * (t1 - t0);
    },
    left: LANE_LEFT,
    right,
  };
}

// c: {t, mean, low, high, level}; actions: [{t, type}]
export function drawCurve(canvas, c, threshold, clearLevel, actions, height = 280) {
  const { g, w } = setupCanvas(canvas, height);
  const padL = 44, padR = 10, padT = 12, padB = 24;
  const n = c.t.length;
  const t0 = c.t[0], t1 = c.t[n - 1] + (c.t[1] - c.t[0] || 1);
  const ymax = Math.max(threshold * 1.3, ...c.high) * 1.05;
  const X = (t) => padL + ((t - t0) / Math.max(1, t1 - t0)) * (w - padL - padR);
  const Y = (v) => padT + (1 - v / ymax) * (height - padT - padB);
  for (let i = 0; i < n; i++) {
    if (!c.level[i]) continue;
    const next = i + 1 < n ? c.t[i + 1] : t1;
    g.fillStyle = css(`--l${c.level[i]}-wash`);
    g.fillRect(X(c.t[i]), padT, X(next) - X(c.t[i]) + 0.6, height - padT - padB);
  }
  g.strokeStyle = css('--line'); g.fillStyle = css('--ink-2');
  const stepY = ymax > 300 ? 100 : 50;
  for (let v = 0; v <= ymax; v += stepY) {
    g.beginPath(); g.moveTo(padL, Y(v) + 0.5); g.lineTo(w - padR, Y(v) + 0.5); g.stroke();
    g.fillText(String(v), 6, Y(v) + 4);
  }
  for (const t of minuteTicks(t0, t1)) g.fillText(clock(t), X(t) - 16, height - 6);
  g.fillStyle = css('--band');
  g.beginPath();
  c.t.forEach((t, i) => (i ? g.lineTo(X(t), Y(c.high[i])) : g.moveTo(X(t), Y(c.high[i]))));
  for (let i = n - 1; i >= 0; i--) g.lineTo(X(c.t[i]), Y(c.low[i]));
  g.fill();
  g.strokeStyle = css('--ink'); g.lineWidth = 1.6;
  g.beginPath();
  c.t.forEach((t, i) => (i ? g.lineTo(X(t), Y(c.mean[i])) : g.moveTo(X(t), Y(c.mean[i]))));
  g.stroke();
  g.lineWidth = 1.5; g.strokeStyle = css('--l2');
  g.beginPath(); g.moveTo(padL, Y(threshold)); g.lineTo(w - padR, Y(threshold)); g.stroke();
  g.setLineDash([5, 4]); g.strokeStyle = css('--ok');
  g.beginPath(); g.moveTo(padL, Y(clearLevel)); g.lineTo(w - padR, Y(clearLevel)); g.stroke();
  g.setLineDash([]); g.lineWidth = 1;
  const tag = (text, color, y) => {  // on a backing, so the curve never runs through it
    const tw = g.measureText(text).width;
    g.fillStyle = css('--surface'); g.fillRect(padL + 4, y - 11, tw + 8, 15);
    g.fillStyle = color; g.fillText(text, padL + 8, y);
  };
  tag(`threshold ${threshold}`, css('--l2'), Y(threshold) - 5);
  tag(`clears below ${clearLevel}`, css('--ok'), Y(clearLevel) + 14);
  for (const a of actions) {
    const ax = X(a.t);
    g.strokeStyle = a.type === 'recovery' ? css('--ok') : css(`--l${a.type.slice(-1)}`);
    g.lineWidth = 2; g.beginPath(); g.moveTo(ax, padT); g.lineTo(ax, padT + 10); g.stroke(); g.lineWidth = 1;
  }
  return { X, Y, t0, t1, padL, padR, height };
}

// A readout that follows the pointer over a curve: time and count at that point.
export function attachReadout(canvas, c, geom, out) {
  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const t = geom.t0 + ((e.clientX - rect.left - geom.padL) / (rect.width - geom.padL - geom.padR)) * (geom.t1 - geom.t0);
    let i = c.t.findIndex((v) => v > t) - 1;
    if (i < 0) i = t < c.t[0] ? 0 : c.t.length - 1;
    out.textContent = `${clock(c.t[i], true)}  ${Math.round(c.mean[i])} birds (${Math.round(c.low[i])}–${Math.round(c.high[i])})${c.level[i] ? `, Level ${c.level[i]}` : ''}`;
  };
  canvas.onmouseleave = () => { out.textContent = ''; };
}
