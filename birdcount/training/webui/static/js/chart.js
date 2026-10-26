// One canvas chart for every plot on the page: training curves, the video
// timeline, GT-vs-prediction scatter. No library, so the panel works offline.

import { el } from './util.js';

const token = (name, fallback) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

export function withAlpha(hex, alpha) {
  const v = hex.replace('#', '');
  const n = Number.parseInt(v.length === 3 ? v.split('').map((c) => c + c).join('') : v, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

/** A grid step of 1 / 2 / 2.5 / 5 x 10^k, so tick labels stay round. */
export function niceStep(range, target = 5) {
  const raw = Math.max(range, 1e-12) / target;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / magnitude;
  return (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * magnitude;
}

export function rollingMean(values, size) {
  if (!(size > 1) || values.length < 3) return values.slice();
  const half = Math.floor(size / 2);
  const prefix = [0];
  for (const v of values) prefix.push(prefix[prefix.length - 1] + v);
  return values.map((_, i) => {
    const from = Math.max(0, i - half);
    const to = Math.min(values.length, i + half + 1);
    return (prefix[to] - prefix[from]) / (to - from);
  });
}

const defaultFormat = (v) => {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 10000) return v.toExponential(1);
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return Number(v.toFixed(2)).toString();
  return Number(v.toPrecision(2)).toString();
};

/** Nearest point index to x in points sorted by x. */
function nearest(points, x) {
  let lo = 0;
  let hi = points.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (points[mid][0] < x) lo = mid; else hi = mid;
  }
  return Math.abs(points[lo][0] - x) <= Math.abs(points[hi][0] - x) ? lo : hi;
}

export class Chart {
  constructor(host, { height = 260 } = {}) {
    this.height = height;
    this.canvas = el('canvas');
    this.tip = el('div', { className: 'chart-tip', hidden: true });
    this.host = el('div', { className: 'chart' }, this.canvas, this.tip);
    host.append(this.host);
    this.opts = null;
    this.hoverX = null;
    this.canvas.addEventListener('pointermove', (e) => this._hover(e));
    this.canvas.addEventListener('pointerleave', () => { this.hoverX = null; this.tip.hidden = true; this._draw(); });
    this.canvas.addEventListener('click', () => this._click());
    this._resize = new ResizeObserver(() => this.opts && this._draw());
    this._resize.observe(this.host);
  }

  draw(opts) {
    this.opts = opts;
    this._draw();
  }

  toPng() { return this.canvas.toDataURL('image/png'); }

  _draw() {
    const o = this.opts;
    const series = (o?.series ?? []).filter((s) => s.points?.length);
    const canvas = this.canvas;
    const width = Math.max(this.host.clientWidth, 280);
    const height = this.height;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = token('--surface', '#fff');  // painted, so a saved PNG reads anywhere
    ctx.fillRect(0, 0, width, height);
    this.layout = null;
    if (!series.length) {
      ctx.fillStyle = token('--fg-faint', '#748298');
      ctx.font = '12px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(o?.empty ?? 'No data yet', width / 2, height / 2);
      return;
    }

    const pad = { l: 56, r: 16, t: o.title ? 26 : 12, b: 30 };
    const plotW = width - pad.l - pad.r;
    const plotH = height - pad.t - pad.b;
    const xs = series.flatMap((s) => [s.points[0][0], s.points[s.points.length - 1][0]]);
    let x0 = o.xMin ?? Math.min(...xs);
    let x1 = o.xMax ?? Math.max(...xs);
    if (o.scatter) {
      x0 = Math.min(...series.flatMap((s) => s.points.map((p) => p[0])));
      x1 = Math.max(...series.flatMap((s) => s.points.map((p) => p[0])));
    }
    const ys = series.flatMap((s) => s.points.map((p) => p[1])).filter((v) => Number.isFinite(v) && (!o.logY || v > 0));
    if (o.threshold?.value) ys.push(o.threshold.value);
    let y0 = o.yMin ?? Math.min(...ys);
    let y1 = o.yMax ?? Math.max(...ys);
    if (o.scatter) { x0 = y0 = Math.min(0, x0, y0); x1 = y1 = Math.max(x1, y1) * 1.04; }
    if (x1 === x0) x1 = x0 + 1;

    let Y;
    let yTicks;
    if (o.logY) {
      const l0 = Math.floor(Math.log10(Math.max(y0, 1e-12)));
      const l1 = Math.ceil(Math.log10(Math.max(y1, 1e-12)));
      const top = l1 === l0 ? l0 + 1 : l1;
      Y = (v) => pad.t + plotH - ((Math.log10(Math.max(v, 1e-12)) - l0) / (top - l0)) * plotH;
      yTicks = [];
      for (let k = l0; k <= top; k++) {
        yTicks.push(10 ** k);
        if (top - l0 <= 2) for (const m of [2, 5]) if (k < top) yTicks.push(m * 10 ** k);
      }
    } else {
      if (!o.scatter) {
        const span = y1 - y0 || Math.abs(y1) || 1;
        y1 += span * 0.06;
        if (o.yMin == null) y0 = y0 >= 0 && y0 < span * 0.6 ? 0 : y0 - span * 0.04;
      }
      if (y1 === y0) y1 = y0 + 1;
      const step = niceStep(y1 - y0, 5);
      y0 = Math.floor(y0 / step) * step;
      y1 = Math.ceil(y1 / step) * step;
      Y = (v) => pad.t + plotH - ((v - y0) / (y1 - y0)) * plotH;
      yTicks = [];
      for (let v = y0; v <= y1 + step * 1e-6; v += step) yTicks.push(v);
    }
    const X = (v) => pad.l + ((v - x0) / (x1 - x0)) * plotW;
    this.layout = { X, Y, x0, x1, pad, plotW, plotH, series };

    ctx.font = '10px ui-monospace, monospace';
    ctx.lineWidth = 1;
    const grid = token('--chart-grid', '#e5eaf1');
    const tick = token('--chart-tick', '#64748b');
    const yFormat = o.yFormat ?? defaultFormat;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const v of yTicks) {
      const y = Y(v);
      if (y < pad.t - 1 || y > pad.t + plotH + 1) continue;
      ctx.strokeStyle = grid;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(width - pad.r, y); ctx.stroke();
      ctx.fillStyle = tick;
      ctx.fillText(yFormat(v), pad.l - 7, y);
    }
    const xTicks = o.xTicks ? o.xTicks(x0, x1, plotW) : (() => {
      const step = niceStep(x1 - x0, Math.max(2, Math.floor(plotW / 90)));
      const out = [];
      for (let v = Math.ceil(x0 / step) * step; v <= x1 + step * 1e-6; v += step) out.push(v);
      return out;
    })();
    const xFormat = o.xFormat ?? defaultFormat;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const v of xTicks) {
      const x = X(v);
      ctx.strokeStyle = grid;
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + plotH); ctx.stroke();
      ctx.fillStyle = tick;
      ctx.fillText(xFormat(v, true), x, pad.t + plotH + 7);
    }

    if (o.scatter) {
      ctx.strokeStyle = token('--chart-guide', '#b8c3d1');
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(X(x0), Y(x0)); ctx.lineTo(X(x1), Y(x1)); ctx.stroke();
      ctx.setLineDash([]);
    }
    if (o.threshold?.value && o.threshold.value < y1) {
      const y = Y(o.threshold.value);
      ctx.fillStyle = withAlpha('#c93632', 0.06);
      ctx.fillRect(pad.l, pad.t, plotW, y - pad.t);
      ctx.strokeStyle = withAlpha('#c93632', 0.75);
      ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(width - pad.r, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = withAlpha('#c93632', 0.9);
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText(o.threshold.label ?? `threshold ${o.threshold.value}`, pad.l + 6, y - 3);
    }
    if (o.vline != null && o.vline.x >= x0 && o.vline.x <= x1) {
      const x = X(o.vline.x);
      ctx.strokeStyle = withAlpha('#16834d', 0.6);
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + plotH); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#16834d';
      ctx.textAlign = x > pad.l + plotW * 0.85 ? 'right' : 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(o.vline.label ?? '', x + (ctx.textAlign === 'left' ? 4 : -4), pad.t + 2);
    }

    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    for (const s of series) {
      const pts = s.points.filter((p) => Number.isFinite(p[1]) && (!o.logY || p[1] > 0));
      if (!pts.length) continue;
      if (o.scatter) {
        ctx.fillStyle = withAlpha(s.color, 0.75);
        for (const [px, py] of pts) { ctx.beginPath(); ctx.arc(X(px), Y(py), 3, 0, Math.PI * 2); ctx.fill(); }
        continue;
      }
      if (s.fill) {
        const gradient = ctx.createLinearGradient(0, pad.t, 0, pad.t + plotH);
        gradient.addColorStop(0, withAlpha(s.color, 0.28));
        gradient.addColorStop(1, withAlpha(s.color, 0.02));
        ctx.fillStyle = gradient;
        ctx.beginPath();
        pts.forEach(([px, py], i) => (i ? ctx.lineTo(X(px), Y(py)) : ctx.moveTo(X(px), Y(py))));
        ctx.lineTo(X(pts[pts.length - 1][0]), pad.t + plotH);
        ctx.lineTo(X(pts[0][0]), pad.t + plotH);
        ctx.closePath();
        ctx.fill();
      }
      ctx.strokeStyle = s.faint ? withAlpha(s.color, 0.35) : s.color;
      ctx.lineWidth = s.width ?? (s.faint ? 1 : 1.7);
      ctx.beginPath();
      pts.forEach(([px, py], i) => (i ? ctx.lineTo(X(px), Y(py)) : ctx.moveTo(X(px), Y(py))));
      ctx.stroke();
      if (s.dots || pts.length === 1) {
        ctx.fillStyle = s.color;
        for (const [px, py] of pts) { ctx.beginPath(); ctx.arc(X(px), Y(py), 2.2, 0, Math.PI * 2); ctx.fill(); }
      }
    }
    if (o.peak) {  // the maximum, named: it is the number the eye looks for
      const s = series[o.peak.series ?? 0];
      let best = 0;
      s.points.forEach((p, i) => { if (p[1] > s.points[best][1]) best = i; });
      const [px, py] = s.points[best];
      ctx.fillStyle = s.color;
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(X(px), Y(py), 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.font = '600 10.5px ui-monospace, monospace';
      ctx.textBaseline = 'bottom';
      ctx.textAlign = X(px) > pad.l + plotW * 0.8 ? 'right' : X(px) < pad.l + plotW * 0.2 ? 'left' : 'center';
      ctx.fillText(o.peak.label(px, py), X(px), Math.max(Y(py) - 8, pad.t + 10));
    }
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillStyle = tick;
    ctx.textBaseline = 'top';
    if (o.title) { ctx.textAlign = 'left'; ctx.fillText(o.title, pad.l, 7); }
    if (o.xTitle) { ctx.textAlign = 'right'; ctx.fillText(o.xTitle, width - pad.r, 7); }

    if (this.hoverX != null && !o.scatter) this._drawHover(ctx);
  }

  _drawHover(ctx) {
    const { X, Y, pad, plotH, series } = this.layout;
    const o = this.opts;
    const primary = series[0];
    const i = nearest(primary.points, this.hoverX);
    const x = primary.points[i][0];
    ctx.strokeStyle = token('--chart-guide', '#b8c3d1');
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(X(x), pad.t); ctx.lineTo(X(x), pad.t + plotH); ctx.stroke();
    ctx.setLineDash([]);
    const rows = [];
    const tolerance = (this.layout.x1 - this.layout.x0) * 0.02;
    for (const s of series) {
      if (s.noTip) continue;
      const j = nearest(s.points, x);
      const [sx, sy] = s.points[j];
      if (Math.abs(sx - x) > tolerance || !Number.isFinite(sy)) continue;
      ctx.fillStyle = s.color;
      ctx.beginPath(); ctx.arc(X(sx), Y(sy), 3, 0, Math.PI * 2); ctx.fill();
      rows.push(el('div', { className: 't-row' }, el('span', { className: 'swatch', style: `background:${s.color}` }),
        s.label, el('b', {}, (s.format ?? o.yFormat ?? defaultFormat)(sy))));
    }
    this.hoverIndex = i;
    this.tip.replaceChildren(el('div', { className: 't-time' }, (o.tipTitle ?? o.xFormat ?? defaultFormat)(x, false, i)),
      ...rows, ...(o.tipHint ? [el('div', { className: 't-hint' }, o.tipHint(i))].filter((n) => n.textContent) : []));
    this.tip.hidden = false;
    const half = this.tip.offsetWidth / 2;
    const left = Math.min(Math.max(X(x), half + 4), this.host.clientWidth - half - 4);
    this.tip.style.left = `${left}px`;
    this.tip.style.top = `${pad.t + 6}px`;
  }

  _hover(event) {
    if (!this.layout) return;
    const { x0, x1, pad, plotW } = this.layout;
    const px = event.clientX - this.canvas.getBoundingClientRect().left;
    this.hoverX = x0 + ((px - pad.l) / plotW) * (x1 - x0);
    this._draw();
  }

  _click() {
    if (this.opts?.onClick && this.hoverX != null && this.hoverIndex != null) this.opts.onClick(this.hoverIndex);
  }
}
