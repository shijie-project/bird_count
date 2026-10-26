// What a run looks like on the page, one view per kind of run. Each view is
// built once per selected run and then fed the increments of later polls.

import { Chart, rollingMean } from './chart.js';
import { card, duration, el, fileUrl, fmt, joinPath, openLightbox, pct, signed, sortableTable, store } from './util.js';

const panel = (title, ...children) => el('div', { className: 'panel' }, el('div', { className: 'panel-head' }, el('h2', {}, title)), ...children);

function segmented(options, active, onPick) {
  const host = el('div', { className: 'metric-switch', role: 'group' });
  const render = (current) => {
    host.replaceChildren(...options.map(([key, label, color]) => el('button', {
      type: 'button', className: key === current ? 'on' : '',
      onclick: () => { render(key); onPick(key); },
    }, color ? el('span', { className: 'swatch', style: `background:${color}` }) : null, label)));
  };
  render(active);
  return host;
}

function toggles(series, state, onToggle) {
  return el('div', { className: 'chart-toggles' }, series.map((s) => el('button', {
    type: 'button', className: state[s.id] ?? s.on ? 'on' : '',
    onclick: (e) => { state[s.id] = !(state[s.id] ?? s.on); e.currentTarget.classList.toggle('on'); onToggle(); },
  }, el('span', { className: 'swatch', style: `background:${s.color}` }), s.label)));
}

// --- training --------------------------------------------------------------------------------

const TRAIN_GROUPS = {
  error: {
    label: 'Error', log: true,
    series: [
      { id: 'train_mae', label: 'train MAE', color: '#2563eb', on: true, get: (r) => r.train?.mae },
      { id: 'train_rmse', label: 'train RMSE', color: '#7c3aed', on: false, get: (r) => r.train?.rmse },
      { id: 'val_mae', label: 'val MAE', color: '#16834d', on: true, val: true, get: (r) => r.val?.mae },
      { id: 'val_rmse', label: 'val RMSE', color: '#c2410c', on: true, val: true, get: (r) => r.val?.rmse },
    ],
  },
  accuracy: {
    label: 'Validation accuracy', log: false, format: (v) => `${(v * 100).toFixed(1)}%`,
    series: [
      { id: 'count_acc', label: 'counting accuracy', color: '#16834d', on: true, val: true, get: (r) => r.val?.counting_accuracy },
      { id: 'primary_acc', label: 'Acc@(5,10%)', color: '#2563eb', on: true, val: true, get: (r) => r.val?.primary_accuracy },
    ],
  },
  loss: {
    label: 'Loss', log: true,
    series: [
      { id: 'loss', label: 'loss', color: '#c93632', on: true, get: (r) => r.train?.loss },
      { id: 'count', label: 'count', color: '#2563eb', on: false, get: (r) => r.train?.count },
      { id: 'ot', label: 'OT', color: '#7c3aed', on: false, get: (r) => r.train?.ot },
      { id: 'tv', label: 'TV', color: '#a56408', on: false, get: (r) => r.train?.tv },
      { id: 'aux', label: 'aux', color: '#16834d', on: false, get: (r) => r.train?.aux },
    ],
  },
  lr: { label: 'Learning rate', log: false, series: [{ id: 'lr', label: 'lr', color: '#2563eb', on: true, get: (r) => r.lr }] },
};

export class TrainView {
  constructor(host, ctx) {
    this.ctx = ctx;
    this.records = [];
    this.group = store.get('train.group', 'error');
    this.on = store.get('train.series', {});
    this.log = store.get('train.log', {});
    this.cards = el('div', { className: 'cards' });
    this.switch = segmented(Object.entries(TRAIN_GROUPS).map(([k, g]) => [k, g.label]), this.group, (k) => {
      this.group = k; store.set('train.group', k); this.renderControls(); this.drawChart();
    });
    this.controls = el('div', { className: 'chart-controls' });
    this.chartHost = el('div', { className: 'chart-host' });
    this.extra = el('div');
    host.append(el('div', { className: 'panel' },
      el('div', { className: 'panel-head' }, el('h2', {}, 'Training'), this.switch),
      this.cards, this.controls, this.chartHost), this.extra);
    this.chart = new Chart(this.chartHost, { height: 280 });
    this.renderControls();
  }

  renderControls() {
    const group = TRAIN_GROUPS[this.group];
    const logBox = el('input', { type: 'checkbox', checked: this.log[this.group] ?? group.log });
    logBox.addEventListener('change', () => { this.log[this.group] = logBox.checked; store.set('train.log', this.log); this.drawChart(); });
    this.controls.replaceChildren(
      toggles(group.series, this.on, () => { store.set('train.series', this.on); this.drawChart(); }),
      el('label', { className: 'check' }, logBox, 'log scale'));
  }

  update(d) {
    this.detail = d;
    if (d.records?.length) this.records.push(...d.records);
    const best = d.summary?.best ?? d.best;
    const last = this.records[this.records.length - 1];
    const epochTime = this.records.length > 2
      ? this.records.slice(-20).reduce((a, r) => a + (r.seconds ?? 0), 0) / Math.min(20, this.records.length) : null;
    this.cards.replaceChildren(
      card('Epochs', `${d.epochs?.[0] ?? 0} / ${d.epochs?.[1] ?? '?'}`),
      card('Best epoch', best?.epoch ?? '—'),
      card('Counting accuracy', pct(best?.counting_accuracy), 'of the best validation'),
      card('Acc@(5,10%)', pct(best?.primary_accuracy), 'images within 5 birds or 10%'),
      card('Best MAE', fmt(best?.mae)),
      card('Best RMSE', fmt(best?.rmse)),
      card('Train MAE', fmt(last?.train?.mae)),
      card('s / epoch', epochTime ? fmt(epochTime, 1) : '—'),
      ...(d.summary ? [card('Wall time', duration(d.summary.seconds))] : []),
    );
    this.drawChart();
    this.renderExtra(d);
  }

  drawChart() {
    const group = TRAIN_GROUPS[this.group];
    const series = group.series.filter((s) => this.on[s.id] ?? s.on).map((s) => ({
      label: s.label, color: s.color, dots: s.val && this.records.filter((r) => r.val).length < 40,
      points: this.records.map((r) => [r.epoch, s.get(r)]).filter((p) => p[1] != null),
    }));
    const best = this.detail?.summary?.best ?? this.detail?.best;
    this.chart.draw({
      series, logY: this.log[this.group] ?? group.log, yFormat: group.format,
      xFormat: (x, axis) => (axis ? String(Math.round(x)) : `epoch ${Math.round(x)}`),
      vline: best ? { x: best.epoch, label: `best ${best.epoch}` } : null,
      empty: 'No epochs yet: curves appear as epochs finish.',
    });
  }

  renderExtra(d) {
    if (this.extraFor === `${d.status}:${d.checkpoints?.length}`) return;
    this.extraFor = `${d.status}:${d.checkpoints?.length}`;
    const actions = el('div', { className: 'actions' });
    if (d.status !== 'running' && d.status !== 'done' && this.records.length) {
      actions.append(el('button', { className: 'btn', onclick: () => this.ctx.resume(d.path) }, 'Resume this run'));
    }
    const ckpts = (d.checkpoints ?? []).filter((n) => n.startsWith('best_')).reverse();
    const list = el('ul', { className: 'ckpt-list' }, ckpts.map((name) => el('li', {},
      el('code', {}, name),
      el('button', { className: 'btn btn-ghost btn-tiny', onclick: () => this.ctx.evaluate(joinPath(d.path, name)) }, 'evaluate'))));
    const config = d.config ? el('details', { className: 'config' }, el('summary', {}, 'config.json'),
      el('table', {}, el('tbody', {}, Object.entries(d.config).map(([k, v]) => el('tr', {}, el('td', {}, k), el('td', {}, JSON.stringify(v))))))) : null;
    this.extra.replaceChildren(panel('Checkpoints', actions.childElementCount ? actions : null,
      ckpts.length ? list : el('div', { className: 'empty' }, 'No checkpoint kept yet.'), config));
  }
}

// --- sweeps ------------------------------------------------------------------------------------

export class SweepView {
  constructor(host, ctx) { this.ctx = ctx; this.host = host; }

  update(d) {
    const rows = d.leaderboard ?? [];
    const table = sortableTable([
      { key: 'run', label: 'run' },
      { key: 'status', label: 'status' },
      { key: 'epoch', label: 'best epoch' },
      { key: 'counting_accuracy', label: 'counting acc', format: (v) => pct(v) },
      { key: 'primary_accuracy', label: 'Acc@(5,10%)', format: (v) => pct(v) },
      { key: 'mae', label: 'MAE', format: (v) => fmt(v) },
      { key: 'rmse', label: 'RMSE', format: (v) => fmt(v) },
      { key: 'score', label: '2·RMSE+MAE', format: (v) => fmt(v) },
    ], rows, { sortKey: 'score', desc: false, onRow: (r) => this.ctx.openRun(`${d.ref}/${r.run}`) });
    this.host.replaceChildren(panel(`Leaderboard · ${rows.length} runs`, rows.length ? table : el('div', { className: 'empty' }, 'No run has finished a validation yet.')));
  }
}

// --- evaluation -----------------------------------------------------------------------------------

const GALLERY_PAGE = 120;

export class EvalView {
  constructor(host, ctx) { this.ctx = ctx; this.host = host; this.shown = null; }

  update(d) {
    const key = `${d.status}:${d.images?.length}:${d.overlays?.kinds?.join()}`;
    if (key === this.shown) return;  // an evaluation's files only change while it runs
    this.shown = key;
    const m = d.metrics;
    const o = m?.overall ?? {};
    const primary = m?.tolerance_accuracy?.primary;
    const pile = m?.pileup_detection;
    const children = [];
    if (m) {
      children.push(panel(`Evaluation · ${m.run?.split} · ${m.run?.images} images`,
        el('div', { className: 'cards' },
          card('Counting accuracy', pct(o.counting_accuracy), '1 − Σ|pred − GT| / ΣGT'),
          card('Acc@(5,10%)', pct(primary?.accuracy), 'images within 5 birds or 10%'),
          card('MAE', fmt(o.mae)), card('RMSE', fmt(o.rmse)), card('Bias', signed(o.bias, 2)),
          card('R²', fmt(o.r2, 3)),
          ...(pile ? [card(`Pile-up F1 (>${pile.threshold})`, fmt(pile.f1, 3))] : []),
          card('Checkpoint', (m.run?.checkpoint ?? '').split(/[\\/]/).pop(), m.run?.checkpoint),
          card('Precision', m.run?.precision ?? '—'), card('Time', duration(m.run?.seconds))),
        this._actions(d)));
    } else {
      children.push(panel('Evaluation', el('div', { className: 'empty' }, d.status === 'running' ? 'Evaluating… results appear when it finishes.' : 'This run wrote no metrics.json.')));
    }
    const images = d.images ?? [];
    if (images.length) {
      const scatterHost = el('div', { className: 'chart-host' });
      const overlay = (row, suffix = 'counts') => (d.overlays ? fileUrl(joinPath(d.overlays.dir, `${row.name}_${suffix}.png`)) : null);
      const table = sortableTable([
        { key: 'name', label: 'image' },
        { key: 'gt', label: 'GT', format: (v) => fmt(v, 0) },
        { key: 'pred', label: 'pred', format: (v) => fmt(v, 1) },
        { key: 'err', label: 'err', format: (v) => signed(v, 1), className: (r) => (Math.abs(r.err) > Math.max(5, 0.1 * r.gt) ? 'bad' : '') },
        { key: 'rel', label: 'err %', format: (v) => (v == null ? '—' : signed(v, 1)) },
      ], images.map((r) => ({ ...r, abs: Math.abs(r.err) })), {
        sortKey: 'abs', onRow: (r) => overlay(r) && openLightbox(overlay(r), `${r.name} — GT ${fmt(r.gt, 0)} · pred ${fmt(r.pred, 1)} · err ${signed(r.err, 1)}`),
      });
      children.push(el('div', { className: 'split-2' }, panel('GT vs prediction', scatterHost), panel('Per image', table)));
      requestAnimationFrame(() => new Chart(scatterHost, { height: 260 }).draw({
        scatter: true, series: [{ label: 'images', color: '#2563eb', points: images.map((r) => [r.gt, r.pred]) }],
        title: 'prediction', xTitle: 'GT',
      }));
    }
    for (const [key, data] of Object.entries(d.regions ?? {})) {
      const suffix = key === 'blobs' ? 'blobs' : 'grid';
      children.push(panel(`Worst ${key === 'blobs' ? 'blobs' : 'grid tiles'} · ${data.total} in all`, sortableTable([
        { key: 'image', label: 'image' },
        { key: key === 'blobs' ? 'id' : 'tile', label: key === 'blobs' ? 'blob' : 'tile' },
        { key: 'gt', label: 'GT', format: (v) => fmt(v, 0) },
        { key: 'count', label: 'pred', format: (v) => fmt(v, 1) },
        { key: 'error', label: 'err', format: (v) => signed(v, 1) },
      ], data.worst.map((r) => ({ ...r, tile: r.row != null ? `r${r.row}c${r.col}` : null, abs: Math.abs(r.error) })), {
        sortKey: 'abs', onRow: (r) => d.overlays && openLightbox(fileUrl(joinPath(d.overlays.dir, `${r.image}_${suffix}.png`)), r.image),
      })));
    }
    if (d.overlays && images.length) children.push(this._gallery(d, images));
    this.host.replaceChildren(...children);
  }

  _actions(d) {
    const blobs = d.regions?.blobs ? joinPath(d.path, 'blobs.json') : null;
    const row = el('div', { className: 'actions' },
      el('a', { className: 'btn btn-ghost btn-tiny', href: fileUrl(joinPath(d.path, 'per_image.csv')), download: 'per_image.csv' }, 'per_image.csv'),
      el('a', { className: 'btn btn-ghost btn-tiny', href: fileUrl(joinPath(d.path, 'metrics.json')), target: '_blank' }, 'metrics.json'),
      blobs ? el('button', { className: 'btn btn-tiny', onclick: () => this.ctx.prelabel(blobs) }, 'Pre-label in Label Studio') : null,
      el('button', { className: 'btn btn-tiny', onclick: () => this.ctx.browse(d.metrics?.run?.split, d.ref) }, 'Browse in the dataset'));
    return row;
  }

  _gallery(d, images) {
    const kinds = d.overlays.kinds;
    let kind = kinds.includes('counts') ? 'counts' : kinds[0];
    let shown = GALLERY_PAGE;
    const search = el('input', { type: 'search', className: 'gallery-search', placeholder: 'search image name…', spellcheck: false });
    const grid = el('div', { className: 'gallery' });
    const note = el('span', { className: 'muted' });
    const more = el('button', { className: 'btn btn-ghost', hidden: true, onclick: () => { shown += GALLERY_PAGE; render(); } }, 'show more');
    const sorted = [...images].sort((a, b) => Math.abs(b.err) - Math.abs(a.err));
    const render = () => {
      const terms = search.value.toLowerCase().split(/\s+/).filter(Boolean);
      const hits = sorted.filter((r) => terms.every((t) => r.name.toLowerCase().includes(t)));
      grid.replaceChildren(...hits.slice(0, shown).map((r) => {
        const src = fileUrl(joinPath(d.overlays.dir, `${r.name}_${kind}.png`));
        const caption = `GT ${fmt(r.gt, 0)} · pred ${fmt(r.pred, 1)} · err ${signed(r.err, 1)}`;
        return el('figure', { onclick: () => openLightbox(src, `${r.name} — ${caption}`) },
          el('img', { src, loading: 'lazy', alt: r.name }),
          el('figcaption', {}, el('span', { className: 'gallery-name', title: r.name }, r.name), el('span', {}, caption)));
      }));
      note.textContent = `${hits.length} images, worst first`;
      more.hidden = hits.length <= shown;
    };
    search.addEventListener('input', () => { shown = GALLERY_PAGE; render(); });
    const sw = kinds.length > 1 ? segmented(kinds.map((k) => [k, k]), kind, (k) => { kind = k; render(); }) : null;
    render();
    return el('div', { className: 'panel' }, el('div', { className: 'panel-head gallery-head' }, el('h2', {}, 'Overlays'), sw,
      el('div', { className: 'gallery-tools' }, search, note)), grid, el('div', { className: 'more' }, more));
  }
}

// --- video ---------------------------------------------------------------------------------------------

const VIDEO_METRICS = {
  peak: { label: 'Busiest patch', color: '#c2410c', digits: 2, index: 3 },
  count: { label: 'Flock count', color: '#2563eb', digits: 0, index: 2 },
};
const TIME_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400, 21600];
const pad2 = (v) => String(v).padStart(2, '0');

export class VideoView {
  constructor(host, ctx) {
    this.ctx = ctx;
    this.samples = [];
    this.metric = store.get('video.metric', 'peak');
    this.smooth = store.get('video.smooth', true);
    this.sub = el('p', { className: 'sub' });
    const smoothBox = el('input', { type: 'checkbox', checked: this.smooth });
    smoothBox.addEventListener('change', () => { this.smooth = smoothBox.checked; store.set('video.smooth', this.smooth); this.drawChart(); });
    this.switch = segmented(Object.entries(VIDEO_METRICS).map(([k, m]) => [k, m.label, m.color]), this.metric, (k) => {
      this.metric = k; store.set('video.metric', k); this.drawChart();
    });
    this.links = el('div', { className: 'timeline-tools' });
    this.chartHost = el('div', { className: 'chart-host' });
    this.cards = el('div', { className: 'cards' });
    this.extra = el('div');
    host.append(el('div', { className: 'panel' },
      el('div', { className: 'panel-head' }, el('h2', {}, 'Timeline'), this.switch,
        el('label', { className: 'check' }, smoothBox, 'smooth'), this.links),
      this.sub, this.chartHost, this.cards), this.extra);
    this.chart = new Chart(this.chartHost, { height: 300 });
  }

  clock(seconds, withSeconds = true) {
    const start = this.start;
    if (start) {
      const at = new Date(start.getTime() + seconds * 1000);
      return `${pad2(at.getHours())}:${pad2(at.getMinutes())}${withSeconds ? `:${pad2(at.getSeconds())}` : ''}`;
    }
    const s = Math.max(0, Math.round(seconds));
    return `${Math.floor(s / 3600)}:${pad2(Math.floor((s % 3600) / 60))}${withSeconds ? `:${pad2(s % 60)}` : ''}`;
  }

  update(d) {
    this.detail = d;
    const summary = d.summary ?? {};
    const settings = summary.settings ?? {};
    const parsed = settings.start_time ? new Date(settings.start_time.replace(' ', 'T')) : null;
    this.start = parsed && !Number.isNaN(parsed.getTime()) ? parsed : null;
    if (d.samples?.length) this.samples.push(...d.samples);
    const v = summary.video ?? {};
    this.sub.textContent = [
      v.name, v.width ? `${v.width}×${v.height} @ ${v.fps} fps` : null,
      `${this.samples.length} samples${settings.interval ? ` every ${settings.interval}s` : ''}`,
      settings.window_px ? `patch ${settings.window_px}px` : null,
      settings.masked === false ? 'UNMASKED' : null,
      d.status === 'running' ? 'sampling…' : null,
    ].filter(Boolean).join(' · ');
    this.renderCards(summary);
    this.renderLinks(d, summary);
    this.drawChart();
    this.renderExtra(d, summary);
  }

  renderCards(summary) {
    const s = summary.summary;
    if (!s) {
      const counts = this.samples.map((x) => x[2]);
      this.cards.replaceChildren(card('Samples', this.samples.length), card('Max count so far', counts.length ? fmt(Math.max(...counts), 0) : '—'));
      return;
    }
    this.cards.replaceChildren(
      card('Peak patch', fmt(s.peak.max, 2), 'most birds inside one patch'), card('Peak at', s.peak.max_clock),
      card('Max count', fmt(s.count.max, 0)), card('Max count at', s.count.max_clock),
      card('Mean count', fmt(s.count.mean, 0)), card('p95 count', fmt(s.count.p95, 0)),
      card('Samples', s.samples), card('Analysis time', duration(summary.seconds)));
  }

  renderLinks(d, summary) {
    const a = summary.artifacts ?? {};
    const link = (name, label) => el('a', { className: 'btn btn-ghost btn-tiny', href: fileUrl(joinPath(d.path, name)), target: '_blank' }, label);
    this.links.replaceChildren(
      ...(a.chart ? [link(a.chart, 'figure')] : []), ...(a.csv ? [link(a.csv, 'CSV')] : []),
      el('button', { className: 'btn btn-ghost btn-tiny', onclick: () => {
        const anchor = el('a', { href: this.chart.toPng(), download: `${summary.video?.name ?? 'timeline'}.png` });
        document.body.append(anchor); anchor.click(); anchor.remove();
      } }, 'save chart'));
  }

  drawChart() {
    const d = this.detail;
    if (!d) return;
    const m = VIDEO_METRICS[this.metric];
    const settings = d.summary?.settings ?? {};
    const raw = this.samples.map((s) => [s[1], s[m.index]]);
    const series = [];
    if (this.smooth && raw.length > 2) {
      const smoothed = rollingMean(raw.map((p) => p[1]), Math.max(Number(settings.smooth) || 5, 3));
      series.push({ label: 'smoothed', color: m.color, width: 2, fill: true, points: raw.map((p, i) => [p[0], smoothed[i]]), format: (y) => y.toFixed(m.digits) });
      series.push({ label: m.label, color: m.color, faint: true, points: raw, format: (y) => y.toFixed(m.digits) });
    } else {
      series.push({ label: m.label, color: m.color, fill: true, points: raw, format: (y) => y.toFixed(m.digits) });
    }
    const threshold = settings.thresholds?.[this.metric];
    const frames = d.summary?.artifacts?.frames;
    this.chart.draw({
      series, yMin: 0,
      title: this.metric === 'peak' ? `birds in the busiest ${settings.window_px ?? ''}px patch` : 'birds in the frame',
      xTitle: this.start ? 'clock time' : 'elapsed',
      threshold: threshold ? { value: threshold } : null,
      peak: { series: series.length - 1, label: (x, y) => `${y.toFixed(m.digits)} @ ${this.clock(x)}` },
      xFormat: (x, axis) => this.clock(x, !axis || this._step < 60),
      xTicks: (x0, x1, width) => {
        const span = x1 - x0;
        const target = Math.max(2, Math.floor(width / 80));
        const step = TIME_STEPS.find((c) => span / c <= target) ?? Math.ceil(span / target);
        this._step = step;
        const offset = this.start ? this.start.getHours() * 3600 + this.start.getMinutes() * 60 + this.start.getSeconds() : 0;
        const out = [];
        for (let t = Math.ceil((x0 + offset) / step) * step - offset; t <= x1; t += step) if (t >= x0) out.push(t);
        return out;
      },
      tipTitle: (x) => this.clock(x),
      tipHint: frames ? () => 'click to open this frame' : null,
      onClick: frames ? (i) => {
        const s = this.samples[i];
        const name = `frame_${String(s[0]).padStart(5, '0')}_${String(Math.floor(s[1])).padStart(6, '0')}s.jpg`;
        openLightbox(fileUrl(joinPath(joinPath(d.path, frames.dir), name)), `${this.clock(s[1])} · count ${fmt(s[2], 0)} · peak ${fmt(s[3], 2)}`);
      } : null,
      empty: d.status === 'running' ? 'Reading the video: the chart fills in as frames are sampled.' : 'No samples.',
    });
  }

  renderExtra(d, summary) {
    const a = summary.artifacts ?? {};
    const key = `${(a.peaks ?? []).length}:${a.video}`;
    if (key === this.extraFor) return;
    this.extraFor = key;
    const children = [];
    if (a.peaks?.length) {
      children.push(panel(`Busiest moments · ${a.peaks.length}`, el('div', { className: 'gallery' }, a.peaks.map((p) => {
        const src = fileUrl(joinPath(d.path, p.path));
        const caption = `${p.clock} · peak ${fmt(p.peak, 2)} · count ${fmt(p.count, 0)}`;
        return el('figure', { onclick: () => openLightbox(src, `${summary.video?.name} — ${caption}`) },
          el('img', { src, loading: 'lazy', alt: caption }), el('figcaption', {}, el('span', { className: 'gallery-name' }, p.clock), caption));
      }))));
    }
    if (a.video) {
      children.push(panel(`Rendered video · ${a.video}`, el('video', { className: 'player', controls: true, preload: 'metadata', src: fileUrl(joinPath(d.path, a.video)) })));
    }
    this.extra.replaceChildren(...children);
  }
}

export const VIEWS = { train: TrainView, sweep: SweepView, eval: EvalView, video: VideoView };
