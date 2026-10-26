// The dataset browser: every image of a split with its annotated points, and
// what an evaluation run of that split predicted for it.

import { api, el, fileUrl, fmt, joinPath, naturalCompare, signed, store, toast } from './util.js';

export class DatasetBrowser {
  constructor(host) {
    this.split = store.get('data.split', 'val');
    this.evalRef = null;
    this.images = [];
    this.preds = new Map();  // name -> {pred, err}
    this.overlayDir = null;
    this.index = 0;
    this.mode = store.get('data.mode', 'points');
    this.sort = store.get('data.sort', 'name');
    this.points = [];
    this.image = new Image();
    this.image.decoding = 'async';
    this.image.onload = () => this.draw();
    this.image.onerror = () => { this.status.textContent = 'image not found'; };

    this.splitSelect = el('select', { onchange: () => this.load(this.splitSelect.value) });
    this.evalSelect = el('select', { onchange: () => this.loadEval(this.evalSelect.value || null) }, el('option', { value: '' }, 'no prediction'));
    this.search = el('input', { type: 'search', className: 'gallery-search', placeholder: 'filter by name…', spellcheck: false, oninput: () => this.renderList() });
    this.sortSelect = el('select', { onchange: () => { this.sort = this.sortSelect.value; store.set('data.sort', this.sort); this.renderList(); } },
      ['name', 'GT', 'worst error'].map((s) => el('option', { value: s, selected: s === this.sort }, `sort: ${s}`)));
    this.modeSwitch = el('div', { className: 'metric-switch' });
    this.list = el('ul', { className: 'image-list', tabIndex: 0 });
    this.canvas = el('canvas', { className: 'viewer-canvas' });
    this.caption = el('div', { className: 'viewer-caption' });
    this.status = el('span', { className: 'muted' });
    const prev = el('button', { className: 'btn btn-ghost btn-tiny', onclick: () => this.step(-1), title: '← / k' }, '‹ prev');
    const next = el('button', { className: 'btn btn-ghost btn-tiny', onclick: () => this.step(1), title: '→ / j' }, 'next ›');
    this.viewer = el('div', { className: 'viewer' }, el('div', { className: 'viewer-bar' }, prev, next, this.modeSwitch, this.status), this.canvas, this.caption);
    host.append(el('div', { className: 'panel browser-panel' },
      el('div', { className: 'panel-head' }, el('h2', {}, 'Dataset'), this.splitSelect, this.evalSelect, this.sortSelect, this.search),
      el('div', { className: 'browser' }, this.list, this.viewer)));
    this.renderMode();
    host.addEventListener('keydown', (e) => {
      if (e.target.matches('input, textarea, select')) return;
      if (e.key === 'ArrowRight' || e.key === 'j') { this.step(1); e.preventDefault(); }
      if (e.key === 'ArrowLeft' || e.key === 'k') { this.step(-1); e.preventDefault(); }
    });
    new ResizeObserver(() => this.draw()).observe(this.viewer);
  }

  renderMode() {
    const modes = [['points', 'GT points'], ['plain', 'image'], ['overlay', 'prediction overlay']];
    this.modeSwitch.replaceChildren(...modes.map(([key, label]) => el('button', {
      type: 'button', className: key === this.mode ? 'on' : '', disabled: key === 'overlay' && !this.overlayDir,
      onclick: () => { this.mode = key; store.set('data.mode', key); this.renderMode(); this.show(this.index); },
    }, label)));
  }

  async init() {
    try {
      const { splits } = await api('/api/dataset');
      this.splitSelect.replaceChildren(...splits.map((s) => el('option', { value: s.name }, `${s.name} · ${s.images}`)));
      if (!splits.some((s) => s.name === this.split) && splits.length) this.split = splits[0].name;
      this.splitSelect.value = this.split;
      await this.load(this.split);
    } catch (err) {
      this.status.textContent = err.message;
    }
  }

  async open(split, evalRef) {
    if (split && split !== this.split) { this.splitSelect.value = split; await this.load(split); }
    if (evalRef) { this.evalSelect.value = evalRef; await this.loadEval(evalRef); }
  }

  async load(split) {
    this.split = split;
    store.set('data.split', split);
    const data = await api(`/api/dataset/${encodeURIComponent(split)}`);
    this.dir = data.dir;
    this.images = data.images;
    this.preds.clear();
    this.overlayDir = null;
    const { items } = await api('/api/runs?page=eval');
    const evals = items.filter((r) => r.type === 'run' && r.split === split && r.status === 'done');
    this.evalSelect.replaceChildren(el('option', { value: '' }, 'no prediction'),
      ...evals.map((r) => el('option', { value: r.ref }, `pred: ${r.name}`)));
    this.renderMode();
    this.renderList();
    this.show(0);
  }

  async loadEval(ref) {
    this.evalRef = ref;
    this.preds.clear();
    this.overlayDir = null;
    if (ref) {
      try {
        const d = await api(`/api/run?ref=${encodeURIComponent(ref)}`);
        for (const r of d.images ?? []) this.preds.set(r.name, r);
        this.overlayDir = d.overlays?.dir ?? null;
      } catch (err) { toast(err.message); }
    }
    if (!this.overlayDir && this.mode === 'overlay') this.mode = 'points';
    this.renderMode();
    this.renderList();
    this.show(this.index);
  }

  visible() {
    const terms = this.search.value.toLowerCase().split(/\s+/).filter(Boolean);
    const rows = this.images.filter((im) => terms.every((t) => im.name.toLowerCase().includes(t)));
    const err = (im) => Math.abs(this.preds.get(im.name)?.err ?? -1);
    if (this.sort === 'GT') rows.sort((a, b) => b.gt - a.gt);
    else if (this.sort === 'worst error') rows.sort((a, b) => err(b) - err(a));
    else rows.sort((a, b) => naturalCompare(a.name, b.name));
    return rows;
  }

  renderList() {
    this.rows = this.visible();
    this.list.replaceChildren(...this.rows.map((im, i) => {
      const p = this.preds.get(im.name);
      return el('li', { dataset: { i }, onclick: () => this.show(i), title: im.name },
        el('span', { className: 'im-name' }, im.name),
        el('span', { className: 'im-stats' }, `GT ${im.gt}`, p ? ` · ${signed(p.err, 1)}` : ''));
    }));
    this.index = Math.min(this.index, Math.max(0, this.rows.length - 1));
    this.highlight();
  }

  highlight() {
    for (const li of this.list.children) li.classList.toggle('sel', Number(li.dataset.i) === this.index);
    this.list.children[this.index]?.scrollIntoView({ block: 'nearest' });
  }

  step(delta) {
    if (this.rows?.length) this.show((this.index + delta + this.rows.length) % this.rows.length);
  }

  async show(i) {
    this.index = i;
    this.highlight();
    const im = this.rows?.[i];
    if (!im) { this.caption.textContent = 'No images.'; return; }
    const p = this.preds.get(im.name);
    this.caption.textContent = `${im.name} — GT ${im.gt}${p ? ` · pred ${fmt(p.pred, 1)} · err ${signed(p.err, 1)}` : ''}`;
    this.status.textContent = `${i + 1} / ${this.rows.length}`;
    this.points = [];
    if (this.mode === 'points') {
      try {
        this.points = (await api(`/api/dataset/${encodeURIComponent(this.split)}/points?name=${encodeURIComponent(im.name)}`)).points;
      } catch (err) { this.status.textContent = err.message; }
      if (this.rows[this.index] !== im) return;  // moved on meanwhile
    }
    const src = this.mode === 'overlay' && this.overlayDir
      ? fileUrl(joinPath(this.overlayDir, `${im.name}_counts.png`))
      : fileUrl(joinPath(this.dir, im.file));
    if (this.image.src.endsWith(src)) this.draw(); else this.image.src = src;
  }

  draw() {
    const img = this.image;
    if (!img.naturalWidth) return;
    const width = Math.max(this.viewer.clientWidth - 2, 200);
    const scale = width / img.naturalWidth;
    const height = Math.round(img.naturalHeight * scale);
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = width * dpr;
    this.canvas.height = height * dpr;
    this.canvas.style.height = `${height}px`;
    const ctx = this.canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.drawImage(img, 0, 0, width, height);
    if (this.mode !== 'points' || !this.points.length) return;
    const r = Math.max(2, Math.min(4, 3 * scale * 1.5));
    ctx.lineWidth = 1;
    for (const [x, y] of this.points) {
      ctx.beginPath();
      ctx.arc(x * scale, y * scale, r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255, 40, 40, 0.85)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.stroke();
    }
  }
}
