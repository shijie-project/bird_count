// Masks: a camera's counted area as polygons over a recent picture of it.
// Drag a point to move it, click an edge to add one, right-click a point to
// drop it, drag inside a polygon to move it all; "New polygon" draws one
// point by point. Apply writes the camera's mask file (the old one kept in its
// history); the runtime reads masks when it starts.

import { api, ago, css, el, short, snapshotUrl, toast } from './util.js';

const HIT_PX = 9;  // how close to a point or an edge a press must land
const CONFIRM_MS = 4000;  // a two-step button waits this long for its second press

let state = null;  // the page on screen: survives nothing but a redraw

function area(poly) {
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i], [x2, y2] = poly[(i + 1) % poly.length];
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a) / 2;
}

function nearestOnSegment(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const len = dx * dx + dy * dy || 1;
  const t = Math.min(1, Math.max(0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len));
  return [a[0] + t * dx, a[1] + t * dy];
}

function cameraOrder(farm) {
  const order = new Set(farm.zones.flatMap((z) => z.macs));
  for (const c of farm.cameras) order.add(c.mac);
  return [...order];
}

// A button that asks once more before it acts: the first press arms it.
function twoStep(label, armed, act, className = 'chip') {
  const b = el('button', { className, textContent: label });
  let timer = 0;
  b.onclick = () => {
    if (b.dataset.armed) { clearTimeout(timer); delete b.dataset.armed; b.textContent = label; act(); return; }
    b.dataset.armed = '1'; b.textContent = armed;
    timer = setTimeout(() => { delete b.dataset.armed; b.textContent = label; }, CONFIRM_MS);
  };
  return b;
}

export async function renderMasks(view, ctx) {
  const order = cameraOrder(ctx.farm);
  const mac = order.includes(ctx.q.camera) ? ctx.q.camera : order[0];
  if (!mac) {
    view.replaceChildren(el('div', { className: 'empty' }, el('h2', { textContent: 'No cameras in the alarm config' })));
    return;
  }
  const info = await api(`/api/masks/${mac}`);
  const [W, H] = info.size;
  const prev = state;
  const s = state = {
    polys: structuredClone(info.polygons), saved: JSON.stringify(info.polygons), undo: [],
    drawing: null, selected: info.polygons.length ? 0 : -1, drag: null, hover: null,
  };
  const dirty = () => JSON.stringify(s.polys) !== s.saved;

  const canvas = el('canvas', { className: 'mask-canvas', role: 'img', 'aria-label': `Region mask of ${mac}` });
  const img = new Image();
  if (info.picture) img.src = snapshotUrl(info.picture.day, info.picture.path);
  img.onload = () => draw();

  // --- geometry ------------------------------------------------------------------------------
  let scale = 1;
  const toMask = (e) => {
    const r = canvas.getBoundingClientRect();
    return [Math.min(W, Math.max(0, (e.clientX - r.left) / scale)), Math.min(H, Math.max(0, (e.clientY - r.top) / scale))];
  };
  const hitPoint = (p) => {
    for (let k = s.polys.length - 1; k >= 0; k--) {
      const j = s.polys[k].findIndex(([x, y]) => Math.hypot(x - p[0], y - p[1]) * scale <= HIT_PX);
      if (j >= 0) return { k, j };
    }
    return null;
  };
  const hitEdge = (p) => {
    for (let k = s.polys.length - 1; k >= 0; k--) {
      const poly = s.polys[k];
      for (let j = 0; j < poly.length; j++) {
        const q = nearestOnSegment(p, poly[j], poly[(j + 1) % poly.length]);
        if (Math.hypot(q[0] - p[0], q[1] - p[1]) * scale <= HIT_PX) return { k, j: j + 1, at: q };
      }
    }
    return null;
  };
  const inside = (p) => {
    for (let k = s.polys.length - 1; k >= 0; k--) {
      const poly = s.polys[k];
      let hit = false;
      for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const [xi, yi] = poly[i], [xj, yj] = poly[j];
        if ((yi > p[1]) !== (yj > p[1]) && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) hit = !hit;
      }
      if (hit) return k;
    }
    return -1;
  };
  const remember = () => { s.undo.push(JSON.stringify(s.polys)); if (s.undo.length > 100) s.undo.shift(); };

  // --- drawing -------------------------------------------------------------------------------
  function draw() {
    if (!document.body.contains(canvas)) return;
    const dpr = window.devicePixelRatio || 1;
    const cw = canvas.clientWidth;
    scale = cw / W;
    const ch = Math.round(H * scale);
    canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr); canvas.style.height = `${ch}px`;
    const g = canvas.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = '#000'; g.fillRect(0, 0, cw, ch);
    if (img.complete && img.naturalWidth) g.drawImage(img, 0, 0, cw, ch);
    g.save(); g.scale(scale, scale);
    g.beginPath(); g.rect(0, 0, W, H);  // the area outside every polygon, darkened as the model sees it
    for (const poly of s.polys) { poly.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.closePath(); }
    g.fillStyle = `rgba(0, 0, 0, ${1 - ctx.farm.mask_shade})`; g.fill('evenodd');
    g.restore();
    const focus = css('--focus') || '#1D5C8A';
    s.polys.forEach((poly, k) => {
      g.beginPath(); poly.forEach(([x, y], i) => (i ? g.lineTo(x * scale, y * scale) : g.moveTo(x * scale, y * scale))); g.closePath();
      g.lineWidth = k === s.selected ? 2.5 : 1.5; g.strokeStyle = k === s.selected ? '#FFD34D' : '#FFFFFF'; g.stroke();
      for (const [x, y] of poly) {
        g.fillStyle = k === s.selected ? '#FFD34D' : '#FFFFFF'; g.strokeStyle = '#000'; g.lineWidth = 1;
        g.fillRect(x * scale - 4, y * scale - 4, 8, 8); g.strokeRect(x * scale - 4, y * scale - 4, 8, 8);
      }
    });
    if (s.drawing) {
      const pts = s.hover ? [...s.drawing, s.hover] : s.drawing;
      g.beginPath(); pts.forEach(([x, y], i) => (i ? g.lineTo(x * scale, y * scale) : g.moveTo(x * scale, y * scale)));
      g.strokeStyle = focus; g.lineWidth = 2; g.setLineDash([6, 4]); g.stroke(); g.setLineDash([]);
      s.drawing.forEach(([x, y], i) => { g.fillStyle = i === 0 ? '#FFD34D' : focus; g.fillRect(x * scale - 4, y * scale - 4, 8, 8); });
    }
    if (!info.picture) { g.fillStyle = '#fff'; g.fillText('No snapshot of this camera yet: drawing on black', 12, 22); }
    side();
  }

  // --- pointer -------------------------------------------------------------------------------
  canvas.oncontextmenu = (e) => {
    e.preventDefault();
    if (s.drawing) { finishDrawing(); return; }
    const h = hitPoint(toMask(e));
    if (!h) return;
    remember();
    if (s.polys[h.k].length > 3) s.polys[h.k].splice(h.j, 1);
    else { s.polys.splice(h.k, 1); s.selected = -1; }  // a triangle losing a point is no polygon: it goes
    draw();
  };
  canvas.onpointerdown = (e) => {
    if (e.button !== 0) return;
    const p = toMask(e);
    if (s.drawing) {
      const first = s.drawing[0];
      if (s.drawing.length >= 3 && Math.hypot(first[0] - p[0], first[1] - p[1]) * scale <= HIT_PX) finishDrawing();
      else s.drawing.push(p);
      draw();
      return;
    }
    const pt = hitPoint(p);
    const edge = pt ? null : hitEdge(p);
    const body = pt || edge ? -1 : inside(p);
    if (pt) { remember(); s.selected = pt.k; s.drag = { kind: 'point', ...pt }; }
    else if (edge) { remember(); s.polys[edge.k].splice(edge.j, 0, edge.at); s.selected = edge.k; s.drag = { kind: 'point', k: edge.k, j: edge.j }; }
    else if (body >= 0) { remember(); s.selected = body; s.drag = { kind: 'body', k: body, from: p, start: structuredClone(s.polys[body]) }; }
    else s.selected = -1;
    if (s.drag) canvas.setPointerCapture(e.pointerId);
    draw();
  };
  canvas.onpointermove = (e) => {
    const p = toMask(e);
    if (s.drawing) { s.hover = p; draw(); return; }
    if (s.drag?.kind === 'point') { s.polys[s.drag.k][s.drag.j] = p; draw(); return; }
    if (s.drag?.kind === 'body') {
      const dx = p[0] - s.drag.from[0], dy = p[1] - s.drag.from[1];
      s.polys[s.drag.k] = s.drag.start.map(([x, y]) => [Math.min(W, Math.max(0, x + dx)), Math.min(H, Math.max(0, y + dy))]);
      draw();
      return;
    }
    canvas.style.cursor = hitPoint(p) ? 'grab' : hitEdge(p) ? 'copy' : inside(p) >= 0 ? 'move' : 'default';
  };
  canvas.onpointerup = () => {
    if (s.drag) { s.drag = null; draw(); }
  };

  function finishDrawing() {
    if (s.drawing && s.drawing.length >= 3) { remember(); s.polys.push(s.drawing); s.selected = s.polys.length - 1; }
    s.drawing = null; s.hover = null;
    draw();
  }

  // --- keys ----------------------------------------------------------------------------------
  if (prev?.keys) document.removeEventListener('keydown', prev.keys);
  const keys = (e) => {
    if (!document.body.contains(canvas) || e.target.closest('input, select, .calendar')) { document.removeEventListener('keydown', keys); return; }
    if (e.key === 'Escape' && s.drawing) { s.drawing = null; s.hover = null; draw(); }
    else if ((e.key === 'Delete' || e.key === 'Backspace') && s.selected >= 0 && !s.drawing) { remember(); s.polys.splice(s.selected, 1); s.selected = -1; draw(); }
    else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && s.undo.length) { e.preventDefault(); s.polys = JSON.parse(s.undo.pop()); s.selected = -1; draw(); }
  };
  s.keys = keys;
  document.addEventListener('keydown', keys);
  const onResize = () => { if (document.body.contains(canvas)) draw(); else window.removeEventListener('resize', onResize); };
  window.addEventListener('resize', onResize);

  // --- side panel ----------------------------------------------------------------------------
  const sidePanel = el('div', { className: 'mask-side' });
  const savedShare = info.keep;

  async function apply() {
    try {
      const out = await api(`/api/masks/${mac}`, { polygons: s.polys });
      toast(`Saved ${out.result.saved}${out.result.backup ? '; the old one is in the history' : ''}. Restart the runtime to use it.`);
      ctx.farm.masks = [...new Set([...(ctx.farm.masks || []), mac])];
      renderMasks(view, ctx);
    } catch (err) { toast(`Not saved: ${err.message}`); }
  }
  async function restore(name) {
    try {
      await api(`/api/masks/${mac}/restore`, { name });
      toast('Restored. Restart the runtime to use it.');
      renderMasks(view, ctx);
    } catch (err) { toast(`Not restored: ${err.message}`); }
  }

  function side() {
    const now = s.polys.reduce((n, p) => n + area(p), 0) / (W * H);
    const changed = dirty();
    sidePanel.replaceChildren(
      el('dl', { className: 'facts-list' },
        el('dt', { textContent: 'File' }), el('dd', { textContent: info.exists ? info.file : `${info.file} (none yet)` }),
        el('dt', { textContent: 'Size' }), el('dd', { textContent: `${W} × ${H}` }),
        el('dt', { textContent: 'Counted' }), el('dd', {}, `${(savedShare * 100).toFixed(1)} % saved`, changed ? el('b', { textContent: `  →  ${(now * 100).toFixed(1)} %` }) : null),
        el('dt', { textContent: 'Polygons' }), el('dd', { textContent: `${s.polys.length}, ${s.polys.reduce((n, p) => n + p.length, 0)} points` })),
      el('div', { className: 'mask-actions' },
        changed && s.polys.length ? twoStep('Apply', 'Replace the mask file?', apply, 'chip primary') : el('button', { className: 'chip', textContent: 'Apply', disabled: true }),
        el('button', { className: 'chip', textContent: 'Undo changes', disabled: !changed, onclick: () => { s.polys = JSON.parse(s.saved); s.undo = []; s.selected = -1; draw(); } }),
        el('button', { className: 'chip', textContent: s.drawing ? 'Drawing… (right-click to close)' : 'New polygon', disabled: !!s.drawing, onclick: () => { s.drawing = []; s.selected = -1; draw(); } }),
        el('button', { className: 'chip', textContent: 'Delete polygon', disabled: s.selected < 0, onclick: () => { remember(); s.polys.splice(s.selected, 1); s.selected = -1; draw(); } })),
      el('p', { className: 'muted mask-help', textContent: 'Drag a point to move it · click an edge to add a point · right-click a point to drop it · drag inside to move the polygon · Ctrl+Z undoes · Esc stops drawing.' }),
      el('div', { className: 'notice' }, el('b', { textContent: 'After Apply: ' }),
        'restart the runtime (python -m birdcount.deploy) to count with the new mask. The camera\'s counts change with it, so its threshold and the threshold suggestion are worth a new look after a few days.'),
      info.history.length ? el('div', {},
        el('h3', { className: 'mask-history-title', textContent: 'Earlier versions' }),
        el('ul', { className: 'mask-history' }, info.history.slice(0, 8).map((v) => el('li', {},
          el('span', { textContent: `${new Date(v.saved_at * 1000).toLocaleString('en-GB')}`, title: v.name }),
          el('span', { className: 'muted', textContent: ` ${ago(Date.now() / 1000 - v.saved_at)} ago` }),
          twoStep('Restore', 'Put it back?', () => restore(v.name)))))) : null);
  }

  // --- camera stepping -----------------------------------------------------------------------
  const at = order.indexOf(mac);
  const step = (d) => order[(at + d + order.length) % order.length];
  const goTo = (d) => ctx.go('masks', null, { camera: step(d) });
  const stepButton = (d, text) => {
    const b = el('button', { className: 'chip', textContent: text, title: d < 0 ? 'Previous camera' : 'Next camera' });
    b.onclick = () => {
      if (dirty() && !b.dataset.armed) { b.dataset.armed = '1'; b.textContent = 'Discard changes?'; setTimeout(() => { delete b.dataset.armed; b.textContent = text; }, CONFIRM_MS); return; }
      goTo(d);
    };
    return b;
  };
  const camSel = el('select', { 'aria-label': 'Camera' }, order.map((m) => el('option', { value: m, textContent: `${short(m)}  ${m}`, selected: m === mac })));
  camSel.onchange = () => {
    if (!dirty()) { ctx.go('masks', null, { camera: camSel.value }); return; }
    camSel.value = mac;
    toast('Apply or undo the changes first');
  };

  view.replaceChildren(
    el('div', { className: 'page-head' }, el('h1', { textContent: 'Masks' }),
      el('span', { className: 'muted', textContent: 'The part of each camera\'s picture that is counted. Bright = counted.' })),
    el('div', { className: 'cam-top' }, el('div', { className: 'cam-nav' }, camSel),
      el('div', { className: 'cam-nav' }, stepButton(-1, `‹ ${short(step(-1))}`), el('span', { className: 'muted', textContent: `${at + 1} of ${order.length}` }), stepButton(1, `${short(step(1))} ›`))),
    el('div', { className: 'panel' }, el('div', { className: 'mask-grid' }, el('div', {}, canvas,
      el('p', { className: 'muted', textContent: info.picture ? `Picture: ${info.picture.day}, ${info.picture.path.split('/').pop()}` : '' })), sidePanel)));
  draw();
}
