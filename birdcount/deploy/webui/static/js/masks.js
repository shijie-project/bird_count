// Masks: a camera's counted area as polygons over the picture it is drawn on
// (the mask's reference; until there is one, the newest count-log snapshot).
// "Fetch new picture" takes a still from the camera now and makes it the
// reference, the old one kept in the history; "Fetch all" does it for every
// camera at once. The picture on screen is replaced; the polygons stay as they are.
// Drag a point to move it, click an edge to add one, right-click a point (or
// click it and press Delete) to drop it, drag inside a polygon to move it all;
// "New polygon" draws one point by point, Backspace taking the last one back.
// Apply writes the camera's mask file (the old one kept in its
// history); the runtime reads masks when it starts, so the notice offers a restart.
// The side panel's buttons are made once per page and only change state on a
// redraw (every pointer move redraws), so an armed Apply stays armed; while a
// save is on its way Apply says so and takes no clicks, and the page that comes
// back says when the mask was saved.

import { CONFIRM_MS, api, ago, clock, css, el, runtimeAction, short, snapshotUrl, toast, twoStep } from './util.js';

const HIT_PX = 9;  // how close to a point or an edge a press must land

let state = null;  // the page on screen: survives nothing but a redraw
let lastSaved = null;  // {mac, at, what}: the save or restore the page reloaded after

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

function pictureUrl(mac, pic) {
  return pic.kind === 'reference' ? `/api/masks/${mac}/picture?v=${pic.taken}` : snapshotUrl(pic.day, pic.path);
}

function pictureText(pic) {
  if (!pic) return 'No picture of this camera yet: fetch one, or draw on black';
  if (pic.kind === 'snapshot') return `Picture: count-log snapshot ${pic.day}, ${pic.path.split('/').pop()}`;
  const day = new Date(pic.taken * 1000).toLocaleDateString('en-GB');
  return `Picture: fetched from the camera ${day} ${clock(pic.taken)}, ${ago(Date.now() / 1000 - pic.taken)} ago`;
}

function cameraOrder(farm) {
  const order = new Set(farm.zones.flatMap((z) => z.macs));
  for (const c of farm.cameras) order.add(c.mac);
  return [...order];
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
    drawing: null, selected: info.polygons.length ? 0 : -1, point: null, drag: null, hover: null,
  };  // point: the clicked point {k, j}, the one Delete drops
  const dirty = () => JSON.stringify(s.polys) !== s.saved;

  const canvas = el('canvas', { className: 'mask-canvas', role: 'img', 'aria-label': `Region mask of ${mac}` });
  const img = new Image();
  if (info.picture) img.src = pictureUrl(mac, info.picture);
  img.onload = () => draw();
  const caption = el('p', { className: 'muted', textContent: pictureText(info.picture) });
  const fetchOne = el('button', {
    className: 'chip', textContent: 'Fetch new picture', disabled: !info.can_fetch,
    title: info.can_fetch ? 'A still from this camera now, to draw on' : 'This camera has no IP in topology.yaml',
  });
  const fetchAll = el('button', {
    className: 'chip', textContent: 'Fetch all', disabled: !info.can_fetch_all,
    title: 'A still from every camera now: each mask gets a new picture to draw on',
  });
  const showPicture = (picture) => {  // the polygons on screen stay, saved or not
    info.picture = picture;
    img.src = pictureUrl(mac, picture);
    caption.textContent = pictureText(picture);
  };
  async function fetchPictures(button, busyText, work) {
    const text = button.textContent;
    fetchOne.disabled = fetchAll.disabled = true; button.textContent = busyText;
    try { await work(); } catch (err) { toast(`No picture: ${err.message}`); }
    fetchOne.disabled = !info.can_fetch; fetchAll.disabled = !info.can_fetch_all; button.textContent = text;
  }
  fetchOne.onclick = () => fetchPictures(fetchOne, 'Fetching…', async () => {
    const out = await api(`/api/masks/${mac}/picture`, {});
    if (state !== s) return;  // the page moved on to another camera meanwhile
    showPicture(out.picture);
    toast('New picture from the camera; the old one is in the history');
  });
  fetchAll.onclick = () => fetchPictures(fetchAll, `Fetching ${order.length} cameras…`, async () => {
    const out = await api('/api/masks/pictures', {});
    const failed = Object.entries(out.failed);
    toast(`New pictures for ${out.fetched.length} of ${out.fetched.length + failed.length} cameras`
      + (failed.length ? `; none from ${failed.map(([m]) => short(m)).join(', ')}` : ''));
    if (failed.length) console.warn('Fetch all: no picture from', out.failed);
    if (state !== s) return;
    if (out.fetched.includes(mac)) showPicture((await api(`/api/masks/${mac}`)).picture);
  });

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
  const dropPoint = ({ k, j }) => {
    remember();
    if (s.polys[k].length > 3) { s.polys[k].splice(j, 1); s.selected = k; }
    else { s.polys.splice(k, 1); s.selected = -1; }  // a triangle losing a point is no polygon: it goes
    s.point = null;
    draw();
  };
  const dropPolygon = () => { remember(); s.polys.splice(s.selected, 1); s.selected = -1; s.point = null; draw(); };

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
      poly.forEach(([x, y], j) => {
        const r = s.point?.k === k && s.point.j === j ? 6 : 4;  // the clicked point: bigger, outlined in the focus colour
        g.fillStyle = k === s.selected ? '#FFD34D' : '#FFFFFF'; g.strokeStyle = r > 4 ? focus : '#000'; g.lineWidth = r > 4 ? 2 : 1;
        g.fillRect(x * scale - r, y * scale - r, 2 * r, 2 * r); g.strokeRect(x * scale - r, y * scale - r, 2 * r, 2 * r);
      });
    });
    if (s.drawing) {
      const pts = s.hover ? [...s.drawing, s.hover] : s.drawing;
      g.beginPath(); pts.forEach(([x, y], i) => (i ? g.lineTo(x * scale, y * scale) : g.moveTo(x * scale, y * scale)));
      g.strokeStyle = focus; g.lineWidth = 2; g.setLineDash([6, 4]); g.stroke(); g.setLineDash([]);
      s.drawing.forEach(([x, y], i) => { g.fillStyle = i === 0 ? '#FFD34D' : focus; g.fillRect(x * scale - 4, y * scale - 4, 8, 8); });
    }
    if (!info.picture) { g.fillStyle = '#fff'; g.fillText('No picture of this camera yet: drawing on black', 12, 22); }
    side();
  }

  // --- pointer -------------------------------------------------------------------------------
  canvas.oncontextmenu = (e) => {
    e.preventDefault();
    if (s.drawing) { finishDrawing(); return; }
    const h = hitPoint(toMask(e));
    if (h) dropPoint(h);
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
    s.point = null;
    if (pt) { remember(); s.selected = pt.k; s.point = pt; s.drag = { kind: 'point', ...pt }; }
    else if (edge) { remember(); s.polys[edge.k].splice(edge.j, 0, edge.at); s.selected = edge.k; s.point = { k: edge.k, j: edge.j }; s.drag = { kind: 'point', k: edge.k, j: edge.j }; }
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
    s.drawing = null; s.hover = null; s.point = null;
    draw();
  }

  // --- keys ----------------------------------------------------------------------------------
  if (prev?.keys) document.removeEventListener('keydown', prev.keys);
  const keys = (e) => {
    if (!document.body.contains(canvas) || e.target.closest('input, select, .calendar')) { document.removeEventListener('keydown', keys); return; }
    const del = e.key === 'Delete' || e.key === 'Backspace';
    if (e.key === 'Escape' && s.drawing) { s.drawing = null; s.hover = null; draw(); }
    else if (del && s.drawing) { e.preventDefault(); s.drawing.pop(); draw(); }  // the last point placed goes
    else if (del && s.point) { e.preventDefault(); dropPoint(s.point); }
    else if (del && s.selected >= 0) { e.preventDefault(); dropPolygon(); }
    else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && s.undo.length) { e.preventDefault(); s.polys = JSON.parse(s.undo.pop()); s.selected = -1; s.point = null; draw(); }
  };
  s.keys = keys;
  document.addEventListener('keydown', keys);
  const onResize = () => { if (document.body.contains(canvas)) draw(); else window.removeEventListener('resize', onResize); };
  window.addEventListener('resize', onResize);

  // --- side panel ----------------------------------------------------------------------------
  const savedShare = info.keep;
  let busy = '';  // 'Saving…' / 'Restoring…' while a request is on its way: no second one

  // One request at a time; on success the page reloads from the file just written.
  async function write(text, request, done) {
    if (busy) return null;
    busy = text; side();
    let out;
    try { out = await request(); } catch (err) { busy = ''; side(); toast(err.message); return null; }
    lastSaved = { mac, at: Date.now() / 1000, what: done };
    if (state === s) {  // else the page moved on to another camera meanwhile
      renderMasks(view, ctx).catch((err) => toast(`${done}, but the page did not reload: ${err.message}`));
    }
    return out;
  }
  async function apply() {
    const out = await write('Saving…', () => api(`/api/masks/${mac}`, { polygons: s.polys, picture: info.picture })
      .catch((err) => { throw new Error(`Not saved: ${err.message}`); }), 'Saved');
    if (!out) return;
    toast(`Saved ${out.result.saved}${out.result.backup ? '; the old one is in the history' : ''}. Restart the runtime to use it.`);
    ctx.farm.masks = [...new Set([...(ctx.farm.masks || []), mac])];
  }
  async function restore(name) {
    const out = await write('Restoring…', () => api(`/api/masks/${mac}/restore`, { name })
      .catch((err) => { throw new Error(`Not restored: ${err.message}`); }), 'Restored');
    if (out) toast('Restored. Restart the runtime to use it.');
  }

  // Made once: a redraw only changes their state, so an armed two-step button stays armed.
  const facts = el('dl', { className: 'facts-list' });
  const applyBtn = twoStep('Apply', 'Replace the mask file?', apply, 'chip primary');
  const undoBtn = el('button', { className: 'chip', textContent: 'Undo changes', onclick: () => { s.polys = JSON.parse(s.saved); s.undo = []; s.selected = -1; s.point = null; draw(); } });
  const newBtn = el('button', { className: 'chip', onclick: () => { s.drawing = []; s.selected = -1; s.point = null; draw(); } });
  const deleteBtn = el('button', { className: 'chip', textContent: 'Delete polygon', onclick: dropPolygon });
  const history = info.history.slice(0, 8);
  const restoreBtns = history.map((v) => twoStep('Restore', 'Put it back?', () => restore(v.name)));
  const sidePanel = el('div', { className: 'mask-side' }, ...[  // a null child would print: no history, no list
    facts,
    el('div', { className: 'mask-actions' }, applyBtn, undoBtn, newBtn, deleteBtn),
    el('p', { className: 'muted mask-help', textContent: 'Drag a point to move it · click an edge to add a point · right-click a point, or click it and press Delete, to drop it · Delete with no point clicked drops the polygon · drag inside to move the polygon · Ctrl+Z undoes · while drawing, Backspace takes the last point back and Esc stops.' }),
    el('div', { className: 'notice' }, el('b', { textContent: 'After Apply: ' }),
      'restart the runtime to count with the new mask. The camera\'s counts change with it, so its threshold and the threshold suggestion are worth a new look after a few days.',
      el('div', { className: 'notice-action' }, twoStep('Restart the runtime', 'Restart now?', () => runtimeAction('restart')))),
    history.length ? el('div', {},
      el('h3', { className: 'mask-history-title', textContent: 'Earlier versions' }),
      el('ul', { className: 'mask-history' }, history.map((v, i) => el('li', {},
        el('span', { textContent: `${new Date(v.saved_at * 1000).toLocaleString('en-GB')}`, title: v.name }),
        el('span', { className: 'muted', textContent: ` ${ago(Date.now() / 1000 - v.saved_at)} ago` }),
        restoreBtns[i])))) : null,
  ].filter(Boolean));

  function side() {
    const now = s.polys.reduce((n, p) => n + area(p), 0) / (W * H);
    const changed = dirty();
    const saved = lastSaved?.mac === mac && !changed ? lastSaved : null;
    facts.replaceChildren(
      el('dt', { textContent: 'File' }), el('dd', { textContent: info.exists ? info.file : `${info.file} (none yet)` }),
      el('dt', { textContent: 'Size' }), el('dd', { textContent: `${W} × ${H}` }),
      el('dt', { textContent: 'Counted' }), el('dd', {}, `${(savedShare * 100).toFixed(1)} % saved`, changed ? el('b', { textContent: `  →  ${(now * 100).toFixed(1)} %` }) : null),
      el('dt', { textContent: 'Polygons' }), el('dd', { textContent: `${s.polys.length}, ${s.polys.reduce((n, p) => n + p.length, 0)} points` }),
      ...(saved ? [el('dt', { textContent: saved.what }), el('dd', {}, el('b', { textContent: clock(saved.at, true) }), ' · restart the runtime to use it')] : []));
    const canApply = changed && s.polys.length > 0 && !busy;
    if (!canApply) delete applyBtn.dataset.armed;
    if (!applyBtn.dataset.armed) applyBtn.textContent = busy === 'Saving…' ? busy : 'Apply';
    applyBtn.disabled = !canApply;
    applyBtn.className = canApply ? 'chip primary' : 'chip';
    undoBtn.disabled = !changed || !!busy;
    newBtn.textContent = s.drawing ? 'Drawing… (right-click to close)' : 'New polygon';
    newBtn.disabled = !!s.drawing || !!busy;
    deleteBtn.disabled = s.selected < 0 || !!busy;
    for (const b of restoreBtns) b.disabled = !!busy;
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
      el('span', { className: 'muted', textContent: 'The part of each camera\'s picture that is counted. Bright = counted.' }),
      el('div', { className: 'mask-fetch-all' }, 'New pictures from every camera', fetchAll)),
    el('div', { className: 'cam-top' }, el('div', { className: 'cam-nav' }, camSel),
      el('div', { className: 'cam-nav' }, stepButton(-1, `‹ ${short(step(-1))}`), el('span', { className: 'muted', textContent: `${at + 1} of ${order.length}` }), stepButton(1, `${short(step(1))} ›`))),
    el('div', { className: 'panel' }, el('div', { className: 'mask-grid' }, el('div', {}, canvas,
      el('div', { className: 'mask-picture' }, caption, fetchOne)), sidePanel)));
  draw();
}
