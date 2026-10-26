// Small helpers every view uses: DOM building, the API, times, the lightbox.

export const $ = (sel, root = document) => root.querySelector(sel);

export function el(tag, { dataset, style, ...props } = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k.includes('-')) e.setAttribute(k, v);  // aria-*: attributes, not properties
    else e[k] = v;
  }
  if (dataset) Object.assign(e.dataset, dataset);
  if (style) e.setAttribute('style', style);
  e.append(...kids.flat().filter((k) => k !== null && k !== undefined && k !== false));
  return e;
}

export async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({ error: `${res.status} ${res.statusText}` }));
  if (!res.ok) throw new Error(data.error || `${res.status}`);
  return data;
}

export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function clock(t, seconds = false) {
  return new Date(t * 1000).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: seconds ? '2-digit' : undefined });
}

export function ago(seconds) {
  if (seconds < 90) return `${Math.max(1, Math.round(seconds))} s`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} min`;
  if (seconds < 172800) return `${Math.round(seconds / 3600)} h`;
  return `${Math.round(seconds / 86400)} days`;
}

export function span(seconds) {
  if (seconds < 60) return `${Math.round(seconds)} s`;
  const m = Math.round(seconds / 60);
  return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${m % 60} min`;
}

export const short = (mac) => mac.slice(-6);
export const levelName = (lv) => (lv ? `Level ${lv}` : 'Normal');
export const ACTION_NAMES = { level1: 'Level 1', level2: 'Level 2', level3: 'Level 3', recovery: 'Recovered' };

export function snapshotUrl(day, path) { return `/api/image?day=${day}&path=${encodeURIComponent(path)}`; }
export function alarmUrl(path) { return `/api/image?alarm=${encodeURIComponent(path)}`; }

export function zoneTitle(name) {
  // Zone_01_Right_Front -> Zone 01 Right Front; axis2 -> Recorder axis2
  if (/^axis\d+$/i.test(name)) return `Recorder ${name}`;
  return name.replace(/_/g, ' ');
}

export function lightbox(src, caption) {
  const d = $('#lightbox');
  d.querySelector('img').src = src;
  d.querySelector('p').textContent = caption || '';
  d.showModal();
  d.onclick = () => d.close();
}

let toastTimer = 0;
export function toast(text) {
  let t = $('.toast');
  if (!t) { t = el('div', { className: 'toast', role: 'status' }); document.body.append(t); }
  t.textContent = text;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.remove(), 2600);
}

export const CONFIRM_MS = 4000;  // a two-step button waits this long for its second press

// A button that asks once more before it acts: the first press arms it.
export function twoStep(label, armed, act, className = 'chip') {
  const b = el('button', { className, textContent: label });
  let timer = 0;
  b.onclick = () => {
    if (b.dataset.armed) { clearTimeout(timer); delete b.dataset.armed; b.textContent = label; act(); return; }
    b.dataset.armed = '1'; b.textContent = armed;
    timer = setTimeout(() => { delete b.dataset.armed; b.textContent = label; }, CONFIRM_MS);
  };
  return b;
}

// Start, stop or restart the runtime; the header follows it through /api/status.
export async function runtimeAction(action) {
  try {
    await api('/api/runtime', { action });
  } catch (err) {
    toast(`Cannot ${action}: ${err.message}`);
  }
  window.dispatchEvent(new Event('runtime-changed'));
}

export function setupCanvas(canvas, height) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.height = `${height}px`;
  const g = canvas.getContext('2d');
  g.scale(dpr, dpr);
  g.font = `12.5px ${css('--font')}`;
  return { g, w };
}

export function minuteTicks(t0, t1, maxTicks = 12) {
  const steps = [60, 300, 600, 900, 1800, 3600, 7200, 10800];
  const step = steps.find((s) => (t1 - t0) / s <= maxTicks) || 21600;
  const out = [];
  for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) out.push(t);
  return out;
}

// A picture with the area outside the camera's mask darkened, as the vision
// model sees it (rubric.SHADE): what is judged is what is counted. Same rule
// for black as the masks themselves: every channel below BLACK_LEVEL.
const BLACK_LEVEL = 10;
const maskCache = new Map();

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`cannot load ${src}`));
    img.src = src;
  });
}

async function maskPixels(mac, w, h) {
  const key = `${mac}:${w}x${h}`;
  if (!maskCache.has(key)) {
    maskCache.set(key, loadImage(`/api/mask/${mac}`).then((m) => {
      const c = new OffscreenCanvas(w, h);
      const g = c.getContext('2d');
      g.imageSmoothingEnabled = false;  // nearest, like read_keep_mask: the mask stays binary
      g.drawImage(m, 0, 0, w, h);
      return g.getImageData(0, 0, w, h).data;
    }));
  }
  return maskCache.get(key);
}

export async function shadedUrl(src, mac, shade) {
  const img = await loadImage(src);
  const w = img.naturalWidth, h = img.naturalHeight;
  const mask = await maskPixels(mac, w, h);
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d');
  g.drawImage(img, 0, 0);
  const frame = g.getImageData(0, 0, w, h);
  const px = frame.data;
  for (let i = 0; i < px.length; i += 4) {
    if (mask[i] < BLACK_LEVEL && mask[i + 1] < BLACK_LEVEL && mask[i + 2] < BLACK_LEVEL) {
      px[i] *= shade; px[i + 1] *= shade; px[i + 2] *= shade;
    }
  }
  g.putImageData(frame, 0, 0);
  return c.toDataURL('image/jpeg', 0.92);
}

export const prefs = {
  get(key, fallback) { try { const v = localStorage.getItem(`farm.${key}`); return v === null ? fallback : JSON.parse(v); } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(`farm.${key}`, JSON.stringify(value)); } catch { /* private mode: not remembered */ } },
};
