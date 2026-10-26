// Small helpers every module uses: DOM building, the JSON API, formatting, storage.

export const $ = (selector, root = document) => root.querySelector(selector);

/** el('div', {className: 'x', onclick}, child, 'text', ...) */
export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key === 'style' && typeof value === 'string') node.style.cssText = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else if (key in node && key !== 'list') node[key] = value;
    else node.setAttribute(key, value === true ? '' : value);
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export async function api(path, { method = 'GET', body, raw } = {}) {
  const options = { method, headers: {} };
  if (raw !== undefined) {
    options.body = raw;
    options.headers['Content-Type'] = 'application/octet-stream';
  } else if (method !== 'GET') {
    options.body = JSON.stringify(body ?? {});
    options.headers['Content-Type'] = 'application/json';
  }
  const response = await fetch(path, options);
  let data = null;
  try { data = await response.json(); } catch { /* empty or not JSON */ }
  if (!response.ok) throw new Error(data?.error ?? `${response.status} ${response.statusText}`);
  return data;
}

let toastTimer = null;
export function toast(message, kind = 'error') {
  const node = $('#toast');
  node.textContent = message;
  node.dataset.kind = kind;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, kind === 'error' ? 7000 : 3500);
}

/** localStorage that never throws (private windows, blocked storage). */
export const store = {
  get(key, fallback = null) {
    try { const v = localStorage.getItem(`birdcount.panel.${key}`); return v == null ? fallback : JSON.parse(v); } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(`birdcount.panel.${key}`, JSON.stringify(value)); } catch { /* ignore */ }
  },
};

export const fileUrl = (path) => `/api/file?path=${encodeURIComponent(path)}`;
/** dir + name, with the separator `dir` already uses (paths come from Windows or POSIX). */
export function joinPath(dir, name) {
  const sep = dir.includes('\\') && !dir.includes('/') ? '\\' : '/';
  return `${dir.replace(/[\\/]+$/, '')}${sep}${name.replace(/[\\/]/g, sep)}`;
}

export function fmt(value, digits = 2) {
  if (value == null || Number.isNaN(value)) return '—';
  return Number(value).toFixed(digits);
}
export const pct = (value, digits = 1) => (value == null ? '—' : `${(value * 100).toFixed(digits)}%`);
export const signed = (value, digits = 1) => (value == null ? '—' : `${value > 0 ? '+' : ''}${value.toFixed(digits)}`);

export function duration(seconds) {
  if (seconds == null) return '—';
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

export function when(epochSeconds) {
  if (!epochSeconds) return '';
  const d = new Date(epochSeconds * 1000);
  const pad = (n) => String(n).padStart(2, '0');
  const today = new Date().toDateString() === d.toDateString();
  return `${today ? '' : `${d.getMonth() + 1}/${d.getDate()} `}${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** Case-insensitive name order with numbers compared as numbers. */
export const naturalCompare = (a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });

export function card(key, value, title) {
  return el('div', { className: 'card', title }, el('div', { className: 'k' }, key), el('div', { className: 'v' }, value));
}

export function openLightbox(src, caption) {
  $('#lightbox-img').src = src;
  $('#lightbox-cap').textContent = caption ?? '';
  $('#lightbox').hidden = false;
}

export function debounce(fn, ms) {
  let timer = null;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
}

/** A sortable table: columns [{key, label, format?, className?}], rows of objects. */
export function sortableTable(columns, rows, { sortKey, desc = true, onRow, limit = 2000 } = {}) {
  const state = { key: sortKey ?? columns[0].key, desc };
  const table = el('table');
  const head = el('tr');
  const body = el('tbody');
  table.append(el('thead', {}, head), body);
  const render = () => {
    const sorted = [...rows].sort((a, b) => {
      const x = a[state.key];
      const y = b[state.key];
      const order = typeof x === 'string' || typeof y === 'string'
        ? naturalCompare(String(x ?? ''), String(y ?? ''))
        : (x ?? -Infinity) - (y ?? -Infinity);
      return state.desc ? -order : order;
    });
    body.replaceChildren(...sorted.slice(0, limit).map((row) => {
      const tr = el('tr', { onclick: onRow ? () => onRow(row) : null, className: onRow ? 'clickable' : '' },
        columns.map((c) => el('td', {
          className: typeof c.className === 'function' ? c.className(row) : c.className ?? '',
          title: typeof row[c.key] === 'string' ? row[c.key] : null,
        }, c.format ? c.format(row[c.key], row) : row[c.key] ?? '—')));
      return tr;
    }));
    for (const th of head.children) {
      th.classList.toggle('sorted', th.dataset.key === state.key);
      th.classList.toggle('asc', th.dataset.key === state.key && !state.desc);
    }
  };
  for (const c of columns) {
    head.append(el('th', {
      dataset: { key: c.key },
      title: c.title,
      onclick: () => {
        state.desc = state.key === c.key ? !state.desc : c.key !== 'name' && c.key !== 'image' && c.key !== 'run';
        state.key = c.key;
        render();
      },
    }, c.label));
  }
  render();
  return el('div', { className: 'table-wrap' }, table);
}
