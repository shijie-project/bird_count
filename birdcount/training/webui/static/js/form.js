// The form of one tool, generated from its argparse spec (GET /api/tools/<key>).

import { api, el, store, toast } from './util.js';

const UPLOADS = { json: { kind: 'json', accept: '.json' }, video: { kind: 'video', accept: 'video/*,.mkv,.ts' } };
const pickerCache = new Map();  // kind -> {at, items}
const PICKER_TTL = 30_000;

async function pickerItems(kind, refresh = false) {
  const hit = pickerCache.get(kind);
  if (!refresh && hit && Date.now() - hit.at < PICKER_TTL) return hit;
  const data = await api(`/api/pickers/${kind}${refresh ? '?refresh=1' : ''}`);
  const entry = { at: Date.now(), items: data.items ?? [], error: data.error };
  pickerCache.set(kind, entry);
  return entry;
}

export function forgetPickers() { pickerCache.clear(); }

function upload(file, kind, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `/api/upload?kind=${kind}&name=${encodeURIComponent(file.name)}`);
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status === 200) resolve(data); else reject(new Error(data?.error ?? `upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error('upload failed'));
    xhr.send(file);
  });
}

const asText = (value) => (Array.isArray(value) ? value.join('\n') : value == null ? '' : String(value));

export class ToolForm {
  /** host: element to render into; onChange(values) after every edit. */
  constructor(host, onChange) {
    this.host = host;
    this.onChange = onChange;
    this.spec = null;
    this.fields = new Map();  // dest -> {opt, get, set, node}
  }

  render(spec, preset = null) {
    this.spec = spec;
    this.fields.clear();
    const saved = { ...store.get(`values.${spec.key}`, {}), ...(preset ?? {}) };
    const open = store.get(`open.${spec.key}`, null);
    const groups = spec.groups.map((group, index) => {
      const details = el('details', { open: open ? open.includes(group.title) : index < 2 || group.title === 'inputs' },
        el('summary', {}, group.title));
      details.addEventListener('toggle', () => this._saveOpen());
      for (const opt of group.options) details.append(this._field(opt, saved));
      return details;
    });
    if (spec.passthrough) {
      const opt = { dest: spec.passthrough_key, flag: '-- ...', kind: 'str', default: '', help: spec.passthrough };
      groups.push(el('details', { open: true }, el('summary', {}, 'extra'), this._field(opt, saved)));
    }
    this.host.replaceChildren(...groups);
    this._changed();
  }

  _saveOpen() {
    const open = [...this.host.querySelectorAll('details')].filter((d) => d.open).map((d) => d.querySelector('summary').textContent);
    store.set(`open.${this.spec.key}`, open);
  }

  _field(opt, saved) {
    const id = `f-${this.spec.key}-${opt.dest}`;
    const initial = opt.dest in saved ? saved[opt.dest] : opt.default;
    const label = el('label', { htmlFor: id }, opt.positional ? opt.dest : opt.flag, opt.required ? el('span', { className: 'req' }, 'required') : null);
    const hint = opt.help ? el('span', { className: 'hint' }, opt.help) : null;
    let input;
    let get;
    let set;
    if (opt.kind === 'bool') {
      input = el('input', { type: 'checkbox', id, checked: Boolean(initial) });
      get = () => input.checked;
      set = (v) => { input.checked = Boolean(v); };
      const node = el('div', { className: 'field is-bool' }, input, el('div', {}, label, hint));
      this.fields.set(opt.dest, { opt, get, set, node });
      input.addEventListener('change', () => this._changed());
      return node;
    }
    if (opt.kind === 'choice') {
      input = el('select', { id }, opt.choices.map((c) => el('option', { value: c }, c)));
      input.value = asText(initial);
    } else if (opt.multi) {
      input = el('textarea', { id, rows: 2, spellcheck: false, placeholder: 'one per line' });
      input.value = asText(initial);
    } else {
      input = el('input', { type: 'text', id, spellcheck: false, autocomplete: 'off',
        inputMode: opt.kind === 'int' || opt.kind === 'float' ? 'decimal' : 'text' });
      input.value = asText(initial);
    }
    get = () => input.value;
    set = (v) => { input.value = asText(v); };
    input.addEventListener('input', () => this._changed());
    input.addEventListener('change', () => this._changed());
    const control = this.spec.pickers?.[opt.dest] ? this._withPicker(opt, input) : input;
    const node = el('div', { className: 'field' }, label, control, hint);
    this.fields.set(opt.dest, { opt, get, set, node });
    return node;
  }

  _withPicker(opt, input) {
    const kind = this.spec.pickers[opt.dest];
    const select = el('select', { className: 'picker', title: 'pick from this machine' }, el('option', { value: '' }, 'browse…'));
    const fill = async (refresh = false) => {
      select.replaceChildren(el('option', { value: '' }, 'loading…'));
      try {
        const { items, error } = await pickerItems(kind, refresh);
        select.replaceChildren(el('option', { value: '' }, error ? `⚠ ${error}` : items.length ? 'browse…' : 'nothing found'),
          ...items.map((item) => el('option', { value: item.value, title: item.value }, item.detail ? `${item.label} — ${item.detail}` : item.label)),
          el('option', { value: '__refresh' }, '↻ refresh'));
      } catch (err) {
        select.replaceChildren(el('option', { value: '' }, `⚠ ${err.message}`));
      }
      select.dataset.loaded = '1';
    };
    select.addEventListener('focus', () => { if (!select.dataset.loaded) fill(); });
    select.addEventListener('pointerdown', () => { if (!select.dataset.loaded) fill(); });
    select.addEventListener('change', () => {
      const value = select.value;
      select.value = '';
      if (value === '__refresh') { fill(true); return; }
      if (!value) return;
      input.value = opt.multi && input.value.trim() ? `${input.value.trim()}\n${value}` : value;
      this._changed();
    });
    const parts = [input, select];
    const up = UPLOADS[kind];
    if (up) {
      const file = el('input', { type: 'file', accept: up.accept, hidden: true });
      const button = el('button', { type: 'button', className: 'btn btn-ghost btn-tiny', title: 'upload from this browser' }, 'upload');
      button.addEventListener('click', () => file.click());
      file.addEventListener('change', async () => {
        const chosen = file.files[0];
        file.value = '';
        if (!chosen) return;
        try {
          button.disabled = true;
          const done = await upload(chosen, up.kind, (f) => { button.textContent = `${Math.round(f * 100)}%`; });
          input.value = opt.multi && input.value.trim() ? `${input.value.trim()}\n${done.path}` : done.path;
          pickerCache.delete(kind);
          this._changed();
        } catch (err) {
          toast(err.message);
        } finally {
          button.disabled = false;
          button.textContent = 'upload';
        }
      });
      parts.push(button, file);
    }
    return el('div', { className: `with-picker${up ? ' is-triple' : ''}` }, parts);
  }

  values() {
    const out = {};
    for (const [dest, field] of this.fields) out[dest] = field.get();
    return out;
  }

  set(values) {
    for (const [dest, value] of Object.entries(values)) this.fields.get(dest)?.set(value);
    this._changed();
  }

  reset() {
    for (const field of this.fields.values()) field.set(field.opt.default);
    this._changed();
  }

  _changed() {
    const values = this.values();
    for (const field of this.fields.values()) {
      const def = field.opt.kind === 'bool' ? Boolean(field.opt.default) : asText(field.opt.default);
      field.node.classList.toggle('changed', String(field.get()) !== String(def));
    }
    if (this.spec) store.set(`values.${this.spec.key}`, values);
    this.onChange?.(values, this.spec?.key);
  }
}
