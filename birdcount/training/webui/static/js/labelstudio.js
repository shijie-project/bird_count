// The Label Studio server and its ngrok tunnel: status, start / stop, logs.

import { api, el, store, toast } from './util.js';

const POLL_MS = 5000;

class LogTail {
  constructor(name) {
    this.name = name;
    this.offset = -1;
    this.node = el('pre', { className: 'service-log' });
  }

  async poll() {
    try {
      const data = await api(`/api/services/${this.name}/log?offset=${this.offset}`);
      if (data.missing) return;
      if (this.offset < 0 || data.offset < this.offset) this.node.textContent = '';
      this.offset = data.offset;
      if (data.text) {
        const atEnd = this.node.scrollHeight - this.node.scrollTop - this.node.clientHeight < 30;
        this.node.textContent = (this.node.textContent + data.text).slice(-200_000);
        if (atEnd) this.node.scrollTop = this.node.scrollHeight;
      }
    } catch { /* the panel is restarting */ }
  }
}

export class LabelStudioPanel {
  constructor(host) {
    this.public = store.get('ls.public', false);
    this.timer = null;
    this.logs = { label_studio: new LogTail('label_studio'), ngrok: new LogTail('ngrok') };
    this.pill = el('span', { className: 'pill', dataset: { status: 'idle' } }, 'checking');
    this.detail = el('span', { className: 'muted' });
    this.url = el('code', {}, '—');
    this.note = el('p', { className: 'ls-note' });
    const publicBox = el('input', { type: 'checkbox', checked: this.public });
    publicBox.addEventListener('change', () => { this.public = publicBox.checked; store.set('ls.public', this.public); this.refresh(); });
    this.startBtn = el('button', { className: 'btn btn-primary', onclick: () => this.act('label_studio', 'start') }, 'Start Label Studio');
    this.stopBtn = el('button', { className: 'btn btn-danger', onclick: () => this.act('label_studio', 'stop') }, 'Stop');
    this.openBtn = el('a', { className: 'btn', target: '_blank', rel: 'noopener' }, 'Open');
    this.ngPill = el('span', { className: 'pill', dataset: { status: 'idle' } }, 'stopped');
    this.ngStart = el('button', { className: 'btn btn-primary', onclick: () => this.act('ngrok', 'start') }, 'Start tunnel');
    this.ngStop = el('button', { className: 'btn btn-danger', onclick: () => this.act('ngrok', 'stop') }, 'Stop');
    this.inspector = el('a', { className: 'btn btn-ghost', target: '_blank', rel: 'noopener' }, 'inspector');
    host.append(
      el('div', { className: 'panel' },
        el('div', { className: 'panel-head' }, el('h2', {}, 'Label Studio'), this.pill, this.detail),
        el('div', { className: 'ls-body' },
          el('label', { className: 'check ls-toggle' }, publicBox, el('span', {}, 'Public link ', el('em', {}, '(the ngrok domain instead of localhost)'))),
          el('div', { className: 'ls-url' }, this.url),
          el('div', { className: 'actions' }, this.startBtn, this.stopBtn, this.openBtn,
            el('button', { className: 'btn btn-ghost', onclick: () => this.copy() }, 'copy link'),
            el('button', { className: 'btn btn-ghost', onclick: () => this.refresh() }, 're-check')),
          this.note),
        el('details', {}, el('summary', {}, 'server log'), this.logs.label_studio.node)),
      el('div', { className: 'panel' },
        el('div', { className: 'panel-head' }, el('h2', {}, 'ngrok tunnel'), this.ngPill),
        el('div', { className: 'ls-body' },
          el('p', { className: 'ls-note' }, 'Forwards the public domain to the local Label Studio port; the public link needs it.'),
          el('div', { className: 'actions' }, this.ngStart, this.ngStop, this.inspector)),
        el('details', {}, el('summary', {}, 'tunnel log'), this.logs.ngrok.node)));
  }

  activate() { this.refresh(); }

  deactivate() { clearTimeout(this.timer); this.timer = null; }

  async refresh() {
    clearTimeout(this.timer);
    try {
      const s = await api(`/api/label-studio?public=${this.public ? 1 : 0}`);
      this.state = s;
      const ls = s.services.label_studio;
      const ng = s.services.ngrok;
      const status = s.reachable ? 'done' : ls.owned ? 'running' : 'idle';
      this.pill.dataset.status = status;
      this.pill.textContent = s.reachable ? 'up' : ls.owned ? 'starting' : 'down';
      this.detail.textContent = s.detail;
      this.url.textContent = s.url || 'no public domain: set LABEL_STUDIO_PUBLIC_URL in .env';
      this.openBtn.href = s.url || '#';
      this.openBtn.classList.toggle('is-disabled', !s.reachable);
      this.startBtn.disabled = ls.owned || s.reachable;
      this.stopBtn.disabled = !ls.owned && !s.reachable;
      this.note.textContent = `Database and uploads in ${s.data_dir}; image files served from ${s.files_root}.`
        + (this.public && !s.tunnel ? ' The public link also needs the tunnel below.' : '');
      this.ngPill.dataset.status = s.tunnel ? 'done' : ng.owned ? 'running' : 'idle';
      this.ngPill.textContent = s.tunnel ? 'up' : ng.owned ? 'starting' : 'stopped';
      this.ngStart.disabled = ng.owned || s.tunnel || !s.public_url;
      this.ngStop.disabled = !ng.owned && !s.tunnel;
      this.inspector.href = s.inspector;
      this.inspector.classList.toggle('is-disabled', !s.tunnel);
      await Promise.all(Object.values(this.logs).map((l) => l.poll()));
    } catch (err) {
      this.pill.dataset.status = 'failed';
      this.pill.textContent = 'panel unreachable';
    }
    this.timer = setTimeout(() => this.refresh(), POLL_MS);
  }

  async act(name, action) {
    try {
      const result = await api(`/api/services/${name}/${action}`, { method: 'POST', body: { public: this.public } });
      if (action === 'stop') {
        if (result.refused?.length) toast(`Left alone: port held by ${result.refused.map((p) => `pid ${p.pid}`).join(', ')}, not ${name}`);
        else if (!result.killed?.length) toast(`${name} was not running`, 'info');
      }
      this.logs[name].offset = -1;
    } catch (err) {
      toast(err.message);
    }
    setTimeout(() => this.refresh(), 600);
  }

  async copy() {
    try { await navigator.clipboard.writeText(this.state?.url ?? ''); toast('link copied', 'info'); } catch { toast('could not copy'); }
  }
}
