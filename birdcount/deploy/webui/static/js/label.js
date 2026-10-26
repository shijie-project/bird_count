// Labelling: one picture at a time, one key per verdict, straight on to the next.
// Verdicts go to the day's labels.csv as they are made.

import { daySelect } from './farm.js';
import { alarmUrl, api, clock, el, prefs, shadedUrl, short, snapshotUrl, toast } from './util.js';

const VERDICTS = [
  ['pileup', 'Pile-up', '1'],
  ['normal', 'Normal', '2'],
  ['unsure', 'Not sure', '3'],
];
const KIND_CHIPS = [['high', 'Near threshold'], ['bin', 'Other counts'], ['alarm', 'Alarm pictures']];
const STATE_CHIPS = [['todo', 'To label'], ['done', 'Labelled'], ['all', 'All']];
const LABEL_TEXT = { pileup: 'Pile-up', normal: 'Normal', unsure: 'Not sure' };

let keyHandler = null;

function why(item) {
  if (item.source === 'alarm') return `the picture sent with ${item.kind === 'recovery' ? 'the recovery' : `the Level ${item.kind.slice(-1)} alarm`}`;
  if (item.kind === 'high') return 'count at 80 % of the threshold or more';
  const lo = Math.floor(item.count / 10) * 10;
  return `a sample of the ${lo}–${lo + 9} count range`;
}

export async function renderLabel(view, ctx) {
  if (keyHandler) document.removeEventListener('keydown', keyHandler);
  if (!ctx.day) {
    view.replaceChildren(el('div', { className: 'empty' }, el('h2', { textContent: 'Nothing to label yet' }), el('p', { textContent: 'Snapshots appear here once the count logger has run.' })));
    return;
  }
  const q = { kind: 'high,bin', state: 'todo', ...ctx.q };
  const kinds = new Set(q.kind.split(','));
  const params = new URLSearchParams({ day: ctx.day, kind: q.kind, state: q.state, ...(q.camera ? { camera: q.camera } : {}) });
  const data = await api(`/api/snapshots?${params}`);
  const items = data.items;
  let i = Math.max(0, q.at ? items.findIndex((it) => it.path === q.at) : 0);
  let labelled = data.labelled;

  const setQ = (patch) => ctx.go('label', null, { ...q, ...patch, at: '', day: ctx.q.day });
  const chip = (label, pressed, onclick) => el('button', { className: 'chip', textContent: label, 'aria-pressed': String(pressed), onclick });
  const camSel = el('select', { 'aria-label': 'Camera' },
    el('option', { value: '', textContent: 'All cameras' }),
    ctx.farm.cameras.map((c) => el('option', { value: c.mac, textContent: `${short(c.mac)}  ${c.mac}`, selected: c.mac === q.camera })));
  camSel.onchange = () => setQ({ camera: camSel.value });
  const filters = el('div', { className: 'filters' },
    STATE_CHIPS.map(([k, label]) => chip(label, q.state === k, () => setQ({ state: k }))),
    el('span', { className: 'sep' }),
    KIND_CHIPS.map(([k, label]) => chip(label, kinds.has(k), () => {
      const next = new Set(kinds);
      if (next.has(k)) next.delete(k); else next.add(k);
      if (next.size) setQ({ kind: [...next].join(',') });
    })),
    el('span', { className: 'sep' }), camSel);

  const head = el('div', { className: 'page-head' }, el('h1', { textContent: 'Label snapshots' }),
    el('span', { className: 'muted', textContent: 'Are birds on top of other birds inside the bright area? Your verdicts check what the alarm and the calibration see.' }),
    daySelect(ctx, 'label'));

  if (!items.length) {
    view.replaceChildren(head, filters, el('div', { className: 'empty' }, el('h2', { textContent: q.state === 'todo' ? 'Everything here is labelled' : 'No snapshots match' }), el('p', { textContent: 'Change the filters above, or pick another day.' })));
    return;
  }

  const img = el('img', { className: 'label-img', alt: 'Snapshot to label' });
  const masks = new Set(ctx.farm.masks || []);
  let shading = prefs.get('shade', true);
  const shadeBox = el('input', { type: 'checkbox', checked: shading });
  shadeBox.onchange = () => { shading = shadeBox.checked; prefs.set('shade', shading); show(i); };
  const shadeToggle = el('label', { className: 'shade-toggle' }, shadeBox, ' Darken outside the mask ', el('kbd', { textContent: 'M' }));
  let showing = 0;  // a newer picture wins over a slower older one
  const verdict = el('div', { className: 'verdict', hidden: true });
  const facts = el('dl', { className: 'facts-list' });
  const buttons = VERDICTS.map(([key, text, kbd]) => el('button', { className: key, onclick: () => decide(key) }, text, el('kbd', { textContent: kbd })));
  const progress = el('div', { className: 'progress' });
  const queue = el('div', { className: 'queue' });
  const thumbs = items.map((it, k) => el('button', { title: `${short(it.mac)} ${clock(it.t)}`, onclick: () => show(k) },
    el('img', { src: it.source === 'alarm' ? alarmUrl(it.path) : snapshotUrl(data.day, it.path), alt: '', loading: 'lazy' })));
  queue.append(...thumbs);

  function show(k) {
    i = Math.min(Math.max(k, 0), items.length - 1);
    const it = items[i];
    const src = it.source === 'alarm' ? alarmUrl(it.path) : snapshotUrl(data.day, it.path);
    const ticket = ++showing;
    if (shading && masks.has(it.mac)) {
      shadedUrl(src, it.mac, ctx.farm.mask_shade).then((url) => { if (ticket === showing) img.src = url; }, () => { if (ticket === showing) img.src = src; });
    } else {
      img.src = src;
    }
    verdict.hidden = !it.label;
    verdict.textContent = it.label ? LABEL_TEXT[it.label] : '';
    facts.replaceChildren(...[
      ['Camera', `${short(it.mac)}  (${it.mac})`],
      ['Time', `${clock(it.t, true)}, ${data.day}`],
      ['Count', `${it.count} birds`],
      ['Threshold', String(it.threshold)],
      ['Alarm then', it.level ? `Level ${it.level}` : 'none'],
      ['Kept as', why(it)],
      ['Model says', it.rating ? `Level ${it.rating.level}: ${it.rating.reason}` : 'not rated yet'],
    ].flatMap(([k2, v]) => [el('dt', { textContent: k2 }), el('dd', { textContent: v })]));
    buttons.forEach((b, j) => b.setAttribute('aria-pressed', String(VERDICTS[j][0] === it.label)));
    thumbs.forEach((t, j) => {
      if (j === i) t.setAttribute('aria-current', 'true'); else t.removeAttribute('aria-current');
      if (items[j].label) t.dataset.label = items[j].label; else delete t.dataset.label;
    });
    thumbs[i].scrollIntoView({ block: 'nearest', inline: 'center' });
    progress.replaceChildren(`${labelled} of ${data.total} labelled  (picture ${i + 1} of ${items.length})`,
      el('div', { className: 'bar' }, el('i', { style: `width:${data.total ? (labelled / data.total) * 100 : 0}%` })));
  }

  async function decide(label) {
    const it = items[i];
    const previous = it.label;
    const next = previous === label ? '' : label;  // pressing the same verdict again clears it
    try {
      await api('/api/labels', { day: data.day, path: it.path, source: it.source || 'count_log', label: next });
    } catch (err) {
      toast(`Not saved: ${err.message}`);
      return;
    }
    it.label = next || null;
    labelled += (it.label ? 1 : 0) - (previous ? 1 : 0);
    if (it.label && i < items.length - 1) show(i + 1); else show(i);
  }

  keyHandler = (e) => {
    if (!document.body.contains(img) || e.target.closest('select, input') || e.ctrlKey || e.metaKey || e.altKey) return;
    const v = VERDICTS.find(([, , k]) => k === e.key);
    if (v) { e.preventDefault(); decide(v[0]); return; }
    if (e.key === ' ' || e.key === 'ArrowRight') { e.preventDefault(); show(i + 1); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); show(i - 1); }
    else if (e.key === 'm' || e.key === 'M') { shadeBox.checked = !shadeBox.checked; shadeBox.onchange(); }
  };
  document.addEventListener('keydown', keyHandler);

  view.replaceChildren(head, filters, el('div', { className: 'panel' }, el('div', { className: 'label-grid' },
    el('div', {}, el('div', { className: 'stage' }, img, verdict), shadeToggle, queue),
    el('div', {}, facts, el('div', { className: 'choice' }, buttons),
      el('div', { className: 'nav-row' }, el('button', { onclick: () => show(i - 1) }, 'Previous ', el('kbd', { textContent: '←' })), el('button', { onclick: () => show(i + 1) }, 'Skip ', el('kbd', { textContent: 'Space' }))),
      progress))));
  show(i);
}
