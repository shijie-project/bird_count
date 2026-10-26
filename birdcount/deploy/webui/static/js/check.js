// Checking alarms: one alarm event at a time, one key per verdict, straight on
// to the next. Was it a real pile-up? Verdicts go to the day's labels.csv; per
// camera they say whether its threshold alarms too often or misses.

import { daySelect } from './farm.js';
import { ACTION_NAMES, alarmUrl, api, clock, el, prefs, shadedUrl, short, span, toast } from './util.js';

const VERDICTS = [
  ['pileup', 'Real pile-up', '1'],
  ['normal', 'False alarm', '2'],
  ['unsure', "Can't tell", '3'],
];
const STATE_CHIPS = [['todo', 'To check'], ['done', 'Checked'], ['all', 'All alarms']];
export const VERDICT_TEXT = { pileup: 'Real pile-up', normal: 'False alarm', unsure: "Can't tell" };

let keyHandler = null;

export async function renderCheck(view, ctx) {
  if (keyHandler) document.removeEventListener('keydown', keyHandler);
  const q = { state: 'todo', ...ctx.q };
  const head = el('div', { className: 'page-head' }, el('h1', { textContent: 'Check alarms' }),
    el('span', { className: 'muted', textContent: 'Was it a real pile-up? Per camera, your verdicts show whether its threshold alarms too often.' }),
    daySelect(ctx, 'check'));
  if (!ctx.day) {
    view.replaceChildren(head, el('div', { className: 'empty' }, el('h2', { textContent: 'No alarms yet' }), el('p', { textContent: 'Alarms appear here once the runtime has raised one.' })));
    return;
  }
  const params = new URLSearchParams({ day: ctx.day, state: q.state, ...(q.camera ? { camera: q.camera } : {}) });
  const data = await api(`/api/alarms?${params}`);
  const items = data.items;
  let i = Math.max(0, q.at ? items.findIndex((it) => it.event_id === q.at) : 0);
  let checked = data.checked;

  const setQ = (patch) => ctx.go('check', null, { ...q, ...patch, at: '', day: ctx.q.day });
  const chip = (label, pressed, onclick) => el('button', { className: 'chip', textContent: label, 'aria-pressed': String(pressed), onclick });
  const camSel = el('select', { 'aria-label': 'Camera' },
    el('option', { value: '', textContent: 'All cameras' }),
    ctx.farm.cameras.map((c) => el('option', { value: c.mac, textContent: `${short(c.mac)}  ${c.mac}`, selected: c.mac === q.camera })));
  camSel.onchange = () => setQ({ camera: camSel.value });
  const filters = el('div', { className: 'filters' },
    STATE_CHIPS.map(([k, label]) => chip(label, q.state === k, () => setQ({ state: k }))),
    el('span', { className: 'sep' }), camSel);

  if (!items.length) {
    const none = q.state === 'todo' && data.total ? 'Every alarm of this day is checked' : 'No alarms match';
    view.replaceChildren(head, filters, el('div', { className: 'empty' }, el('h2', { textContent: none }), el('p', { textContent: 'Change the filters above, or pick another day.' })));
    return;
  }

  const img = el('img', { className: 'label-img', alt: 'Alarm picture' });
  const masks = new Set(ctx.farm.masks || []);
  let shading = prefs.get('shade', true);
  let picture = 0;  // which of the event's pictures is shown
  const shadeBox = el('input', { type: 'checkbox', checked: shading });
  shadeBox.onchange = () => { shading = shadeBox.checked; prefs.set('shade', shading); showPicture(picture); };
  const shadeToggle = el('label', { className: 'shade-toggle' }, shadeBox, ' Darken outside the mask ', el('kbd', { textContent: 'M' }));
  let showing = 0;  // a newer picture wins over a slower older one
  const verdict = el('div', { className: 'verdict', hidden: true });
  const facts = el('dl', { className: 'facts-list' });
  const strip = el('div', { className: 'event-pictures' });
  const buttons = VERDICTS.map(([key, text, kbd]) => el('button', { className: key, onclick: () => decide(key) }, text, el('kbd', { textContent: kbd })));
  const progress = el('div', { className: 'progress' });
  const queue = el('div', { className: 'queue' });
  const thumbs = items.map((it, k) => el('button', { title: `${short(it.mac)} ${clock(it.t)}`, onclick: () => show(k) },
    el('img', { src: alarmUrl(it.pictures[0].path), alt: '', loading: 'lazy' })));
  queue.append(...thumbs);

  function showPicture(k) {
    const it = items[i];
    picture = Math.min(Math.max(k, 0), it.pictures.length - 1);
    const src = alarmUrl(it.pictures[picture].path);
    const ticket = ++showing;
    if (shading && masks.has(it.mac)) {
      shadedUrl(src, it.mac, ctx.farm.mask_shade).then((url) => { if (ticket === showing) img.src = url; }, () => { if (ticket === showing) img.src = src; });
    } else {
      img.src = src;
    }
    [...strip.children].forEach((b, j) => { if (j === picture) b.setAttribute('aria-current', 'true'); else b.removeAttribute('aria-current'); });
  }

  function show(k) {
    i = Math.min(Math.max(k, 0), items.length - 1);
    const it = items[i];
    const here = new URLSearchParams(location.hash.split('?')[1] || '');
    here.set('at', it.event_id);  // a reload, or coming back, opens this alarm again
    history.replaceState(null, '', `#/check?${here}`);
    strip.replaceChildren(...it.pictures.map((pic, j) => el('button', { onclick: () => showPicture(j) },
      el('img', { src: alarmUrl(pic.path), alt: '', loading: 'lazy' }),
      el('span', {}, el('b', { textContent: ACTION_NAMES[pic.type] || pic.type }), ` ${clock(pic.t, true)}, ${pic.count}`))));
    showPicture(0);
    verdict.hidden = !it.label;
    verdict.textContent = it.label ? VERDICT_TEXT[it.label] : '';
    const levels = ctx.farm.levels;
    facts.replaceChildren(...[
      ['Camera', `${short(it.mac)}  (${it.mac})`],
      ['Started', `${clock(it.t, true)}, ${data.day}`],
      ['Count then', `${it.count} birds (threshold ${it.threshold})`],
      ['Reached', `Level ${it.level}`],
      ['Lasted', it.end ? span(it.end - it.t) : 'still open at the end of the record'],
      ['Model says', it.rating ? `${it.rating.level} ${levels[it.rating.level].name}` : 'not rated yet'],
    ].flatMap(([k2, v]) => [el('dt', { textContent: k2 }), el('dd', { textContent: v, title: k2 === 'Model says' ? it.rating?.reason || '' : '' })]));
    buttons.forEach((b, j) => b.setAttribute('aria-pressed', String(VERDICTS[j][0] === it.label)));
    thumbs.forEach((t, j) => {
      if (j === i) t.setAttribute('aria-current', 'true'); else t.removeAttribute('aria-current');
      if (items[j].label) t.dataset.label = items[j].label; else delete t.dataset.label;
    });
    // centre its thumbnail in the queue without scrolling the page itself
    queue.scrollLeft = thumbs[i].offsetLeft - queue.offsetLeft - (queue.clientWidth - thumbs[i].offsetWidth) / 2;
    progress.replaceChildren(`${checked} of ${data.total} checked  (alarm ${i + 1} of ${items.length})`,
      el('div', { className: 'bar' }, el('i', { style: `width:${data.total ? (checked / data.total) * 100 : 0}%` })));
  }

  async function decide(label) {
    const it = items[i];
    const previous = it.label;
    const next = previous === label ? '' : label;  // pressing the same verdict again clears it
    try {
      await api('/api/labels', { day: data.day, path: it.pictures[0].path, source: 'alarm', label: next });
    } catch (err) {
      toast(`Not saved: ${err.message}`);
      return;
    }
    it.label = next || null;
    checked += (it.label ? 1 : 0) - (previous ? 1 : 0);
    if (it.label && i < items.length - 1) show(i + 1); else show(i);
  }

  keyHandler = (e) => {
    if (!document.body.contains(img) || e.target.closest('select, input, .calendar') || e.ctrlKey || e.metaKey || e.altKey) return;
    const v = VERDICTS.find(([, , k]) => k === e.key);
    if (v) { e.preventDefault(); decide(v[0]); return; }
    if (e.key === ' ' || e.key === 'ArrowRight') { e.preventDefault(); show(i + 1); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); show(i - 1); }
    else if (e.key === 'm' || e.key === 'M') { shadeBox.checked = !shadeBox.checked; shadeBox.onchange(); }
  };
  document.addEventListener('keydown', keyHandler);

  view.replaceChildren(head, filters, el('div', { className: 'panel' }, el('div', { className: 'label-grid' },
    el('div', {}, el('div', { className: 'stage' }, img, verdict), shadeToggle, strip, queue),
    el('div', {}, facts, el('div', { className: 'choice' }, buttons),
      el('div', { className: 'nav-row' }, el('button', { onclick: () => show(i - 1) }, 'Previous ', el('kbd', { textContent: '←' })), el('button', { onclick: () => show(i + 1) }, 'Skip ', el('kbd', { textContent: 'Space' }))),
      progress))));
  show(i);
}
