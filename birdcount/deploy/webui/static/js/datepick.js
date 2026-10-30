// The day picker: a button naming the day shown, opening a month calendar
// where only days with a count log can be picked, between steps to the day
// with a count log before and after it. Offline: no library.

import { el } from './util.js';

const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

let openPicker = null;  // the popover on screen, so a page's timed re-render can wait for it

export const pickerOpen = () => openPicker !== null && document.body.contains(openPicker);

const iso = (y, m, d) => `${y}-${String(m + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;

// days: the days with a count log, newest first; onPick(day) gets the chosen one.
export function datePicker(days, current, onPick) {
  const have = new Set(days);
  const latest = days[0];
  const button = el('button', { className: 'date-button', 'aria-haspopup': 'dialog', 'aria-expanded': 'false' },
    current, current === latest ? el('span', { className: 'muted', textContent: ' latest' }) : null, el('span', { className: 'caret', textContent: '▾' }));
  const at = days.indexOf(current);  // newest first: the day before is the next one in the list
  const before = at >= 0 ? days[at + 1] : undefined;
  const after = at > 0 ? days[at - 1] : undefined;
  const stepTo = (day, text, label) => el('button', {
    className: 'chip day-step', textContent: text, disabled: !day,
    title: day ? `${label}: ${day}` : `No ${label.toLowerCase()} with a count log`, onclick: () => day && onPick(day),
  });
  const wrap = el('div', { className: 'date-pick' },
    stepTo(before, `‹ ${before ? before.slice(5) : ''}`.trim(), 'Day before'),
    button,
    stepTo(after, `${after ? after.slice(5) : ''} ›`.trim(), 'Day after'));
  let [y, m] = current.split('-').map(Number);
  m -= 1;
  let pop = null;

  const close = () => {
    pop?.remove();
    pop = openPicker = null;
    button.setAttribute('aria-expanded', 'false');
    document.removeEventListener('pointerdown', outside, true);
    document.removeEventListener('keydown', onKey, true);
  };
  const outside = (e) => { if (!wrap.contains(e.target)) close(); };
  const onKey = (e) => { if (e.key === 'Escape') { e.stopPropagation(); close(); button.focus(); } };
  const pick = (day) => { close(); if (day !== current) onPick(day); };

  function month(step) {
    m += step;
    if (m < 0) { m = 11; y -= 1; } else if (m > 11) { m = 0; y += 1; }
    draw();
  }

  function draw() {
    const first = (new Date(y, m, 1).getDay() + 6) % 7;  // Monday first
    const n = new Date(y, m + 1, 0).getDate();
    const cells = [];
    for (let i = 0; i < first; i++) cells.push(el('span'));
    for (let d = 1; d <= n; d++) {
      const day = iso(y, m, d);
      cells.push(el('button', {
        className: 'day', textContent: String(d), disabled: !have.has(day),
        title: have.has(day) ? day : `${day}: no count log`,
        'aria-pressed': String(day === current),
        dataset: { latest: day === latest ? '1' : '' },
        onclick: () => pick(day),
      }));
    }
    const monthKey = (yy, mm) => `${yy}-${String(mm + 1).padStart(2, '0')}`;
    const months = new Set(days.map((d) => d.slice(0, 7)));
    const earlier = [...months].some((k) => k < monthKey(y, m));
    const later = [...months].some((k) => k > monthKey(y, m));
    pop.replaceChildren(
      el('div', { className: 'cal-head' },
        el('button', { className: 'cal-step', textContent: '‹', 'aria-label': 'Previous month', disabled: !earlier, onclick: () => month(-1) }),
        el('b', { textContent: `${MONTHS[m]} ${y}` }),
        el('button', { className: 'cal-step', textContent: '›', 'aria-label': 'Next month', disabled: !later, onclick: () => month(1) })),
      el('div', { className: 'cal-grid' }, WEEKDAYS.map((w) => el('span', { className: 'wd', textContent: w })), cells),
      el('div', { className: 'cal-foot' },
        el('span', { className: 'muted', textContent: `${days.length} day${days.length === 1 ? '' : 's'} recorded` }),
        el('button', { className: 'chip', textContent: 'Latest day', disabled: current === latest, onclick: () => pick(latest) })));
  }

  button.onclick = () => {
    if (pop) { close(); return; }
    [y, m] = current.split('-').map(Number);
    m -= 1;
    pop = openPicker = el('div', { className: 'calendar', role: 'dialog', 'aria-label': 'Pick a day' });
    wrap.append(pop);
    button.setAttribute('aria-expanded', 'true');
    draw();
    document.addEventListener('pointerdown', outside, true);
    document.addEventListener('keydown', onKey, true);
    pop.querySelector('.day[aria-pressed="true"]')?.focus();
  };
  return wrap;
}
