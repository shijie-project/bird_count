// Model ratings: how crowded the vision model judged each of a day's pictures,
// next to the count and the threshold, every camera at once. One lane per camera
// (across = count, up = level 1-5); a lane opens that camera's page, where its
// pictures show with the model's level and reason.

import { datePicker } from './datepick.js';
import { api, css, el, setupCanvas, short } from './util.js';

const ROW_H = 46;
const TOP = 30;
const LEFT = 118;
const STEP = 7;  // pixels between two levels in a lane

// rows: [{mac, threshold, items: [{count, level}]}]
function drawRatings(canvas, rows, lo, hi) {
  const h = TOP + rows.length * ROW_H + 4;
  const { g, w } = setupCanvas(canvas, h);
  const X = (c) => LEFT + ((c - lo) / (hi - lo)) * (w - LEFT - 12);
  const Y = (top, lv) => top + 7 + (5 - lv) * STEP;
  const ink = css('--ink'), ink2 = css('--ink-2'), line = css('--line'), thr = css('--l2');
  g.fillStyle = ink2;
  g.fillText('Camera', 0, 18); g.fillText('Thr.', 70, 18);
  g.strokeStyle = line;
  const tick = [20, 40, 50, 100].find((s) => X(lo + s) - X(lo) >= 34) || 100;  // labels never touch on a phone
  for (let c = Math.ceil(lo / tick) * tick; c <= hi; c += tick) {
    g.fillText(String(c), X(c) - 8, 18);
    g.beginPath(); g.moveTo(X(c) + 0.5, TOP - 6); g.lineTo(X(c) + 0.5, h); g.stroke();
  }
  rows.forEach((r, i) => {
    const top = TOP + i * ROW_H;
    g.fillStyle = ink; g.fillText(short(r.mac), 0, top + 24);
    g.fillStyle = ink2; g.fillText(String(r.threshold), 70, top + 24);
    g.fillStyle = line;  // the lane's floor (level 1) and a faint line at 4, where a full shed normally sits
    g.fillRect(LEFT, Y(top, 1), w - LEFT - 12, 1);
    g.globalAlpha = 0.5; g.fillRect(LEFT, Y(top, 4), w - LEFT - 12, 1); g.globalAlpha = 1;
    g.fillStyle = ink2;
    g.fillText('5', LEFT - 12, Y(top, 5) + 4); g.fillText('1', LEFT - 12, Y(top, 1) + 4);
    g.strokeStyle = thr; g.lineWidth = 2;
    g.beginPath(); g.moveTo(X(r.threshold), top + 3); g.lineTo(X(r.threshold), top + ROW_H - 5); g.stroke();
    g.lineWidth = 1;
    g.fillStyle = ink; g.globalAlpha = 0.55;
    for (const it of r.items) { g.beginPath(); g.arc(X(it.count), Y(top, it.level) + 0.5, 3.2, 0, 7); g.fill(); }
    g.globalAlpha = 1;
    if (i) { g.fillStyle = line; g.fillRect(0, top, w, 1); }
  });
  return (event) => {  // which lane a click landed on
    const rect = canvas.getBoundingClientRect();
    return rows[Math.floor((event.clientY - rect.top - TOP) / ROW_H)] || null;
  };
}

export async function renderRatings(view, ctx) {
  const allDays = ctx.q.span === 'all';
  const latest = ctx.days[0];
  const range = ctx.days.length ? el('div', { className: 'day-pick' },
    el('button', { className: 'chip', textContent: 'All days', 'aria-pressed': String(allDays),
      onclick: () => ctx.go('ratings', null, allDays ? { day: ctx.q.day } : { span: 'all', day: ctx.q.day }) }),
    el('span', { style: allDays ? 'opacity:.55' : '' }, 'Day '),  // picking a day leaves All days
    el('span', { style: allDays ? 'opacity:.55' : '' }, datePicker(ctx.days, ctx.day || latest, (day) => ctx.go('ratings', null, { day: day === latest ? '' : day })))) : null;
  const head = el('div', { className: 'page-head' }, el('h1', { textContent: 'Model ratings' }),
    el('span', { className: 'muted', textContent: 'How crowded the vision model judged each picture. Nothing here changes the alarm.' }),
    range);
  if (!ctx.day) {
    view.replaceChildren(head, el('div', { className: 'empty' }, el('h2', { textContent: 'No count log yet' })));
    return;
  }
  const data = await api(`/api/ratings?day=${allDays ? 'all' : ctx.day}`);
  const levels = data.levels;
  const rows = ctx.farm.cameras.map((c) => ({ mac: c.mac, ...data.cameras[c.mac] })).filter((r) => r.pictures);
  const all = rows.flatMap((r) => r.items);
  const pictures = rows.reduce((n, r) => n + r.pictures, 0);
  if (!all.length) {
    view.replaceChildren(head, el('div', { className: 'empty' },
      el('h2', { textContent: allDays ? 'No ratings yet' : 'No ratings for this day' }),
      el('p', {}, `${pictures} pictures are waiting. Run `, el('code', { textContent: 'python -m birdcount.deploy.calibrate rate' }),
        ' (needs ANTHROPIC_API_KEY in .env), or pick another day.')));
    return;
  }

  const byLevel = [1, 2, 3, 4, 5].map((lv) => all.filter((it) => it.level === lv).length);
  const stats = el('div', { className: 'stats', style: 'margin-bottom:4px' },
    el('span', {}, el('b', { textContent: all.length }), `of ${pictures} pictures rated`),
    allDays ? el('span', {}, 'over ', el('b', { textContent: data.days.length }), `day${data.days.length === 1 ? '' : 's'}, against today's thresholds`) : null,
    el('span', {}, 'Model ', el('b', { textContent: data.model || '—' }), `, rubric ${data.rubric}`),
    byLevel.map((n, i) => el('span', { title: levels[i + 1].text }, el('b', { textContent: n }), `${i + 1} ${levels[i + 1].name}`)));

  const counts = [...all.map((it) => it.count), ...rows.map((r) => r.threshold)];
  const lo = Math.max(0, Math.floor(Math.min(...counts) / 20) * 20 - 10);
  const hi = Math.ceil(Math.max(...counts) / 20) * 20 + 10;
  const canvas = el('canvas', { className: 'clickable', role: 'img', 'aria-label': 'Each camera\'s rated pictures by count and crowding level, with its threshold' });
  const lanes = el('div', { className: 'panel' },
    el('div', { className: 'panel-head' }, el('h2', { textContent: 'Count against crowding' }),
      el('span', { className: 'muted', textContent: 'one dot per rated picture: across = the count, up = the level (1 sparse to 5 crammed); the coloured line is the threshold. Click a camera to see its pictures.' })),
    canvas);

  view.replaceChildren(head, stats, lanes);
  const hit = drawRatings(canvas, rows, lo, hi);
  canvas.onclick = (e) => {
    const r = hit(e);
    if (r) ctx.go('camera', r.mac, { day: ctx.q.day });
  };
}
