// Model ratings: how crowded the vision model judged each of a day's pictures,
// next to the count and the threshold. One lane per camera (across = count,
// up = level 1-5); a lane opens that camera's pictures with the model's reasons.

import { daySelect } from './farm.js';
import { alarmUrl, api, clock, css, el, lightbox, prefs, setupCanvas, shadedUrl, short, snapshotUrl } from './util.js';

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

function caption(it, levels) {
  const where = it.source === 'alarm' ? `alarm picture (${it.kind})` : clock(it.t);
  return `Level ${it.level} ${levels[it.level].name}, ${it.count} birds, ${where}`;
}

export async function renderRatings(view, ctx) {
  const head = el('div', { className: 'page-head' }, el('h1', { textContent: 'Model ratings' }),
    el('span', { className: 'muted', textContent: 'How crowded the vision model judged each picture. Nothing here changes the alarm.' }),
    daySelect(ctx, 'ratings'));
  if (!ctx.day) {
    view.replaceChildren(head, el('div', { className: 'empty' }, el('h2', { textContent: 'No count log yet' })));
    return;
  }
  const data = await api(`/api/ratings?day=${ctx.day}`);
  const levels = data.levels;
  const rows = ctx.farm.cameras.map((c) => ({ mac: c.mac, ...data.cameras[c.mac] })).filter((r) => r.pictures);
  const all = rows.flatMap((r) => r.items);
  const pictures = rows.reduce((n, r) => n + r.pictures, 0);
  if (!all.length) {
    view.replaceChildren(head, el('div', { className: 'empty' },
      el('h2', { textContent: 'No ratings for this day' }),
      el('p', {}, `${pictures} pictures are waiting. Run `, el('code', { textContent: 'python -m birdcount.deploy.calibrate rate' }),
        ' (needs ANTHROPIC_API_KEY in .env), or pick another day.')));
    return;
  }

  const byLevel = [1, 2, 3, 4, 5].map((lv) => all.filter((it) => it.level === lv).length);
  const stats = el('div', { className: 'stats', style: 'margin-bottom:4px' },
    el('span', {}, el('b', { textContent: all.length }), `of ${pictures} pictures rated`),
    el('span', {}, 'Model ', el('b', { textContent: data.model || '—' }), `, rubric ${data.rubric}`),
    byLevel.map((n, i) => el('span', { title: levels[i + 1].text }, el('b', { textContent: n }), `${i + 1} ${levels[i + 1].name}`)));

  const counts = [...all.map((it) => it.count), ...rows.map((r) => r.threshold)];
  const lo = Math.max(0, Math.floor(Math.min(...counts) / 20) * 20 - 10);
  const hi = Math.ceil(Math.max(...counts) / 20) * 20 + 10;
  const canvas = el('canvas', { className: 'clickable', role: 'img', 'aria-label': 'Each camera\'s rated pictures by count and crowding level, with its threshold' });
  const lanes = el('div', { className: 'panel' },
    el('div', { className: 'panel-head' }, el('h2', { textContent: 'Count against crowding' }),
      el('span', { className: 'muted', textContent: 'one dot per rated picture: across = the count, up = the level (1 sparse to 5 crammed); the coloured line is the threshold. Click a camera for its pictures.' })),
    canvas);

  const mac = rows.some((r) => r.mac === ctx.q.camera) ? ctx.q.camera : null;
  const parts = [head, stats, lanes];
  if (mac) parts.push(gallery(ctx, data, rows.find((r) => r.mac === mac)));
  view.replaceChildren(...parts);
  const hit = drawRatings(canvas, rows, lo, hi);
  canvas.onclick = (e) => {
    const r = hit(e);
    if (r) ctx.go('ratings', null, { day: ctx.q.day, camera: r.mac === mac ? '' : r.mac });
  };
  if (mac && !ctx.quiet) parts.at(-1).scrollIntoView({ block: 'nearest' });
}

function gallery(ctx, data, row) {
  const levels = data.levels;
  const want = Number(ctx.q.level) || 0;
  const items = row.items.filter((it) => !want || it.level === want).sort((a, b) => a.count - b.count || a.t - b.t);
  const setLevel = (lv) => ctx.go('ratings', null, { day: ctx.q.day, camera: row.mac, level: lv || '' });
  const chips = el('div', { className: 'filters' },
    el('button', { className: 'chip', textContent: `All ${row.items.length}`, 'aria-pressed': String(!want), onclick: () => setLevel(0) }),
    [1, 2, 3, 4, 5].map((lv) => {
      const n = row.items.filter((it) => it.level === lv).length;
      return el('button', { className: 'chip', textContent: `${lv} ${levels[lv].name}  ${n}`, title: levels[lv].text, disabled: !n, 'aria-pressed': String(want === lv), onclick: () => setLevel(lv) });
    }));
  const masks = new Set(ctx.farm.masks || []);
  const shade = prefs.get('shade', true) && masks.has(row.mac);
  const cards = items.map((it) => {
    const src = it.source === 'alarm' ? alarmUrl(it.path) : snapshotUrl(data.day, it.path);
    const img = el('img', { src, alt: '', loading: 'lazy' });
    const text = caption(it, levels);
    const card = el('button', { className: 'rated', dataset: { level: String(it.level) }, onclick: () => lightbox(img.src, `${text}. ${it.reason}`) },
      img,
      el('span', { className: 'rated-head' }, el('b', { className: 'rated-level', textContent: `${it.level} ${levels[it.level].name}` }), `${it.count} birds`,
        el('span', { className: 'muted', textContent: it.source === 'alarm' ? 'alarm' : clock(it.t) })),
      el('span', { className: 'rated-reason', textContent: it.reason }));
    if (shade) shadedUrl(src, row.mac, ctx.farm.mask_shade).then((url) => { img.src = url; }, () => {});
    return card;
  });
  return el('div', { className: 'panel' },
    el('div', { className: 'panel-head' }, el('h2', { textContent: `${short(row.mac)}: ${row.items.length} rated pictures` }),
      el('span', { className: 'muted', textContent: `threshold ${row.threshold}; lowest count first${shade ? ', outside the mask darkened as the model saw it' : ''}` }),
      el('a', { className: 'back-link', style: 'margin-left:auto', href: `#/label?camera=${row.mac}&state=all${ctx.q.day ? `&day=${ctx.q.day}` : ''}`, textContent: 'Label these pictures' })),
    chips,
    items.length ? el('div', { className: 'rated-grid' }, cards) : el('p', { className: 'muted', textContent: 'No pictures at this level.' }));
}
