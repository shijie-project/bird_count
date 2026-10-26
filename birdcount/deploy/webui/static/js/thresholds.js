// Thresholds: what the pictures' crowding suggests for each camera, next to
// the threshold in force. Proposals only; configs/alarm.json is never changed here.

import { ago, api, css, el, setupCanvas, short } from './util.js';


function share(n, k) { return n ? `${k} of ${n}` : '—'; }
function pct(x) { return x == null ? '—' : `${Math.round(x * 100)} %`; }

// One camera: share of pictures rated dense per 10-bird bin (bar height; bar
// opacity grows with the bin's ratings), the monotone fit, and both thresholds.
function drawBins(canvas, p) {
  const height = 220;
  const { g, w } = setupCanvas(canvas, height);
  const padL = 40, padR = 10, padT = 10, padB = 24;
  const lo = Math.min(p.bins[0].lo, p.threshold, p.suggested ?? Infinity) - 10;
  const hi = Math.max(p.bins.at(-1).lo + 10, p.threshold, p.suggested ?? 0) + 10;
  const X = (c) => padL + ((c - lo) / (hi - lo)) * (w - padL - padR);
  const Y = (v) => padT + (1 - v) * (height - padT - padB);
  const maxN = Math.max(...p.bins.map((b) => b.n));
  g.strokeStyle = css('--line'); g.fillStyle = css('--ink-2');
  for (const v of [0, 0.5, 1]) { g.beginPath(); g.moveTo(padL, Y(v) + 0.5); g.lineTo(w - padR, Y(v) + 0.5); g.stroke(); g.fillText(`${v * 100}%`, 4, Y(v) + 4); }
  for (let c = Math.ceil(lo / 20) * 20; c <= hi; c += 20) g.fillText(String(c), X(c) - 8, height - 6);
  for (const b of p.bins) {
    g.globalAlpha = 0.25 + 0.75 * (b.n / maxN);
    g.fillStyle = css('--ink-2');
    g.fillRect(X(b.lo) + 1, Y(b.share), X(b.lo + 10) - X(b.lo) - 2, Y(0) - Y(b.share));
  }
  g.globalAlpha = 1;
  g.strokeStyle = css('--ink'); g.lineWidth = 1.8; g.beginPath();
  p.bins.forEach((b, i) => (i ? g.lineTo : g.moveTo).call(g, X(b.lo + 5), Y(b.fitted)));
  g.stroke(); g.lineWidth = 1.5;
  const vline = (c, color, text, dash, leftSide) => {  // the lower line labels to its left, so two close lines never overlap
    g.strokeStyle = color; g.setLineDash(dash); g.beginPath(); g.moveTo(X(c), padT); g.lineTo(X(c), Y(0)); g.stroke(); g.setLineDash([]);
    g.fillStyle = color; g.textAlign = leftSide ? 'right' : 'left';
    g.fillText(text, X(c) + (leftSide ? -4 : 4), padT + 12);
    g.textAlign = 'left';
  };
  const lower = p.suggested != null && p.suggested < p.threshold;
  vline(p.threshold, css('--l2'), `now ${p.threshold}`, [], !lower && p.suggested != null);
  if (p.suggested != null && p.suggested !== p.threshold) vline(p.suggested, css('--focus'), `suggested ${p.suggested}`, [5, 4], lower);
}

export async function renderThresholds(view, ctx) {
  const data = await api('/api/calibration');
  const head = el('div', { className: 'page-head' }, el('h1', { textContent: 'Thresholds' }),
    el('span', { className: 'muted', textContent: 'Suggested from how crowded each camera\'s pictures look. Nothing here changes the alarm.' }));
  const prop = data.proposal;
  if (!prop) {
    view.replaceChildren(head, el('div', { className: 'empty' },
      el('h2', { textContent: 'No suggestion yet' }),
      el('p', {}, 'Add ANTHROPIC_API_KEY to .env, then run ', el('code', { textContent: 'python -m birdcount.deploy.calibrate run' }), ' (nightly on the farm). It rates the count log\'s snapshots and suggests a threshold per camera.')));
    return;
  }
  const rows = prop.proposals.map((p) => ({ ...p, now: data.thresholds[p.mac] ?? p.threshold }));
  const changed = rows.filter((r) => r.suggested != null && Math.abs(r.suggested - r.now) >= 5).length;
  const detail = el('div');
  const select = (r) => {
    if (!r.bins.length) { detail.replaceChildren(); return; }
    const canvas = el('canvas', { role: 'img', 'aria-label': `Share of dense pictures by count for ${r.mac}` });
    detail.replaceChildren(el('div', { className: 'panel' },
      el('div', { className: 'panel-head' }, el('h2', { textContent: `${short(r.mac)}: from which count are its birds piling up?` }),
        el('span', { className: 'muted', textContent: 'share of pictures rated piled up (level 5) per 10-bird bin (darker = more ratings), the fitted curve, and the thresholds' })),
      canvas, r.note ? el('p', { className: 'muted', textContent: r.note }) : null));
    drawBins(canvas, { ...r, threshold: r.now });
    detail.scrollIntoView({ block: 'nearest' });
  };
  const body = rows.map((r) => {
    const delta = r.suggested == null ? null : r.suggested - r.now;
    const tr = el('tr', { tabIndex: 0 },
      el('td', {}, el('b', { textContent: short(r.mac) }), el('span', { className: 'muted', textContent: `  ${r.camera_id}` })),
      el('td', { textContent: r.now }),
      el('td', {}, r.suggested == null ? '—' : el('b', { textContent: r.suggested }), delta ? el('span', { className: 'muted', textContent: `  ${delta > 0 ? '+' : ''}${delta}` }) : null),
      el('td', { textContent: r.step == null ? '—' : Math.round(r.step) }),
      el('td', { textContent: share(r.ratings, r.dense) }),
      el('td', { textContent: `${pct(r.error_now)} → ${pct(r.error_suggested)}` }),
      el('td', { textContent: share(r.alarm_pictures, r.alarm_dense) }),
      el('td', { textContent: share(r.labels_compared, r.labels_agree) }),
      el('td', { className: 'l muted', textContent: r.note }));
    const pick = () => {
      for (const other of tr.parentNode.children) other.removeAttribute('aria-selected');
      tr.setAttribute('aria-selected', 'true');
      select(r);
    };
    tr.onclick = pick;
    tr.onkeydown = (e) => { if (e.key === 'Enter') pick(); };
    return tr;
  });
  const cols = ['Camera', 'Now', 'Suggested', 'Next step', 'Rated piled up', 'Misjudged now → suggested', 'Alarm pictures piled up', 'Agrees with your labels', 'Note'];
  view.replaceChildren(head,
    el('div', { className: 'stats', style: 'margin-bottom:12px' },
      el('span', {}, el('b', { textContent: changed }), 'cameras where the suggestion differs by 5 or more'),
      el('span', {}, 'Made ', el('b', { textContent: ago(Date.now() / 1000 - prop.made_at) }), ' ago'),
      el('span', {}, 'Model ', el('b', { textContent: prop.model })),
      el('span', {}, el('b', { textContent: prop.days.length }), ` day${prop.days.length === 1 ? '' : 's'} of pictures`)),
    el('div', { className: 'panel table-wrap' }, el('table', {},
      el('thead', {}, el('tr', {}, cols.map((c, i) => el('th', { textContent: c, className: i === 0 || i === cols.length - 1 ? 'l' : '' })))),
      el('tbody', {}, body))),
    detail);
}
