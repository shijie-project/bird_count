// One camera's pictures of a day as cards: the picture (shaded outside the mask
// as the model saw it), its count and time, the model's level (its reason on
// hover). Filter by level, sort by count or time; a card opens the picture large.

import { clock, el, lightbox, prefs, shadedUrl, snapshotUrl } from './util.js';

// items: [{path, t, count, kind, level (alarm level then), label, rating: {level, reason} | null}]
export function pictureGallery(ctx, { day, mac, items }) {
  const levels = ctx.farm.levels;
  const shade = prefs.get('shade', true) && (ctx.farm.masks || []).includes(mac);
  let want = 0;  // a model level, -1 = not rated yet, 0 = all
  let sort = prefs.get('gallery.sort', 'count');
  const grid = el('div', { className: 'rated-grid' });
  const shaded = new Map();  // src -> promise of the shaded picture, made once per card
  const chips = el('div', { className: 'filters' });
  const n = (lv) => items.filter((it) => (lv === -1 ? !it.rating : it.rating?.level === lv)).length;

  function card(it) {
    const src = snapshotUrl(day, it.path);
    const img = el('img', { src, alt: '', loading: 'lazy' });
    if (shade) {
      if (!shaded.has(src)) shaded.set(src, shadedUrl(src, mac, ctx.farm.mask_shade));
      shaded.get(src).then((url) => { img.src = url; }, () => {});
    }
    const lv = it.rating?.level;
    return el('button', {
      className: 'rated', dataset: { level: String(lv || 0) },
      title: it.rating?.reason || '',  // the reason on hover only
      onclick: () => lightbox(img.src, `${lv ? `${lv} ${levels[lv].name}, ` : ''}${it.count} birds, ${clock(it.t, true)}${it.rating ? `. ${it.rating.reason}` : ''}`),
    },
    img,
    el('span', { className: 'rated-head' },
      el('b', { className: 'rated-level', textContent: lv ? `${lv} ${levels[lv].name}` : 'not rated' }),
      `${it.count} birds`,
      it.level ? el('span', { className: `lv-mark l${it.level}`, textContent: `L${it.level}`, title: `alarm Level ${it.level} at the time` }) : null,
      el('span', { className: 'muted', textContent: clock(it.t) })));
  }

  function draw() {
    const chip = (text, pressed, onclick, extra = {}) => el('button', { className: 'chip', textContent: text, 'aria-pressed': String(pressed), onclick, ...extra });
    chips.replaceChildren(...[  // replaceChildren takes nodes only: flatten, drop the absent chip
      chip(`All ${items.length}`, want === 0, () => { want = 0; draw(); }),
      [1, 2, 3, 4, 5].map((lv) => chip(`${lv} ${levels[lv].name}  ${n(lv)}`, want === lv, () => { want = lv; draw(); }, { title: levels[lv].text, disabled: !n(lv) })),
      n(-1) ? chip(`Not rated  ${n(-1)}`, want === -1, () => { want = -1; draw(); }) : null,
      el('span', { className: 'sep' }),
      chip('Lowest count first', sort === 'count', () => { sort = 'count'; prefs.set('gallery.sort', sort); draw(); }),
      chip('In time order', sort === 'time', () => { sort = 'time'; prefs.set('gallery.sort', sort); draw(); }),
    ].flat().filter(Boolean));
    const shown = items
      .filter((it) => want === 0 || (want === -1 ? !it.rating : it.rating?.level === want))
      .sort((a, b) => (sort === 'count' ? a.count - b.count || a.t - b.t : a.t - b.t));
    grid.replaceChildren(...(shown.length ? shown.map(card) : [el('p', { className: 'muted', textContent: 'No pictures here.' })]));
  }

  draw();
  return el('div', {}, chips, grid);
}
