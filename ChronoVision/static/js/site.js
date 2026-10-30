/* ChronoVision project page.
   Every number below is copied from the paper's tables (arXiv 2608.05631); nothing is a screenshot. */
(() => {
'use strict';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const SVGNS = 'http://www.w3.org/2000/svg';
const svg = (tag, attrs = {}, parent) => {
  const el = document.createElementNS(SVGNS, tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(el);
  return el;
};
const txt = (parent, attrs, s) => { const t = svg('text', attrs, parent); t.textContent = s; return t; };
const fmt = (v, d = 1) => Number(v).toFixed(d);
const signed = (v, d = 1) => (v > 0 ? '+' : v < 0 ? '−' : '±') + fmt(Math.abs(v), d);

/* ------------------------------------------------------------------ data */
// Vbvr-VQA categories, in the paper's table order. Column indices point into MODELS rows.
const DIMS = [
  { key: 'fluid', short: 'Fluid', name: 'Fluid Intelligence', color: 'var(--d-fluid)', id: 5, ood: 11,
    desc: 'New problems solved by <strong>planning</strong> rather than recall: collect keys in a maze by the shortest route, route curves between matching shapes without crossing, sort shapes by type and size.' },
  { key: 'crystallized', short: 'Crystallized', name: 'Crystallized Intelligence', color: 'var(--d-cryst)', id: 6, ood: 12,
    desc: 'Tasks that lean on <strong>learned knowledge</strong> of how things work: read a clock four hours later, bend light through glass, let liquid settle in connected tubes, operate an interface slider.' },
  { key: 'visuospatial', short: 'Visuospatial', name: 'Visuospatial Cognition', color: 'var(--d-visuo)', id: 7, ood: 13,
    desc: '<strong>Spatial structure and search</strong>: shortest paths on graphs and grids, the highest-cost route through a cost map, finding hollow circles or a single Chinese character among distractors.' },
  { key: 'simulation', short: 'Simulation', name: 'Mental Simulation', color: 'var(--d-sim)', id: 8, ood: 14,
    desc: '<strong>Run physics forward</strong> in your head: a bouncing ball, domino chains that may break, projectile motion under a given gravity, faces sliding into a size-sorted row.' },
  { key: 'transformation', short: 'Transformation', name: 'Transformation', color: 'var(--d-trans)', id: 9, ood: 15,
    desc: 'Track how an <strong>object or layout changes</strong>: placing bricks, sliding-tile puzzles, rotating pipe pieces, a camera orbiting a block sculpture, restacking blocks one top block at a time.' },
];
const BENCH_H = [1188, 870, 726, 734, 734];

// [name, group, logo, overall, ID avg, ID flu, cry, vis, men, trans, OOD avg, OOD flu, cry, vis, men, trans]
const MODELS = [
  ['Qwen3.5-9B', 'open', 'qwen', 14.2, 16.8, 10.8, 13.3, 20.0, 40.0, 12.7, 11.6, 13.3, 20.0, 5.0, 12.0, 10.0],
  ['GLM-4.6V', 'open', 'Z', 21.4, 19.2, 10.8, 22.2, 21.7, 20.0, 23.6, 23.6, 15.6, 30.0, 25.0, 16.0, 26.0],
  ['GPT-5.4', 'prop', 'gpt', 33.8, 28.0, 23.1, 31.1, 25.0, 32.0, 32.7, 39.6, 20.0, 47.5, 30.0, 40.0, 49.0],
  ['Gemini 3.1 Pro', 'prop', 'gemini', 41.8, 52.8, 32.3, 66.7, 66.7, 60.0, 47.3, 30.8, 42.2, 62.5, 47.5, 48.0, 2.0],
  ['GPT-o3', 'prop', 'gpt', 46.0, 42.8, 26.2, 60.0, 45.0, 64.0, 36.4, 49.2, 31.1, 62.5, 45.0, 60.0, 51.0],
  ['Gemini 3.0 Flash', 'prop', 'gemini', 46.6, 47.2, 30.8, 51.1, 53.3, 80.0, 41.8, 46.0, 31.1, 37.5, 45.0, 48.0, 56.0],
  ['Qwen3.5-397B', 'open', 'qwen', 49.4, 45.6, 21.5, 60.0, 58.3, 44.0, 49.1, 53.2, 42.2, 67.5, 47.5, 44.0, 57.0],
  ['Claude Opus 4.6', 'prop', 'claude', 55.8, 50.8, 30.8, 66.7, 60.0, 56.0, 49.1, 60.8, 48.9, 67.5, 42.5, 56.0, 72.0],
  ['ChronoVision', 'ours', 'ours', 73.2, 74.8, 72.3, 77.8, 75.0, 68.0, 78.2, 71.6, 66.7, 75.0, 72.5, 64.0, 74.0],
];
const GROUP_LABEL = { prop: 'Closed-source', open: 'Open-source', ours: 'Ours' };
const VIEWS = {
  summary: { label: 'Summary', cols: [3, 4, 10], names: ['Overall', 'In-domain', 'Out-of-domain'], avg: 0 },
  id: { label: 'In-domain', cols: [5, 6, 7, 8, 9, 4], names: ['Fluid', 'Crystallized', 'Visuospatial', 'Mental Sim.', 'Transform.', 'ID Avg.'], avg: 5 },
  ood: { label: 'Out-of-domain', cols: [11, 12, 13, 14, 15, 10], names: ['Fluid', 'Crystallized', 'Visuospatial', 'Mental Sim.', 'Transform.', 'OOD Avg.'], avg: 5 },
};

// Kendall's tau (x100): [name, ID flu, cry, vis, men, trans, OOD flu, cry, vis, men, trans]
const TAU = [
  ['Qwen3.5-9B', 29.5, 31.4, 36.8, 51.2, 30.9, 31.6, 36.2, 25.1, 30.7, 29.4],
  ['Gemini 3.0 Flash', 55.6, 68.0, 69.6, 84.4, 62.3, 55.9, 59.4, 64.0, 65.8, 71.4],
  ['Gemini 3.1 Pro', 56.1, 78.4, 78.8, 73.8, 65.8, 63.0, 76.2, 66.6, 66.9, 24.8],
  ['Claude Opus 4.6', 58.8, 80.1, 76.2, 74.1, 69.9, 70.1, 81.0, 66.4, 74.2, 83.5],
  ['ChronoVision', 88.4, 90.7, 89.5, 87.8, 91.0, 86.9, 89.5, 88.8, 85.5, 89.3],
];

// [label, group, ours?, overall, ID avg, OOD avg]
const ABLATION = [
  ['SFT, no RVH', 'Training process', 0, 66.0, 66.8, 65.2],
  ['SFT + RVH', 'Training process', 0, 69.0, 70.0, 68.0],
  ['SFT + RVH + ROI', 'Training process', 0, 70.2, 71.6, 68.8],
  ['+ RL (ChronoVision)', 'Training process', 1, 73.2, 74.8, 71.6],
  ['RL without answer reward', 'Reward components', 0, 69.8, 71.2, 68.4],
  ['RL without latent-grounding reward', 'Reward components', 0, 71.0, 72.8, 69.2],
  ['RL without focus reward', 'Reward components', 0, 71.8, 74.0, 69.6],
  ['Full reward (ChronoVision)', 'Reward components', 1, 73.2, 74.8, 71.6],
];
// same backbone and SFT data: [label, ours?, overall, ID avg, OOD avg]
const RELATED = [
  ['SFT + Latent Sketchpad', 0, 69.8, 71.2, 68.4],
  ['SFT + R1-VL', 0, 70.4, 72.0, 68.8],
  ['SFT + VL-Cogito', 0, 70.8, 72.4, 69.2],
  ['ChronoVision', 1, 73.2, 74.8, 71.6],
];
const INTPHYS = [['Easy', 51.0, 62.5], ['Medium', 48.5, 54.8], ['Hard', 47.6, 53.0], ['Overall', 48.5, 55.0]];
const VIDEO = {
  holmes: { label: 'Video-Holmes', note: 'Overall score. Every model uses 32 uniformly sampled frames, except Gemini-2.5-Pro, whose frame count is not reported.',
    rows: [['Qwen2.5-VL-7B', 27.80], ['Video-R1', 36.50], ['GPT-4o', 42.00], ['Gemini-2.5-Pro', 45.00], ['ChronoVision', 45.89, 1]], d: 2 },
  longvideo: { label: 'LongVideo-Reason', note: 'Overall score. ChronoVision is also highest on each of the temporal, goal, plot and spatial categories.',
    rows: [['LongVILA-7B', 62.7], ['Video-R1-7B', 68.1], ['Gemini-1.5-Pro', 69.3], ['LongVILA-R1-7B', 72.0], ['ChronoVision', 74.7, 1]], d: 1 },
};
const GENERAL = [
  ['MMMU', 78.4, 78.8], ['MMBench', 90.1, 88.2], ['MMStar', 79.7, 78.5], ['MathVista', 85.7, 85.9],
  ['AI2D', 90.2, 90.2], ['HallusionBench', 69.3, 69.6], ['BabyVision', 25.8, 27.3],
];

/* ------------------------------------------------------------- theming */
const root = document.documentElement;
$('#themeToggle').addEventListener('click', () => {
  const dark = root.dataset.theme ? root.dataset.theme === 'dark'
    : matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
  try { localStorage.setItem('cv-theme', root.dataset.theme); } catch (e) {}
});

/* ---------------------------------------------------- top bar + nav */
const topbar = $('#topbar');
new IntersectionObserver(([e]) => topbar.classList.toggle('show', !e.isIntersecting), { rootMargin: '-56px 0px 0px 0px' })
  .observe($('.hero-links'));

const navLinks = [...$$('.topbar nav a'), ...$$('.sidenav a')];
const sections = ['top', 'takeaways', 'benchmark', 'method', 'results', 'analysis', 'generalization', 'examples', 'cite']
  .map(id => document.getElementById(id)).filter(Boolean);
const setActive = id => navLinks.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + id));
const spy = () => {
  const y = window.innerHeight * 0.4;
  let cur = 'top';
  for (const s of sections) if (s.id !== 'top' && s.getBoundingClientRect().top <= y) cur = s.id;
  setActive(cur);
};
addEventListener('scroll', spy, { passive: true }); spy();

/* --------------------------------------------------------------- reveal */
const rev = new IntersectionObserver(es => es.forEach(e => {
  if (e.isIntersecting) { e.target.classList.add('in'); rev.unobserve(e.target); }
}), { rootMargin: '0px 0px -8% 0px' });
$$('.reveal').forEach(el => rev.observe(el));

/* ----------------------------------------------------------------- math */
const renderMath = () => {
  if (!window.katex) return false;
  $$('.eq[data-tex]').forEach(el => katex.render(el.dataset.tex, el, { displayMode: false, throwOnError: false }));
  return true;
};
if (!renderMath()) addEventListener('load', renderMath);

/* -------------------------------------------------- segmented controls */
function seg(el, options, initial, onPick) {
  options.forEach(([label, value]) => {
    const b = document.createElement('button');
    b.textContent = label; b.setAttribute('aria-pressed', String(value === initial));
    b.addEventListener('click', () => { $$('button', el).forEach(x => x.setAttribute('aria-pressed', String(x === b))); onPick(value); });
    el.appendChild(b);
  });
}

/* --------------------------------------------------- category explorer */
const OURS = MODELS.find(m => m[1] === 'ours');
const bestOf = (col, groups) => MODELS.filter(m => groups.includes(m[1])).reduce((a, b) => (b[col] > a[col] ? b : a));
const tabs = $('#dimtabs'), info = $('#diminfo'), stage = $('#vstage'), thumbs = $('#vthumbs');

DIMS.forEach((d, i) => {
  const b = document.createElement('button');
  b.className = 'dimtab'; b.setAttribute('role', 'tab'); b.style.setProperty('--c', d.color);
  b.innerHTML = `<span class="sw"></span>${d.name}`;
  b.addEventListener('click', () => selectDim(i));
  tabs.appendChild(b);
});

function selectDim(i) {
  const d = DIMS[i], tau = TAU.find(t => t[0] === 'ChronoVision');
  $$('.dimtab', tabs).forEach((b, j) => b.setAttribute('aria-selected', String(i === j)));
  const bi = bestOf(d.id, ['prop', 'open']), bo = bestOf(d.ood, ['prop', 'open']);
  const tile = (v, t, ours) => `<div class="pp${ours ? ' ours' : ''}"><div class="v">${fmt(v)}</div><div class="t">${t}</div></div>`;
  info.style.setProperty('--c', d.color);
  info.innerHTML = `
    <h3><span class="sw"></span>${d.name}</h3>
    <div class="theory">Exact-match accuracy, 5 examples below</div>
    <p class="desc">${d.desc}</p>
    <div class="dim-perf two">
      <div class="row-l">In-domain</div>
      ${tile(bi[d.id], `Best baseline<br><b>${bi[0]}</b>`)}
      ${tile(OURS[d.id], 'ChronoVision<br>(ours)', true)}
      <div class="row-l">Out-of-domain</div>
      ${tile(bo[d.ood], `Best baseline<br><b>${bo[0]}</b>`)}
      ${tile(OURS[d.ood], 'ChronoVision<br>(ours)', true)}
    </div>
    <p class="desc" style="font-size:.85rem;margin-top:14px">Partial credit (Kendall's τ) for ChronoVision:
      <strong>${fmt(tau[1 + i])}</strong> in-domain, <strong>${fmt(tau[6 + i])}</strong> out-of-domain.</p>`;
  stage.innerHTML = ''; thumbs.innerHTML = '';
  for (let k = 1; k <= 5; k++) {
    const src = `static/images/web/bench_${d.key}_${k}.webp`;
    const img = document.createElement('img');
    img.src = src; img.alt = `${d.name} example ${k}`; img.width = 760; img.height = BENCH_H[k - 1];
    if (k === 1) img.className = 'on';
    stage.appendChild(img);
    const t = document.createElement('button');
    t.setAttribute('role', 'tab'); t.setAttribute('aria-selected', String(k === 1)); t.setAttribute('aria-label', `Example ${k}`);
    t.innerHTML = `<img src="${src}" alt="" loading="lazy">`;
    t.addEventListener('click', () => showEx(k - 1));
    thumbs.appendChild(t);
  }
}
function showEx(k) {
  $$('img', stage).forEach((im, j) => im.classList.toggle('on', j === k));
  $$('button', thumbs).forEach((b, j) => b.setAttribute('aria-selected', String(j === k)));
}
selectDim(0);

/* ---------------------------------------------------------- leaderboard */
const LOGO = { gpt: ['gpt.svg', true], gemini: ['gemini.svg'], claude: ['claude.svg'], qwen: ['qwen.svg'] };
const logoHTML = k => {
  if (k === 'ours') return '<img src="static/images/web/logo.webp" alt="">';
  if (LOGO[k]) return `<img src="static/images/logos/${LOGO[k][0]}" alt=""${LOGO[k][1] ? ' class="mono"' : ''}>`;
  return `<span class="ph">${k}</span>`;
};
const RANK = new Map(MODELS.slice().sort((a, b) => b[3] - a[3]).map((m, i) => [m[0], i + 1]));
let view = 'summary', sortCol = 0, sortDir = -1, group = 'all';

function drawLB() {
  const V = VIEWS[view], t = $('#lb');
  const colRank = V.cols.map(c => {
    const vals = MODELS.map(m => m[c]).sort((a, b) => b - a);
    return { best: vals[0], second: vals.find(v => v < vals[0]), baseline: Math.max(...MODELS.filter(m => m[1] !== 'ours').map(m => m[c])) };
  });
  const head = `<thead><tr><th>#</th><th>Model</th><th>Type</th>${V.names.map((n, i) =>
    `<th class="sortable${i === V.avg ? ' c-avg' : ''}${i === sortCol ? ' sorted' : ''}" data-c="${i}">${n}<span class="arrow">${sortDir < 0 ? '▼' : '▲'}</span></th>`).join('')}</tr></thead>`;
  const rows = MODELS.filter(m => group === 'all' || m[1] === group || m[1] === 'ours')
    .sort((a, b) => sortDir * (a[V.cols[sortCol]] - b[V.cols[sortCol]]) || (b[3] - a[3]));
  const row = m => {
    const isO = m[1] === 'ours', rk = RANK.get(m[0]);
    const cells = V.cols.map((c, i) => {
      const v = m[c], cr = colRank[i];
      const cls = v === cr.best ? ' best' : v === cr.second ? ' second' : '';
      const dv = v - cr.baseline;
      const d = isO ? `<span class="delta${dv < 0 ? ' neg' : ''}">${signed(dv)}</span>` : '';
      return `<td class="${i === V.avg ? 'c-avg' : ''}"><span class="cell${cls}">${fmt(v)}<span class="bar" style="width:${v}%"></span></span>${d}</td>`;
    }).join('');
    return `<tr class="${isO ? 'ours' : ''}"><td><span class="rank${rk <= 3 ? ' r' + rk : ''}">${rk}</span></td><td><span class="mname">${logoHTML(m[2])}${m[0]}${isO ? '<span class="tag">ours · 9B</span>' : ''}</span></td><td class="grp">${GROUP_LABEL[m[1]]}</td>${cells}</tr>`;
  };
  t.innerHTML = head + '<tbody>' + rows.map(row).join('') + '</tbody>';
  $$('th.sortable', t).forEach(th => th.addEventListener('click', () => {
    const c = +th.dataset.c;
    if (c === sortCol) sortDir = -sortDir; else { sortCol = c; sortDir = -1; }
    drawLB();
  }));
}
seg($('#lbview'), Object.entries(VIEWS).map(([k, v]) => [v.label, k]), view, v => { view = v; sortCol = VIEWS[v].avg; sortDir = -1; drawLB(); });
$$('#lbtools .chip').forEach(b => b.addEventListener('click', () => {
  group = b.dataset.g;
  $$('#lbtools .chip').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
  drawLB();
}));
drawLB();
$('.lb-note').innerHTML = '<b>Bold</b>: best. <u>Underline</u>: runner-up. Numbers next to ChronoVision: its margin over the best baseline in that column (grey when behind). Ranks follow overall accuracy.';

/* Kendall's tau */
(function tauTable() {
  const t = $('#tau'), cats = DIMS.map(d => d.short);
  t.innerHTML = `<thead><tr><th class="l" rowspan="2">Model</th><th colspan="5">In-domain</th><th colspan="5" class="gap">Out-of-domain</th></tr>
    <tr>${cats.map(c => `<th>${c}</th>`).join('')}${cats.map((c, i) => `<th class="${i === 0 ? 'gap' : ''}">${c}</th>`).join('')}</tr></thead>
    <tbody>${TAU.map(r => `<tr class="${r[0] === 'ChronoVision' ? 'ours' : ''}"><td class="l">${r[0]}</td>${r.slice(1).map((v, i) =>
      `<td class="${i === 5 ? 'gap' : ''}"><span style="display:inline-block;min-width:42px;padding:2px 0;border-radius:4px;background:color-mix(in srgb, var(--c1) ${Math.round(v * 0.32)}%, transparent)">${fmt(v)}</span></td>`).join('')}</tr>`).join('')}</tbody>`;
})();

/* ------------------------------------------------------------- gap chart */
(function gapChart() {
  const el = $('#gapchart');
  const C = { closed: 'var(--ink-44)', open: 'var(--c2)', ours: 'var(--c1)' };
  $('#gaplegend').innerHTML = `<span><i style="background:${C.closed}"></i>Best closed-source</span><span><i style="background:${C.open}"></i>Best open-source</span><span><i style="background:${C.ours}"></i>ChronoVision</span>`;
  let split = 'id';
  function draw() {
    el.innerHTML = '';
    const W = 640, L = 176, Rr = 64, top = 30, rowH = 58;
    const rows = [...DIMS.map(d => ({ name: d.name, col: d[split], color: d.color })),
      { name: 'Average', col: split === 'id' ? 4 : 10, color: 'var(--ink-44)' }];
    const H = top + rows.length * rowH + 6;
    el.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const x = v => L + (v / 100) * (W - L - Rr);
    const g = svg('g', { class: 'grid' }, el);
    [0, 25, 50, 75, 100].forEach(v => { svg('line', { x1: x(v), x2: x(v), y1: top - 8, y2: H - 6 }, g); txt(el, { x: x(v), y: top - 14, 'text-anchor': 'middle' }, v); });
    txt(el, { x: W - 4, y: top - 14, 'text-anchor': 'end', style: 'font-size:11px;fill:var(--ink-44)' }, 'Δ best');
    rows.forEach((r, i) => {
      const y = top + i * rowH + rowH / 2 - 4;
      const bc = bestOf(r.col, ['prop']), bo = bestOf(r.col, ['open']), ours = OURS[r.col];
      const best = bc[r.col] >= bo[r.col] ? bc : bo;
      svg('rect', { x: 0, y: y - 5, width: 9, height: 10, rx: 2, style: `fill:${r.color}` }, el);
      txt(el, { x: 16, y: y + 4, class: 'lbl', style: r.col === 4 || r.col === 10 ? 'font-weight:600' : '' }, r.name);
      txt(el, { x: 16, y: y + 19, style: 'font-size:10.5px;fill:var(--ink-44)' }, `vs ${best[0]} · ${fmt(best[r.col])}`);
      const vals = [bc[r.col], bo[r.col], ours];
      svg('line', { x1: x(Math.min(...vals)), x2: x(Math.max(...vals)), y1: y, y2: y, class: 'stem' }, el);
      const dot = (v, fill, rad, title) => {
        const c = svg('circle', { cx: x(v), cy: y, r: rad, style: `fill:${fill};stroke:var(--paper);stroke-width:2` }, el);
        svg('title', {}, c).textContent = title;
      };
      dot(bc[r.col], C.closed, 6, `${bc[0]}: ${fmt(bc[r.col])}`);
      dot(bo[r.col], C.open, 6, `${bo[0]}: ${fmt(bo[r.col])}`);
      dot(ours, C.ours, 7, `ChronoVision: ${fmt(ours)}`);
      txt(el, { x: x(ours), y: y - 13, 'text-anchor': 'middle', class: 'val', style: 'fill:var(--c1-ink);font-weight:600' }, fmt(ours));
      const dv = ours - best[r.col];
      txt(el, { x: W - 4, y: y + 4, 'text-anchor': 'end', class: 'val', style: `font-weight:600;fill:${dv >= 0 ? 'var(--c1-ink)' : 'var(--ink-44)'}` }, signed(dv));
    });
  }
  seg($('#gapseg'), [['In-domain', 'id'], ['Out-of-domain', 'ood']], split, v => { split = v; draw(); });
  draw();
})();

/* -------------------------------------------------------------- ablation */
(function ablation() {
  const t = $('#abtable');
  let col = 3;
  let last = '';
  t.innerHTML = `<thead><tr><th class="l">Variant</th><th class="l">Accuracy</th></tr></thead><tbody>${ABLATION.map(a => {
    const head = a[1] !== last ? `<tr class="grp-row"><td colspan="2" class="l" style="padding-top:14px;font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-44);font-weight:500">${a[1]}</td></tr>` : '';
    last = a[1];
    return head + `<tr class="${a[2] ? 'ours' : ''}" data-row><td class="l">${a[0]}</td><td class="barcell"><div class="ab-bar"><div class="f" style="width:0"></div><span class="t"></span></div></td></tr>`;
  }).join('')}</tbody>`;
  function draw() {
    $$('tr[data-row]', t).forEach((tr, i) => {
      const v = ABLATION[i][col];
      $('.f', tr).style.width = v + '%';
      $('.t', tr).textContent = fmt(v);
    });
  }
  seg($('#abseg'), [['Overall', 3], ['In-domain', 4], ['Out-of-domain', 5]], col, v => { col = v; draw(); });
  new IntersectionObserver(([e], o) => { if (e.isIntersecting) { draw(); o.disconnect(); } }, { threshold: .25 }).observe(t);
})();

/* ------------------------------------------------- horizontal bar groups */
function hbars(el, rows, d = 1) {
  el.innerHTML = rows.map(([n, v, ours]) => `
    <div class="hb${ours ? ' ours' : ''}"><span class="nm">${n}</span>
      <div class="ab-bar"><div class="f" style="width:${v}%"></div><span class="t">${fmt(v, d)}</span></div></div>`).join('');
}
(function related() {
  let col = 2;
  const draw = () => hbars($('#relbars'), RELATED.map(r => [r[0], r[col], r[1]]));
  seg($('#relseg'), [['Overall', 2], ['In-domain', 3], ['Out-of-domain', 4]], col, v => { col = v; draw(); });
  draw();
})();
(function video() {
  let k = 'holmes';
  const draw = () => { hbars($('#vidbars'), VIDEO[k].rows, VIDEO[k].d); $('#vidnote').textContent = VIDEO[k].note; };
  seg($('#vidseg'), Object.entries(VIDEO).map(([key, v]) => [v.label, key]), k, v => { k = v; draw(); });
  draw();
})();

/* -------------------------------------------------------- analysis charts */
function axes(el, W, H, m, xs, ys, xl, yfmt) {
  const X = i => m.l + (i / (xs.length - 1)) * (W - m.l - m.r);
  const Y = v => H - m.b - ((v - ys[0]) / (ys[ys.length - 1] - ys[0])) * (H - m.t - m.b);
  const g = svg('g', { class: 'grid' }, el);
  ys.forEach(v => { svg('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); txt(el, { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', style: 'font-size:10.5px' }, yfmt(v)); });
  xs.forEach((s, i) => txt(el, { x: X(i), y: H - m.b + 16, 'text-anchor': 'middle', style: 'font-size:10.5px' }, s));
  txt(el, { x: (m.l + W - m.r) / 2, y: H - 2, 'text-anchor': 'middle', style: 'font-size:10.5px;fill:var(--ink-44)' }, xl);
  return { X, Y };
}
const line = (el, pts, color, dash) => {
  svg('path', { d: 'M' + pts.map(p => p.join(',')).join('L'), style: `fill:none;stroke:${color};stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round${dash ? ';stroke-dasharray:5 4' : ''}` }, el);
  pts.forEach(p => svg('circle', { cx: p[0], cy: p[1], r: 3.6, style: `fill:${color};stroke:var(--paper-2);stroke-width:1.5` }, el));
};
const STEPS = ['0', '10', '20', '30', '40'];
const CLEAN = [.512, .364, .245, .182, .146];
const RED = '#e5484d';

(function conv() {
  const el = $('#ch-conv'), W = 320, H = 210, m = { l: 40, r: 12, t: 26, b: 34 };
  el.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const { X, Y } = axes(el, W, H, m, STEPS, [0, .1, .2, .3, .4, .5, .6], 'CoT step', v => v.toFixed(1));
  const wrong = [.508, .456, .412, .435, .428];
  line(el, wrong.map((v, i) => [X(i), Y(v)]), 'var(--ink-44)', true);
  line(el, CLEAN.map((v, i) => [X(i), Y(v)]), 'var(--c1)');
  txt(el, { x: X(4) - 2, y: Y(.428) - 9, 'text-anchor': 'end', style: 'font-size:10.5px;fill:var(--ink-60);font-weight:600' }, 'incorrect 0.428');
  txt(el, { x: X(4) - 2, y: Y(.146) + 17, 'text-anchor': 'end', style: 'font-size:10.5px;fill:var(--c1-ink);font-weight:600' }, 'correct 0.146');
})();

(function noise() {
  const el = $('#ch-noise'), W = 320, H = 210, m = { l: 40, r: 12, t: 26, b: 34 };
  el.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const { X, Y } = axes(el, W, H, m, STEPS, [0, .1, .2, .3, .4, .5, .6], 'CoT step', v => v.toFixed(1));
  svg('rect', { x: X(3) - 9, y: m.t - 4, width: 18, height: H - m.t - m.b + 4, rx: 4, style: `fill:${RED};opacity:.12` }, el);
  txt(el, { x: X(3), y: m.t - 9, 'text-anchor': 'middle', style: `font-size:10.5px;fill:${RED};font-weight:600` }, 'noise');
  const noisy = [.512, .364, .245, .425, .468];
  line(el, noisy.map((v, i) => [X(i), Y(v)]), RED, true);
  line(el, CLEAN.map((v, i) => [X(i), Y(v)]), 'var(--c1)');
  txt(el, { x: X(4) - 2, y: Y(.468) - 9, 'text-anchor': 'end', style: `font-size:10.5px;fill:${RED};font-weight:600` }, 'noised 0.468');
  txt(el, { x: X(4) - 2, y: Y(.146) + 17, 'text-anchor': 'end', style: 'font-size:10.5px;fill:var(--c1-ink);font-weight:600' }, 'clean 0.146');
})();

(function patch() {
  const el = $('#ch-patch'), W = 320, H = 210, m = { l: 12, r: 12, t: 30, b: 34 };
  el.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const lo = -.5, hi = .5, X = v => m.l + ((v - lo) / (hi - lo)) * (W - m.l - m.r);
  const g = svg('g', { class: 'grid' }, el);
  [-.4, -.2, 0, .2, .4].forEach(v => {
    svg('line', { x1: X(v), x2: X(v), y1: m.t - 6, y2: H - m.b, style: v === 0 ? 'stroke:var(--ink-44)' : '' }, g);
    txt(el, { x: X(v), y: H - m.b + 16, 'text-anchor': 'middle', style: 'font-size:10.5px' }, (v > 0 ? '+' : '') + v.toFixed(1));
  });
  txt(el, { x: W / 2, y: H - 2, 'text-anchor': 'middle', style: 'font-size:10.5px;fill:var(--ink-44)' }, 'Δ p(correct order), mean of 5 categories');
  [['incorrect latent → correct run', -.44, RED], ['correct latent → incorrect run', .28, 'var(--c1)']].forEach(([lab, v, c], i) => {
    const y = m.t + 18 + i * 64;
    txt(el, { x: v < 0 ? X(0) + 6 : X(0) - 6, y: y - 8, 'text-anchor': v < 0 ? 'start' : 'end', style: 'font-size:11px;fill:var(--ink)' }, lab);
    svg('rect', { x: Math.min(X(0), X(v)), y, width: Math.abs(X(v) - X(0)), height: 24, rx: 5, style: `fill:${c}` }, el);
    txt(el, { x: v < 0 ? X(v) + 8 : X(v) - 8, y: y + 16, 'text-anchor': v < 0 ? 'start' : 'end', style: 'font-size:12px;font-weight:700;fill:#fff' }, signed(v, 2));
  });
})();

(function probe() {
  const el = $('#ch-probe'), W = 320, H = 210, m = { l: 40, r: 12, t: 26, b: 34 };
  el.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const acc = [18.4, 31.6, 49.2, 65.5, 74.1], lo = 0, hi = 80;
  const Y = v => H - m.b - ((v - lo) / (hi - lo)) * (H - m.t - m.b);
  const g = svg('g', { class: 'grid' }, el);
  [0, 20, 40, 60, 80].forEach(v => { svg('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); txt(el, { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', style: 'font-size:10.5px' }, v + '%'); });
  const bw = (W - m.l - m.r) / STEPS.length;
  const defs = svg('defs', {}, el), gr = svg('linearGradient', { id: 'pg', x1: 0, x2: 0, y1: 1, y2: 0 }, defs);
  svg('stop', { offset: 0, 'stop-color': '#2b3cb3' }, gr); svg('stop', { offset: 1, 'stop-color': '#a94fd6' }, gr);
  acc.forEach((v, i) => {
    const x = m.l + i * bw + bw * .2, w = bw * .6;
    svg('rect', { x, y: Y(v), width: w, height: Y(0) - Y(v), rx: 4, fill: 'url(#pg)', opacity: .35 + .65 * (i / 4) }, el);
    txt(el, { x: x + w / 2, y: Y(v) - 6, 'text-anchor': 'middle', style: 'font-size:10.5px;fill:var(--ink);font-weight:600' }, fmt(v));
    txt(el, { x: x + w / 2, y: H - m.b + 16, 'text-anchor': 'middle', style: 'font-size:10.5px' }, STEPS[i]);
  });
  txt(el, { x: (m.l + W - m.r) / 2, y: H - 2, 'text-anchor': 'middle', style: 'font-size:10.5px;fill:var(--ink-44)' }, 'CoT step · probe accuracy (in-domain)');
})();

(function prefix() {
  const el = $('#ch-prefix'), W = 320, H = 210, m = { l: 40, r: 70, t: 22, b: 34 };
  el.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const { X, Y } = axes(el, W, H, m, ['standard', 'wrong prefix'], [40, 50, 60, 70, 80], 'inference', v => v);
  [['SFT, no RVH', 66.8, 50.8, 'var(--ink-44)', 'var(--ink-60)'], ['ChronoVision', 74.8, 63.2, 'var(--c1)', 'var(--c1-ink)']].forEach(([n, a, b, c, ink]) => {
    line(el, [[X(0), Y(a)], [X(1), Y(b)]], c);
    txt(el, { x: X(0) - 8, y: Y(a) + 4, 'text-anchor': 'end', class: 'val', style: `fill:${ink};font-weight:600` }, fmt(a));
    txt(el, { x: X(1) + 8, y: Y(b) - 3, class: 'val', style: `fill:${ink};font-weight:600` }, `${fmt(b)}  ${signed(b - a)}`);
    txt(el, { x: X(1) + 8, y: Y(b) + 11, style: `font-size:10px;fill:${ink}` }, n);
  });
})();

(function cot() {
  const el = $('#ch-cot'), W = 320, H = 210, L = 92, Rr = 14, top = 22, rowH = 26;
  el.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const rows = [['Fluid', 66.2, 72.3], ['Crystallized', 71.1, 77.8], ['Visuospatial', 68.3, 75.0], ['Mental Sim.', 60.0, 68.0], ['Transform.', 69.1, 78.2], ['Average', 67.6, 74.8]];
  const lo = 55, hi = 85, x = v => L + ((v - lo) / (hi - lo)) * (W - L - Rr);
  const g = svg('g', { class: 'grid' }, el);
  [55, 65, 75, 85].forEach(v => { svg('line', { x1: x(v), x2: x(v), y1: top - 6, y2: top + rows.length * rowH }, g); txt(el, { x: x(v), y: H - 18, 'text-anchor': 'middle', style: 'font-size:10.5px' }, v); });
  rows.forEach(([n, a, b], i) => {
    const y = top + i * rowH + rowH / 2;
    txt(el, { x: 0, y: y + 4, style: `font-size:11px;fill:var(--ink)${n === 'Average' ? ';font-weight:600' : ''}` }, n);
    svg('line', { x1: x(a), x2: x(b), y1: y, y2: y, class: 'stem', style: 'stroke-width:3' }, el);
    const ca = svg('circle', { cx: x(a), cy: y, r: 5, style: 'fill:var(--ink-44);stroke:var(--paper-2);stroke-width:1.5' }, el); svg('title', {}, ca).textContent = `no CoT: ${fmt(a)}`;
    const cb = svg('circle', { cx: x(b), cy: y, r: 5.5, style: 'fill:var(--c1);stroke:var(--paper-2);stroke-width:1.5' }, el); svg('title', {}, cb).textContent = `with CoT: ${fmt(b)}`;
  });
  txt(el, { x: (L + W - Rr) / 2, y: H - 2, 'text-anchor': 'middle', style: 'font-size:10.5px;fill:var(--ink-44)' }, 'grey: no CoT · violet: with CoT');
})();

/* --------------------------------------------------------- generalization */
(function intphys() {
  $('#intbars').innerHTML = INTPHYS.map(([n, b, o]) => `
    <div style="margin-bottom:14px">
      <div style="display:flex;justify-content:space-between;font-size:.9rem;margin-bottom:6px"><span style="font-weight:500;color:var(--ink)">IntPhys 2 · ${n}</span><span class="delta" style="font-size:.85rem;color:var(--c1-ink)">${signed(o - b)}</span></div>
      <div class="ab-bar" style="margin-bottom:4px"><div class="f" style="width:${b}%;background:var(--ink-20)"></div><span class="t" style="left:8px;right:auto;font-weight:500;color:var(--ink-80)">Qwen3.5-9B · ${fmt(b)}</span></div>
      <div class="ab-bar"><div class="f" style="width:${o}%;background:var(--grad)"></div><span class="t" style="left:8px;right:auto;color:#fff">ChronoVision · ${fmt(o)}</span></div>
    </div>`).join('');
})();

(function general() {
  const el = $('#genchart');
  const W = 640, L = 140, Rr = 58, top = 26, rowH = 40;
  const H = top + GENERAL.length * rowH + 10;
  el.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const lo = 20, hi = 100, x = v => L + ((v - lo) / (hi - lo)) * (W - L - Rr);
  const g = svg('g', { class: 'grid' }, el);
  [20, 40, 60, 80, 100].forEach(v => { svg('line', { x1: x(v), x2: x(v), y1: top - 6, y2: H - 8 }, g); txt(el, { x: x(v), y: top - 12, 'text-anchor': 'middle' }, v); });
  txt(el, { x: W - 4, y: top - 12, 'text-anchor': 'end', style: 'font-size:11px;fill:var(--ink-44)' }, 'Δ');
  GENERAL.forEach(([n, b, o], i) => {
    const cy = top + i * rowH + rowH / 2;
    txt(el, { x: 0, y: cy + 4, class: 'lbl' }, n);
    svg('line', { x1: x(b), x2: x(o), y1: cy, y2: cy, class: 'stem', style: 'stroke-width:4' }, el);
    const cb = svg('circle', { cx: x(b), cy, r: 6, style: 'fill:var(--ink-44);stroke:var(--paper);stroke-width:2' }, el);
    svg('title', {}, cb).textContent = `Qwen3.5-9B: ${fmt(b)}`;
    const co = svg('circle', { cx: x(o), cy, r: 6.5, style: 'fill:var(--c1);stroke:var(--paper);stroke-width:2' }, el);
    svg('title', {}, co).textContent = `ChronoVision: ${fmt(o)}`;
    txt(el, { x: x(Math.max(b, o)) + 12, y: cy + 4, class: 'val' }, fmt(o));
    const d = o - b;
    txt(el, { x: W - 4, y: cy + 4, 'text-anchor': 'end', class: 'val', style: `font-weight:600;fill:${d > 0 ? 'var(--c1-ink)' : 'var(--ink-44)'}` }, signed(d));
  });
})();

/* ------------------------------------------------------------ qualitative */
const QCAP = [
  '<strong>Transformation</strong> · ground truth E F D B A C. Grok 4.20 Beta and Claude Sonnet 4.5 break the order; ChronoVision recovers it.',
  '<strong>Mental simulation</strong> · ground truth C E B D F A. Qwen3.5 397B and GLM-4.6V break the order; ChronoVision recovers it.',
  '<strong>Transformation</strong> · ground truth E D C B A F. Gemini 3.0 Flash and GPT-o3 break the order; ChronoVision recovers it.',
];
const setQ = i => {
  $$('#qtabs .chip').forEach((x, j) => x.setAttribute('aria-pressed', String(i === j)));
  $$('#qstage .figcard').forEach((f, j) => f.classList.toggle('on', i === j));
  $('#qcap').innerHTML = QCAP[i];
};
$$('#qtabs .chip').forEach(b => b.addEventListener('click', () => setQ(+b.dataset.i)));
setQ(0);

/* ---------------------------------------------------------------- bibtex */
$('#copybib').addEventListener('click', async e => {
  const b = e.currentTarget;
  try { await navigator.clipboard.writeText($('#bibtex').textContent); b.textContent = 'Copied'; b.classList.add('ok'); }
  catch (err) { b.textContent = 'Select & copy'; }
  setTimeout(() => { b.textContent = 'Copy'; b.classList.remove('ok'); }, 1800);
});
})();
