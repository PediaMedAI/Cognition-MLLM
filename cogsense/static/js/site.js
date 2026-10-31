/* Cognitive Supersensing project page.
   Every number below is copied from the paper's tables; nothing is a screenshot. */
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
const fmt = (v, d = 1) => Number(v).toFixed(d);

/* ------------------------------------------------------------------ data */
const DIMS = [
  { key: 'fluid', short: 'Fluid', name: 'Fluid Intelligence', color: 'var(--d-fluid)', share: 28,
    theory: 'Cattell (1963) · Structure-Mapping Theory (Gentner, 1983)',
    desc: 'Solve <strong>novel reasoning problems</strong> without relying on prior knowledge. The model must look past surface attributes and map <strong>higher-order relational rules</strong>, as in Raven-style progressive matrices.',
    srcs: [['MaRs-VQA', '1.4K', 13], ['PGM', '10K', 94], ['RAVEN', '18K', 169]], train: '29.4K', test: 276, n: 4 },
  { key: 'crystallized', short: 'Crystallized', name: 'Crystallized Intelligence', color: 'var(--d-cryst)', share: 37,
    theory: 'Cattell (1963) · Prototype Theory (Rosch, 1973)',
    desc: 'Use <strong>learned world knowledge</strong> to abstract a semantic concept shared by a set of real images, then pick the option that belongs to it. Relies on <strong>inductive reasoning</strong> over visual variance.',
    srcs: [['Bongard-RWR+', '16K', 152], ['Bongard-HOI', '23K', 216]], train: '39K', test: 368, n: 4 },
  { key: 'visuospatial', short: 'Visuospatial', name: 'Visuospatial Cognition', color: 'var(--d-visuo)', share: 11,
    theory: 'Gestalt laws (Wertheimer, 1923) · Recognition-by-Components (Biederman, 1987)',
    desc: 'Reconstruct <strong>structural relationships</strong>: group discrete strokes into coherent wholes and decompose shapes into <strong>geometric primitives</strong> to decide which concept a drawing follows.',
    srcs: [['Bongard-LOGO', '12K', 113]], train: '12K', test: 113, n: 4 },
  { key: 'simulation', short: 'Simulation', name: 'Mental Simulation', color: 'var(--d-sim)', share: 14,
    theory: 'Intuitive physics engine (Battaglia et al., 2013) · Hypothetico-deductive reasoning',
    desc: 'Act as a <strong>simulation engine</strong>: infer the hidden transformation from an example, then <strong>apply it</strong> to a new input, from object rotations to ARC-style grid programs.',
    srcs: [['KiVA', '1.4K', 13], ['STARE', '4K', 26], ['ARC-AGI', '1.6K', 31], ['ARC-AGI-2', '8K', 80]], train: '15K', test: 150, n: 4 },
  { key: 'routines', short: 'Routines', name: 'Visual Routines', color: 'var(--d-rout)', share: 10,
    theory: 'Visual Routines (Ullman, 1984) · Focused attention',
    desc: 'Compose <strong>elementary visual operations</strong> for efficient search. Finding the odd one out requires binding features with focused attention and <strong>inhibiting</strong> distracting similarities.',
    srcs: [['CVR', '10K', 93]], train: '10K', test: 93, n: 3 },
];

// [name, group, logo, fluid, cryst, visuo, sim, routines, avg]
const MODELS = [
  ['Human', 'human', 'human', 82.7, 91.3, 88.5, 97.9, 78.7, 88.4],
  ['Gemini 2.5 Flash', 'prop', 'gemini', 23.2, 40.2, 31.0, 40.0, 45.2, 34.9],
  ['GPT-o3', 'prop', 'gpt', 4.7, 51.4, 20.4, 38.7, 43.0, 32.3],
  ['GPT-5.2', 'prop', 'gpt', 29.3, 35.9, 57.5, 60.0, 37.6, 40.3],
  ['Claude Sonnet 4', 'prop', 'claude', 22.5, 31.3, 26.5, 58.0, 34.4, 32.6],
  ['Grok 4 Fast', 'prop', 'X', 13.0, 45.4, 41.6, 21.3, 37.6, 31.7],
  ['Llama-4-Scout-17B', 'open', 'meta', 20.3, 29.9, 35.4, 48.7, 41.9, 31.8],
  ['Gemma-3-27B', 'open', 'google', 18.5, 29.3, 39.8, 55.3, 43.0, 32.7],
  ['Ministral-3-14B', 'open', 'mistral', 9.4, 10.6, 10.6, 39.3, 31.2, 16.5],
  ['Intern-S1', 'open', 'In', 6.5, 20.9, 12.4, 17.3, 43.0, 17.5],
  ['Qwen3-VL-30B', 'open', 'qwen', 30.8, 34.0, 37.2, 56.0, 40.9, 37.4],
  ['Qwen3-VL-8B', 'ft', 'qwen', 51.1, 76.6, 63.7, 59.3, 41.9, 62.3],
  ['InternVL3.5-8B', 'ft', 'In', 52.2, 80.2, 63.7, 59.3, 41.9, 63.9],
  ['Qwen3.5-9B', 'ft', 'qwen', 56.5, 89.1, 61.9, 62.7, 46.2, 69.1],
  ['CogSense-8B', 'ours', 'ours', 63.8, 91.0, 69.0, 68.0, 50.5, 73.8],
];
const GROUP_LABEL = { human: 'Reference', prop: 'Closed-source', open: 'Open-source', ft: 'Fine-tuned', ours: 'Ours' };
const COLS = ['Fluid Intel.', 'Crystallized Intel.', 'Visuospatial Cog.', 'Mental Simu.', 'Visual Rout.', 'Avg.'];

// [label, SFT, LVIP, GRPO, LR, fluid, cryst, visuo, sim, routines, avg]
const ABLATION = [
  ['Qwen3-VL-8B (base)', 0, 0, 0, 0, 31.2, 34.8, 31.0, 45.3, 40.9, 35.5],
  ['SFT', 1, 0, 0, 0, 51.1, 76.6, 63.7, 59.3, 41.9, 62.3],
  ['SFT + LVIP', 1, 1, 0, 0, 55.4, 88.6, 61.1, 61.3, 44.1, 68.0],
  ['SFT + GRPO', 1, 0, 1, 0, 55.8, 79.9, 63.7, 63.3, 43.0, 65.5],
  ['SFT + LVIP + GRPO', 1, 1, 1, 0, 59.1, 89.9, 64.6, 65.3, 46.2, 70.8],
  ['SFT + LR', 1, 0, 0, 1, 57.2, 80.7, 65.5, 65.3, 45.2, 66.9],
  ['SFT + GRPO + LR', 1, 0, 1, 1, 56.5, 80.4, 64.6, 64.0, 44.1, 66.2],
  ['SFT + LVIP + GRPO + LR', 1, 1, 1, 1, 59.8, 90.2, 65.5, 66.7, 47.3, 71.5],
  ['CogSense-8B', 1, 1, 0, 1, 63.8, 91.0, 69.0, 68.0, 50.5, 73.8],
];

const GENERAL = [
  ['GQA', 71.4, 71.8, 'General'], ['ScienceQA', 92.6, 92.6, 'General'], ['RealWorldQA', 71.5, 71.9, 'General'],
  ['BLINK', 64.7, 65.3, 'General'], ['HallusionBench', 61.1, 60.5, 'General'],
  ['MMSI-Bench', 28.0, 32.4, 'Spatial'], ['ViewSpatial-Bench', 38.6, 40.8, 'Spatial'], ['ERQA', 46.8, 48.8, 'Spatial'],
];
const EMMA = [['Chemistry', 39.2, 45.4], ['Mathematics', 26.0, 34.8]];

/* ------------------------------------------------------------- theming */
const root = document.documentElement;
$('#themeToggle').addEventListener('click', () => {
  const dark = root.dataset.theme ? root.dataset.theme === 'dark'
    : matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
  try { localStorage.setItem('cs-theme', root.dataset.theme); } catch (e) {}
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

/* --------------------------------------------------- benchmark explorer */
const best = (i, groups) => MODELS.filter(m => groups.includes(m[1])).reduce((a, b) => (b[i] > a[i] ? b : a));
let curDim = 0;
const tabs = $('#dimtabs'), info = $('#diminfo'), stage = $('#vstage'), thumbs = $('#vthumbs');

DIMS.forEach((d, i) => {
  const b = document.createElement('button');
  b.className = 'dimtab'; b.setAttribute('role', 'tab'); b.style.setProperty('--c', d.color);
  b.innerHTML = `<span class="sw"></span>${d.name}`;
  b.addEventListener('click', () => selectDim(i));
  tabs.appendChild(b);
});

function selectDim(i) {
  curDim = i;
  const d = DIMS[i], col = 3 + i;
  $$('.dimtab', tabs).forEach((b, j) => b.setAttribute('aria-selected', String(i === j)));
  const human = MODELS[0][col], ours = MODELS.find(m => m[1] === 'ours')[col];
  const bc = best(col, ['prop']);
  info.style.setProperty('--c', d.color);
  info.innerHTML = `
    <h3><span class="sw"></span>${d.name}</h3>
    <div class="theory">${d.theory}</div>
    <p class="desc">${d.desc}</p>
    <table class="srcs">
      <thead><tr><th>Source dataset</th><th># Train</th><th># Test</th></tr></thead>
      <tbody>
        ${d.srcs.map(s => `<tr><td>${s[0]}</td><td>${s[1]}</td><td>${s[2]}</td></tr>`).join('')}
        <tr class="tot"><td>${d.share}% of train split</td><td>${d.train}</td><td>${d.test}</td></tr>
      </tbody>
    </table>
    <div class="dim-perf">
      <div class="pp"><div class="v">${fmt(bc[col])}</div><div class="t">Best closed<br>${bc[0]}</div></div>
      <div class="pp ours"><div class="v">${fmt(ours)}</div><div class="t">CogSense-8B<br>(ours)</div></div>
      <div class="pp"><div class="v">${fmt(human)}</div><div class="t">Human<br>reference</div></div>
    </div>`;
  stage.innerHTML = ''; thumbs.innerHTML = '';
  for (let k = 1; k <= d.n; k++) {
    const src = `static/images/web/bench_${d.key}_${k}.webp`;
    const img = document.createElement('img');
    img.src = src; img.alt = `${d.name} example ${k}`; img.width = 580; img.height = 849;
    if (k === 1) img.className = 'on';
    stage.appendChild(img);
    const t = document.createElement('button');
    t.setAttribute('role', 'tab'); t.setAttribute('aria-selected', String(k === 1)); t.setAttribute('aria-label', `Example ${k}`);
    t.innerHTML = `<img src="${src}" alt="" loading="lazy">`;
    t.addEventListener('click', () => showEx(k - 1));
    thumbs.appendChild(t);
  }
  hiDonut(i);
}
function showEx(k) {
  $$('img', stage).forEach((im, j) => im.classList.toggle('on', j === k));
  $$('button', thumbs).forEach((b, j) => b.setAttribute('aria-selected', String(j === k)));
}

/* ---------------------------------------------------------------- donut */
const donut = $('#donut'), legend = $('#donutlegend'), dwrap = $('#donutwrap');
const cx = 130, cy = 130, R = 118, r0 = 72;
let a0 = -Math.PI / 2;
const arcs = [];
const pt = (rad, a) => [cx + rad * Math.cos(a), cy + rad * Math.sin(a)];
DIMS.forEach((d, i) => {
  const a1 = a0 + (d.share / 100) * Math.PI * 2, gap = 0.012;
  const s = a0 + gap, e = a1 - gap, large = e - s > Math.PI ? 1 : 0;
  const [x0, y0] = pt(R, s), [x1, y1] = pt(R, e), [x2, y2] = pt(r0, e), [x3, y3] = pt(r0, s);
  const p = svg('path', { d: `M${x0},${y0} A${R},${R} 0 ${large} 1 ${x1},${y1} L${x2},${y2} A${r0},${r0} 0 ${large} 0 ${x3},${y3}Z`, style: `fill:${d.color}` }, donut);
  const mid = (a0 + a1) / 2, [tx, ty] = pt((R + r0) / 2, mid);
  const t = svg('text', { x: tx, y: ty + 4, 'text-anchor': 'middle', style: 'font-size:12px;font-weight:600;fill:#1a1a1a;pointer-events:none' }, donut);
  t.textContent = d.share + '%';
  arcs.push(p);
  p.addEventListener('mouseenter', () => hiDonut(i));
  p.addEventListener('click', () => selectDim(i));
  a0 = a1;
  const li = document.createElement('li');
  li.style.setProperty('--c', d.color);
  li.innerHTML = `<span class="sw"></span><span class="nm">${d.name}<span class="ds">${d.srcs.map(s => `${s[0]} (${s[1]})`).join(', ')}</span></span><span class="pc">${d.share}%</span>`;
  li.addEventListener('mouseenter', () => hiDonut(i));
  li.addEventListener('click', () => selectDim(i));
  legend.appendChild(li);
});
const c1 = svg('text', { x: cx, y: cy - 4, 'text-anchor': 'middle', style: 'font-size:22px;font-weight:600;fill:var(--ink);letter-spacing:-.5px' }, donut);
const c2 = svg('text', { x: cx, y: cy + 16, 'text-anchor': 'middle', style: 'font-size:11px;fill:var(--ink-60)' }, donut);
const setCenter = (a, b) => { c1.textContent = a; c2.textContent = b; };
setCenter('105.4K', 'train VQA pairs');
function hiDonut(i, hover) {
  if (!arcs.length) return;
  dwrap.classList.add('dim-dim');
  arcs.forEach((p, j) => p.style.opacity = j === i ? 1 : .35);
  $$('li', legend).forEach((l, j) => l.classList.toggle('hot', j === i));
  setCenter(DIMS[i].train, DIMS[i].short + ' pairs');
}
dwrap.addEventListener('mouseleave', () => {
  dwrap.classList.remove('dim-dim');
  arcs.forEach(p => p.style.opacity = 1);
  setCenter('105.4K', 'train VQA pairs');
});
selectDim(0);
// the donut starts neutral; the explorer only highlights it once the user picks a dimension
dwrap.classList.remove('dim-dim'); arcs.forEach(p => p.style.opacity = 1); setCenter('105.4K', 'train VQA pairs');

/* ---------------------------------------------------------- leaderboard */
const LOGO = {
  gpt: ['gpt.svg', true], gemini: ['gemini.svg'], claude: ['claude.svg'], meta: ['meta.svg'], google: ['google.svg'],
  mistral: ['mistral.svg'], qwen: ['qwen.svg'],
};
const logoHTML = k => {
  if (k === 'ours') return '<img src="static/images/web/logo.webp?v=2" alt="">';
  if (k === 'human') return '<svg class="ph" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="background:none"><circle cx="12" cy="8" r="4"/><path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6" stroke-linecap="round"/></svg>';
  if (LOGO[k]) return `<img src="static/images/logos/${LOGO[k][0]}" alt=""${LOGO[k][1] ? ' class="mono"' : ''}>`;
  return `<span class="ph">${k}</span>`;
};
const ranked = MODELS.filter(m => m[1] !== 'human').slice().sort((a, b) => b[8] - a[8]);
const RANK = new Map(ranked.map((m, i) => [m[0], i + 1]));
const STRONGEST = MODELS.find(m => m[0] === 'Qwen3.5-9B');
const colRank = COLS.map((_, c) => {
  const vals = MODELS.filter(m => m[1] !== 'human').map(m => m[3 + c]).sort((a, b) => b - a);
  return { best: vals[0], second: vals.find(v => v < vals[0]) };
});
let sortCol = 5, sortDir = -1, group = 'all';

function drawLB() {
  const t = $('#lb');
  const head = `<thead><tr><th>#</th><th>Model</th><th>Type</th>${COLS.map((c, i) =>
    `<th class="sortable${i === 5 ? ' c-avg' : ''}${i === sortCol ? ' sorted' : ''}" data-c="${i}">${c}<span class="arrow">${sortDir < 0 ? '▼' : '▲'}</span></th>`).join('')}</tr></thead>`;
  let rows = MODELS.filter(m => m[1] !== 'human' && (group === 'all' || m[1] === group || m[1] === 'ours'));
  rows.sort((a, b) => sortDir * (a[3 + sortCol] - b[3 + sortCol]) || (b[8] - a[8]));
  const human = MODELS[0];
  const row = m => {
    const isH = m[1] === 'human', isO = m[1] === 'ours', rk = RANK.get(m[0]);
    const rankCell = isH ? '' : `<span class="rank${rk <= 3 ? ' r' + rk : ''}">${rk}</span>`;
    const tag = m[1] === 'ft' ? '<span class="tag">FT</span>' : isO ? '<span class="tag">ours</span>' : '';
    const cells = COLS.map((_, c) => {
      const v = m[3 + c], cr = colRank[c];
      const cls = isH ? '' : v === cr.best ? ' best' : v === cr.second ? ' second' : '';
      const d = isO ? `<span class="delta">+${fmt(v - STRONGEST[3 + c])}</span>` : '';
      return `<td class="${c === 5 ? 'c-avg' : ''}"><span class="cell${cls}">${fmt(v)}${isH ? '' : `<span class="bar" style="width:${v}%"></span>`}</span>${d}</td>`;
    }).join('');
    return `<tr class="${isH ? 'human' : isO ? 'ours' : ''}"><td>${rankCell}</td><td><span class="mname">${logoHTML(m[2])}${m[0]}${tag}</span></td><td class="grp">${GROUP_LABEL[m[1]]}</td>${cells}</tr>`;
  };
  t.innerHTML = head + '<tbody>' + row(human) + '<tr class="sep"><td colspan="9"></td></tr>' + rows.map(row).join('') + '</tbody>';
  $$('th.sortable', t).forEach(th => th.addEventListener('click', () => {
    const c = +th.dataset.c;
    if (c === sortCol) sortDir = -sortDir; else { sortCol = c; sortDir = -1; }
    drawLB();
  }));
}
$$('#lbtools .chip').forEach(b => b.addEventListener('click', () => {
  group = b.dataset.g;
  $$('#lbtools .chip').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
  drawLB();
}));
drawLB();
$('.lb-note').innerHTML = '<b>Bold</b>: best model. <u>Underline</u>: runner-up. Red numbers: CogSense-8B\'s margin over the strongest baseline, Qwen3.5-9B (FT). The human row is a reference and is not ranked.';

/* ------------------------------------------------------------- gap chart */
(function gapChart() {
  const el = $('#gapchart');
  const W = 640, L = 190, Rr = 24, top = 30, rowH = 62;
  const rows = [...DIMS.map((d, i) => ({ name: d.name, col: 3 + i, color: d.color })), { name: 'Average', col: 8, color: 'var(--ink-44)' }];
  const H = top + rows.length * rowH + 10;
  el.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const x = v => L + (v / 100) * (W - L - Rr);
  const g = svg('g', { class: 'grid' }, el);
  [0, 25, 50, 75, 100].forEach(v => {
    svg('line', { x1: x(v), x2: x(v), y1: top - 8, y2: H - 10 }, g);
    const t = svg('text', { x: x(v), y: top - 14, 'text-anchor': 'middle' }, el); t.textContent = v;
  });
  const C = { closed: 'var(--ink-44)', ft: 'var(--blue)', ours: 'var(--orange)' };
  const lg = $('#gaplegend');
  lg.innerHTML = `<span><i style="background:${C.closed}"></i>Best closed-source</span><span><i style="background:${C.ft}"></i>Best fine-tuned baseline</span><span><i style="background:${C.ours}"></i>CogSense-8B</span><span><i class="ring"></i>Human</span>`;
  rows.forEach((r, i) => {
    const y = top + i * rowH + rowH / 2;
    const bc = best(r.col, ['prop']), bf = best(r.col, ['ft']);
    const ours = MODELS.find(m => m[1] === 'ours')[r.col], human = MODELS[0][r.col];
    const lab = svg('text', { x: 16, y: y + 4, class: 'lbl', style: r.col === 8 ? 'font-weight:600' : '' }, el); lab.textContent = r.name;
    svg('rect', { x: 0, y: y - 5, width: 9, height: 10, rx: 2, style: `fill:${r.color}` }, el);
    svg('line', { x1: x(bc[r.col]), x2: x(human), y1: y, y2: y, class: 'stem' }, el);
    const dot = (v, fill, rad, title) => {
      const c = svg('circle', { cx: x(v), cy: y, r: rad, style: `fill:${fill};stroke:var(--paper);stroke-width:2` }, el);
      svg('title', {}, c).textContent = title;
      return c;
    };
    const hc = svg('circle', { cx: x(human), cy: y, r: 9, style: 'fill:none;stroke:var(--ink-60);stroke-width:2' }, el);
    svg('title', {}, hc).textContent = `Human: ${fmt(human)}`;
    dot(bc[r.col], C.closed, 6, `${bc[0]}: ${fmt(bc[r.col])}`);
    dot(bf[r.col], C.ft, 6, `${bf[0]} (FT): ${fmt(bf[r.col])}`);
    dot(ours, C.ours, 7, `CogSense-8B: ${fmt(ours)}`);
    const tn = svg('text', { x: x(bc[r.col]) - 12, y: y + 4, 'text-anchor': 'end', style: 'font-size:10.5px;fill:var(--ink-44)' }, el);
    tn.textContent = `${bc[0]} · ${fmt(bc[r.col])}`;
    const to = svg('text', { x: x(ours), y: y + 23, 'text-anchor': 'middle', class: 'val', style: 'fill:var(--orange-ink);font-weight:600' }, el); to.textContent = fmt(ours);
    const th = svg('text', { x: x(human), y: y - 15, 'text-anchor': 'middle', class: 'val' }, el); th.textContent = fmt(human);
  });
})();

/* -------------------------------------------------------------- ablation */
(function ablation() {
  const seg = $('#abseg'), t = $('#abtable');
  const opts = [['Avg.', 10], ['Fluid', 5], ['Crystallized', 6], ['Visuospatial', 7], ['Simulation', 8], ['Routines', 9]];
  let col = 10;
  opts.forEach(([n, c]) => {
    const b = document.createElement('button'); b.textContent = n; b.setAttribute('aria-pressed', String(c === col));
    b.addEventListener('click', () => { col = c; $$('button', seg).forEach(x => x.setAttribute('aria-pressed', String(x === b))); draw(); });
    seg.appendChild(b);
  });
  const mark = v => v ? '<span class="chk">✓</span>' : '<span class="no">–</span>';
  t.innerHTML = `<thead><tr><th class="l">Variant</th><th>SFT</th><th>LVIP</th><th>GRPO</th><th>Latent<br>Rationales</th><th class="l">Accuracy</th></tr></thead><tbody>${
    ABLATION.map((a, i) => `<tr class="${i === ABLATION.length - 1 ? 'ours' : ''}"><td class="l">${a[0]}</td><td>${mark(a[1])}</td><td>${mark(a[2])}</td><td>${mark(a[3])}</td><td>${mark(a[4])}</td><td class="barcell"><div class="ab-bar"><div class="f" style="width:0"></div><span class="t"></span></div></td></tr>`).join('')
  }</tbody>`;
  function draw() {
    $$('tbody tr', t).forEach((tr, i) => {
      const v = ABLATION[i][col];
      $('.f', tr).style.width = v + '%';
      $('.t', tr).textContent = fmt(v);
    });
  }
  // grow the bars when they first scroll into view
  new IntersectionObserver(([e], o) => { if (e.isIntersecting) { draw(); o.disconnect(); } }, { threshold: .25 }).observe(t);
})();

/* -------------------------------------------------------- analysis charts */
function axes(el, W, H, m, xs, ys, xl, yfmt) {
  const X = i => m.l + (i / (xs.length - 1)) * (W - m.l - m.r);
  const Y = v => H - m.b - ((v - ys[0]) / (ys[ys.length - 1] - ys[0])) * (H - m.t - m.b);
  const g = svg('g', { class: 'grid' }, el);
  ys.forEach(v => {
    svg('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g);
    const t = svg('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', style: 'font-size:10.5px' }, el); t.textContent = yfmt(v);
  });
  xs.forEach((s, i) => { const t = svg('text', { x: X(i), y: H - m.b + 16, 'text-anchor': 'middle', style: 'font-size:10.5px' }, el); t.textContent = s; });
  const xt = svg('text', { x: (m.l + W - m.r) / 2, y: H - 2, 'text-anchor': 'middle', style: 'font-size:10.5px;fill:var(--ink-44)' }, el); xt.textContent = xl;
  return { X, Y };
}
const line = (el, pts, color, dash) => {
  svg('path', { d: 'M' + pts.map(p => p.join(',')).join('L'), style: `fill:none;stroke:${color};stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round${dash ? ';stroke-dasharray:5 4' : ''}` }, el);
  pts.forEach(p => svg('circle', { cx: p[0], cy: p[1], r: 3.6, style: `fill:${color};stroke:var(--paper-2);stroke-width:1.5` }, el));
};

(function noise() {
  const el = $('#ch-noise'), W = 320, H = 210, m = { l: 40, r: 12, t: 26, b: 34 };
  el.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const steps = ['0', '10', '20', '30', '40'];
  const { X, Y } = axes(el, W, H, m, steps, [0, .1, .2, .3, .4, .5], 'CoT step', v => v.toFixed(1));
  svg('rect', { x: X(3) - 9, y: m.t - 4, width: 18, height: H - m.t - m.b + 4, rx: 4, style: 'fill:var(--orange);opacity:.12' }, el);
  const nt = svg('text', { x: X(3), y: m.t - 9, 'text-anchor': 'middle', style: 'font-size:10.5px;fill:var(--orange-ink);font-weight:600' }, el); nt.textContent = 'noise';
  const clean = [.492, .368, .215, .184, .158], noisy = [.492, .368, .215, .374, .418];
  line(el, noisy.map((v, i) => [X(i), Y(v)]), 'var(--orange)', true);
  line(el, clean.map((v, i) => [X(i), Y(v)]), 'var(--blue)');
  const l1 = svg('text', { x: X(4) - 2, y: Y(.418) - 9, 'text-anchor': 'end', style: 'font-size:10.5px;fill:var(--orange-ink);font-weight:600' }, el); l1.textContent = 'noised 0.418';
  const l2 = svg('text', { x: X(4) - 2, y: Y(.158) + 17, 'text-anchor': 'end', style: 'font-size:10.5px;fill:var(--blue-ink);font-weight:600' }, el); l2.textContent = 'clean 0.158';
})();

(function patch() {
  const el = $('#ch-patch'), W = 320, H = 210, m = { l: 12, r: 12, t: 30, b: 34 };
  el.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const lo = -.5, hi = .5, X = v => m.l + ((v - lo) / (hi - lo)) * (W - m.l - m.r);
  const g = svg('g', { class: 'grid' }, el);
  [-.4, -.2, 0, .2, .4].forEach(v => {
    svg('line', { x1: X(v), x2: X(v), y1: m.t - 6, y2: H - m.b, style: v === 0 ? 'stroke:var(--ink-44)' : '' }, g);
    const t = svg('text', { x: X(v), y: H - m.b + 16, 'text-anchor': 'middle', style: 'font-size:10.5px' }, el); t.textContent = (v > 0 ? '+' : '') + v.toFixed(1);
  });
  const xt = svg('text', { x: W / 2, y: H - 2, 'text-anchor': 'middle', style: 'font-size:10.5px;fill:var(--ink-44)' }, el); xt.textContent = 'Δ p(correct answer)';
  const bars = [['correct → incorrect patch', -.43, 'var(--orange)'], ['incorrect → correct patch', .22, 'var(--blue)']];
  bars.forEach(([lab, v, c], i) => {
    const y = m.t + 18 + i * 64;
    const t = svg('text', { x: v < 0 ? X(0) + 6 : X(0) - 6, y: y - 8, 'text-anchor': v < 0 ? 'start' : 'end', style: 'font-size:11px;fill:var(--ink)' }, el); t.textContent = lab;
    svg('rect', { x: Math.min(X(0), X(v)), y, width: Math.abs(X(v) - X(0)), height: 24, rx: 5, style: `fill:${c}` }, el);
    const vt = svg('text', { x: v < 0 ? X(v) + 8 : X(v) - 8, y: y + 16, 'text-anchor': v < 0 ? 'start' : 'end', style: 'font-size:12px;font-weight:700;fill:#fff' }, el);
    vt.textContent = (v > 0 ? '+' : '−') + Math.abs(v).toFixed(2);
  });
})();

(function probe() {
  const el = $('#ch-probe'), W = 320, H = 210, m = { l: 40, r: 12, t: 26, b: 34 };
  el.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const steps = ['0', '10', '20', '30', '40'], acc = [.173, .237, .394, .552, .621];
  const lo = 0, hi = .8;
  const Y = v => H - m.b - ((v - lo) / (hi - lo)) * (H - m.t - m.b);
  const g = svg('g', { class: 'grid' }, el);
  [0, .2, .4, .6, .8].forEach(v => {
    svg('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g);
    const t = svg('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', style: 'font-size:10.5px' }, el); t.textContent = v.toFixed(1);
  });
  const bw = (W - m.l - m.r) / steps.length;
  const defs = svg('defs', {}, el), gr = svg('linearGradient', { id: 'pg', x1: 0, x2: 0, y1: 1, y2: 0 }, defs);
  svg('stop', { offset: 0, 'stop-color': '#2f9be8' }, gr); svg('stop', { offset: 1, 'stop-color': '#f08419' }, gr);
  acc.forEach((v, i) => {
    const x = m.l + i * bw + bw * .2, w = bw * .6;
    svg('rect', { x, y: Y(v), width: w, height: Y(0) - Y(v), rx: 4, fill: 'url(#pg)', opacity: .35 + .65 * (i / 4) }, el);
    const t = svg('text', { x: x + w / 2, y: Y(v) - 6, 'text-anchor': 'middle', style: 'font-size:10.5px;fill:var(--ink);font-weight:600' }, el); t.textContent = v.toFixed(2);
    const s = svg('text', { x: x + w / 2, y: H - m.b + 16, 'text-anchor': 'middle', style: 'font-size:10.5px' }, el); s.textContent = steps[i];
  });
  const xt = svg('text', { x: (m.l + W - m.r) / 2, y: H - 2, 'text-anchor': 'middle', style: 'font-size:10.5px;fill:var(--ink-44)' }, el); xt.textContent = 'CoT step · probe accuracy';
})();

/* --------------------------------------------------------- generalization */
(function emma() {
  const el = $('#emmabars');
  el.innerHTML = EMMA.map(([n, b, o]) => `
    <div style="margin-bottom:16px">
      <div style="display:flex;justify-content:space-between;font-size:.9rem;margin-bottom:6px"><span style="font-weight:500;color:var(--ink)">EMMA · ${n}</span><span class="delta" style="font-size:.85rem">+${fmt(o - b)}</span></div>
      <div class="ab-bar" style="margin-bottom:4px"><div class="f" style="width:${b}%;background:var(--ink-20)"></div><span class="t" style="left:8px;right:auto;font-weight:500;color:var(--ink-80)">Qwen3-VL-8B · ${fmt(b)}</span></div>
      <div class="ab-bar"><div class="f" style="width:${o}%;background:var(--grad)"></div><span class="t" style="left:8px;right:auto;color:#fff">CogSense-8B · ${fmt(o)}</span></div>
    </div>`).join('');
})();

(function general() {
  const el = $('#genchart');
  const W = 640, L = 140, Rr = 58, top = 26, rowH = 40, gapH = 26;
  const H = top + GENERAL.length * rowH + gapH * 2 + 10;
  el.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const lo = 20, hi = 100, x = v => L + ((v - lo) / (hi - lo)) * (W - L - Rr);
  const g = svg('g', { class: 'grid' }, el);
  [20, 40, 60, 80, 100].forEach(v => {
    svg('line', { x1: x(v), x2: x(v), y1: top - 6, y2: H - 8 }, g);
    const t = svg('text', { x: x(v), y: top - 12, 'text-anchor': 'middle' }, el); t.textContent = v;
  });
  const dh = svg('text', { x: W - 4, y: top - 12, 'text-anchor': 'end', style: 'font-size:11px;fill:var(--ink-44)' }, el); dh.textContent = 'Δ';
  let y = top, last = '';
  GENERAL.forEach(([n, b, o, grp]) => {
    if (grp !== last) {
      y += gapH;
      const gt = svg('text', { x: 0, y: y - 6, style: 'font-size:10.5px;font-weight:600;letter-spacing:.08em;fill:var(--ink-44)' }, el); gt.textContent = grp.toUpperCase();
      last = grp;
    }
    const cy = y + rowH / 2 - 6;
    const lt = svg('text', { x: 0, y: cy + 4, class: 'lbl' }, el); lt.textContent = n;
    svg('line', { x1: x(b), x2: x(o), y1: cy, y2: cy, class: 'stem', style: 'stroke-width:4' }, el);
    const cb = svg('circle', { cx: x(b), cy, r: 6, style: 'fill:var(--ink-44);stroke:var(--paper);stroke-width:2' }, el);
    svg('title', {}, cb).textContent = `Qwen3-VL-8B: ${fmt(b)}`;
    const co = svg('circle', { cx: x(o), cy, r: 6.5, style: 'fill:var(--orange);stroke:var(--paper);stroke-width:2' }, el);
    svg('title', {}, co).textContent = `CogSense-8B: ${fmt(o)}`;
    const vt = svg('text', { x: x(Math.max(b, o)) + 12, y: cy + 4, class: 'val' }, el); vt.textContent = fmt(o);
    const d = o - b;
    const dt = svg('text', { x: W - 4, y: cy + 4, 'text-anchor': 'end', class: 'val', style: `font-weight:600;fill:${d > 0 ? 'var(--orange-ink)' : d < 0 ? 'var(--ink-44)' : 'var(--ink-44)'}` }, el);
    dt.textContent = d === 0 ? '±0.0' : (d > 0 ? '+' : '−') + fmt(Math.abs(d));
    y += rowH;
  });
})();

/* ------------------------------------------------------------ qualitative */
$$('#qtabs .chip').forEach(b => b.addEventListener('click', () => {
  const i = +b.dataset.i;
  $$('#qtabs .chip').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
  $$('#qstage .figcard').forEach((f, j) => f.classList.toggle('on', i === j));
}));

/* ---------------------------------------------------------------- bibtex */
$('#copybib').addEventListener('click', async e => {
  const b = e.currentTarget;
  try { await navigator.clipboard.writeText($('#bibtex').textContent); b.textContent = 'Copied'; b.classList.add('ok'); }
  catch (err) { b.textContent = 'Select & copy'; }
  setTimeout(() => { b.textContent = 'Copy'; b.classList.remove('ok'); }, 1800);
});
})();
