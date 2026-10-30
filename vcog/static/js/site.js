/* "What is the Visual Cognition Gap between Humans and Multimodal LLMs?" project page.
   Every number below is copied from the paper's tables (COLM 2025, arXiv 2406.10424); nothing is a screenshot. */
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
const fmt = (v, d = 2) => Number(v).toFixed(d);
const signed = (v, d = 2) => (v > 0 ? '+' : v < 0 ? '−' : '±') + fmt(Math.abs(v), d);

/* ------------------------------------------------------------------ data */
// Example items from the paper's difficulty table. T1 is the MaRs-IB target; `order` fixes how options are shown.
const LEVELS = [
  { n: 1, attrs: 'Shape + Size', rule: 'Only one attribute changes, or two simple ones together, such as shape and size.', count: 90, gpt4o: 57.78, order: ['t3', 't1', 't4', 't2'] },
  { n: 2, attrs: 'Color + Multi-object', rule: 'Several objects per cell, plus one more changing attribute such as color.', count: 96, gpt4o: 27.08, order: ['t2', 't3', 't1', 't4'] },
  { n: 3, attrs: 'Shape + Color + Position', rule: 'Three simple attributes change at once.', count: 84, gpt4o: 27.38, order: ['t2', 't4', 't3', 't1'] },
  { n: 4, attrs: 'Shape + Color + Multi-object', rule: 'Several objects per cell, plus two more changing attributes.', count: 72, gpt4o: 19.43, order: ['t1', 't4', 't2', 't3'] },
  { n: 5, attrs: 'Shape + Color + Position + Multi-object', rule: 'Four or more attributes change at once.', count: 138, gpt4o: 21.74, order: ['t3', 't1', 't2', 't4'] },
];

// [name, group, logo, scale, MaRs-VQA (4 options), RAVEN (8 options)]
const MODELS = [
  ['Human', 'human', 'human', '–', 81.00, 84.41],
  ['LLaVA-NeXT', 'open', 'Lv', '7B', 16.88, 14.29],
  ['InternVL-2.5', 'open', 'In', '8B', 20.00, 13.21],
  ['Qwen2-VL', 'open', 'qwen', '7B', 23.75, 29.27],
  ['Claude 3 Sonnet', 'prop', 'claude', '–', 23.22, 13.39],
  ['Claude 3 Opus', 'prop', 'claude', '–', 24.13, 11.95],
  ['Claude 3.5 Sonnet', 'prop', 'claude', '–', 24.28, 15.36],
  ['GPT-4V', 'prop', 'gpt', '–', 33.13, 15.63],
  ['GPT-4o', 'prop', 'gpt', '–', 33.96, 25.89],
  ['GPT-o1', 'prop', 'gpt', '–', 52.29, 25.36],
  ['Qwen2-VCog', 'ours', 'ours', '7B', 72.71, 31.96],
  ['Random', 'random', 'random', '–', 25.00, 12.50],
];
const GROUP_LABEL = { human: 'Reference', random: 'Reference', prop: 'Closed-source', open: 'Open-source', ours: 'Ours (SFT)' };
const COLS = ['MaRs-VQA', 'RAVEN (OOD)'];

const DATASETS = [
  ['kosmos-iq50', 'NeurIPS 2023', 'RAVEN-IQ test', 'cmp_raven_iq', '50', [0, 0, 1, 0, 0]],
  ['Visual Reasoning Benchmark', 'COLM 2024', 'Mensa test, RAVEN, IntelligenceTest', 'cmp_vrb', '241', [0, 0, 0, 0, 0]],
  ['MaRs-VQA (ours)', 'COLM 2025', 'MaRs-IB questionnaires', 'cmp_marsvqa', '1,440', [1, 1, 1, 1, 1]],
];
const PROMPTS = [
  ['GPT-4o + CoT', 33.96], ['+ 1-shot', 35.22], ['+ 3-shot', 36.10], ['+ 5-shot', 36.03],
  ['Multi-round CoT', 41.96], ['Multi-round + 1-shot', 42.08], ['GPT-o1 (reference)', 52.29, 'ref'],
];
const GENERAL = [
  ['MME', 1666, 1680, 0], ['HallusionBench', 53.83, 55.73, 2], ['POPE', 88.84, 88.71, 2],
  ['VQAv2 (val)', 79.88, 79.45, 2], ['ScienceQA-Image', 83.73, 83.29, 2], ['SEED-Bench', 69.16, 68.87, 2],
];

/* ------------------------------------------------------------- theming */
const root = document.documentElement;
$('#themeToggle').addEventListener('click', () => {
  const dark = root.dataset.theme ? root.dataset.theme === 'dark'
    : matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
  try { localStorage.setItem('vcog-theme', root.dataset.theme); } catch (e) {}
});

/* ---------------------------------------------------- top bar + nav */
const topbar = $('#topbar');
new IntersectionObserver(([e]) => topbar.classList.toggle('show', !e.isIntersecting), { rootMargin: '-56px 0px 0px 0px' })
  .observe($('.hero-links'));

const navLinks = [...$$('.topbar nav a'), ...$$('.sidenav a')];
const sections = ['top', 'takeaways', 'benchmark', 'method', 'results', 'analysis', 'examples', 'cite']
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

/* ------------------------------------------------------- try-it puzzle */
const lvtabs = $('#lvtabs'), lvinfo = $('#lvinfo'), opts = $('#pzopts'), msg = $('#pzmsg');
let cur = 0;
LEVELS.forEach((L, i) => {
  const b = document.createElement('button');
  b.className = 'dimtab lvtab'; b.setAttribute('role', 'tab'); b.style.setProperty('--c', `var(--l${L.n})`);
  b.innerHTML = `<span class="sw"></span>Level ${L.n}`;
  b.addEventListener('click', () => selectLevel(i));
  lvtabs.appendChild(b);
});
function selectLevel(i) {
  cur = i;
  const L = LEVELS[i];
  $$('.dimtab', lvtabs).forEach((b, j) => b.setAttribute('aria-selected', String(i === j)));
  lvinfo.style.setProperty('--c', `var(--l${L.n})`);
  lvinfo.innerHTML = `
    <h3><span class="sw"></span>Level ${L.n}</h3>
    <div class="theory">Changing in this example: ${L.attrs}</div>
    <p class="desc">${L.rule}</p>
    <div class="dim-perf">
      <div class="pp"><div class="v">${L.count}</div><div class="t">test items<br>at this level</div></div>
      <div class="pp ours"><div class="v">${fmt(L.gpt4o, 1)}</div><div class="t">GPT-4o + CoT<br>accuracy (%)</div></div>
      <div class="pp"><div class="v">25.0</div><div class="t">chance with<br>four options</div></div>
    </div>`;
  const m = $('#pzm');
  m.src = `static/images/web/puzzles/l${L.n}_m.webp`; m.alt = `Level ${L.n} matrix with the bottom-right cell missing`;
  opts.innerHTML = '';
  L.order.forEach((t, k) => {
    const b = document.createElement('button');
    b.className = 'opt'; b.dataset.t = t;
    b.setAttribute('aria-label', `Option ${'ABCD'[k]}`);
    b.innerHTML = `<span class="lt">${'ABCD'[k]}</span><img src="static/images/web/puzzles/l${L.n}_${t}.webp" alt="" width="128" height="128">`;
    b.addEventListener('click', () => answer(b));
    opts.appendChild(b);
  });
  msg.innerHTML = 'Pick an option. Each row and column follows the same rule.';
}
function answer(btn) {
  const L = LEVELS[cur], all = $$('.opt', opts);
  const right = all.find(b => b.dataset.t === 't1'), letter = 'ABCD'[all.indexOf(right)];
  all.forEach(b => { b.disabled = true; });
  right.classList.add('right');
  if (btn !== right) btn.classList.add('wrong');
  msg.innerHTML = `${btn === right ? '<span class="ok">Correct.</span>' : `<span class="no">Not quite.</span> The answer is <strong>${letter}</strong>.`}
    GPT-4o with chain-of-thought solves <strong>${fmt(L.gpt4o, 1)}%</strong> of the ${L.count} Level ${L.n} test items.
    <button class="btn again" type="button">${cur < LEVELS.length - 1 ? 'Next level' : 'Start over'}</button>`;
  $('.again', msg).addEventListener('click', () => selectLevel((cur + 1) % LEVELS.length));
}
selectLevel(0);

/* level chart */
(function levelChart() {
  const el = $('#levelchart'), W = 560, H = 270, m = { l: 40, r: 70, t: 24, b: 44 };
  el.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const Y = v => H - m.b - (v / 70) * (H - m.t - m.b);
  const g = svg('g', { class: 'grid' }, el);
  [0, 20, 40, 60].forEach(v => { svg('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); txt(el, { x: m.l - 8, y: Y(v) + 4, 'text-anchor': 'end', style: 'font-size:11px' }, v + '%'); });
  const bw = (W - m.l - m.r) / LEVELS.length;
  LEVELS.forEach((L, i) => {
    const x = m.l + i * bw + bw * .2, w = bw * .6;
    const r = svg('rect', { x, y: Y(L.gpt4o), width: w, height: Y(0) - Y(L.gpt4o), rx: 5, style: `fill:var(--l${L.n})` }, el);
    svg('title', {}, r).textContent = `Level ${L.n}: ${fmt(L.gpt4o)}% (${L.count} items)`;
    txt(el, { x: x + w / 2, y: Y(L.gpt4o) - 7, 'text-anchor': 'middle', class: 'val', style: 'fill:var(--ink);font-weight:600' }, fmt(L.gpt4o, 1));
    txt(el, { x: x + w / 2, y: H - m.b + 17, 'text-anchor': 'middle', class: 'lbl' }, `Level ${L.n}`);
    txt(el, { x: x + w / 2, y: H - m.b + 32, 'text-anchor': 'middle', style: 'font-size:10.5px;fill:var(--ink-44)' }, `${L.count} items`);
  });
  svg('line', { x1: m.l, x2: W - m.r, y1: Y(25), y2: Y(25), style: 'stroke:var(--ink-44);stroke-width:1.5;stroke-dasharray:5 4' }, el);
  txt(el, { x: W - m.r + 8, y: Y(25) + 4, style: 'font-size:10.5px;fill:var(--ink-60)' }, 'chance 25%');
})();

/* dataset comparison */
(function cmp() {
  const mark = v => v ? '<span class="y">✓</span>' : '<span class="n">✗</span>';
  $('#cmp').innerHTML = `<thead><tr><th class="l">Dataset</th><th>Example</th><th>Items</th><th>RGB images</th><th>Human study</th><th>Psychological validity</th><th>Open-source</th><th>VQA annotation</th></tr></thead>
    <tbody>${DATASETS.map(d => `<tr class="${d[0].includes('ours') ? 'ours' : ''}"><td class="l">${d[0]}<span class="src">${d[1]} · ${d[2]}</span></td>
      <td><img src="static/images/web/${d[3]}.webp" alt="" loading="lazy"></td><td>${d[4]}</td>${d[5].map(mark).map(x => `<td>${x}</td>`).join('')}</tr>`).join('')}</tbody>`;
})();

/* ---------------------------------------------------------- leaderboard */
const LOGO = { gpt: ['gpt.svg', true], claude: ['claude.svg'], qwen: ['qwen.svg'] };
const logoHTML = k => {
  if (k === 'ours') return '<img src="static/images/web/logo.svg" alt="">';
  if (k === 'human') return '<svg class="ph" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="background:none"><circle cx="12" cy="8" r="4"/><path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6" stroke-linecap="round"/></svg>';
  if (k === 'random') return '<svg class="ph" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="background:none"><rect x="4" y="4" width="16" height="16" rx="3"/><circle cx="9" cy="9" r="1.2" fill="currentColor"/><circle cx="15" cy="15" r="1.2" fill="currentColor"/></svg>';
  if (LOGO[k]) return `<img src="static/images/logos/${LOGO[k][0]}" alt=""${LOGO[k][1] ? ' class="mono"' : ''}>`;
  return `<span class="ph">${k}</span>`;
};
const isRef = m => m[1] === 'human' || m[1] === 'random';
const RANKED = MODELS.filter(m => !isRef(m));
const RANK = new Map(RANKED.slice().sort((a, b) => b[4] - a[4]).map((m, i) => [m[0], i + 1]));
const colRank = COLS.map((_, c) => {
  const vals = RANKED.map(m => m[4 + c]).sort((a, b) => b - a);
  return { best: vals[0], second: vals.find(v => v < vals[0]), baseline: Math.max(...RANKED.filter(m => m[1] !== 'ours').map(m => m[4 + c])) };
});
let sortCol = 0, sortDir = -1, group = 'all';
function drawLB() {
  const t = $('#lb');
  const head = `<thead><tr><th>#</th><th>Model</th><th>Type</th><th>Scale</th>${COLS.map((c, i) =>
    `<th class="sortable${i === 0 ? ' c-avg' : ''}${i === sortCol ? ' sorted' : ''}" data-c="${i}">${c}<span class="arrow">${sortDir < 0 ? '▼' : '▲'}</span></th>`).join('')}</tr></thead>`;
  const rows = RANKED.filter(m => group === 'all' || m[1] === group || m[1] === 'ours')
    .sort((a, b) => sortDir * (a[4 + sortCol] - b[4 + sortCol]) || (b[4] - a[4]));
  const row = m => {
    const ref = isRef(m), isO = m[1] === 'ours', rk = RANK.get(m[0]);
    const cells = COLS.map((_, c) => {
      const v = m[4 + c], cr = colRank[c];
      const cls = ref ? '' : v === cr.best ? ' best' : v === cr.second ? ' second' : '';
      const d = isO ? `<span class="delta">${signed(v - cr.baseline, 1)}</span>` : '';
      return `<td class="${c === 0 ? 'c-avg' : ''}"><span class="cell${cls}">${fmt(v)}${ref ? '' : `<span class="bar" style="width:${v}%"></span>`}</span>${d}</td>`;
    }).join('');
    return `<tr class="${ref ? 'ref' : isO ? 'ours' : ''}"><td>${ref ? '' : `<span class="rank${rk <= 3 ? ' r' + rk : ''}">${rk}</span>`}</td><td><span class="mname">${logoHTML(m[2])}${m[0]}${isO ? '<span class="tag">ours</span>' : ''}</span></td><td class="grp">${GROUP_LABEL[m[1]]}</td><td class="sc">${m[3]}</td>${cells}</tr>`;
  };
  const human = MODELS.find(m => m[1] === 'human'), random = MODELS.find(m => m[1] === 'random');
  t.innerHTML = head + '<tbody>' + row(human) + '<tr class="sep"><td colspan="6"></td></tr>' + rows.map(row).join('') +
    '<tr class="sep"><td colspan="6"></td></tr>' + row(random) + '</tbody>';
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
$('.lb-note').innerHTML = '<b>Bold</b>: best model. <u>Underline</u>: runner-up. Red numbers: Qwen2-VCog\'s margin over the best zero-shot model in that column. Human and random rows are references and are not ranked. Qwen2-VCog is fine-tuned on the MaRs-VQA train split only, so RAVEN is out-of-domain for it.';

/* ------------------------------------------------------- dumbbell chart */
(function gapChart() {
  const el = $('#gapchart');
  const rows = [MODELS.find(m => m[1] === 'human'), ...RANKED.slice().sort((a, b) => b[4] - a[4]), MODELS.find(m => m[1] === 'random')];
  const W = 640, L = 150, Rr = 104, top = 30, rowH = 32;
  const H = top + rows.length * rowH + 6;
  el.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const x = v => L + (v / 100) * (W - L - Rr);
  const g = svg('g', { class: 'grid' }, el);
  [0, 25, 50, 75, 100].forEach(v => { svg('line', { x1: x(v), x2: x(v), y1: top - 8, y2: H - 6 }, g); txt(el, { x: x(v), y: top - 14, 'text-anchor': 'middle' }, v); });
  txt(el, { x: W - 56, y: top - 14, 'text-anchor': 'end', style: 'font-size:10.5px;fill:var(--c2-ink);font-weight:600' }, 'MaRs');
  txt(el, { x: W - 4, y: top - 14, 'text-anchor': 'end', style: 'font-size:10.5px;fill:var(--amber);font-weight:600' }, 'RAVEN');
  rows.forEach((m, i) => {
    const y = top + i * rowH + rowH / 2, ref = isRef(m), ours = m[1] === 'ours';
    if (ours) svg('rect', { x: -4, y: y - rowH / 2 + 2, width: W + 8, height: rowH - 4, rx: 8, style: 'fill:var(--c1-soft)' }, el);
    txt(el, { x: 0, y: y + 4, class: 'lbl', style: `${ref ? 'font-style:italic;fill:var(--ink-60);' : ''}${ours ? 'font-weight:600' : ''}` }, m[0]);
    const a = m[4], b = m[5];
    svg('line', { x1: x(Math.min(a, b)), x2: x(Math.max(a, b)), y1: y, y2: y, class: 'stem', style: ours ? 'stroke:var(--c1);opacity:.55' : '' }, el);
    const dot = (v, fill, title) => {
      const c = svg('circle', { cx: x(v), cy: y, r: ours ? 6.5 : 5.5, style: ref && m[1] === 'human' ? `fill:none;stroke:${fill};stroke-width:2.5` : `fill:${fill};stroke:var(--paper);stroke-width:2` }, el);
      svg('title', {}, c).textContent = title;
    };
    dot(b, 'var(--amber)', `${m[0]} · RAVEN: ${fmt(b)}`);
    dot(a, 'var(--c2)', `${m[0]} · MaRs-VQA: ${fmt(a)}`);
    txt(el, { x: W - 56, y: y + 4, 'text-anchor': 'end', class: 'val', style: ours ? 'font-weight:700;fill:var(--c1-ink)' : '' }, fmt(a, 1));
    txt(el, { x: W - 4, y: y + 4, 'text-anchor': 'end', class: 'val', style: ours ? 'font-weight:700;fill:var(--c1-ink)' : '' }, fmt(b, 1));
  });
})();

/* -------------------------------------------------------- analysis charts */
function hbars(el, rows, d = 2) {
  el.innerHTML = rows.map(([n, v, cls]) => `
    <div class="hb ${cls || ''}"><span class="nm">${n}</span>
      <div class="ab-bar"><div class="f" style="width:${v / 0.6}%"></div><span class="t">${fmt(v, d)}</span></div></div>`).join('');
}
hbars($('#promptbars'), PROMPTS);

function groupedBars(el, groups, series, max, opts = {}) {
  const W = 320, H = 210, m = { l: 34, r: 8, t: 26, b: 40 };
  el.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const Y = v => H - m.b - (v / max) * (H - m.t - m.b);
  const g = svg('g', { class: 'grid' }, el);
  for (let v = 0; v <= max; v += max / 4) { svg('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); txt(el, { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', style: 'font-size:10.5px' }, v); }
  const gw = (W - m.l - m.r) / groups.length, bw = Math.min(30, gw * .7 / series.length);
  groups.forEach((gname, gi) => {
    const gx = m.l + gi * gw + gw / 2 - (bw * series.length) / 2;
    series.forEach((s, si) => {
      const v = s.vals[gi], x = gx + si * bw;
      const r = svg('rect', { x: x + 1.5, y: Y(v), width: bw - 3, height: Y(0) - Y(v), rx: 4, style: `fill:${s.color}` }, el);
      svg('title', {}, r).textContent = `${s.name} · ${gname}: ${fmt(v)}`;
      txt(el, { x: x + bw / 2, y: Y(v) - 5, 'text-anchor': 'middle', style: 'font-size:10px;fill:var(--ink);font-weight:600' }, fmt(v, 1));
    });
    gname.split('\n').forEach((ln, li) => txt(el, { x: m.l + gi * gw + gw / 2, y: H - m.b + 15 + li * 12, 'text-anchor': 'middle', style: 'font-size:10.5px' }, ln));
  });
  if (opts.chance != null)
    svg('line', { x1: m.l, x2: W - m.r, y1: Y(opts.chance), y2: Y(opts.chance), style: 'stroke:var(--ink-44);stroke-width:1.2;stroke-dasharray:4 3' }, el);
  const lg = svg('g', {}, el);
  let lx = m.l;
  series.forEach(s => {
    svg('rect', { x: lx, y: 6, width: 9, height: 9, rx: 2, style: `fill:${s.color}` }, lg);
    txt(lg, { x: lx + 13, y: 14, style: 'font-size:10.5px;fill:var(--ink-80)' }, s.name);
    lx += 13 + s.name.length * 5.6 + 14;
  });
  if (opts.chance != null) {
    svg('line', { x1: lx, x2: lx + 14, y1: 10.5, y2: 10.5, style: 'stroke:var(--ink-44);stroke-width:1.2;stroke-dasharray:4 3' }, lg);
    txt(lg, { x: lx + 18, y: 14, style: 'font-size:10.5px;fill:var(--ink-80)' }, opts.chanceLabel);
  }
}
groupedBars($('#ch-reason'), ['MaRs-VQA', 'RAVEN (OOD)'], [
  { name: 'with reasoning', color: 'var(--c1)', vals: [72.71, 31.96] },
  { name: 'answers only', color: 'var(--ink-20)', vals: [54.82, 29.30] },
], 80);
groupedBars($('#ch-vl'), ['question\nonly', '+ option\nimages', '+ option\ndescriptions'], [
  { name: 'GPT-4o', color: 'var(--amber)', vals: [24.58, 33.96, 36.46] },
  { name: 'Qwen2-VCog', color: 'var(--c1)', vals: [26.32, 72.71, 71.43] },
], 80, { chance: 25, chanceLabel: 'chance (25%)' });

(function gen() {
  $('#gentable').innerHTML = `<thead><tr><th>Benchmark</th><th>Qwen2-VL-7B</th><th>Qwen2-VCog-7B</th><th>Δ</th></tr></thead><tbody>${
    GENERAL.map(([n, a, b, d]) => `<tr><td>${n}</td><td>${fmt(a, d)}</td><td>${fmt(b, d)}</td><td class="d ${b > a ? 'up' : 'dn'}">${signed(b - a, d)}</td></tr>`).join('')}</tbody>`;
})();

/* ---------------------------------------------------------------- bibtex */
$('#copybib').addEventListener('click', async e => {
  const b = e.currentTarget;
  try { await navigator.clipboard.writeText($('#bibtex').textContent); b.textContent = 'Copied'; b.classList.add('ok'); }
  catch (err) { b.textContent = 'Select & copy'; }
  setTimeout(() => { b.textContent = 'Copy'; b.classList.remove('ok'); }, 1800);
});
})();
