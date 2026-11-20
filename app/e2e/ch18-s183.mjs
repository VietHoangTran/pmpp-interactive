// Browser check for chapter 18, section 18.3 "Vertices, edges or a frontier": the five
// listings (as printed and corrected, the changed lines marked), the two Exercise 1c
// predictions and their jump into the explorer, the explorer itself (Exercise 1 graph, pull,
// call 2; run to the end; the chain and the star), the "Two threads, one neighbour" stepper
// (the hoisted expected value misses vertex 4; every order for each visit form), the HIP and
// Metal tabs, the narrated wavefront scene, and two questions. Every number it expects comes
// from src/sim/bfs.ts, src/sim/bfsvisit.ts or labs/ch18/snippets/s183_variants.cu.
// Passes with or without content/ch18/narration/manifest.json.
// Usage: node e2e/ch18-s183.mjs [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto, BASE } from './browser.mjs';
import { EX1, MAZE, MAZE_GOAL, MAZE_ROOT, PRESETS, backtrace, depth, levels, parentChoices, runBFS } from '../src/sim/bfs.ts';
import { FORMS, SCENARIOS, changedLines, enumerate, region, runOrder } from '../src/sim/bfsvisit.ts';

const here = path.dirname(fileURLToPath(import.meta.url));
const SRC = fs.readFileSync(path.join(here, '../../labs/ch18/snippets/s183_variants.cu'), 'utf8');
const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const bench = (title) => `section.bench[aria-label="${title}"]`;
const X = bench('One graph, five ways to run a level');
const V = bench('Two threads, one neighbour');
const S = 'section.scene[aria-label="The wavefront"]';
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const pressIn = async (scope, label, exact = false) => {
  const ok = await page.evaluate((scope, label, exact) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => (exact ? x.textContent.trim() === label : x.textContent.trim().startsWith(label)));
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label, exact);
  await pause(150);
  return ok;
};
const clickSvg = async (sel) => {
  const ok = await page.evaluate((sel) => {
    const el = document.querySelector(sel);
    if (!el) return false;
    el.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    return true;
  }, sel);
  await pause(150);
  return ok;
};
const choose = async (scope, label, value) => {
  await page.select(`${scope} select[aria-label="${label}"]`, String(value));
  await pause(150);
};
const question = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);

await goto(page, 'ch18/18.3');
await page.waitForSelector(X);

// ---------------------------------------------------------------- the five listings
const listing = 'figure.code';
const figureOf = async () => page.evaluateHandle(() => [...document.querySelectorAll('figure.code')].find((f) => f.textContent.includes('from labs/ch18/snippets/s183_variants.cu') && /Fig\. 18\.\d+ \w+, (as printed|corrected)/.test(f.textContent)));
await page.waitForFunction(() => [...document.querySelectorAll('figure.code')].some((f) => f.textContent.includes('s183_variants.cu')));
const FIGS = [['fig18_6', '18.6'], ['fig18_8', '18.8'], ['fig18_10', '18.10'], ['fig18_12', '18.12'], ['fig18_14', '18.14']];
const scopeL = 'div.wide:has(> .row [aria-label="Listing version"])';
let listingsOk = true;
let detail = '';
for (const [id, label] of FIGS) {
  for (const fixed of [false, true]) {
    await pressIn(scopeL, fixed ? 'corrected' : 'as printed', true);
    await pressIn(scopeL, label, true);
    const fig = await figureOf();
    const got = await fig.evaluate((f) => ({
      lines: [...f.querySelectorAll('span.ln')].map((s) => s.textContent.replace(/\n$/, '')),
      hl: [...f.querySelectorAll('span.ln.hl')].map((s) => Number(s.dataset.n)),
    }));
    const printed = region(SRC, id + '_printed');
    const corrected = region(SRC, id + '_fixed');
    const want = (fixed ? corrected : printed).split('\n');
    const wantHl = fixed ? changedLines(printed, corrected) : [];
    if (!same(got.lines.map((l) => l.trimEnd()), want.map((l) => l.trimEnd())) || !same(got.hl, wantHl)) {
      listingsOk = false;
      detail += `${id} ${fixed ? 'fixed' : 'printed'}: ${got.lines.length} lines, hl ${got.hl} vs ${wantHl}; `;
    }
  }
}
check('each of the five listings, as printed and corrected, is its region of the snippet file, with the changed lines marked', listingsOk, detail);
await pressIn(scopeL, 'corrected', true);
await pressIn(scopeL, '18.6', true);
const hl186 = await (await figureOf()).evaluate((f) => [...f.querySelectorAll('span.ln.hl')].map((s) => Number(s.dataset.n)));
check('corrected Fig. 18.6 marks lines 05, 09, 10 and 11', same(hl186, [5, 9, 10, 11]), JSON.stringify(hl186));
const cap = await (await figureOf()).evaluate((f) => f.querySelector('figcaption').textContent);
check('its caption says what changed', cap.includes('relaxed atomic loads and stores'), cap);
if (out) {
  await (await figureOf()).evaluate((f) => f.scrollIntoView({ block: 'center' }));
  await page.screenshot({ path: out + '-listing.png' });
}
await pressIn(scopeL, 'as printed', true);

// ---------------------------------------------------------------- the explorer, by hand
const pull = runBFS(EX1, { variant: 'pull', root: 0, blockSize: 4, warpWidth: 4 });
await choose(X, 'Graph', 1);
await choose(X, 'Root', 0);
await choose(X, 'Kernel', 'pull');
await pressIn(X, 'Step', true);
await pressIn(X, 'Step', true);
const summary = () => page.$eval(`${X} [data-call-summary]`, (e) => ({ ...e.dataset }));
let sm = await summary();
check('Exercise 1 graph, root 0, pull, call 2: 5 threads loop and 3 label', (await page.$eval(`${X} [data-call]`, (e) => e.dataset.call)) === '2' && sm.looping === '5' && sm.labelling === '3', JSON.stringify(sm));
const roles = await page.$$eval(`${X} [data-t]`, (s) => s.map((x) => x.dataset.role));
check('the thread cells carry the model\'s roles', same(roles, pull.calls[1].threads.map((t) => t.role)), roles.join(','));
const loopers = await page.$$eval(`${X} [data-t][data-role="labels"], ${X} [data-t][data-role="iterates"]`, (s) => s.map((x) => x.firstChild.textContent.trim()));
check('the five that loop are vertices 1, 3, 4, 6 and 7', same(loopers, ['v1', 'v3', 'v4', 'v6', 'v7']), loopers.join(','));
const row2 = await page.$$eval(`${X} tr[data-row="2"] td`, (t) => t.map((x) => x.textContent.trim()));
check('the table row for call 2: 8 launched, 5 past the check, 3 labels, frontier 1, 3, 7', row2[2] === '8' && row2[3] === '5' && row2[5] === '3' && row2[6] === '1, 3, 7', row2.join(' | '));
// Click the thread on vertex 4: it scans in-edges and labels nothing.
await page.$eval(`${X} [data-t="4"]`, (e) => e.click());
await pause(120);
let narr = await text(`${X} [data-thread-text]`);
check('the thread on 4 scans its in-edges, finds no parent at level 1, labels nothing', narr.includes('owns vertex 4') && narr.includes('No parent at level 1'), narr);
if (out) {
  await page.$eval(X, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-explorer-pull.png' });
}

// Run to the end: the levels are the model's, and the bound is the book's O(...) counted with the empty call.
await pressIn(X, 'Run to end', true);
const tags = await page.$$eval(`${X} svg [data-vertex]`, (vs) => vs.map((v) => v.querySelector('[data-tag]')?.textContent ?? '·'));
const show = (lv) => lv.map((L) => (L === 4294967295 ? '·' : String(L)));
check('run to the end: the levels match the model on the Exercise 1 graph', same(tags, show(levels(EX1, 0))), tags.join(','));
const workPull = await text(`${X} [data-work]`);
check('the end-of-run sentence gives the p. 439 bound in O form and our (d + 1) count', workPull.includes('The p. 439 bound O(d·n + d·m), counted here with the final empty call as (d + 1)·(n + m)'), workPull);
check('the explorer has no block-size, warp-width or edit-edges controls (condensed tier)',
  !(await page.$(`${X} [aria-label="Block size"]`)) && !(await page.$(`${X} [aria-label="Warp width"]`)) && !(await text(X)).includes('edit edges'));

// The chain: push against frontier, whole search.
const chainRun = (variant) => runBFS(PRESETS[2], { variant, root: 0, blockSize: 4, warpWidth: 4 });
await choose(X, 'Graph', 2);
await choose(X, 'Kernel', 'push');
await pressIn(X, 'Run to end', true);
let tot = await page.$$eval(`${X} tr[data-totals] td`, (t) => t.map((x) => x.textContent.trim()));
let work = await text(`${X} [data-work]`);
const cp = chainRun('push');
check('the chain, push: launched total and work as the model counts them', tot[1] === String(cp.totals.launched) && work.includes(`${cp.totals.work} units of work`), tot.join('|') + ' ' + work);
await choose(X, 'Kernel', 'frontier');
await pressIn(X, 'Run to end', true);
tot = await page.$$eval(`${X} tr[data-totals] td`, (t) => t.map((x) => x.textContent.trim()));
work = await text(`${X} [data-work]`);
const cf = chainRun('frontier');
check('the chain, frontier: work is n + m = 28, far below push', work.includes(`${cf.totals.work} units of work`) && cf.totals.work === 10 + 18 && cf.totals.work < cp.totals.work, work);

// The star from a leaf, push, call 2: the hub's thread loops alone in its warp.
const starRun = runBFS(PRESETS[3], { variant: 'push', root: 1, blockSize: 4, warpWidth: 4 });
await choose(X, 'Graph', 3);
await choose(X, 'Kernel', 'push');
await pressIn(X, 'Step', true);
await pressIn(X, 'Step', true);
const warp0 = await page.$eval(`${X} [data-warp="0"]`, (e) => e.textContent.trim());
check('the star, push, call 2: warp 0 makes 10 passes for 10 useful lane slots of 40', warp0 === `${starRun.calls[1].warps[0].passes} passes, 10 of 40 lane slots used`, warp0);
if (out) await page.screenshot({ path: out + '-explorer-star.png' });

// ---------------------------------------------------------------- the two predictions
for (const [needle, value, variant, call, key, want] of [
  ['How many threads loop over their incoming edges', 4, 'pull', 2, 'looping', '5'],
  ['How many threads find their edge\'s source at level 2', 8, 'edge', 3, 'past', '8'],
]) {
  const pq = await question(needle);
  if (await pq.evaluate((e) => !!e.querySelector('input'))) {
    await (await pq.$('input')).type(String(value));
    await pq.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.includes('Commit')).click());
    await pause(200);
    const reveal = await pq.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
    const wantReveal = variant === 'pull' ? 'It is 5: the unvisited vertices 1, 3, 4, 6 and 7' : 'Right: 8: the edges out of 1, 3 and 7';
    check(`the ${variant} reveal answers the committed ${value}`, reveal.includes(wantReveal), reveal);
  }
  await pq.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.includes('Show it in the explorer')).click());
  await pause(500);
  sm = await summary();
  const at = await page.$eval(`${X} [data-call]`, (e) => e.dataset.call);
  const kernel = await page.$eval(`${X} select[aria-label="Kernel"]`, (e) => e.value);
  const graph = await page.$eval(`${X} select[aria-label="Graph"]`, (e) => e.value);
  check(`the ${variant} reveal opens the explorer on the Exercise 1 graph at call ${call}, showing ${want}`, graph === '1' && kernel === variant && at === String(call) && sm[key] === want, JSON.stringify({ graph, kernel, at, sm }));
}
check('edge-centric call 3: four of the eight may label', sm.labelling === '4', JSON.stringify(sm));

// ---------------------------------------------------------------- the stepper
const final = () => page.$eval(`${V} [data-final]`, (e) => e.dataset.final).catch(() => null);
const levelCells = () => page.$$eval(`${V} [data-levels] [data-v]`, (s) => Object.fromEntries(s.map((x) => [x.dataset.v, x.firstChild.textContent])));
await choose(V, 'Scenario', 'trap');
await choose(V, 'Visit code', 'hoisted');
await pressIn(V, 'Run forward order', true);
let lc = await levelCells();
check('expected hoisted, trap, forward order: the outcome is "missed", and level[4] stays UNVISITED', (await final()) === 'missed' && lc['4'] === '·' && lc['3'] === '2', JSON.stringify(lc));
const hf = runOrder(FORMS.hoisted, SCENARIOS.trap, 'forward', 12);
check('the stepper ends on the model\'s last event', (await text(`${V} [data-narrate]`)).startsWith(hf.events.at(-1).text.slice(0, 40)), await text(`${V} [data-narrate]`));
if (out) {
  await page.$eval(V, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-stepper-hoisted.png' });
}
await pressIn(V, 'Run reverse order', true);
check('the same code in reverse order is correct', (await final()) === 'correct');

// Every order, for each form, against the model's enumeration.
const every = () => page.$eval(`${V} [data-every]`, (e) => e.dataset.every);
let allOk = true;
detail = '';
for (const [form, sp] of [['plain', 0], ['plain-append', 0], ['cas', 0], ['hoisted', 0], ['weak', 1], ['weak-loop', 1], ['weak-reset', 0], ['weak-correct', 2]]) {
  await choose(V, 'Visit code', form);
  if (FORMS[form].weak) await page.evaluate((V, sp) => [...document.querySelectorAll(V + ' [aria-label="Spurious failures allowed"] button')].find((b) => b.textContent.trim() === String(sp)).click(), V, sp);
  await pause(100);
  await pressIn(V, 'Try every order', true);
  const got = await every();
  const want = enumerate(FORMS[form], SCENARIOS.trap, { spurious: FORMS[form].weak ? sp : 0 }).outcomes.map((o) => o.kind).join(',');
  if (got !== want) {
    allOk = false;
    detail += `${form}: ${got} vs ${want}; `;
  }
}
check('"Try every order" lists the model\'s outcomes for all eight visit forms', allOk, detail);
// At 1440 px every form's listing shows whole: no line is cut at the right edge of the code box.
let cut = '';
for (const form of Object.keys(FORMS)) {
  await choose(V, 'Visit code', form);
  const fit = await page.$eval(`${V} [data-visit-code] pre`, (p) => ({ sw: p.scrollWidth, cw: p.clientWidth }));
  if (fit.sw > fit.cw + 1) cut += `${form}: ${fit.sw} > ${fit.cw}; `;
}
check('the stepper\'s code box is as wide as its longest line, for every visit form', cut === '', cut);
await choose(V, 'Visit code', 'cas');
await pressIn(V, 'Try every order', true);
check('compare-and-swap: every order is correct', (await every()) === 'correct' && (await text(`${V} [data-every]`)).startsWith('Every order is correct'));
await choose(V, 'Visit code', 'plain-append');
await pressIn(V, 'Try every order', true);
await page.evaluate((V) => [...document.querySelectorAll(V + ' li')].find((l) => l.querySelector('[data-outcome="duplicate-entry"]')).querySelector('button').click(), V);
await pause(150);
const fr = await page.$eval(`${V} [data-frontier]`, (e) => e.dataset.frontier);
check('plain with an append: the loaded order puts 3 in the frontier twice', (await final()) === 'duplicate-entry' && fr.split(',').filter((x) => x === '3').length === 2, fr);
await choose(V, 'Visit code', 'weak-loop');
await pressIn(V, 'Run forward order', true);
check('while (!cas(e)) {}: forward order appends 3 twice', (await final()) === 'duplicate-entry' && (await page.$eval(`${V} [data-frontier]`, (e) => e.dataset.frontier)) === runOrder(FORMS['weak-loop'], SCENARIOS.trap, 'forward').states.at(-1).frontier.join(','));
await choose(V, 'Visit code', 'weak-reset');
await pressIn(V, 'Run forward order', true);
check('while (!cas(e)) e = UNVISITED: forward order never terminates', (await final()) === 'never-terminates');
// A spurious failure by hand: the weak form with no loop leaves 3 unvisited.
await choose(V, 'Visit code', 'weak');
await page.$eval(`${V} tr[data-thread="0"] button`, (b) => b.click()); // take neighbour 3
await pause(100);
await page.evaluate((V) => [...document.querySelectorAll(V + ' tr[data-thread="0"] button')].find((b) => b.textContent.includes('fail spuriously')).click(), V);
await pause(120);
narr = await text(`${V} [data-narrate]`);
check('a spurious failure can be injected by hand, and e stays UNVISITED', narr.includes('fails spuriously') && (await page.$eval(`${V} tr[data-thread="0"] td.t-reg`, (e) => e.textContent)) === 'UNVISITED', narr);
// Switch to the book's scenario and check the graph and level cells.
await choose(V, 'Scenario', 'book');
await choose(V, 'Visit code', 'cas');
await pressIn(V, 'Run forward order', true);
lc = await levelCells();
check("the book's scenario: compare-and-swap labels 8 at level 3 once", (await final()) === 'correct' && lc['8'] === '3', JSON.stringify(lc));

// ---------------------------------------------------------------- the HIP and Metal tabs
const backends = await page.evaluateHandle(() => [...document.querySelectorAll('figure.code')].find((f) => [...f.querySelectorAll('[role=tab]')].some((t) => t.textContent === 'Metal')));
const cudaTab = await backends.evaluate((f) => f.querySelector('pre').textContent);
check('the CUDA and HIP tab is the corrected Fig. 18.14 with an atomicCAS branch for HIP', cudaTab.includes('compare_exchange_strong') && cudaTab.includes('atomicCAS') && cudaTab.includes('__HIPCC__'), cudaTab.slice(0, 200));
await backends.evaluate((f) => [...f.querySelectorAll('[role=tab]')].find((t) => t.textContent === 'Metal').click());
await pause(150);
const metalTab = await backends.evaluate((f) => f.querySelector('pre').textContent);
check('the Metal tab is the lab\'s weak compare-and-swap retry loop', metalTab.includes('atomic_compare_exchange_weak_explicit') && metalTab.includes('expected != UNVISITED'), metalTab.slice(0, 200));
// Its status comes from this machine's reports: the provided frontier test of lab 18.1 on Metal.
const reportsNow = await (await fetch(`${BASE}/api/reports`)).json().catch(() => ({}));
const frontierTests = (reportsNow.metal?.results ?? []).filter((r) => r.lab === 'ch18/lab1_frontier' && r.built)
  .flatMap((r) => r.tests.filter((t) => t.case.startsWith('provided frontier')));
const wantStatus = !frontierTests.length ? 'unverified (no run' : frontierTests.every((t) => t.pass) ? 'verified on Metal' : 'unverified (failed';
const metalStatus = await backends.evaluate((f) => f.querySelector('.status')?.textContent ?? '');
check('the Metal tab\'s status is derived from the reports, in the fixed vocabulary', metalStatus.startsWith(wantStatus), metalStatus);
const colW = await page.$$eval('table[data-backends] thead th', (ths) => ths.map((t) => Math.round(t.getBoundingClientRect().width)));
const tblFit = await page.$eval('table[data-backends]', (t) => t.scrollWidth <= t.clientWidth + 1);
check('the HIP and Metal table gives every column a readable width and does not overflow', colW.length === 4 && colW.every((w) => w >= 140) && tblFit, JSON.stringify(colW));

// ---------------------------------------------------------------- the wavefront scene
const LV = levels(MAZE, MAZE_ROOT);
const D = depth(LV);
const SIZES = Array.from({ length: D + 1 }, (_, L) => LV.filter((x) => x === L).length);
const PUSH = runBFS(MAZE, { variant: 'push', root: MAZE_ROOT, blockSize: 1, warpWidth: 1 });
const FRONT = runBFS(MAZE, { variant: 'frontier', root: MAZE_ROOT, blockSize: 1, warpWidth: 1 });
const PATH = backtrace(MAZE, MAZE_ROOT, MAZE_GOAL);
const CH = parentChoices(MAZE, MAZE_ROOT);
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const scripts = [];
for (let n = 1; n <= 5; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch18/narration/wavefront-${n}.txt`)).text()));
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);
const seek = async (v) => {
  await page.$eval(S + ' input[type=range]', (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(150);
};
const stage = () => page.$eval(S + ' g[data-beat]', (g) => ({ ...g.dataset }));
const caption = () => page.$eval(S + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim()).catch(() => null);
check('the scene has five parts and a captions option', (await page.$$(S + ' .seg button')).length === 5 && (await text(S)).includes('captions'));
const hasManifest = await (await fetch(`${BASE}/content/ch18/narration/manifest.json`)).json().then((m) => !!m?.clips?.['wavefront-1']).catch(() => false);
if (!hasManifest) check('with no audio yet, part 1 shows its script as a caption', (await caption()) === scripts[0], String(await caption()));
// Beat by beat, at the end and at the midpoint, against the model.
const LABELS = ['Blocks and wires', 'A thread per cell', 'A thread per frontier cell', 'A launch per level', 'Walking back'];
const callOf = (p) => 1 + Math.min(D, Math.floor(Math.max(0, Math.min(1, (p - 0.04) / 0.9)) * (D + 1)));
const sum = (a) => a.reduce((s, x) => s + x, 0);
const want = [
  { level: String(D), wave: String(SIZES[D]) },
  { call: String(D + 1), launched: String(MAZE.n), total: String(PUSH.totals.items), working: String(SIZES[D]) },
  { call: String(D + 1), 'threads-total': String(FRONT.totals.items), 'edges-total': String(MAZE.edges.length), work: String(FRONT.totals.work), 'push-work': String(PUSH.totals.work) },
  { launches: String(D + 1), largest: String(Math.max(...SIZES)) },
  { shown: String(PATH.length), 'path-length': String(PATH.length), ties: String(PATH.filter((v) => CH[v] > 1).length), target: String(LV[MAZE_GOAL]) },
];
let sceneOk = true;
detail = '';
for (let k = 0; k < 5; k++) {
  await pressIn(S, `${k + 1}. ${LABELS[k]}`, true);
  if (!hasManifest && (await caption()) !== scripts[k]) {
    sceneOk = false;
    detail += `caption ${k + 1}; `;
  }
  await seek(1000);
  const st = await stage();
  for (const [key, v] of Object.entries(want[k])) if (st[key.replace(/-([a-z])/g, (_, c) => c.toUpperCase())] !== v) {
    sceneOk = false;
    detail += `part ${k + 1} ${key}: ${JSON.stringify(st)} vs ${v}; `;
  }
  if (out && (k === 1 || k === 4)) await page.screenshot({ path: `${out}-scene-${k + 1}.png` });
}
check('each part\'s end state matches the model: 30 levels, 6107 push threads, 869 against 6779, 31 launches, a 27-block wire with 9 ties', sceneOk, detail);
await pressIn(S, '2. A thread per cell', true);
await seek(500);
const mid = await stage();
const c = callOf(0.5);
check('halfway through part 2 the working threads are the model\'s wavefront for that call', mid.call === String(c) && mid.working === String(PUSH.calls[c - 1].pastCheck) && mid.total === String(sum(PUSH.calls.slice(0, c).map((x) => x.items))), JSON.stringify(mid));
const dots = await page.$$eval(S + ' circle[data-dot="work"]', (d) => d.length);
check('and draws one dark dot per working thread', dots === PUSH.calls[c - 1].pastCheck, `${dots}`);
await pressIn(S, '1. Blocks and wires', true);
await pressIn(S, 'Play', true);
await pause(1500);
const moved = await page.$eval(S + ' input[type=range]', (e) => Number(e.value));
check('it plays on its own clock', moved > 0, String(moved));
await pressIn(S, 'Pause', true);

// ---------------------------------------------------------------- questions
const qh = await question('never in the last level during a search whose host loop makes 10 kernel');
await (await qh.$('input[type=number]')).type('10000');
await qh.evaluate((e) => e.querySelector('button.primary').click());
await pause(150);
check('the hub question accepts 10,000 reads', (await qh.evaluate((e) => e.textContent)).includes('Correct.'));
const qs = await question('once, above the edge loop');
await qs.evaluate((e) => e.querySelectorAll('input[type=radio]')[0].click());
await qs.evaluate((e) => e.querySelector('button.primary').click());
await pause(150);
check('the stale-expected bug hunt accepts "4 is missed"', (await qs.evaluate((e) => e.textContent)).includes('Correct.'));

// ---------------------------------------------------------------- light, dark, narrow
for (const theme of ['light', 'dark']) {
  await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, theme);
  await page.$eval(X, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(200);
  if (out) await page.screenshot({ path: `${out}-explorer-${theme}.png` });
  await page.$eval(V, (e) => e.scrollIntoView({ block: 'start' }));
  if (out) await page.screenshot({ path: `${out}-stepper-${theme}.png` });
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  if (out) await page.screenshot({ path: `${out}-scene-${theme}.png` });
}
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(400);
const narrow = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
check('at a narrow width the page does not scroll sideways', narrow);
if (out) {
  for (const [sel, name] of [[X, 'explorer'], [V, 'stepper'], [S, 'scene']]) {
    await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
    await page.screenshot({ path: `${out}-narrow-${name}.png` });
  }
}

const expected404 = (u) => /\/content\/ch18\/narration\/(manifest\.json|wavefront-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
