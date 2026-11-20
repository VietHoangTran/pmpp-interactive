// Browser check for section 22.2: the explorer "Who writes, who conflicts" (every preset, both
// cuts, a selected thread's mapping, the GEMV sector counts), the classification "Sort the
// book's tour" (several cards, both traps), the stepper "Split-K, warp by warp" (to the first
// lost update with plain +=, and to a correct end with atomics), the two kernels, and questions.
// Every expected number comes from src/sim/decompose.ts and src/sim/coalesce.ts.
// Usage: node e2e/ch22-s222.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { warpAccess } from '../src/sim/coalesce.ts';
import {
  PRESETS, SPLITK_TINY, TOUR, decomposeState, describeSKStep, describeWorker, matvec, splitKOrder, splitKRun, warp0Load,
} from '../src/sim/decompose.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (t) => `section.bench[aria-label="${t}"]`;
const E = bench('Who writes, who conflicts');
const T = bench('Sort the book\'s tour');
const K = bench('Split-K, warp by warp');
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
/** Click the button labelled `label` inside the group `group` (aria-label) of bench `scope`. */
const press = async (scope, group, label) => {
  const ok = await page.evaluate((scope, group, label) => {
    const root = document.querySelector(scope);
    const g = group ? root.querySelector(`[aria-label="${group}"]`) : root;
    const b = [...g.querySelectorAll('button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, group, label);
  await pause(80);
  return ok;
};
const select = async (scope, label, value) => {
  await page.evaluate((scope, label, value) => {
    const l = [...document.querySelectorAll(scope + ' label.field')].find((x) => x.textContent.trim().startsWith(label));
    const s = l.querySelector('select');
    s.value = String(value);
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, scope, label, value);
  await pause(100);
};
const metrics = () => page.$$eval(E + ' [data-metric]', (cs) => Object.fromEntries(cs.map((c) => [c.dataset.metric, c.textContent.trim()])));
const writers = () => page.$eval(E + ' [data-row="out"]', (e) => e.dataset.writers);
const fmt = (v) => v.toLocaleString('en-US');

await goto(page, 'ch22/22.2');
await page.waitForSelector(E);

// ---------------------------------------------------------------- explorer: the four presets
for (const p of PRESETS) {
  check(`preset "${p.label}" is offered`, await press(E, 'Presets', p.label));
  const d = decomposeState(p.state);
  const m = await metrics();
  check(`preset "${p.label}": threads, atomics and writers as the model`,
    m.threads === fmt(d.metrics.threads) && m.atomics === fmt(d.metrics.atomicUpdates) && m.worst === fmt(d.metrics.worstWriters)
    && m['outputs-atomic'] === `${fmt(d.metrics.outputsNeedingAtomics)} of ${fmt(d.problem.nOut)}` && (await writers()) === d.writers.join(' '),
    JSON.stringify({ m, want: d.metrics }));
  if (out) await page.screenshot({ path: `${out}-preset-${p.id}.png` });
}
// Spell out the numbers the lesson's questions rely on.
await press(E, 'Presets', 'Convolution, scattered');
let m = await metrics();
check('convolution scattered: 12 of 12 outputs need atomics, 54 updates, 5 writers at most', m['outputs-atomic'] === '12 of 12' && m.atomics === '54' && m.worst === '5', JSON.stringify(m));
await press(E, 'Decomposition', 'output-centric (gather)');
m = await metrics();
check('convolution gathered: no atomics, one writer each, 42 additions in registers', m.atomics === '0' && m.worst === '1' && m.registers === '42' && (await writers()) === Array(12).fill(1).join(' '), JSON.stringify(m));
await press(E, 'Presets', 'Histogram, a thread per bin');
m = await metrics();
check('histogram, a thread per bin: 4 threads, each inspects all 16 inputs, 64 in all, no atomics', m.threads === '4' && m['inspected-max'] === '16' && m.inspected === '64' && m.atomics === '0', JSON.stringify(m));
// Click bin 0's thread: its mapping sentence is the model's.
await page.$eval(E + ' [data-worker="0"]', (e) => e.dispatchEvent(new MouseEvent('click', { bubbles: true })));
await pause(100);
const hist0 = decomposeState(PRESETS[1].state);
check('clicking a thread prints its index mapping', (await text(E + ' [data-mapping-text]')) === describeWorker(hist0, 0), await text(E + ' [data-mapping-text]'));
await press(E, 'Decomposition', 'input-centric: a thread per input');
m = await metrics();
check('histogram, a thread per input: 16 threads, 1 input each, atomics on every bin', m.threads === '16' && m.inspected === '16' && m['outputs-atomic'] === '4 of 4', JSON.stringify(m));
await press(E, 'Presets', 'Skewed sparse matrix, COO');
m = await metrics();
check('COO on the skewed matrix: balanced, 20 atomic updates', m.imbalance === '1' && m.atomics === '20', JSON.stringify(m));
await press(E, 'Decomposition', 'output-centric: a thread per row (CSR)');
m = await metrics();
check('CSR on the same matrix: no atomics, the busiest thread does 3 times the mean', m.atomics === '0' && m.imbalance === '3', JSON.stringify(m));

// ---------------------------------------------------------------- explorer: decode GEMV sectors
await press(E, 'Presets', 'Narrow decode GEMV, split-K');
const ww = 32;
const sectors = (N, K, mapping, S = 4) => warpAccess(warp0Load(matvec(N, K), mapping, { warpWidth: ww, S }), 4, 32).transactions;
m = await metrics();
check('narrow GEMV split-K: 256 threads, 8 atomic updates on 2 outputs, 4 sectors in one run',
  m.threads === '256' && m.atomics === '8' && m['outputs-atomic'] === '2 of 2' && m.sectors === `${sectors(2, 256, 'splitk')}, one run`, JSON.stringify(m));
await press(E, 'Decomposition', 'a thread per output');
m = await metrics();
check('narrow GEMV, a thread per output: 2 threads, no atomics', m.threads === '2' && m.atomics === '0' && m.sectors.startsWith(String(sectors(2, 256, 'output'))), JSON.stringify(m));
await select(E, 'Outputs N', 32);
m = await metrics();
check('32 outputs, a thread per output: warp 0\'s first load touches 32 sectors', m.sectors === String(sectors(32, 256, 'output')) && m.sectors === '32', JSON.stringify(m));
if (out) await page.screenshot({ path: `${out}-gemv-thread.png` });
await press(E, 'Decomposition', 'a warp per output');
m = await metrics();
check('32 outputs, a warp per output: 4 sectors in one contiguous run', m.sectors === `${sectors(32, 256, 'warp')}, one run` && m.sectors === '4, one run' && m.threads === '1,024', JSON.stringify(m));
await select(E, 'Warp width', 'amd');
m = await metrics();
check('a 64-wide warp per output: 8 sectors, 2,048 threads', m.sectors === '8, one run' && m.threads === '2,048', JSON.stringify(m));
await select(E, 'Warp width', 'nv');
await press(E, 'Decomposition', 'split the input dimension: S warps per output');
await select(E, 'Chunks S', 8);
const sk = decomposeState({ ...PRESETS[3].state, matvec: { N: 32, K: 256, S: 8, warpWidth: 32 } });
m = await metrics();
check('split-K, S = 8: S atomic updates per output', m.atomics === fmt(32 * 8) && m.worst === '8' && m.threads === fmt(sk.metrics.threads), JSON.stringify(m));
// Click a cell of W: the owning warp's mapping appears.
const grid = await page.$(E + ' [data-grid]');
await grid.evaluate((e) => e.scrollIntoView({ block: 'center' }));
await pause(150);
const gb = await grid.boundingBox();
await page.mouse.click(gb.x + gb.width * (100.5 / 256), gb.y + gb.height * (1.5 / 32));
await pause(120);
check('clicking W[1][100] selects warp 1·8 + 3 = 11 and prints its mapping', (await text(E + ' [data-mapping-text]')) === describeWorker(sk, 11), await text(E + ' [data-mapping-text]'));
await press(E, 'Decomposition', 'input-centric: a thread per input');
m = await metrics();
check('a thread per input: 256 threads, every output written by all 256', m.threads === '256' && m.worst === '256' && m.atomics === fmt(256 * 32), JSON.stringify(m));
if (out) await page.screenshot({ path: `${out}-gemv-input.png` });

// ---------------------------------------------------------------- classification
const card = async (id, decomp, crit) => {
  const k = TOUR.findIndex((r) => r.id === id);
  await page.$eval(`${T} [data-dot="${id}"]`, (b) => b.click());
  await pause(60);
  await press(T, 'Decomposition', decomp);
  await press(T, 'What decided it', crit);
  await press(T, null, 'Check');
  const fb = await text(T + ' [data-feedback]');
  const verdict = await page.$eval(T + ' [data-card]', (e) => e.dataset.verdict);
  return { k, fb, verdict };
};
let c = await card('merge', 'output-centric', 'avoids atomics');
check('trap: merge decided by atomics is marked wrong on the criterion, right on the decomposition',
  c.verdict === 'partly' && c.fb.includes('decomposition right') && c.fb.includes('decided by: load balance') && c.fb.includes('Book pp. 535'), c.fb);
c = await card('merge', 'output-centric', 'load balance');
check('merge, load balance: fully right', c.verdict === 'right' && c.fb.includes('criterion right'), c.fb);
c = await card('bfs-push', 'input-centric', 'avoids atomics');
check('trap: push BFS is not about atomics (idempotent)', c.verdict === 'partly' && c.fb.includes('idempotent') && c.fb.includes('exposes parallelism'), c.fb);
c = await card('bfs-push', 'input-centric', 'load balance');
check('push BFS, load balance: fully right', c.verdict === 'right', c.fb);
c = await card('coo', 'output-centric', 'avoids atomics');
check('COO as output-centric is wrong both ways', c.verdict === 'wrong' && c.fb.includes('it is input-centric'), c.fb);
c = await card('oddeven', 'the same either way', 'atomics not needed either way');
check('odd-even sort: the same either way', c.verdict === 'right', c.fb);
c = await card('filter', 'input-centric', 'inputs easy to find');
check('filter: input-centric, and the feedback mentions the scan inside it', c.verdict === 'right' && c.fb.includes('scan is itself output-centric'), c.fb);
c = await card('tokencounts', 'input-centric', 'inputs easy to find');
check('a card of ours is labelled and graded', c.verdict === 'right' && (await text(T + ' [data-card]')).includes('ours, not in the book'), c.fb);
check('the score counts fully right cards', (await page.$eval(T + ' [data-score]', (e) => e.dataset.score)) === '5' && (await text(T + ' [data-score]')).startsWith('6 of 21 checked'), await text(T + ' [data-score]'));
if (out) await page.screenshot({ path: `${out}-tour.png` });

// ---------------------------------------------------------------- split-K stepper
const ys = () => page.$$eval(K + ' [data-y]', (gs) => gs.map((g) => `${g.dataset.y}=${g.dataset.value}|${g.dataset.parts}`));
const modelYs = (run, n) => {
  const s = run.snaps[n];
  return s.Y.map((v, cell) => `${Math.floor(cell / SPLITK_TINY.N)},${cell % SPLITK_TINY.N}=${v}|${s.parts[cell].join(',')}`);
};
// S = 2 with atomics, warps one at a time: step once, then run to the end.
await press(K, 'Chunks S', '2');
await press(K, null, 'Step');
const cfgA = { ...SPLITK_TINY, S: 2, atomics: true };
const runA = splitKRun(cfgA, splitKOrder(cfgA, 'sequential'));
check('step 1 is warp 0\'s first load, with lane 1\'s index arithmetic', (await text(K + ' [data-sk-text]')) === describeSKStep(runA, 0, 1), await text(K + ' [data-sk-text]'));
for (let i = 0; i < 8; i++) await press(K, null, 'Run to the next commit');
check('with atomics every Y cell holds both chunks and the reference', (await page.$eval(K + ' [data-result]', (e) => e.dataset.result)) === 'right'
  && JSON.stringify(await ys()) === JSON.stringify(modelYs(runA, runA.steps.length)), JSON.stringify(await ys()));
check('S = 2: 24 threads, 12 atomic updates', (await page.$eval(K + ' [data-sk-threads]', (e) => e.dataset.skThreads)) === '24' && (await page.$eval(K + ' [data-sk-atomics]', (e) => e.dataset.skAtomics)) === '12');
// Plain +=, all warps together: go to the first lost update.
await press(K, 'Commit', 'a plain +=');
await select(K, 'Warp order', 'round-robin');
const cfgP = { ...SPLITK_TINY, S: 2, atomics: false };
const runP = splitKRun(cfgP, splitKOrder(cfgP, 'round-robin'));
const first = runP.steps.findIndex((s) => s.lost);
check('plain += with all warps together offers a lost update to jump to', await press(K, null, 'Go to the first lost update'));
const narr = await text(K + ' [data-sk-text]');
check('the first lost update is the model\'s: warp 1 overwrites chunk 0 of output 0', narr === describeSKStep(runP, first, 1) && narr.includes('chunk 0\'s contribution is overwritten and lost'), narr);
check('the narration is marked as a failure', await page.$eval(K + ' .narrate', (e) => e.classList.contains('bad')));
check('Y after that step is the model\'s', JSON.stringify(await ys()) === JSON.stringify(modelYs(runP, first + 1)), JSON.stringify(await ys()));
if (out) await page.screenshot({ path: `${out}-stepper-lost.png` });
for (let i = 0; i < 12; i++) await press(K, null, 'Run to the next commit');
check('at the end Y is wrong, with the model\'s count of lost contributions',
  (await page.$eval(K + ' [data-result]', (e) => e.dataset.result)) === 'wrong' && (await text(K + ' [data-result]')).includes(`${runP.lostCount} contributions lost`), await text(K));
// The same plain += one warp at a time is right: the emulator's situation.
await select(K, 'Warp order', 'sequential');
for (let i = 0; i < 12; i++) await press(K, null, 'Run to the next commit');
check('plain += with one warp at a time comes out right', (await page.$eval(K + ' [data-result]', (e) => e.dataset.result)) === 'right');
// S = 1: a plain store, no atomics.
await press(K, 'Chunks S', '1');
check('S = 1: no atomic updates and no lost update to go to', (await page.$eval(K + ' [data-sk-atomics]', (e) => e.dataset.skAtomics)) === '0' && !(await press(K, null, 'Go to the first lost update')));

// ---------------------------------------------------------------- the two kernels and questions
const codes = await page.$$eval('.code', (fs) => fs.map((f) => f.textContent));
check('the scatter and the gather kernels are shown', codes.some((t) => t.includes('out[o] += filter[t] * v;')) && codes.some((t) => t.includes('out[o] = acc;')));
check('their status lines use the project\'s vocabulary', codes.some((t) => t.includes('front-end-checked; wrong on purpose')) && codes.some((t) => t.includes('emulator-checked and front-end-checked')));

const ask = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const numeric = async (needle, v) => {
  const q = await ask(needle);
  await (await q.$('input[type=number]')).type(String(v));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
const choose = async (needle, idx) => {
  const q = await ask(needle);
  const radios = await q.$$('input[type=radio]');
  await radios[idx].click();
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
check('q-conv-writers accepts 5', (await numeric('Away from the edges, how many threads write each output?', 5)).includes('Correct.'));
check('q-hist-inspect accepts 256', (await numeric('How many times as many input reads', 256)).includes('Correct.'));
check('q-merge: the load-balance answer is correct', (await choose('atomics are not the reason. What is?', 0)).includes('Correct.'));
const wrong = await choose('in all three thread orders. Is it correct?', 1);
check('q-plain-scatter: "emu would have caught it" is wrong, with the reason', wrong.includes('Not quite.') && wrong.includes('never interrupted'), wrong);
check('q-plain-scatter: "no" is correct', (await choose('in all three thread orders. Is it correct?', 0)).includes('Correct.'));
check('q-metal-max: restructuring is correct', (await choose('atomicMax on floats', 0)).includes('Correct.'));

// ---------------------------------------------------------------- narrow screen
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
const wide = await page.evaluate(() => document.documentElement.scrollWidth);
check('at 390 px the page does not scroll sideways', wide <= 392, String(wide));
if (out) {
  await page.$eval(E, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: `${out}-narrow-explorer.png` });
  await page.$eval(K, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: `${out}-narrow-stepper.png` });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
