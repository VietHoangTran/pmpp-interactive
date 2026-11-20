// Browser check for section 6.9, optimisation strategy: the bottleneck loop explorer
// (tiling a shared-memory-limited kernel makes it slower; relieving a non-bottleneck
// leaves the time unchanged; coarsening too far leaves a partial wave; the bottleneck
// moves; undo and reset), the stopping-rule panel (every row's percentage of the roof
// recomputed here from /api/reports, or the "no measurement" sentence; the threshold
// moves the verdicts; the kernel-of-your-own calculator), the profiler table, three
// questions, and the page at a narrow width. Expected values come from the model,
// src/sim/bottleneck.ts, imported here directly.
// Usage: node e2e/ch06-s69.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { RESOURCE_NAME, evaluate, presetById, roofFraction, run, stopPoints, stopVerdict } from '../src/sim/bottleneck.ts';

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
const L = 'section.bench[aria-label="Bottleneck loop"]';
const R = 'section.bench[aria-label="Stopping rule on your measurements"]';
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const click = async (sel) => {
  await page.$eval(sel, (e) => e.click());
  await pause(80);
};
const setInput = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(80);
};
const setSelect = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, v);
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(100);
};
const close = (a, b) => Math.abs(a - b) <= 1e-3 * Math.max(1, Math.abs(b));

await goto(page, 'ch06/6.9');
await page.waitForSelector(L);

// ---- the explorer, against the model
/** What the explorer shows now, compared with the model after these ops from this preset. */
const state = async (preset, ops, label) => {
  const { kernel, steps } = run(presetById(preset).kernel, ops);
  const e = evaluate(kernel);
  const shown = await page.evaluate((L) => ({
    total: Number(document.querySelector(L + ' [data-total]').dataset.total),
    bars: Object.fromEntries([...document.querySelectorAll(L + ' [data-resource]')].map((b) => [b.dataset.resource, Number(b.dataset.time)])),
    bottleneck: [...document.querySelectorAll(L + ' [data-resource]')].find((b) => b.dataset.bottleneck === 'yes')?.dataset.resource,
    label: document.querySelector(L + ' [data-bottleneck-label]').textContent,
    history: [...document.querySelectorAll(L + ' [data-history] li')].map((l) => [l.dataset.step, l.dataset.verdict]),
  }), L);
  const ok =
    close(shown.total, e.total) &&
    Object.keys(e.times).every((r) => close(shown.bars[r], e.times[r])) &&
    shown.bottleneck === e.bottleneck &&
    shown.label === RESOURCE_NAME[e.bottleneck].toLowerCase() &&
    JSON.stringify(shown.history) === JSON.stringify(steps.map((s) => [s.op, s.verdict]));
  check(label + ` (model: ${e.total.toFixed(2)}, ${e.bottleneck})`, ok, JSON.stringify(shown));
  return { e, steps };
};

check('starts on the shared-memory-limited kernel', (await page.$eval(`${L} button[data-preset="shared-limited"]`, (b) => b.getAttribute('aria-pressed'))) === 'true');
await state('shared-limited', [], 'shared-limited: bars, total and bottleneck match the model');
await click(`${L} button[data-op="tile"]`);
let st = await state('shared-limited', ['tile'], 'tiling it: matches the model');
check('tiling a shared-memory-limited kernel makes it slower (40 to 80)', st.steps[0].verdict === 'slower' && close(st.e.total, 80));
const occ = await text(L + ' [data-occupancy]');
check('the occupancy readout shows 1 of 8 blocks, limited by shared memory', occ.startsWith('1 of 8 blocks per SM, limited by shared memory'), occ);
if (out) await page.$eval(L, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-tiled-slower.png' });
await page.evaluate((L) => [...document.querySelectorAll(L + ' button')].find((b) => b.textContent === 'Undo').click(), L);
await pause(80);
await state('shared-limited', [], 'undo returns to the start');
await click(`${L} button[data-op="vector"]`);
st = await state('shared-limited', ['vector'], 'vector loads at low occupancy: matches the model');
check('vector loads on the shared-limited kernel are faster (more bytes in flight)', st.steps[0].verdict === 'faster');
check('vector loads cannot be applied twice', await page.$eval(`${L} button[data-op="vector"]`, (b) => b.disabled));

// relieving a non-bottleneck
await click(`${L} button[data-preset="vecadd"]`);
check('choosing a preset clears the history', (await page.$$(L + ' [data-history] li')).length === 0);
await click(`${L} button[data-op="vector"]`);
st = await state('vecadd', ['vector'], 'vector addition, vector loads: matches the model');
const last = await text(`${L} [data-history] li:last-child`);
check('relieving a non-bottleneck leaves the time unchanged, and the history says so', st.steps[0].verdict === 'same' && last.includes('unchanged') && last.includes('not the bottleneck'), last);
await page.evaluate((L) => [...document.querySelectorAll(L + ' button')].find((b) => b.textContent === 'Reset').click(), L);
await pause(80);
for (let i = 0; i < 3; i++) await click(`${L} button[data-op="coarsen"]`);
st = await state('vecadd', ['coarsen', 'coarsen', 'coarsen'], 'reset, then coarsening three times: matches the model');
check('coarsening: unchanged twice, then slower on a partial wave', JSON.stringify(st.steps.map((s) => s.verdict)) === JSON.stringify(['same', 'same', 'slower']));
const occ2 = await text(L + ' [data-occupancy]');
check('the readout names the partial wave', occ2.includes('the last one partial'), occ2);
check('coarsening is disabled after three', await page.$eval(`${L} button[data-op="coarsen"]`, (b) => b.disabled));
check('padding is disabled with a reason when there is no shared memory', (await page.$eval(`${L} button[data-op="pad"]`, (b) => b.disabled)) && (await text(L)).includes('no shared memory to pad'));

// the bottleneck moves
await click(`${L} button[data-preset="naive"]`);
await click(`${L} button[data-op="tile"]`);
await click(`${L} button[data-op="unroll"]`);
st = await state('naive', ['tile', 'unroll'], 'naive matmul, tiling then unrolling: matches the model');
check('the bottleneck moves: global, then arithmetic, then global again', st.steps.map((s) => s.after.bottleneck).join(',') === 'compute,global' && st.steps[0].before.bottleneck === 'global');
await page.evaluate((L) => [...document.querySelectorAll(L + ' button')].find((b) => b.textContent === 'Reset').click(), L);
await pause(80);
await state('naive', [], 'reset returns to the starting kernel');
await click(`${L} button[data-preset="exchange"]`);
await click(`${L} button[data-op="double-buffer"]`);
st = await state('exchange', ['double-buffer'], 'neighbour exchange, double buffering: matches the model');
check('double buffering a barrier-bound kernel is faster and moves the bottleneck', st.steps[0].verdict === 'faster' && st.e.bottleneck === 'global');

// ---- the stopping rule
await page.waitForFunction((R) => !document.querySelector(R + ' [data-stop="loading"]'), {}, R);
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const gpus = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const roofs = gpus.flatMap((b) => {
  const r = reports[b].results.filter((x) => x.lab === 'ch05/lab1_roofline' && x.variant === 'solution').flatMap((x) => x.bench).find((x) => x.name === 'roofline');
  return r && r.compute_gflops > 0 && r.bandwidth_gb_per_s > 0 ? [{ b, peak: r.compute_gflops, bw: r.bandwidth_gb_per_s }] : [];
});
if (!roofs.length) {
  check('no lab 5.1 roof on this machine: the panel says "no measurement" and names the command', (await page.$(R + ' [data-stop="none"]')) !== null && (await text(R)).includes('python3 labs/pmpp.py run 5.1'));
} else {
  const roof = roofs[0];
  check(`roof line names ${roof.b}`, (await page.$eval(R + ' [data-roof]', (e) => e.dataset.roof)) === roof.b);
  // the rows the page should place: per lab, the reader's work rows if any, else the reference's
  const results = reports[roof.b].results;
  const labs = [...new Set(results.map((r) => r.lab))];
  const pairs = labs.flatMap((lab) => {
    const work = results.filter((r) => r.lab === lab && r.variant === 'work').flatMap((r) => r.bench);
    const rows = work.length ? work : results.filter((r) => r.lab === lab && r.variant === 'solution').flatMap((r) => r.bench);
    return rows.map((row) => ({ lab, row }));
  });
  const want = stopPoints(pairs).map((p) => roofFraction(p.flop, p.bytes, p.seconds, roof.peak, roof.bw).fraction);
  const shown = await page.$$eval(R + ' tr[data-point]', (t) => t.map((r) => [r.dataset.point, Number(r.dataset.fraction), r.dataset.verdict]));
  if (!want.length) {
    check('no placeable lab rows: the panel says "no measurement"', (await page.$(R + ' [data-stop="none"]')) !== null);
  } else {
    check(`the table shows ${want.length} kernels with "% of roof"`, shown.length === want.length && (await text(R + ' table[data-stop-table] thead')).includes('% of roof'), `${shown.length} vs ${want.length}`);
    check('every percentage of the roof matches the model on the reported rows', shown.every((s, i) => Math.abs(s[1] - want[i]) < 1e-4), JSON.stringify(shown.map((s) => s[1])));
    check('verdicts at the default 80% threshold match the model', shown.every((s, i) => s[2] === stopVerdict(want[i], 0.8)), JSON.stringify(shown.map((s) => s[2])));
    await setInput(R + ' input[aria-label="threshold"]', 100);
    const at100 = await page.$$eval(R + ' tr[data-point]', (t) => t.map((r) => r.dataset.verdict));
    check('at a 100% threshold the verdicts move with it', at100.every((v, i) => v === stopVerdict(want[i], 1)), at100.join(','));
    await setInput(R + ' input[aria-label="threshold"]', 50);
    const at50 = await page.$$eval(R + ' tr[data-point]', (t) => t.map((r) => r.dataset.verdict));
    check('at 50% too', at50.every((v, i) => v === stopVerdict(want[i], 0.5)), at50.join(','));
    if (want.some((f) => f > 1.1)) check('a point above the roof gets the explanation about the byte count', (await page.$(R + ' [data-above-note]')) !== null);
    const placed = new Set(shown.map((s) => s[0]));
    const missing = await page.$$eval(R + ' [data-missing] li', (l) => l.map((x) => x.textContent));
    check('labs with no rows are listed with the command to run', ['5.2', '5.3', '6.2', '6.4', '6.5'].filter((n) => !placed.has(n)).length === missing.length, missing.join(' | '));
  }
}
// a kernel of your own: the defaults against the book's example, then q-stop's numbers
let mine = Number(await page.$eval(R + ' [data-mine-result]', (e) => e.dataset.mineResult));
check('kernel of your own: the default against the book’s example matches the model', close(mine, roofFraction(2e9, 8e9, 0.004, 66900, 3350).fraction), String(mine));
const inputs = await page.$$(R + ' [data-mine] input');
for (const [i, v] of [1, 12, 5].entries()) {
  await page.evaluate((el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, inputs[i], v);
  await pause(60);
}
mine = Number(await page.$eval(R + ' [data-mine-result]', (e) => e.dataset.mineResult));
check('kernel of your own: 1 GFLOP, 12 GB, 5 ms on the book’s example is 2400 / 3350 of the roof', close(mine, 2400 / 3350), String(mine));
if (roofs.length) {
  await setSelect(R + ' [data-mine] select', 'yours');
  mine = Number(await page.$eval(R + ' [data-mine-result]', (e) => e.dataset.mineResult));
  check('and against your own roof', close(mine, roofFraction(1e9, 12e9, 0.005, roofs[0].peak, roofs[0].bw).fraction), String(mine));
}
if (out) await page.$eval(R, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-stop.png' });

// ---- the profiler table
const prof = await page.$$eval('table[data-profilers] tbody tr', (t) => t.map((r) => r.firstElementChild.textContent));
check('profiler table: CUDA, HIP, Metal', JSON.stringify(prof) === JSON.stringify(['CUDA', 'HIP', 'Metal']), prof.join(','));

// ---- questions
const q = async (promptPart) => page.evaluateHandle((p) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(p)), promptPart);
let h = await q('moves 12 GB in 5 ms');
await h.evaluate((e) => {
  const el = e.querySelector('input[type=number]');
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '80');
  el.dispatchEvent(new Event('input', { bubbles: true }));
});
await pause(60);
await (await h.$('button.primary')).click();
await pause(100);
let v = await h.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
check('q-stop: 80% is correct', v.includes('Correct.') && v.includes('1.25'), v);
const choose = async (promptPart, optionPart) => {
  const h = await q(promptPart);
  await h.evaluate((e, o) => [...e.querySelectorAll('.opts label')].find((x) => x.textContent.includes(o)).querySelector('input').click(), optionPart);
  await pause(60);
  await (await h.$('button.primary')).click();
  await pause(100);
  return h.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
};
v = await choose('Which do you trust?', 'The duration.');
check('q-sol: the duration is correct, with facts A17', v.includes('Correct.') && v.includes('A17'), v);
v = await choose('unrolls its loops', 'At most about 10% faster');
check('q-nonbottleneck: correct', v.includes('Correct.') && v.includes('p. 153'), v);

// ---- narrow screen
await page.setViewport({ width: 390, height: 900 });
await pause(300);
const fits = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
check('at 390 px nothing overflows the page', fits);
if (out) await page.$eval(L, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-narrow.png' });
if (out) await page.$eval(R, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-narrow-stop.png' });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
