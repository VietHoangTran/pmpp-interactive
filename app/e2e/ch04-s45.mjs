// Browser check for section 4.5, "Control divergence": the divergence simulator (four
// statements, the stepper, the order switch), the batch of loops, the exercise 1
// prediction, the boundary-check counter with its erratum hunt, the lab 4.2 prediction
// and the questions. Everything the page shows is compared with the models it runs on
// (src/sim/divergence.ts and src/sim/warps.ts, imported here directly).
// The narrated animation of this section has its own check, e2e/ch04-scene-warp-branch.mjs.
// Usage: node e2e/ch04-s45.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import {
  exercise1, fig410Trips, ifElse, loop, loopBatch, loopRows, maskBelow, maskModulo, run, select, shuffledTrips, sortedTrips,
} from '../src/sim/divergence.ts';
import { BOOK_CASES, boundary1D, boundary2D, splitWarp1D, warpsInRightEdgeBlocks } from '../src/sim/warps.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });

// Leave no trace in the real progress file, and start from an empty one so the
// predictions are uncommitted whatever the reader has already done. Reports are
// replaced too: first by none at all, then by a made-up one (clearly named as a
// fixture) to drive the places that read the reader's own device and lab 4.2.
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
const branch = (pattern, ratio) => ({ name: 'branch', pattern, width: 32, threads_per_block: 256, n: 1048576, iters: 4096, fraction_b: 0.5, seconds: 0.1 * ratio, sorted_seconds: 0.1, ratio_to_sorted: ratio });
const FIXTURE = {
  cuda: {
    generated: '2026-11-20T00:00:00',
    device: { kind: 'device', backend: 'cuda', available: true, name: 'e2e fixture device', warpSize: '32', maxThreadsPerBlock: '1024' },
    results: [{
      lab: 'ch04/lab2_divergence', variant: 'solution', backend: 'cuda', tile: 16, built: true, test_rc: 0,
      tests: [{ case: 'fixture', pass: true }],
      bench: [
        branch('all A', 1),
        branch('alternating warps', 1.02),
        branch("half of each warp's lanes", 1.9),
        branch('alternating lanes', 1.95),
        { name: 'select', pattern: "half of each warp's lanes", width: 32, seconds: 0.2, sorted_seconds: 0.1, ratio_to_sorted: 2, ratio_to_branch: 1.05 },
        { name: 'loop', order: 'sorted', width: 32, seconds: 0.1, ratio_to_sorted: 1 },
        { name: 'loop', order: 'shuffled', width: 32, seconds: 0.17, ratio_to_sorted: 1.7 },
      ],
    }],
    work_state: {},
  },
  hip: {
    generated: '2026-11-20T00:00:00',
    device: { kind: 'device', backend: 'hip', available: true, name: 'e2e fixture wide device', warpSize: '64', maxThreadsPerBlock: '256' },
    results: [],
    work_state: {},
  },
};
let fixture = false;
const saved = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  const u = r.url();
  if (u.includes('/api/progress')) {
    if (r.method() === 'PUT') saved.push(r.postData() ?? '');
    return r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : JSON.stringify(EMPTY) });
  }
  if (u.includes('/api/reports')) return r.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(fixture ? FIXTURE : {}) });
  return r.continue();
});
// The narration manifest and audio for this chapter may not exist yet; those 404s are expected.
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const big = (v) => v.toLocaleString('en-US');
const pct = (v) => `${Math.round(v * 1000) / 10}%`;
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const bench = (label) => `section.bench[aria-label="${label}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.trim());
/** Press the button with exactly this text inside `scope` (optionally inside the group with this aria-label). */
const press = async (scope, label, group) => {
  const done = await page.evaluate((scope, label, group) => {
    const root = document.querySelector(group ? `${scope} [role="group"][aria-label="${group}"]` : scope);
    const btn = root && [...root.querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label, group ?? null);
  await pause(80);
  return done;
};
/** Type into a number or range input the way React notices. */
const setInput = async (scope, aria, value) => {
  await page.$eval(`${scope} input[aria-label="${aria}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(90);
};
const setSelect = async (scope, aria, value) => {
  await page.$eval(`${scope} select[aria-label="${aria}"]`, (el, v) => {
    el.value = v;
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await pause(90);
};
const clickSvg = async (sel) => {
  await page.$eval(sel, (el) => el.dispatchEvent(new MouseEvent('click', { bubbles: true })));
  await pause(80);
};
const outOf = (scope, readout, name) => text(`${scope} [data-readout="${readout}"] [data-out="${name}"]`);
const question = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const shot = async (sel, name) => {
  if (!out) return;
  await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};

const D = bench('Divergence simulator');
const L = bench('One loop per thread, lengths from the data');
const B = bench('Boundary-check counter');
const EX1 = '[data-predict="ch04/4.5/p-ex1"]';

await goto(page, 'ch04/4.5');
await page.waitForSelector(D);
await page.waitForSelector(B);

// ---------------------------------------------------------------- the page
check('every simulation says what it leaves out', (await page.$$('.note.caveat')).length >= 4, String((await page.$$('.note.caveat')).length));
check('the page marks what is not in the book', (await page.$$('.note.beyond')).length >= 5);
const code = await page.evaluate(() => {
  const f = [...document.querySelectorAll('figure.code')].find((x) => x.textContent.includes('foo_kernel'));
  return f ? { text: f.textContent, status: f.querySelector('.status')?.textContent ?? '', hl: [...f.querySelectorAll('.ln.hl')].map((l) => Number(l.dataset.n)) } : null;
});
check("exercise 1's kernel is shown as stored, and says how it was checked", !!code && code.text.includes('if(threadIdx.x < 40 || threadIdx.x >= 104)') && code.text.includes('foo_kernel<<<(N + 128 - 1)/128, 128>>>') && code.status.includes('front-end-checked'), code?.status);
check('lines 3, 4, 6, 7, 9 and 10 of the kernel are marked', !!code && eq(code.hl, [3, 4, 6, 7, 9, 10]), JSON.stringify(code?.hl));

// ---------------------------------------------------------------- the divergence simulator
const sim = async () => {
  const names = ['passes', 'spent', 'uniform', 'overhead', 'idle', 'efficiency'];
  const o = {};
  for (const n of names) o[n] = await outOf(D, 'trace', n);
  return o;
};
const rows = () => page.$$eval(`${D} svg[data-view="lanes"] g[data-row]`, (gs) => gs.map((g) => ({ count: Number(g.dataset.count), part: g.dataset.part })));
const wants = () => page.$$eval(`${D} svg[data-view="lanes"] g[data-head] text`, (ts) => ts.slice(1).map((t) => t.textContent));
const warpButtons = () => page.$$eval(`${D} [data-warp-summary]`, (bs) => bs.map((b) => b.textContent.trim()));
const agrees = (o, t) => o.spent === String(t.spent) && Number(o.uniform) === Number(t.uniform.toFixed(2)) && o.idle === String(t.idle) && o.efficiency === pct(t.efficiency);

let t = run(ifElse(32, maskBelow(32, 24), 3, 3));
let o = await sim();
check('the simulator opens on Fig. 4.9: 2 passes, 8 warp-steps where 5 would do, 96 idle lane-steps', o.passes === '2' && agrees(o, t) && t.spent === 8 && t.uniform === 5 && t.idle === 96, JSON.stringify(o));
let r = await rows();
check('its rows are the model\'s: the test by 32, A by 24 three times, B by 8 three times, C by 32', eq(r.map((x) => x.count), t.issues.map((x) => x.count)) && eq(r.map((x) => x.count), [32, 24, 24, 24, 8, 8, 8, 32]) && eq(r.map((x) => x.part), ['op', 'A', 'A', 'A', 'B', 'B', 'B', 'op']), JSON.stringify(r));
check('24 lanes want A and 8 want B', eq(await wants(), maskBelow(32, 24).map((a) => (a ? 'A' : 'B'))));
check('the other warps of the block are uniform', eq(await warpButtons(), ['warp 0: 24 A, 8 B', 'warp 1: all B', 'warp 2: all B', 'warp 3: all B']), JSON.stringify(await warpButtons()));
await shot(D, 'sim-fig49');

for (const T of [16, 1]) {
  await setInput(D, 'Threshold T', T);
  t = run(ifElse(32, maskBelow(32, T), 3, 3));
  o = await sim();
  check(`T = ${T}: still 2 passes and 8 warp-steps, whatever the split`, o.passes === '2' && o.spent === '8' && agrees(o, t), JSON.stringify(o));
}
await setInput(D, 'Threshold T', 32);
t = run(ifElse(32, maskBelow(32, 32), 3, 3));
o = await sim();
check('T = 32 falls on a warp boundary: one pass, no idle lane, 5 warp-steps', o.passes === '1' && o.idle === '0' && o.spent === '5' && agrees(o, t), JSON.stringify(o));
check('and the next warp is uniform too', eq(await warpButtons(), ['warp 0: all A', 'warp 1: all B', 'warp 2: all B', 'warp 3: all B']));
await press(D, 'warp 1: all B', 'Warp of the block');
o = await sim();
check('warp 1 makes one pass over B', o.passes === '1' && o.idle === '0' && eq((await rows()).map((x) => x.part), ['op', 'B', 'B', 'B', 'op']), JSON.stringify(o));

await setSelect(D, 'Warp width', 'w64');
check('at 64 wide the same threshold splits warp 0 of the block 32 against 32', eq(await warpButtons(), ['warp 0: 32 A, 32 B', 'warp 1: all B']), JSON.stringify(await warpButtons()));
await press(D, 'warp 0: 32 A, 32 B', 'Warp of the block');
t = run(ifElse(64, maskBelow(64, 32), 3, 3));
o = await sim();
check('that warp makes 2 passes with half its lanes idle in each', o.passes === '2' && agrees(o, t) && eq((await rows()).map((x) => x.count), [64, 32, 32, 32, 32, 32, 32, 64]), JSON.stringify(o));
check('the drawing has 64 lanes', (await page.$$eval(`${D} svg[data-view="lanes"] g[data-row="0"] rect[data-lane]`, (x) => x.length)) === 64);
await shot(D, 'sim-64');

await setSelect(D, 'Warp width', 'book');
await setInput(D, 'Threshold T', 24);
await press(D, 'interleaved', 'Order of the passes');
const ti = run(ifElse(32, maskBelow(32, 24), 3, 3), 'interleaved');
o = await sim();
r = await rows();
check('interleaved: the same instructions in another order, with the same totals', eq(r.map((x) => x.part), ['op', 'A', 'B', 'A', 'B', 'A', 'B', 'op']) && eq(r.map((x) => x.count), ti.issues.map((x) => x.count)) && o.spent === '8' && o.idle === '96' && o.passes === '2', JSON.stringify(r));
await press(D, 'one path, then the other', 'Order of the passes');

for (const k of [1, 8]) {
  await setInput(D, 'Instructions in A', k);
  await setInput(D, 'Instructions in B', k);
  t = run(ifElse(32, maskBelow(32, 24), k, k));
  const s = run(select(32, k, k));
  o = await sim();
  const sel = [await outOf(D, 'select', 'select-spent'), await outOf(D, 'select', 'select-discarded')];
  check(`paths of ${k}: the branch spends ${t.spent}, compute-both-and-select ${s.spent} with no lane waiting`, o.spent === String(t.spent) && sel[0] === String(s.spent) && s.idle === 0 && sel[1] === String(8 * k + 24 * k), JSON.stringify([o, sel]));
}
await setInput(D, 'Instructions in A', 3);
await setInput(D, 'Instructions in B', 0);
t = run(ifElse(32, maskBelow(32, 24), 3, 0));
o = await sim();
check('an if with no else: one pass, and the lanes that skip it still wait', o.passes === '1' && o.idle === String(t.idle) && t.idle === 24 && agrees(o, t), JSON.stringify(o));
await setInput(D, 'Instructions in B', 3);

// The stepper.
const stepText = () => text(`${D} [data-readout="step"]`);
await press(D, 'Restart');
check('Restart clears the rows', (await rows()).length === 0 && (await stepText()).startsWith('Nothing issued yet'), await stepText());
await press(D, 'Step');
check('step 1 is the test, executed by every lane', (await rows()).length === 1 && (await stepText()) === 'Step 1 of 8: test: threadIdx.x < 24. All 32 lanes execute it.', await stepText());
await press(D, 'Step');
check('step 2 is the first instruction of A: 24 execute and 8 wait', (await stepText()) === 'Step 2 of 8: A, instruction 1. 24 of 32 lanes execute it and 8 wait.', await stepText());
await press(D, 'Back');
check('Back removes a row', (await rows()).length === 1);
await press(D, 'Play');
await pause(2100);
const played = (await rows()).length;
check('Play issues instructions by itself', played >= 4 && played <= 7, String(played));
await press(D, 'Pause');
await press(D, 'Show all');
check('Show all ends on C and sums up', (await rows()).length === 8 && (await stepText()).includes('That is the whole program: 8 warp-steps, where the same work with no lane waiting would take 5.'), await stepText());

// if (i % m == 0)
await setSelect(D, 'Statement', 'modulo');
t = run(ifElse(32, maskModulo(32, 2), 3, 3));
o = await sim();
check('i % 2 == 0: every other lane, 2 passes, half the lanes idle in each', o.passes === '2' && agrees(o, t) && eq((await rows()).map((x) => x.count), [32, 16, 16, 16, 16, 16, 16, 32]), JSON.stringify(o));
check('every warp of the block is split the same way', (await warpButtons()).every((b) => b.endsWith('16 A, 16 B')), JSON.stringify(await warpButtons()));
await setInput(D, 'm', 4);
check('i % 4 == 0: 8 lanes take A', (await wants()).filter((x) => x === 'A').length === 8 && (await outOf(D, 'trace', 'spent')) === '8');

// A painted mask.
await setSelect(D, 'Statement', 'paint');
check('the painted mask starts with the first 16 lanes on A', eq(await wants(), maskBelow(32, 16).map((a) => (a ? 'A' : 'B'))));
await clickSvg(`${D} svg[data-view="lanes"] [data-head-lane="20"]`);
check('clicking a lane flips it', (await wants())[20] === 'A' && (await wants()).filter((x) => x === 'A').length === 17 && (await rows())[1].count === 17);
await press(D, 'all A', 'Mask presets');
o = await sim();
check('all A: one pass', o.passes === '1' && o.idle === '0');
await clickSvg(`${D} svg[data-view="lanes"] [data-head-lane="0"]`);
t = run(ifElse(32, maskBelow(32, 32).map((v, i) => (i === 0 ? false : v)), 3, 3));
o = await sim();
check('one lane out of step costs the whole warp a second pass', o.passes === '2' && o.spent === '8' && agrees(o, t) && eq((await rows()).map((x) => x.count), [32, 31, 31, 31, 1, 1, 1, 32]), JSON.stringify(o));
await shot(D, 'sim-paint');

// The loop of Fig. 4.10.
await setSelect(D, 'Statement', 'loop');
const loopOut = async () => ({ ...(await sim()), u: await outOf(D, 'trace', 'uniform-iterations'), d: await outOf(D, 'trace', 'divergent-iterations') });
t = run(loop(32, fig410Trips(32)));
o = await loopOut();
check('the loop opens on Fig. 4.10\'s printed counts: 8 iterations, 4 with every lane and 4 with lanes waiting', o.passes === '8' && o.u === '4' && o.d === '4' && agrees(o, t), JSON.stringify(o));
check('lanes drop out as the model says: 32, 32, 32, 32, 28, 24, 16, 8', eq((await rows()).map((x) => x.count), [32, 32, 32, 32, 32, 28, 24, 16, 8]) && eq(t.issues.map((x) => x.count), [32, 32, 32, 32, 32, 28, 24, 16, 8]));
check('the trip counts above the lanes are the printed values, repeated', eq((await wants()).map(Number), fig410Trips(32)));
await shot(D, 'sim-loop');
await press(D, 'the same counts, sorted', 'Trip count presets');
const sortedOut = await loopOut();
check('sorting inside one warp changes nothing', sortedOut.spent === o.spent && sortedOut.idle === o.idle && eq((await wants()).map(Number), sortedTrips(fig410Trips(32))));
await press(D, "Fig. 4.10's printed values, repeated", 'Trip count presets');
await clickSvg(`${D} svg[data-view="trips"] rect[data-lane="3"][data-trip="10"]`);
const edited = fig410Trips(32).map((v, i) => (i === 3 ? 10 : v));
t = run(loop(32, edited));
o = await loopOut();
check('clicking a column sets that lane: one lane of 10 makes the warp run 10 iterations', o.passes === '10' && (await wants())[3] === '10' && agrees(o, t), JSON.stringify(o));
await press(D, 'all 6', 'Trip count presets');
o = await loopOut();
check('equal trip counts do not diverge', o.passes === '6' && o.u === '6' && o.d === '0' && o.idle === '0', JSON.stringify(o));
await press(D, 'one lane of 10, the rest 2', 'Trip count presets');
t = run(loop(32, [10, ...Array(31).fill(2)]));
o = await loopOut();
check('one long lane holds 31 finished lanes for 8 iterations', o.passes === '10' && o.idle === '248' && agrees(o, t), JSON.stringify(o));
await setInput(D, 'Instructions in the loop body', 2);
check('a body of 2 instructions doubles the rows', (await rows()).length === 21 && (await outOf(D, 'trace', 'spent')) === '21');
await setInput(D, 'Instructions in the loop body', 1);

// ---------------------------------------------------------------- the batch of loops
const batchRow = (id) => page.$eval(`${L} tr[data-batch="${id}"]`, (tr) => Object.fromEntries([...tr.querySelectorAll('[data-out]')].map((c) => [c.dataset.out, c.textContent.trim()])));
const longest = (i) => page.$$eval(`${L} svg`, (svgs, i) => [...svgs[i].querySelectorAll('g[data-batch-warp]')].map((g) => Number(g.dataset.longest)), i);
const batchAgrees = async (warps, width, seed, why) => {
  const trips = shuffledTrips(warps * width, 1, 64, seed);
  const sh = loopBatch(trips, width);
  const so = loopBatch(sortedTrips(trips), width);
  const a = await batchRow('shuffled');
  const b = await batchRow('sorted');
  const ok = a.spent === String(sh.spent) && b.spent === String(so.spent) && a.useful === big(sh.useful) && b.useful === big(so.useful)
    && a.idle === big(sh.idle) && b.idle === big(so.idle) && a.efficiency === pct(sh.efficiency) && b.efficiency === pct(so.efficiency)
    && eq(await longest(0), sh.warps.map((w) => w.longest)) && eq(await longest(1), so.warps.map((w) => w.longest));
  check(why(sh, so), ok && sh.useful === so.useful && so.spent < sh.spent, JSON.stringify([a, b]));
  return { sh, so };
};
let bt = await batchAgrees(8, 32, 7, (sh, so) => `8 warps of 32: the same work, ${sh.spent} warp-iterations shuffled against ${so.spent} sorted`);
check('sorted, the warps run 8, 16, ... 64 iterations: 288 in all, where 260 would do with no waiting', bt.so.spent === 288 && eq(await longest(1), [8, 16, 24, 32, 40, 48, 56, 64]) && (await outOf(L, 'batch', 'uniform')) === '260');
await shot(L, 'batch');
await press(L, '16', 'Number of warps');
await batchAgrees(16, 32, 7, (sh, so) => `16 warps: ${sh.spent} against ${so.spent}`);
await setSelect(L, 'Warp width of the batch', 'w64');
await batchAgrees(16, 64, 7, (sh, so) => `16 warps of 64: ${sh.spent} against ${so.spent}`);
await press(L, 'Shuffle again');
await batchAgrees(16, 64, 8, (sh, so) => `another shuffle: ${sh.spent} against ${so.spent}, the sorted batch unchanged`);

// ---------------------------------------------------------------- exercise 1
const e1 = exercise1(32);
const w0 = (k) => e1.detail.find((x) => x.block === 0 && x.warp === k);
const model = {
  'l4-active': e1.line4.active, 'l4-divergent': e1.line4.divergent,
  'l4-eff0': 100 * w0(0).line4.efficiency, 'l4-eff1': 100 * w0(1).line4.efficiency, 'l4-eff3': 100 * w0(3).line4.efficiency,
  'l7-active': e1.line7.active, 'l7-divergent': e1.line7.divergent, 'l7-eff0': 100 * w0(0).line7.efficiency,
  'loop-uniform': e1.loop.uniform, 'loop-divergent': e1.loop.divergent,
};
check('the model gives the answers of exercise 1: 24, 16, 100, 25, 75; 32, 32, 50; 3 and 2', eq(Object.values(model), [24, 16, 100, 25, 75, 32, 32, 50, 3, 2]));
check('the exercise asks for ten numbers and shows no answer before they are committed', (await page.$$(`${EX1} input[data-field]`)).length === 10 && !(await page.$(`${EX1} [data-reveal]`)));
const commitDisabled = () => page.$eval(`${EX1} button.primary`, (b) => b.disabled);
const mine = { ...model, 'l4-divergent': 8, 'loop-divergent': 5 }; // two wrong on purpose
const fillField = (k, v) => page.$eval(`${EX1} input[data-field="${k}"]`, (el, v) => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
  el.dispatchEvent(new Event('input', { bubbles: true }));
}, v);
for (const [k, v] of Object.entries(mine)) if (k !== 'l4-active') await fillField(k, v);
await pause(100);
check('it cannot be committed with a field empty', await commitDisabled());
await fillField('l4-active', mine['l4-active']);
await pause(100);
await page.$eval(`${EX1} button.primary`, (b) => b.click());
await pause(250);
const revealed = await page.$$eval(`${EX1} tr[data-field]`, (trs) => trs.map((tr) => ({ key: tr.dataset.field, ok: tr.dataset.ok, model: tr.querySelector('[data-out="model"]').textContent.trim() })));
check('the reveal lists the model\'s ten counts', eq(revealed.map((x) => parseFloat(x.model)), Object.values(model)), JSON.stringify(revealed));
check('and marks the two entries that differ', eq(revealed.filter((x) => x.ok === 'no').map((x) => x.key), ['l4-divergent', 'loop-divergent']) && (await text(`${EX1} [data-reveal]`)).startsWith('8 of 10 match the model.'));
const drawn = await page.$$eval(`${EX1} svg[data-view="lanes"]`, (svgs) => svgs.map((s) => [...s.querySelectorAll('g[data-row]')].map((g) => Number(g.dataset.count))));
check('line 4 in block 0: 32, 8, 0 and 24 lanes execute it', eq(drawn[0], [0, 1, 2, 3].map((k) => w0(k).line4.count)) && eq(drawn[0], [32, 8, 0, 24]), JSON.stringify(drawn[0]));
check('line 7 in block 0: 16 lanes in every warp', eq(drawn[1], [16, 16, 16, 16]));
check('the loop in warp 0: 32, 32, 32, then 22 and 11 lanes', eq(drawn[2], loopRows(w0(0).trips).map((a) => a.filter(Boolean).length)) && eq(drawn[2], [32, 32, 32, 22, 11]), JSON.stringify(drawn[2]));
await pause(600); // the progress file is written after a short delay
const stored = saved.map((s) => { try { return JSON.parse(s).answers?.['ch04/4.5/p-ex1']?.value; } catch { return undefined; } }).filter(Boolean).pop();
check('the ten numbers are saved under ch04/4.5/p-ex1', eq(stored, Object.values(mine)), JSON.stringify(stored));
await shot(EX1, 'ex1');

// ---------------------------------------------------------------- the boundary-check counter
const counts = async () => {
  const o = {};
  for (const n of ['blocks', 'threads', 'warps', 'divergent', 'idle', 'full', 'share']) o[n] = await outOf(B, 'boundary', n);
  return o;
};
const same = (o, b) => o.blocks.startsWith(big(b.blocks)) && o.threads === big(b.threads) && o.warps === big(b.warps) && o.divergent === big(b.divergent)
  && o.idle === big(b.idle) && o.full === big(b.full) && o.share === pct(b.share);
const drawnStates = () => page.$$eval(`${B} svg[data-view="launch"] rect[data-state]`, (rs) => rs.map((r) => r.dataset.state));
const tally = (states) => ({ full: states.filter((s) => s === 'full').length, divergent: states.filter((s) => s === 'divergent').length, idle: states.filter((s) => s === 'idle').length });

let b = boundary1D(1003, 64, 32);
let c = await counts();
check('the counter opens on the book\'s vector of 1003: 16 blocks, 1,024 threads, 1 of 32 warps split', same(c, b) && c.warps === '32' && c.divergent === '1' && c.threads === '1,024', JSON.stringify(c));
const sp = splitWarp1D(1003, 64, 32);
check('it names the warp: warp 1 of block 15, 11 threads in range and 21 out', (await text(`${B} [data-readout="split"]`)).startsWith(`The one warp that splits is warp ${sp.warp} of block ${sp.block}: ${sp.valid} threads pass the test and ${sp.invalid} do not.`) && sp.block === 15 && sp.invalid === 21, await text(`${B} [data-readout="split"]`));
const split = await page.$eval(`${B} svg[data-view="launch"] rect[data-state="divergent"]`, (r) => [r.dataset.block, r.dataset.warp]);
check('and draws it: 31 full warps and one split, in block 15', eq(tally(await drawnStates()), { full: 31, divergent: 1, idle: 0 }) && eq(split, ['15', '1']), JSON.stringify(split));
await shot(B, 'counter-1d');
for (const [n, of] of [[100, 4], [1000, 32], [10000, 314]]) {
  await setInput(B, 'Elements n', n);
  b = boundary1D(n, 64, 32);
  c = await counts();
  check(`${big(n)} elements: 1 of ${of} warps split`, same(c, b) && c.divergent === '1' && c.warps === String(of), JSON.stringify(c));
}
check('10,000 elements: 313 warps hold an element and one more is launched and idle', b.warps - b.idle === 313 && c.idle === '1' && eq(tally(await drawnStates()), { full: 312, divergent: 1, idle: 1 }));
await setInput(B, 'Elements n', 96);
c = await counts();
check('96 elements end on a warp boundary: nothing splits, one warp idle', c.divergent === '0' && c.idle === '1' && (await text(`${B} [data-readout="split"]`)).startsWith('The last element is the last lane of a warp, so nothing splits.'), JSON.stringify(c));
await setSelect(B, 'Warp width of the launch', 'w64');
c = await counts();
check('the same 96 elements at 64 wide: one warp split', same(c, boundary1D(96, 64, 64)) && c.divergent === '1', JSON.stringify(c));
await setSelect(B, 'Warp width of the launch', 'book');

await press(B, '2D: a picture or a matrix', 'Dimensions');
const p76 = { width: 76, height: 62, blockX: 16, blockY: 16 };
b = boundary2D(p76, 32);
c = await counts();
check('2D opens on the 76 by 62 picture: 20 blocks, 31 of 160 warps split', same(c, b) && c.divergent === '31' && c.warps === '160', JSON.stringify(c));
const regionRows = () => page.$$eval(`${B} tr[data-region]`, (trs) => trs.map((tr) => ['blocks', 'warps', 'divergent', 'idle'].map((k) => Number(tr.querySelector(`[data-out="${k}"]`).textContent))));
check('its four regions: 12, 3, 4 and 1 blocks, with 0, 24, 0 and 7 warps split', eq(await regionRows(), b.regions.map((x) => [x.blocks, x.warps, x.divergent, x.idle])) && eq(b.regions.map((x) => x.divergent), [0, 24, 0, 7]), JSON.stringify(await regionRows()));
check('every warp is drawn with its state', eq(tally(await drawnStates()), { full: b.full, divergent: b.divergent, idle: b.idle }), JSON.stringify(tally(await drawnStates())));
const bottom = await page.$$eval(`${B} svg[data-view="launch"] rect[data-by="3"][data-bx="0"]`, (rs) => rs.map((r) => r.dataset.state));
check('a block on the bottom edge has seven full warps and one idle, none split', eq(bottom, [...Array(7).fill('full'), 'idle']), JSON.stringify(bottom));
await shot(B, 'counter-2d');
await setInput(B, 'Height (rows)', 61);
const odd = boundary2D({ ...p76, height: 61 }, 32).regions[2];
check('an odd height splits one warp in each bottom-edge block too', eq((await regionRows())[2].slice(2), [odd.divergent, odd.idle]) && odd.divergent === 4, JSON.stringify(await regionRows()));
await setInput(B, 'Height (rows)', 62);
await setSelect(B, 'Warp width of the launch', 'w64');
b = boundary2D(p76, 64);
c = await counts();
check('at 64 wide the same picture has 20 of 80 warps split', same(c, b) && c.divergent === '20' && c.warps === '80' && eq((await regionRows()).map((x) => x[2]), [0, 12, 4, 4]), JSON.stringify(c));
await shot(B, 'counter-2d-64');
await setSelect(B, 'Warp width of the launch', 'book');

const table = await page.$$eval(`${B} tr[data-case]`, (trs) => trs.map((tr) => ({ id: tr.dataset.case, model: tr.querySelector('[data-out="model"]').textContent.trim(), differs: tr.hasAttribute('data-differs') })));
check('the table of the book\'s examples holds the model\'s count for each', eq(table.map((x) => x.id), BOOK_CASES.map((x) => x.id)) && BOOK_CASES.every((k, i) => { const m = boundary2D(k.picture, 32); return table[i].model.startsWith(`${m.divergent} of ${big(m.warps)}`); }), JSON.stringify(table));
check('three printed figures differ from the count: 10,000 elements and both readings of 200 × 150', eq(table.filter((x) => x.differs).map((x) => x.id), ['v10000', 'p200x150', 'p150x200']));
const loadCase = async (id) => { await page.$eval(`${B} tr[data-case="${id}"] button`, (x) => x.click()); await pause(120); };
await loadCase('p200x150');
c = await counts();
check('200 wide and 150 high: 75 of 1,040 split', c.divergent === '75' && c.warps === '1,040' && same(c, boundary2D({ width: 200, height: 150, blockX: 16, blockY: 16 }, 32)), JSON.stringify(c));
await loadCase('p150x200');
c = await counts();
check('150 wide and 200 high: 100 of 1,040 split, 9.6%', c.divergent === '100' && c.share === '9.6%', JSON.stringify(c));
await loadCase('v1003');
check('loading a vector switches back to one dimension', (await outOf(B, 'boundary', 'warps')) === '32' && !!(await page.$(`${B} [data-readout="split"]`)));
await press(B, '2D: a picture or a matrix', 'Dimensions');
await setInput(B, 'Width (columns)', 4096);
await setInput(B, 'Height (rows)', 4096);
c = await counts();
check('a large matrix is counted without being drawn', same(c, boundary2D({ width: 4096, height: 4096, blockX: 16, blockY: 16 }, 32)) && c.divergent === '0' && !!(await page.$(`${B} [data-undrawn]`)) && !(await page.$(`${B} svg[data-view="launch"]`)), JSON.stringify(c));
await setInput(B, 'blockDim.x', 64);
await setInput(B, 'blockDim.y', 64);
check('a block over the cap is refused', (await text(`${B} [data-over]`)).includes('4096 threads in a block is more than the 1024'), await page.$eval(B, (e) => e.textContent.slice(0, 200)));
await setInput(B, 'blockDim.x', 16);
await setInput(B, 'blockDim.y', 16);

// The erratum hunt.
const H = `${B} [data-hunt]`;
const pickHunt = async (i) => {
  await page.$$eval(`${H} input[type=radio]`, (rs, i) => rs[i].click(), i);
  await pause(60);
  await page.$eval(`${H} button.primary`, (x) => x.click());
  await pause(100);
};
await pickHunt(0);
check('the hunt: really split warps at 200 wide give 75, not 80', (await text(`${H} .verdict`)).startsWith('That gives 75, not 80.'), await text(`${H} .verdict`));
const huntCounts = await page.$$eval(`${H} [data-hunt-count]`, (es) => es.map((e) => Number(e.dataset.huntCount)));
const wide = { width: 200, height: 150, blockX: 16, blockY: 16 };
const tall = { width: 150, height: 200, blockX: 16, blockY: 16 };
check('every way of counting shows the model\'s number', eq(huntCounts, [boundary2D(wide, 32).divergent, boundary2D(tall, 32).divergent, warpsInRightEdgeBlocks(wide, 32), warpsInRightEdgeBlocks(tall, 32), boundary2D(wide, 64).divergent]) && eq(huntCounts, [75, 100, 80, 104, 50]), JSON.stringify(huntCounts));
await pickHunt(2);
check('counting every warp of the blocks on the right edge gives the printed 80', (await text(`${H} .verdict`)).startsWith('That gives 80.') && (await text(`${H} .verdict`)).includes('72 + 8 = 80'), await text(`${H} .verdict`));
await shot(H, 'hunt');

// ---------------------------------------------------------------- questions
const answerNumber = async (needle, value) => {
  const q = await question(needle);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(150);
  return q.evaluate((e) => e.textContent);
};
const answerChoice = async (needle, index) => {
  const q = await question(needle);
  await (await q.$$('input[type=radio]'))[index].click();
  await (await q.$('button.primary')).click();
  await pause(150);
  return q.evaluate((e) => e.textContent);
};
check('exercise 2 accepts 2048 threads', (await answerNumber('exercise 2', 2048)).includes('Correct.'));
check('the 150 by 200 picture rejects the book\'s 80 and gives 100', (await answerNumber('150 pixels wide and 200 high', 80)).includes('Not quite. The answer is 100 warps.'));
check('exercise 3: one warp diverges', (await answerChoice('exercise 3', 1)).includes('Correct.'));
check('AMD\'s example on a 64-wide device: a wrong option explains itself', (await answerChoice('whose warps are 64 wide', 0)).includes('Here the first boundary is at 64.'));
check('the causal mask: about 3%', (await answerChoice('attention scores', 1)).includes('992 split warps out of 32,768'));
const selectText = await answerChoice('compute both results', 0);
check('compute both and select: the model\'s 202 against 203', selectText.includes('Correct.') && selectText.includes('spends 202') && selectText.includes('spends 203'), selectText.slice(-400));
check('all eight questions of the section are on the page', (await page.$$eval('.q', (qs) => qs.filter((q) => q.querySelector('.kind')?.textContent.match(/Check your reasoning|Work it out/)).length)) === 8);

// ---------------------------------------------------------------- the lab 4.2 prediction
const P = () => question('before you run lab 4.2');
const commit = async (v) => {
  const q = await P();
  await (await q.$('input[type=number]')).type(String(v));
  await (await q.$('button.primary')).click();
  await pause(200);
  return q;
};
let q = await commit(2);
check('with no benchmark on this machine the prediction says what to run', (await q.evaluate((e) => e.querySelector('[data-measured]')?.dataset.measured)) === 'none' && (await q.evaluate((e) => e.textContent)).includes('python3 labs/pmpp.py run 4.2 --bench'), await q.evaluate((e) => e.textContent.slice(0, 200)));

fixture = true;
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector('.page[data-ready]');
await page.waitForSelector(D);
await pause(500);
q = await commit(3);
const measured = await q.evaluate((e) => { const m = e.querySelector('[data-measured]'); return m ? { backend: m.dataset.measured, mixed: m.dataset.mixed, text: m.textContent, patterns: [...m.querySelectorAll('tr[data-pattern]')].length } : null; });
check('with a report it shows the measured ratio beside the prediction', !!measured && measured.backend === 'cuda' && measured.mixed === '1.9' && measured.text.includes('e2e fixture device') && measured.text.includes('took 1.9 times as long') && measured.text.includes('You expected more.'), JSON.stringify(measured)?.slice(0, 300));
check('and the other patterns, the select kernel and the loop kernel', !!measured && measured.patterns === 4 && measured.text.includes('1.05 times as long as the branch') && measured.text.includes('1.7 times as long as with the same counts sorted'));
check('it says the numbers are measurements, and gives the page\'s pass count of 2', (await q.evaluate((e) => e.textContent)).includes('not expected values') && (await q.evaluate((e) => e.textContent)).includes('says 2 for long, equal paths'));
const widths = await page.$$eval(`${D} select[aria-label="Warp width"] option`, (os) => os.map((x) => x.textContent));
check('the simulator offers the widths the devices reported', widths.includes('yours: e2e fixture device (cuda), 32') && widths.includes('yours: e2e fixture wide device (hip), 64'), JSON.stringify(widths));
await setSelect(D, 'Warp width', 'hip');
check('choosing the 64-wide device draws 64 lanes and two warps in the block', (await page.$$eval(`${D} svg[data-view="lanes"] g[data-row="0"] rect[data-lane]`, (x) => x.length)) === 64 && (await warpButtons()).length === 2);
await press(B, '2D: a picture or a matrix', 'Dimensions');
await setSelect(B, 'Warp width of the launch', 'hip');
check('a 16 × 16 block fits that device\'s cap of 256 and is cut into four-row warps', (await outOf(B, 'boundary', 'divergent')) === '20');
await setInput(B, 'blockDim.x', 32);
check('a 32 × 16 block is over that device\'s cap and is refused', (await text(`${B} [data-over]`)).includes('more than the 256'));

// ---------------------------------------------------------------- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await setSelect(D, 'Warp width', 'book');
await shot(D, 'light-sim');
await shot(L, 'light-batch');
await setInput(B, 'blockDim.x', 16);
await shot(B, 'light-counter');
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(400);
const narrow = await page.evaluate((scopes) => ({
  page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1,
  benches: scopes.every((s) => document.querySelector(s).getBoundingClientRect().width <= document.documentElement.clientWidth),
}), [D, L, B]);
check('at a narrow width nothing makes the page scroll sideways', narrow.page && narrow.benches, JSON.stringify(narrow));
await shot(D, 'narrow-sim');
await shot(B, 'narrow-counter');
await shot(EX1, 'narrow-ex1');

// ---------------------------------------------------------------- errors
const expected404 =(u) => /\/content\/ch04\/narration\/(manifest\.json|warp-branch-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
