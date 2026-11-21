// Browser check for section 10.7: the listings, the shuffle explorer, the warp_reduce
// stepper, the hand-off explorer and two questions. Every number the page shows is
// compared with the model it runs on (src/sim/shuffle.ts, imported here directly).
// The narrated scene has its own check (e2e/ch10-scene-shuffle.mjs).
// Usage: node e2e/ch10-s107.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { callText, hybrid, ones, shuffle, smallValues, warpReduceTrace } from '../src/sim/shuffle.ts';

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
const lanes = (n) => Array.from({ length: n }, (_, i) => i);
const bench = (t) => `section.bench[aria-label="${t}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const clickText = async (scope, label) => {
  const done = await page.evaluate((sel, label) => {
    const btn = [...document.querySelector(sel).querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, bench(scope), label);
  await pause(80);
  return done;
};
const setSelect = async (scope, aria, value) => {
  await page.$eval(`${bench(scope)} select[aria-label="${aria}"]`, (s, v) => {
    s.value = String(v);
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await pause(100);
};
const setRange = async (scope, aria, value) => {
  await page.$eval(`${bench(scope)} input[aria-label="${aria}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(100);
};
const setCheck = async (scope, word, on) => {
  await page.evaluate((sel, word, on) => {
    const box = [...document.querySelector(sel).querySelectorAll('label.check')].find((l) => l.textContent.includes(word)).querySelector('input');
    if (box.checked !== on) box.click();
  }, bench(scope), word, on);
  await pause(100);
};

await goto(page, 'ch10/10.7');
await page.waitForSelector(bench('One shuffle across a warp'));

// ---- listings
await page.waitForFunction(() => [...document.querySelectorAll('.code .tabs button')].some((b) => b.textContent === 'portable (lab 10.1)'), { timeout: 10000 }).catch(() => {});
const tabNames = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent));
check('warp_reduce is shown as printed, portable and for Metal', ['as printed (CUDA only)', 'portable (lab 10.1)', 'Metal (lab 10.1)'].every((t) => tabNames.includes(t)), tabNames.join('|'));
const codeOf = async (label) => page.evaluate((label) => {
  const fig = [...document.querySelectorAll('figure.code')].find((f) => [...f.querySelectorAll('.tabs button')].some((b) => b.textContent === label));
  [...fig.querySelectorAll('.tabs button')].find((b) => b.textContent === label).click();
  return new Promise((r) => setTimeout(() => r(fig.querySelector('pre').textContent), 80));
}, label);
const printed = await codeOf('as printed (CUDA only)');
check('the as-printed tab is the book\'s loop with the 32-bit mask', printed.includes('#define WARP_SIZE 32') && printed.includes('__shfl_down_sync(0xffffffff, partialSum, stride)'));
const portable = await codeOf('portable (lab 10.1)');
check('the portable tab has a per-vendor mask type and starts at warpSize / 2', portable.includes('typedef unsigned long long lane_mask_t') && portable.includes('warpSize / 2') && !/0xffffffff/.test(portable));
const metal = await codeOf('Metal (lab 10.1)');
check('the Metal tab uses simd_shuffle_down with no mask', metal.includes('simd_shuffle_down(partialSum') && !metal.includes('mask'));
const hybridCode = await codeOf('Fig. 10.14, corrected');
const hybridLines = hybridCode.replace(/\n$/, '').split('\n');
check('the corrected Fig. 10.14 keeps the book\'s 19 lines, the barrier on line 11 and the write on lines 15 to 17',
  hybridLines.length === 19 && hybridLines[10].trim() === '__syncthreads();' && hybridLines[14].includes('threadIdx.x == 0') && hybridLines[15].includes('*output = partialSum') && hybridLines[4].includes('stride >= (unsigned int)warpSize'),
  JSON.stringify([hybridLines.length, hybridLines[10], hybridLines[15]]));

// ---- shuffle explorer
const X = 'One shuffle across a warp';
const outCells = () => page.$$eval(`${bench(X)} rect[data-outcome]`, (rs) => rs.map((r) => [r.dataset.outcome, r.dataset.value]));
const expect = (res) => res.map((r) => [r.outcome, r.outcome === 'absent' ? '' : r.value == null ? '?' : String(r.value)]);
const clickLane = async (lane) => {
  await page.$eval(`${bench(X)} rect[data-in-lane="${lane}"]`, (r) => r.dispatchEvent(new MouseEvent('click', { bubbles: true })));
  await pause(100);
};
const mode = (label) => clickText(X, label);
const full32 = new Array(32).fill(true);

check('default call is the down shuffle by 16 on a 32-lane warp', (await text(`${bench(X)} [data-call]`)) === callText('down', 16, 32, full32), await text(`${bench(X)} [data-call]`));
check('every lane\'s result matches the model: lanes 0 to 15 get 16 to 31, the top 16 keep their own', same(await outCells(), expect(shuffle('down', lanes(32), 16))));
check('lane 20 is explained: past the end of the warp, keeps 20', (await text(`${bench(X)} [data-explain]`)).includes('past the end of the warp') && (await text(`${bench(X)} [data-explain]`)).includes('keeps its own value, 20'), await text(`${bench(X)} [data-explain]`));
check('the tally counts 16 moved and 16 kept', (await text(`${bench(X)} [data-tally]`)).startsWith('16 lanes received another lane\'s value, 16 lanes kept their own, 0 lanes got an undefined value, 0 lanes did not call'));
if (out) await page.$eval(bench(X), (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-shuffle-down16.png' });

await setRange(X, 'delta', 2);
await setSelect(X, 'width argument', 8);
check('width 8, delta 2: the call shows the width and the results match the model', (await text(`${bench(X)} [data-call]`)) === callText('down', 2, 8, full32) && same(await outCells(), expect(shuffle('down', lanes(32), 2, { width: 8 }))));
await clickLane(14);
let ex = await text(`${bench(X)} [data-explain]`);
check('lane 14 is position 6 of the group 8 to 15 and keeps 14', ex.includes('position 6 of its group, lanes 8 to 15') && ex.includes('keeps its own value, 14'), ex);
await clickLane(13);
ex = await text(`${bench(X)} [data-explain]`);
check('lane 13 receives 15', ex.includes('It receives 15'), ex);
if (out) await page.screenshot({ path: out + '-shuffle-width8.png' });

// lanes that return, and lanes out of the mask
await setSelect(X, 'width argument', 32);
await setRange(X, 'delta', 16);
await mode('makes it return');
await clickLane(20);
const exited = lanes(32).map((i) => i === 20);
check('a lane that returned gives an undefined value to the lane that reads it', same(await outCells(), expect(shuffle('down', lanes(32), 16, { exited }))) && (await outCells())[4][0] === 'undefined');
await mode('explains it');
await clickLane(4);
ex = await text(`${bench(X)} [data-explain]`);
check('and the explanation says lane 20 has returned', ex.includes('lane 20 has returned') && ex.includes('undefined'), ex);
await clickText(X, 'All lanes back');
await setRange(X, 'delta', 1);
await mode('takes it out of the mask');
await clickLane(9);
const mask = lanes(32).map((i) => i !== 9);
check('a lane outside the mask: its reader gets an undefined value, and the mask literal changes', same(await outCells(), expect(shuffle('down', lanes(32), 1, { mask }))) && (await text(`${bench(X)} [data-call]`)) === callText('down', 1, 32, mask), await text(`${bench(X)} [data-call]`));
await clickText(X, 'All lanes back');
await mode('explains it');

// xor and by source lane
await setSelect(X, 'shuffle function', 'xor');
await setRange(X, 'laneMask', 8);
await setSelect(X, 'width argument', 8);
check('xor 8 with width 8: lane 3 keeps its own (later group), lane 11 reads lane 3 (earlier group)',
  same(await outCells(), expect(shuffle('xor', lanes(32), 8, { width: 8 }))) && (await outCells())[3][0] === 'own' && (await outCells())[11][1] === '3');
await clickLane(3);
check('the explanation says xor cannot read a later group', (await text(`${bench(X)} [data-explain]`)).includes('later group'));
await setSelect(X, 'shuffle function', 'idx');
await setRange(X, 'srcLane', 10);
check('by source lane 10 with width 8: lane 3 reads lane 2 of its own group', same(await outCells(), expect(shuffle('idx', lanes(32), 10, { width: 8 }))) && (await outCells())[3][1] === '2');
await clickLane(3);
check('and the explanation takes 10 modulo 8', (await text(`${bench(X)} [data-explain]`)).includes('taken modulo 8'));

// other warp widths
await setSelect(X, 'shuffle function', 'down');
await setSelect(X, 'warp width', 'w64');
await setRange(X, 'delta', 16);
const full64 = new Array(64).fill(true);
check('a 64-lane warp: 64 results, a 64-bit mask, and the model\'s values', (await outCells()).length === 64 && (await text(`${bench(X)} [data-call]`)) === callText('down', 16, 64, full64) && same(await outCells(), expect(shuffle('down', lanes(64), 16))), await text(`${bench(X)} [data-call]`));
if (out) await page.screenshot({ path: out + '-shuffle-64.png' });
await setSelect(X, 'warp width', 'toy');
check('the 8-lane toy has 8 lanes', (await outCells()).length === 8);
await setSelect(X, 'warp width', 'book');

// ---- warp_reduce stepper
const S = 'warp_reduce, step by step';
const rows = () => page.$$eval(`${bench(S)} svg[data-stepper-grid] g[data-row]`, (gs) => gs.map((g) => g.dataset.values));
const traceRows = (t, upto) => [t.initial.map((v, i) => (t.live[i] ? (v == null ? '?' : String(v)) : '-')).join(' '),
  ...t.steps.slice(0, upto).map((s) => s.lanes.map((l) => (l.live ? (l.after == null ? '?' : String(l.after)) : '-')).join(' '))];
const counts = () => text(`${bench(S)} [data-counts]`);
const down = warpReduceTrace(smallValues(32), { pattern: 'down' });
check('stepper starts with the starting values only', same(await rows(), traceRows(down, 0)));
await clickText(S, 'Step');
const n1 = await text(`${bench(S)} [data-narrate]`);
check('step 1 is stride 16: 16 lanes receive, 16 add their own, 16 of 32 additions reach lane 0',
  n1.startsWith('Stride 16') && n1.includes('16 lanes received') && n1.includes('16 lanes had no lane 16 above them') && n1.includes('16 of these 32 additions reach lane 0'), n1);
check('after step 1 the rows are the model\'s', same(await rows(), traceRows(down, 1)));
for (let i = 0; i < 4; i++) await clickText(S, 'Step');
check('five steps: every row matches the model', same(await rows(), traceRows(down, 5)));
let c = await counts();
check('160 lane-operations, 31 reach lane 0, 129 wasted, and lane 0 holds 162', c.includes('Lane-operations so far: 160; reaching lane 0: 31; wasted: 129') && c.includes('Lane 0 holds 162, the sum of all 32 values') && down.lane0 === 162, c);
check('cone cells: 32 starting values and 31 lane-steps', (await page.$$(`${bench(S)} svg[data-stepper-grid] g[data-cone="1"]`)).length === 63);
if (out) await page.$eval(bench(S), (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-stepper-down.png' });

await clickText(S, 'xor butterfly');
await clickText(S, 'Run to the end');
const xor = warpReduceTrace(smallValues(32), { pattern: 'xor' });
c = await counts();
check('xor: same 160 lane-operations, all needed, every lane holds 162', same(await rows(), traceRows(xor, 5)) && c.includes('Lane-operations so far: 160; reaching every lane: 160; wasted: 0') && c.includes('Every lane holds 162'), c);

await clickText(S, 'down (Fig. 10.13)');
await setSelect(S, 'case', 'wave64');
await setSelect(S, 'values', 'ones');
await clickText(S, 'Run to the end');
const w64 = warpReduceTrace(ones(64), { pattern: 'down', loopWidth: 32 });
c = await counts();
check('the 32-lane loop on 64 lanes: 5 steps, lane 0 and lane 32 hold 32 each, marked wrong',
  same(await rows(), traceRows(w64, 5)) && c.includes('Lane 0 holds 32 and lane 32 holds 32. The sum is 64') && (await page.$eval(`${bench(S)} [data-result]`, (e) => e.dataset.result)) === 'fail', c);
if (out) await page.screenshot({ path: out + '-stepper-wave64.png' });

await setSelect(S, 'case', 'return');
await setSelect(S, 'values', 'small');
await setRange(S, 'n', 20);
await clickText(S, 'Run to the end');
const gone = warpReduceTrace(smallValues(32), { pattern: 'down', tail: { from: 20, mode: 'return' } });
c = await counts();
check('lanes from 20 returned: rows match the model, question marks appear, lane 0 is undefined',
  same(await rows(), traceRows(gone, 5)) && (await rows())[2].split(' ')[0] === '?' && c.includes('Lane 0\'s result is undefined') && (await page.$eval(`${bench(S)} [data-result]`, (e) => e.dataset.result)) === 'fail', c);
if (out) await page.screenshot({ path: out + '-stepper-return.png' });
await setSelect(S, 'case', 'zero');
await clickText(S, 'Run to the end');
c = await counts();
check('contributing 0 instead: lane 0 holds 96, the sum of lanes 0 to 19', c.includes('Lane 0 holds 96, the sum of lanes 0 to 19') && (await page.$eval(`${bench(S)} [data-result]`, (e) => e.dataset.result)) === 'ok', c);

// ---- hand-off explorer
const H = 'Where to hand over';
const cell = (k) => text(`${bench(H)} [data-k="${k}"]`);
const slotStates = () => page.$$eval(`${bench(H)} g[data-slot]`, (gs) => gs.map((g) => g.dataset.state));
const tally = (arr) => arr.reduce((a, s) => ({ ...a, [s]: (a[s] ?? 0) + 1 }), {});
let h = hybrid(ones(2048), 1024, 32, 'printed');
check('as printed at 1024 / 32: strides 512 to 32, 6 barriers, 2016 / 2016 shared accesses, 5 shuffles, 2048',
  (await cell('strides')) === '512, 256, 128, 64, 32' && (await cell('barriers')) === '6' && (await cell('shared')) === `${h.sharedWrites} / ${h.sharedReads}` && (await cell('shuffles')) === '5' && (await cell('result')) === '2048');
check('the hand-off strip: 32 slots read and live, 32 already added into lower slots', same(tally(await slotStates()), { used: 32, idle: 32 }), JSON.stringify(tally(await slotStates())));
await setCheck(H, 'line 11', false);
check('without line 11 the printed hand-off is still right, and the page says why', (await cell('result')) === '2048' && (await cell('barriers')) === '5' && (await text(`${bench(H)} [data-handoff-says]`)).includes('reads only a slot it wrote itself'));
await setCheck(H, 'line 11', true);
await setSelect(H, 'hand-off', 'early');
h = hybrid(ones(2048), 1024, 32, 'early');
check('one level early: half the sum, 32 slots dropped', (await cell('result')) === String(h.result) && h.result === 1024 && tally(await slotStates()).dropped === 32);
if (out) await page.$eval(bench(H), (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-handoff-early.png' });
await setSelect(H, 'hand-off', 'double');
h = hybrid(ones(2048), 1024, 32, 'double');
check('shared loop to 1 and warp_reduce too: counts stale slots again', (await cell('result')) === String(h.result) && h.result > 2048 && tally(await slotStates()).again === 31, JSON.stringify(tally(await slotStates())));
await setSelect(H, 'hand-off', 'early2');
check('one level early with two slots per lane: right with line 11', (await cell('result')) === '2048');
await setCheck(H, 'line 11', false);
check('and a data race without it', (await cell('result')) === 'undefined' && (await text(`${bench(H)} [data-handoff-says]`)).includes('data race'), await text(`${bench(H)} [data-handoff-says]`));
await setCheck(H, 'line 11', true);
await setSelect(H, 'hand-off', 'printed');
await setSelect(H, 'block size', 64);
await setSelect(H, 'warp width', 'w64');
h = hybrid(ones(128), 64, 64, 'printed');
check('a block of one 64-wide warp: no shared iterations, one barrier, right answer', (await cell('strides')) === 'none' && (await cell('barriers')) === String(h.barriers) && h.barriers === 1 && (await cell('result')) === '128');

// ---- questions
const answer = async (needle, value) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  if (typeof value === 'number') await (await q.$('input[type=number]')).type(String(value));
  else await q.evaluate((e, v) => [...e.querySelectorAll('.opts label')].find((l) => l.textContent.includes(v)).querySelector('input').click(), value);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
check('q-top-lanes accepts 20', (await answer('What does lane 20 hold afterwards', 20)).includes('Correct.'));
check('q-line11 accepts "each lane reads the slot it wrote"', (await answer('is the __syncthreads() on line 11', 'loads only the slot it wrote in the last iteration')).includes('Correct.'));
check('q-wave64 rejects 64 and gives 32', (await answer('What does lane 0 return?', 64)).includes('The answer is 32'));
check('q-hip-mask: the compile-time answer with a per-vendor mask type is right', (await answer('smallest change that keeps one source', 'At compile time, on every AMD target')).includes('Correct.'));
check('q-syncwarp: "the right sum, a warp issues one instruction" is wrong and says why', (await answer('An older tutorial finishes a block sum', 'a warp issues one instruction for all its lanes')).includes('names this very pattern'));

// ---- the backends table and a narrow screen
check('the backends table has the seven rows', (await page.$$('table[data-backends] tbody tr')).length === 7);
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
check('at a narrow width nothing makes the page scroll sideways', overflow <= 1, String(overflow));
await setSelect(X, 'warp width', 'book');
const lane20 = await page.$eval(`${bench(X)} rect[data-lane="20"]`, (r) => { const b = r.getBoundingClientRect(); const box = r.closest('section').getBoundingClientRect(); return { l: b.left, r: b.right, boxL: box.left, boxR: box.right, vw: innerWidth }; });
const lanesRows = await page.$eval(`${bench(X)} svg[data-rows]`, (e) => e.dataset.rows);
check('narrow: the explorer folds a 32-lane warp into 2 rows and lane 20 is on screen', lanesRows === '2' && lane20.l >= lane20.boxL && lane20.r <= Math.min(lane20.boxR, lane20.vw), JSON.stringify({ lanesRows, lane20 }));
if (out) {
  await page.$eval(bench(X), (e) => e.scrollIntoView());
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (u) => /\/content\/ch10\/narration\/(manifest\.json|shuffle-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
