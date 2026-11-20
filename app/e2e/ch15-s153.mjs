// Browser check for section 15.3: the lab 15.1 listing and its three slices, the two
// predictions, the "Who owns what" explorer (one- and two-dimensional launches) and the
// loadTile stepper (both tiles, the presets, the vector-load toggle). Every number the page
// shows is compared with the model (src/sim/gemmmap.ts, imported directly).
// Usage: APP_URL=http://localhost:5173 node e2e/ch15-s153.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { launch2D, owns, slotOf, loadPlan, tileLoadSummary, warpLoad } from '../src/sim/gemmmap.ts';

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
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const O = bench('Who owns what');
const L = bench('loadTile, one sub-tile at a time');

/** Set a React-controlled input or select by its label text (or aria-label), inside `scope`. */
async function setField(scope, label, value) {
  const ok = await page.evaluate((scope, label, value) => {
    const root = document.querySelector(scope);
    let el = [...root.querySelectorAll('input, select')].find((e) => e.getAttribute('aria-label') === label);
    if (!el) {
      const lab = [...root.querySelectorAll('label.field')].find((l) => l.textContent.trim().startsWith(label));
      el = lab?.querySelector('input[type=number], select');
    }
    if (!el) return false;
    const proto = el.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, String(value));
    el.dispatchEvent(new Event(el.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true }));
    return true;
  }, scope, label, value);
  await pause(120);
  return ok;
}
async function click(scope, label, group) {
  const ok = await page.evaluate((scope, label, group) => {
    const root = document.querySelector(scope);
    const where = group ? root.querySelector(`[aria-label="${group}"]`) : root;
    const b = [...where.querySelectorAll('button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label, group);
  await pause(120);
  return ok;
}
async function tick(scope, label, on) {
  await page.evaluate((scope, label, on) => {
    const box = [...document.querySelector(scope).querySelectorAll('label.check')].find((l) => l.textContent.includes(label)).querySelector('input');
    if (box.checked !== on) box.click();
  }, scope, label, on);
  await pause(150);
}
const vals = (scope) => page.$$eval(scope + ' [data-v]', (els) => Object.fromEntries(els.map((e) => [e.dataset.v, e.textContent.trim()])));

await goto(page, 'ch15/15.3');
await page.waitForSelector(O);
await page.waitForSelector('.code .tabs button');
await pause(300);

// ---- the listing and the slices are the lab's files
const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent.trim()));
check('the listing has CUDA, Metal kernel and Metal host tabs', ['CUDA', 'Metal kernel', 'Metal host'].every((t) => tabs.includes(t)), tabs.join('|'));
const first = await text('.code pre');
check('the CUDA tab is lab 15.1\'s reference solution', first.includes('gemmCoarseKernel') && first.includes('launchGemmCoarse'));
const statuses = await page.$$eval('.code .tabs .status', (s) => s.map((x) => x.textContent.trim()));
check('every listing carries a status in the course vocabulary', statuses.length >= 4 && statuses.every((s) => /emulator|verified on Metal|unverified|no run/.test(s) && !/\bverified on CUDA\b|\bverified on HIP\b/.test(s.replace(/unverified on CUDA and HIP hardware|unverified on (CUDA|HIP)/g, ''))), statuses.join(' | '));
const slices = await page.$$eval('figure.code', (fs) => fs.map((f) => ({
  tab: f.querySelector('.tabs button')?.textContent.trim(),
  firstN: Number(f.querySelector('.ln')?.dataset.n),
  firstLine: f.querySelector('.ln')?.textContent ?? '',
  last: [...f.querySelectorAll('.ln')].at(-1)?.textContent.trim(),
})));
for (const fn of ['loadTile', 'mm', 'writeTile']) {
  const s = slices.find((x) => x.tab === `${fn}, lab 15.1`);
  check(`the ${fn} slice starts at its definition with the file's line number and ends at its brace`,
    !!s && s.firstN > 30 && new RegExp(`void ${fn}\\(`).test(s.firstLine) && s.last === '}', JSON.stringify(s));
}
await page.evaluate(() => [...document.querySelectorAll('.code .tabs button')].find((b) => b.textContent.trim() === 'Metal kernel').click());
await pause(150);
check('the Metal tab is the Metal kernel', (await text('.code pre')).includes('kernel void gemm_coarse'));

// ---- predictions
async function predict(id, value) {
  const q = await page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(t)), id);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(150);
  return q.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
}
const t17 = owns(1024, 1024, 2, 1, 17);
const t255 = owns(1024, 1024, 2, 1, 255);
let r = await predict('first row of C that thread 17 owns', 136);
check('prediction for thread 17 reveals row 136 and columns 264 to 271', r.includes(`Row ${t17.rows[0]}.`) && r.includes(`${t17.cols[0]} to ${t17.cols[1]}`) && r.includes('You had it.'), r);
r = await predict('last column of C that thread 255 owns', 300);
check('prediction for thread 255 reveals column 383 and corrects a wrong guess', r.includes(`Column ${t255.cols[1]}.`) && !r.includes('You had it.'), r);

// ---- explorer: one-dimensional block
await page.$eval(O, (e) => e.scrollIntoView({ block: 'start' }));
check('block (2, 1) can be selected', await setField(O, 'block', '2,1'));
await setField(O, 'threadIdx.x value', 17);
let v = await vals(O + ' [data-readout]');
check('thread 17 of block (2, 1): bRow 128, bCol 256, ty 1, tx 1, tRow 8, tCol 8',
  v.bRow === String(t17.bRow) && v.bCol === String(t17.bCol) && v.ty === '1' && v.tx === '1' && v.tRow === '8' && v.tCol === '8', JSON.stringify(v));
let owned = await text(O + ' [data-owns]');
check('it owns rows 136 to 143 and columns 264 to 271, and reads A_s rows 8 to 15 and B_s columns 8 to 15',
  owned.includes('rows 136 to 143') && owned.includes('columns 264 to 271') && owned.includes('rows 8 to 15 of A_s') && owned.includes('columns 8 to 15 of B_s'), owned);
check('the block streams rows 128 to 255 of A and columns 256 to 383 of B', owned.includes('rows 128 to 255 of A') && owned.includes('columns 256 to 383 of B'), owned);
const mine = await page.$$eval(O + ' [data-view="block"] [data-state="mine"]', (e) => e.map((x) => x.dataset.p));
check('the block view marks patch 17 as the thread\'s', mine.length === 1 && mine[0] === '17', mine.join(','));
const strip = await page.$eval(O + ' [data-strip="A_s"]', (e) => Number(e.getAttribute('y')));
const tile0 = await page.$eval(O + ' [data-view="block"] [data-p="0"]', (e) => Number(e.getAttribute('y')));
check('the A_s strip sits at tRow (8 rows = 16 px below the tile top)', strip - tile0 === 16, `${strip} vs ${tile0}`);
await setField(O, 'threadIdx.x value', 255);
owned = await text(O + ' [data-owns]');
check('thread 255 owns the bottom-right patch: rows 248 to 255, columns 376 to 383', owned.includes(`rows ${t255.rows[0]} to ${t255.rows[1]}`) && owned.includes(`columns ${t255.cols[0]} to ${t255.cols[1]}`), owned);
if (out) await page.screenshot({ path: out + '-owners.png' });

// edges
await setField(O, 'M (rows of C)', 130);
await setField(O, 'block', '0,1');
await setField(O, 'threadIdx.x value', 17);
owned = await text(O + ' [data-owns]');
check('M = 130, block (0, 1), thread 17: the patch lies outside C and nothing is stored', owns(130, 1024, 0, 1, 17).writes === 0 && owned.includes('writeTile stores nothing'), owned);
await setField(O, 'threadIdx.x value', 0);
owned = await text(O + ' [data-owns]');
check('thread 0 there stores only 2 × 8 = 16 elements', owns(130, 1024, 0, 1, 0).writes === 16 && owned.includes('2 × 8 = 16'), owned);
check('the block view shades what lies outside C', (await text(O + ' [data-view="block"]')).includes('outside C (2 rows, 128 columns inside)'));

// two-dimensional launch
const l2 = launch2D(16, 16);
await tick(O, 'launch dim3(16, 16) instead', true);
await setField(O, 'threadIdx.x value', 5);
await setField(O, 'threadIdx.y value', 9);
const two = await text(O + ' [data-twod]');
check('dim3(16, 16): 16 of 256 patches written, each by 16 threads', two.includes(`${l2.patchesWritten} of 256 patches written`) && two.includes(`each by ${l2.writers[0]} threads`), two);
check('rows 8 to 127 never written; 64 of 1024 slots of each tile loaded; no patch has all its inputs',
  two.includes(`(${l2.rowsNeverWritten} rows)`) && two.includes(`${l2.aLoadedCount} of 1024 slots of A_s`) && two.includes(`${l2.bLoadedCount} of 1024 of B_s`) && two.includes(`all loaded: ${l2.correctPatches}`), two);
const states = await page.$$eval(O + ' [data-view="block"] [data-p]', (e) => e.map((x) => x.dataset.state));
check('the block view marks 16 patches written and 240 never written', states.filter((s) => s === 'written').length === 16 && states.filter((s) => s === 'never').length === 240);
const readout = await text(O + ' [data-readout]');
check('thread (x 5, y 9) behaves as thread 5', readout.includes('exactly what thread 5 of the intended block does') && (await vals(O + ' [data-readout]')).tx === '5');
if (out) await page.screenshot({ path: out + '-2d.png' });
await tick(O, 'launch dim3(16, 16) instead', false);

// ---- stepper: loadTile
await page.$eval(L, (e) => e.scrollIntoView({ block: 'start' }));
const P = { M: 1024, N: 1024, K: 1024 };
v = await vals(L + ' [data-warp-readout]');
let w = warpLoad('A', P, 0, 0, 0, 0, 0);
check('A tile, warp 0: 4 sectors in 4 runs, no zero fill', v.sectors === String(w.sectors.length) && v.runs === String(w.runs.length) && v.zero === '0' && w.sectors.length === 4, JSON.stringify(v));
let strips = await page.$$eval(L + ' [data-strip] [data-sector]', (e) => e.length);
check('the memory strip draws those 4 sectors', strips === 4, String(strips));
await click(L, 'Step');
await click(L, 'Step');
check('after two steps the stepper is on sub-tile 2', (await text(L + ' [data-sub]')).startsWith('sub-tile 2 of 4'));
const at = slotOf(loadPlan(128, 8, 256), 77, 2);
const t77 = await page.$eval(L + ` [data-cell="${at.row},${at.col}"]`, (e) => e.dataset.t).catch(() => null);
check('in sub-tile 2, element (73, 5) of the A tile is loaded by thread 77', at.row === 73 && at.col === 5 && t77 === '77', String(t77));
await click(L, 'warp 2', 'Warp');
v = await vals(L + ' [data-warp-readout]');
w = warpLoad('A', P, 0, 0, 0, 2, 2);
check('sub-tile 2, warp 2: the model\'s sectors and runs', v.sectors === String(w.sectors.length) && v.runs === String(w.runs.length), JSON.stringify(v));
let summ = await vals(L + ' [data-summary]');
let model = tileLoadSummary('A', P, 0, 0, 0);
check('whole A tile: 128 sectors, nothing diverges', summ['total-sectors'] === String(model.sectors) && summ.divergent === '0' && model.sectors === 128, JSON.stringify(summ));

await click(L, 'B tile (8 × 128)', 'Tile');
v = await vals(L + ' [data-warp-readout]');
w = warpLoad('B', P, 0, 0, 0, 0, 2);
check('B tile, warp 2: 4 sectors in one run', v.sectors === '4' && v.runs === '1' && w.runs.length === 1, JSON.stringify(v));
if (out) await page.screenshot({ path: out + '-loadtile-b.png' });

// unaligned K
await click(L, 'K = 1001', 'Presets');
await click(L, 'A tile (128 × 8)', 'Tile');
const P2 = { M: 1024, N: 1024, K: 1001 };
const counts = [];
for (let k = 0; k < 8; k++) {
  await click(L, `warp ${k}`, 'Warp');
  counts.push(Number((await vals(L + ' [data-warp-readout]')).sectors));
}
const expected = Array.from({ length: 8 }, (_, k) => warpLoad('A', P2, 0, 0, 0, 0, k).sectors.length);
check('K = 1001: every A warp\'s sector count matches the model, and some need 8', JSON.stringify(counts) === JSON.stringify(expected) && Math.max(...counts) === 8, JSON.stringify({ counts, expected }));

// edge block
await click(L, 'Edge block: M = 130, K = 9', 'Presets');
await click(L, 'warp 0', 'Warp');
const P3 = { M: 130, N: 256, K: 9 };
v = await vals(L + ' [data-warp-readout]');
w = warpLoad('A', P3, 0, 1, 1, 0, 0);
check('edge block, A, warp 0: 30 of 32 lanes zero-fill and the warp diverges', v.zero === String(w.zeroLanes) && w.zeroLanes === 30 && 'diverges' in v, JSON.stringify(v));
const zeros = await page.$$eval(L + ' [data-detail] [data-zero="1"]', (e) => e.length);
check('the detail marks the zero-filled cells of this sub-tile', zeros === 256 - 2, String(zeros));
summ = await vals(L + ' [data-summary]');
model = tileLoadSummary('A', P3, 0, 1, 1);
check('whole A tile of the edge block: 1 divergent warp instruction', summ.divergent === String(model.divergent) && model.divergent === 1, JSON.stringify(summ));
await click(L, 'B tile (8 × 128)', 'Tile');
summ = await vals(L + ' [data-summary]');
check('the B tile of the same block: no divergence (the K edge is warp-uniform for B)', summ.divergent === String(tileLoadSummary('B', P3, 0, 1, 1).divergent) && summ.divergent === '0', JSON.stringify(summ));
if (out) await page.screenshot({ path: out + '-edge.png' });

// vector loads
await click(L, 'Aligned: M = N = K = 1024', 'Presets');
await tick(L, 'vector loads (Exercise 1)', true);
check('vector loads: one sub-tile', (await text(L + ' [data-sub]')).startsWith('sub-tile 0 of 1'));
v = await vals(L + ' [data-warp-readout]');
w = warpLoad('B', P, 0, 0, 0, 0, 0, undefined, 'float4');
check('vector loads, B tile: 16 sectors in one run per warp', v.sectors === String(w.sectors.length) && v.runs === '1' && w.sectors.length === 16, JSON.stringify(v));
await click(L, 'A tile (128 × 8)', 'Tile');
await click(L, 'warp 3', 'Warp');
v = await vals(L + ' [data-warp-readout]');
check('vector loads, A tile: 16 sectors in 16 runs (16 rows of A)', v.sectors === '16' && v.runs === '16', JSON.stringify(v));
const groups = await page.$$eval(L + ' [data-detail] [data-t]', (e) => e.length);
check('the detail shows one float4 per thread: 64 groups in a 32-row window', groups === 64, String(groups));
await click(L, 'K = 1001', 'Presets');
check('vector loads with K = 1001 flag the misaligned addresses', (await text(L + ' [data-warp-readout]')).includes('not 16-byte aligned') && (await page.$(L + ' .narrate.bad')) !== null);
if (out) await page.screenshot({ path: out + '-vector.png' });
await tick(L, 'vector loads (Exercise 1)', false);

// ---- questions
async function numeric(prompt, value) {
  const q = await page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(t)), prompt);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(150);
  return q.evaluate((e) => e.textContent.includes('Correct.'));
}
async function choice(prompt, optionText) {
  const q = await page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(t)), prompt);
  await q.evaluate((e, o) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes(o)).querySelector('input').click(), optionText);
  await pause(80);
  await (await q.$('button.primary')).click();
  await pause(150);
  return q.evaluate((e) => e.textContent.includes('Correct.'));
}
check('question: thread 77 in sub-tile 2 loads row 73', await numeric('Which row of the tile does thread 77 load', 73));
check('question: K = 1001 zero-fills 7 columns', await numeric('K = 1001. In the last tile pair', 7));
check('question: float4 loads, 2 instructions per thread per tile pair', await numeric('How many load instructions', 2));
check('question: a 16 x 16 launch duplicates the top row of patches', await choice('so you launch the kernel with', 'ty is always 0'));
check('question: the batch-1 layer launches 32 blocks against 256', await choice('Lab 5.3\'s benchmark layer', '32 against 256'));
check('question: sectors of an A and a B load', await choice('Counting NVIDIA\'s 32-byte sectors', 'Both touch four sectors'));
check('a wrong answer is marked wrong', !(await choice('You delete the', 'The kernel no longer compiles')));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
