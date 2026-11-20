// Browser check for section 6.1: the coalescing explorer and its presets, stepping k,
// the corner-turning stepper in both arrangements, the exercise 1 drill, two
// questions, both predictions, and the "your device" panel (against this machine's
// /api/reports, against a fixture report, and with no reports at all). Every
// expected number comes from the model the page runs on (src/sim/coalesce.ts).
// Usage: node e2e/ch06-s61.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { exercise1, exprAccess, tileLoad, strideEfficiency } from '../src/sim/coalesce.ts';
import { tileAccess } from '../src/sim/banks.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file; optionally replace the reports.
let fakeReports = null;
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.method() === 'PUT' && r.url().includes('/api/progress')) return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  if (fakeReports && r.method() === 'GET' && r.url().includes('/api/reports')) return r.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(fakeReports) });
  return r.continue();
});
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const clickText = async (scope, label) => {
  const done = await page.evaluate((sel, label) => {
    const root = document.querySelector(sel);
    const btn = [...root.querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, bench(scope), label);
  await pause(80);
  return done;
};
const select = async (scope, field, value) => {
  const ok = await page.evaluate((sel, field, value) => {
    const l = [...document.querySelectorAll(sel + ' label.field')].find((x) => x.textContent.trim().startsWith(field));
    const s = l?.querySelector('select');
    if (!s) return false;
    const set = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set;
    set.call(s, String(value));
    s.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  }, bench(scope), field, value);
  await pause(80);
  return ok;
};
const slide = async (scope, aria, value) => {
  await page.$eval(`${bench(scope)} input[aria-label="${aria}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(80);
};
const readout = (scope) => page.$eval(`${bench(scope)} [data-readout]`, (e) => ({ ...e.dataset }));
const P = (o = {}) => ({ Width: 1024, k: 0, row: 0, stride: 1, offset: 0, warpWidth: 32, lane0: 0, ...o });

await goto(page, 'ch06/6.1');
const E = 'Coalescing explorer';
await page.waitForSelector(bench(E));

// ---- explorer presets, each against the model
const presets = [
  ['Fig. 6.2: row-major N', exprAccess('n-row', P(), 32), 4],
  ['Fig. 6.3: column-major N', exprAccess('n-col', P(), 32), 32],
  ['The M read', exprAccess('m-row', P(), 32), 1],
  ['One float off', exprAccess('offset', P({ offset: 1 }), 32), 5],
];
for (const [label, model, want] of presets) {
  await clickText(E, label);
  const r = await readout(E);
  check(`preset "${label}": ${want} sectors, as the model says`, Number(r.transactions) === want && model.transactions === want && Number(r.needed) === model.bytesNeeded, JSON.stringify(r));
}
await clickText(E, 'Stride 2');
let r = await readout(E);
check('preset "Stride 2": 50 % of fetched bytes used', Number(r.efficiency) === 0.5 && (await text(`${bench(E)} [data-readout]`)).includes('50 % of fetched bytes used'), JSON.stringify(r));
await slide(E, 'stride', 8);
r = await readout(E);
check('stride 8: 32 sectors at 12.5 %', Number(r.transactions) === 32 && Number(r.efficiency) === 0.125, JSON.stringify(r));
await select(E, 'Element size', 16);
await slide(E, 'stride', 1);
r = await readout(E);
check('float4 elements at stride 1: 16 sectors (Nsight\'s 128-bit optimum)', Number(r.transactions) === exprAccess('stride', P(), 32, 16).transactions && Number(r.transactions) === 16, JSON.stringify(r));

// ---- stepping k on the column-major read: totals grow eightfold apart
await clickText(E, 'Fig. 6.3: column-major N');
for (let i = 0; i < 3; i++) await clickText(E, 'k + 1');
const tot = await page.$eval(`${bench(E)} [data-total-fetched]`, (e) => ({ f: Number(e.dataset.totalFetched), n: Number(e.dataset.totalNeeded), t: e.textContent }));
check('after three steps of k: 4 × 1024 bytes fetched for 4 × 128 needed', tot.f === 4096 && tot.n === 512 && tot.t.includes('8 times as much'), JSON.stringify(tot));
check('the narration says which elements lanes 0 and 1 read at k = 3', (await text(`${bench(E)} .narrate`)).includes('k = 3: lane 0 reads element 3, lane 1 reads 1027, 1024 elements further on'), await text(`${bench(E)} .narrate`));
await slide(E, 'Width', 4);
r = await readout(E);
check('Width 4: only four lanes active, 16 bytes apart, two sectors', Number(r.transactions) === exprAccess('n-col', P({ Width: 4, k: 3 }), 32).transactions && Number(r.transactions) === 2 && (await text(`${bench(E)} .narrate`)).includes('Only 4 lanes are active'), JSON.stringify(r));
if (out) await page.$eval(bench(E), (e) => e.scrollIntoView()) , await page.screenshot({ path: out + '-explorer.png' });

// ---- block width 16 splits the M read; 64 lanes and AMD lines; Metal is a choice
await clickText(E, 'The M read');
await select(E, 'Block width', 16);
r = await readout(E);
check('M read with 16-wide blocks: two addresses, two sectors', Number(r.transactions) === exprAccess('m-row', P({ blockDimX: 16 }), 32).transactions && Number(r.transactions) === 2, JSON.stringify(r));
await clickText(E, 'Fig. 6.2: row-major N');
await select(E, 'Warp width', 64);
await select(E, 'Segment model', 'amd128');
r = await readout(E);
check('64-lane warp of consecutive floats under AMD\'s 128-byte lines: two lines', Number(r.transactions) === 2 && Number(r.needed) === 256, JSON.stringify(r));
await select(E, 'Segment model', 'metal');
check('Apple: the size is the reader\'s choice and says so', (await text(bench(E))).includes('Your choice (not a fact)') && (await text(bench(E))).includes('Apple documents no coalescing rule'));
await select(E, 'Your choice', 32);
r = await readout(E);
check('with a 32-byte choice the same warp is 8 segments', Number(r.transactions) === 8, JSON.stringify(r));
await select(E, 'Warp width', 32);
await select(E, 'Segment model', 'nvidia');

// ---- device widths, from this machine's reports
const reports = await (await fetch(BASE + '/api/reports')).json();
const gpus = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race' && !/emulator/i.test(String(reports[b].device?.name ?? '')));
const rowsOf = (b, lab) => {
  const res = reports[b].results.filter((x) => x.lab === lab);
  const s = res.filter((x) => x.variant === 'solution').flatMap((x) => x.bench);
  return s.length ? s : res.filter((x) => x.variant === 'work').flatMap((x) => x.bench);
};
const widths = gpus.flatMap((b) => {
  const w = rowsOf(b, 'ch06/lab1_coalescing').find((x) => x.name === 'simd_width')?.width ?? Number(reports[b].device.warpSize);
  return Number.isFinite(w) && w ? [w] : [];
});
const shownWidths = await page.$eval(`${bench(E)} [data-device-widths]`, (e) => e.dataset.deviceWidths);
check(`device widths shown are the reports' (${widths.join(',') || 'none'})`, shownWidths === widths.join(','), shownWidths);

// ---- corner-turning stepper
const C = 'Corner-turning stepper';
await page.$eval(bench(C), (e) => e.scrollIntoView());
let n = await text(`${bench(C)} .narrate`);
check('stepper starts on Fig. 6.4\'s toy: (a) thread (3, 1) loads 121, its neighbour 105', n.includes('= 121') && n.includes('tx 2 loads 105'), n);
const toyA = tileLoad({ layout: 'col', arrangement: 'ch5', T: 4, bx: 1, ph: 2, Width: 16, warpWidth: 4 }).warps[0];
r = await readout(C);
check(`(a), toy warp of four: ${toyA.access.transactions} sectors, four runs`, Number(r.transactions) === toyA.access.transactions && Number(r.runs) === 4, JSON.stringify(r));
await clickText(C, '(b) corner turned');
n = await text(`${bench(C)} .narrate`);
check('(b): the same thread loads 91, its neighbour 90, and stores to Nds[tx][ty]', n.includes('= 91') && n.includes('tx 2 loads 90, 1 element away') && n.includes('Nds[tx][ty]'), n);
r = await readout(C);
check('(b), toy warp: one sector, one run', Number(r.transactions) === 1 && Number(r.runs) === 1, JSON.stringify(r));
if (out) await page.$eval(bench(C), (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-stepper-toy.png' });
await select(C, 'Tile width', 32);
await select(C, 'Width', 1024);
const real = (arr) => tileLoad({ layout: 'col', arrangement: arr, T: 32, bx: 1, ph: 2, Width: 1024, warpWidth: 32 });
r = await readout(C);
check('(b), T = 32, 32-lane warps: four sectors, one run', Number(r.transactions) === real('turned').warps[0].access.transactions && Number(r.transactions) === 4 && Number(r.runs) === 1, JSON.stringify(r));
const passes = await page.$eval(`${bench(C)} [data-store-passes]`, (e) => Number(e.dataset.storePasses));
check('(b), T = 32: the shared store is a 32-way conflict, handed to 6.4', passes === tileAccess({ T: 32, pitch: 32, orientation: 'x-first' }).passes && passes === 32 && (await text(bench(C))).includes('Section 6.4 is about this'), String(passes));
await clickText(C, 'Next warp');
await clickText(C, 'Next warp');
check('stepping warps: warp 3 of 32', (await text(bench(C))).includes('warp 3 of 32'));
await clickText(C, '(a) chapter 5\'s mapping');
r = await readout(C);
check('(a), T = 32: one sector per lane, 12.5 %', Number(r.transactions) === real('ch5').warps[2].access.transactions && Number(r.transactions) === 32 && Number(r.efficiency) === 0.125, JSON.stringify(r));
await select(C, 'N is stored', 'row');
r = await readout(C);
check('row-major N with chapter 5\'s mapping: four sectors (chapter 5 was already coalesced)', Number(r.transactions) === 4, JSON.stringify(r));
await select(C, 'Tile width', 16);
r = await readout(C);
check('Exercise 3 at T = 16, row-major: two runs but every byte used', Number(r.runs) === 2 && Number(r.efficiency) === 1, JSON.stringify(r));
if (out) await page.$eval(bench(C), (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-stepper-32.png' });

// ---- exercise 1 drill
const D = 'Exercise 1 drill';
const answers = exercise1();
for (const x of answers)
  await page.evaluate((sel, id, v) => [...document.querySelector(`${sel} [aria-label="Access (${id})"]`).querySelectorAll('button')].find((b) => b.textContent === v).click(), bench(D), x.id, x.verdict);
await pause(100);
await clickText(D, 'Check my answers');
check('drill: the model\'s answers score 9 of 9', (await text(bench(D))).includes('9 of 9 right.'));
const dRow = await text(`${bench(D)} tr[data-access="d"]`);
check('drill feedback for (d): 16 sectors, 25 % used', dRow.includes('16 sectors') && dRow.includes('25 % used'), dRow);
const hRow = await text(`${bench(D)} tr[data-access="h"]`);
check('drill feedback for (h): shared, a 4-way conflict on 8 banks (section 6.4)', hRow.includes('4-way on 8 banks'), hRow);
await page.evaluate((sel) => [...document.querySelector(`${sel} [aria-label="Access (g)"]`).querySelectorAll('button')].find((b) => b.textContent === 'uncoalesced').click(), bench(D));
await pause(80);
await clickText(D, 'Check my answers');
check('one wrong call is marked wrong, the rest stay right', (await page.$eval(`${bench(D)} tr[data-access="g"]`, (e) => e.dataset.result)) === 'wrong' && (await text(bench(D))).includes('8 of 9 right.'));
if (out) await page.$eval(bench(D), (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-drill.png' });

// ---- questions and predictions
const answerNumeric = async (needle, v) => {
  const q = await page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), needle);
  await (await q.$('input[type=number]')).type(String(v));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
check('q-cornerturn accepts 91', (await answerNumeric('which flat index', 91)).includes('Correct.'));
check('q-align accepts 5', (await answerNumeric('one float past a 32-byte boundary', 5)).includes('Correct.'));
const commit = async (needle, v) => {
  const q = await page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), needle);
  await (await q.$('input[type=number]')).type(String(v));
  await (await q.$('button.primary')).click();
  await pause(200);
  return q.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
};
const ps = await commit('How many 32-byte sectors does it touch', 32);
check('p-sectors reveal: the model says 32, lanes 4,096 bytes apart, row-major 4', ps.includes('The model says 32') && ps.includes('4,096 bytes apart') && ps.includes('touches 4') && ps.includes('You had it.'), ps);

// ---- your device, against this machine's reports
const measuredGpus = gpus.filter((b) => rowsOf(b, 'ch06/lab1_coalescing').some((x) => x.name === 'strided'));
const Y = 'Your device: coalescing';
await page.$eval(bench(Y), (e) => e.scrollIntoView());
const drawn = await page.$$eval(`${bench(Y)} [data-measured]`, (gs) => gs.map((g) => g.dataset.measured));
check(`your device: one measured curve per GPU with lab 6.1 results (${measuredGpus.join(',') || 'none'})`, JSON.stringify(drawn) === JSON.stringify(measuredGpus), JSON.stringify(drawn));
check('three model curves, 32, 64 and 128 bytes', JSON.stringify(await page.$$eval(`${bench(Y)} [data-model]`, (g) => g.map((x) => x.dataset.model))) === '["32","64","128"]');
if (!measuredGpus.length) check('with no measurement the panel says what to run', (await text(bench(Y))).includes('python3 labs/pmpp.py run 6.1 --solution --bench'));
const pt = await commit('time of the transpose with strided stores', 1);
const tRows = measuredGpus.flatMap((b) => {
  const rows = rowsOf(b, 'ch06/lab1_coalescing');
  const a = rows.find((x) => x.name === 'transpose' && (x.variant === 'read_rows' || x.kernel === 'read_rows'))?.seconds;
  const w = rows.find((x) => x.name === 'transpose' && (x.variant === 'write_rows' || x.kernel === 'write_rows'))?.seconds;
  return a && w ? [a / w] : [];
});
check('p-transpose reveal uses this machine\'s transposes, or says what to run', tRows.length ? pt.includes('took') && pt.includes('strided loads') : pt.includes('No measurement on this machine yet'), pt);
if (out) await page.screenshot({ path: out + '-device.png' });
const choose = async (needle, label) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await q.evaluate((e, l) => [...e.querySelectorAll('label')].find((x) => x.textContent.trim().startsWith(l)).querySelector('input').click(), label);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
};
check('q-store: "the book says so" is wrong, with NVIDIA\'s GDDR/HBM2 reason',
  (await choose('whether the strided stores are what costs', 'Nothing: the book says')).includes('depends on the device'));
const qst = await choose('whether the strided stores are what costs', 'Time the mirror image');
check('q-store: time the mirror-image transpose; the explanation quotes NVIDIA and says the read-modify-write is undescribed',
  qst.includes('Correct.') && qst.includes('Scattered accesses increase ECC memory transfer overhead') && qst.includes('described in none of the pages we read'), qst);

// ---- your device, against a fixture report (the rendering path, whatever this machine has)
fakeReports = {
  fixturegpu: {
    generated: '2026-11-20', device: { name: 'Fixture GPU', warpSize: '64' },
    results: [{
      lab: 'ch06/lab1_coalescing', variant: 'solution', backend: 'fixturegpu', tile: 16, built: true, test_rc: 0, tests: [],
      bench: [
        { name: 'copy', n: 1, seconds: 1, gb_per_s: 500 },
        ...[1, 2, 4, 8, 16, 32, 64].map((s, i) => ({ name: 'strided', stride: s, n: 1, seconds: 1, gb_per_s: 500 / (i + 1), relative_to_stride1: 1 / (i + 1) })),
        { name: 'transpose', variant: 'read_rows', rows: 4096, cols: 4096, seconds: 0.002, gb_per_s: 100 },
        { name: 'transpose', variant: 'write_rows', rows: 4096, cols: 4096, seconds: 0.001, gb_per_s: 200 },
        { name: 'simd_width', width: 48 },
      ],
    }],
  },
};
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector('.page[data-ready]');
await page.waitForSelector(bench(Y));
await pause(400);
check('fixture: the reported width 48 is offered in the explorer', (await page.$$eval(`${bench(E)} option`, (o) => o.map((x) => x.textContent))).includes('48 lanes (your device: Fixture GPU)'));
await select(E, 'Warp width', 48);
r = await readout(E);
check('fixture: a 48-lane warp of consecutive floats is 6 sectors', Number(r.transactions) === exprAccess('n-row', P({ warpWidth: 48 }), 32).transactions && Number(r.transactions) === 6, JSON.stringify(r));
check('fixture: one measured curve', JSON.stringify(await page.$$eval(`${bench(Y)} [data-measured]`, (g) => g.map((x) => x.dataset.measured))) === '["fixturegpu"]');
const fx = await text(`${bench(Y)} tr[data-device="fixturegpu"]`);
check('fixture: the table shows copy 500, strided stores 100 GB/s, loads 200 GB/s, ratio 2×', fx.includes('500') && fx.includes('100 GB/s') && fx.includes('200 GB/s') && fx.includes('2×'), fx);
const pt2 = await commit('time of the transpose with strided stores', 2);
check('fixture: the p-transpose reveal reads 2× and calls a prediction of 2 close', pt2.includes('took 2×') && pt2.includes('close to what you predicted'), pt2);
check('fixture: the model curve at stride 64 is 12.5 % for 32-byte segments', strideEfficiency(64, 32) === 0.125);

// ---- and with no reports at all
fakeReports = {};
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector('.page[data-ready]');
await page.waitForSelector(bench(Y));
await pause(400);
check('no reports: the panel says no measurement yet and what to run', (await text(bench(Y))).includes('No measurement yet') && (await page.$$(`${bench(Y)} [data-measured]`)).length === 0);
check('no reports: the explorer says no width has been reported', (await text(bench(E))).includes('No warp width reported by your devices yet'));
fakeReports = null;

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
check('at 390 px nothing scrolls the page sideways', narrow);
const fitSel = await page.$eval(bench(E), (b) => {
  const right = b.getBoundingClientRect().right;
  return [...b.querySelectorAll('select')].map((s) => [s.closest('label')?.firstChild?.textContent.trim() ?? '?', Math.round(s.getBoundingClientRect().right - right)]);
});
check('at 390 px every explorer select fits inside the bench', fitSel.every(([, over]) => over <= 1), JSON.stringify(fitSel));
const svgs = await page.$eval(bench(E), (b) => [...b.querySelectorAll('svg[aria-label^="Lanes"], svg[aria-label^="Logical view"]')].map((s) => {
  const p = s.parentElement;
  return { w: Math.round(s.getBoundingClientRect().width), scrolls: getComputedStyle(p).overflowX === 'auto' && p.scrollWidth > p.clientWidth };
}));
check('at 390 px the explorer drawings keep at least 600 px and scroll inside the bench instead of shrinking',
  svgs.length === 2 && svgs.every((x) => x.w >= 600 && x.scrolls), JSON.stringify(svgs));
if (out) {
  await page.$eval(bench(E), (e) => e.scrollIntoView());
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (u) => /\/content\/ch06\/narration\/(manifest\.json|[\w-]+\.mp3)$/.test(u);
const realErrors = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', realErrors.length === 0 && missing.every(expected404), [...realErrors, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
