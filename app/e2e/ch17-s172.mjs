// Browser check for section 17.2 "Formats decide the kernel": the format explorer (every tab,
// stepping, the ELL loop and stride switches, the hybrid's K, the COO order, editing the matrix,
// hovering), the "Two threads, one y[0]" stepper, the CSR prediction, the "open in the explorer"
// buttons, the three snippets, the comparison table, the narrated scene "Pad, turn, flatten"
// and some questions. Expected values come from the model, src/sim/sparse.ts, imported inside the
// page through the app's own module server (it imports its neighbours without file extensions,
// which Node cannot resolve on its own). Passes with or without narration audio.
// Usage: APP_URL=http://localhost:6521 node e2e/ch17-s172.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
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
const bench = (t) => `section.bench[aria-label="${t}"]`;
const EX = bench('Format explorer');
const RS = bench('Two threads, one y[0]');
const TB = bench('All six formats on one matrix');
const SC = 'section.scene[aria-label="Pad, turn, flatten"]';
const squash = (s) => s.replace(/\s+/g, ' ').trim();
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const clickText = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const btn = [...document.querySelector(scope).querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(90);
  return done;
};
const clickSel = async (sel) => { await page.$eval(sel, (e) => e.click()); await pause(90); };
const setRange = async (scope, aria, value) => {
  await page.evaluate((scope, aria, value) => {
    const el = document.querySelector(`${scope} input[aria-label="${aria}"]`);
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(value));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, scope, aria, value);
  await pause(120);
};
const model = (fn, ...args) =>
  page.evaluate(async (src, args) => {
    const m = await import('/src/sim/sparse.ts');
    return new Function('m', 'args', `return (${src})(m, ...args)`)(m, args);
  }, fn.toString(), args);
const arr = (name) => page.$$eval(`${EX} .mx[data-array="${name}"] > span[data-i]`, (els) => els.map((e) => e.childNodes[0]?.textContent ?? ''));
const reads = () => page.$$eval(`${EX} tr[data-thread]`, (trs) => trs.map((t) => t.dataset.read));
const kv = (k, scope = EX) => text(`${scope} [data-k="${k}"]`);
const step = () => clickText(EX, 'Step');
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

await goto(page, 'ch17/17.2');
await page.waitForSelector(EX);
await page.waitForSelector('.code .tabs button');

// ---------------------------------------------------------------- explorer: COO
let want = await model((m) => { const f = m.toCOO(m.FIG17_1); return [f.rowIdx.map(String), f.colIdx.map(String), f.value.map(String)]; });
check('COO tab on Fig. 17.1: the three arrays are Fig. 17.3', same(await arr('rowIdx'), want[0]) && same(await arr('colIdx'), want[1]) && same(await arr('value'), want[2]), JSON.stringify(await arr('value')));
let s = await model((m) => m.summarize(m.spmvTrace(m.toCOO(m.FIG17_1)), 4, 4));
check('COO: 8 atomic updates, 3 y elements shared, one run per group', (await kv('atomics')).startsWith(`${s.atomicUpdates} atomic`) && s.atomicUpdates === 8
  && (await kv('shared-y')).startsWith(`${s.sharedY} y`) && s.sharedY === 3 && (await kv('runs-now')).includes('1 run in each of 2 groups'), await kv('runs-now'));
const yRow = await kv('y');
check('COO: y ends as the row sums 8, 17, 10, 6', yRow.includes('[8, 17, 10, 6]') && yRow.includes('the row sums of A'), yRow);
await clickText(EX, "Fig. 17.6's order");
want = await model((m) => m.toCOO(m.FIG17_1, m.FIG17_6_ORDER).value.map(String));
check('Fig. 17.6\'s order permutes the arrays together and y does not change', same(await arr('value'), want) && same(await arr('rowIdx'), ['1', '3', '2', '2', '0', '1', '1', '0']) && (await kv('y')) === yRow, JSON.stringify(await arr('value')));
await clickText(EX, 'shuffled');
check('a shuffled order gives the same y', (await kv('y')) === yRow);
await clickText(EX, 'row order (Fig. 17.3)');
// Hover a matrix cell: its array slots get the ring.
await page.hover(`${EX} [data-matrix] span[data-r="1"][data-c="2"]`);
await pause(120);
const ring = await page.$$eval(`${EX} .mx[data-array="value"] > span[data-i]`, (els) => els.map((e) => e.style.boxShadow ? Number(e.dataset.i) : -1).filter((i) => i >= 0));
check('hovering the matrix cell (1, 2) rings value[3] in COO', same(ring, [3]), JSON.stringify(ring));
await page.hover(`${EX} .mx[data-array="value"] > span[data-i="6"]`);
await pause(120);
const ringM = await page.$$eval(`${EX} [data-matrix] span[data-r]`, (els) => els.filter((e) => e.style.boxShadow).map((e) => `${e.dataset.r},${e.dataset.c}`));
check('hovering value[6] rings the matrix cell (2, 2)', same(ringM, ['2,2']), JSON.stringify(ringM));
await page.mouse.move(5, 5);
if (out) await page.screenshot({ path: out + '-coo.png' });

// ---------------------------------------------------------------- CSR
await clickSel(`${EX} button[data-tab="csr"]`);
check('CSR: rowPtrs is Fig. 17.7 with its end marker', same(await arr('rowPtrs'), ['0', '2', '5', '7', '8']));
const csrReads = await model((m) => m.spmvTrace(m.toCSR(m.FIG17_1)).steps.map((s) => s.lanes.map((l) => (l.index == null ? '-' : String(l.index)))));
check('CSR iteration 0 reads 0, 2, 5, 7: 4 runs (p. 410)', same(await reads(), csrReads[0]) && same(csrReads[0], ['0', '2', '5', '7']) && (await kv('runs-now')).includes('4 runs'), await kv('runs-now'));
await step();
check('CSR iteration 1 reads 1, 3, 6 and thread 3 is idle', same(await reads(), csrReads[1]) && same(csrReads[1], ['1', '3', '6', '-']) && (await kv('idle-now')).startsWith('1 of 4'), JSON.stringify(await reads()));
await step();
check('CSR iteration 2: only thread 1 works; Step is then disabled', same(await reads(), ['-', '4', '-', '-']) && !(await clickText(EX, 'Step')));
check('CSR: no atomics, 4 lane-iterations wait', (await kv('atomics')).startsWith('0 atomic') && (await kv('waiting')) === '4' && (await kv('useful')) === '8');
if (out) await page.screenshot({ path: out + '-csr.png' });

// ---------------------------------------------------------------- ELL
await clickSel(`${EX} button[data-tab="ell"]`);
want = await model((m) => m.toELL(m.FIG17_1).at.map((e) => (e ? e.label : '0')));
check('ELL: value is Fig. 17.10, column-major with pads holding 0, colIdx holding -1', same(await arr('value'), want) && same(await arr('colIdx'), ['0', '0', '1', '3', '1', '2', '2', '-1', '-1', '3', '-1', '-1']), JSON.stringify(await arr('value')));
check('ELL: 4 pads, 28 numbers stored against 16 dense', (await kv('pads')) === '4' && (await kv('total')) === '28' && (await kv('dense')) === '16');
const ellReads = await model((m) => m.spmvTrace(m.toELL(m.FIG17_1)).steps.map((s) => s.lanes.map((l) => (l.index == null ? '-' : String(l.index)))));
let ok = true;
for (let t = 0; t < 3; t++) {
  if (t) await step();
  ok = ok && same(await reads(), ellReads[t]) && (await kv('runs-now')) === 'matrix reads in this step: 1 run';
}
check('ELL, loop to nnzPerRow: one run per iteration, the same idle lanes as CSR', ok && same(ellReads[1], ['4', '5', '6', '-']) && (await kv('waiting')) === '4', JSON.stringify(ellReads));
await clickText(EX, 'full width, test col >= 0');
await step(); await step();
const padRows = await page.$$eval(`${EX} tr[data-thread]`, (trs) => trs.map((t) => t.dataset.pad));
check('ELL, full width: iteration 2 reads 8 to 11, three of them pads; 0 waiting, 4 on pads', same(await reads(), ['8', '9', '10', '11']) && same(padRows, ['1', '0', '1', '1'])
  && (await kv('waiting')) === '0' && (await kv('pad-work')) === '4', JSON.stringify(padRows));
if (out) await page.screenshot({ path: out + '-ell-full.png' });
await clickText(EX, 'stride a multiple of 8');
const v8 = await arr('value');
check('ELL, stride 8: 24 slots, the 9 at slot 17 (question q-ell-index)', v8.length === 24 && v8[17] === '9' && (await kv('pads')) === '16', JSON.stringify(v8));
await clickText(EX, 'stride = numRows');
await clickText(EX, 'loop to nnzPerRow (the book)');
// Edit: one long row multiplies the pads.
for (const c of [0, 1, 2]) await clickSel(`${EX} [data-matrix] span[data-r="3"][data-c="${c}"]`);
const edited = await model((m) => { let x = m.FIG17_1; for (const c of [0, 1, 2]) x = m.toggle(x, 3, c); const st = m.storage(m.toELL(x), x); return [st.pads, st.total]; });
check('adding three non-zeros to row 3 makes it the longest: the model\'s pads and total', (await kv('pads')) === String(edited[0]) && (await kv('total')) === String(edited[1])
  && (await text(EX)).includes('edited'), `${await kv('pads')} ${await kv('total')} vs ${edited}`);
await clickSel(`${EX} button[data-preset="fig17_1"]`);
check('loading Fig. 17.1 again restores 4 pads', (await kv('pads')) === '4');

// ---------------------------------------------------------------- hybrid on Fig. 17.13
await clickSel(`${EX} button[data-preset="fig17_13"]`);
await clickSel(`${EX} button[data-tab="hyb"]`);
await setRange(EX, 'K', 2);
check('hybrid, Fig. 17.13, K = 2: 3 pads; COO part rows 1, 1, 1, 6, 6 (p. 414)', (await kv('pads')) === '3' && same(await arr('rowIdx (COO part)'), ['1', '1', '1', '6', '6']), `${await kv('pads')} ${await arr('rowIdx (COO part)')}`);
const hs = await model((m) => m.summarize(m.spmvTrace(m.toHybrid(m.FIG17_13, 2)), 4, 8));
check('hybrid, K = 2: the model\'s divergence and 5 atomic adds', (await kv('group-iters')) === String(hs.groupIterations) && (await kv('atomics')).startsWith('5 atomic') && hs.atomicUpdates === 5);
if (out) await page.screenshot({ path: out + '-hybrid.png' });
await setRange(EX, 'K', 5);
check('hybrid, K = 5: pure ELL, 22 pads, empty COO part', (await kv('pads')) === '22' && (await arr('rowIdx (COO part)')).length === 0);
await page.select(`${EX} select[aria-label="Threads per group"]`, 'rows');
await pause(120);
const hs8 = await model((m) => m.summarize(m.spmvTrace(m.toHybrid(m.FIG17_13, 5)), 8, 8));
check('one group of 8: the model\'s count of group-iterations', (await kv('group-iters')) === String(hs8.groupIterations), `${await kv('group-iters')} vs ${hs8.groupIterations}`);
await page.select(`${EX} select[aria-label="Threads per group"]`, 'four');

// ---------------------------------------------------------------- JDS on Fig. 17.14
await clickSel(`${EX} button[data-preset="fig17_14"]`);
await clickSel(`${EX} button[data-tab="jds"]`);
check('JDS on Fig. 17.14: row and iterPtr are the figure\'s', same(await arr('row'), ['1', '3', '5', '2', '4', '0']) && same(await arr('iterPtr'), ['0', '6', '11', '14', '15'])
  && (await arr('value')).join('') === 'bhmfkacingldjoe', JSON.stringify(await arr('iterPtr')));
for (let t = 0; t < 3; t++) await step();
check('JDS, last iteration: only thread 0 (row 1) works; the active set shrank from the end', same(await reads(), ['14', '-', '-', '-', '-', '-']));
if (out) await page.screenshot({ path: out + '-jds.png' });

// ---------------------------------------------------------------- CSC on Fig. 17.1
await clickSel(`${EX} button[data-preset="fig17_1"]`);
await clickSel(`${EX} button[data-tab="csc"]`);
check('CSC: colPtrs and rowIdx are Fig. 17.16', same(await arr('colPtrs'), ['0', '2', '4', '6', '8']) && same(await arr('rowIdx'), ['0', '1', '0', '2', '1', '2', '1', '3']));
check('CSC iteration 0 reads 0, 2, 4, 6 (p. 420); x read in one run; atomics collide on 3 y elements',
  same(await reads(), ['0', '2', '4', '6']) && (await kv('x-runs-now')) === 'x reads in this step: 1 run' && (await kv('shared-y')).startsWith('3 y'), await kv('x-runs-now'));
await step();
check('CSC iteration 1 reads 1, 3, 5, 7 and no x', same(await reads(), ['1', '3', '5', '7']) && (await kv('x-runs-now')) === 'x reads in this step: 0 runs', await kv('x-runs-now'));

// ---------------------------------------------------------------- open-in-explorer buttons and the prediction
await clickSel('button[data-open="csr:1"]');
await pause(400);
check('"Show CSR at iteration 1" opens the CSR tab at step 2', (await page.$eval(`${EX} button[data-tab="csr"]`, (b) => b.getAttribute('aria-pressed'))) === 'true'
  && (await page.$eval(`${EX} [data-step]`, (e) => e.dataset.step)) === '1' && same(await reads(), ['1', '3', '6', '-']));
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('which index of value does thread 2 read')));
await (await pred.$('input[type=number]')).type('6');
await (await pred.$('button.primary')).click();
await pause(150);
const pt = squash(await pred.evaluate((e) => e.textContent));
check('the CSR prediction reveals value[1], value[3], value[6] and nothing for thread 3', pt.includes('You predicted 6') && pt.includes('Right.') && pt.includes('value[1], value[3], value[6], nothing'), pt);

// ---------------------------------------------------------------- the stepper
await page.$eval(RS, (e) => e.scrollIntoView({ block: 'start' }));
await clickText(RS, 'Step thread 0');
for (let i = 0; i < 3; i++) await clickText(RS, 'Step thread 1');
await clickText(RS, 'Step thread 0');
await clickText(RS, 'Step thread 0');
check('thread 0 loads, thread 1 runs through, thread 0 finishes: y[0] = 1, thread 1\'s update lost', (await kv('y0', RS)) === '1' && (await kv('rmw-result', RS)).includes('1 instead of 8'), await kv('rmw-result', RS));
await clickText(RS, 'Both load, then both store');
const outs = await model((m) => [m.cooUpdates([0, 1, 0, 1, 0, 1]).y, m.cooOutcomes(), m.cooOutcomes({ atomic: true })]);
check('"both load, then both store" gives the model\'s 7', (await kv('y0', RS)) === String(outs[0]) && outs[0] === 7);
const found = await page.$$eval(`${RS} [data-found]`, (els) => els.map((e) => Number(e.dataset.found)));
check('the outcomes found so far are 1 and 7', same(found, [1, 7]), JSON.stringify(found));
await clickText(RS, 'Show every interleaving');
const tally = await model((m) => { const t = {}; for (const s of m.interleavings([3, 3])) { const y = m.cooUpdates(s).y; t[y] = (t[y] ?? 0) + 1; } return t; });
const allTxt = await kv('all-interleavings', RS);
check('every interleaving: the model\'s 20, ending at 1, 7 or 8', same(outs[1], [1, 7, 8]) && allTxt.includes('The 20 ways') && Object.entries(tally).every(([y, n]) => allTxt.includes(`${y} (${n} of them)`)), allTxt);
await page.evaluate((sel) => { const b = [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('atomic add')).querySelector('input'); b.click(); }, RS);
await pause(100);
await clickText(RS, 'Step thread 1');
await clickText(RS, 'Step thread 0');
check('with the atomic, one step each, and y[0] = 8 (both updates landed)', same(outs[2], [8]) && (await kv('y0', RS)) === '8' && (await kv('rmw-result', RS)).includes('both updates landed'));
for (const k of [0, 1, 2]) await clickSel(`${RS} button[data-term="${k}"]`);
const f012 = await model((m) => m.floatSum(m.ORDER_TERMS, [0, 1, 2]));
check('1e8, 1, -1e8 in that order: float32 gives 0, the exact sum is 1', f012 === 0 && (await kv('float-result', RS)).startsWith('0;') && (await kv('float-all', RS)).includes('0 in 4, 1 in 2'), await kv('float-result', RS));
await clickText(RS, 'Clear');
for (const k of [0, 2, 1]) await clickSel(`${RS} button[data-term="${k}"]`);
check('1e8, -1e8, 1: the large terms cancel first and the 1 survives', (await kv('float-result', RS)).startsWith('1;'));
await clickText(RS, 'Shuffle');
check('Shuffle picks a full order', (await page.$(`${RS} [data-k="float-result"]`)) !== null);
if (out) await page.screenshot({ path: out + '-stepper.png' });

// ---------------------------------------------------------------- code
const codes = await page.$$eval('.code', (els) => els.map((e) => ({ tab: e.querySelector('.tabs button')?.textContent, status: e.querySelector('.status')?.textContent ?? '', hl: [...e.querySelectorAll('.ln.hl')].map((l) => l.textContent.trim()), text: e.querySelector('pre').textContent })));
const byTab = Object.fromEntries(codes.map((c) => [c.tab, c]));
check('three snippets are shown (CSC is named in the prose, not shown), each with its verification status', ['spmv_coo.cu', 'spmv_csr.cu', 'spmv_ell.cu'].every((t) => byTab[t]?.status.includes('emulator-checked and front-end-checked')) && !byTab['spmv_csc.cu'], JSON.stringify(Object.keys(byTab)));
check('the shown files are the snippets, without their header comments', codes.every((c) => !c.text.trim().startsWith('//')) && byTab['spmv_ell.cu'].text.includes('spmv_ell_full_kernel'));
const ellHl = byTab['spmv_ell.cu'].hl;
check('ELL highlights the index line, the accumulate and the pad test', ellHl.filter((l) => l === 'unsigned int i = t*ellMatrix.stride + row;').length === 2
  && ellHl.filter((l) => l === 'y[row] += sum;').length === 2 && ellHl.includes('if (col >= 0) sum += x[col]*ellMatrix.value[i];'), JSON.stringify(ellHl));
check('COO highlights its two atomic lines', byTab['spmv_coo.cu'].hl.length === 2 && byTab['spmv_coo.cu'].hl[1].startsWith('y_ref.fetch_add'));

// ---------------------------------------------------------------- comparison table
await page.$eval(TB, (e) => e.scrollIntoView({ block: 'start' }));
await clickText(TB, 'Fig. 17.1');
const cell = (f, c) => text(`${TB} tr[data-format="${f}"] td[data-col="${c}"]`);
check('table, Fig. 17.1: COO and CSC 8 atomics; CSR worst 4 runs, ELL 1; ELL full width 0 waiting and 4 on pads',
  (await cell('coo', 'atomics')).startsWith('8') && (await cell('csc', 'atomics')).startsWith('8') && (await cell('csr', 'worst')) === '4' && (await cell('ell', 'worst')) === '1'
  && (await cell('ell-full', 'waiting')) === '0' && (await cell('ell-full', 'padwork')) === '4' && (await cell('ell', 'waiting')) === '4');
await clickText(TB, 'Fig. 17.13');
const tPads = await model((m) => [m.storage(m.toELL(m.FIG17_13), m.FIG17_13).pads, m.storage(m.toHybrid(m.FIG17_13, 5), m.FIG17_13).pads]);
check('table, Fig. 17.13: ELL pads 22, the hybrid at the explorer\'s K the model\'s count', (await cell('ell', 'pads')) === '22' && (await cell('hyb', 'pads')) === String(tPads[1]), `${await cell('hyb', 'pads')}`);
check('the table\'s matrix is the explorer\'s', (await page.$eval(`${EX} button[data-preset="fig17_13"]`, (b) => b.getAttribute('aria-pressed'))) === 'true');
if (out) await page.screenshot({ path: out + '-table.png' });

// ---------------------------------------------------------------- questions
const ask = async (needle, f) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await f(q);
  await pause(150);
  return squash(await q.evaluate((e) => e.textContent));
};
let qt = await ask('How many threads add into', async (q) => { await (await q.$('input[type=number]')).type('3'); await (await q.$('button.primary')).click(); });
check('q-coo-y1: 3 is correct', qt.includes('Correct.'), qt);
qt = await ask('What is the flat index of row 1', async (q) => { await (await q.$('input[type=number]')).type('9'); await (await q.$('button.primary')).click(); });
check('q-ell-index: 9 (the book\'s stride) is not accepted; the answer is 17', qt.includes('Not quite. The answer is 17'), qt);
qt = await ask('x[0] is +Inf', async (q) => { const r = await q.$$('input[type=radio]'); await r[0].click(); await (await q.$('button.primary')).click(); });
check('q-pad-marker: NaN in every padded row is correct', qt.includes('Correct.') && qt.includes('rows 0, 2 and 3'), qt);
qt = await ask('cannot be computed from m', async (q) => { const r = await q.$$('input[type=radio]'); await r[3].click(); await (await q.$('button.primary')).click(); });
check('q-ex2-missing: "only ELL" is wrong and says why', qt.includes('Not quite.') && qt.includes('K + 1'), qt);

// ---------------------------------------------------------------- the scene
const N = 6;
const LABELS = ['CSR reads', 'Idle lanes', 'Pad', 'Turn', 'Run', 'The bill'];
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const res = await fetch(BASE + '/content/ch17/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch17/narration/ell-${n}.txt`)).text()));
const words = scripts.map((x) => x.split(' ').length);
check('six scripts exist, each 35 to 60 words', words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((x) => !/[<>=×%/_(){}#*+]/.test(x)), scripts.find((x) => /[<>=×%/_(){}#*+]/.test(x)));
const book = await model((m) => [m.exercise2Count(1000, 10000, 200), m.storage(m.toELL(m.FIG17_1), m.FIG17_1)]);
check('the numbers the scripts say are the model\'s: 0, 2, 5, 7; 4 pads for 8; 12 slots; 2 and 40 percent',
  scripts[0].includes('0, 2, 5 and 7') && scripts[2].includes(`${book[1].pads} pads, to hold 8 non-zeros`) && scripts[5].includes(`${book[1].values} slots hold 8 non-zeros`)
  && Math.round((100 * book[0].csr) / 1e6) === 2 && scripts[5].includes('about 2 percent') && (100 * book[0].ell) / 1e6 === 40 && scripts[5].includes('40 percent'));
if (manifest?.clips?.['ell-1']) check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);
const labels = await page.$$eval(SC + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered, with their names', labels.length === N && LABELS.every((l, i) => labels[i] === `${i + 1}. ${l}`), labels.join(' | '));
const box = (word) => page.evaluateHandle((sel, word) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes(word)).querySelector('input'), SC, word);
const setBox = async (word, on) => { const b = await box(word); if ((await b.evaluate((e) => e.checked)) !== on) await b.click(); await pause(120); };
const hasCaption = () => page.$(SC + ' .caption').then((e) => !!e);
const caption = () => page.$eval(SC + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim());
check('there is a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const audible = !!manifest?.clips?.['ell-1'];
if (audible) check('with audio and captions off, no caption is shown', !(await hasCaption()));
else check('a part with no audio shows its script even with captions off', (await hasCaption()) && (await caption()) === scripts[0]);
await setBox('captions', true);
check('captions on: part 1\'s script', (await hasCaption()) && (await caption()) === scripts[0]);
const press = async (label) => { await page.evaluate((sel, label) => [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim().includes(label)).click(), SC, label); await pause(150); };
const seek = async (v) => {
  await page.$eval(SC + ' input[type=range]', (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(150);
};
const current = () => page.$$eval(SC + ' .seg button', (bs) => bs.findIndex((b) => b.getAttribute('aria-pressed') === 'true') + 1);
const drawn = () => page.evaluate((S) => {
  const svg = document.querySelector(S + ' .stage svg');
  const g = (a) => svg.querySelector(`[${a}]`)?.getAttribute(a) ?? null;
  return { iter: g('data-iter'), reads: g('data-reads'), runs: g('data-runs'), idle: g('data-idle'), pads: g('data-pads'), slots: g('data-slots'), flat: g('data-flat'), upto: g('data-upto'),
    bars: Object.fromEntries([...svg.querySelectorAll('[data-bar]')].map((b) => [b.dataset.bar, Number(b.dataset.frac)])), headline: svg.querySelector('[data-headline]')?.textContent.trim() ?? '' };
}, SC);
const sceneCsr = await model((m) => m.spmvTrace(m.toCSR(m.FIG17_1)).steps.map((s) => [s.lanes.map((l) => (l.index == null ? '-' : l.index)).join(' '), m.matrixRuns(s, 4)[0], m.idleLanes(s)]));
const sceneEll = await model((m) => [m.spmvTrace(m.toELL(m.FIG17_1)).steps.map((s) => [s.lanes.map((l) => (l.index == null ? '-' : l.index)).join(' '), m.matrixRuns(s, 4)[0]]), m.toELL(m.FIG17_1).at.map((e) => (e ? e.label : 'pad')).join(' ')]);
await setBox('sound', false);
await press('Play');
await pause(1500);
check('with sound off the animation runs', (await page.$eval(SC + ' input[type=range]', (e) => Number(e.value))) > 20);
await press('Pause');
await seek(970);
await press('Resume');
await pause(2600);
check('at the end of part 1, part 2 starts by itself with its caption', (await current()) === 2 && (await caption()) === scripts[1], `part ${await current()}`);
await press('Pause');
await press('1. CSR reads');
await seek(1000);
let d = await drawn();
check('part 1: CSR iteration 0, reads 0 2 5 7, 4 separate places', d.iter === '0' && d.reads === sceneCsr[0][0] && d.reads === '0 2 5 7' && d.runs === String(sceneCsr[0][1]) && d.runs === '4', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-scene1.png' });
await press('2. Idle lanes');
await seek(250);
d = await drawn();
check('part 2, first half: iteration 1, thread 3 idle', d.iter === '1' && d.reads === sceneCsr[1][0] && d.idle === String(sceneCsr[1][2]), JSON.stringify(d));
await seek(900);
d = await drawn();
check('part 2, second half: iteration 2, only thread 1 runs', d.iter === '2' && d.reads === '- 4 - -' && d.idle === '3', JSON.stringify(d));
await press('3. Pad');
await seek(1000);
d = await drawn();
check('part 3 ends: 4 pads, 12 slots', d.pads === String(book[1].pads) && d.slots === String(book[1].values) && d.pads === '4' && d.slots === '12', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-scene3.png' });
await press('4. Turn');
await seek(500);
d = await drawn();
check('part 4 midway: the flat array is filling in column order', Number(d.upto) > 2 && Number(d.upto) < 12 && sceneEll[1].startsWith(d.flat), JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 4 ends: 1 5 2 6 7 3 8 pad pad 9 pad pad, Fig. 17.10\'s order', d.flat === sceneEll[1] && d.flat === '1 5 2 6 7 3 8 pad pad 9 pad pad', JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-scene4.png' });
let runOk = true;
await press('5. Run');
for (const [pos, t] of [[100, 0], [500, 1], [900, 2]]) {
  await seek(pos);
  d = await drawn();
  runOk = runOk && d.iter === String(t) && d.reads === sceneEll[0][t][0] && d.runs === '1' && sceneEll[0][t][1] === 1;
}
check('part 5: iterations 0, 1, 2 each read one run; the idle lanes match CSR\'s', runOk, JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-scene5.png' });
await press('6. The bill');
await seek(1000);
d = await drawn();
check('part 6 ends: the book\'s example from exercise 2\'s counts, 2.1% and 40%', Math.abs(d.bars.csr - book[0].csr / 1e6) < 1e-3 && d.bars.ell === 0.4 && d.slots === '12', JSON.stringify(d.bars));
if (out) await page.screenshot({ path: out + '-scene6.png' });

// ---------------------------------------------------------------- themes and a narrow screen
if (out) {
  await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
  await page.evaluate(() => { delete document.documentElement.dataset.theme; });
  await press('5. Run');
  await seek(500);
  await page.screenshot({ path: out + '-scene5-dark.png' });
  await page.$eval(EX, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-explorer-dark.png' });
}
await setBox('captions', false);
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(400);
const narrow = await page.evaluate((S, E) => ({
  page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1,
  scene: document.querySelector(S).getBoundingClientRect().width <= document.documentElement.clientWidth,
  explorer: document.querySelector(E).getBoundingClientRect().width <= document.documentElement.clientWidth,
}), SC, EX);
check('at a narrow width nothing makes the page scroll sideways', narrow.page && narrow.scene && narrow.explorer, JSON.stringify(narrow));
if (out) {
  await page.$eval(EX, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-explorer.png' });
  await page.$eval(RS, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-stepper.png' });
}

const expected404 = (u) => /\/content\/ch17\/narration\/(manifest\.json|[\w-]+\.mp3)$/.test(u);
const realErrors = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest or clip is allowed)', realErrors.length === 0 && missing.every(expected404), [...realErrors, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
