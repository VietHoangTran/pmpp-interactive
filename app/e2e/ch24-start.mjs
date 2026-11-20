// Browser check for chapter 24's start page: the steps map, the "what limits what"
// classification, the audit of Fig. 24.1 with its explorer, and the threshold strip.
// Usage: node e2e/ch24-start.mjs [screenshot-prefix]
// E2E_WIDTH=390 runs the same checks at a phone width; E2E_THEME=light|dark fixes the theme.
import { open, goto, BASE } from './browser.mjs';

const out = process.argv[2];
const theme = process.env.E2E_THEME;
const { browser, page, errors } = await open({ width: Number(process.env.E2E_WIDTH || 1440), height: 1000, dark: theme === 'dark' });
// The page is read against a known progress file (one section of chapter 5 done, nothing
// answered), and nothing is written to the real one.
const START = { version: 1, sections: { 'ch05/5.1': { done: true, visited: '2026-11-20T00:00:00.000Z' } }, answers: {}, review: {} };
const puts = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
  if (r.method() === 'PUT') {
    puts.push(JSON.parse(r.postData()));
    return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  }
  return r.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(START) });
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent);
const squash = (s) => s.replace(/\s+/g, ' ');
const question = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const clickText = async (scope, label) => {
  const done = await page.evaluate(
    (scope, label) => {
      const root = [...document.querySelectorAll('section.bench')].find((b) => b.getAttribute('aria-label') === scope);
      const btn = [...root.querySelectorAll('button')].find((b) => b.textContent.trim() === label);
      if (!btn || btn.disabled) return false;
      btn.click();
      return true;
    },
    scope,
    label,
  );
  await pause(80);
  return done;
};
const setRange = (sel, value) =>
  page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
const shot = async (name, sel) => {
  if (!out) return;
  await pause(150);
  await (await page.$(sel)).screenshot({ path: `${out}-${name}.png` });
};

const reports = await (await fetch(BASE + '/api/reports')).json();
const real = Object.keys(reports).filter((b) => b !== 'emu');

await goto(page, 'ch24/start');
if (theme) await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, theme);

// ---- 1. the steps map ----
const S = "The book's three steps";
await page.waitForSelector(bench(S) + ' [data-chip]');
const lane = (n) => page.$$eval(`${bench(S)} [data-lane="${n}"] [data-chip]`, (c) => c.map((x) => x.dataset.chip));
const [l1, l2, l3, lo] = [await lane('1'), await lane('2'), await lane('3'), await lane('outside')];
check('lanes hold 5, 9 and 8 chapters', l1.length === 5 && l2.length === 9 && l3.length === 8, [l1.length, l2.length, l3.length].join());
check('step one is chapters 2 to 6, step three ends at chapter 23', l1.join() === 'ch02,ch03,ch04,ch05,ch06' && l3[0] === 'ch16' && l3[7] === 'ch23', l1.join() + ' / ' + l3.join());
check('the fourth lane holds chapters 1 and 24, appendices A and C and modules X1 to X3', lo.join() === 'ch01,ch24,appA,appC,x1,x2,x3', lo.join());
const lit = () => page.$$eval(`${bench(S)} [data-chip][data-lit="true"]`, (c) => c.map((x) => x.dataset.chip));
check('matmul lights chapters 3, 4, 5, 6, 15 and 19', (await lit()).join() === 'ch03,ch04,ch05,ch06,ch15,ch19', (await lit()).join());
await clickText(S, 'sampling');
check('sampling lights chapters 9, 10, 11, 12, 14 and module X1', (await lit()).join() === 'ch09,ch10,ch11,ch12,ch14,x1', (await lit()).join());
let r = squash(await text(bench(S) + ' [data-readout="goal"]'));
check('sampling is all step two plus a module the book lacks', r.includes('step two (9, 10, 11, 12, 14)') && r.includes("outside the book's steps (X1)") && r.includes('module X1 does'), r);
check('a lit chip shows its project milestone', (await text(`${bench(S)} [data-chip="ch10"]`)).includes('M4 softmax and argmax'));
check('modules are marked as ours', (await text(`${bench(S)} [data-chip="x1"]`)).includes('(ours)'));
// Every goal draws on more than one lane ("other backends" on one lane plus every lab).
const spans = [];
for (const g of ['matmul', 'attention', 'sampling', 'quantised ops', 'other backends', 'reading real code']) {
  await clickText(S, g);
  const lanes = await page.$$eval(`${bench(S)} [data-lane]`, (ls) => ls.filter((l) => l.querySelector('[data-lit="true"]')).map((l) => l.dataset.lane));
  const everyLab = (await text(bench(S) + ' [data-readout="goal"]')).includes('and every lab');
  spans.push(`${g}:${lanes.join('+')}${everyLab ? '+labs' : ''}`);
}
check('every goal draws on more than one step', spans.every((s) => s.split(':')[1].split('+').length > 1), spans.join(' | '));
check('matmul spans all three steps', spans[0] === 'matmul:1+2+3', spans[0]);
// The reader's record: sections from the progress file, labs counted from reports by the chapter prefix.
const ch5 = squash(await text(`${bench(S)} [data-chip="ch05"] [data-record]`));
const labs5 = new Set(Object.values(reports).flatMap((run) => run.results.map((x) => x.lab).filter((l) => l.startsWith('ch05/'))));
const pass5 = [...labs5].filter((lab) =>
  real.some((b) => {
    if (reports[b].work_state?.[lab] === 'untouched') return false;
    const rs = reports[b].results.filter((x) => x.lab === lab && x.variant === 'work' && x.tests.length);
    return rs.length && rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass));
  }),
).length;
check('a chip shows sections marked done', ch5.startsWith('1 of '), ch5);
check('a chip counts labs passing in work/ on a real backend', labs5.size ? ch5.includes(`${pass5} of ${labs5.size} labs passing`) : !ch5.includes('labs'), ch5);
const linked = await page.$$eval(`${bench(S)} [data-chip]`, (cs) => cs.map((c) => ({ id: c.dataset.chip, a: c.querySelector('a')?.getAttribute('href') ?? null, planned: c.textContent.includes('planned') })));
check('built chapters are links, planned ones plain text', linked.every((c) => (c.a === null) === c.planned) && linked.find((c) => c.id === 'ch05').a === '#/ch05/start', JSON.stringify(linked.filter((c) => (c.a === null) !== c.planned)));
await clickText(S, 'sampling');
await shot('steps', bench(S));

let q = await question('where is the material');
await (await q.$$('input[type=radio]'))[1].click();
await (await q.$('button.primary')).click();
await pause(150);
check('q-compose: the material is step two, assembled in a module of ours', (await q.evaluate((e) => e.textContent)).includes('Correct.'));

// ---- 2. what limits what ----
const L = 'What limits what';
const place = async (qid, value) => {
  await page.select(`${bench(L)} [data-card="${qid}"] select`, value);
  await pause(40);
};
const P = 'ch24/start/q-limit-';
check('Check is disabled until all five are placed', (await clickText(L, 'Check')) === false);
await place(P + 'load', 'hostlink');
await place(P + 'decode', 'compute'); // wrong on purpose
await place(P + 'prefill', 'compute');
await place(P + 'fit', 'memcap');
await place(P + 'allreduce', 'gpulink');
await clickText(L, 'Check');
r = squash(await text(bench(L)));
check('four of five right with one misplaced card', r.includes('4 of 5 right at the last check'), r.slice(-200));
const wrongCard = squash(await text(`${bench(L)} [data-card="${P}decode"]`));
check('a wrong placement gets a one-line reason', wrongCard.includes('not compute throughput') && wrongCard.includes('leave the arithmetic units waiting'), wrongCard);
check('a right placement gets its reason and source', squash(await text(`${bench(L)} [data-card="${P}load"]`)).includes('ch. 2 §2.4'));
await pause(500); // let the first placements reach the progress file before the fix
await place(P + 'decode', 'membw');
await clickText(L, 'Check');
r = squash(await text(bench(L)));
check('all five right after the fix', r.includes('All five right.') && r.includes('touches four of them'), r.slice(-260));
const results = await page.$$eval(`${bench(L)} [data-card]`, (c) => c.map((x) => x.dataset.result));
check('each row of the table took exactly one card', results.every((x) => x === 'right') && results.length === 5, results.join());
await shot('limits', bench(L));

// ---- 3. check the table ----
const T = 'Check the table';
check('the table is hidden until the prediction is committed', !!(await page.$(bench(T) + ' [data-gate="table"]')) && !(await page.$('[data-printed-table]')));
q = await question('By what factor does it say the host link');
check('the prediction hides its answer', !(await q.evaluate((e) => e.textContent)).includes('8 GB/s'));
await (await q.$('input[type=number]')).type('100');
await (await q.$('button.primary')).click();
await pause(200);
let pred = squash(await q.evaluate((e) => e.textContent));
check('p-hostlink reveals 8x and reacts to a guess near 100', pred.includes('You predicted 100 x') && pred.includes('8x as printed: 8 GB/s to 64 GB/s') && pred.includes('kept pace') && pred.includes('about 16x'), pred);
await page.waitForSelector('[data-printed-table]');
const printed = await page.$$eval('[data-printed-table] [data-cell]', (c) => Object.fromEntries(c.map((x) => [x.dataset.cell, x.textContent.trim()])));
check('15 tickable cells, printed verbatim', Object.keys(printed).length === 15 && printed['tensor.ratio'] === '638x' && printed['nvlink.b200'] === '1,800 TB/s' && printed['tensor.b200'] === '2200 TFLOPS (Tensor Core FP32)' && printed['memcap.g80'] === '1.5GB' && printed['pcie.g80'] === '8 GB/s (Gen2 x16, each way)', JSON.stringify(printed));
check('the explorer is not shown before checking', !(await page.$(bench(T) + ' [data-explorer]')));
const tick = (id) => page.$eval(`[data-printed-table] [data-cell="${id}"] input`, (e) => e.click());
for (const id of ['fp32.ratio', 'tensor.b200', 'tensor.ratio', 'membw.ratio', 'memcap.ratio', 'pcie.ratio']) await tick(id);
await clickText(T, 'Check');
r = squash(await text(bench(T) + ' [data-readout="audit"]'));
check('two cells are wrong as printed; one found, five other ticks', r.includes('Two cells are wrong as printed.') && r.includes('You found 1 of 2') && r.includes('ticked 5 other cells'), r);
const states = await page.$$eval('[data-printed-table] [data-cell]', (c) => c.filter((x) => x.dataset.state === 'wrong').map((x) => x.dataset.cell));
check('the wrong cells are the 638x ratio and the TB/s unit', states.join() === 'tensor.ratio,nvlink.b200', states.join());
const v = squash(await text(bench(T) + ' [data-verdicts]'));
for (const a of ['80 / 0.345 = 232', '8000 / 86.4 = 93', '192 / 1.5 = 128', '64 / 8 = 8', '2200 / 0.345 = 6,377'])
  check(`a ticked cell gets its arithmetic: ${a}`, v.includes(a), v);
check('638x would need 220 TFLOPS', squash(await text(`${bench(T)} [data-verdict="tensor.ratio"]`)).includes('would need 220 TFLOPS'));
check('the 2200 figure stands', squash(await text(`${bench(T)} [data-verdict="tensor.b200"]`)).includes('the figure stands and the ratio is the error') && (await page.$eval('[data-cell="tensor.b200"]', (e) => e.dataset.state)) === 'stands');
const okCells = await page.$$eval('[data-printed-table] [data-cell][data-state="ok"]', (c) => c.map((x) => x.dataset.cell));
check('the four consistent ratios are confirmed', okCells.join() === 'fp32.ratio,membw.ratio,memcap.ratio,pcie.ratio', okCells.join());
const missed = squash(await text(`${bench(T)} [data-verdict="nvlink.b200"]`));
check('the unit error is listed as missed, with its arithmetic', missed.startsWith('missed') && missed.includes('1,800,000 GB/s / 8,000 GB/s = 225') && missed.includes('1,800 GB/s'), missed);
check('three printed ratios are right and one is off by ten', v.includes('4 of the 5 printed ratios do') && v.includes('Off by ten'), v.slice(-300));
await shot('audit', bench(T));

// The explorer: recomputed ratios under each reading.
const ratios = () => page.$$eval('[data-recomputed] tbody tr', (rows) => Object.fromEntries(rows.map((x) => [x.dataset.row, x.querySelector('[data-ratio]').textContent.trim()])));
const seg = async (label, option) => {
  await page.evaluate(
    (label, option) => [...document.querySelector(`[data-seg="${label}"]`).querySelectorAll('button')].find((b) => b.textContent.trim().startsWith(option)).click(),
    label,
    option,
  );
  await pause(80);
};
const order = async () => squash(await text(bench(T) + ' [data-readout="order"]'));
let x = await ratios();
check('the book as printed recomputes to 232x, 6,377x, 93x, 128x and 8x', [x.fp32, x.tensor, x.membw, x.memcap, x.pcie].join() === '232x,6,377x,93x,128x,8x' && x.nvlink === '', JSON.stringify(x));
const flagged = await page.$$eval('[data-recomputed] tbody tr', (rows) => rows.filter((x) => x.querySelector('[data-printed] .t-fail')).map((x) => x.dataset.row + ':' + x.querySelector('[data-printed]').textContent.trim()));
check('only the printed 638x disagrees with its recomputed ratio', flagged.join() === 'tensor:638x', flagged.join());
check('the host link grew least on the printed figures', (await order()).includes('Least growth: host link (PCIe x16, each way), 8x'), await order());
await seg('Figures', "NVIDIA's documents");
x = await ratios();
check("NVIDIA's documents: 217x, 6,366x, 104x, 120x and 16x", [x.fp32, x.tensor, x.membw, x.memcap, x.pcie].join() === '217x,6,366x,104x,120x,16x', JSON.stringify(x));
check('none of the five printed ratios survives', (await page.$$('[data-recomputed] [data-printed] .t-fail')).length === 5);
check('the host link still grew least', (await order()).includes('Least growth: host link (PCIe x16, each way), 16x'), await order());
await seg('Tensor-core figure', 'dense');
check('dense TF32 gives 3,183x', (await ratios()).tensor === '3,183x', (await ratios()).tensor);
await seg('PCIe generation of the 2007 board', '2');
check('a Gen2 board against the Gen5 rate gives 8x', (await ratios()).pcie === '8x', (await ratios()).pcie);
await seg('PCIe generation of the 2007 board', '1');
const cells2025 = () => page.$$eval('[data-recomputed] tbody tr', (rows) => Object.fromEntries(rows.map((x) => [x.dataset.row, x.children[2].textContent.trim()])));
check('NVLink both directions is 1,800 GB/s', (await cells2025()).nvlink === '1,800 GB/s', (await cells2025()).nvlink);
await seg('NVLink counted', 'each way');
check('NVLink each way is 900 GB/s', (await cells2025()).nvlink === '900 GB/s', (await cells2025()).nvlink);
check('NVLink each way is about 14 times a Gen5 x16 link', (await order()).includes('14.3 times a Gen5 x16 host link'), await order());
// Bars are on a log scale and the host link's is the shortest.
const bars = await page.$$eval('[data-recomputed] tbody tr', (rows) => Object.fromEntries(rows.filter((x) => x.querySelector('[data-bar]')).map((x) => [x.dataset.row, x.querySelector('[data-bar]').getBoundingClientRect().width])));
check('the host-link bar is the shortest of five', Object.keys(bars).length === 5 && Object.entries(bars).every(([k, w]) => k === 'pcie' || w > bars.pcie * 1.5), JSON.stringify(bars));
check('bar lengths follow log10 of the ratio', Math.abs(bars.memcap / bars.pcie - Math.log10(120) / Math.log10(63.0153846 / 4)) < 0.03, JSON.stringify(bars));
// Selecting a row shows how its number is built.
const select = async (row) => {
  await page.$eval(`[data-recomputed] tr[data-row="${row}"] input[type=radio]`, (b) => b.click());
  await pause(80);
  return squash(await text(bench(T) + ' [data-readout="built"]'));
};
let b = await select('fp32');
check('compute: 128 x 2 x 1.35 GHz = 345.6 GFLOPS', b.includes('128 multiply-adds per clock × 2 × 1.35 GHz = 345.6 GFLOPS') && b.includes('600 / 8 = 75 TFLOPS'), b);
b = await select('membw');
check('memory bandwidth of the C870: 76.8 GB/s at 800 MHz', b.includes('384 bits × 2 transfers per clock × 800 MHz / 8 = 76.8 GB/s'), b);
b = await select('pcie');
check('PCIe: line rate x encoding x 16 lanes, 4.0 and 63.0 GB/s', b.includes('2.5 GT/s × 8/10 encoding × 16 lanes / 8 = 4.0 GB/s') && b.includes('32 GT/s × 128/130 encoding × 16 lanes / 8 = 63.0 GB/s'), b);
b = await select('tensor');
check('the tensor-core row is TF32, dense when asked', b.includes('1.1 petaFLOPS, TF32 on Tensor Cores, dense'), b);
await shot('explorer-vendor', bench(T) + ' [data-explorer]');
await seg('Figures', 'the book');
b = await select('membw');
check('the book set: 86.4 GB/s at 900 MHz, another board', b.includes('900 MHz / 8 = 86.4 GB/s') && b.includes('768 MB GeForce 8800 GTX'), b);
b = await select('pcie');
check('the book set returns to the printed Gen2: 8.0 GB/s', b.includes('5 GT/s × 8/10 encoding × 16 lanes / 8 = 8.0 GB/s') && (await ratios()).pcie === '8x', b);

q = await question('What 2025 figure would make 638x correct');
await (await q.$('input[type=number]')).type('220');
await (await q.$('button.primary')).click();
await pause(150);
check('q-638: 220 TFLOPS would make the printed ratio right', (await q.evaluate((e) => e.textContent)).includes('Correct.'));

// ---- 4. where the threshold went ----
const K = 'Where the threshold went';
check('the strip is hidden until the prediction is committed', !!(await page.$(bench(K) + ' [data-gate="knee"]')) && !(await page.$(bench(K) + ' svg')));
q = await question('What is the threshold in the 2025 column');
await (await q.$('input[type=number]')).type('2');
await (await q.$('button.primary')).click();
await pause(200);
pred = squash(await q.evaluate((e) => e.textContent));
check('p-knee reveals 10 FLOP/B, 2.5 times, 275, and reacts to a guess below 4', pred.includes('= 10 FLOP/B') && pred.includes('2.5 times the 2007') && pred.includes('it is 275') && pred.includes('upside down'), pred);
await page.waitForSelector(bench(K) + ' svg');
const knee = async () => squash(await text(bench(K) + ' [data-readout="knee"]'));
const need = () => text(bench(K) + ' [data-need]');
const regimeOf = async (id) => squash(await text(`${bench(K)} [data-mark-row="${id}"]`));
const device = async (key) => {
  await page.select(bench(K) + ' select', key);
  await pause(100);
};
let k = await knee();
check('2007 column: threshold 3.99 FLOP/B, smallest T is 16', k.includes('threshold 3.99 FLOP/B') && (await need()) === '16', k);
check('T = 15 is memory-bound on the 2007 figures', k.includes('At T = 15 the tiled kernel reaches 3.75 FLOP/B: memory-bound'), k);
await setRange(bench(K) + ' input[type=range]', 16);
await pause(100);
k = await knee();
check('T = 16 is exactly enough', k.includes('At T = 16 the tiled kernel reaches 4 FLOP/B: compute-bound'), k);
const batch1 = [await regimeOf('lin1')];
await shot('strip-2007', bench(K));
await device('book2025');
k = await knee();
check('2025 column, FP32: threshold 10, smallest T is 40, 2.5 times the 2007 value', k.includes('threshold 10 FLOP/B') && (await need()) === '40' && k.includes("2.5 times the 2007 column's 3.99"), k);
check('a 128 x 128 output tile clears the FP32 threshold', (await regimeOf('out128')).includes('32compute-bound') || (await regimeOf('out128')).includes('32 compute-bound'), await regimeOf('out128'));
batch1.push(await regimeOf('lin1'));
await device('book2025tc');
k = await knee();
check('2025 column, tensor-core figure: threshold 275, smallest T is 1,100, about 69 times', k.includes('threshold 275 FLOP/B') && (await need()) === '1,100' && k.includes("69 times the 2007 column's") && k.includes('far off this slider'), k);
await setRange(bench(K) + ' input[type=range]', 32);
await pause(100);
check("neither chapter 5's 32-wide tile nor chapter 15's output tile reaches it", (await knee()).includes('reaches 8 FLOP/B: memory-bound') && (await regimeOf('out128')).includes('memory-bound'), await regimeOf('out128'));
batch1.push(await regimeOf('lin1'));
await shot('strip-tensor', bench(K));
await page.$eval(bench(K) + ' label.check input', (e) => e.click());
await pause(100);
k = await knee();
check("dense TF32 from NVIDIA's documents: threshold 137.5, smallest T is 550", k.includes('threshold 137.5 FLOP/B') && (await need()) === '550', k);
batch1.push(await regimeOf('lin1'));
await device('ch5');
k = await knee();
check("the book's chapter 5 example: threshold 20, smallest T is 80", k.includes('threshold 20 FLOP/B') && (await need()) === '80', k);
batch1.push(await regimeOf('lin1'));
check('batch-1 generation is memory-bound on every published device', batch1.length === 5 && batch1.every((s) => s.includes('memory-bound')), batch1.join(' | '));
check('batch 512 is compute-bound on the chapter 5 example', (await regimeOf('lin512')).includes('compute-bound'));
// "Yours" is offered only for a real backend that lab 5.1 has benchmarked.
const roof = (bk) => reports[bk].results.find((x) => x.lab === 'ch05/lab1_roofline' && x.variant === 'solution')?.bench.find((x) => x.name === 'roofline');
const mine = real.filter((bk) => roof(bk)?.compute_gflops && roof(bk)?.bandwidth_gb_per_s);
const offered = await page.$$eval(bench(K) + ' select option', (o) => o.map((x) => x.value));
check('device list: four sets of published figures plus each measured backend', offered.join() === ['book2007', 'book2025', 'book2025tc', 'ch5', ...mine].join(), offered.join());
if (mine.length) {
  const row = roof(mine[0]);
  const thr = row.compute_gflops / row.bandwidth_gb_per_s;
  await device(mine[0]);
  k = await knee();
  check('my device: threshold is the measured compute over the measured bandwidth of lab 5.1', k.includes(`threshold ${Number(thr.toPrecision(3))} FLOP/B`) && (await need()) === String(Math.ceil(thr * 4)), k);
  check('my device is labelled as measured, not a vendor peak', squash(await text(bench(K))).includes('It is not a vendor peak'));
  check('batch-1 generation is memory-bound on my device too', (await regimeOf('lin1')).includes('memory-bound') === thr > 0.9988);
  await shot('strip-mine', bench(K));
} else {
  check('with no measurement the strip says so', (await knee()).includes('lab 5.1 has not been benchmarked'));
}

q = await question('How wide must T be');
await (await q.$('input[type=number]')).type('1100');
await (await q.$('button.primary')).click();
await pause(150);
const qt = squash(await q.evaluate((e) => e.textContent));
check('q-tile: T = 1100, about 9.7 MB of shared memory, 16 on the 2007 figures', qt.includes('Correct.') && qt.includes('about 9.7 MB') && qt.includes('the answer was 16'), qt);

// ---- the backends table and its question ----
const rows = await page.$$eval('table[data-backends] tbody tr', (rs) => rs.map((x) => x.textContent));
check('the backends table has five rows and three backends', rows.length === 5 && (await page.$$('table[data-backends] thead th')).length === 5);
const all = rows.join(' ');
check('identifiers are spelled as in the facts file', ['cuda::tiles', 'cuda_tile.h', 'mpp::tensor_ops::matmul2d', '<metal_simdgroup_matrix>', 'ncclXxx', 'mx.distributed', 'MFMA', 'Infinity Fabric'].every((s) => all.includes(s)), all);
check('no AMD intrinsic spelling is printed', !(await text('.prose')).includes('__builtin_amdgcn'));
q = await question('has no counterpart on Apple silicon');
await (await q.$$('input[type=radio]'))[2].click();
await (await q.$('button.primary')).click();
await pause(600);
check('q-norow: the host-to-device link', (await q.evaluate((e) => e.textContent)).includes('Correct.'));

// ---- page rules ----
const afterBench = await page.$$eval('.prose > .note', (ns) => ns.filter((n) => n.previousElementSibling?.tagName !== 'P').map((n) => n.textContent.slice(0, 40)));
check('every margin note directly follows a paragraph', afterBench.length === 0, afterBench.join(' | '));
check('the page has no code block and no narrated scene', !(await page.$('.prose .code')) && !(await page.$('.prose .scene')));
check('the page links to the wrap page', !!(await page.$('.prose a[href="#/ch24/wrap"]')));

// ---- progress ----
const last = puts[puts.length - 1] ?? { answers: {}, review: {} };
const A = last.answers;
check('predictions are recorded under their ids', A['ch24/start/p-hostlink']?.value === 100 && A['ch24/start/p-knee']?.value === 2, JSON.stringify([A['ch24/start/p-hostlink'], A['ch24/start/p-knee']]));
check('question answers are recorded under ch24/start ids', ['q-compose', 'q-638', 'q-tile', 'q-norow'].every((id) => A['ch24/start/' + id]?.correct === true), JSON.stringify(Object.keys(A)));
check('each classification card is recorded as its own question', ['load', 'decode', 'prefill', 'fit', 'allreduce'].every((id) => A[P + id]?.correct === true) && A[P + 'load'].value === 3 && A[P + 'decode'].value === 1 && A[P + 'decode'].attempts === 2 && A[P + 'load'].attempts === 1, JSON.stringify(A[P + 'decode']));
check('a misplaced card enters the review schedule as a lapse, a right one without', puts.some((p) => p.answers?.[P + 'decode']?.correct === false) && last.review?.[P + 'decode']?.lapses === 1 && last.review?.[P + 'load']?.lapses === 0 && last.review?.[P + 'load']?.box === 0, JSON.stringify([last.review?.[P + 'decode'], last.review?.[P + 'load']]));
check('nothing was written outside ch24/start', Object.keys(A).every((id) => id.startsWith('ch24/start/')) && Object.keys(A).length === 11, Object.keys(A).join());

// Nothing makes the page itself scroll sideways, at any width.
check('the page does not scroll sideways', await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), String(await page.evaluate(() => document.documentElement.scrollWidth)));
if (out) await page.screenshot({ path: `${out}-page.png`, fullPage: true });
check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
