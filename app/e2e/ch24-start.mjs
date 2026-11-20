// Browser check for chapter 24's start page: the prerequisite check ("what limits what"),
// the steps map, the audit of Fig. 24.1 with its explorer, and the threshold strip.
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
// GPU backends only: emu and race are both the CPU emulator.
const real = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const narrow = Number(process.env.E2E_WIDTH || 1440) <= 600;

await goto(page, 'ch24/start');
if (theme) await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, theme);

// ---- 0. page order: the prerequisite check comes first, as on the template's start pages ----
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
check('sections in order, starting with the prerequisite check', JSON.stringify(h2) === JSON.stringify(['Before you start', 'What you can now do', "The book's one table", 'Reading the table as a kernel writer', 'The same rows on AMD and Apple', 'The rest of the outlook']), JSON.stringify(h2));

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

let q = await question('min-p cut-off itself need');
await (await q.$$('input[type=radio]'))[0].click();
await (await q.$('button.primary')).click();
await pause(150);
let qc = squash(await q.evaluate((e) => e.textContent));
check('q-compose: "sort then scan, as top-p" is wrong, and says why', qc.includes('Not quite.') && qc.includes('depends on one number, the largest probability'), qc);
await (await q.$$('input[type=radio]'))[1].click();
await (await q.$('button.primary')).click();
await pause(150);
qc = squash(await q.evaluate((e) => e.textContent));
check('q-compose: min-p needs a reduction and a filter, no sort; labelled as ours', qc.includes('Correct.') && qc.includes('No sort is needed') && qc.includes('Min-p is not in the book'), qc);

// ---- 2. what limits what (the prerequisite check) ----
const L = 'What limits what';
const place = async (qid, value) => {
  await page.select(`${bench(L)} [data-card="${qid}"] select`, value);
  await pause(40);
};
const P = 'ch24/start/q-limit-';
const froms = await page.$$eval(`${bench(L)} [data-card] [data-from]`, (f) => f.map((x) => x.textContent.trim()));
check('each card is labelled with the chapters it draws on', froms.join('|') === 'From chapter 2|From chapters 5 and 20|From chapter 20|From chapter 20|From chapter 23', froms.join('|'));
check('the all-reduce card pins the topology to NVLink', (await text(`${bench(L)} [data-card="${P}allreduce"]`)).includes('joined to each other by NVLink'));
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
const proseBefore = squash(await text('.prose'));
check('nothing outside the gate gives the host-link answer or the audit away', !proseBefore.includes('16x') && !proseBefore.includes('contradicts itself twice') && !(await question('what 2007 throughput would make 638x correct')).asElement(), '');
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
check('the task says what the audit checks', squash(await text(bench(T) + ' header')).includes('Tick every cell the table itself contradicts: a ratio its two figures do not give, or a unit that cannot be meant'));
check('before Check, the discussion below waits for it', !!(await page.$('.prose [data-gate="audit"]')) && !(await page.$('.prose [data-after-audit]')));
const tick = (id) => page.$eval(`[data-printed-table] [data-cell="${id}"] input`, (e) => e.click());
for (const id of ['fp32.ratio', 'tensor.b200', 'tensor.ratio', 'membw.ratio', 'memcap.ratio', 'pcie.g80']) await tick(id);
await clickText(T, 'Check');
r = squash(await text(bench(T) + ' [data-readout="audit"]'));
check('the table contradicts itself in two cells; one found; of five other ticks two are corrected, three check out', r.includes('The table contradicts itself in two cells.') && r.includes('You found 1 of 2') && r.includes("ticked 5 other cells: 2 that NVIDIA's documents correct and 3 that check out."), r);
const byState = await page.$$eval('[data-printed-table] [data-cell]', (c) => {
  const o = {};
  for (const x of c) if (x.dataset.state) (o[x.dataset.state] ??= []).push(x.dataset.cell);
  return o;
});
check('red: the 638x ratio and the TB/s unit, and nothing else', byState.wrong?.join() === 'tensor.ratio,nvlink.b200', JSON.stringify(byState));
check("neutral, not green: the 2200 label and the Gen2 host link, which NVIDIA's documents correct", byState.differs?.join() === 'tensor.b200,pcie.g80', JSON.stringify(byState));
check('green: only the three consistent ratios ticked', byState.consistent?.join() === 'fp32.ratio,membw.ratio,memcap.ratio', JSON.stringify(byState));
const washes = await page.$$eval('[data-printed-table] [data-cell]', (c) => Object.fromEntries(c.filter((x) => x.dataset.state).map((x) => [x.dataset.cell, x.style.background])));
check('a corrected cell is never washed green', !washes['pcie.g80'].includes('ok-wash') && !washes['tensor.b200'].includes('ok-wash') && washes['fp32.ratio'].includes('ok-wash'), JSON.stringify(washes));
const v = squash(await text(bench(T) + ' [data-verdicts]'));
for (const a of ['80 / 0.345 = 232', '8000 / 86.4 = 93', '192 / 1.5 = 128', '64 / 8 = 8', '2200 / 0.345 = 6,377'])
  check(`a ticked cell gets its arithmetic: ${a}`, v.includes(a), v);
check('638x would need 220 TFLOPS', squash(await text(`${bench(T)} [data-verdict="tensor.ratio"]`)).includes('would need 220 TFLOPS'));
const tcv = squash(await text(`${bench(T)} [data-verdict="tensor.b200"]`));
check('the 2200 figure stands, its label does not (A9)', tcv.includes("NVIDIA's documents differ") && tcv.includes('the figure stands and the ratio is the error') && tcv.includes('TF32 on Tensor Cores') && tcv.includes('[facts A9]'), tcv);
const pcv = squash(await text(`${bench(T)} [data-verdict="pcie.g80"]`));
check('the Gen2 cell: arithmetic holds, NVIDIA says generation 1 (A6)', pcv.startsWith("NVIDIA's documents differ") && pcv.includes("The table's arithmetic holds") && pcv.includes('Generation 1, 4 GB/s each way') && pcv.includes('[facts A6]'), pcv);
const missed = squash(await text(`${bench(T)} [data-verdict="nvlink.b200"]`));
check('the unit error is listed as missed, with its arithmetic', missed.startsWith('missed') && missed.includes('1,800,000 GB/s / 8,000 GB/s = 225') && missed.includes('1,800 GB/s'), missed);
check('four printed ratios follow from their figures, one is off by ten; five cells are corrected', v.includes('4 of the 5 printed ratios follow') && v.includes('Off by ten') && v.includes("NVIDIA's documents correct 5 cells that are not contradictions"), v.slice(-400));
await shot('audit', bench(T));
// What the audit finds is discussed below it only now.
const after = squash(await text('.prose [data-after-audit]'));
check('after Check, the discussion counts two contradictions and two corrections', after.includes('The table contradicts itself twice') && squash(await text('.prose')).includes('Two more cells are wrong against NVIDIA') && squash(await text('.prose')).includes('about two orders of magnitude or more'), after);

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
const differs = await page.$$eval('[data-recomputed] [data-printed][data-differs]', (c) => c.map((x) => x.closest('tr').dataset.row + ':' + x.dataset.differs));
check('none of the five printed ratios survives NVIDIA\'s figures', differs.length === 5, differs.join());
check('only 638x, which its own figures contradict, takes the failure colour; the other four are marked neutrally', (await page.$$('[data-recomputed] [data-printed] .t-fail')).length === 1 && differs.join() === 'fp32:figures,tensor:contradicted,membw:figures,memcap:figures,pcie:figures', differs.join());
check('the explorer says what its marks mean', (await order()).includes('“≠” marks a printed ratio that follows from the book\'s figures but not from the ones selected'), await order());
check('the host link still grew least', (await order()).includes('Least growth: host link (PCIe x16, each way), 16x'), await order());
await seg('Tensor-core figure', 'dense');
check('dense TF32 gives 3,183x', (await ratios()).tensor === '3,183x', (await ratios()).tensor);
await seg('PCIe generation of the 2007 board', '2');
check('a Gen2 board against the Gen5 rate gives 8x', (await ratios()).pcie === '8x', (await ratios()).pcie);
await seg('PCIe generation of the 2007 board', '1');
const cells2025 = () => page.$$eval('[data-recomputed] tbody tr', (rows) => Object.fromEntries(rows.map((x) => [x.dataset.row, x.querySelector('[data-to]').textContent.trim()])));
check('NVLink both directions is 1,800 GB/s', (await cells2025()).nvlink === '1,800 GB/s', (await cells2025()).nvlink);
await seg('NVLink counted', 'each way');
check('NVLink each way is 900 GB/s', (await cells2025()).nvlink === '900 GB/s', (await cells2025()).nvlink);
check('NVLink each way is about 14 times a Gen5 x16 link', (await order()).includes('14.3 times a Gen5 x16 host link'), await order());
// Bars are on a log scale and the host link's is the shortest.
const bars = await page.$$eval('[data-recomputed] tbody tr', (rows) => Object.fromEntries(rows.filter((x) => x.querySelector('[data-bar]')).map((x) => [x.dataset.row, x.querySelector('[data-bar]').getBoundingClientRect().width])));
check('the host-link bar is the shortest of five', Object.keys(bars).length === 5 && Object.entries(bars).every(([k, w]) => k === 'pcie' || w > bars.pcie * 1.5), JSON.stringify(bars));
check('bar lengths follow log10 of the ratio', Math.abs(bars.memcap / bars.pcie - Math.log10(120) / Math.log10(63.0153846 / 4)) < 0.03, JSON.stringify(bars));
// Growth, printed ratio and bar stay in view: the table fits its panel at this width.
const fit = await page.$eval('[data-recomputed]', (t) => ({ table: t.scrollWidth, panel: t.parentElement.clientWidth, growth: t.querySelector('tbody tr [data-ratio]').getBoundingClientRect().right, printed: t.querySelector('tbody tr [data-printed]').getBoundingClientRect().right, view: window.innerWidth }));
check(`the explorer table fits its panel at ${narrow ? 'phone' : 'desktop'} width`, fit.table <= fit.panel + 1 && fit.printed <= fit.view, JSON.stringify(fit));
check(narrow ? 'at phone width the 2007 and 2025 figures sit under the row name' : 'at desktop width the 2007 and 2025 figures have their own columns',
  (await page.$$('[data-recomputed] thead th')).length === (narrow ? 4 : 6), String((await page.$$('[data-recomputed] thead th')).length));
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

q = await question('what 2007 throughput would make 638x correct');
await (await q.$('input[type=number]')).type('3.45');
await (await q.$('button.primary')).click();
await pause(150);
const q638 = squash(await q.evaluate((e) => e.textContent));
check('q-638: 3.45 TFLOPS in 2007 would make the printed ratio right; both inputs stand, so the ratio is the error', q638.includes('Correct.') && q638.includes('2200 / 638 = 3.45 TFLOPS') && q638.includes('345.6 GFLOPS') && q638.includes('6,377x'), q638);
check('the audit and its explorer do not print the answer to q-638', !squash(await text(bench(T))).includes('3.45'));

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
check('batch-1 generation is memory-bound in every column', batch1.length === 4 && batch1.every((s) => s.includes('memory-bound')), batch1.join(' | '));
// The strip keeps to the book's columns; other devices are chapter 5's roofline.
const offered = await page.$$eval(bench(K) + ' select option', (o) => o.map((x) => x.value));
check("device list: the book's 2007 column and its two 2025 figures, nothing else", offered.join() === 'book2007,book2025,book2025tc', offered.join());
check('the strip points to the roofline of chapter 5 for other devices', !!(await page.$(`${bench(K)} footer a[href="#/ch05/5.1"]`)));
check('the strip says the 16-bit marks are indicative', squash(await text('.prose')).includes('two linear-layer marks are 16-bit kernels set against FP32 and TF32 peaks'));
// The whole strip, threshold and compute-bound side included, fits its box at this width.
const strip = await page.$eval(bench(K) + ' [data-strip]', (s) => {
  const r = s.getBoundingClientRect();
  const label = s.querySelector('[data-threshold-label]').getBoundingClientRect();
  const fs = parseFloat(getComputedStyle(s.querySelector('[data-mark] text')).fontSize) * (r.width / s.viewBox.baseVal.width);
  return { w: r.width, box: s.parentElement.clientWidth, labelRight: label.right, right: r.right, fontPx: fs };
});
check(`the strip fits its box and its threshold label is in view (${narrow ? 'phone' : 'desktop'})`, strip.w <= strip.box + 1 && strip.labelRight <= strip.right + 1, JSON.stringify(strip));
check('strip labels stay legible (at least 9 px)', strip.fontPx >= 9, JSON.stringify(strip));

q = await question('How many megabytes would one block need');
check('q-tile states T = 1,100 and asks for something the strip does not show', squash(await q.evaluate((e) => e.textContent)).includes('reaches only at T = 1,100'));
await (await q.$('input[type=number]')).type('9.68');
await (await q.$('button.primary')).click();
await pause(150);
const qt = squash(await q.evaluate((e) => e.textContent));
check('q-tile: 2 x 1,100 x 1,100 x 4 bytes = 9.68 MB; 2.4 MB at T = 550; 2 KB at T = 16', qt.includes('Correct.') && qt.includes('= 9.68 MB per block') && qt.includes('T = 550 still needs 2.4 MB') && qt.includes('T = 16 needed 2 KB'), qt);

// ---- the backends table and its question ----
const rows = await page.$$eval('table[data-backends] tbody tr', (rs) => rs.map((x) => x.textContent));
check('the backends table has five rows and three backends', rows.length === 5 && (await page.$$('table[data-backends] thead th')).length === 5);
const all = rows.join(' ');
check('identifiers are spelled as in the facts file', ['cuda::tiles', 'cuda_tile.h', 'mpp::tensor_ops::matmul2d', '<metal_simdgroup_matrix>', 'ncclXxx', 'mx.distributed', 'MFMA', 'Infinity Fabric'].every((s) => all.includes(s)), all);
check("Apple's “Neural Accelerators” are named, with no link to MSL stated (C1)", all.includes('Neural Accelerators') && all.includes('no link to MSL stated'), all);
check("the book's own §15.9 names WMMA and WGMMA for CUDA", all.includes("The book's §15.9 (p. 369) also names the WMMA and WGMMA instructions"), all);
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
