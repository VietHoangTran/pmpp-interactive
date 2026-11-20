// Browser check for section 9.5: the "Blocks, copies, commits" calculator, the
// "Who reads what in iteration c" explorer, the listing, a prediction and questions.
// Expected values come from the model (src/sim/histcoarsen.ts), imported directly.
// Usage: node e2e/ch09-s95.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { EXERCISE5, index, kernelTotals, plan, warpFootprint } from '../src/sim/histcoarsen.ts';

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
/** Set a <select> or <input> inside a bench by its aria-label; ranges need the native setter and an input event. */
const setField = async (scope, aria, value) => {
  const ok = await page.evaluate(
    (scope, aria, value) => {
      const root = [...document.querySelectorAll('section.bench')].find((b) => b.getAttribute('aria-label') === scope);
      const el = root.querySelector(`[aria-label="${aria}"]`);
      if (!el) return false;
      const proto = el.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, String(value));
      el.dispatchEvent(new Event(el.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true }));
      return true;
    },
    scope,
    aria,
    value,
  );
  await pause(120);
  return ok;
};
const readout = (scope) => page.$eval(bench(scope) + ' [data-readout]', (e) => ({ elements: e.dataset.elements, transactions: Number(e.dataset.transactions) }));

await goto(page, 'ch09/9.5');
await page.waitForSelector(bench('Who reads what in iteration c'));

// ---------------------------------------------------------------- calculator
const P = 'Blocks, copies, commits';
check('calculator: exercise 5 preset exists', await clickText(P, 'Exercise 5 (p. 220)'));
const planRows = async () =>
  page.$$eval(bench(P) + ' tr[data-kernel]', (rs) => Object.fromEntries(rs.map((r) => [r.dataset.kernel, { blocks: Number(r.dataset.blocks), global: Number(r.dataset.global) }])));
let rows = await planRows();
check('exercise 5a: no privatisation, 524,288 global atomics', rows['9.6']?.global === 524288, JSON.stringify(rows));
check('exercise 5b: factor 1, 65,536', rows['9.10']?.global === 65536, JSON.stringify(rows));
check('exercise 5c: factor 4, 16,384 in 128 blocks', rows['9.14']?.global === 16384 && rows['9.14']?.blocks === 128, JSON.stringify(rows));
check('the page shows 16,384', (await page.$eval(bench(P), (e) => e.textContent)).includes('16,384'));
await setField(P, 'coarse factor', 16);
rows = await planRows();
const p16 = plan(EXERCISE5.n, EXERCISE5.blockDim, 16, EXERCISE5.bins);
check('factor 16 matches the model', rows['9.14']?.global === p16.commitAtomicsMax && rows['9.14']?.blocks === p16.blocks, JSON.stringify(rows['9.14']));
await setField(P, 'bins', 4096);
await setField(P, 'coarse factor', 1);
rows = await planRows();
check('a block commits no more bins than it has pixels (1024 pixels, 4096 bins)', rows['9.10']?.global === plan(EXERCISE5.n, 1024, 1, 4096).commitAtomicsMax && rows['9.10'].global === 512 * 1024, JSON.stringify(rows['9.10']));

// ---------------------------------------------------------------- explorer
const X = 'Who reads what in iteration c';
const range = (a, b, step = 1) => Array.from({ length: Math.floor((b - a) / step) + 1 }, (_, k) => a + k * step).join(',');
await clickText(X, 'Figs. 9.11 and 9.13');
await clickText(X, 'Contiguous (Fig. 9.12)');
let r = await readout(X);
check('Fig. 9.11, first iteration: cells 0, 3, ..., 21', r.elements === range(0, 21, 3), r.elements);
await clickText(X, 'Step');
r = await readout(X);
check('Fig. 9.11, second iteration: cells 1, 4, ..., 22', r.elements === range(1, 22, 3), r.elements);
const book = { n: 24, blockDim: 8, coarse: 3, block: 0, warp: 0, warpWidth: 32, elemBytes: 1, segmentBytes: 32 };
check('drawn cells agree with the model', r.elements === warpFootprint({ ...book, p: 'contiguous', c: 1 }).elements.join(','));
const hot = await page.$$eval(bench(X) + ' g[data-hot="1"]', (gs) => gs.map((g) => g.dataset.cell).join(','));
check('the strip highlights the same cells', hot === range(1, 22, 3), hot);
if (out) await page.screenshot({ path: out + '-fig911.png' });
await clickText(X, 'Interleaved (Fig. 9.14)');
await clickText(X, 'c = 0');
r = await readout(X);
check('Fig. 9.13, first iteration: cells 0 to 7', r.elements === range(0, 7), r.elements);
await clickText(X, 'Step');
r = await readout(X);
check('Fig. 9.13, second iteration: cells 8 to 15', r.elements === range(8, 15), r.elements);

// At the drawing's scale both orders fetch the same single segment, and the text says so.
const tot0 = await page.$$eval(bench(X) + ' tr[data-p]', (rs) => Object.fromEntries(rs.map((x) => [x.dataset.p, Number(x.dataset.transactions)])));
check('drawing preset: both orders fetch the same segments (3 loads of one 32-byte segment)', tot0.contiguous === 3 && tot0.interleaved === 3 && tot0.contiguous === kernelTotals({ ...book, p: 'contiguous' }).transactions, JSON.stringify(tot0));
const prose95 = await page.$eval('.prose', (e) => e.textContent.replace(/\s+/g, ' '));
check('the text says the drawing shows no cost difference, and where it appears', prose95.includes('At this scale the two orders still cost the same') && prose95.includes('The difference appears with the second preset'));
check('the CPU section says a Mac may go either way', prose95.includes('on a Mac lab 9.2\'s measurement may go either way'));
const notes95 = await page.$$eval('.note', (n) => n.map((x) => x.className + ': ' + x.textContent));
check('the calculator and the explorer each state what they simplify', notes95.some((t) => t.includes('caveat') && t.includes('The last column is a bound')) && notes95.some((t) => t.includes('caveat') && t.includes('Aligned segments per warp-wide load')));
const misplaced = await page.$$eval('.prose > .note', (ns) => ns.filter((n) => n.previousElementSibling?.tagName !== 'P').map((n) => n.textContent.slice(0, 60)));
check('every margin note follows a paragraph', misplaced.length === 0, misplaced.join(' | '));

// Click a cell: its owner's line 12 is spelled out.
await page.evaluate((sel) => document.querySelector(sel + ' g[data-cell="13"]').dispatchEvent(new MouseEvent('click', { bubbles: true })), bench(X));
await pause(100);
let formula = await page.$eval(bench(X) + ' [data-formula]', (e) => ({ v: Number(e.dataset.formula), t: e.textContent }));
check('clicking cell 13 picks thread 5, whose iteration-1 element is 13', formula.v === index('interleaved', 0, 5, 1, 8, 3) && formula.v === 13 && formula.t.includes('Thread 5'), formula.t);

// The coalescing discovery: one-byte pixels at factor 32.
await clickText(X, 'One-byte pixels, factor 32');
await clickText(X, 'Contiguous (Fig. 9.12)');
r = await readout(X);
check('contiguous, factor 32: 32 segments for 32 one-byte lanes', r.transactions === 32, String(r.transactions));
await clickText(X, 'Interleaved (Fig. 9.14)');
r = await readout(X);
check('interleaved, factor 32: 1 segment', r.transactions === 1, String(r.transactions));
const tot = await page.$$eval(bench(X) + ' tr[data-p]', (rs) => Object.fromEntries(rs.map((x) => [x.dataset.p, Number(x.dataset.transactions)])));
const machine = { n: 1 << 20, blockDim: 256, coarse: 32, warpWidth: 32, elemBytes: 1, segmentBytes: 32 };
check('whole-kernel totals match the model', tot.contiguous === kernelTotals({ ...machine, p: 'contiguous' }).transactions && tot.interleaved === kernelTotals({ ...machine, p: 'interleaved' }).transactions, JSON.stringify(tot));
check('contiguous fetches 32 times as many segments', tot.contiguous === 32 * tot.interleaved, JSON.stringify(tot));
await setField(X, 'coarse factor', 2);
await clickText(X, 'Contiguous (Fig. 9.12)');
r = await readout(X);
check('contiguous, factor 2: lanes two bytes apart need 2 segments', r.transactions === 2, String(r.transactions));
if (out) await page.screenshot({ path: out + '-factor.png' });

// Vendor choices: a 64-lane warp, 4-byte ids, AMD's 128-byte lines, interleaved: ceil(64·4/128) = 2.
await clickText(X, 'Interleaved (Fig. 9.14)');
await setField(X, 'warp width', 64);
await setField(X, 'element size', 4);
await setField(X, 'segment', 'amd128');
r = await readout(X);
check('64 lanes of 4-byte ids in 128-byte lines: 2 segments', r.transactions === 2, String(r.transactions));
await setField(X, 'segment', 'apple');
check('Apple: the segment size becomes the reader’s choice', (await page.$(bench(X) + ' [aria-label="segment bytes"]')) !== null);
await setField(X, 'warp width', 0);
await setField(X, 'custom warp width', 16);
r = await readout(X);
check('a typed warp width of 16 gives 16 lanes', (await page.$eval(bench(X) + ' [data-readout]', (e) => Number(e.dataset.active))) === 16);

// ---------------------------------------------------------------- listing
const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent));
const missing = await page.$('[data-listing="missing"]');
if (missing) check('listing: lab 9.2 not in place yet (says so)', true);
else {
  check('listing shows interleaved, contiguous and Metal tabs, with short labels', ['Interleaved', 'Contiguous', 'Metal'].every((t) => tabs.includes(t)), tabs.join('|'));
  const hl = await page.$eval('.code .ln.hl', (e) => e.textContent);
  check('interleaved tab highlights line 12 of Fig. 9.14', hl.includes('i = segment + c*blockDim.x + threadIdx.x'), hl);
  await page.evaluate(() => [...document.querySelectorAll('.code .tabs button')].find((b) => b.textContent === 'Contiguous').click());
  await pause(100);
  const hl2 = await page.$$eval('.code .ln.hl', (e) => e.map((x) => x.textContent).join('|'));
  check('contiguous tab highlights line 12 of Fig. 9.12, and only it', hl2.includes('i = segment + threadIdx.x*coarseFactor + c') && !hl2.includes('|'), hl2);
}

// ---------------------------------------------------------------- questions and prediction
const answer = async (needle, value) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
check('q-index-interleaved accepts 2821', (await answer('Which element does thread 5 of block 2 read', 2821)).includes('Correct.'));
check('q-ex5c accepts 16384', (await answer('Exercise 5c: 524,288 elements', 16384)).includes('Correct.'));
check('q-segments rejects 1 and gives 32', (await answer('How many segments does one warp-wide load touch', 1)).includes('The answer is 32'));
const choice = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('so the whole image is one')));
await choice.evaluate((e) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes('rest of the device idles')).querySelector('input').click());
await (await choice.$('button.primary')).click();
await pause(120);
check('q-overcoarsen: the idle-device answer is right', (await choice.evaluate((e) => e.textContent)).includes('Correct.'));

const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Predict first') && e.textContent.includes('COARSE_FACTOR')));
await (await pred.$('input[type=number]')).type('3');
await (await pred.$('button.primary')).click();
await page.waitForFunction((e) => e.querySelector('[data-gain]'), {}, pred);
const gain = await pred.evaluate((e) => ({ t: e.textContent, g: e.querySelector('[data-gain]').dataset.gain }));
check('prediction committed, then a measurement or a plain "no measurement yet"', gain.t.includes('You predicted 3') && (gain.g === 'none' ? gain.t.includes('No measurement on this machine yet') : Number(gain.g) > 0), gain.t.slice(0, 200));
// Direction first: a prediction above 1 and a measurement below 1 are never called close.
const dir = await pred.evaluate((e) => [...e.querySelectorAll('[data-gain]')].filter((p) => p.dataset.gain !== 'none').map((p) => ({ g: Number(p.dataset.gain), d: p.dataset.direction, t: p.textContent })));
for (const x of dir)
  check(`measured gain ${x.g.toFixed(3)} against a prediction of 3: direction "${x.d}" and the matching sentence`,
    x.g < 1 ? x.d === 'opposite' && x.t.includes('Contiguous won on this device') && !x.t.includes('close to what you predicted') : x.d === 'same', x.t.slice(0, 300));

// ---------------------------------------------------------------- narrow screen
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
const wide = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
check('no horizontal page scroll at 390 px', wide <= 0, `${wide}px too wide`);

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
