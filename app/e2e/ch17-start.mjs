// Browser check for chapter 17, "Before you start": the redrawn Fig. 17.1 (compared with
// the model it is drawn from, src/sim/sparseroute.ts, imported here directly), the bytes
// prediction and its reveal (src/sim/sparsescale.ts, which imports other models and is
// therefore loaded inside the page through the dev server), every question, the margin notes and
// their placement, the links of the chapter map, and the page at phone width.
// Usage: node e2e/ch17-start.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { fig171Dense, exclusiveScan, countByRow, FIG171 } from '../src/sim/sparseroute.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });

// Leave no trace in the real progress file, and start from an empty one so that no
// question or prediction is prefilled whatever the reader has already done.
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
const saved = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.url().includes('/api/progress')) {
    if (r.method() === 'PUT') saved.push(r.postData() ?? '');
    return r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : JSON.stringify(EMPTY) });
  }
  return r.continue();
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const qHandle = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const answerNumeric = async (needle, value) => {
  const q = await qHandle(needle);
  const input = await q.$('input[type=number]');
  // React owns the field: set it through the native setter so a previous answer is replaced, not appended to
  await input.evaluate((el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
const answerChoice = async (needle, index) => {
  const q = await qHandle(needle);
  const radios = await q.$$('input[type=radio]');
  await radios[index].click();
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};

await goto(page, 'ch17/start');
await page.waitForSelector('.q');
const M = await page.evaluate(async () => {
  const m = await import('/src/sim/sparsescale.ts');
  return { pruned: m.bytesPerPosition(0.1, 2, 4, Infinity), breakEven: m.breakEvenDensity(0.5, 2, 2) * 100 };
});

// ---------------------------------------------------------------- the figure against the model
const dense = fig171Dense();
const cells = await page.$$eval('figure svg g[data-cell]', (g) => g.map((x) => [x.dataset.cell, Number(x.dataset.value)]));
check('the figure has 16 cells', cells.length === 16, String(cells.length));
check(
  "every cell of the figure holds the model's value (Fig. 17.1)",
  cells.every(([rc, v]) => {
    const [r, c] = rc.split(',').map(Number);
    return dense[r][c] === v;
  }),
  JSON.stringify(cells),
);
const shownValues = await page.$$eval('figure svg g[data-cell] text', (t) => t.map((x) => Number(x.textContent)));
check('the non-zeros are printed in row order as COO lists them', JSON.stringify(shownValues) === JSON.stringify(FIG171.value), JSON.stringify(shownValues));
const fills = await page.$$eval('figure svg g[data-cell]', (g) => g.map((x) => [Number(x.dataset.value), x.querySelector('rect').getAttribute('fill')]));
check(
  'non-zeros use the global-memory colour variable, zeros the plain cell',
  fills.every(([v, f]) => (v ? f === 'var(--global-wash)' : f === 'var(--cell)')),
  JSON.stringify(fills),
);
const rowLabels = await page.$$eval('figure svg g[data-row] > text', (t) => t.map((x) => x.textContent));
check('rows are labelled, not coloured', JSON.stringify(rowLabels) === JSON.stringify(['row 0', 'row 1', 'row 2', 'row 3']), JSON.stringify(rowLabels));
const vectors = await page.$$eval('figure svg g[data-vector]', (g) => g.map((x) => x.dataset.vector));
check('the figure shows x, y and the new y', JSON.stringify(vectors) === JSON.stringify(['x', 'y', 'y']), JSON.stringify(vectors));

// ---------------------------------------------------------------- margin notes and their placement
const notes = await page.$$eval('aside.note', (a) => a.map((x) => x.textContent));
check('SpMV is cited to §17.1, pp. 402–403', notes.some((t) => t.includes('§17.1, pp. 402–403')));
check('the library formula is a dated Today note (A4, A10)', notes.some((t) => t.startsWith('Today') && t.includes('A4, A10') && t.includes('usmv')));
check("the book's warning is cited to p. 422", notes.some((t) => t.includes('pp. 409, 422')));
check('PyTorch is quoted in a dated note (D1)', notes.some((t) => t.startsWith('Today') && t.includes('D1')));
check('the LLM framing is marked as not in the book', notes.some((t) => t.startsWith('Not in the book')));
check('the backends sentence carries B1, C1', notes.some((t) => t.includes('B1, C1')));
const placement = await page.$$eval('.prose > aside.note', (a) => a.map((x) => x.previousElementSibling?.tagName ?? 'none'));
check('every margin note follows a paragraph (no gap after a figure, table or question)', placement.every((t) => t === 'P'), placement.join(','));
const labels = await page.$$eval('.q .kind', (k) => k.map((x) => x.textContent));
check('the bytes prediction comes first', labels[0] === 'Predict first', labels.join('|'));
for (const l of ['Before any format', 'From chapter 4', 'From chapter 6', 'From chapter 9', 'From appendix A', 'From chapter 11'])
  check(`a question labelled "${l}"`, labels.includes(l), labels.join('|'));
const links = await page.$$eval('.prose a', (a) => a.map((x) => x.getAttribute('href')));
for (const l of ['#/ch17/17.2', '#/ch17/17.5', '#/ch17/labs', '#/ch17/wrap', '#/review']) check(`the page links to ${l}`, links.includes(l), links.join('|'));
const crit = await page.$$eval('table[data-criteria] tbody td:first-child', (t) => t.map((x) => x.textContent));
check('the five criteria are listed in the book\'s order', JSON.stringify(crit) === JSON.stringify(['Space efficiency', 'Flexibility', 'Accessibility', 'Memory-access efficiency', 'Load balance']), JSON.stringify(crit));

// ---------------------------------------------------------------- the prediction
{
  const q = await qHandle('pruned to keep 10%');
  const input = await q.$('input[type=number]');
  await input.type('1');
  await (await q.$('button.primary')).click();
  await pause(200);
  const t = await q.evaluate((e) => e.textContent);
  const pruned = await page.$eval('[data-reveal="ch17/start/p-bytes"] [data-out=pruned]', (e) => e.textContent);
  const fewer = await page.$eval('[data-reveal="ch17/start/p-bytes"] [data-out=fewer]', (e) => e.textContent);
  check('the prediction commits and shows what was predicted', t.includes('You predicted 1 bytes.'), t.slice(0, 200));
  check("the reveal gives the model's 0.6 bytes per position", Math.abs(Number(pruned) - M.pruned) < 1e-9, pruned);
  check('the reveal gives 3.3 times fewer than dense fp16', fewer === '3.3', fewer);
  check('the reveal turns to 4-bit dense weights, which win', t.includes('Pruning to 10% loses to it'), t.slice(0, 600));
}

// ---------------------------------------------------------------- the questions
let t = await answerNumeric('Below what density', 25);
check('break-even: 25 is wrong, the answer is 12.5', t.includes('Not quite. The answer is 12.5 %.'), t.slice(0, 300));
t = await answerNumeric('Below what density', M.breakEven);
check('break-even: 12.5 is correct', t.includes('Correct.') && t.includes('four times the quantised matrix'), t.slice(0, 500));
const lens = countByRow(FIG171.rowIdx, 4);
t = await answerNumeric('trip counts', 66.7);
check('loop: 66.7 is accepted', t.includes('Correct.') && t.includes(`of which ${lens.reduce((a, b) => a + b, 0)} work`), t.slice(0, 400));
t = await answerNumeric('index 16*t', 4);
check('stride: 4 is wrong, the answer is 32', t.includes('Not quite. The answer is 32 segments.'), t.slice(0, 300));
t = await answerNumeric('index 16*t', 32);
check('stride: 32 is correct and ties to CSR', t.includes('Correct.') && t.includes('value[16*t]'), t.slice(0, 400));
t = await answerChoice('Which final values of', 1);
check('lost update: "only 8" is wrong, with the reason', t.includes('Not quite.') && t.includes('a load, an add and a store'), t.slice(0, 400));
t = await answerChoice('Which final values of', 0);
check('lost update: 1, 7 or 8 is correct, and leads to the COO atomic', t.includes('Correct.') && t.includes('Fig. 17.5'), t.slice(0, 500));
t = await answerChoice('three products are 1e8', 0);
check('order: "always 1" is wrong', t.includes('Not quite.') && t.includes('neither fixes the order'), t.slice(0, 400));
t = await answerChoice('three products are 1e8', 1);
check('order: 0 or 1 is correct, with the model\'s list and the spacing of 8', t.includes('Correct.') && t.includes('[0, 1]') && t.includes('8 apart'), t.slice(0, 600));
t = await answerNumeric('exclusive scan of', exclusiveScan(lens)[2]);
check('scan: 5 is correct, and the scan is CSR\'s rowPtrs', t.includes('Correct.') && t.includes('[0, 2, 5, 7, 8]'), t.slice(0, 400));

await pause(700); // the app saves progress after a short delay
const last = saved.at(-1) ?? '';
check(
  'answers were saved through PUT /api/progress (intercepted)',
  ['p-bytes', 'q-breakeven', 'q-warp-loop', 'q-stride16', 'q-lost-update', 'q-order', 'q-scan'].every((id) => last.includes('ch17/start/' + id)),
  last.slice(0, 300),
);
if (out) await page.screenshot({ path: out + '-page.png', fullPage: true });

// ---------------------------------------------------------------- dark theme and phone width
if (out) {
  await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
  await pause(200);
  const fig = await page.$('figure');
  await fig.screenshot({ path: out + '-figure-dark.png' });
  await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }]);
}
await page.setViewport({ width: 390, height: 860, deviceScaleFactor: 1 });
await pause(300);
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('no horizontal scroll at 390 px', overflow <= 0, String(overflow));
const figW = await page.$eval('figure svg', (s) => s.getBoundingClientRect().width);
check('the figure shrinks to fit at 390 px', figW > 250 && figW <= 390, String(figW));
if (out) await page.screenshot({ path: out + '-narrow.png', fullPage: true });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
