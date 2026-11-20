// Browser check for section 10.1: the listing is the checked snippet, and the fold
// explorer shows the model's float32 trace, its verdict and its random-row count for
// every operator and start value it is driven through. Expected values come from
// src/sim/reducetree.ts, imported here directly.
// Usage: node e2e/ch10-s101.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { BOOK_LIST, HUGE_ROW, NEGATIVE_ROW, f32short, f64short, foldSequential, randomRow, randomRowTest } from '../src/sim/reducetree.ts';

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
const B = 'section.bench[aria-label="One loop, any operator"]';
const press = async (group, label) => {
  const ok = await page.evaluate((B, group, label) => {
    const root = document.querySelector(B);
    const scope = group ? root.querySelector(`[aria-label="${group}"]`) : root;
    const btn = [...scope.querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, B, group, label);
  await pause(80);
  return ok;
};
// The explorer's state, read from the page.
const state = () => page.evaluate((B) => {
  const root = document.querySelector(B);
  const res = root.querySelector('[data-readout="result"]');
  return {
    rows: [...root.querySelectorAll('[data-part="trace"] tbody tr')].map((r) => ({ row: r.dataset.row, cells: [...r.querySelectorAll('td')].map((c) => c.textContent.trim()) })),
    ok: res.dataset.ok,
    result: res.querySelector('[data-value="result"]')?.textContent.trim() ?? null,
    truth: res.querySelector('[data-value="truth"]')?.textContent.trim() ?? null,
    chip: res.querySelector('.chip')?.textContent.trim() ?? null,
    random: root.querySelector('[data-readout="random"]').textContent.replace(/\s+/g, ' ').trim(),
    progress: root.querySelector('[data-readout="progress"]').textContent.trim(),
    list: root.querySelector('input[aria-label="The list"]').value,
  };
}, B);
const accs = (s) => s.rows.filter((r) => r.row !== 'start').map((r) => r.cells[2]);
const pairText = (v, i) => `(${f32short(v)}, ${i < 0 ? 'none' : i})`;
const randomText = (op, start) => `on ${randomRowTest(op, start).rows} random rows of 8 logits: wrong on ${randomRowTest(op, start).wrong}`;

await goto(page, 'ch10/10.1');
await page.waitForSelector(B);
await page.waitForFunction(() => document.querySelector('.code pre')?.textContent.includes('reduce_seq'));

// ---- the listing
const code = await page.$eval('.code pre', (e) => e.textContent);
check('the listing is labs/ch10/snippets/fold.h, loaded from the repository', code.includes('T reduce_seq(const T* input, std::size_t n, T start, Op op)') && code.includes('-FLT_MAX') && code.includes('argmax_row'));
const hl = await page.$$eval('.code .ln.hl', (ls) => ls.map((l) => l.textContent.trim()));
check('the three highlighted lines are the start, the loop and the operator', hl.length === 3 && hl[0] === 'T acc = start;' && hl[2] === 'acc = op(acc, input[i]);', hl.join(' | '));

// ---- the book's sum, in float32
let s = await state();
const book = foldSequential(BOOK_LIST, 'sum');
check('the book\'s list is the default', s.list === '7.0, 2.1, 5.3, 9.0, 11.2', s.list);
check('the trace is the model\'s float32 trace: 7, 9.1, 14.400001, 23.400002, 34.600002', JSON.stringify(accs(s)) === JSON.stringify(book.steps.map((x) => f32short(x.acc))) && accs(s).at(-1) === '34.600002', JSON.stringify(accs(s)));
check('it starts at the identity 0 and is right: 34.600002 against 34.6', s.rows[0].cells[2] === '0' && s.ok === '1' && s.result === '34.600002' && s.truth === f64short(book.truth) && s.truth === '34.6' && s.chip === 'right', JSON.stringify(s));
check('and no random row trips it', s.random.includes(randomText('sum', 'identity')) && randomRowTest('sum', 'identity').wrong === 0, s.random);

// Restart, step, play.
await press(null, 'Restart');
s = await state();
check('Restart leaves only the start row, and the answer waits', s.rows.length === 1 && s.ok === '' && s.result === null, JSON.stringify(s.rows));
await press(null, 'Step');
await press(null, 'Step');
s = await state();
check('two Steps show iterations 0 and 1', s.rows.length === 3 && s.rows[2].cells[0] === 'i = 1' && s.rows[2].cells[2] === '9.1' && s.progress.includes('Shown: 2 of 5'), JSON.stringify(s.rows));
await press(null, 'Play');
await pause(2600);
s = await state();
check('Play runs the rest of the loop and gives the verdict', s.rows.length === 6 && s.ok === '1', JSON.stringify(s.rows.length));
if (out) await page.screenshot({ path: out + '-sum.png' });

// ---- max on an all-negative row
await press('Operator', 'max');
await press('Lists', 'logits, all negative');
await press('Start value', '0');
s = await state();
let m = foldSequential(NEGATIVE_ROW, 'max', 'zero');
check('max started at 0 on an all-negative row returns 0 and is marked wrong', s.ok === '0' && s.result === '0' && m.result === 0 && !m.ok && s.chip === 'wrong' && s.truth === f64short(Math.max(...NEGATIVE_ROW)), JSON.stringify([s.result, s.truth, s.chip]));
check('random rows almost never catch it: the model\'s count is shown', s.random.includes(randomText('max', 'zero')) && randomRowTest('max', 'zero').wrong > 0 && s.random.includes('could easily miss it'), s.random);
if (out) await page.screenshot({ path: out + '-max-zero.png' });
await press('Lists', 'one huge logit');
s = await state();
check('the same start on a row with one huge logit is right', s.ok === '1' && s.result === f32short(31.7) && foldSequential(HUGE_ROW, 'max', 'zero').ok, JSON.stringify([s.result, s.ok]));
await press('Lists', 'logits, all negative');
await press('Start value', '−FLT_MAX');
s = await state();
check('−FLT_MAX is a safe start for max: right, −0.9', s.ok === '1' && s.result === '−0.9' && s.rows[0].cells[2] === '−3.4028235e+38' && s.random.includes('wrong on 0'), JSON.stringify([s.result, s.rows[0]]));
await press('Start value', 'the identity, −∞');
s = await state();
check('the identity of max is −∞', s.rows[0].cells[2] === '−∞' && s.ok === '1');

// ---- argmax, min, product, element 0
await press('Operator', 'argmax');
await press('Start value', '0');
s = await state();
const a = foldSequential(NEGATIVE_ROW, 'argmax', 'zero');
check('argmax started at (0, none) never finds an index on this row', s.result === pairText(a.result, a.resultIndex) && s.result === '(0, none)' && s.ok === '0' && s.truth === `index ${a.truthIndex} (value ${f64short(a.truth)})`, JSON.stringify([s.result, s.truth]));
await press('Start value', 'element 0');
s = await state();
const a0 = foldSequential(NEGATIVE_ROW, 'argmax', 'first');
check('started from element 0, argmax is right and takes one iteration fewer', s.ok === '1' && s.result === pairText(a0.result, a0.resultIndex) && s.rows.length === NEGATIVE_ROW.length && s.progress.includes(`Started from element 0, so ${NEGATIVE_ROW.length} elements take ${NEGATIVE_ROW.length - 1} iterations`), JSON.stringify([s.result, s.progress]));
await press('Operator', 'min');
await press('Start value', '−FLT_MAX');
s = await state();
check('−FLT_MAX as the start of a min is wrong on every row', s.ok === '0' && s.random.includes(randomText('min', 'negmax')) && randomRowTest('min', 'negmax').wrong === 1000, s.random);
await press('Operator', 'product');
await press('Start value', '0');
s = await state();
check('a product started at 0 is 0', s.ok === '0' && s.result === '0');
await press('Operator', 'sum');
await press('Start value', '−FLT_MAX');
s = await state();
check('a sum started at −FLT_MAX is wrong', s.ok === '0' && s.chip === 'wrong');

// ---- typed lists and random rows
await press('Start value', 'the identity, 0');
await page.$eval(B + ' input[aria-label="The list"]', (el) => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '3, -1, 2.5');
  el.dispatchEvent(new Event('input', { bubbles: true }));
});
await pause(100);
s = await state();
check('a typed list is used at once', s.result === '4.5' && s.rows.length === 4, JSON.stringify([s.result, s.rows.length]));
await page.$eval(B + ' input[aria-label="The list"]', (el) => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '3, x');
  el.dispatchEvent(new Event('input', { bubbles: true }));
});
await pause(100);
s = await state();
check('a list that does not parse is flagged, and the last good one is kept', (await page.$eval(B, (e) => e.textContent)).includes('Type between 1 and 16 numbers') && s.result === '4.5');
await press('Lists', 'a random row');
s = await state();
const r1 = randomRow(8, 1);
check('"a random row" loads the model\'s seeded row', s.list === r1.map((x) => (Number.isInteger(x) ? x.toFixed(1) : String(x))).join(', ') && s.rows.length === 9, s.list);

// ---- questions
const question = (text) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt').textContent.includes(t)), text);
async function choose(text, option) {
  const q = await question(text);
  await q.evaluate((e, o) => [...e.querySelectorAll('.opts label')].find((l) => l.textContent.includes(o)).querySelector('input').click(), option);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
}
let t = await choose('starts each thread\'s accumulator at', 'all negative');
check('q-identity-max: the all-negative row', t.includes('Correct.') && t.includes('a value that is not in the row'));
t = await choose('infinities are undefined', 'first element');
check('q-finite-identity: the first element is wrong for a thread with no element', t.includes('Not quite.') && t.includes('has nothing to start from'));
t = await choose('infinities are undefined', '-FLT_MAX');
check('q-finite-identity: −FLT_MAX', t.includes('Correct.') && t.includes('ch10-facts.md C3'));
t = await choose('std::accumulate', 'std::reduce, because');
check('q-reduce-order: std::reduce may regroup', t.includes('Correct.') && t.includes('non-deterministic'));

// ---- notes
const notes = await page.$$eval('.note', (n) => n.map((x) => x.className + ': ' + x.textContent));
check('the errata of p. 222 are noted in the margin', notes.some((x) => x.includes('minimum (maximum)') && x.includes('iteration 5')));
check('Metal fast math and the library links carry Today notes', notes.some((x) => x.includes('today') && x.includes('§8.1')) && notes.some((x) => x.includes('today') && x.includes('cccl/unstable')));
check('the explorer states what it simplifies, before the bench', await page.evaluate((B) => {
  const bench = document.querySelector(B);
  const prev = bench.previousElementSibling;
  return prev.classList.contains('caveat') && prev.textContent.includes('float32');
}, B));

// ---- narrow screen
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
check('at a narrow width nothing scrolls sideways', await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
const clipped = await page.evaluate((B) => {
  const b = document.querySelector(B);
  const r = b.getBoundingClientRect();
  return [...b.querySelectorAll('button, input')].filter((x) => {
    const q = x.getBoundingClientRect();
    return q.width > 0 && (q.right > r.right + 1 || q.left < r.left - 1);
  }).map((x) => (x.textContent || x.getAttribute('aria-label')).trim());
}, B);
check('and every control of the explorer sits inside it', clipped.length === 0, clipped.join(' | '));
if (out) {
  await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
