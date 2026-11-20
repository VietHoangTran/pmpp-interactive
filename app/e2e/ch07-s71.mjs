// Browser check for section 7.1: the sliding-window explorer (index, presets, policy,
// flip, editing), the Exercise 1 prediction, the patch explorer, the ghost counter and
// its formulas, and the questions. Expected values come from src/sim/convolve.ts.
// Usage: node e2e/ch07-s71.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { ASYMMETRIC, BOOK_1D, EX2, conv1d, ghostCounts2d, ghostFormula2d } from '../src/sim/convolve.ts';

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
const norm = (s) => s.replace(/\s+/g, ' ').trim();
const bench = (t) => `section.bench[aria-label="${t}"]`;
const SW = bench('The sliding window');
const PE = bench("One output's patch");
const GC = bench('Count the ghost cells');
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

async function setRange(sel, v) {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(80);
}
async function setSelect(sel, v) {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(80);
}
async function setText(sel, v) {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(80);
}
async function press(scope, label) {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(80);
  return ok;
}
const sum = () => page.$eval(SW + ' svg', (e) => e.dataset.sum);
const yRow = () => page.$eval(SW + ' [data-out="y"]', (e) => JSON.parse(e.dataset.y));
const swText = () => page.$eval(SW + ' [data-out="y"]', (e) => e.textContent.replace(/\s+/g, ' '));

const question = (text) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.replace(/\s+/g, ' ').includes(t)), text);
async function answerChoice(text, optionStart) {
  const q = await question(text);
  const found = await q.evaluate((e, o) => {
    const label = [...e.querySelectorAll('.opts label')].find((l) => l.textContent.replace(/\s+/g, ' ').trim().startsWith(o));
    if (!label) return false;
    label.querySelector('input').click();
    return true;
  }, optionStart);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return { found, text: norm(await q.evaluate((e) => e.textContent)) };
}
async function answerNumber(text, value) {
  const q = await question(text);
  const input = await q.$('input[type=number]');
  await input.click();
  const len = await input.evaluate((e) => e.value.length);
  for (let i = 0; i < len; i++) await page.keyboard.press('Backspace');
  await input.type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return norm(await q.evaluate((e) => e.textContent));
}

await goto(page, 'ch07/7.1');
await page.waitForSelector(SW);
const progress = await page.evaluate(() => fetch('/api/progress').then((r) => r.json()));
const answers = progress.answers ?? {};

// ---- the sliding window, the book's data
check('starts at y[2] = 52 (Fig. 7.1)', (await sum()) === '52', await sum());
await setRange(SW + ' input[type=range]', 3);
check('y[3] = 47 (Fig. 7.2)', (await sum()) === '47', await sum());
await setRange(SW + ' input[type=range]', 1);
check('y[1] = 53 with one zero ghost cell (Fig. 7.3)', (await sum()) === '53', await sum());
check('the sum is spelled out from the products', norm(await page.$eval(SW + ' [data-out="sum"]', (e) => e.textContent)) === 'y[1] = 0 + 24 + 10 + 15 + 4 = 53');
const zero = conv1d(BOOK_1D.x, BOOK_1D.f);
await setRange(SW + ' input[type=range]', 0);
if (!answers['ch07/7.1/p-y0']) {
  check('before a prediction, y[0] (Exercise 1) is hidden in the explorer', (await sum()) === '' && (await yRow())[0] === null, JSON.stringify(await yRow()));
  const p = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Exercise 1. With zero ghost cells')));
  await (await p.$('input[type=number]')).type('51');
  await (await p.$('button.primary')).click();
  await pause(150);
}
const reveal = norm(await page.evaluate(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Exercise 1. With zero ghost cells')).textContent));
check('the prediction reveals y[0] = 51 with its working', reveal.includes('You predicted') && reveal.includes('1·0 + 3·0 + 5·8 + 3·2 + 1·5 = 51'), reveal);
check('after the prediction the explorer shows y[0] = 51', (await sum()) === '51', await sum());
check('the y row is the model\'s', same(await yRow(), zero), JSON.stringify(await yRow()));
check('it names the outputs whose window reaches a ghost cell: y[0], y[1], y[5], y[6]', (await swText()).includes('reaches a ghost cell: y[0], y[1], y[5], y[6].'), await swText());

// ---- policies
for (const [label, policy] of [['replicate the edge', 'replicate'], ['wrap around', 'wrap']]) {
  await press(SW, label);
  const want = conv1d(BOOK_1D.x, BOOK_1D.f, { policy });
  check(`${policy}: y row is the model's (${want.join(', ')})`, same(await yRow(), want), JSON.stringify(await yRow()));
  check(`${policy}: it changed exactly y[0], y[1], y[5], y[6]`, (await swText()).includes('this policy changed y[0], y[1], y[5], y[6].'), await swText());
}
if (out) await page.$eval(SW, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-window-wrap.png' });
await press(SW, 'zero ghost cells');

// ---- presets and the flip
const flipBox = async (on) => {
  await page.evaluate((SW, on) => {
    const b = [...document.querySelectorAll(SW + ' label.check')].find((l) => l.textContent.includes('flip')).querySelector('input');
    if (b.checked !== on) b.click();
  }, SW, on);
  await pause(80);
};
await flipBox(true);
check('flipping the book\'s symmetric filter changes nothing', (await swText()).includes('the flip changed nothing'), await swText());
await flipBox(false);
await setSelect(SW + ' select', 'ex2');
if (!answers['ch07/7.1/q-ex2']) {
  check('before Exercise 2 is answered, the explorer hides its outputs', (await yRow()).every((v) => v === null) && (await sum()) === '' &&
    norm(await page.$eval(SW + ' [data-out="sum"]', (e) => e.textContent)).includes('Exercise 2: answer it below'), JSON.stringify(await yRow()));
  await setSelect(SW + ' select', 'book');
  await setText(SW + ' input[aria-label="x"]', '4, 1, 3, 2, 3');
  await setText(SW + ' input[aria-label="f"]', '2, 1, 4');
  check('typing Exercise 2\'s data by hand hides them too', (await yRow()).every((v) => v === null), JSON.stringify(await yRow()));
  await setSelect(SW + ' select', 'book');
}
// ---- the Exercise 2 question, then its data in the explorer
let r = await answerChoice('Exercise 2. Convolve', '{6, 23, 11, 20, 11}');
check('q-ex2: the flipped answer is explained as textbook convolution', r.text.includes('Not quite.') && r.text.includes('textbook convolution'), r.text.slice(0, 300));
r = await answerChoice('Exercise 2. Convolve', '{8, 21, 13, 20, 7}');
check('q-ex2: right answer accepted', r.text.includes('Correct.'), r.text.slice(-200));
await setSelect(SW + ' select', 'ex2');
check('Exercise 2: 8, 21, 13, 20, 7', same(await yRow(), [8, 21, 13, 20, 7]), JSON.stringify(await yRow()));
await flipBox(true);
check('Exercise 2 flipped (textbook convolution): 6, 23, 11, 20, 11', same(await yRow(), [6, 23, 11, 20, 11]), JSON.stringify(await yRow()));
check('and it says which outputs the flip changed', (await swText()).includes('the flip changed y[0], y[1], y[2], y[4]'), await swText());
await flipBox(false);
await setSelect(SW + ' select', 'ex3c');
check('Exercise 3 (c), [1, 0, 0], moves every value one place right', same(await yRow(), [0, 8, 2, 5, 4, 1, 7]), JSON.stringify(await yRow()));
await setSelect(SW + ' select', 'ex3b');
check('Exercise 3 (b), [0, 0, 1], moves every value one place left', same(await yRow(), [2, 5, 4, 1, 7, 3, 0]), JSON.stringify(await yRow()));
await setSelect(SW + ' select', 'asym');
await flipBox(true);
check('the asymmetric filter flipped is the model\'s', same(await yRow(), conv1d(BOOK_1D.x, ASYMMETRIC, { flip: true })), JSON.stringify(await yRow()));
await flipBox(false);
await setSelect(SW + ' select', 'book');
await setText(SW + ' input[aria-label="f"]', '1, 3, 5, 3');
check('an even-length filter is refused with a message', (await page.$(SW + ' [role="alert"]')) !== null && (await page.$eval(SW + ' [role="alert"]', (e) => e.textContent)).includes('odd number'));
await setText(SW + ' input[aria-label="f"]', '1/3, 1/3, 1/3');
const avg = conv1d(BOOK_1D.x, [1 / 3, 1 / 3, 1 / 3]).map((v) => Math.round(v * 1000) / 1000);
check('fractions can be typed: a 3-point average', same(await yRow(), avg), JSON.stringify(await yRow()));
await setSelect(SW + ' select', 'book');

// ---- questions on 1D
r = await answerChoice('moves every value one place to the right', '[1, 0, 0]');
check('q-ex3: [1, 0, 0]', r.text.includes('Correct.') && r.text.includes('central difference'), r.text.slice(-300));
r = await answerChoice('Switching from zero ghost cells', 'y[0], y[1], y[5] and y[6].');
check('q-policy: the first and last r', r.text.includes('Correct.'), r.text.slice(-200));

// ---- the patch explorer (Figs. 7.4 and 7.5)
const patch = () => page.$eval(PE + ' [data-out="patch"]', (e) => ({ sum: Number(e.dataset.sum), ghost: Number(e.dataset.ghost), text: e.textContent.replace(/\s+/g, ' ') }));
let pt = await patch();
check('P[2][2] = 321 with row sums 27 + 56 + 95 + 84 + 59', pt.sum === 321 && pt.ghost === 0 && pt.text.includes('27 + 56 + 95 + 84 + 59 = 321'), pt.text);
await page.click(PE + ' [data-grid="P"] [data-rc="1,0"]');
await pause(100);
pt = await patch();
check('clicking P[1][0] gives 112 with 13 ghost taps', pt.sum === 112 && pt.ghost === 13, JSON.stringify(pt));
check('the 13 ghost products are drawn dashed', (await page.$$(PE + ' [data-grid="products"] span.void')).length === 13);
check('and the ghost cells appear around N', (await page.$$(PE + ' [data-grid="N"] span.void')).length === 13);
await page.click(PE + ' [data-grid="P"] [data-rc="0,0"]');
await pause(100);
pt = await patch();
check('the corner P[0][0] has 16 ghost taps', pt.ghost === 16, JSON.stringify(pt));
if (out) await page.$eval(PE, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-patch.png' });

// ---- the ghost counter (Exercises 4 to 6)
const counts = () => page.$$eval(GC + ' [data-out="counts"] tbody tr', (trs) => trs.map((t) => [...t.querySelectorAll('td')].slice(1).map((c) => Number(c.textContent.replace(/,/g, '')))));
const formulaBtn = () => page.evaluate((GC) => { const b = [...document.querySelectorAll(GC + ' button')].find((x) => x.textContent.trim() === 'Show the formulas'); return b ? b.disabled : null; }, GC);
if (!answers['ch07/7.1/q-ex5']) check('the formulas stay closed until Exercise 5 is answered', (await formulaBtn()) === true);
let c = await counts();
const b75 = ghostCounts2d(7, 7, 5, 5);
check(`7×7 with 5×5 by default: ${b75.ghostCells}, ${b75.allTaps}, ${b75.realTaps}`, same(c.map((x) => x[0]), [b75.ghostCells, b75.allTaps, b75.realTaps]), JSON.stringify(c));
await setRange(GC + ' input[type=range]', 8);
await setSelect(GC + ' select', 3);
c = await counts();
if (!answers['ch07/7.1/q-ex5']) {
  const hidden = await page.$eval(GC + ' [data-out="counts"] tr[data-row="real"]', (e) => e.textContent);
  check('before Exercise 5 is answered, 8×8 with 3×3 hides the skipped count and the saving',
    c[0][0] === 36 && c[1][0] === 576 && hidden.includes('?') && norm(await page.$eval(GC + ' [data-out="share"]', (e) => e.textContent)).includes('Exercise 5 question'), JSON.stringify(c));
}
const t = await answerNumber('Exercise 5 for an 8×8 image', 484);
check('q-ex5: 484', t.includes('Correct.') && t.includes('22² = 484'), t.slice(-200));
c = await counts();
check('8×8 with 3×3: 36 ghost cells, 576 and 484 multiplications', same(c.map((x) => x[0]), [36, 576, 484]), JSON.stringify(c));
check('answering Exercise 5 opens the formulas button', (await formulaBtn()) === false);
await press(GC, 'Show the formulas');
c = await counts();
const f83 = ghostFormula2d(8, 8, 3, 3);
check('the formula column agrees with the count for 8×8, 3×3', same(c.map((x) => x[1]), [f83.ghostCells, f83.allTaps, f83.realTaps]), JSON.stringify(c));
check('the formulas are shown, with 4r(N + r) and (N·M − r(r + 1))²', norm(await page.$eval(GC + ' [data-out="formulas"]', (e) => e.textContent)).includes('4r(N + r)'));
await setRange(GC + ' input[type=range]', 2);
await setSelect(GC + ' select', 7);
c = await counts();
const bad = ghostCounts2d(2, 2, 7, 7);
check('below N = r the closed form disagrees, and the panel says so', c[2][0] === bad.realTaps && c[2][1] !== bad.realTaps && norm(await page.$eval(GC + ' [data-out="formulas"]', (e) => e.textContent)).includes('Here N < r'), JSON.stringify(c));
await setRange(GC + ' input[type=range]', 64);
await setSelect(GC + ' select', 3);
const share = norm(await page.$eval(GC + ' [data-out="share"]', (e) => e.textContent));
const g64 = ghostCounts2d(64, 64, 3, 3);
check('for 64×64 the saving is small: the model\'s share', share.includes(`${Math.round(((g64.allTaps - g64.realTaps) / g64.allTaps) * 1000) / 10} %`), share);
if (out) await page.$eval(GC, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-counter.png' });

r = await answerChoice('torch.nn.functional.conv2d on a 1×1×7×7', 'Only the 3×3 outputs');
check('q-pytorch: padding=0 means no ghost cells and a 3×3 result', r.text.includes('Correct.') && r.text.includes('padding=2'), r.text.slice(-300));

// ---- dark theme and a narrow width
await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
if (out) await page.$eval(SW, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-dark.png' });
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
const over = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', over <= 1, `overflow ${over}px`);
if (out) await page.$eval(SW, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-narrow.png' });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
