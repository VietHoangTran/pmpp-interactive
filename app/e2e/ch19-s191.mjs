// Browser check for section 19.1: the term-by-term explorer of Fig. 19.2 (with the 180°
// rotation switch), the LeNet-5 explorer, the CPU reference from lab 19.1, the prediction
// and the questions. Expected values come from src/sim/cnnlayer.ts.
// Usage: node e2e/ch19-s191.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { BOOK_F, BOOK_LAYER, BOOK_X, batchTraffic, changedByFlip, forward, lenetLayer, perInputMap, shapes, terms } from '../src/sim/cnnlayer.ts';

const out = process.argv.slice(2).find((a) => !a.startsWith('--')); // a screenshot prefix; flags are ignored
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
const ONE = bench('One output, term by term');
const LN = bench("LeNet-5's convolution layers");
const S = BOOK_LAYER;

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
const term = () => page.$eval(ONE + ' [data-out="term"]', (e) => ({ k: Number(e.dataset.k), sum: Number(e.dataset.sum), text: e.textContent.replace(/\s+/g, ' ') }));
const clickY = async (m, h, w) => {
  await page.$eval(`${ONE} [data-y="${m},${h},${w}"]`, (e) => e.click());
  await pause(80);
};
const question = (text) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.replace(/\s+/g, ' ').includes(t)), text);
async function answerNumber(text, value) {
  const q = await question(text);
  const input = await q.$('input[type=number]');
  await input.click({ clickCount: 3 });
  await input.type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return norm(await q.evaluate((e) => e.textContent));
}
async function answerChoice(text, optionStart) {
  const q = await question(text);
  const found = await q.evaluate((e, o) => {
    const l = [...e.querySelectorAll('.opts label')].find((x) => x.textContent.replace(/\s+/g, ' ').trim().startsWith(o));
    if (!l) return false;
    l.querySelector('input').click();
    return true;
  }, optionStart);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return { found, text: norm(await q.evaluate((e) => e.textContent)) };
}

await goto(page, 'ch19/19.1');
await page.waitForSelector(ONE);

// ---- the prediction comes before the explorer
const predict = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('pixel (0, 1) of output map 0')));
const order = await page.evaluate((one) => {
  const q = [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('pixel (0, 1) of output map 0'));
  return !!(q.compareDocumentPosition(document.querySelector(one)) & Node.DOCUMENT_POSITION_FOLLOWING);
}, ONE);
check('the prediction sits above the explorer', order);
await (await predict.$('input[type=number]')).type('20');
await (await predict.$('button.primary')).click();
await pause(120);
const pv = norm(await predict.evaluate((e) => e.textContent));
const p01 = perInputMap(S, BOOK_X, BOOK_F, 0, 0, 1);
check('committing 20 reveals 10 + 8 + 2 = 20 from the model', pv.includes(`${p01.join(' + ')} = 20`) && pv.includes('That is what you had.'), pv);

// ---- one output, term by term
let t = await term();
check('starts on Y[0, 0, 0] with nothing run', t.k === 0 && t.text.includes('Y[0, 0, 0] adds 12 products'), t.text);
const yVals = await page.$$eval(ONE + ' [data-y]', (es) => es.map((e) => Number(e.dataset.value)));
check('the eight outputs are the book\'s 14 20 15 24 / 12 24 17 26', JSON.stringify(yVals) === JSON.stringify(forward(S, BOOK_X, BOOK_F)), JSON.stringify(yVals));
await press(ONE, 'Step');
t = await term();
const tr000 = terms(S, BOOK_X, BOOK_F, 0, 0, 0);
check('step 1: c = 0, p = 0, q = 0, X[0, 0, 0] × F[0, 0, 0, 0] = 1 × 1', t.k === 1 && t.sum === tr000[0].sum && t.text.includes('X[0, 0, 0] × F[0, 0, 0, 0] = 1 × 1 = 1'), t.text);
check('the X value and the filter weight in use are lit', (await page.$$(ONE + ' [data-grid="X0"] span.hot')).length === 1 && (await page.$$(ONE + ' [data-grid="F00"] span.hot')).length === 1);
check('the patch is outlined: 4 cells in each input map', (await page.$$eval(ONE + ' [data-grid^="X"] span.on, ' + ONE + ' [data-grid^="X"] span.hot', (s) => s.length)) === 12);
for (let i = 0; i < 4; i++) await press(ONE, 'Step');
t = await term();
check('after 5 steps the running sum is the model\'s', t.k === 5 && t.sum === tr000[4].sum, JSON.stringify(t));
check('map 0 has finished and shows its share, 7', (await page.$eval(ONE + ' [data-part="0"]', (e) => e.textContent)) === '7');
await press(ONE, 'Finish');
t = await term();
check('finished: 7 + 5 + 2 = 14', t.k === 12 && t.sum === 14 && t.text.includes('= 7 + 5 + 2 = 14'), t.text);
await press(ONE, 'Back');
t = await term();
check('Back goes one term back', t.k === 11 && t.sum === tr000[10].sum);
// another output
await clickY(0, 0, 1);
t = await term();
check('clicking Y0 (0, 1) selects it and restarts', t.k === 0 && t.text.includes('Y[0, 0, 1]'), t.text);
await press(ONE, 'Step');
check('its first term reads X0[0][1] = 2', (await page.$eval(ONE + ' [data-grid="X0"] span.hot', (e) => e.textContent)) === '2');
await press(ONE, 'Finish');
t = await term();
check('Y[0, 0, 1] = 10 + 8 + 2 = 20', t.sum === 20 && t.text.includes('10 + 8 + 2 = 20'), t.text);
await clickY(1, 1, 1);
await press(ONE, 'Play');
await pause(1400);
t = await term();
check('Play advances on its own', t.k >= 2 && t.k < 12, JSON.stringify(t));
await press(ONE, 'Pause');
// the rotation
await clickY(0, 0, 0);
await page.evaluate((one) => [...document.querySelectorAll(one + ' label.check')].find((l) => l.textContent.includes('rotate')).querySelector('input').click(), ONE);
await pause(120);
await press(ONE, 'Finish');
t = await term();
check('rotated: Y[0, 0, 0] = 8 + 5 + 2 = 15', t.sum === 15 && t.text.includes('8 + 5 + 2 = 15'), t.text);
const flipped = await page.$$eval(ONE + ' [data-y]', (es) => es.map((e) => ({ v: Number(e.dataset.value), c: e.dataset.changed })));
const fwdFlip = forward(S, BOOK_X, BOOK_F, true);
const changed = new Set(changedByFlip(S, BOOK_X, BOOK_F));
check('rotated outputs are the model\'s, and the changed ones are marked', flipped.every((x, i) => x.v === fwdFlip[i] && (x.c === 'yes') === changed.has(i)), JSON.stringify(flipped));
const plain = forward(S, BOOK_X, BOOK_F);
const changesText = norm(await page.$eval(ONE + ' [data-out="changes"]', (e) => e.textContent));
check(`${changed.size} of 8 outputs change, each listed old → new`, changesText.includes(`${changed.size} of 8 outputs differ`) && [...changed].every((i) => changesText.includes(`${plain[i]} → ${fwdFlip[i]}`)), changesText);
const f00 = await page.$$eval(ONE + ' [data-grid="F00"] span', (s) => s.map((x) => x.textContent).join(' '));
check('the first filter is drawn rotated: 2 2 / 1 1', f00 === '2 2 1 1', f00);
if (out) await page.$eval(ONE, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-oneoutput.png' });
await page.evaluate((one) => [...document.querySelectorAll(one + ' label.check')].find((l) => l.textContent.includes('rotate')).querySelector('input').click(), ONE);
await pause(80);

// ---- LeNet
const stage = (id) => page.evaluate((ln, id) => [...document.querySelectorAll(ln + ' [role=group] button')].find((b) => b.textContent.trim().startsWith(id + ' ') || b.textContent.trim() === id).click(), LN, id);
const field = (k) => page.$eval(`${LN} [data-out="${k}"]`, (e) => e.textContent.trim());
check('eight stages in the strip', (await page.$$(LN + ' [role=group] button')).length === 8);
check('C1 by default: X 1 × 1 × 32 × 32, Y 1 × 6 × 28 × 28, 6 filters, 150 weights',
  (await field('X')) === '1 × 1 × 32 × 32' && (await field('Y')) === '1 × 6 × 28 × 28' && (await field('filters')) === '6' && (await field('weights')) === '150');
await stage('C3');
await pause(80);
const c3 = shapes(lenetLayer('C3'));
check('C3: 96 filters, 2,400 weights, F 16 × 6 × 5 × 5', (await field('filters')) === String(c3.filters) && (await field('weights')) === '2,400' && (await field('F')) === '16 × 6 × 5 × 5');
await stage('C5');
await pause(80);
check('C5: 1,920 filters, Y 1 × 120 × 1 × 1', (await field('filters')) === '1,920' && (await field('Y')) === '1 × 120 × 1 × 1');
const share1 = await field('fshare');
check('C5 at batch 1: F is 98.9% of the ideal bytes', share1 === (batchTraffic(lenetLayer('C5'), 1).fShare * 100).toFixed(1) + '%', share1);
await page.$eval(LN + ' input[type=range]', (el) => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '8');
  el.dispatchEvent(new Event('input', { bubbles: true }));
});
await pause(120);
const t256 = batchTraffic(lenetLayer('C5'), 256);
check('at N = 256 the share falls to the model\'s 26.5%', (await field('fshare')) === (t256.fShare * 100).toFixed(1) + '%', await field('fshare'));
check('F bytes per sample fall as 1/N: 750', (await field('fpersample')) === String(t256.fBytesPerSample));
const rows = await page.$$eval(LN + ' tr[data-row]', (r) => r.map((x) => ({ id: x.dataset.row, s: Number(x.dataset.fshare) })));
check('the comparison table lists C1, C3, C5 with the model\'s shares', rows.map((r) => r.id).join() === 'C1,C3,C5' && rows.every((r) => Math.abs(r.s - batchTraffic(lenetLayer(r.id), 256).fShare) < 1e-12), JSON.stringify(rows));
await stage('S2');
await pause(80);
check('S2 is named but not computed', (await field('not-computed')).includes('not computed here'));
if (out) await page.$eval(LN, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-lenet.png' });

// ---- the CPU reference from the lab
await page.waitForFunction(() => [...document.querySelectorAll('figure.code')].some((f) => f.textContent.includes('cpuConvLayer')), { timeout: 8000 }).catch(() => undefined);
const ref = await page.evaluate(() => [...document.querySelectorAll('figure.code')].map((f) => ({ code: f.querySelector('pre').textContent, cap: f.querySelector('figcaption')?.textContent ?? '' })).find((x) => x.code.includes('cpuConvLayer')));
check('the CPU reference region of lab 19.1 is shown, markers stripped', !!ref && !ref.code.includes('[lesson:') && ref.cap.startsWith('The CPU reference') && /for \(int q = 0; q < K; q\+\+\)/.test(ref.code), ref ? ref.code.slice(0, 160) : 'missing');
check('and only that region (not the unroll reference after it)', !!ref && !ref.code.includes('cpuUnroll'));

// ---- questions
let r = await answerNumber('How many weights does F hold', 2400);
check('q-weights: 2400 is correct', r.includes('Correct.'), r.slice(0, 120));
r = await answerNumber('What would Y[0, 0, 0] of the book', 15);
check('q-flip: 15 is correct', r.includes('Correct.'));
r = await answerNumber('What is the height of its output', 14);
check('q-outsize: 14 is marked wrong and the answer 10 shown', r.includes('The answer is 10'));
const b = await answerChoice('what can happen to the traffic for F', 'It can fall as 1/N');
check('q-batch: 1/N is correct', b.found && b.text.includes('Correct.'), b.text.slice(0, 160));

// ---- dark theme and a narrow width
await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
await pause(200);
if (out) await page.$eval(ONE, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-dark.png' });
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
const over = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', over <= 1, `overflow ${over}px`);
if (out) await page.$eval(ONE, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-narrow.png' });
if (out) await page.$eval(LN, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-narrow-lenet.png' });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
