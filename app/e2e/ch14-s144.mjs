// Browser check for section 14.4: the radix stepper (against src/sim/radixsort.ts), the raw-bits
// prediction, the "Logits to keys" explorer (against src/sim/floatkey.ts), the snippet listing
// and the five questions. The narrated scene has its own check (ch14-scene-lsd-passes.mjs).
// Usage: APP_URL=http://localhost:5173 node e2e/ch14-s144.mjs [--dark] [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto } from './browser.mjs';
import { BOOK_KEYS, bin, lsdSort, parseKeys, promptKey } from '../src/sim/radixsort.ts';
import { PRESETS, explore, items } from '../src/sim/floatkey.ts';

const dark = process.argv.includes('--dark');
const out = process.argv.slice(2).find((a) => !a.startsWith('--'));
const { browser, page, errors } = await open({ width: 1440, height: 1000, dark });
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
const bench = (t) => `section.bench[aria-label="${t}"]`;
const ST = bench('Radix sort, pass by pass');
const EX = bench('Logits to keys');
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const press = async (scope, label) => {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(80);
  return ok;
};
const setField = async (scope, label, value) => {
  await page.evaluate((scope, label, value) => {
    const l = [...document.querySelectorAll(scope + ' label.field')].find((x) => x.textContent.trim().startsWith(label));
    const s = l.querySelector('select');
    s.value = String(value);
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, scope, label, value);
  await pause(100);
};
const setCheck = async (scope, label, on) => {
  await page.evaluate((scope, label, on) => {
    const i = [...document.querySelectorAll(scope + ' label.check')].find((x) => x.textContent.includes(label)).querySelector('input');
    if (i.checked !== on) i.click();
  }, scope, label, on);
  await pause(100);
};
const typeInto = async (sel, value) => {
  const input = await page.$(sel);
  await input.click();
  await page.evaluate((sel) => {
    const el = document.querySelector(sel);
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '');
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, sel);
  await input.type(String(value));
  await pause(80);
};
const row = (which) => page.$$eval(`${ST} g[data-row="${which}"]`, (gs) => gs.map((g) => g.dataset.key));
const show4 = (ks) => ks.map((k) => bin(k, 4));

await goto(page, 'ch14/14.4');
await page.waitForSelector(ST);

// ---------------------------------------------------------------- the stepper
const run = lsdSort(BOOK_KEYS, 4, 1);
const p1 = run.passes[0];
const ask = promptKey(p1);
let t = await text(ST + ' [data-part="prompt"]');
check(`before pass 1 the stepper asks about ${bin(BOOK_KEYS[ask], 4)} at position ${ask}`, t.includes(`key ${bin(BOOK_KEYS[ask], 4)} is at position ${ask}`), t);
check('the asked key is highlighted in the input row', (await page.$$eval(`${ST} g[data-row="top"][data-hot="1"]`, (g) => g.map((x) => Number(x.dataset.slot)))).join() === String(ask));
await typeInto(`${ST} input[aria-label="your guess"]`, p1.dst[ask]);
await press(ST, 'Run pass 1');
t = await text(ST + ' .narrate');
check(`after pass 1: it landed at ${p1.dst[ask]}, and the guess is called right`, t.includes(`landed at position ${p1.dst[ask]}`) && t.includes('right'), t);
check('after pass 1: the narration gives the buckets', t.includes('8 with 0 to positions 0–7') && t.includes('in order by its low bit'), t);
check("after pass 1: the output row is Fig. 14.4's iteration 1", JSON.stringify(await row('bottom')) === JSON.stringify(show4(p1.output)));
check('after pass 1: 16 arrows, each to its model destination', JSON.stringify(await page.$$eval(`${ST} line[data-arrow]`, (ls) => ls.map((l) => Number(l.dataset.to)))) === JSON.stringify(p1.dst));
check('after pass 1: the asked key is outlined where it landed', (await page.$$eval(`${ST} g[data-row="bottom"][data-ring="1"]`, (g) => g.map((x) => Number(x.dataset.slot)))).join() === String(p1.dst[ask]));
await typeInto(`${ST} input[aria-label="your guess"]`, 0);
await press(ST, 'Run pass 2');
t = await text(ST + ' .narrate');
check('a wrong guess is reported without "right"', !t.includes('right') && t.includes('you said 0'), t);
check('after pass 2: in order by the low two bits', (await page.$eval(`${ST} [data-sorted-by]`, (e) => e.dataset.sortedBy)) === '2');
await press(ST, 'Run all');
check("run all: the final row is Fig. 14.4's sorted list", JSON.stringify(await row('bottom')) === JSON.stringify(show4(run.result)));
t = await text(ST + ' [data-part="outcome"]');
check('run all: outcome sorted, ties in input order', t.includes('sorted') && t.includes('Equal keys are in their input order too'), t);
if (out) await page.screenshot({ path: out + '-stepper-sorted.png' });

// unstable pass 4
await page.select(`${ST} select[aria-label="Make a pass unstable"]`, '4');
await pause(100);
await press(ST, 'Run all');
const bad4 = lsdSort(BOOK_KEYS, 4, 1, { unstablePass: 4 });
t = await text(ST + ' [data-part="outcome"]');
check('unstable pass 4: not sorted, the first out-of-order pair named', t.includes('not sorted') && t.includes(`${bin(bad4.result[bad4.firstDisorder], 4)} at position ${bad4.firstDisorder} comes before ${bin(bad4.result[bad4.firstDisorder + 1], 4)}`), t);
check('unstable pass 4: that pair is marked failed', (await page.$$eval(`${ST} g[data-row="bottom"][data-bad="1"]`, (g) => g.map((x) => Number(x.dataset.slot)))).join() === `${bad4.firstDisorder},${bad4.firstDisorder + 1}`);
check('unstable pass 4: the reversed bucket\'s arrows are drawn as failures', (await page.$$(`${ST} line[stroke="var(--fail)"]`)).length === bad4.passes[3].counts[1]);
if (out) await page.screenshot({ path: out + '-stepper-unstable.png' });

// unstable pass 1: sorted, but two equal keys swapped
await page.select(`${ST} select[aria-label="Make a pass unstable"]`, '1');
await pause(100);
await press(ST, 'Run all');
t = await text(ST + ' [data-part="outcome"]');
check('unstable pass 1: the keys are sorted but the two 1001s are out of input order', t.startsWith('sorted') && t.includes('two equal keys (1001'), t);
await page.select(`${ST} select[aria-label="Make a pass unstable"]`, '0');
await pause(100);

// most significant bit first
await setCheck(ST, 'most significant bit first', true);
await press(ST, 'Run all');
const msd = lsdSort(BOOK_KEYS, 4, 1, { order: 'msd' });
t = await text(ST + ' .narrate');
check('MSB first: the narration says the last pass has the final word, bit 0 first', t.includes('ordered by bit 0, then bit 1, then bit 2, then bit 3 among equals'), t);
check('MSB first: not sorted, as the model says', (await text(ST + ' [data-part="outcome"]')).includes('not sorted') && !msd.sorted);
check('MSB first: the result is the model\'s', JSON.stringify(await row('bottom')) === JSON.stringify(show4(msd.result)));
await setCheck(ST, 'most significant bit first', false);

// 2-bit radix
await press(ST, '2 bits');
check('2 bits: two passes', (await text(ST)).includes('0 of 2 passes'));
await press(ST, 'Run pass 1');
const two = lsdSort(BOOK_KEYS, 4, 2);
check("2 bits, pass 1: Fig. 14.10's first row, four buckets", JSON.stringify(await row('bottom')) === JSON.stringify(show4(two.passes[0].output)) && (await page.$$(`${ST} [data-bucket]`)).length === 4);
await press(ST, 'Run all');
check('2 bits: sorted after two passes', (await text(ST + ' [data-part="outcome"]')).startsWith('sorted') && JSON.stringify(await row('bottom')) === JSON.stringify(show4(two.result)));
await press(ST, '1 bit');

// your own keys
await setField(ST, 'Keys', 'own');
const mine = '1111 0 1010 0110 3 1001 0001 1110';
await typeInto(`${ST} input[aria-label="your eight keys"]`, mine);
await press(ST, 'Run all');
const own = lsdSort(parseKeys(mine, 4, 8).keys, 4, 1);
check('your own eight keys sort as the model says', JSON.stringify(await row('bottom')) === JSON.stringify(show4(own.result)));
await typeInto(`${ST} input[aria-label="your eight keys"]`, '1 2 3');
check('bad input is reported', (await text(ST)).includes('type exactly 8 keys'));
await setField(ST, 'Keys', 'book');

// ---------------------------------------------------------------- the prediction
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Predict first') && e.textContent.includes('raw bits')));
await (await pred.$('input[type=number]')).type('2');
await (await pred.$('button.primary')).click();
await pause(150);
t = await pred.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
check('prediction p-raw-order: revealed, position 2, with the bits', t.includes('position 2, as you said') && t.includes('−1.0 = 0xBF800000') && t.includes('0.5 = 0x3F000000'), t);

// ---------------------------------------------------------------- the explorer
const outTokens = () => page.$$eval(`${EX} [data-part="output"] span`, (s) => s.map((x) => Number(x.dataset.token)));
const badCells = () => page.$$eval(`${EX} [data-part="output"] span.bad`, (s) => s.length);
const verdict = () => text(`${EX} [data-part="verdict"]`);
const expectOrder = async (name, presetId, o) => {
  const r = explore(items(PRESETS.find((p) => p.id === presetId).bits), o);
  const got = await outTokens();
  check(name + ': output tokens as the model', JSON.stringify(got) === JSON.stringify(r.order), JSON.stringify(got) + ' vs ' + JSON.stringify(r.order));
  check(name + (r.firstMismatch < 0 ? ': matches the reference' : `: differs first at position ${r.firstMismatch}`),
    r.firstMismatch < 0 ? (await verdict()).includes('matches the reference') && (await badCells()) === 0
      : (await verdict()).includes(`first at position ${r.firstMismatch}`) && (await badCells()) === r.match.filter((m) => !m).length,
    await verdict());
  return r;
};
await expectOrder('raw bits, logits with ties', 'ties', { transform: 'none', zero: false, descBy: 'invert' });
await press(EX, 'flip sign bit');
await expectOrder('sign bit only', 'ties', { transform: 'sign', zero: false, descBy: 'invert' });
await press(EX, 'full');
await expectOrder('full transform', 'ties', { transform: 'full', zero: false, descBy: 'invert' });
await setField(EX, 'Row of logits', 'zeros');
const z = await expectOrder('signs and zeros without the -0.0 step', 'zeros', { transform: 'full', zero: false, descBy: 'invert' });
check('without the step, exactly the two zeros are marked', z.match.filter((m) => !m).length === 2);
if (out) await page.screenshot({ path: out + '-explorer-zeros.png' });
await setCheck(EX, '−0.0 step', true);
await expectOrder('signs and zeros with the step', 'zeros', { transform: 'full', zero: true, descBy: 'invert' });
const zrow = await page.$$eval(`${EX} tr[data-token="2"] td`, (tds) => tds.map((td) => td.textContent));
check('the step changes the -0.0 row\'s bits to 0x00000000', zrow[2] === '0x80000000' && zrow[3] === '0x00000000', JSON.stringify(zrow));
await setField(EX, 'Row of logits', 'ties');
await press(EX, 'reversing an ascending sort');
await expectOrder('reversing an ascending sort, logits with ties', 'ties', { transform: 'full', zero: true, descBy: 'reverse' });
await press(EX, 'inverting the keys');
await setField(EX, 'Row of logits', 'edges');
await expectOrder('edges with the full pipeline', 'edges', { transform: 'full', zero: true, descBy: 'invert' });
check("edges: the reference is CUB's bits rule", (await text(`${EX} footer`)).includes("CUB's rule"));
if (out) await page.screenshot({ path: out + '-explorer-edges.png' });

// ---------------------------------------------------------------- listings
const codes = await page.$$eval('.code', (cs) => cs.map((c) => c.textContent));
check('the snippet listing is shown with its status', codes.some((c) => c.includes('float_to_key_desc') && c.includes('emulator-checked; unverified on CUDA and HIP hardware')));
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const lab3 = fs.existsSync(path.join(repo, 'labs/ch14/lab3_sortrows/solution/kernel.metal'));
check(lab3 ? "lab 14.3's Metal kernel is shown, its key transform marked" : 'lab 14.3 is not written yet, so no Metal listing is shown',
  lab3 ? codes.some((c) => c.includes('Metal (lab 14.3)')) && (await page.$$eval('.code', (cs) => cs.filter((c) => c.textContent.includes('Metal (lab 14.3)')).flatMap((c) => [...c.querySelectorAll('.ln.hl')]).length)) > 0
    : !codes.some((c) => c.includes('Metal (lab 14.3)')));

// ---------------------------------------------------------------- questions
const question = (txt) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(t)), txt);
const choose = async (txt, start) => {
  const q = await question(txt);
  await q.evaluate((e, s) => [...e.querySelectorAll('.opts label')].find((x) => x.textContent.trim().startsWith(s)).querySelector('input').click(), start);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
};
const q2 = await question('After pass 2 (bits 0 and 1)');
await (await q2.$('input[type=number]')).type(String(run.passes[1].output.indexOf(0b0110)));
await (await q2.$('button.primary')).click();
await pause(120);
check('q-pass2: position 8', (await q2.evaluate((e) => e.textContent)).includes('Correct.'));
check('q-msb: the bit-reversed key', (await choose('from the most significant bit down', 'By the bit-reversed key')).includes('Correct.'));
check('q-negatives: -2.0 still lands above -1.0', (await choose('Flip only the sign bit of every value', '−2.0 still lands above')).includes('Correct.'));
check('q-ties-desc: token 7', (await choose('Tokens 3 and 7 have exactly equal probability', 'Token 7')).includes('Correct.'));
check('q-zero: +0.0 first', (await choose('Without the −0.0 step, a row holds', '+0.0 (token 9)')).includes('Correct.'));

// ---------------------------------------------------------------- phone width
await page.setViewport({ width: 390, height: 800, deviceScaleFactor: 1 });
await pause(300);
const sw = await page.evaluate(() => document.documentElement.scrollWidth);
check('no sideways page scroll at 390 px', sw <= 390, String(sw));
if (out) {
  await page.$eval(ST, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-stepper.png' });
  await page.$eval(EX, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-explorer.png' });
}

// A lab file that is not written yet answers 404; that is the lab's, not this page's.
const errs = errors.filter((e) => !e.includes('status of 404'));
check('no console or page errors', errs.length === 0, errs.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
