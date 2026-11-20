// Browser check for section A.2: the prediction that unlocks the probe, the "Number line"
// explorer (counts, the gap ratio, Exercises 1 and 2 one slider move away, the three
// conventions, the probe under all three), the "Spacing in real formats" explorer (the
// 70,000 and 1.001 bridge, FP8 overflow with and without saturation, flush-to-zero) and the
// questions. Expected values come from src/sim/minifloat.ts, imported here directly.
// Usage: APP_URL=http://localhost:5173 node e2e/appA-sa2.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { BF16, E4M3, E5M2, FP16, FP32, classify, makeFormat, maxFinite, round, show, values } from '../src/sim/minifloat.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
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
const N = bench('Number line');
const S = bench('Spacing in real formats');
const text = (sel) => page.evaluate((s) => document.querySelector(s)?.textContent.replace(/\s+/g, ' ').trim() ?? '', sel);
const press = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(80);
  return done;
};
const setInput = async (sel, value) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(100);
};
const tick = async (scope, label) => {
  await page.evaluate((scope, label) => [...document.querySelectorAll(scope + ' label.check')].find((l) => l.textContent.includes(label)).querySelector('input').click(), scope, label);
  await pause(100);
};
const summary = () => text(`${N} [data-part="summary"]`);
const counts = () => page.$$eval(`${N} svg:first-of-type g[data-conv]`, (gs) => Object.fromEntries(gs.map((g) => [g.dataset.conv, Number(g.dataset.count)])));
const probeRows = () => page.$$eval(`${N} tr[data-conv]`, (rs) => Object.fromEntries(rs.map((r) => [r.dataset.conv, { v: Number(r.dataset.nearest), err: Number(r.dataset.error) }])));
const spacingRows = () => page.$$eval(`${S} tr[data-format]`, (rs) => Object.fromEntries(rs.map((r) => [r.dataset.format, { v: Number(r.dataset.value), status: r.dataset.status, text: r.textContent }])));
const conv = (c) => makeFormat(2, 2, c);
const near = (a, b) => Math.abs(a - b) < 1e-12;

await goto(page, 'appA/A.2');
await page.waitForSelector(N);
await page.waitForSelector(S);

// ---- the prediction gates the probe readout
check('before the prediction the probe errors are hidden', (await text(`${N} [data-part="locked"]`)).includes('Commit your prediction'));
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('0.2 is rounded to the nearest')));
await (await pred.$('input[aria-label="your prediction"]')).type('0.3');
await pred.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Commit prediction').click());
await pause(150);
const pt = await pred.evaluate((e) => e.textContent);
check('the reveal gives 0.3 and acknowledges the right prediction', pt.includes('0.3') && pt.includes('That is what you had.'), pt.slice(0, 200));

// ---- the book's 5-bit format and the probe 0.2
let pr = await probeRows();
const want = { nozero: round(conv('nozero'), 0.2), abrupt: round(conv('abrupt'), 0.2), denorm: round(conv('denorm'), 0.2) };
check('probe 0.2: no zero 0.5, abrupt 0, denormals 0.25', near(pr.nozero.v, want.nozero) && near(pr.abrupt.v, want.abrupt) && near(pr.denorm.v, want.denorm) && want.nozero === 0.5 && want.abrupt === 0 && want.denorm === 0.25, JSON.stringify(pr));
check('probe 0.2 errors: 0.3, 0.2, 0.05', near(pr.nozero.err, 0.3) && near(pr.abrupt.err, 0.2) && near(pr.denorm.err, 0.05), JSON.stringify(pr));
let c = await counts();
const pos = (f) => values(f).filter((v) => v > 0).length;
check('Fig. A.4 counts: 12, 8 and 11 positive values', c.nozero === pos(conv('nozero')) && c.abrupt === pos(conv('abrupt')) && c.denorm === pos(conv('denorm')) && c.nozero === 12, JSON.stringify(c));
let s = await summary();
check('no zero: 3 intervals, the gap is 4 times the next spacing', s.includes('in 3 intervals') && (await text(`${N} [data-part="ratio"]`)) === '4', s);
const tableRow0 = () => page.$eval(`${N} tr[data-e="0"]`, (r) => r.textContent.replace(/\s+/g, ' '));
check('values table, E = 00 under no zero: 0.5 to 0.875', (await tableRow0()).includes('0.5, 0.625, 0.75, 0.875'), await tableRow0());

// ---- Exercise 1: m = 3
await setInput(`${N} input[aria-label="mantissa bits"]`, 3);
c = await counts();
check('Exercise 1: m = 3 gives 24 positive values with no zero', c.nozero === 24 && c.nozero === pos(makeFormat(2, 3, 'nozero')), JSON.stringify(c));
check('the gap is now 8 times the next spacing', (await text(`${N} [data-part="ratio"]`)) === '8');
// ---- Exercise 2: e = 3, m = 2
await setInput(`${N} input[aria-label="mantissa bits"]`, 2);
await setInput(`${N} input[aria-label="exponent bits"]`, 3);
s = await summary();
check('Exercise 2: e = 3 gives 7 intervals, largest 14', s.includes('in 7 intervals') && s.includes('largest 14') && maxFinite(makeFormat(3, 2, 'nozero')) === 14, s);
check('with e = 3 the region near 0 is magnified', (await page.$$(`${N} svg`)).length === 2);
if (out) await page.screenshot({ path: out + '-numberline.png' });

// ---- the conventions, back on the 5-bit format
await setInput(`${N} input[aria-label="exponent bits"]`, 2);
await press(N, 'abrupt underflow');
s = await summary();
check('abrupt underflow: smallest positive 1, gap still 4 times the spacing', s.includes('smallest 1,') && (await text(`${N} [data-part="ratio"]`)) === '4', s);
check('values table, E = 00 under abrupt underflow: all read as 0', (await tableRow0()).includes('all read as 0'));
await press(N, 'denormals');
check('denormals: the ratio drops to 1', (await text(`${N} [data-part="ratio"]`)) === '1');
check('values table, E = 00 with denormals: 0 to 0.75', (await tableRow0()).includes('0, 0.25, 0.5, 0.75'), await tableRow0());
await setInput(`${N} input[aria-label="probe value"]`, '3.9');
pr = await probeRows();
check('a probe beyond the largest value is clamped to 3.5 in every convention', Object.values(pr).every((r) => r.v === 3.5), JSON.stringify(pr));
await setInput(`${N} input[aria-label="probe value"]`, '0.6');
pr = await probeRows();
check('probe 0.6 matches the model in all three conventions', ['nozero', 'abrupt', 'denorm'].every((k) => near(pr[k].v, round(conv(k), 0.6))), JSON.stringify(pr));

// ---- Spacing in real formats
let r = await spacingRows();
check('x = 1 is exact everywhere', Object.values(r).every((x) => x.v === 1 && x.status === 'normal'), JSON.stringify(r));
await press(S, '70,000');
r = await spacingRows();
check('70,000: FP16 overflows to infinity, BF16 stores 70144, FP32 is exact',
  r.fp16.status === 'inf' && r.bf16.v === round(BF16, 70000) && r.bf16.v === 70144 && r.fp32.v === 70000, JSON.stringify({ fp16: r.fp16.status, bf16: r.bf16.v }));
await press(S, '1.001');
r = await spacingRows();
check('1.001: BF16 rounds to 1, FP16 keeps 1.0009765625', r.bf16.v === 1 && r.fp16.v === round(FP16, 1.001) && r.fp16.v === 1.0009765625 && r.fp32.v === round(FP32, 1.001));
await press(S, '500');
r = await spacingRows();
check('500: E4M3 overflows to NaN (non-saturating), E5M2 stores the model value', r.e4m3.status === 'nan' && r.e5m2.v === round(E5M2, 500), JSON.stringify(r.e4m3));
await tick(S, 'FP8: saturating');
r = await spacingRows();
check('500 with saturation: E4M3 saturates at 448', r.e4m3.status === 'saturated' && r.e4m3.v === 448 && classify(E4M3, 500, { overflow: 'saturate' }).value === 448);
await tick(S, 'FP8: saturating');
await press(S, '2^-20');
r = await spacingRows();
check('2^-20 is subnormal in FP16 and normal in FP32', r.fp16.status === 'subnormal' && r.fp32.status === 'normal' && r.fp16.v === 2 ** -20);
await tick(S, 'denormals off');
r = await spacingRows();
check('denormals off: FP16 flushes 2^-20 to 0, FP32 and BF16 are untouched', r.fp16.status === 'flushed' && r.fp16.v === 0 && r.fp32.status === 'normal' && r.bf16.status === 'normal', JSON.stringify(r.fp16));
await setInput(`${S} input[aria-label="log2 of x"]`, 10);
r = await spacingRows();
check('the slider sets x = 2^10: exact in every format but E2M1', r.fp32.v === 1024 && r.fp16.v === 1024 && r.e2m1.status === 'saturated', JSON.stringify(r.e2m1));
if (out) await page.screenshot({ path: out + '-spacing.png' });

// ---- questions
const q = async (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const numeric = async (needle, v) => {
  const h = await q(needle);
  await (await h.$('input[type=number]')).type(String(v));
  await h.evaluate((e) => e.querySelector('button.primary').click());
  await pause(120);
  return h.evaluate((e) => e.textContent);
};
const choose = async (needle, index) => {
  const h = await q(needle);
  await h.evaluate((e, i) => e.querySelectorAll('.opts input')[i].click(), index);
  await pause(50);
  await h.evaluate((e) => e.querySelector('button.primary').click());
  await pause(120);
  return h.evaluate((e) => e.textContent);
};
check('q-ex1 accepts 24', (await numeric('How many positive values can it represent', 24)).includes('Correct.'));
check('q-ex2 accepts 14', (await numeric('What is the largest representable value', 14)).includes('Correct.'));
check('q-int-half rejects 2048', (await numeric('smallest positive integer that FP16', 2048)).includes('Not quite. The answer is 2049.'));
check('q-ftz accepts the flush answer', (await choose('--ftz=true', 0)).includes('Correct.'));
check('q-range-vs-bits accepts FP32 only', (await choose('activation of 70,000', 3)).includes('Correct.'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
