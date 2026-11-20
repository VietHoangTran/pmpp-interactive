// Browser check for section A.1: the "Excess codes" explorer (predictions for both codings,
// the table at every width), the "Bits and value" explorer (bit flips, typed values, format
// switches that keep or re-encode the bits, next up / next down, overflow messages) and the
// questions. Expected values come from src/sim/minifloat.ts, imported here directly.
// Usage: APP_URL=http://localhost:5173 node e2e/appA-sa1.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import {
  BF16, E4M3, FP16, FP32, TOY6,
  decode, encode, excessCode, excessTable, nextUp, round, show, showBits, twosComplement,
} from '../src/sim/minifloat.ts';

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
const X = bench('Excess codes');
const B = bench('Bits and value');
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
const setInput = async (sel, value, event = 'input') => {
  await page.$eval(sel, (el, v, ev) => {
    const proto = el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event(ev, { bubbles: true }));
  }, value, event);
  await pause(80);
};
const part = (scope, name) => text(`${scope} [data-part="${name}"]`);

await goto(page, 'appA/A.1');
await page.waitForSelector(X);
await page.waitForSelector(B);

// ---- Excess codes: the book's pair, exponents -2 and 1 in excess-3
const row = (k) => `${X} [data-coding="${k}"]`;
await page.evaluate((s) => [...document.querySelectorAll(s + ' button')].find((b) => b.textContent.trim() === "B's code").click(), row('excess'));
await pause(80);
let t = await text(row('excess'));
check('excess prediction reveals 001 and 100, B larger, and says right',
  t.includes('001') && t.includes('100') && t.includes("B's code is larger") && t.includes('you said B: right') && excessCode(3, -2) === 1 && excessCode(3, 1) === 4, t);
check('excess codes keep the exponents\' order', (await page.$eval(row('excess'), (e) => e.dataset.sameOrder)) === 'true');
await page.evaluate((s) => [...document.querySelectorAll(s + ' button')].find((b) => b.textContent.trim() === "B's code").click(), row('twos'));
await pause(80);
t = await text(row('twos'));
check("two's complement: -2 is 110, 1 is 001, A larger, the prediction B is marked wrong",
  t.includes('110') && t.includes('001') && t.includes("A's code is larger") && t.includes('you said B: wrong') && twosComplement(3, -2) === 6, t);
check("two's-complement codes reverse the order here", (await page.$eval(row('twos'), (e) => e.dataset.sameOrder)) === 'false');
const twosOrder = excessTable(3).slice().sort((a, b) => a.twos - b.twos).map((r) => (r.value === null ? '(reserved)' : String(r.value).replace('-', '−'))).join(', ');
check('the two\'s-complement order line matches the model', (await part(X, 'twos-order')).includes(twosOrder), twosOrder);
const tableAt = async () => page.$$eval(X + ' tbody tr', (rs) => rs.map((r) => [...r.querySelectorAll('td')].map((c) => c.textContent.trim())));
let rows = await tableAt();
check('Fig. A.1 at e = 3, row by row', JSON.stringify(rows.map((r) => r.slice(0, 3))) === JSON.stringify(
  excessTable(3).map((r) => [r.twos.toString(2).padStart(3, '0'), r.value === null ? 'reserved' : String(r.value).replace('-', '−'), r.excess.toString(2).padStart(3, '0')])), JSON.stringify(rows));
check('the picked exponents are marked in the table', rows.find((r) => r[3] === 'A')?.[1] === '−2' && rows.find((r) => r[3] === 'B')?.[1] === '1');
for (const e of [2, 4, 5]) {
  await setInput(`${X} input[aria-label="exponent bits"]`, e);
  rows = await tableAt();
  const model = excessTable(e);
  const ok = rows.length === 2 ** e && rows.every((r, i) => r[2] === model[i].excess.toString(2).padStart(e, '0')) && rows[rows.length - 1][1] === 'reserved';
  check(`width ${e}: ${2 ** e} rows in excess order, the all-ones code reserved`, ok, JSON.stringify(rows.slice(-2)));
}
check('changing the width clears the predictions', !(await text(X)).includes('you said'));

// ---- Bits and value
check("starts at the book's worked example: 0.5 is 0 010 00", (await part(B, 'pattern')) === '0 010 00' && (await part(B, 'value')) === '0.5');
const v9 = (v) => show(v, 9);
check('exponent row says E - bias = -1, hidden bit 1', (await part(B, 'unbiased')) === '−1' && (await part(B, 'hidden')) === '1');
check("the book's format has no not-in-the-book badge", (await page.$(`${B} [data-part="badge"]`)) === null);
await setInput(`${B} input[aria-label="value to encode"]`, '0.75');
check('typing 0.75 gives 0 010 10 (the q-encode answer)', (await part(B, 'pattern')) === showBits(TOY6, encode(TOY6, 0.75)) && (await part(B, 'pattern')) === '0 010 10');
check('0.75 is exact in the 6-bit format', (await part(B, 'typed')).includes('Exact'));
await setInput(`${B} input[aria-label="value to encode"]`, '0.6');
check('0.6 rounds to the nearest 6-bit value', (await part(B, 'typed')).includes(show(round(TOY6, 0.6))), await part(B, 'typed'));

await press(B, 'FP16');
check('switching width re-encodes the value', (await part(B, 'note')).includes('Re-encoded') && (await part(B, 'pattern')) === showBits(FP16, encode(FP16, 0.6)));
check('FP16 carries the not-in-the-book badge', (await part(B, 'badge')) === 'not in the book');
await setInput(`${B} input[aria-label="value to encode"]`, '1');
check('1 in FP16 is 0 01111 0000000000', (await part(B, 'pattern')) === '0 01111 0000000000');
await press(B, 'BF16');
const same = decode(BF16, 0x3c00).value;
check('switching FP16 to BF16 keeps the 16 bits and changes the value', (await part(B, 'note')).includes('Same 16 bits') && (await part(B, 'value')) === v9(same), `${await part(B, 'value')} vs ${v9(same)}`);
await page.click(`${B} button[data-bit="14"]`);
await pause(80);
check('flipping BF16 bit 14 gives the model\'s value', (await part(B, 'value')) === v9(decode(BF16, 0x3c00 ^ 0x4000).value), await part(B, 'value'));
await page.click(`${B} button[data-bit="14"]`);
await pause(80);
await press(B, 'Next value up');
check('next value up walks to the model\'s neighbour', (await part(B, 'value')) === v9(nextUp(BF16, same)), await part(B, 'value'));
if (out) await page.screenshot({ path: out + '-bits.png' });

await press(B, 'FP16');
await setInput(`${B} input[aria-label="value to encode"]`, '70000');
check('70,000 in FP16 overflows to infinity', (await part(B, 'typed')).includes('overflows') && (await part(B, 'value')) === '+inf' && (await part(B, 'class')) === 'infinity');
await press(B, 'E4M3');
await setInput(`${B} input[aria-label="value to encode"]`, '1000');
check('1000 in E4M3 has no infinity to go to: NaN', (await part(B, 'typed')).includes('no infinity') && (await part(B, 'value')) === 'NaN');
check('next buttons are disabled on NaN', !(await press(B, 'Next value up')));
await setInput(`${B} input[aria-label="value to encode"]`, '300');
check('300 in E4M3 rounds to the model value', (await part(B, 'value')) === v9(round(E4M3, 300)));
await press(B, 'FP32');
await setInput(`${B} input[aria-label="value to encode"]`, '0.1');
check('0.1 in FP32 is the platform float32, 0.100000001 to 9 digits', (await part(B, 'value')) === v9(Math.fround(0.1)) && (await part(B, 'value')) === '0.100000001' && (await part(B, 'pattern')) === showBits(FP32, encode(FP32, 0.1)));
check('a formula is shown with numbers substituted', (await part(B, 'formula')).includes('127'), await part(B, 'formula'));
await setInput(`${B} input[aria-label="value to encode"]`, '1e-40');
check('1e-40 in FP32 is subnormal with the hidden bit 0', (await part(B, 'class')).startsWith('subnormal') && (await part(B, 'hidden')) === '0');

// ---- questions
const q = async (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const choose = async (needle, index) => {
  const h = await q(needle);
  await h.evaluate((e, i) => e.querySelectorAll('.opts input')[i].click(), index);
  await pause(50);
  await h.evaluate((e) => e.querySelector('button.primary').click());
  await pause(120);
  return h.evaluate((e) => e.textContent);
};
check('q-encode accepts 001010', (await choose('which pattern stores 0.75', 0)).includes('Correct.'));
check('q-unsigned-compare accepts yes and explains the negative case', (await choose('sorts those integers', 0)).includes('§14.4'));
check('q-bf16-top marks the round-to-nearest option wrong', (await choose('top 16 bits of an FP32 value', 1)).includes('Not quite.'));
const mh = await q('largest finite value');
await (await mh.$('input[type=number]')).type('65504');
await mh.evaluate((e) => e.querySelector('button.primary').click());
await pause(120);
check('q-max-half accepts 65504', (await mh.evaluate((e) => e.textContent)).includes('Correct.'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
