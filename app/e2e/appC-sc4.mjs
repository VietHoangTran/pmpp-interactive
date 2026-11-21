// Browser check for section C.4: the "Bits to bytes" explorer (address width, marks,
// element and index type, the reader's devices), the "your devices" panel and the
// questions. Expected values come from src/sim/addrwidth.ts, imported here directly,
// and from the app's own /api/reports.
// Usage: APP_URL=http://localhost:5173 node e2e/appC-sc4.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { MARKS, arrayLimit, binary, memoryFigures, reach } from '../src/sim/addrwidth.ts';

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
const B = 'section.bench[aria-label="Bits to bytes"]';
const text = (sel) => page.evaluate((s) => document.querySelector(s)?.textContent.replace(/\s+/g, ' ').trim() ?? '', sel);
const attr = (sel, a) => page.evaluate((s, a) => document.querySelector(s)?.getAttribute(a), sel, a);
const setRange = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(80);
};
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
const reachOut = async () => ({ text: await text(B + ' [data-out="reach"]'), bytes: Number(await attr(B + ' [data-out="reach"]', 'data-bytes')) });
const limitOut = async () => ({ text: await text(B + ' [data-out="limit"]'), bytes: Number(await attr(B + ' [data-out="limit"]', 'data-bytes')), binds: await attr(B + ' [data-out="limit"]', 'data-binds') });
const answer = async (match, value) => {
  const q = await page.evaluateHandle((m) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(m)), match);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
const choose = async (match, option) => {
  const q = await page.evaluateHandle((m) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(m)), match);
  await q.evaluate((e, o) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes(o)).querySelector('input').click(), option);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};

const reports = await (await fetch(BASE + '/api/reports')).json();
const figs = memoryFigures(reports);

await goto(page, 'appC/C.4');
await page.waitForSelector(B);

// ---- questions first, as the page asks
check('q-4gib: 4 is correct', (await answer('A byte address is 32 bits wide', 4)).includes('Correct.'));
check('q-why-special: the 48-bit host address is the reason', (await choose('needed a special allocation', 'A 48-bit host virtual address')).includes('Correct.'));
check('q-fermi-tib: 1 TiB is correct', (await answer("Fermi's unified address space was 40 bits wide", 1)).includes('Correct.'));
check('q-int-index: 8 GiB is correct, and it links to chapter 3 §3.4', (await answer('About how many GiB can the array hold', 8)).includes('Correct.') && (await page.$('.q a[href="#/ch03/3.4"]')) !== null);

// ---- the reader's devices panel
const rows = await page.$$eval('[data-your-memory] tr[data-figure]', (r) => r.map((x) => ({ f: x.dataset.figure, b: Number(x.dataset.bytes) })));
check('your-devices panel lists exactly what the reports hold', JSON.stringify(rows) === JSON.stringify(figs.map((f) => ({ f: f.field, b: f.bytes }))), JSON.stringify(rows));
if (!figs.length) check('with no devices the panel says how to record them', (await text('[data-your-memory]')).includes('python3 labs/pmpp.py run appC/lab1 --bench'));

// ---- the width slider and the marks
let r = await reachOut();
check('starts at 32 bits: 4 GiB', r.text === '4 GiB' && r.bytes === reach(32), JSON.stringify(r));
await setRange(B + ' input[type=range]', 40);
r = await reachOut();
check('40 bits: 1 TiB', r.text === '1 TiB' && r.bytes === reach(40), JSON.stringify(r));
await setRange(B + ' input[type=range]', 33);
const r33 = await reachOut();
await setRange(B + ' input[type=range]', 34);
const r34 = await reachOut();
check('each extra bit doubles the reach (33: 8 GiB, 34: 16 GiB)', r33.text === '8 GiB' && r34.bytes === 2 * r33.bytes, JSON.stringify([r33, r34]));
for (const m of MARKS) {
  await page.click(`${B} button[data-goto="${m.bits}"]`);
  await pause(60);
  r = await reachOut();
  check(`mark ${m.bits} bits jumps the slider and shows ${m.label}`, r.text === m.label && r.text === binary(reach(m.bits)).text && Number(await page.$eval(B + ' input[type=range]', (e) => e.value)) === m.bits, JSON.stringify(r));
}
check('the bar draws all five marks', (await page.$$(B + ' svg [data-mark]')).length === MARKS.length);
check('the bar draws one marker per device figure', (await page.$$(B + ' svg [data-part="device"]')).length === figs.length);

// ---- the wall against the reader's devices
await setRange(B + ' input[type=range]', 32);
if (figs.length) {
  const over = figs.filter((f) => f.bytes > reach(32)).length;
  check('at 32 bits the wall names every device figure above 4 GiB', Number(await attr(B + ' [data-out="wall"]', 'data-over')) === over, String(over));
  await setRange(B + ' input[type=range]', 48);
  check('at 48 bits every figure fits', Number(await attr(B + ' [data-out="wall"]', 'data-over')) === figs.filter((f) => f.bytes > reach(48)).length);
} else check('with no device figures there is no wall line', (await page.$(B + ' [data-out="wall"]')) === null);
if (out) await page.screenshot({ path: out + '-c4-bar.png' });

// ---- the index limit
await setRange(B + ' input[type=range]', 64);
let l = await limitOut();
let e = arrayLimit('int32', 4, 64);
check('int over floats at 64 bits: 8 GiB, the index binds', l.bytes === e.bytes && l.text === '8 GiB' && l.binds === 'index', JSON.stringify(l));
await page.select(B + ' select[aria-label="Element type"]', '2');
await pause(60);
l = await limitOut();
check('int over 2-byte values: 4 GiB', l.bytes === arrayLimit('int32', 2, 64).bytes && l.text === '4 GiB', JSON.stringify(l));
await page.select(B + ' select[aria-label="Element type"]', '0.5');
await pause(60);
l = await limitOut();
check('int over 4-bit values: 1 GiB', l.bytes === arrayLimit('int32', 0.5, 64).bytes && l.text === '1 GiB', JSON.stringify(l));
await page.select(B + ' select[aria-label="Element type"]', '4');
await pause(60);
await press(B + ' [aria-label="Index type"]', 'unsigned int');
l = await limitOut();
check('unsigned int over floats: 16 GiB', l.bytes === arrayLimit('uint32', 4, 64).bytes && l.text === '16 GiB', JSON.stringify(l));
await press(B + ' [aria-label="Index type"]', 'int64_t');
l = await limitOut();
e = arrayLimit('int64', 4, 64);
check('int64_t over floats: the 64-bit address binds, 16 EiB', l.binds === 'address' && e.binds === 'address' && l.text === '16 EiB', JSON.stringify(l));
await press(B + ' [aria-label="Index type"]', 'int');
await setRange(B + ' input[type=range]', 32);
l = await limitOut();
check('int over floats at 32 bits: the address binds at 4 GiB', l.binds === 'address' && l.text === '4 GiB' && arrayLimit('int32', 4, 32).binds === 'address', JSON.stringify(l));
check('the explanation says the address runs out first', (await text(B + ' [data-out="limit-why"]')).startsWith('The address runs out first'));
await setRange(B + ' input[type=range]', 40);
check('at 40 bits the int index binds again', (await limitOut()).binds === 'index');
if (out) await page.screenshot({ path: out + '-c4-limit.png' });

// ---- narrow screen
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((B) => ({ page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, bench: document.querySelector(B).getBoundingClientRect().width <= document.documentElement.clientWidth }), B);
check('at a narrow width nothing scrolls the page sideways', narrow.page && narrow.bench, JSON.stringify(narrow));
if (out) {
  await page.$eval(B, (el) => el.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-c4-narrow.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
