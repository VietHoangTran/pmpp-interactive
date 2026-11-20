// Browser check for section 6.5: the coarsening toy listing, the "who handles what"
// stepper, the pitfall cards, the wave calculator and the questions. Expected values
// are computed by the models the page runs on (src/sim/coarsen.ts and
// src/sim/coalesce.ts), imported directly.
// Usage: node e2e/ch06-s65.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { BOOK_CAPACITY, PITFALLS, calculate, warpInstruction, warpVectorInstruction } from '../src/sim/coarsen.ts';
import { warpAccess } from '../src/sim/coalesce.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
let puts = 0;
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.method() === 'PUT' && r.url().includes('/api/progress')) {
    puts++;
    return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  }
  return r.continue();
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (t) => `section.bench[aria-label="${t}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const clickText = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const root = document.querySelector(scope);
    const btn = [...root.querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(80);
  return done;
};
const setInput = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
    set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(80);
};
const setSelect = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    el.value = String(v);
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(80);
};
const fmt = (v) => Number(v.toPrecision(3)).toString();

await goto(page, 'ch06/6.5');
await page.waitForSelector(bench('Wave calculator'));

// ---- the toy listing
const code = await text('figure.code pre');
check('the toy listing is the snippet, with the coarsening loop highlighted',
  code.includes('unsigned int iStart = COARSEN*(blockIdx.x*blockDim.x + threadIdx.x);') && (await text('figure.code .ln.hl')).includes('iStart'));

// ---- who handles what
const W = bench('Who handles what');
const sectors = () => page.$eval(W + ' [data-k="sectors"]', (e) => Number(e.dataset.sectors));
const outlined = () => page.$$eval(W + ' rect[data-sector]', (r) => r.length);
const loopRow = () => page.$$eval(W + ' tr[data-k="loop"] td', (t) => t.map((x) => x.textContent.trim()));
const contig = warpAccess(warpInstruction(0, 4, 'contiguous', 32, 64), 4, 32);
const strided = warpAccess(warpInstruction(0, 4, 'strided', 32, 64), 4, 32);
const vec = warpAccess(warpVectorInstruction(32), 16, 32);
check('C = 4, contiguous: the instruction touches 16 sectors, as the model says', (await sectors()) === 16 && contig.transactions === 16 && (await outlined()) === 16, String(await sectors()));
check('and uses a quarter of the bytes fetched', (await text(W + ' [data-k="sectors"]')).includes('25%'));
let row = await loopRow();
check('the whole loop: 4 instructions, 64 sectors', row[1] === '4' && row[2] === '64', row.join('|'));
const vrow = await page.$$eval(W + ' tr[data-k="vector"] td', (t) => t.map((x) => x.textContent.trim()));
check('the 16-byte load: 1 instruction, 16 sectors, 100% used', vrow[1] === '1' && vrow[2] === String(vec.transactions) && vrow[3] === '100%', vrow.join('|'));
await clickText(W, 'Next c');
check('Next c moves to c = 1, still 16 sectors', (await text(W + ' [data-k="iteration"]')).includes('c = 1') && (await sectors()) === 16);
if (out) await page.screenshot({ path: out + '-mapping-contiguous.png' });
await clickText(W, 'strided');
check('strided: 4 sectors per instruction, 100% used', (await sectors()) === strided.transactions && strided.transactions === 4 && (await outlined()) === 4 && (await text(W + ' [data-k="sectors"]')).includes('100%'));
row = await loopRow();
check('strided whole loop: 4 instructions, 16 sectors, and no vector row', row[2] === '16' && !(await page.$(W + ' tr[data-k="vector"]')), row.join('|'));
await clickText(W, 'contiguous (the book\'s)');
await clickText(W, 'C = 8');
check('C = 8 contiguous: 32 sectors, one per lane', (await sectors()) === warpAccess(warpInstruction(0, 8, 'contiguous', 32, 64), 4, 32).transactions && (await sectors()) === 32);
const t31 = await page.$$eval(W + ' table.threads tbody tr', (r) => r.map((x) => [...x.querySelectorAll('td')].map((td) => td.textContent.replace(/\s+/g, ' ').trim()).join(': ')));
check("the thread table follows the book's mapping: thread 1 handles 8 to 15 at C = 8", t31[1] === '1: 8 9 10 11 12 13 14 15' && t31[3] === '31: 248 249 250 251 252 253 254 255', t31.join(' / '));

// ---- pitfall cards
const P = bench('Helps or hurts?');
for (const p of PITFALLS) {
  const wrong = ['helps', 'little or no change', 'can hurt'].find((v) => v !== p.answer);
  await page.evaluate((P, id, label) => [...document.querySelector(`${P} [data-card="${id}"]`).querySelectorAll('button')].find((b) => b.textContent === label).click(), P, p.id, wrong);
  await pause(60);
  const r1 = await text(`${P} [data-card="${p.id}"] [data-k="reveal"]`);
  await page.evaluate((P, id, label) => [...document.querySelector(`${P} [data-card="${id}"]`).querySelectorAll('button')].find((b) => b.textContent === label).click(), P, p.id, p.answer);
  await pause(60);
  const r2 = await text(`${P} [data-card="${p.id}"] [data-k="reveal"]`);
  check(`card "${p.id}": a wrong pick shows the answer, the right one says so`, r1.startsWith(`the answer: ${p.answer}`) && r2.startsWith(`right: ${p.answer}`) && r2.includes(p.pages), r1 + ' / ' + r2);
}
// The bandwidth-bound vector addition agrees with 6.3's q-when, 6.8's "vector-full" card and 6.9's model.
const vcard = await text(`${P} [data-card="vector"] [data-k="reveal"]`);
check('card "vector": little or no change for a kernel already at its bandwidth, citing pp. 139, 142',
  vcard.startsWith('right: little or no change') && vcard.includes('(pp. 139, 142)') && vcard.includes('below full occupancy'), vcard);
check('one card helps: coarsening the tiled matmul (p. 142)', PITFALLS.filter((p) => p.answer === 'helps').map((p) => p.id).join() === 'matmul-tiles');

// ---- wave calculator
const C = bench('Wave calculator');
const val = (k) => text(`${C} [data-k="${k}"]`);
const waveRows = () => page.$$eval(`${C} svg text[data-wave]`, (t) => t.map((x) => Number(x.dataset.fill)));
const expectCase = async (name, units, factor, capacity) => {
  const r = calculate(units, 256, factor, capacity);
  const got = { blocks: await val('blocks'), waves: await val('waves'), idle: await val('idle'), verdict: await val('verdict'), rows: await waveRows() };
  const ok = got.blocks === r.blocks.toLocaleString('en-US') && got.waves === fmt(r.waves) && got.idle === `${fmt(100 * r.fill.idle)}%`
    && JSON.stringify(got.rows) === JSON.stringify(r.fill.perWave) && got.verdict.includes(r.fill.whole ? 'only whole waves' : `${r.fill.tail} of ${capacity} slots busy`);
  check(name, ok, JSON.stringify(got));
  return r;
};
const devOpts = await page.$$eval(`${C} select[aria-label="device"] option`, (o) => o.map((x) => ({ t: x.textContent, d: x.disabled })));
check("the device list offers the book's hypothetical 264-block device, and custom", devOpts[0].t === "The book's hypothetical device (264)" && devOpts.some((o) => o.t === 'Custom'), JSON.stringify(devOpts));
const metalOpt = devOpts.find((o) => o.t.includes('(metal)'));
if (metalOpt) check('a Metal report offers no figure, and the option is disabled', metalOpt.d && metalOpt.t.endsWith('no figure'), JSON.stringify(metalOpt));
await clickText(C, "The book's 1056 blocks");
let r = await expectCase('1056 blocks on 264 at once: 4 full waves', 1056 * 256, 1, BOOK_CAPACITY);
check('the model agrees: 4 waves', r.waves === 4);
await setInput(`${C} input[aria-label="coarsening factor"]`, 8);
r = await expectCase('coarsened by 8: 132 blocks, 0.5 wave, half the slots idle', 1056 * 256, 8, BOOK_CAPACITY);
check('the model agrees: 132 blocks, 0.5 wave', r.blocks === 132 && r.waves === 0.5 && r.fill.idle === 0.5);
check('wave time doubles: 8 against 4 uncoarsened', (await val('time')) === '8' && (await text(C + ' table')).includes('against 4 uncoarsened'));
if (out) await page.screenshot({ path: out + '-waves-half.png' });
await clickText(C, "The book's 1584 blocks");
check('the preset resets the factor to 1: 6 waves', (await val('waves')) === '6');
await setInput(`${C} input[aria-label="coarsening factor"]`, 4);
r = await expectCase('1584 blocks coarsened by 4: 396 blocks, 1.5 waves, a partial last wave', 1584 * 256, 4, BOOK_CAPACITY);
check('the model agrees: 1.5 waves, 264 then 132', r.waves === 1.5 && JSON.stringify(r.fill.perWave) === '[264,132]');
await setInput(`${C} input[aria-label="coarsening factor"]`, 6);
await expectCase('a factor of 6 leaves one whole wave', 1584 * 256, 6, BOOK_CAPACITY);
// q-factor's setting on a custom device.
await setSelect(`${C} select[aria-label="device"]`, 'custom');
await setInput(`${C} input[aria-label="custom capacity"]`, 100);
await setInput(`${C} input[aria-label="work units"]`, 102400);
await setInput(`${C} input[aria-label="coarsening factor"]`, 1);
await expectCase('custom device of 100: 102,400 units at factor 1 are 4 whole waves', 102400, 1, 100);
await setInput(`${C} input[aria-label="coarsening factor"]`, 8);
await expectCase('and at factor 8, half a wave', 102400, 8, 100);
await setInput(`${C} input[aria-label="coarsening factor"]`, 4);
await expectCase('and at factor 4, exactly one wave', 102400, 4, 100);
if (out) await page.screenshot({ path: out + '-waves-custom.png' });

// ---- questions
const answerNumeric = async (needle, v) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await (await q.$('input[type=number]')).type(String(v));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
const answerChoice = async (needle, label) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await q.evaluate((e, label) => [...e.querySelectorAll('label')].find((l) => l.textContent.startsWith(label)).querySelector('input').click(), label);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
check('q-factor accepts 4', (await answerNumeric('102,400 work units', 4)).includes('Correct.'));
check('q-occupancy accepts 50', (await answerNumeric('raises registers per thread from 32 to 64', 50)).includes('Correct.'));
check('q-mapping: the right option is marked correct', (await answerChoice('thread i handles elements 4i to 4i + 3', 'Lanes 16 bytes apart')).includes('Correct.'));
check('q-independent: a wrong option explains itself', (await answerChoice('A ReLU kernel: one element per thread', 'About 4×')).includes('Not quite.'));
check('answers went to the (intercepted) progress API', puts >= 1, String(puts));

// ---- narrow screen
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(400);
const narrow = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
check('at a narrow width the page does not scroll sideways', narrow);
if (out) {
  await page.$eval(C, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
