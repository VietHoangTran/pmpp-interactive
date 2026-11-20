// Browser check for section 14.9: the "Bitonic network" stepper (N = 8 and 16, stage by stage,
// the same comparators after a shuffle, the (value, index) comparison, typed values), the
// library table, and two questions. Every expected result is recomputed here from
// src/sim/sortnet.ts.
// Usage: node e2e/ch14-s149.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { TIES16, TIES8, applyNetwork, bitonicCount, bitonicStages, byValue, tieGroups, transpositionCount } from '../src/sim/sortnet.ts';

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
const B = 'section.bench[aria-label="Bitonic network"]';
const clickText = async (label) => {
  const done = await page.evaluate((sel, label) => {
    const btn = [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, B, label);
  await pause(80);
  return done;
};
const fmtP = (p) => (Number.isInteger(p) ? String(p) : p.toFixed(3).replace(/0+$/, '').replace(/\.$/, ''));
const enc = (items) => items.map((x) => `${fmtP(x.key)}:${x.id}`).join(' ');
const shownValues = () => page.$eval(B + ' svg', (e) => e.dataset.values);
const stage = () => page.$eval(B + ' [data-stage]', (e) => Number(e.dataset.stage));
const counter = (k) => page.$eval(`${B} [data-counter="${k}"]`, (e) => e.textContent.trim());
const comps = () => page.$$eval(B + ' [data-comp]', (cs) => cs.map((c) => c.dataset.comp));
const inputItems = () => page.$$eval(B + ' [data-in]', (ts) => ts.map((t, id) => ({ key: Number(t.firstChild.textContent), id })));
const groups = () => page.$$eval(B + ' [data-group]', (gs) => gs.map((g) => `${g.dataset.group}:${g.dataset.ok}`));
const want = (items, byIndex) => applyNetwork(items, bitonicStages(items.length), byValue(byIndex));
const wantGroups = (items, byIndex) => tieGroups(want(items, byIndex).out).map((g) => `${fmtP(g.key)}:${g.increasing ? 'yes' : 'no'}`);
const shot = async (name) => {
  if (!out) return;
  await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(100);
  await page.screenshot({ path: `${out}-${name}.png` });
};

await goto(page, 'ch14/14.9');
await page.waitForSelector(B);

// N = 8, ties preset, stage by stage.
const run8 = want(TIES8, false);
check('starts at stage 0 with the eight-value ties preset', (await stage()) === 0 && (await shownValues()) === enc(TIES8));
check('the network has 24 comparators, drawn whatever the data', (await comps()).length === 24 && bitonicCount(8) === 24);
await clickText('Step');
check('stage 1: the model\'s values, 4 comparators applied', (await shownValues()) === enc(run8.stages[0].after) && (await counter('applied')) === '4');
const sw1 = await page.$$eval(B + ' [data-swapped="yes"]', (cs) => cs.length);
check('stage 1: the comparators drawn as swapped are the model\'s', sw1 === run8.stages[0].swapped.filter(Boolean).length && (await counter('swaps')) === String(sw1));
await clickText('Step');
await clickText('Step');
check('stage 3 matches the model', (await stage()) === 3 && (await shownValues()) === enc(run8.stages[2].after));
await clickText('Back');
check('back returns to stage 2', (await shownValues()) === enc(run8.stages[1].after));
await clickText('Run');
check('run: sorted largest first, as the model', (await shownValues()) === enc(run8.out) && (await counter('applied')) === '24');
check('odd-even transposition for 8 values: 28 comparators', (await counter('transposition')) === String(transpositionCount(8)));
const g8 = await groups();
check('the network reorders the 0.08 and 0.05 groups and keeps the 0.12 pair', g8.join() === wantGroups(TIES8, false).join() && g8.join() === '0.12:yes,0.08:no,0.05:no', g8.join());
await shot('n8');

// The same comparators after a shuffle; only which ones swap changes.
const before = await comps();
await clickText('shuffle values');
const shuffled = await inputItems();
check('shuffle changes the input order', enc(shuffled) !== enc(TIES8.map((x, id) => ({ key: x.key, id }))) || shuffled.some((x, k) => x.key !== TIES8[k].key));
check('shuffle keeps exactly the same comparators', (await comps()).join() === before.join());
await clickText('Run');
check('the shuffled input sorts to the model\'s result', (await shownValues()) === enc(want(shuffled, false).out));

// (value, index) on the ties preset.
await clickText('ties preset');
await page.evaluate((sel) => document.querySelector(sel + ' label.check input').click(), B);
await pause(80);
await clickText('Run');
const tiesStable = await page.$eval(B + ' [data-ties]', (e) => e.dataset.ties);
check('with (value, index) every tie group is in token order', tiesStable === 'stable' && (await shownValues()) === enc(want(TIES8, true).out), tiesStable);
await page.evaluate((sel) => document.querySelector(sel + ' label.check input').click(), B);
await pause(80);

// N = 16.
await clickText('N = 16');
check('N = 16: 80 comparators in 10 stages', (await comps()).length === 80 && (await stage()) === 0 && (await page.$eval(B + ' [data-stage]', (e) => e.textContent)).includes('of 10'));
await clickText('Run');
check('N = 16: the model\'s output and 80 applied, against 120 for transposition',
  (await shownValues()) === enc(want(TIES16, false).out) && (await counter('applied')) === '80' && (await counter('transposition')) === '120');
const g16 = await groups();
check('N = 16: tie groups as the model (three reordered)', g16.join() === wantGroups(TIES16, false).join() && g16.filter((g) => g.endsWith(':no')).length === 3, g16.join());
await shot('n16');

// Typed values.
await clickText('N = 8');
await page.$eval(B + ' input[aria-label="values"]', (e) => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(e, '0.3 0.1 0.3 0.2 0.1 0.3 0.2 0.1');
  e.dispatchEvent(new Event('input', { bubbles: true }));
});
await pause(120);
await clickText('Run');
const typed = [0.3, 0.1, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1].map((key, id) => ({ key, id }));
check('typed values run through the same network to the model\'s result', (await shownValues()) === enc(want(typed, false).out));
await page.$eval(B + ' input[aria-label="values"]', (e) => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(e, '0.3 0.1');
  e.dispatchEvent(new Event('input', { bubbles: true }));
});
await pause(120);
check('a wrong count of values is refused with a message', (await page.$eval(B, (e) => e.textContent)).includes('exactly 8 numbers, none negative'));

// The library table.
const lib = await page.$eval('table[data-table="libraries"]', (e) => e.textContent);
check('the library table names the stable calls and warns off the unstable segmented sort',
  lib.includes('cub::DeviceRadixSort::SortPairsDescending') && lib.includes('hipcub::DeviceRadixSort::SortPairsDescending')
  && lib.includes('StableSortPairsDescending') && lib.includes('never DeviceSegmentedSort::SortPairsDescending') && lib.includes('argSort(_:axis:descending:name:)'));

// Questions.
const q = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('bitonic sorting network for N = 16')));
await (await q.$('input[type=number]')).type('80');
await (await q.$('button.primary')).click();
await pause(150);
check('numeric question: 80 comparators', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
const qn = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('sorts 64 candidates per row')));
await qn.evaluate((e) => [...e.querySelectorAll('.opts label')].find((l) => l.textContent.includes('compares (value, index) pairs')).querySelector('input').click());
await pause(80);
await (await qn.$('button.primary')).click();
await pause(150);
check('choice question: a network on (value, index) pairs', (await qn.evaluate((e) => e.textContent)).includes('Correct.'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
