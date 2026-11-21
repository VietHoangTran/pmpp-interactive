// Browser check for section 12.5 (the stable filter by scan): the prediction, the explorer
// "Keep, scan, scatter" in its three skins, the block stepper "Inside one block", the lab
// listing and three questions. Expected values are the ones src/sim/stablescan.test.ts pins
// (Fig. 12.7's offsets 0 0 1 1 2 3 3 4 4 5 5 6 7 7 7 8 and total 8, p-size 5, q-blockrank 7,
// q-ex1-bytes 30); where the page computes from seeded data, the check recomputes from
// what the page shows.
// Usage: APP_URL=http://localhost:5173 node e2e/ch12-s125.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';
import { checkLabStatuses } from './labstatus.mjs';

const out = process.argv[2] && !process.argv[2].startsWith('--') ? process.argv[2] : null;
const dark = process.argv.includes('--dark');
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
const bench = (title) => `section.bench[aria-label="${title}"]`;
const E = bench('Keep, scan, scatter');
const B = bench('Inside one block: votes, then a scan of warp totals');
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const row = (scope, name) => page.$$eval(`${scope} [data-row="${name}"] > span`, (cs) => cs.map((c) => c.childNodes[0]?.textContent ?? c.textContent));
const clickIn = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const btn = [...document.querySelectorAll(scope + ' button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(80);
  return done;
};
const slide = async (scope, label, value) => {
  await page.$eval(`${scope} input[aria-label="${label}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(100);
};
const total = () => page.$eval(E + ' [data-total]', (e) => Number(e.dataset.total));

await goto(page, 'ch12/12.5');
await page.waitForSelector(E);

// ---------------------------------------------------------------- the prediction
const pq = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Predict first') && e.textContent.includes('1 0 0 1 1 0 1 1')));
await (await pq.$('input[type=number]')).type('5');
await (await pq.$('button.primary')).click();
await pause(150);
const reveal = await pq.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
check('the prediction reveals an output size of 5 from offset 4 plus keep 1', reveal.includes('the size is 5') && reveal.includes('offset is 4'), reveal);

// ---------------------------------------------------------------- the explorer, keys
check('Fig. 12.7\'s offsets', JSON.stringify(await row(E, 'offset')) === JSON.stringify(['0', '0', '1', '1', '2', '3', '3', '4', '4', '5', '5', '6', '7', '7', '7', '8']), JSON.stringify(await row(E, 'offset')));
check('and total 8', (await total()) === 8 && (await text(E)).includes('= 8 + 0 = 8'));
check('the output is k1 k3 k4 k6 k8 k10 k11 k14', JSON.stringify(await row(E, 'output')) === JSON.stringify(['k1', 'k3', 'k4', 'k6', 'k8', 'k10', 'k11', 'k14']), JSON.stringify(await row(E, 'output')));
check('the offsets of dropped keys are greyed', (await page.$$(E + ' [data-row="offset"] > span.void')).length === 8);
await page.$eval(E + ' [data-row="input"] [data-i="11"]', (e) => e.click());
await pause(100);
check('clicking k11 shows the six earlier keepers its offset counts', (await text(E)).includes('k11 has offset 6: the kept keys before it are k1, k3, k4, k6, k8, k10') && (await page.$$(E + ' [data-row="input"] > span.good')).length === 6);
await page.$eval(E + ' [data-row="keep"] [data-i="0"]', (e) => e.click());
await pause(100);
check('keeping k0 too shifts every offset by one: total 9', (await total()) === 9 && (await row(E, 'offset'))[15] === '9');
await clickIn(E, 'The book\'s flags');
check('"The book\'s flags" restores total 8', (await total()) === 8);
if (out) await page.$eval(E, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-keys.png' });

// ---------------------------------------------------------------- logits
await clickIn(E, 'logits, min-p');
await slide(E, 'min-p', 10);
const label = await text(E + ' label.field');
const thr = Number(/= (-?\d+\.\d+)$/.exec(label)?.[1]);
const logits = (await row(E, 'input')).map(Number);
const keepL = (await row(E, 'keep')).map(Number);
check('min-p at 0.10: keep is exactly logit >= max + ln p', Number.isFinite(thr) && Math.abs(thr - (Math.max(...logits) + Math.log(0.1))) < 0.006 && keepL.every((k, i) => k === (logits[i] >= thr ? 1 : 0)), `${label} ${JSON.stringify(keepL)}`);
check('and the total counts them', (await total()) === keepL.reduce((a, b) => a + b, 0));
await slide(E, 'min-p', 100);
check('min-p at 1.00 keeps only the maximum', (await total()) === logits.filter((l) => l === Math.max(...logits)).length);

// ---------------------------------------------------------------- top-p
await clickIn(E, 'sorted probabilities, top-p');
let allPrefix = true;
for (const p of [5, 30, 55, 80, 95, 100]) {
  await slide(E, 'top-p', p);
  const k = (await row(E, 'keep')).map(Number);
  const mass = (await row(E, 'mass')).map(Number);
  const prefix = (await page.$eval(E + ' [data-prefix]', (e) => e.dataset.prefix)) === 'yes';
  const cut = k.indexOf(0) < 0 ? k.length : k.indexOf(0);
  if (!prefix || !k.every((x, i) => x === (mass[i] < p / 100 ? 1 : 0)) || k.slice(cut).some((x) => x) || (await total()) !== cut) allPrefix = false;
}
check('top-p keeps a prefix for every p, keep = mass before < p, total = cut index', allPrefix);
if (out) await page.screenshot({ path: out + '-topp.png' });

// ---------------------------------------------------------------- the block stepper
for (let i = 0; i < 6; i++) await clickIn(B, 'Step');
check('six steps end on the block offsets', (await page.$eval(B + ' [data-phase]', (e) => e.dataset.phase)) === '6');
check('the vote-and-scan offsets equal the plain scan (toy warps of 8)', (await page.$eval(B + ' [data-same]', (e) => e.dataset.same)) === 'yes');
const flags = await page.$$eval(B + ' [data-warp] .mx.shared > span', (cs) => cs.map((c) => Number(c.textContent)));
const totals = await page.$$eval(B + ' [data-total]', (cs) => cs.map((c) => Number(c.textContent.replace(/\D/g, ''))));
const starts = await page.$$eval(B + ' [data-start]', (cs) => cs.map((c) => Number(c.textContent.replace(/\D/g, ''))));
const sum = (a) => a.reduce((x, y) => x + y, 0);
check('warp totals count each warp\'s flags', totals.length === 4 && totals.every((t, w) => t === sum(flags.slice(w * 8, w * 8 + 8))), JSON.stringify(totals));
check('warp starts are the exclusive scan of the totals', starts.every((s, w) => s === sum(totals.slice(0, w))), JSON.stringify(starts));
const masks = await page.$$eval(B + ' [data-mask] code', (cs) => cs.map((c) => c.textContent));
check('masks are printed lane 0 on the right', masks.every((m, w) => [...m].reverse().join('') === flags.slice(w * 8, w * 8 + 8).join('')), JSON.stringify(masks));
// Follow lane 5 of warp 2.
await page.$eval(B + ' [data-ranks="2"] [data-t="21"]', (e) => e.click());
await pause(100);
const arith = await text(B + ' [data-lane-arith]');
const rank = sum(flags.slice(16, 21));
check('lane 5 of warp 2: offset = start of warp 2 + rank', arith.includes(`rank = popcount`) && arith.includes(`= ${starts[2]} + ${rank} = ${starts[2] + rank}`), arith);
await page.$eval(B + ' [data-warp="0"] .mx.shared > span[data-t="0"]', (e) => e.click());
await pause(100);
check('flipping a flag keeps the offsets equal to the plain scan', (await page.$eval(B + ' [data-same]', (e) => e.dataset.same)) === 'yes');
await clickIn(B, 'warps of 32 lanes, 128 threads');
check('128 threads in warps of 32: four warps, still equal to the plain scan', (await page.$$(B + ' [data-warp]')).length === 4 && (await page.$$(B + ' [data-warp="0"] .mx.shared > span')).length === 32 && (await page.$eval(B + ' [data-same]', (e) => e.dataset.same)) === 'yes');
if (out) await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-block.png' });

// ---------------------------------------------------------------- the listing
const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
check('the listing has CUDA, Metal and the provided scan', ['CUDA', 'Metal kernel', 'the provided scan'].every((t) => tabs.includes(t)), tabs.join('|'));
const cu = await text('figure.code pre');
check('the CUDA tab is lab 12.3 part A: keep flags, scatter and their launch', cu.includes('keepFlagsKernel') && cu.includes('scatterKernel') && cu.includes('launchExclusiveScanU32') && !cu.includes('stagedScatterKernel'));

// ---------------------------------------------------------------- questions
const question = (needle) => page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), needle);
for (const [needle, value, name] of [['in-warp rank of 4', '7', 'q-blockrank accepts 7'], ['how many bytes of global traffic per', '30', 'q-ex1-bytes accepts 30']]) {
  const q = await question(needle);
  await (await q.$('input[type=number]')).type(value);
  await (await q.$('button.primary')).click();
  await pause(150);
  check(name, (await q.evaluate((e) => e.textContent)).includes('Correct.'));
}
const qt = await question('What does the keep row look like');
await qt.evaluate((e) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes('a prefix')).querySelector('input').click());
await (await qt.$('button.primary')).click();
await pause(150);
check('q-topp: a prefix is right', (await qt.evaluate((e) => e.textContent)).includes('Correct.'));

// The listings' statuses: LabKit's words from this machine's reports, and "failed on this machine" when a run failed.
await checkLabStatuses(check, 'ch12/12.5', [
  { label: 'CUDA', lab: 'ch12/lab3_stable', file: 'kernel.cu' },
  { label: 'Metal kernel', lab: 'ch12/lab3_stable', file: 'kernel.metal' },
]);

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
