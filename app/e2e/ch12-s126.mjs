// Browser check for section 12.6 (staging and coarsening): the prediction, the explorer
// "Who writes each output slot?", the lab listing, the bytes table and three questions. The
// narrated scene has its own check (ch12-scene-stage-write.mjs). Expected values are the ones
// src/sim/compactstore.test.ts and stablescan.test.ts pin (preset: 32 direct stores and 88
// segments, 18 staged stores and 77 segments; 30, 8 + 4f + 16/256 and 6 bytes per key).
// Usage: APP_URL=http://localhost:5173 node e2e/ch12-s126.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';

const out = process.argv[2] && !process.argv[2].startsWith('--') ? process.argv[2] : null;
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000, dark });
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (title) => `section.bench[aria-label="${title}"]`;
const E = bench('Who writes each output slot?');
const BT = bench('Bytes per key, by design');
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const select = async (scope, label, value) => {
  await page.evaluate((scope, label, value) => {
    const s = document.querySelector(`${scope} select[aria-label="${label}"]`);
    s.value = String(value);
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, scope, label, value);
  await pause(150);
};
const slide = async (scope, label, value) => {
  await page.$eval(`${scope} input[aria-label="${label}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(150);
};
const totals = () => page.$$eval(E + ' tr[data-design]', (rs) => Object.fromEntries(rs.map((r) => [r.dataset.design, {
  stores: Number(r.querySelector('[data-stores]').dataset.stores),
  segments: Number(r.querySelector('[data-segments]').dataset.segments),
  chunks: Number(r.querySelector('[data-chunks]').dataset.chunks),
  staging: Number(r.querySelector('[data-staging]').dataset.staging),
}])));

await goto(page, 'ch12/12.6');
await page.waitForSelector(E);

// ---------------------------------------------------------------- the prediction
const pq = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Predict first') && e.textContent.includes('How many does the staged version issue')));
check('the prediction states the preset and the direct count of 32', (await pq.evaluate((e) => e.textContent)).includes('issues 32 store instructions'));
await (await pq.$('input[type=number]')).type('18');
await (await pq.$('button.primary')).click();
await pause(150);
const reveal = await pq.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
check('the reveal gives the model\'s 18 and the segments 88 to 77', reveal.includes('The model says 18, as you said') && reveal.includes('from 88 to 77'), reveal);

// ---------------------------------------------------------------- the explorer
let t = await totals();
check('preset: direct 32 stores and 88 segments; staged 18 and 77', t.direct.stores === 32 && t.direct.segments === 88 && t.staged.stores === 18 && t.staged.segments === 77, JSON.stringify(t));
check('preset: four blocks, so four chunks either way', t.direct.chunks === 4 && t.staged.chunks === 4);
const strip = await page.$$eval(E + ' [data-strip="0"] [data-slot]', (gs) => gs.filter((g) => g.dataset.store !== '').length);
const runText = await text(E + ' .mx-label');
const runLen = Number(/\((\d+) keys\)/.exec(runText)?.[1]);
check('the strip shades exactly block 0\'s run', strip === runLen && runLen > 0, `${strip} vs ${runText}`);
await page.$eval(E + ' details', (d) => { d.open = true; });
await pause(100);
const listed = await page.$$eval(E + ' details tbody tr', (r) => r.length);
check('block 0 lists one store per warp in the direct scatter (8)', listed === 8, String(listed));
await select(E, 'Design', 'staged');
check('staged: block 0 needs four or five stores', [4, 5].includes(await page.$$eval(E + ' details tbody tr', (r) => r.length)));
const lanes = await page.$$eval(E + ' details tbody tr', (rs) => rs.map((r) => r.children[4].textContent));
check('staged stores write ranges of consecutive slots', lanes.every((x) => /^\d+–\d+$/.test(x)), JSON.stringify(lanes));
if (out) await page.$eval(E, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-explorer.png' });

await slide(E, 'Fraction kept', 25);
t = await totals();
check('a quarter kept: staging cuts the stores by more than half', t.staged.stores * 2 < t.direct.stores, JSON.stringify(t));
await slide(E, 'Fraction kept', 100);
t = await totals();
check('everything kept: staging changes nothing in the stores or segments', t.staged.stores === t.direct.stores && t.staged.segments === t.direct.segments, JSON.stringify(t));
check('everything kept: coarsened without staging touches more segments than staged', t['direct-coarse'].segments > 2 * t['staged-coarse'].segments, JSON.stringify(t));
await slide(E, 'Fraction kept', 50);
await select(E, 'Design', 'staged-coarse');
await select(E, 'Threads per block', 128);
await select(E, 'Coarsening factor', 1);
const c1 = (await totals())['staged-coarse'].chunks;
await select(E, 'Coarsening factor', 4);
const c4 = (await totals())['staged-coarse'];
check('coarsening by 4 divides the chunks by 4 (8 to 2 for 1024 keys, 128 threads)', c1 === 8 && c4.chunks === 2, `${c1} -> ${c4.chunks}`);
check('staging 128 x 4 keys takes 2048 bytes', c4.staging === 2048);
await select(E, 'Pattern', 'alternate');
check('every other key: the fraction slider is disabled', await page.$eval(E + ' input[aria-label="Fraction kept"]', (e) => e.disabled));
if (out) await page.screenshot({ path: out + '-explorer-coarse.png' });

// ---------------------------------------------------------------- the listing
const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
check('the listing has CUDA, Metal kernel and Metal host', ['CUDA', 'Metal kernel', 'Metal host'].every((x) => tabs.includes(x)), tabs.join('|'));
const cu = await text('figure.code pre');
check('the CUDA tab is lab 12.3 part B', cu.includes('countKeptKernel') && cu.includes('stagedScatterKernel') && cu.includes('extern __shared__') && !cu.includes('keepFlagsKernel'));
await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')].find((b) => b.textContent === 'Metal host').click());
await pause(100);
check('the Metal host tab sizes threadgroup memory against the device limit', (await text('figure.code pre')).includes('setThreadgroupMemoryLength') && (await text('figure.code pre')).includes('maxThreadgroupMemoryLength'));

// ---------------------------------------------------------------- the bytes table
const bytes = () => page.$$eval(BT + ' tr[data-traffic]', (rs) => Object.fromEntries(rs.map((r) => [r.dataset.traffic, Number(r.querySelector('[data-total]').dataset.total)])));
let b = await bytes();
check('at half kept: 30, 10.06 and 6 bytes per key', b['three-kernels'] === 30 && Math.abs(b.staged - (10 + 16 / 256)) < 1e-9 && b['single-pass'] === 6, JSON.stringify(b));
await slide(BT, 'Fraction kept for the bytes table', 0);
b = await bytes();
check('nothing kept: 28, 8.06 and 4', b['three-kernels'] === 28 && Math.abs(b.staged - (8 + 16 / 256)) < 1e-9 && b['single-pass'] === 4, JSON.stringify(b));
const hasRun = (await page.$$(BT + ' thead th')).length > 4;
check('a measured column only when lab 12.3 has run, else the instructions to run it (no ./verify.sh needed)', hasRun || ((await text(BT)).includes('python3 labs/pmpp.py run 12.3 --solution --bench') && !(await text(BT)).includes('verify.sh')));

// ---------------------------------------------------------------- questions
const question = (needle) => page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), needle);
for (const [needle, value, name] of [['How many contiguous chunks', '8', 'q-chunks accepts 8'], ['needs how many bytes of shared memory', '8192', 'q-budget accepts 8192']]) {
  const q = await question(needle);
  await (await q.$('input[type=number]')).type(value);
  await (await q.$('button.primary')).click();
  await pause(150);
  check(name, (await q.evaluate((e) => e.textContent)).includes('Correct.'));
}
const qd = await question('12 of a warp\'s 32 lanes');
await qd.evaluate((e) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes('consecutive output slots')).querySelector('input').click());
await (await qd.$('button.primary')).click();
await pause(150);
check('q-direct: consecutive slots is right', (await qd.evaluate((e) => e.textContent)).includes('Correct.'));

const expected404 = (u) => /\/content\/ch12\/narration\/(manifest\.json|stage-write-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
