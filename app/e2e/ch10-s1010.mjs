// Browser check for section 10.10: the coarsened listing, the coarsening explorer (presets,
// every control, the sweep chart, the R choices including "yours" from an injected CUDA
// report), the prediction and its reveal, and the questions. Expected values come from
// src/sim/multiblock.ts and src/sim/coarsen.ts, imported here.
// Usage: APP_URL=http://localhost:5173 node e2e/ch10-s1010.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { checkLabStatuses } from './labstatus.mjs';
import { CH6_DEVICE_R, coarsenSteps, fewestStepsC, justEnoughC, softmaxPreset } from '../src/sim/multiblock.ts';
import { deviceSlots, residentUpperBound } from '../src/sim/coarsen.ts';

const out = process.argv[2];
const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));

async function page0(reports) {
  const { browser, page, errors } = await open({ width: 1440, height: 1000 });
  await page.setRequestInterception(true);
  page.on('request', (r) => {
    if (r.method() === 'PUT' && r.url().includes('/api/progress'))
      return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
    if (r.method() === 'GET' && r.url().endsWith('/api/progress'))
      return r.respond({ status: 200, contentType: 'application/json', body: JSON.stringify({ version: 1, sections: {}, answers: {}, review: {} }) });
    if (reports && r.method() === 'GET' && r.url().includes('/api/reports'))
      return r.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(reports) });
    return r.continue();
  });
  return { browser, page, errors };
}

// A synthetic CUDA report, used only by this check: 10 SMs, 16 block slots and 1536 thread
// slots each, so 256-thread blocks fit 6 per SM and R = 60.
const DEVICE = { name: 'Test GPU (injected by this check)', multiProcessorCount: '10', maxBlocksPerMultiProcessor: '16', maxThreadsPerMultiProcessor: '1536', warpSize: '32' };
const R_YOURS = residentUpperBound(deviceSlots(DEVICE), 256);
const SWEEP = [[1, 100], [2, 180], [4, 260], [8, 300], [16, 310], [32, 290], [64, 240], [128, 200]];
const fake = {
  cuda: {
    generated: 'test',
    device: DEVICE,
    results: [{ lab: 'ch10/lab1_sum', variant: 'solution', backend: 'cuda', tile: 0, built: true, test_rc: 0, tests: [{ case: 'x', pass: true }],
      bench: SWEEP.map(([c, g]) => ({ name: 'any', n: 67108864, block: 256, coarse: c, blocks: 1, seconds: 1, gb_per_s: g })) }],
  },
};

const { browser, page, errors } = await page0(null);
const X = 'section.bench[aria-label="Coarsening explorer"]';
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const click = async (scope, label) => {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(100);
  return ok;
};
const setField = async (scope, label, value) => {
  await page.evaluate((scope, label, value) => {
    const el = document.querySelector(`${scope} [aria-label="${label}"]`);
    const proto = el.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, String(value));
    el.dispatchEvent(new Event(el.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true }));
  }, scope, label, value);
  await pause(120);
};
const shown = () => page.$eval(X + ' [data-testid=outputs]', (e) => ({ ...e.dataset }));
const same = (got, r) =>
  Number(got.blocks) === r.blocks && Number(got.waves) === r.waves && Number(got.total) === r.totalSteps &&
  Number(got.full) === r.fullSteps && Number(got.under) === r.underSteps && Number(got.commits) === r.commits && got.toofew === (r.tooFew ? '1' : '0');

await goto(page, 'ch10/10.10');
await page.waitForSelector(X);

// ---------------------------------------------------------------- listing
await page.waitForFunction(() => document.querySelectorAll('figure.code .ln').length > 5, { timeout: 8000 }).catch(() => undefined);
const hl = await page.$$eval('figure.code .ln.hl', (l) => l.map((x) => x.textContent.trim()));
check('the listing marks the segment, the first load and the coarsening loop', hl[0]?.startsWith('unsigned int segment = coarse * 2') && hl[1]?.startsWith('float partialSum = i < N') && hl[2]?.startsWith('for (unsigned int c = 1') && hl.length === 8, JSON.stringify(hl));

// ---------------------------------------------------------------- explorer: default
let r = coarsenSteps({ N: 1 << 20, block: 256, C: 1, R: CH6_DEVICE_R });
check('default (2^20 floats, 256 threads, C = 1, R = 264) matches the model: 2048 blocks, 8 waves, 72 steps', same(await shown(), r) && r.totalSteps === 72, JSON.stringify(await shown()));
check('the device is full at C = 1', (await text(X + ' [data-testid=warning]')).includes('the device is full'));
check('the sweep chart marks the fewest-steps factor (8) and the book\'s rule (7)',
  (await page.$eval(X + ' svg[data-best]', (e) => [e.dataset.best, e.dataset.rule])).join() === [fewestStepsC({ N: 1 << 20, block: 256, R: 264 }, 512), justEnoughC({ N: 1 << 20, block: 256, R: 264 }, 512)].join() );

// Fig. 10.21
await click(X, 'Fig. 10.21 (a)');
r = coarsenSteps({ N: 32, block: 8, C: 1, R: 1 });
check('Fig. 10.21 (a): 2 blocks one after the other, 8 steps, 2 fully used and 6 underused', same(await shown(), r) && r.totalSteps === 8 && r.underSteps === 6, JSON.stringify(await shown()));
await click(X, 'Fig. 10.21 (b)');
r = coarsenSteps({ N: 32, block: 8, C: 2, R: 1 });
check('Fig. 10.21 (b): one coarsened block, 6 steps, 3 and 3', same(await shown(), r) && r.totalSteps === 6 && r.fullSteps === 3, JSON.stringify(await shown()));
if (out) await page.screenshot({ path: out + '-fig1021.png' });

// q-fig1021 by hand: N 64, C 4, then C 1
await setField(X, 'N', 64);
await setField(X, 'C', 4);
check('N 64, 8 threads, C 4, R 1: 10 steps', Number((await shown()).total) === 10);
await setField(X, 'C', 1);
check('and C 1: 16 steps in 4 waves', Number((await shown()).total) === 16 && Number((await shown()).waves) === 4);

// 2^20 floats on 264 slots: wave quantisation around C = 7 and 8.
await click(X, '2²⁰ floats');
await setField(X, 'C', 7);
r = coarsenSteps({ N: 1 << 20, block: 256, C: 7, R: CH6_DEVICE_R });
check('C = 7: 293 blocks, 2 waves (the last holds 29), 42 steps', same(await shown(), r) && (await text(X + ' [data-testid=outputs]')).includes('the last holds 29 blocks'));
await setField(X, 'C', 8);
r = coarsenSteps({ N: 1 << 20, block: 256, C: 8, R: CH6_DEVICE_R });
check('C = 8: 256 blocks, one wave, 23 steps, and the warning: 8 of 264 slots idle', same(await shown(), r) && (await text(X + ' [data-testid=warning]')).includes('256 blocks for 264 slots: 97% of the device has work'));
await setField(X, 'C', 6);
check('C = 6 takes fewer steps than C = 7 (38 against 42)', Number((await shown()).total) === 38);
if (out) await page.screenshot({ path: out + '-quantisation.png' });

// Clicking a bar picks that factor.
await page.evaluate((X) => document.querySelectorAll(X + ' svg[data-best] rect')[3].dispatchEvent(new MouseEvent('click', { bubbles: true })), X);
await pause(120);
check('clicking the fourth bar sets C = 4', (await page.$eval(X + ' [aria-label="C"]', (e) => e.value)) === '4' && Number((await shown()).blocks) === 512);

// R: choose 256, the q-just-enough device.
await click(X, 'choose');
await setField(X, 'R', 256);
await setField(X, 'C', 8);
r = coarsenSteps({ N: 1 << 20, block: 256, C: 8, R: 256 });
check('R = 256, C = 8 (q-just-enough): exactly one full wave', same(await shown(), r) && (await text(X + ' [data-testid=warning]')).includes('100% of the block slots are busy'));
check('and the chart\'s two markers agree on 8', (await page.$eval(X + ' svg[data-best]', (e) => [e.dataset.best, e.dataset.rule])).join() === '8,8');
await click(X, '1 (Fig. 10.21)');
check('R = 1: every block is its own wave', Number((await shown()).waves) === Number((await shown()).blocks));

// Rows and threads per block.
await click(X, '264 (the book\'s example, ch. 6)');
await setField(X, 'threads per block', 1024);
await setField(X, 'rows', 3);
r = coarsenSteps({ N: 1 << 20, block: 1024, C: 8, R: 264, rows: 3 });
check('3 rows of 2^20 in 1024-thread blocks at C = 8: the model\'s counts', same(await shown(), r), JSON.stringify(await shown()));

// Softmax presets: one 256-thread block per row.
for (const b of [1, 8, 64]) {
  await click(X, `softmax, batch ${b}`);
  r = coarsenSteps(softmaxPreset(b));
  check(`softmax, batch ${b}: ${b} block(s), 601 steps, too few blocks`, same(await shown(), r) && r.blocks === b && r.totalSteps === 601 && (await text(X + ' [data-testid=warning]')).includes(`${b} block${b === 1 ? '' : 's'} for 264 slots`));
}
if (out) await page.screenshot({ path: out + '-softmax.png' });
check('no "yours" option without a CUDA or HIP report; the note says Metal reports neither',
  !(await click(X, 'yours (cuda)')) && (await text(X + ' [data-testid=r-note]')).includes('Metal reports neither'));

// ---------------------------------------------------------------- questions
const ask = async (needle) => page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), needle);
let q = await ask('How many steps in total with a coarsening factor of 4');
await (await q.$('input[type=number]')).type('10');
await (await q.$('button.primary')).click();
await pause(120);
check('q-fig1021 accepts 10', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
q = await ask('Which coarsening factor does the book\'s rule pick');
await q.evaluate((e) => [...e.querySelectorAll('label')].find((l) => l.textContent.trim() === 'C = 8.').querySelector('input').click());
await (await q.$('button.primary')).click();
await pause(120);
check('q-just-enough accepts C = 8', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
q = await ask('Why step by');
await q.evaluate((e) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes('bank conflicts')).querySelector('input').click());
await (await q.$('button.primary')).click();
await pause(120);
check('q-stride rejects bank conflicts and says why', (await q.evaluate((e) => e.textContent)).includes('These loads go to global memory'));

// Prediction with this machine's reports.
let p = await ask('Which factor will be fastest');
await (await p.$('input[type=number]')).type('16');
await p.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.includes('Commit')).click());
await pause(300);
const rv = await p.evaluate((e) => e.textContent);
check('the reveal shows this machine\'s sweep, or says how to measure it', rv.includes('You predicted 16') && (rv.includes('the fastest factor') || rv.includes('No measurement on this machine yet')), rv.slice(0, 160));
// The listings' statuses: LabKit's words from this machine's reports, and "failed on this machine" when a run failed.
await checkLabStatuses(check, 'ch10/10.10', [
  { label: 'CUDA', lab: 'ch10/lab1_sum', file: 'kernel.cu' },
]);

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();

// ---------------------------------------------------------------- injected CUDA report
const second = await page0(fake);
await goto(second.page, 'ch10/10.10');
await second.page.waitForSelector(X);
const pg = second.page;
await pg.evaluate((X) => [...document.querySelectorAll(X + ' button')].find((b) => b.textContent.trim() === 'yours (cuda)').click(), X);
await pause(150);
const note = await pg.$eval(X + ' [data-testid=r-note]', (e) => e.textContent.replace(/\s+/g, ' '));
check(`"yours" reads the injected report: at most ${R_YOURS} blocks of 256 threads`, R_YOURS === 60 && note.includes('Test GPU (injected by this check)') && note.includes('at most 60'), note);
const d = await pg.$eval(X + ' [data-testid=outputs]', (e) => ({ ...e.dataset }));
r = coarsenSteps({ N: 1 << 20, block: 256, C: 1, R: R_YOURS });
check('the explorer then counts waves of 60', Number(d.waves) === r.waves && Number(d.total) === r.totalSteps, JSON.stringify(d));
p = await pg.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Which factor will be fastest')));
await (await p.$('input[type=number]')).type('4');
await p.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.includes('Commit')).click());
await pause(300);
const best = await p.evaluate((e) => e.querySelector('[data-backend="cuda"]')?.dataset.best);
const said = await p.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
check('the reveal marks the fastest measured factor (16) and compares the pick (4: 260 GB/s, 84%)', best === '16' && said.includes('C = 16, at 310 GB/s') && said.includes('ran at 260 GB/s, 84% of the fastest'), said.slice(0, 300));
check('one bar per measured factor', (await p.evaluate((e) => e.querySelectorAll('[data-backend="cuda"] svg rect').length)) === SWEEP.length);
if (out) await pg.screenshot({ path: out + '-reveal.png' });
// A phone: the charts keep their own scale and scroll inside their box; the legend wraps.
await pg.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await pg.evaluate((X) => {
  const sweep = document.querySelector(X + ' svg[data-best]');
  const wave = document.querySelector(X + ' svg[aria-label="Steps, wave by wave"]');
  const legend = document.querySelector(X + ' [data-part="sweep-legend"]');
  const vb = (e) => Number(e.getAttribute('viewBox').split(' ')[2]);
  return {
    overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    sweep: [sweep.getBoundingClientRect().width, vb(sweep)],
    wave: [wave.getBoundingClientRect().width, vb(wave)],
    legend: legend.getBoundingClientRect().right <= innerWidth && legend.textContent.includes('fewest steps'),
  };
}, X);
check('narrow: no sideways page scroll; both charts at full scale (text never shrinks); the legend fits on screen',
  narrow.overflow <= 1 && narrow.sweep[0] >= narrow.sweep[1] && narrow.wave[0] >= narrow.wave[1] && narrow.legend, JSON.stringify(narrow));
check('no console or page errors with injected reports', second.errors.length === 0, second.errors.join(' | '));
await second.browser.close();

console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
