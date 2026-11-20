// Browser check for section 15.7: the prediction that opens the register and
// shared-memory budget, the calculator itself (presets, inputs, devices, the cap on
// registers per thread, the Metal case), and the section's questions. Expected values
// come from the model the page runs on (src/sim/gemmbudget.ts), imported directly, and
// from this machine's reports.
// Usage: node e2e/ch15-s157.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { BOOK_LIMITS, BOOK_SHAPE, budget } from '../src/sim/gemmbudget.ts';

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
const B = 'section.bench[aria-label="Register and shared-memory budget"]';
const fmt = (v) => String(Number(v.toPrecision(3)));
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
  await pause(100);
};
const preset = async (key) => {
  await page.$eval(`${B} button[data-preset="${key}"]`, (b) => b.click());
  await pause(80);
};
/** What the calculator shows. */
const shown = () =>
  page.$eval(B, (b) => {
    const o = b.querySelector('[data-out="occupancy"]');
    const regs = b.querySelector('[data-out="regs"]');
    return {
      occupancy: o.dataset.occupancy,
      blocks: o.dataset.blocks,
      binding: o.dataset.binding,
      wanted: Number(regs.dataset.wanted),
      regs: Number(regs.dataset.regs),
      bound: b.querySelector('[data-out="bound"]')?.dataset.bound ?? null,
      shared: Number(b.querySelector('[data-out="shared"]').dataset.bytes),
      fits: b.querySelector('[data-out="fits"]')?.textContent ?? null,
      text: b.textContent.replace(/\s+/g, ' '),
      rows: Object.fromEntries([...b.querySelectorAll('tr[data-limit]')].map((r) => [r.dataset.limit, r.dataset.fit])),
    };
  });
/** Does the page show what the model gives for this shape and these limits? */
const agrees = (s, b) =>
  s.wanted === b.wanted &&
  s.regs === b.regs &&
  s.occupancy === (b.occupancy == null ? 'none' : fmt(b.occupancy)) &&
  s.blocks === String(b.blocks ?? 'none') &&
  s.binding === b.binding.join('+') &&
  s.shared === b.shared &&
  (b.threadBound == null ? s.bound == null : s.bound === String(b.threadBound));

await goto(page, 'ch15/15.7');
await page.waitForSelector('.q');

// ---- the first question comes before the arithmetic
const q = async (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const numeric = async (needle, v) => {
  const h = await q(needle);
  await (await h.$('input[type=number]')).type(String(v));
  await (await h.$('button.primary')).click();
  await pause(120);
  return h.evaluate((e) => e.textContent);
};
const choose = async (needle, label) => {
  const h = await q(needle);
  await h.evaluate((e, label) => [...e.querySelectorAll('label')].find((l) => l.textContent.startsWith(label)).querySelector('input').click(), label);
  await (await h.$('button.primary')).click();
  await pause(120);
  return h.evaluate((e) => e.textContent);
};
const q80 = await numeric('fewest registers it needs', 79);
check('q-80: 79 is marked wrong and the answer, 80, is given', q80.includes('Not quite. The answer is 80'));
{
  const h = await q('fewest registers it needs');
  await h.evaluate((e) => {
    const el = e.querySelector('input[type=number]');
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '80');
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await pause(80);
  await (await h.$('button.primary')).click();
  await pause(120);
  check('q-80: 80 registers is accepted, with 64 + 8 + 8', (await h.evaluate((e) => e.textContent)).includes('Correct.') && (await h.evaluate((e) => e.textContent)).includes('64 + 8 + 8 = 80'));
}

// ---- the prediction opens the calculator
const gated = !!(await page.$('[data-out="gate"]'));
if (gated) {
  check('before the prediction the calculator is closed', !(await page.$(B)));
  const p = await q('how many blocks of 256 threads');
  await (await p.$('input[type=number]')).type('1');
  await (await p.$('button.primary')).click();
  await pause(200);
  const t = await p.evaluate((e) => e.textContent);
  check('committing 1 reveals the arithmetic: 65,536 / 255 = 257, one block, 12.5%', t.includes('You predicted 1 blocks') && t.includes('65,536 / 255 = 257') && t.includes('That is what you had.') && t.includes('12.5%'), t);
} else {
  check('a prediction from an earlier visit already opens the calculator', true);
}
await page.waitForSelector(B);
await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));
check('the calculator opens on the book\'s kernel', await page.$eval(`${B} button[data-preset="book"]`, (b) => b.getAttribute('aria-pressed') === 'true'));

// ---- the book's numbers
const book = { ...BOOK_SHAPE, extra: 175 };
let s = await shown();
let m = budget(book, BOOK_LIMITS);
check('book preset agrees with the model: 255 registers, 257 threads, 1 block, 12.5%, registers bind', agrees(s, m) && s.occupancy === '12.5' && s.bound === '257' && s.blocks === '1' && s.binding === 'registers', JSON.stringify(s).slice(0, 300));
check('the table shows 1 block by registers, 8 by thread slots, 28 by shared memory', s.rows.registers === '1' && s.rows.threads === '8' && s.rows.shared === '28', JSON.stringify(s.rows));
check('to fit a second block a thread would need 128 registers', (await page.$eval(`${B} [data-out="next-regs"]`, (e) => e.textContent)) === '128');
check('shared memory: 4,096 + 4,096 = 8,192 B', s.shared === 8192 && s.text.includes('4,096 B (4 KB)'));
if (out) await page.screenshot({ path: out + '-book.png' });

for (const [key, shape, want] of [
  ['tiles', { ...BOOK_SHAPE, extra: 0 }, { regs: 80, blocks: '3', occupancy: '37.5' }],
  ['r128', { ...BOOK_SHAPE, extra: 48 }, { regs: 128, blocks: '2', occupancy: '25' }],
  ['pipe', { ...BOOK_SHAPE, extra: 175, pad: 1, buffers: 2 }, { regs: 255, blocks: '1', occupancy: '12.5' }],
]) {
  await preset(key);
  s = await shown();
  m = budget(shape, BOOK_LIMITS);
  check(`preset ${key}: ${want.regs} registers, ${want.blocks} block(s), ${want.occupancy}%, as the model says`, agrees(s, m) && s.regs === want.regs && s.blocks === want.blocks && s.occupancy === want.occupancy, JSON.stringify(s).slice(0, 300));
}
check('the padded, double-buffered tiles take 17,408 bytes', s.shared === 17408 && s.text.includes('twice over'));

// ---- typing: more registers than a thread may have
await preset('book');
await setInput(`${B} input[aria-label="Extra registers"]`, 300);
s = await shown();
check('asking for 380 registers is capped at 255 and flagged', s.wanted === 380 && s.regs === 255 && s.text.includes('over the limit') && agrees(s, budget({ ...BOOK_SHAPE, extra: 300 }, BOOK_LIMITS)));
await setInput(`${B} input[aria-label="Threads per block"]`, 128);
s = await shown();
check('128 threads of 8 x 8 do not cover a 128 x 128 tile, and the page says so', s.text.includes('check the shape') && agrees(s, budget({ ...BOOK_SHAPE, extra: 300, threads: 128 }, BOOK_LIMITS)));
await preset('book');

// ---- custom limits: a 1536-thread SM
await setSelect(`${B} select[aria-label="Limits of one SM"]`, 'custom');
await setInput(`${B} input[aria-label="Thread slots per SM"]`, 1536);
s = await shown();
check('on a custom SM with 1536 thread slots the same block is 16.7%', s.occupancy === '16.7' && agrees(s, budget(book, { ...BOOK_LIMITS, threadsPerSm: 1536 })), JSON.stringify(s).slice(0, 200));

// ---- the reader's devices
const rep = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const real = Object.entries(rep).filter(([b]) => b !== 'emu' && b !== 'race');
const options = await page.$$eval(`${B} select[aria-label="Limits of one SM"] option`, (o) => o.map((x) => [x.value, x.textContent]));
const metal = real.filter(([, r]) => 'maxThreadgroupMemoryLength' in r.device);
const gpus = real.filter(([, r]) => 'regsPerMultiprocessor' in r.device && 'maxThreadsPerMultiProcessor' in r.device);
check(`the device list is the book's example, ${metal.length + gpus.length} reported device(s) and custom numbers`,
  options.length === 2 + metal.length + gpus.length && options[0][1] === "The book's example SM" && options.at(-1)[0] === 'custom', JSON.stringify(options));
for (const [b, r] of metal) {
  await setSelect(`${B} select[aria-label="Limits of one SM"]`, b);
  s = await shown();
  const cap = Number(r.device.maxThreadgroupMemoryLength);
  const lim = { regsPerSm: 0, threadsPerSm: 0, maxRegsPerThread: 0, sharedPerBlock: cap, sharedPerSm: 0 };
  check(`Metal (${b}): registers "not documented", no occupancy, no register table`, s.occupancy === 'none' && s.text.includes('not documented') && Object.keys(s.rows).length === 0 && agrees(s, budget(book, lim)));
  check(`Metal (${b}): 8,192 B ${8192 <= cap ? 'fits' : 'does not fit'} the reported ${cap} B`, s.fits === (8192 <= cap ? 'fits' : 'does not fit'), String(s.fits));
  await preset('pipe');
  s = await shown();
  check(`Metal (${b}): the double-buffered 17,408 B against the same limit`, s.fits === (17408 <= cap ? 'fits' : 'does not fit'));
  const runs = await page.$eval(`${B} [data-out="measured"]`, (e) => [Number(e.dataset.runs), e.textContent]);
  const labRows = r.results.filter((x) => x.lab.startsWith('ch15/') && x.variant === 'solution').flatMap((x) => x.bench.filter((y) => typeof (y.max_threads_per_threadgroup ?? y.maxTotalThreadsPerThreadgroup) === 'number'));
  check(`Metal (${b}): the labs' pipeline rows are shown (${labRows.length})`, runs[0] === labRows.length && (labRows.length ? runs[1].includes('threads per threadgroup') : runs[1].includes('No chapter 15 lab')), runs[1]);
  if (out) await page.screenshot({ path: out + '-metal.png' });
  await preset('book');
}
for (const [b, r] of gpus) {
  await setSelect(`${B} select[aria-label="Limits of one SM"]`, b);
  s = await shown();
  const d = r.device;
  const lim = {
    regsPerSm: Number(d.regsPerMultiprocessor), threadsPerSm: Number(d.maxThreadsPerMultiProcessor),
    maxRegsPerThread: b === 'hip' ? 256 : b === 'cuda' ? 255 : 0, sharedPerBlock: Number(d.sharedMemPerBlock) || 0, sharedPerSm: Number(d.sharedMemPerMultiprocessor) || 0,
  };
  check(`${b}: the budget uses the device's own registers and thread slots`, agrees(s, budget(book, lim)), JSON.stringify(s).slice(0, 200));
}
await setSelect(`${B} select[aria-label="Limits of one SM"]`, 'book');

// ---- questions
check('q-1536: 16.7% is accepted', (await numeric('thread limit is 1536', 16.7)).includes('Correct.'));
const r128 = await numeric('fitted the kernel in 128 registers', 2);
check('q-128regs: 2 blocks, and the occupancy API is named for the exact figure', r128.includes('Correct.') && r128.includes('cudaOccupancyMaxActiveBlocksPerMultiprocessor'));
check('q-two-phases: coalescing is wrong, with the reason', (await choose('no longer do for you', 'Coalesce the loads')).includes('not about how many blocks'));
check('q-two-phases: the overlap of phases is right', (await choose('no longer do for you', 'Overlap one block')).includes('Correct.'));
const lb = await choose('__launch_bounds__(256)', 'Without it HIP compiles');
check('q-launch-bounds: right, with AMD quoted and labelled not in the book', lb.includes('Correct.') && lb.includes('unconstrained compilation') && lb.includes('not in the book'));
const mlx = await choose("MLX's Metal GEMM", 'It is a quarter');
check('q-mlx: reading hardware off a library choice is wrong', mlx.includes('Not quite.') && mlx.includes("library's choice"));
check('q-mlx: nothing documented is right', (await choose("MLX's Metal GEMM", 'Nothing documented')).includes('Correct.'));
check('answers went to the (intercepted) progress API', puts >= 1, String(puts));

// ---- the page's own claims
const body = await page.$eval('.prose', (e) => e.textContent.replace(/\s+/g, ' '));
check('the margin note gives the documented thread limits and the chunk caveat', body.includes('1536 on RTX 30, 40 and 50') && body.includes('chunks of undocumented size'));
check('Metal: the labs declare max_total_threads_per_threadgroup(256)', body.includes('[[max_total_threads_per_threadgroup(256)]]'));

// ---- dark theme and narrow screen
await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
await pause(200);
if (out) {
  await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-dark.png' });
}
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
check('at a narrow width the page does not scroll sideways', await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
if (out) {
  await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
