// Browser check for section 4.7, resource partitioning and occupancy: the occupancy
// calculator and the cliff explorer against the model (src/sim/occupancy.ts, imported
// here directly), the exercises 7 and 8 checker, three predictions (one of them
// revealed from this machine's lab 4.4 benchmark when it exists), six questions, the
// code shown, and the page at a narrow width.
// Usage: node e2e/ch04-s47.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import {
  BOOK, EX6, EX7, EX7_ASSIGNMENTS, EX8, EX8_KERNELS, assignment, limitsFromReport, occupancy, pct, regValues, scan, steps,
} from '../src/sim/occupancy.ts';

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
const C = 'section.bench[aria-label="Occupancy calculator"]';
const F = 'section.bench[aria-label="Find the cliff"]';
const X = 'section.bench[aria-label="Exercises 7 and 8"]';
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const attr = (sel, a) => page.$eval(sel, (e, a) => e.getAttribute(a), a);
const setInput = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(80);
};
const setSelect = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, v);
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(100);
};
const kernel = async (threads, regs = 0, shared = 0) => {
  await setInput(`${C} input[aria-label="Threads per block"]`, threads);
  await setInput(`${C} input[aria-label="Registers per thread"]`, regs);
  await setInput(`${C} input[aria-label="Shared memory per block (bytes)"]`, shared);
};
/** What the calculator shows. */
const calc = async () => page.evaluate((C) => {
  const o = document.querySelector(C + ' [data-out=occupancy]');
  const n = (s) => document.querySelector(C + ' ' + s)?.textContent.replace(/,/g, '');
  return {
    value: o.dataset.value,
    binding: o.dataset.binding,
    blocks: n('[data-out=blocks]'),
    idle: n('[data-out=idle]'),
    byThreads: document.querySelector(C + ' [data-out=by-threads]')?.dataset.value ?? null,
    fits: Object.fromEntries([...document.querySelectorAll(C + ' tr[data-budget]')].map((r) => [r.dataset.budget, r.dataset.fit])),
    binds: [...document.querySelectorAll(C + ' tr[data-binds]')].map((r) => r.dataset.budget),
    verdict: document.querySelector(C + ' [data-out=verdict]').textContent.replace(/\s+/g, ' ').trim(),
    panel: document.querySelector(C + ' [aria-live]').textContent.replace(/\s+/g, ' ').trim(),
  };
}, C);
/** The model's answer in the calculator's terms. */
const want = (limits, threads, regs = 0, shared = 0) => {
  const o = occupancy(limits, { threads, regs, shared });
  return {
    value: o.usable ? String(pct(o.byWarps)) : 'n/a',
    binding: o.binding.join('+'),
    blocks: String(o.blocks),
    idle: String(o.idleSlots),
    fits: Object.fromEntries(o.rows.map((r) => [r.key, r.fit == null ? 'none' : String(r.fit)])),
    byThreads: o.usable && o.paddedLanes > 0 ? String(pct(o.byThreads)) : null,
  };
};
const agrees = (s, m) => s.value === m.value && s.binding === m.binding && (m.value === 'n/a' || (s.blocks === m.blocks && s.idle === m.idle))
  && JSON.stringify(s.fits) === JSON.stringify(m.fits) && s.byThreads === m.byThreads && s.binds.join('+') === m.binding;
const show = (s, m) => JSON.stringify({ shown: { ...s, verdict: undefined, panel: undefined }, model: m });
const q = (words) => page.evaluateHandle((w) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(w)), words);
const predict = async (words, v) => {
  const h = await q(words);
  await (await h.$('input[type=number]')).type(String(v));
  await page.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.includes('Commit')).click(), h);
  await pause(200);
  return h;
};
const answerNumeric = async (words, v) => {
  const h = await q(words);
  await (await h.$('input[type=number]')).type(String(v));
  await page.evaluate((e) => e.querySelector('button.primary').click(), h);
  await pause(120);
  return h.evaluate((e) => e.textContent);
};
const answerChoice = async (words, optionWords) => {
  const h = await q(words);
  const picked = await h.evaluate((e, w) => {
    const l = [...e.querySelectorAll('.opts label')].find((x) => x.textContent.includes(w));
    if (!l) return false;
    l.querySelector('input').click();
    return true;
  }, optionWords);
  await pause(60);
  await page.evaluate((e) => e.querySelector('button.primary').click(), h);
  await pause(120);
  return { picked, text: await h.evaluate((e) => e.textContent) };
};

const reports = await (await fetch(BASE + '/api/reports')).json();
const cudaLike = Object.keys(reports).filter((b) => limitsFromReport(reports[b].device, b));

await goto(page, 'ch04/4.7');
await page.waitForSelector(C);
await page.waitForSelector(F);

// ---- margin notes
const notes = await page.$$eval('aside.note', (a) => a.map((n) => n.textContent.replace(/\s+/g, ' ').trim()));
check('book claims carry their pages (p. 85, pp. 85–86, p. 86, p. 87, the exercises pp. 91–92)', ['§4.7, p. 85', '§4.7, pp. 85–86', '§4.7, p. 86', '§4.7, p. 87', '§4.10, pp. 91–92'].every((p) => notes.some((n) => n.startsWith('Book ' + p))), notes.filter((n) => n.startsWith('Book')).join(' | '));
check('vendor facts are dated with their items (A12, A13, A14; B9, B10; C5)', ['A13', 'A12', 'A14', 'B10', 'C5'].every((k) => notes.some((n) => n.includes(k))), notes.join(' | '));
check('the register model says it is the book\'s, and that the vendor function is the authority', notes.some((n) => n.startsWith('What this model leaves out') && n.includes("book's model") && n.includes('authority')));

// ---- predictions before the calculator
let h = await predict('blocks of 32 threads. What is the occupancy', 50);
let t = await h.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
check('prediction "32 threads": the reveal gives 50%, the model\'s, and the 64-thread minimum', t.includes('50%.') && t.includes('That is what you had.') && t.includes('= 64') && pct(occupancy({ ...BOOK, regs: null, shared: null }, { threads: 32, regs: 0, shared: 0 }).byWarps) === 50, t.slice(0, 300));

// ---- the calculator on the book's example
const SLOTS = BOOK; // the calculator's book entry carries registers and shared memory too; with 0 they are not counted
let s = await calc();
let m = want(SLOTS, 256);
check('starts at 256 threads on the book\'s example: 8 blocks, 100%, thread slots bind', agrees(s, m) && s.value === '100' && s.binding === 'threads', show(s, m));
check('with no CUDA or HIP report it says the list has no entry of yours', cudaLike.length ? !(await page.$(C + ' [data-out=no-device]')) : !!(await page.$(C + ' [data-out=no-device]')));
const presets = { t1024: [2, '100'], t512: [4, '100'], t256: [8, '100'], t128: [16, '100'], t64: [32, '100'], t32: [32, '50'], t768: [2, '75'] };
for (const [key, [blocks, occ]] of Object.entries(presets)) {
  await page.$eval(`${C} button[data-preset="${key}"]`, (b) => b.click());
  await pause(80);
  s = await calc();
  m = want(SLOTS, Number(key.slice(1)));
  check(`the book's worked case ${key.slice(1)} threads per block: ${blocks} blocks, ${occ}% (pp. 85–86)`, agrees(s, m) && s.blocks === String(blocks) && s.value === occ, show(s, m));
}
await page.$eval(`${C} button[data-preset="t768"]`, (b) => b.click());
await pause(80);
s = await calc();
check('768: neither limit reached, 512 slots idle, and the verdict says thread slots bind but not cleanly', s.idle === '512' && s.verdict.startsWith('Thread slots bind, but not cleanly.'), s.verdict);
await page.$eval(`${C} button[data-preset="t32"]`, (b) => b.click());
await pause(80);
s = await calc();
check('32: block slots bind, half the thread slots idle', s.binds.join() === 'blocks' && s.idle === '1024' && s.verdict.startsWith('Block slots bind.'), s.verdict);
await page.$eval(`${C} button[data-preset="t48"]`, (b) => b.click());
await pause(80);
s = await calc();
m = want(SLOTS, 48);
check('48 threads: 100% by warps and 75% by threads, the padded lanes named', agrees(s, m) && s.value === '100' && s.byThreads === '75' && s.panel.includes('16 padded lanes'), show(s, m));
if (out) await page.$eval(C, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-calc-48.png' });

await kernel(512, 31);
s = await calc();
m = want(BOOK, 512, 31);
check('512 threads at 31 registers: 4 blocks, 100%; thread slots and registers bind together', agrees(s, m) && s.value === '100' && s.binding === 'threads+regs', show(s, m));
await kernel(512, 33);
s = await calc();
m = want(BOOK, 512, 33);
check('512 threads at 33 registers: the cliff, 3 blocks and 75%, registers bind (p. 86)', agrees(s, m) && s.value === '75' && s.blocks === '3' && s.binding === 'regs', show(s, m));
await kernel(256, 0, 38 * 1024);
s = await calc();
m = want(BOOK, 256, 0, 38 * 1024);
check('256 threads with 38 KB of shared memory per block: shared memory binds at 75% (ch. 5, p. 116)', agrees(s, m) && s.value === '75' && s.binding === 'shared', show(s, m));
await kernel(1024, 100);
s = await calc();
check('1024 threads at 100 registers: no block fits, and the verdict says so', s.value === '0' && s.verdict.includes('does not fit'), s.verdict);
await kernel(2048, 0);
check('a block larger than the device accepts is flagged as an invalid launch', (await text(C + ' [aria-live]')).includes('invalid launch'));

// ---- other limits
await setSelect(`${C} select[aria-label="Limits of one SM"]`, 't1536');
const T1536 = { threadSlots: 1536, blockSlots: 16, blockSlotsDerived: false, regs: 65536, shared: null, width: 32, maxThreadsPerBlock: 1024 };
await kernel(768);
s = await calc();
m = want(T1536, 768);
check('on NVIDIA\'s 1536 / 16 row, 768 threads give 100%, not 75%', agrees(s, m) && s.value === '100', show(s, m));
await kernel(1024);
s = await calc();
m = want(T1536, 1024);
check('and 1024 threads give 66.7%', agrees(s, m) && s.value === '66.7', show(s, m));
check('its smallest filling block is 96 and full occupancy allows 42 registers', (await text(C + ' [data-out=smallest]')) === '96' && (await text(C + ' [data-out=fill-regs]')) === '42');
await setSelect(`${C} select[aria-label="Limits of one SM"]`, 'ex6');
for (const [threads, resident] of [[128, 512], [256, 1024], [512, 1536], [1024, 1024]]) {
  await kernel(threads);
  s = await calc();
  m = want(EX6, threads);
  check(`exercise 6's device at ${threads} threads per block: ${resident} threads resident`, agrees(s, m) && occupancy(EX6, { threads, regs: 0, shared: 0 }).residentThreads === resident, show(s, m));
}
await setSelect(`${C} select[aria-label="Limits of one SM"]`, 'custom');
await setInput(`${C} input[aria-label="Thread slots per SM"]`, 0);
await kernel(256);
s = await calc();
check('custom limits without thread slots: no occupancy is stated', s.value === 'n/a' && s.verdict.startsWith('Occupancy needs two numbers'), JSON.stringify(s));
await setInput(`${C} input[aria-label="Thread slots per SM"]`, 3072);
await setInput(`${C} input[aria-label="Block slots per SM"]`, 8);
s = await calc();
m = want({ threadSlots: 3072, blockSlots: 8, blockSlotsDerived: false, regs: 65536, shared: 228 * 1024, width: 32, maxThreadsPerBlock: null }, 256);
check('custom limits of 3072 threads and 8 blocks at 256 threads per block: block slots bind at 66.7%', agrees(s, m) && s.value === '66.7' && s.binding === 'blocks', show(s, m));
await setSelect(`${C} select[aria-label="Limits of one SM"]`, 'book');

// ---- the cliff
h = await predict('You declare two more local variables', 90);
t = await h.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
check('prediction "the cliff": the reveal gives 75% with the book\'s register counts', t.includes('75%.') && t.includes('63,488') && t.includes('67,584') && t.includes('50,688') && t.includes('performance cliff'), t.slice(0, 400));

const cliff = () => page.evaluate((F) => {
  const o = document.querySelector(F + ' [data-out=cliff-occupancy]');
  return {
    value: Number(o.dataset.value), blocks: Number(o.dataset.blocks), binding: o.dataset.binding,
    steps: [...document.querySelectorAll(F + ' button[data-step]')].map((b) => Number(b.dataset.step)),
    next: document.querySelector(F + ' [data-out=next-step]').textContent.replace(/\s+/g, ' ').trim(),
    regsPlot: { x: Number(document.querySelector(F + ' svg[data-plot=regs]')?.dataset.x), occ: Number(document.querySelector(F + ' svg[data-plot=regs]')?.dataset.occ) },
    threadsPlot: { x: Number(document.querySelector(F + ' svg[data-plot=threads]').dataset.x), occ: Number(document.querySelector(F + ' svg[data-plot=threads]').dataset.occ) },
  };
}, F);
const regSteps = (limits, threads) => steps(scan({ ...limits, shared: null }, { threads, regs: 0, shared: 0 }, 'regs', regValues())).map((x) => x.at);
let c = await cliff();
check('the cliff explorer starts at 512 threads and 31 registers: 100%, next step two registers away, a quarter of the SM', c.value === 100 && c.blocks === 4 && c.next.startsWith('2 more registers per thread') && c.next.includes('a quarter of the SM'), JSON.stringify(c));
check('its steps along registers are the model\'s: 33, 43, 65, 129', JSON.stringify(c.steps) === JSON.stringify(regSteps(BOOK, 512)) && JSON.stringify(c.steps) === '[33,43,65,129]', JSON.stringify(c.steps));
await page.$eval(`${F} button[data-step="33"]`, (b) => b.click());
await pause(100);
c = await cliff();
check('clicking the first step lands on 33 registers: 3 blocks, 75%, both plots marked there', c.value === 75 && c.blocks === 3 && c.binding === 'regs' && c.regsPlot.x === 33 && c.regsPlot.occ === 75 && c.threadsPlot.x === 512 && c.threadsPlot.occ === 75, JSON.stringify(c));
await setInput(`${F} input[aria-label="Threads per block"]`, 1024);
c = await cliff();
check('at 1024 threads per block the steps are 33 and 65, and one costs half the SM', JSON.stringify(c.steps) === JSON.stringify(regSteps(BOOK, 1024)) && JSON.stringify(c.steps) === '[33,65]' && c.value === 50 && c.next.includes('half of the SM'), JSON.stringify(c));
await setInput(`${F} input[aria-label="Registers per thread"]`, 64);
c = await cliff();
check('64 registers at 1024 threads: one block, 50%, the model\'s value', c.value === pct(occupancy(BOOK, { threads: 1024, regs: 64, shared: 0 }).byWarps) && c.value === 50 && c.blocks === 1, JSON.stringify(c));
await setInput(`${F} input[aria-label="Registers per thread"]`, 31);
await setInput(`${F} input[aria-label="Threads per block"]`, 768);
c = await cliff();
check('the block-size bars dip at 768 (75%) where the staircase does not', c.threadsPlot.x === 768 && c.threadsPlot.occ === 75 && c.value === 75, JSON.stringify(c));
await setSelect(`${F} select[aria-label="Limits of one SM for the staircase"]`, 't1536');
await setInput(`${F} input[aria-label="Threads per block"]`, 512);
c = await cliff();
check('on the 1536 / 16 limits the first step at 512 threads moves to 43 registers', c.steps[0] === 43 && c.steps[0] === regSteps(T1536, 512)[0], JSON.stringify(c.steps));
if (out) await page.$eval(F, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-cliff.png' });

// ---- exercises 7 and 8
const ex7Model = EX7_ASSIGNMENTS.map((a) => { const r = assignment(EX7, a.blocks, a.threads); return r.possible ? String(pct(r.occupancy)) : 'no'; });
const mine7 = [...ex7Model.slice(0, 4), '50']; // the last one deliberately wrong
for (let i = 0; i < 5; i++) await setSelect(`${X} select[aria-label="Exercise 7 (${'abcde'[i]})"]`, mine7[i]);
await page.evaluate((X) => [...document.querySelectorAll(X + ' button')].find((b) => b.textContent.includes('Check exercise 7')).click(), X);
await pause(120);
const models7 = await page.$$eval(`${X} [data-ex="7"] td[data-model]`, (tds) => tds.map((td) => td.dataset.model));
check('exercise 7: the checker marks 4 of 5 and shows the model\'s 50, 50, 50, 100, 100', (await text(`${X} [data-out=ex7-score]`)) === '4 of 5 right.' && JSON.stringify(models7) === JSON.stringify(ex7Model) && JSON.stringify(ex7Model) === '["50","50","50","100","100"]', JSON.stringify(models7));
const ex8Model = EX8_KERNELS.map((k) => { const o = occupancy(EX8, k); return o.byWarps === 1 ? 'full' : o.binding[0]; });
check('exercise 8 shows no bars before it is answered', (await page.$$(`${X} [data-ex8-bars]`)).length === 0);
for (let i = 0; i < 3; i++) await setSelect(`${X} select[aria-label="Exercise 8 (${'abc'[i]})"]`, ex8Model[i]);
await page.evaluate((X) => [...document.querySelectorAll(X + ' button')].find((b) => b.textContent.includes('Check exercise 8')).click(), X);
await pause(120);
const bars = await page.$$eval(`${X} [data-ex8-bars]`, (ps) => ps.map((p) => p.textContent.replace(/\s+/g, ' ')));
check('exercise 8: full, block slots, registers, 3 of 3, and the bars appear with 87.5% for (c)', (await text(`${X} [data-out=ex8-score]`)) === '3 of 3 right.' && JSON.stringify(ex8Model) === '["full","blocks","regs"]' && bars.length === 3 && bars[2].includes('7 blocks, 87.5%') && bars[1].includes('32 blocks, 50%'), JSON.stringify(bars.map((b) => b.slice(0, 80))));
if (out) await page.$eval(X, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-exercises.png' });

// ---- questions
let a = await answerNumeric('launched with 768 threads per block', 100);
check('q-768: 100% is accepted', a.includes('Correct.'));
a = await answerChoice('Which block size puts the most threads on', '512 threads per block');
check('q-ex6: 512 is right', a.picked && a.text.includes('Correct.'));
a = await answerChoice('Blocks of 48 threads on the book', '100% by warps and 75% by threads');
check('q-twodefs: 100% by warps, 75% by threads', a.picked && a.text.includes('Correct.'));
a = await answerNumeric('needs 64 registers per thread', 50);
check('q-regs64: 50% is accepted', a.includes('Correct.'));
a = await answerNumeric('uses 34 registers per thread', 87.5);
check('q-ex8c: 87.5% is accepted', a.includes('Correct.'), a.slice(0, 200));
a = await answerChoice('and gets some number N', 'number of wave slots');
check('q-hipslots: the derived wave-slot answer is right', a.picked && a.text.includes('Correct.'));

// ---- the code shown and the tools
const codes = await page.$$eval('figure.code', (fs) => fs.map((f) => ({ code: f.querySelector('pre').textContent, status: f.querySelector('.status')?.textContent ?? '' })));
check("lab 4.4's occupancy call is shown from the lab file, labelled front-end-checked and unverified on hardware", codes.some((c) => c.code.includes('cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, gatherSumKernel, threadsPerBlock, 0)') && c.status.includes('front-end-checked') && c.status.includes('unverified')), JSON.stringify(codes.map((c) => c.status)));
check('the tools table lists seven tools', (await page.$$('table[data-tools] tbody tr')).length === 7);

// ---- the lab 4.4 prediction, revealed from this machine's benchmark when there is one
h = await predict('how many times as long will the slowest block size', 2);
const meas = await h.evaluate((e) => [...e.querySelectorAll('[data-measured]')].map((d) => ({ v: d.dataset.measured, b: d.dataset.backend ?? null, sizes: d.querySelectorAll('tr[data-size]').length, models: [...d.querySelectorAll('td[data-model]')].map((x) => x.dataset.model) })));
const rowsFor = (b) => {
  const run = reports[b];
  const pick = (variant) => {
    const out = [];
    let name = null;
    for (const r of run.results.filter((x) => x.lab === 'ch04/lab4_blocksize' && x.variant === variant).flatMap((x) => x.bench)) {
      if (typeof r.threads_per_block !== 'number' || !(r.seconds > 0)) continue;
      if (name == null) name = r.name;
      if (r.name !== name) continue;
      if (!out.some((x) => x.size === r.threads_per_block)) out.push({ size: r.threads_per_block, seconds: r.seconds });
    }
    return out;
  };
  const workPass = run.work_state?.['ch04/lab4_blocksize'] !== 'untouched' && run.results.some((x) => x.lab === 'ch04/lab4_blocksize' && x.variant === 'work' && x.tests.length && x.built && x.test_rc === 0 && x.tests.every((t) => t.pass));
  const mine = workPass ? pick('work') : [];
  return mine.length >= 2 ? mine : pick('solution');
};
const measured = Object.keys(reports).filter((b) => b !== 'emu').map((b) => ({ b, rows: rowsFor(b) })).filter((x) => x.rows.length >= 2);
const fmt = (v) => Number(v.toPrecision(3)).toString();
if (measured.length) {
  const okAll = measured.every(({ b, rows }) => {
    const got = meas.find((x) => x.b === b);
    const ratio = Math.max(...rows.map((r) => r.seconds)) / Math.min(...rows.map((r) => r.seconds));
    const lim = limitsFromReport(reports[b].device, b);
    return got && got.v === fmt(ratio) && got.sizes === rows.length && (lim ? true : got.models.every((x) => x === 'none'));
  });
  check(`the reveal is computed from lab 4.4's rows in this machine's reports (${measured.map((x) => x.b).join(', ')}), with no occupancy column where no limits exist`, okAll, JSON.stringify(meas));
} else {
  check('with no lab 4.4 benchmark yet, the reveal says how to run it', meas.length === 1 && meas[0].v === 'none', JSON.stringify(meas));
}

// ---- narrow screen
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((C, F) => {
  const inside = (sel) => {
    const b = document.querySelector(sel).getBoundingClientRect();
    return [...document.querySelectorAll(sel + ' select')].every((s) => s.getBoundingClientRect().right <= b.right + 1);
  };
  return { page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, calcSelect: inside(C), cliffSelect: inside(F) };
}, C, F);
check('at a narrow width nothing overflows, and the device lists fit their benches', narrow.page && narrow.calcSelect && narrow.cliffSelect, JSON.stringify(narrow));
if (out) await page.$eval(C, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-narrow.png' });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
