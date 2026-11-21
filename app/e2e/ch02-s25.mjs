// Browser check for section 2.5, "Kernel functions and threading": the narrated
// animation, the kernel listing, the line-by-line stepper, the thread index explorer
// with its locked mappings, the prediction, the questions and the qualifier table.
// Everything asserted is computed from the model the page runs on (src/sim/grid1d.ts)
// or read from the lab files and narration scripts on disk.
// Usage: node e2e/ch02-s25.mjs [screenshot-prefix]
//   E2E_DARK=1, E2E_LIGHT=1 and E2E_WIDTH=<px> run the same checks in either theme or at another width.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto, BASE } from './browser.mjs';
import { tabStatus as sharedTabStatus } from './labstatus.mjs';
import { counts, coverage, loopView, makeData, mapping, snapBlock, span, stepKernel, working } from '../src/sim/grid1d.ts';
import { CALLERS, EXECUTOR_TEXT, functionSource, legalCall } from '../src/sim/toolchain.ts';

const out = process.argv[2];
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const { browser, page, errors } = await open({ width: Number(process.env.E2E_WIDTH || 1440), height: 1000, dark: process.env.E2E_DARK === '1' });
// E2E_LIGHT=1 forces the light theme even when the operating system is set to dark.
if (process.env.E2E_LIGHT === '1') await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }]);
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
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent);
const exists = async (sel) => (await page.$(sel)) !== null;
const num = (v) => v.toLocaleString('en-US');
const shot = async (name, sel) => {
  if (!out) return;
  await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};
const clickText = async (scope, label) => {
  const done = await page.evaluate(
    (scope, label) => {
      const root = [...document.querySelectorAll('section.bench')].find((b) => b.getAttribute('aria-label') === scope);
      const btn = [...root.querySelectorAll('button')].find((b) => b.textContent.trim() === label);
      if (!btn || btn.disabled) return false;
      btn.click();
      return true;
    },
    scope,
    label,
  );
  await pause(80);
  return done;
};
const click = async (sel) => {
  await page.$eval(sel, (e) => e.dispatchEvent(new MouseEvent('click', { bubbles: true })));
  await pause(80);
};
/** Set an input, range or select the way a person would, so React sees the change. */
const setValue = async (sel, value) => {
  await page.$eval(
    sel,
    (e, value) => {
      const proto = e.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(proto, 'value').set.call(e, value);
      e.dispatchEvent(new Event(e.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true }));
    },
    String(value),
  );
  await pause(80);
};
const question = (part) => page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(part)), part);
const qText = (q) => q.evaluate((e) => e.textContent);
const typeAnswer = async (q, v) => {
  const input = await q.$('input[type=number]');
  await input.evaluate((e) => e.scrollIntoView({ block: 'center' }));
  await input.focus();
  await input.evaluate((e) => e.select());
  await input.type(String(v));
  await (await q.$('button.primary')).click();
  await pause(150);
};
const pick = async (q, label) => {
  const found = await q.evaluate((e, label) => {
    const l = [...e.querySelectorAll('.opts label')].find((x) => x.textContent.trim().startsWith(label));
    if (!l) return false;
    l.querySelector('input').click();
    return true;
  }, label);
  await (await q.$('button.primary')).click();
  await pause(150);
  return found;
};

await goto(page, 'ch02/2.5');
const SC = 'section.scene';
const STEP = 'Run the kernel on every thread';
const EXPL = 'Thread index explorer';
await page.waitForSelector(bench(STEP));
await page.waitForFunction(() => document.querySelectorAll('.code pre').length >= 3, { timeout: 15000 });
await pause(300);

// ---------------------------------------------------------------- the narrated animation
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').trim();
const scripts = [1, 2, 3, 4, 5, 6].map((k) => strip(fs.readFileSync(path.join(repo, 'content/ch02/narration', `loop-to-grid-${k}.txt`), 'utf8')));
const manifest = await fetch(BASE + '/content/ch02/narration/manifest.json').then((r) => (r.ok ? r.json() : null)).catch(() => null);
const press = async (label) => {
  await page.evaluate((sel, label) => [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim().includes(label)).click(), SC, label);
  await pause(150);
};
const seek = async (v) => {
  await page.$eval(SC + ' input[type=range]', (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(150);
};
const drawn = () => page.$$eval(SC + ' .stage svg text', (ts) => ts.map((x) => x.textContent.trim()).filter(Boolean));
const has = (list, s) => list.some((x) => x.includes(s));
const box = (word) => page.evaluateHandle((sel, word) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes(word)).querySelector('input'), SC, word);

await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' }));
await pause(200);
check('the animation is titled "Where the loop went" and has six parts', (await page.$eval(SC, (e) => e.getAttribute('aria-label'))) === 'Where the loop went' && (await page.$$(SC + ' .seg button')).length === 6);
check('six scripts, each 35 to 60 words', scripts.every((s) => { const w = s.split(/\s+/).length; return w >= 35 && w <= 60; }), scripts.map((s) => s.split(/\s+/).length).join(' '));
check('it has a captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
await (await box('captions')).click();
await pause(120);
const LABELS = ['The loop', 'Trips become threads', 'Blocks', 'Two built-in numbers', 'The guard', 'On Metal'];
const frames = [];
const captionsOk = [];
for (let k = 0; k < 6; k++) {
  await press(`${k + 1}. ${LABELS[k]}`);
  captionsOk.push((await page.$eval(SC + ' .caption', (e) => e.textContent.trim())) === scripts[k]);
  await seek(1000);
  frames.push(await drawn());
  await shot(`scene-${k + 1}`, SC);
}
const N = 10, B = 4;
const sc = coverage({ n: N, blockDim: B });
const trips = loopView(N);
const m1 = mapping('one');
check('each part shows its own script as the caption', captionsOk.every(Boolean), JSON.stringify(captionsOk));
check('part 1: the loop, then 10 trips by one thread, ending on i = 9',
  has(frames[0], 'for (int i = 0; i < n; ++i)') && has(frames[0], `${N} trips, one after another, all by one thread`) && has(frames[0], `i = ${N - 1}`), frames[0].join('|'));
check('part 1 ends with C holding the loop\'s sums', trips.every((t) => frames[0].includes(String(t.c))));
check('part 2: every trip is a thread with its own i', has(frames[1], `${N} of ${N} trips are now threads, each with its own i`) && trips.every((t) => frames[1].includes(`i = ${t.i}`)), frames[1].join('|'));
check(`part 3: ${sc.blocks} blocks of ${B} is ${sc.launched} threads, ${sc.idle} more than the trips`,
  has(frames[2], `${sc.blocks} blocks of ${B} is ${sc.launched} threads: ${sc.idle} more than the loop had trips`) && [0, 1, 2].every((b) => frames[2].includes(`block ${b}`)), frames[2].join('|'));
check('part 4: the arithmetic worked for three threads gives 1, 6 and 11',
  [[0, 1], [1, 2], [2, 3]].every(([b, t]) => has(frames[3], `block ${b}, thread ${t}:  ${working(b, t, B)}`)) && [1, 6, 11].every((v, k) => v === m1.first(...[[0, 1], [1, 2], [2, 3]][k], B)) &&
  has(frames[3], 'the same arithmetic in every thread, a different i in each') && has(frames[3], 'b2 t3'), frames[3].join('|'));
check(`part 5: the test moves into the kernel and threads ${sc.idleThreads.map((t) => t.i).join(' and ')} fail it`,
  has(frames[4], `threads ${sc.idleThreads.map((t) => t.i).join(' and ')} fail the guard and do nothing`) && sc.idleThreads.every((t) => has(frames[4], `thread i = ${t.i}:  ${t.i} < ${N} is false`)) &&
  has(frames[4], 'int i = blockIdx.x * blockDim.x + threadIdx.x;') && has(frames[4], 'if (i < n)'), frames[4].join('|'));
const exact = counts({ n: N, blockDim: N });
check('part 6: on Metal the index is an argument and a grid of exactly 10 threads has none spare',
  has(frames[5], 'uint i [[thread_position_in_grid]]') && has(frames[5], `a grid of exactly ${exact.launched} threads`) && has(frames[5], `${exact.launched} threads for ${N} elements: ${exact.idle} spare, nothing to guard against`), frames[5].join('|'));
if (manifest) check('the manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
else check('with no manifest the player says it runs on estimated timings', (await page.$eval(SC + ' footer', (e) => e.textContent)).includes('No narration manifest found'));
const soundBox = await box('sound');
if (await soundBox.evaluate((e) => e.checked)) await soundBox.click();
await press('1. The loop');
await press('Play');
await pause(1500);
const moved = await page.$eval(SC + ' input[type=range]', (e) => Number(e.value));
await press('Pause');
check('pressing Play moves the animation', moved > 20, String(moved));
await (await box('captions')).click();

// ---------------------------------------------------------------- the reader's devices
const reports = await (await fetch(BASE + '/api/reports')).json();
const limits = await text('[data-out="your-limits"]');
const metalWidth = reports.metal?.device?.['maxThreadsPerThreadgroup.width'];
if (metalWidth) check('the block-size paragraph quotes this machine\'s Metal report, not a fixed number', limits.includes(`at most ${num(Number(metalWidth))} threads along one dimension of a threadgroup`) && limits.includes(String(reports.metal.device.name)), limits);
else check('without a device report the page says how to get one', limits.includes('python3 labs/pmpp.py doctor'), limits);
check('the emulator\'s synthetic figures are not presented as a device', !limits.includes('CPU emulator'), limits);

// ---------------------------------------------------------------- the kernel listing
const lab = (rel) => fs.readFileSync(path.join(repo, 'labs', rel), 'utf8');
const kCu = functionSource(lab('ch02/lab1_vecadd/solution/kernel.cu'), 'vecAddKernel');
const kMetal = functionSource(lab('ch02/lab1_vecadd/solution/kernel.metal'), 'vec_add');
const passed = (backend) => {
  const rs = (reports[backend]?.results ?? []).filter((x) => x.lab === 'ch02/lab1_vecadd' && x.variant === 'solution' && x.tests.length);
  return rs.length > 0 && rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass));
};
// LabKit's solutionTabStatus, from the shared helper, with this machine's runs and its CUDA
// front-end check (GET /api/frontend-check).
const frontend = await (await fetch(BASE + '/api/frontend-check')).json();
const tabStatus = (labId, file) => sharedTabStatus(reports, labId, file, frontend);
const LISTING = await page.evaluateHandle(() => [...document.querySelectorAll('figure.code')].find((f) => [...f.querySelectorAll('.tabs button')].map((b) => b.textContent).join('|') === 'CUDA|Metal'));
const listingLines = () => LISTING.evaluate((f) => [...f.querySelectorAll('pre .ln')].map((l) => l.textContent.replace(/\n$/, '')));
const listingStatus = () => LISTING.evaluate((f) => f.querySelector('.status')?.textContent ?? '');
check('the listing has a CUDA tab and a Metal tab', !!(await LISTING.evaluate((f) => !!f)));
check('the CUDA tab is vecAddKernel from the lab 2.1 solution, line for line', JSON.stringify(await listingLines()) === JSON.stringify(kCu.text.split('\n')));
const wantCu = tabStatus('ch02/lab1_vecadd', 'kernel.cu');
check('its status follows this machine\'s recorded runs, in the fixed vocabulary', (await listingStatus()) === wantCu, `${await listingStatus()} / ${wantCu}`);
await LISTING.evaluate((f) => [...f.querySelectorAll('.tabs button')].find((b) => b.textContent === 'Metal').click());
await pause(100);
check('the Metal tab is vec_add from kernel.metal, line for line', JSON.stringify(await listingLines()) === JSON.stringify(kMetal.text.split('\n')));
check('the Metal status is "verified on Metal" only with a passing run here', (await listingStatus()) === tabStatus('ch02/lab1_vecadd', 'kernel.metal'), await listingStatus());
const numbering = () => LISTING.evaluate((f) => {
  const n = [...f.querySelectorAll('pre .ln')].map((l) => Number(l.dataset.n));
  return { first: n[0], last: n.at(-1), src: f.querySelector('[data-code-source]')?.textContent ?? '' };
});
const metalNums = await numbering();
check('the Metal tab is numbered as kernel.metal is, and names the file', metalNums.first === kMetal.firstLine && metalNums.last === kMetal.lastLine && metalNums.src === 'labs/ch02/lab1_vecadd/solution/kernel.metal', JSON.stringify(metalNums));
await LISTING.evaluate((f) => [...f.querySelectorAll('.tabs button')].find((b) => b.textContent === 'CUDA').click());
await pause(100);
const cuNums = await numbering();
check('the CUDA tab is numbered as kernel.cu is, and names the file', cuNums.first === kCu.firstLine && cuNums.last === kCu.lastLine && cuNums.src === 'labs/ch02/lab1_vecadd/solution/kernel.cu', JSON.stringify(cuNums));

// ---------------------------------------------------------------- q-own
let q = await question('Which element does thread 5 of block 3 own?');
await typeAnswer(q, 768);
check('q-own: 768 (the start of block 3) is marked wrong with the answer given', (await qText(q)).includes(`The answer is ${m1.first(3, 5, 256)}.`));
await typeAnswer(q, m1.first(3, 5, 256));
check('q-own: 773 is accepted and the working is shown', (await qText(q)).includes('Correct.') && (await qText(q)).includes(working(3, 5, 256)));

// ---------------------------------------------------------------- A: the stepper
const S = bench(STEP);
const o = (name) => page.$eval(`${S} [data-out="${name}"]`, (e) => e.textContent.trim());
const threadStates = () => page.$$eval(`${S} [data-thread]`, (gs) => gs.map((g) => ({ g: Number(g.dataset.thread), state: g.dataset.state, text: g.textContent.trim() })));
const cell = (name, i) => page.$eval(`${S} [data-cell="${name}:${i}"]`, (g) => ({ text: g.textContent.trim(), touched: g.dataset.touched })).catch(() => null);
const hl = () => page.$$eval(`${S} .code .ln.hl`, (ls) => ls.map((l) => l.textContent));
const fact = (key) => page.$eval(`${S} [data-fact="${key}"]`, (e) => e.textContent.trim()).catch(() => null);
const counters = async () => [await o('blocks'), await o('launched'), await o('working'), await o('idle')];
const cfg = { n: 10, blockDim: 4 };
const c10 = counts(cfg);
check('the stepper opens at n = 10, blocks of 4, guard on, before line 1', (await o('n')) === '10' && (await o('B')) === '4' && (await page.$eval(S + ' [data-stepper]', (e) => e.dataset.line)) === '0');
check(`its counters: ${c10.blocks} blocks, ${c10.launched} threads, ${c10.working} with an element, ${c10.idle} without`, JSON.stringify(await counters()) === JSON.stringify([c10.blocks, c10.launched, c10.working, c10.idle].map(String)));
let ts = await threadStates();
check('before line 1 no thread has an i', ts.length === c10.launched && ts.every((t) => t.state === 'new'), JSON.stringify(ts.slice(0, 3)));
await clickText(STEP, 'Step');
let st = stepKernel(cfg, 1);
ts = await threadStates();
let said = await o('said');
check('line 1: every thread shows its own i, as the model computes it', ts.every((t) => t.state === 'has-i' && t.text.endsWith(String(st.threads[t.g].i))), JSON.stringify(ts.map((t) => t.text)));
check('line 1: the selected thread\'s arithmetic is spelled out and the copies are counted', said.includes(`For thread 2 of block 1: ${working(1, 2, 4)}`) && said.includes(`There are ${c10.launched} threads and ${c10.launched} separate copies of i`), said);
check('line 1: the source line marked in the listing is the index line', (await hl()).length === 1 && (await hl())[0].includes('int i ='), JSON.stringify(await hl()));
check('the stepper names the statement rather than numbering it, so it cannot disagree with the listing’s numbers', said.startsWith('The index line.') && (await page.$eval(S + ' [aria-live]', (e) => e.textContent.trim())) === 'after the index line, 1 of 3', said);
await clickText(STEP, 'Step');
st = stepKernel(cfg, 2);
ts = await threadStates();
said = await o('said');
const offI = st.threads.filter((t) => t.pass === false).map((t) => t.i);
check(`line 2: threads with i = ${span(offI)} fail the guard and grey out`, ts.every((t) => t.state === (st.threads[t.g].pass ? 'has-i' : 'off')) && said.includes(`${c10.working} threads pass`) && said.includes(`with i = ${span(offI)} fail`), said);
check('line 2: the marked line is the guard', (await hl())[0]?.includes('if (i < n)'));
check('line 2: the narration and the counter call it the guard', said.startsWith('The guard.') && (await page.$eval(S + ' [aria-live]', (e) => e.textContent.trim())) === 'after the guard, 2 of 3', said);
await clickText(STEP, 'Step');
st = stepKernel(cfg, 3);
const data = makeData(10);
const cOk = [];
for (let i = 0; i < 10; i++) { const c = await cell('C', i); cOk.push(c && c.text === String(st.C[i]) && c.touched === 'true'); }
check('line 3: every element of C holds A[i] + B[i], as the model stores it', cOk.every(Boolean), JSON.stringify(cOk));
check('line 3: no index past the end was touched', (await cell('A', 10)) === null && !(await exists(`${S} [data-out="oob"]`)));
check('line 3: the marked line is the addition', (await hl())[0]?.includes('C[i] = A[i] + B[i];'));
await shot('stepper-line3', S);
// A thread and an element, picked by clicking.
await click(`${S} [data-thread="9"]`);
const t9 = st.threads[9];
check('clicking thread 9 shows its built-ins, its private i and what it did',
  (await text(`${S} [data-out="detail"]`)).includes('Thread 1 of block 2') && (await fact('blockIdx.x')) === '2' && (await fact('threadIdx.x')) === '1' && (await fact('i')) === working(2, 1, 4) &&
  (await fact('addition')) === `loads A[9] = ${data.A[9]} and B[9] = ${data.B[9]}, stores ${t9.store.value} in C[9]`, await text(`${S} [data-out="detail"]`));
await click(`${S} [data-cell="C:3"]`);
check('clicking C[3] names the one thread that wrote it', (await o('writer')).includes(`C[3] holds ${st.C[3]}. It was written by thread 3 of block 0`), await o('writer'));
// The guard off.
await page.$eval(S, (root) => [...root.querySelectorAll('label.check')].find((l) => l.textContent.includes('guard')).querySelector('input').click());
await pause(120);
const ng = stepKernel({ ...cfg, guard: false }, 3);
ts = await threadStates();
said = await o('said');
check('guard off at line 3: the surplus threads touch indices 10 and 11, outside the arrays', JSON.stringify(ng.outOfRange) === '[10,11]' && (await o('oob')) === 'Indices touched outside the arrays: 10, 11.' &&
  said.includes('The addition, with no guard') && said.includes(`${3 * ng.outOfRange.length} accesses outside the arrays`), said);
check('guard off: those two threads are drawn as stray, the others as done', ts.every((t) => t.state === (t.g >= 10 ? 'stray' : 'done')));
check('guard off: cells past the end of A, B and C are drawn', (await cell('A', 11))?.touched === 'true' && (await cell('C', 10))?.touched === 'true');
check('guard off: the listing says to read it without the guard', (await o('guard-off')).includes('as if line'));
await shot('stepper-noguard', S);
await clickText(STEP, 'Back');
check('Back returns to line 2, where without a guard nothing is tested', (await o('said')).includes('The guard has been removed') && !(await exists(`${S} [data-out="oob"]`)));
await setValue(`${S} input[aria-label="Number of elements"]`, 12);
await clickText(STEP, 'Step');
check('n = 12 with blocks of 4 and no guard: nothing outside, and the lesson says why', (await o('said')).includes('the missing guard did no harm') && !(await exists(`${S} [data-out="oob"]`)) && (await o('idle')) === '0');
await setValue(`${S} input[aria-label="Threads per block"]`, 8);
await setValue(`${S} input[aria-label="Number of elements"]`, 21);
const c21 = counts({ n: 21, blockDim: 8 });
const ng21 = stepKernel({ n: 21, blockDim: 8, guard: false }, 3);
check(`n = 21, blocks of 8, no guard: ${c21.blocks} blocks, ${c21.launched} threads, indices ${span(ng21.outOfRange)} touched`,
  JSON.stringify(await counters()) === JSON.stringify([c21.blocks, c21.launched, c21.working, c21.idle].map(String)) && (await o('oob')) === `Indices touched outside the arrays: ${ng21.outOfRange.join(', ')}.`, await o('said'));
await page.$eval(S, (root) => [...root.querySelectorAll('label.check')].find((l) => l.textContent.includes('guard')).querySelector('input').click());
await pause(100);
await clickText(STEP, 'Restart');
check('Restart goes back before line 1', (await page.$eval(S + ' [data-stepper]', (e) => e.dataset.line)) === '0' && (await threadStates()).every((t) => t.state === 'new'));

// ---------------------------------------------------------------- q-private
q = await question('How many copies of');
check('q-private: "one per block" is refused with its reason', (await pick(q, 'One per block')) && (await qText(q)).includes('Not quite.') && (await qText(q)).includes('chapter 5'));
await pick(q, '10,000: one per thread');
check('q-private: one per thread is accepted', (await qText(q)).includes('Correct.'));

// ---------------------------------------------------------------- the prediction
const p37 = counts({ n: 100, blockDim: 32 });
const P = await question('How many threads fail the guard?');
await (await P.$('input[type=number]')).type(String(p37.launched));
await (await P.$('button.primary')).click();
await pause(150);
let t = await qText(P);
check(`p-idle: the reveal gives ${p37.blocks} blocks, ${p37.launched} threads and ${p37.idle} idle, from the model`,
  t.includes(`launches ${p37.blocks}: ${p37.blocks} × 32 = ${p37.launched} threads`) && t.includes(`The other ${p37.idle}, all in block ${p37.blocks - 1}`), t);
check('p-idle: a prediction of 128 is told that is the number launched', t.includes(`${p37.launched} is the number of threads launched`), t);

// ---------------------------------------------------------------- B: the explorer
const E = bench(EXPL);
const eo = (name) => page.$eval(`${E} [data-out="${name}"]`, (e) => e.textContent.trim());
const ecount = async () => [await eo('blocks'), await eo('launched'), await eo('working'), await eo('idle')];
const wantCount = (c) => [c.blocks, c.launched, c.working, c.idle].map(num);
const cells = () => page.$$eval(`${E} [data-strip] [data-e]`, (gs) => gs.map((g) => ({ e: Number(g.dataset.e), state: g.dataset.state, block: Number(g.dataset.block), thread: Number(g.dataset.thread), mine: g.dataset.mine === 'true' })));
const stripMatches = async (n, B, kind) => {
  const m = mapping(kind);
  const cs = await cells();
  const bad = cs.filter((c) => {
    const ow = m.owner(c.e, B);
    const state = c.e < n ? 'in' : m.elements(ow.block, ow.thread, B).some((x) => x < n) ? 'tail' : 'idle';
    return ow.block !== c.block || ow.thread !== c.thread || state !== c.state;
  });
  return { ok: cs.length > 0 && bad.length === 0, detail: JSON.stringify(bad.slice(0, 3)) + ` of ${cs.length}` };
};
await page.$eval(E, (e) => e.scrollIntoView({ block: 'start' }));
check('the explorer opens on the book\'s first case: n = 100, blocks of 32', JSON.stringify(await ecount()) === JSON.stringify(wantCount(p37)), JSON.stringify(await ecount()));
check('the thread in view is the owner of the last element', (await eo('thread')).startsWith(`Thread 3 of block 3: i = ${working(3, 3, 32)}. It owns element 99.`), await eo('thread'));
let sm = await stripMatches(100, 32, 'one');
check('every drawn cell is coloured by the block that owns it, with the 28 idle positions dashed', sm.ok && (await cells()).filter((c) => c.state === 'idle').length === 28, sm.detail);
await shot('explorer-100', E);
await clickText(EXPL, 'n = 1,000, blocks of 256 (pp. 38–39)');
const p1000 = counts({ n: 1000, blockDim: 256 });
check('the second case: 4 blocks, 1,024 threads, 1,000 working, 24 idle', JSON.stringify(await ecount()) === JSON.stringify(wantCount(p1000)) && p1000.idle === 24, JSON.stringify(await ecount()));
sm = await stripMatches(1000, 256, 'one');
check('the strip still matches the model at blocks of 256', sm.ok, sm.detail);
await page.hover(`${E} [data-e="1000"]`);
await pause(80);
check('pointing at index 1,000 names its thread and says it is past the end', (await eo('hover')) === `Index 1,000: block 3, thread 232. Past the end.`, await eo('hover'));
await setValue(`${E} [data-in="block"]`, 2);
await setValue(`${E} [data-in="thread"]`, 5);
check('typing block 2, thread 5 shows i = 2 × 256 + 5 = 517 and marks that cell', (await eo('thread')).startsWith(`Thread 5 of block 2: i = ${working(2, 5, 256)}. It owns element 517.`) && (await cells()).some((c) => c.e === 517 && c.mine), await eo('thread'));
await click(`${E} [data-e="600"]`);
const o600 = mapping('one').owner(600, 256);
check('clicking a cell picks its owner', (await eo('thread')).startsWith(`Thread ${o600.thread} of block ${o600.block}: i = ${working(o600.block, o600.thread, 256)}. It owns element 600.`), await eo('thread'));
// Free n and block size, with and without the snap to multiples of 32.
await setValue(`${E} [data-in="n"]`, 777);
await setValue(`${E} [data-in="B"]`, 100);
const snapped = snapBlock(100);
check(`with the snap on, a block size of 100 becomes ${snapped}`, (await page.$eval(`${E} [data-in="B"]`, (e) => e.value)) === String(snapped) && JSON.stringify(await ecount()) === JSON.stringify(wantCount(counts({ n: 777, blockDim: snapped }))), JSON.stringify(await ecount()));
await page.$eval(`${E} [data-in="snap"]`, (e) => e.click());
await pause(80);
await setValue(`${E} [data-in="B"]`, 100);
const c777 = counts({ n: 777, blockDim: 100 });
check(`with the snap off, n = 777 and blocks of 100: ${c777.blocks} blocks, ${c777.idle} idle, fewer than one block`, JSON.stringify(await ecount()) === JSON.stringify(wantCount(c777)) && c777.idle < 100, JSON.stringify(await ecount()));
sm = await stripMatches(777, 100, 'one');
check('the strip matches the model at a block size that is not a multiple of 32', sm.ok, sm.detail);
// The overview strip moves the window.
await page.$eval(`${E} [data-overview]`, (svg) => {
  const r = svg.getBoundingClientRect();
  svg.dispatchEvent(new MouseEvent('click', { bubbles: true, clientX: r.left + 2, clientY: r.top + r.height / 2 }));
});
await pause(100);
check('clicking the start of the overview shows the first block in full', (await eo('window')).startsWith('Drawn in full: positions 0 to'), await eo('window'));

// The two other mappings are locked until their questions are answered.
const kindBtn = (k) => `${E} [data-kind-choice="${k}"]`;
check('the two-element mappings start locked', (await page.$eval(kindBtn('two-adjacent'), (b) => b.disabled)) && (await page.$eval(kindBtn('two-sections'), (b) => b.disabled)) && (await exists(`${E} [data-out="locked"]`)));
q = await question('two adjacent');
check('q-two-adjacent: the shifted index is refused with its reason', (await pick(q, 'i = blockIdx.x * blockDim.x + threadIdx.x + 2;')) && (await qText(q)).includes('Not quite.') && (await qText(q)).includes('overlap'));
await pick(q, 'i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;');
t = await qText(q);
check('q-two-adjacent: accepted; the explanation gives the tail guard and the MLX loop', t.includes('Correct.') && t.includes('that is 1 thread') && t.includes('index + i < size'), t);
check('answering unlocks the adjacent mapping only', !(await page.$eval(kindBtn('two-adjacent'), (b) => b.disabled)) && (await page.$eval(kindBtn('two-sections'), (b) => b.disabled)));
await click(kindBtn('two-adjacent'));
const adj = counts({ n: 777, blockDim: 100, mapping: 'two-adjacent' });
const fm = await eo('formula');
check('two adjacent: the formula, the second element and one guard per access are shown', fm.includes('i = (blockIdx.x * blockDim.x + threadIdx.x) * 2') && fm.includes('i + 1') && fm.includes('i < n') && fm.includes('i + 1 < n'), fm);
check(`two adjacent, n = 777: ${adj.blocks} blocks, ${adj.partial} thread with its second element past the end`, JSON.stringify(await ecount()) === JSON.stringify(wantCount(adj)) && (await eo('partial')) === String(adj.partial) && adj.partial === 1, JSON.stringify([await ecount(), await eo('partial')]));
check('the thread in view is that one, and the lesson names its second test', (await eo('thread')).includes('Element 777 is past the end') && (await eo('thread')).includes('i + 1 < n'), await eo('thread'));
sm = await stripMatches(777, 100, 'two-adjacent');
check('the strip recolours by the new owners, with the tail position marked', sm.ok && (await cells()).some((c) => c.e === 777 && c.state === 'tail'), sm.detail);
await shot('explorer-adjacent', E);
q = await question('in two sections');
await pick(q, 'i = blockIdx.x * blockDim.x * 2 + threadIdx.x;');
t = await qText(q);
check('q-two-sections: accepted, with the note that the answers are ours', t.includes('Correct.') && t.includes('Both answers here are ours'), t);
await click(kindBtn('two-sections'));
const sec = counts({ n: 777, blockDim: 100, mapping: 'two-sections' });
check(`two sections, n = 777: ${sec.blocks} blocks, ${sec.partial} threads with a second element past the end`, JSON.stringify(await ecount()) === JSON.stringify(wantCount(sec)) && (await eo('partial')) === String(sec.partial), JSON.stringify([await ecount(), await eo('partial')]));
check('two sections: the second element is blockDim.x further on', (await eo('formula')).includes('i + blockDim.x') && (await eo('formula')).includes('blockIdx.x * blockDim.x * 2 + threadIdx.x'));
sm = await stripMatches(777, 100, 'two-sections');
check('the strip matches the sectioned owners', sm.ok, sm.detail);
await shot('explorer-sections', E);
await click(kindBtn('one'));
check('back to one element per thread', (await page.$eval(`${E} [data-explorer]`, (e) => e.dataset.kind)) === 'one' && !(await exists(`${E} [data-out="partial"]`)));

// ---------------------------------------------------------------- the other questions
q = await question('maps the thread and block indices to the data index');
check('q-index: blockIdx.x + threadIdx.x is refused because blocks overlap', (await pick(q, 'i = blockIdx.x + threadIdx.x;')) && (await qText(q)).includes('overlap'));
await pick(q, 'i = blockIdx.x * blockDim.x + threadIdx.x;');
check('q-index: the book\'s formula is accepted', (await qText(q)).includes('Correct.'));
const exLabels = await page.$$eval('.q .kind', (ks) => ks.map((k) => k.textContent.trim()));
check('the book’s exercises 1 to 3 are labelled as such', ['Book exercise 1', 'Book exercise 2', 'Book exercise 3'].every((l) => exLabels.includes(l)), exLabels.join(' | '));
const ngq = coverage({ n: 1000, blockDim: 256, guard: false });
q = await question('someone deletes');
await pick(q, 'The launch fails with an error code,');
check('q-noguard: "the launch fails" is refused', (await qText(q)).includes('Not quite.'));
await pick(q, `${ngq.idle} threads read and write elements ${span(ngq.outOfRange)}`);
t = await qText(q);
check(`q-noguard: ${ngq.idle} threads, elements ${span(ngq.outOfRange)}, and AddressSanitizer is named`, t.includes('Correct.') && t.includes('AddressSanitizer'), t);
q = await question('thread_position_in_grid');
await pick(q, 'Not when the host calls dispatchThreads');
t = await qText(q);
check('q-metal-guard: dispatchThreads needs no guard, dispatchThreadgroups does', t.includes('Correct.') && t.includes('llama.cpp uses the'), t);

// ---------------------------------------------------------------- the qualifier table and the bridge
const qrows = await page.$$eval('table[data-table="qualifiers"] tbody tr', (rs) => rs.map((r) => [r.dataset.qualifier, ...[...r.querySelectorAll('td')].map((c) => c.textContent.trim())]));
const wantRows = ['host', 'global', 'device'].map((qq) => {
  const rules = CALLERS.map((c) => legalCall(c, qq, 'cuda'));
  const sure = rules.filter((r) => r.book === 'legal');
  return [qq, sure[0].executedBy];
});
check('the qualifier table has the book\'s three rows, executed-by read from the model',
  qrows.length === 3 && wantRows.every(([qq, ex], i) => qrows[i][0] === qq && qrows[i][4] === EXECUTOR_TEXT[ex]) &&
  qrows[0][1].startsWith('__host__') && qrows[0][1].includes('the default') && qrows[1][2] === 'host code (or device code, with dynamic parallelism)' && qrows[2][2] === 'device code', JSON.stringify(qrows));
const llama = await page.evaluate(() => {
  const f = [...document.querySelectorAll('figure.code')].find((x) => x.querySelector('.tabs button')?.textContent === 'llama.cpp, ggml-cuda/scale.cu');
  return f ? { code: f.querySelector('pre').textContent, status: f.querySelector('.status')?.textContent ?? '' } : null;
});
check('the llama.cpp bridge shows the index and the early-return guard, marked unverified', !!llama && llama.code.includes('blockIdx.x * blockDim.x + threadIdx.x') && llama.code.includes('if (i >= k) { return; }') && llama.status.startsWith('unverified'), JSON.stringify(llama));

// ---------------------------------------------------------------- the page as a whole
const whole = await text('.page');
check('the page cites its book pages and the facts items', whole.includes('§2.5, pp. 33–34') && whole.includes('Fig. 2.10') && whole.includes('notes/today/ch02-facts.md, A6') && whole.includes('A10, B5'));
check('each simulation says what it leaves out (animation, stepper, explorer)', (await page.$$('.note.caveat')).length === 3);
check('answers and the prediction went to the (intercepted) progress API', puts > 0, String(puts));
const expected404 = (u) => /\/content\/ch02\/narration\/(manifest\.json|loop-to-grid-\d\.mp3)$/.test(u);
const real = errors.filter((x) => !/Failed to load resource: the server responded with a status of 404/.test(x));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
if (out) {
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: `${out}-page.png`, fullPage: true });
}

await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
