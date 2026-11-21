// Browser check for section 4.3: the exercise 4 prediction, the barrier timeline
// (presets, typing, dragging), the listing of Fig. 4.4, the bug hunt "Is this barrier
// legal?" (marking, then every kernel under both rules and both thread orders), the
// waves explorer, the backend table and the five questions. Expected values are
// computed here by the models the page runs on (src/sim/barrier.ts and
// src/sim/blocksched.ts), imported directly. The narrated scene on this page has its
// own check, ch04-scene-waves.mjs.
// Usage: node e2e/ch04-s43.mjs [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto, BASE } from './browser.mjs';
import { EXERCISE_4, KERNELS, finding, run, timeline } from '../src/sim/barrier.ts';
import {
  FIG45,
  TAIL_EXAMPLE,
  capacity,
  deviceFromReport,
  equalDurations,
  isGpuBackend,
  profile,
  schedule,
  tail,
  unevenDurations,
  usage,
  waveSizes,
  waves,
} from '../src/sim/blocksched.ts';

const out = process.argv[2];
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
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
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const clean = (s) => s.replace(/\s+/g, ' ').trim();
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
// The page's number format (src/lib/reports.ts, fmt).
const fmt = (v, digits = 3) => (!isFinite(v) ? 'n/a' : v === 0 ? '0' : Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : Number(v.toPrecision(digits)).toString());
const f1 = (v) => v.toFixed(1);
const bench = (title) => `section.bench[aria-label="${title}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const has = (sel) => page.$(sel).then((e) => !!e);
const clickText = async (scope, label, wait = 60) => {
  const done = await page.evaluate((scope, label) => {
    const btn = [...document.querySelectorAll(scope + ' button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(wait);
  return done;
};
// A number input: React only sees a value set through the native setter.
const setInput = async (scope, aria, value) => {
  await page.evaluate((scope, aria, value) => {
    const el = document.querySelector(`${scope} input[aria-label="${aria}"]`);
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(value));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, scope, aria, value);
  await pause(100);
};
const setSelect = async (scope, aria, value) => {
  await page.evaluate((scope, aria, value) => {
    const s = document.querySelector(`${scope} select[aria-label="${aria}"]`);
    s.value = String(value);
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, scope, aria, value);
  await pause(100);
};
const question = (part) => page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(part)), part);
const qText = (q) => q.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
const typeAnswer = async (q, v) => {
  const input = await q.$('input[type=number]');
  await input.focus();
  await input.evaluate((e) => e.select());
  await input.type(String(v));
  await (await q.$('button.primary')).click();
  await pause(120);
};
const pick = async (q, label) => {
  await q.evaluate((e, label) => {
    [...e.querySelectorAll('.opts label')].find((l) => l.textContent.trim().startsWith(label)).querySelector('input').click();
  }, label);
  await (await q.$('button.primary')).click();
  await pause(120);
};
const shot = async (scope, name) => {
  if (!out) return;
  await page.$eval(scope, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};

await goto(page, 'ch04/4.3');
const TL = bench('Barrier timeline');
const HUNT = bench('Is this barrier legal?');
const WAVES = bench('Waves and the tail');
await page.waitForSelector(WAVES);
const prose = clean(await page.$eval('.prose', (e) => e.textContent));

// ---------------------------------------------------------------- prediction: exercise 4
const ex = timeline(EXERCISE_4);
let q = await question('What percentage of the threads\' total time');
check('the prediction quotes the eight times of exercise 4', (await qText(q)).includes(EXERCISE_4.map(f1).join(', ')), await qText(q));
await (await q.$('input[type=number]')).type('25');
await (await q.$('button.primary')).click();
await pause(200);
let t = await qText(q);
check('prediction reveals 17.1%: released at 3.0, 4.1 of 24.0 microseconds waiting', t.includes(`${f1(ex.share * 100)}%. All are released at ${f1(ex.release)} µs`) && t.includes(`is ${f1(ex.total)} µs of thread time, of which ${f1(ex.waiting)} µs is waiting`) && f1(ex.share * 100) === '17.1' && f1(ex.waiting) === '4.1' && f1(ex.total) === '24.0', t);
const waitsDrawn = await q.evaluate((e) => [...e.querySelectorAll('svg g[data-thread]')].map((g) => [...g.querySelectorAll('text')].pop().textContent));
check('the reveal draws the eight threads with the model\'s waits', same(waitsDrawn, ex.waits.map((w) => (w > 0 ? `waits ${f1(w)}` : 'last to arrive'))) && waitsDrawn[2] === 'last to arrive', waitsDrawn.join(' | '));

// ---------------------------------------------------------------- barrier timeline
const summary = async () => ({
  release: await text(TL + ' [data-k="release"]'),
  waiting: await text(TL + ' [data-k="waiting"]'),
  total: await text(TL + ' [data-k="total"]'),
  share: await text(TL + ' [data-k="share"]'),
  all: await text(TL + ' .summary'),
});
const times = () => page.$$eval(TL + ' input[type=number]', (i) => i.map((x) => Number(x.value)));
const agreesTL = (s, d) => {
  const m = timeline(d);
  return s.release === f1(m.release) && s.waiting === f1(m.waiting) && s.total === f1(m.total) && s.share === f1(m.share * 100) + '%';
};
let d = await times();
check('the timeline opens on eight threads, and its summary is the model\'s for the times shown', d.length === 8 && agreesTL(await summary(), d), JSON.stringify(await summary()));
await clickText(TL, 'exercise 4');
d = await times();
let s = await summary();
check('preset "exercise 4": 3.0, 4.1 of 24.0, 17.1%', same(d, [...EXERCISE_4]) && s.release === '3.0' && s.waiting === '4.1' && s.total === '24.0' && s.share === '17.1%' && s.all.includes('when thread 2 arrives'), s.all);
await shot(TL, 'timeline');
await clickText(TL, 'one slow thread');
d = await times();
s = await summary();
const slow = timeline(d);
check('preset "one slow thread": the slowest thread sets every wait (three quarters of the others\' time)', agreesTL(s, d) && slow.slowest.length === 1 && slow.waits.filter((w) => w > 0).every((w) => w / slow.release === 0.75) && s.all.includes('when thread 7 arrives'), s.all);
await clickText(TL, 'balanced');
s = await summary();
check('preset "balanced": nobody waits', agreesTL(s, await times()) && s.waiting === '0.0' && s.share === '0.0%' && s.all.includes('threads 0, 1, 2, 3, 4, 5, 6 and 7 arrive'), s.all);
await setInput(TL, 'time of thread 3', 4.5);
d = await times();
s = await summary();
check('typing a time: thread 3 at 4.5 makes it the last, and the summary follows the model', d[3] === 4.5 && agreesTL(s, d) && s.release === '4.5' && s.all.includes('when thread 3 arrives'), s.all);
await setInput(TL, 'time of thread 0', 9);
check('a typed time is kept inside the axis', (await times())[0] === 5 && agreesTL(await summary(), await times()));
// Drag the end of thread 5's bar to the left.
await clickText(TL, 'exercise 4');
const handle = await page.$(TL + ' svg g[data-thread="5"] circle');
await handle.evaluate((e) => e.scrollIntoView({ block: 'center' }));
await pause(150);
const hb = await handle.boundingBox();
await page.mouse.move(hb.x + hb.width / 2, hb.y + hb.height / 2);
await page.mouse.down();
await page.mouse.move(hb.x + hb.width / 2 - 40, hb.y + hb.height / 2, { steps: 6 });
await page.mouse.move(hb.x + hb.width / 2 - 80, hb.y + hb.height / 2, { steps: 6 });
await page.mouse.up();
await pause(150);
d = await times();
s = await summary();
check('dragging the end of a bar changes that thread\'s time, and only that one', d[5] < EXERCISE_4[5] && d[5] >= 0.1 && d.every((v, i) => i === 5 || v === EXERCISE_4[i]), JSON.stringify(d));
check('after the drag the summary is the model\'s for the new times', agreesTL(s, d) && s.release === '3.0', s.all);
const waitText = await page.$eval(TL + ' svg g[data-thread="5"]', (g) => [...g.querySelectorAll('text')].pop().textContent);
check('and the dragged thread\'s wait is drawn', waitText === `waits ${f1(timeline(d).waits[5])}`, waitText);

// ---------------------------------------------------------------- Fig. 4.4, the listing
const listing = await page.evaluateHandle(() => [...document.querySelectorAll('.prose > figure.code')].find((f) => f.querySelector('figcaption')));
const file = fs.readFileSync(path.join(repo, 'labs/ch04/snippets/incorrect_barrier.cu'), 'utf8').trimEnd().split('\n');
const lines = await listing.evaluate((f) => [...f.querySelectorAll('pre .ln')].map((l) => l.textContent.replace(/\n$/, '')));
check('the listing is labs/ch04/snippets/incorrect_barrier.cu, line for line', same(lines, file), lines.join('\\n'));
const hl = await listing.evaluate((f) => [...f.querySelectorAll('pre .ln.hl')].map((l) => [Number(l.dataset.n), l.textContent.trim()]));
check('the two barrier calls, lines 5 and 9, are marked', same(hl, [[5, '__syncthreads();'], [9, '__syncthreads();']]) && same(KERNELS.incorrect_barrier.sites, [5, 9]), JSON.stringify(hl));
check('the corrected listing has __global__ and states its status exactly', lines[0].startsWith('__global__ void incorrect_barrier_example') && (await listing.evaluate((f) => f.querySelector('.status').textContent)) === 'front-end-checked; wrong on purpose, never run');
check('the apparent errata of Fig. 4.4 are stated', prose.includes('The printed listing has no __global__') && prose.includes('04, 06 and 10 where the listing numbers them 3, 5 and 9'));
check('NVIDIA\'s wording is carried as a dated note', prose.includes('"may hang or produce unintended side effects"') && prose.includes('"non-exited" threads'));

// ---------------------------------------------------------------- bug hunt
const RULE = { 'all-arrive': 'every thread must arrive (the book)', 'exited-excluded': 'ended threads do not block (NVIDIA, Apple)' };
const ORDER = { ascending: 'thread 0 runs first', descending: 'thread 3 runs first' };
const TEXT = {
  'stuck-sites': 'stuck: threads wait at two different calls',
  'stuck-ended': 'stuck: waiting for a thread that has ended',
  'ran-ahead': 'finished, but a thread started a phase too early',
  partial: 'finished: ended threads were not waited for',
  clean: 'finished: every thread met at every barrier',
};
const CASES = [
  { id: 'incorrect_barrier', tab: '1. Fig. 4.4', verdict: 'Undefined' },
  { id: 'barrier_uniform', tab: '2. if (blockIdx.x > 0)', verdict: 'Legal' },
  { id: 'barrier_per_thread', tab: '3. if (threadIdx.x > 0)', verdict: 'Undefined' },
  { id: 'barrier_early_return', tab: '4. early return', verdict: 'Depends on whose rule you read' },
  { id: 'barrier_loop', tab: '5. barrier in a loop', verdict: 'Undefined' },
  { id: 'barrier_guarded', tab: '4, rewritten', verdict: null },
];
const tabs = () => page.$$eval(HUNT + ' .seg[aria-label="Kernel"] button', (b) => b.map((x) => x.textContent.trim()));
const openTab = async (i) => {
  await page.evaluate((HUNT, i) => document.querySelectorAll(HUNT + ' .seg[aria-label="Kernel"] button')[i].click(), HUNT, i);
  await pause(100);
};
const mark = async (label) => {
  await page.evaluate((HUNT, label) => {
    [...document.querySelectorAll(HUNT + ' .panel.mark label')].find((l) => l.textContent.trim() === label).querySelector('input').click();
  }, HUNT, label);
  await pause(60);
  await clickText(HUNT, 'Mark it', 120);
};
const threads = () => page.$$eval(HUNT + ' table.threads tbody tr', (trs) => trs.map((tr) => ({ status: tr.dataset.status, text: tr.children[1].textContent.trim(), cur: tr.classList.contains('cur') })));
const outcome = async () => ((await has(HUNT + ' .chip.outcome')) ? page.$eval(HUNT + ' .chip.outcome', (e) => ({ finding: e.dataset.finding, text: e.textContent.trim(), cls: e.className })) : null);
const stateText = (st) => (st.status === 'ready' && st.line == null ? 'not started' : st.status === 'waiting' ? `waiting at line ${st.site}` : st.status === 'ended' ? 'ended' : 'released, ready to run');
const highlighted = () => page.$$eval(HUNT + ' .code pre .ln.hl', (l) => l.map((x) => Number(x.dataset.n)));
const setRule = (r) => clickText(HUNT, RULE[r], 100);
const setOrder = (o) => clickText(HUNT, ORDER[o], 100);

check('six kernels are offered, five of them to be marked', same(await tabs(), CASES.map((c) => c.tab)) && (await text(HUNT + ' .score')) === '0 of 5 marked, 0 right', (await tabs()).join(' | '));
check('before a kernel is marked its stepper is closed', !(await has(HUNT + ' table.threads')) && (await text(HUNT)).includes('Mark it, then the stepper for this kernel opens.'));
check('the bug hunt states what "stuck" stands for', prose.includes('"Stuck" stands for what the vendors call undefined'));

for (let i = 0; i < CASES.length; i++) {
  const c = CASES[i];
  await openTab(i);
  const shownCode = await page.$$eval(HUNT + ' .code pre .ln', (l) => l.map((x) => x.textContent.replace(/\n$/, '')));
  const src = fs.readFileSync(path.join(repo, 'labs/ch04/snippets', c.id + '.cu'), 'utf8').trimEnd().split('\n');
  check(`kernel ${i + 1}: the code shown is labs/ch04/snippets/${c.id}.cu`, same(shownCode, src) && (await text(HUNT + ' .code .tabs')).startsWith(c.id + '.cu'));
  if (c.verdict) {
    // Kernel 1 is marked wrong on purpose, to see the page say so.
    const mine = i === 0 ? 'Legal' : c.verdict;
    await mark(mine);
    const box = await text(HUNT + ' .panel.verdict-box');
    check(`kernel ${i + 1}: marked "${mine}", the page answers "${c.verdict}"`, box.startsWith(`${mine === c.verdict ? 'Right' : 'Not quite'} ${c.verdict}.`), box);
  } else {
    check('the rewrite is shown for comparison, with nothing to mark', (await text(HUNT + ' .panel.verdict-box')).startsWith('For comparison.') && !(await has(HUNT + ' .panel.mark')));
  }

  // Default rule and order, one step at a time against the model.
  const model = run(KERNELS[c.id].build(4, 1), 'all-arrive', 'ascending');
  let ok = (await text(HUNT + ' .narrate')).startsWith('Nothing has run yet.');
  let where = ok ? '' : 'opening text';
  for (let n = 1; ok && n <= model.steps.length; n++) {
    await clickText(HUNT, 'Step', 30);
    const st = model.steps[n - 1];
    const rows = await threads();
    const want = st.states.map((x, tIdx) => ({ status: x.status, text: stateText(x), cur: st.kind === 'advance' && st.thread === tIdx }));
    const wantHl = st.kind === 'advance' ? st.lines : st.kind === 'release' ? [st.site] : st.waiting.map((g) => g.site);
    if (!same(rows, want) || !same(await highlighted(), [...new Set(wantHl)].sort((a, b) => a - b))) { ok = false; where = `step ${n}: ${JSON.stringify(rows)} hl ${JSON.stringify(await highlighted())}`; }
  }
  check(`kernel ${i + 1}: every step shows the model's thread states and marks the lines just run`, ok, where);

  // All four combinations of rule and order, run to the end.
  const got = [];
  const want = [];
  for (const rule of ['all-arrive', 'exited-excluded'])
    for (const order of ['ascending', 'descending']) {
      await setRule(rule);
      await setOrder(order);
      await clickText(HUNT, 'Run to the end', 80);
      const o = await outcome();
      const f = finding(run(KERNELS[c.id].build(4, 1), rule, order));
      got.push(`${o?.finding}/${o?.text}`);
      want.push(`${f}/${TEXT[f]}`);
    }
  check(`kernel ${i + 1}: the outcome under both rules and both thread orders is the model's (${[...new Set(want.map((w) => w.split('/')[0]))].join(', ')})`, same(got, want), got.join(' | '));
  await setRule('all-arrive');
  await setOrder('ascending');
}

check('the score counts the marks: 5 of 5, 4 right', (await text(HUNT + ' .score')) === '5 of 5 marked, 4 right', await text(HUNT + ' .score'));
check('the tabs show which marks were right', same(await tabs(), ['1. Fig. 4.4 ✗', '2. if (blockIdx.x > 0) ✓', '3. if (threadIdx.x > 0) ✓', '4. early return ✓', '5. barrier in a loop ✓', '4, rewritten']), (await tabs()).join(' | '));

// Fig. 4.4: who waits where.
await openTab(0);
await clickText(HUNT, 'Run to the end', 100);
let rows = await threads();
let narr = await text(HUNT + ' .narrate');
check('Fig. 4.4: threads 0 and 2 wait at line 5, threads 1 and 3 at line 9', same(rows.map((r) => r.text), ['waiting at line 5', 'waiting at line 9', 'waiting at line 5', 'waiting at line 9']), JSON.stringify(rows));
check('and the page says why nothing can move: two calls are two barriers', narr === 'Nothing can move. Threads 0 and 2 wait at line 5; threads 1 and 3 wait at line 9. Two calls are two barriers, and neither will ever have the whole block.' && (await page.$eval(HUNT + ' .narrate', (e) => e.classList.contains('bad'))), narr);
check('both call sites are marked in the code', same(await highlighted(), [5, 9]));
await setRule('exited-excluded');
await clickText(HUNT, 'Run to the end', 100);
check('Fig. 4.4 is stuck under the other rule too', (await outcome()).finding === 'stuck-sites' && (await outcome()).cls.includes('fail'));
await shot(HUNT, 'hunt-fig44');
await setRule('all-arrive');

// Kernel 2: a block-uniform condition is legal, and legal is not the same as right.
await openTab(1);
await clickText(HUNT, 'Run to the end', 100);
check('kernel 2 in block 1: all four meet at the barrier and are released', (await outcome()).finding === 'clean' && finding(run(KERNELS.barrier_uniform.build(4, 1), 'all-arrive')) === 'clean' && (await outcome()).cls.includes('ok'));
await setSelect(HUNT, 'Block', 0);
await clickText(HUNT, 'Run to the end', 100);
const block0 = run(KERNELS.barrier_uniform.build(4, 0), 'all-arrive');
check('kernel 2 in block 0: nobody calls the barrier, nothing hangs, and a thread runs ahead', (await outcome()).finding === finding(block0) && finding(block0) === 'ran-ahead' && block0.releases === 0, JSON.stringify(await outcome()));
await setSelect(HUNT, 'Block', 1);

// Kernel 3: how a misplaced barrier passes a test.
await openTab(2);
await setRule('exited-excluded');
await setOrder('descending');
await clickText(HUNT, 'Run to the end', 100);
const lucky = run(KERNELS.barrier_per_thread.build(4, 1), 'exited-excluded', 'descending');
check('kernel 3, second rule, thread 3 first: it finishes with nothing flagged, as the lesson says', (await outcome()).finding === 'partial' && finding(lucky) === 'partial' && lucky.ahead.length === 0 && !(await outcome()).cls.includes('fail') && !(await page.$eval(HUNT + ' .narrate', (e) => e.classList.contains('bad'))), JSON.stringify(await outcome()));
await setOrder('ascending');
await clickText(HUNT, 'Run to the end', 100);
check('with thread 0 first the same kernel is caught: thread 0 ran ahead', (await outcome()).finding === 'ran-ahead' && (await outcome()).cls.includes('fail'));
await clickText(HUNT, 'Restart', 60);
await clickText(HUNT, 'Step', 60);
narr = await text(HUNT + ' .narrate');
check('and the step that does it is explained', narr.includes('Thread 0 runs lines 4 and 8 and ends. It started phase 1 while threads 1, 2 and 3 still had earlier work to do: nothing separated the phases.'), narr);
await setRule('all-arrive');

// Kernel 4: legal under one wording, not under the other; the rewrite is legal under both.
await openTab(3);
await clickText(HUNT, 'Run to the end', 100);
narr = await text(HUNT + ' .narrate');
check('kernel 4, the book\'s rule: stuck waiting for the thread that returned', (await outcome()).finding === 'stuck-ended' && narr === 'Nothing can move. Threads 0, 1 and 2 wait at line 6 for thread 3, which ended without reaching the call and never will.', narr);
await setRule('exited-excluded');
await clickText(HUNT, 'Run to the end', 100);
check('kernel 4, NVIDIA\'s and Apple\'s wording: released, and nobody ran ahead', (await outcome()).finding === 'partial' && run(KERNELS.barrier_early_return.build(4, 1), 'exited-excluded').ahead.length === 0);
await clickText(HUNT, 'Back', 60);
await clickText(HUNT, 'Back', 60);
await clickText(HUNT, 'Back', 60);
await clickText(HUNT, 'Back', 60);
narr = await text(HUNT + ' .narrate');
check('the release under that rule says who was not waited for', narr.includes('Threads 0, 1 and 2 wait at the call on line 6. Thread 3 ended without reaching it. Under this rule a thread that has ended does not count, so the barrier opens.'), narr);
await shot(HUNT, 'hunt-early-return');
await setRule('all-arrive');
await openTab(5);
check('the rewrite has an accurate status: checked by the front end, not wrong on purpose', (await text(HUNT + ' .code .status')) === 'front-end-checked; never run');
await clickText(HUNT, 'Run to the end', 100);
check('the rewrite is clean under the book\'s rule', (await outcome()).finding === 'clean');

// ---------------------------------------------------------------- wider scopes, the scene
check('wider scopes: the cooperative launch and the removal in CUDA 13 are a dated note', prose.includes('cudaLaunchCooperativeKernel') && prose.includes('was removed in CUDA 13') && prose.includes('Its worked example is in chapter 18'));
check('the narrated scene "One grid, two devices" sits on this page with six parts and a captions switch', (await page.$$('section.scene[aria-label="One grid, two devices"] .seg button')).length === 6 && (await text('section.scene')).includes('captions'));

// ---------------------------------------------------------------- waves and the tail
const W = (k) => text(`${WAVES} [data-k="${k}"]`);
const wavesShown = async () => ({
  capacity: await W('capacity'),
  waves: await W('waves'),
  sizes: (await has(WAVES + ' [data-k="sizes"]')) ? await W('sizes') : null,
  idle: (await has(WAVES + ' [data-k="idle"]')) ? await W('idle') : null,
  idleShare: (await has(WAVES + ' [data-k="idle-share"]')) ? await W('idle-share') : null,
  utilisation: await W('utilisation'),
  bars: await page.$$eval(WAVES + ' svg rect[fill="var(--ink-2)"]', (r) => r.length),
  noTail: await page.evaluate((WAVES) => !!document.querySelector(WAVES + ' .stats .chip.ok'), WAVES),
});
const wavesWant = (blocks, units, slots, uneven = false) => {
  const cap = capacity(units, slots);
  const sch = schedule({ blocks, units, slotsPerUnit: slots, durations: uneven ? unevenDurations(blocks) : equalDurations(blocks) });
  const sizes = waveSizes(blocks, cap);
  const tl = tail(blocks, cap);
  return {
    capacity: fmt(cap, 6),
    waves: fmt(waves(blocks, cap), 4),
    sizes: uneven ? null : sizes.length <= 6 ? sizes.join(', ') : `${sizes.slice(0, 3).join(', ')}, … , ${sizes[sizes.length - 1]}`,
    idle: uneven ? null : fmt(tl.idle, 6),
    idleShare: uneven ? null : fmt(tl.idleShare * 100, 3) + '%',
    utilisation: fmt(usage(sch).utilisation * 100, 3) + '%',
    bars: profile(sch).length,
    noTail: !uneven && tl.idle === 0,
  };
};
const setGrid = async (blocks, units, slots) => {
  await setInput(WAVES, 'Blocks in the grid', blocks);
  await setInput(WAVES, 'SMs', units);
  await setInput(WAVES, 'Blocks per SM', slots);
};
const T = TAIL_EXAMPLE;
let w = await wavesShown();
check('it opens on the book\'s example: 660 on 264 is 2.5 waves of 264, 264 and 132', same(w, wavesWant(T.blocks, T.units, T.slotsPerUnit)) && w.capacity === '264' && w.waves === '2.5' && w.sizes === '264, 264, 132', JSON.stringify(w));
check('the last wave leaves 132 slots, 50%, idle; 83.3% of the capacity is used overall', w.idle === '132' && w.idleShare === '50%' && w.utilisation === '83.3%' && !w.noTail, JSON.stringify(w));
await shot(WAVES, 'waves-tail');
await clickText(WAVES, 'Round the grid up to full waves', 120);
w = await wavesShown();
check('rounding the grid up to 792 blocks removes the tail', (await page.$eval(WAVES + ' input[aria-label="Blocks in the grid"]', (e) => e.value)) === '792' && same(w, wavesWant(792, T.units, T.slotsPerUnit)) && w.idle === '0' && w.noTail && w.utilisation === '100%' && w.waves === '3', JSON.stringify(w));
check('with a full last wave there is nothing left to round', await page.evaluate((WAVES) => [...document.querySelectorAll(WAVES + ' button')].find((b) => b.textContent.trim() === 'Round the grid up to full waves').disabled, WAVES));
await clickText(WAVES, 'Fig. 4.5, small device', 120);
w = await wavesShown();
check('Fig. 4.5, small device: 8 blocks, 2 at once, 4 waves', same(w, wavesWant(FIG45.blocks, FIG45.small.units, FIG45.small.slotsPerUnit)) && w.waves === '4' && w.capacity === '2', JSON.stringify(w));
await clickText(WAVES, 'Fig. 4.5, large device', 120);
w = await wavesShown();
check('Fig. 4.5, large device: the same 8 blocks, 4 at once, 2 waves', same(w, wavesWant(FIG45.blocks, FIG45.large.units, FIG45.large.slotsPerUnit)) && w.waves === '2' && w.capacity === '4', JSON.stringify(w));
await clickText(WAVES, "the book's 660 on 264", 120);
await setInput(WAVES, 'Blocks in the grid', 1000);
w = await wavesShown();
check('1000 blocks on 264: 56 slots, 21.2%, idle in the last wave (the question below)', same(w, wavesWant(1000, T.units, T.slotsPerUnit)) && w.idle === '56' && w.idleShare === '21.2%' && w.sizes === '264, 264, 264, 208', JSON.stringify(w));
const sampling = clean(await page.evaluate(() => [...document.querySelectorAll('.prose > p')].find((p) => p.textContent.includes('Sampling works on'))?.textContent ?? ''));
check('the sampling bridge states its arithmetic: 32,000 / 256 = 125 blocks, 0.47 of a wave; 128,000 is 500 blocks, 1.9 waves; four per thread, 125',
  sampling.includes('vocabulary ÷ (threads per block × elements per thread) blocks') && sampling.includes('is 125 blocks: 0.47 of a wave on the book\'s 264')
  && sampling.includes('128,000 at one element per thread is 500 blocks, 1.9 waves') && sampling.includes('back to 125 blocks'), sampling);
await setInput(WAVES, 'Blocks in the grid', 125);
w = await wavesShown();
check('125 blocks on 264, the sampling case: a fraction of one wave', same(w, wavesWant(125, T.units, T.slotsPerUnit)) && Number(w.waves) < 1 && w.sizes === '125' && w.idle === '139', JSON.stringify(w));
await setGrid(5000, 80, 16);
w = await wavesShown();
check('a typed device (80 SMs of 16 blocks, 5000 blocks) follows the model', same(w, wavesWant(5000, 80, 16)) && w.capacity === '1,280', JSON.stringify(w));
await setGrid(T.blocks, T.units, T.slotsPerUnit);
await clickText(WAVES, 'uneven block times', 200);
w = await wavesShown();
const un = wavesWant(T.blocks, T.units, T.slotsPerUnit, true);
check('uneven block times: no clean waves are claimed, and the utilisation is the scheduler\'s', same(w, un) && w.sizes === null && (await text(WAVES + ' .stats')).includes('there are no clean waves to count'), JSON.stringify(w));
await setInput(WAVES, 'Blocks in the grid', 2640);
const many = await wavesShown();
check('uneven blocks: ten waves use more of the device than two and a half', same(many, wavesWant(2640, T.units, T.slotsPerUnit, true)) && parseFloat(many.utilisation) > parseFloat(w.utilisation), `${w.utilisation} -> ${many.utilisation}`);
await shot(WAVES, 'waves-uneven');
await clickText(WAVES, 'equal block times', 120);
await setInput(WAVES, 'Blocks in the grid', T.blocks);

// The reader's own device: a preset only when a report carries an SM count.
const reports = await (await fetch(BASE + '/api/reports')).json();
const mine = Object.keys(reports).filter(isGpuBackend).map((b) => deviceFromReport(b, reports[b].device)).filter((x) => x.units != null);
if (mine.length) {
  const label = `your ${mine[0].name}: ${fmt(mine[0].units, 6)} from multiProcessorCount`;
  await clickText(WAVES, label, 120);
  check('your device\'s multiProcessorCount is offered as a preset and sets the SM count', (await page.$eval(WAVES + ' input[aria-label="SMs"]', (e) => e.value)) === String(mine[0].units) && !(await has(WAVES + ' .own-device')));
} else {
  check('with no SM count in this machine\'s reports, no preset is invented and the page says so', (await text(WAVES + ' .own-device')).includes('No device report on this machine carries an SM count') && !(await text(WAVES)).includes('from multiProcessorCount'));
}
check('the explorer states what it simplifies', prose.includes('"blocks per SM" is a number you type'));
check('wave, tail effect and transparent scalability are called the book\'s terms', prose.includes('"Wave", "tail effect" and "transparent scalability" are the book\'s terms') && prose.includes('"in more waves"'));

// ---------------------------------------------------------------- backend table
const BT = 'table[aria-label="Barriers across backends"]';
const cells = await page.$$eval(BT + ' tbody tr', (trs) => trs.map((tr) => [...tr.querySelectorAll('td')].map((td) => td.textContent.trim())));
check('the backend table has five rows and a column per backend', cells.length === 5 && cells.every((r) => r.length === 4), JSON.stringify(cells.map((r) => r[0])));
check('block barrier: __syncthreads() on CUDA and HIP, threadgroup_barrier on Metal with the flag left open', cells[0][1] === '__syncthreads()' && cells[0][2] === 'same spelling' && cells[0][3].replace(/\s+/g, ' ').startsWith('threadgroup_barrier(mem_flags::…). The flag names the memory to order: mem_threadgroup for threadgroup memory, mem_device for device buffers'), cells[0].join(' | '));
check('conditional rule: NVIDIA\'s is taught as the portable one and labelled as taken from CUDA\'s documentation', cells[1][2].includes('not stated in AMD\'s pages') && cells[1][2].includes('taken from CUDA\'s documentation, not AMD\'s') && cells[1][3].includes('conditionals and for loops'), cells[1].join(' | '));
check('grid-wide barrier: cooperative launch on CUDA and HIP, none on Metal', cells[4][1].includes('cudaLaunchCooperativeKernel') && cells[4][2].includes('hipLaunchCooperativeKernel') && cells[4][3].startsWith('none documented'), cells[4].join(' | '));
check('the table is marked as not in the book and not run on CUDA or HIP hardware', prose.includes('Nothing in this table has been run on CUDA or HIP hardware by this course.'));

// ---------------------------------------------------------------- questions
q = await question('Who waits where');
await pick(q, 'Nobody waits for long');
check('Fig. 4.4 question: "each pair is complete" is marked wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('A barrier counts the whole block'), await qText(q));
await pick(q, 'Threads 0 and 2 wait at the call on line 5');
check('Fig. 4.4 question: the model\'s answer is accepted', (await qText(q)).includes('Correct.') && same(run(KERNELS.incorrect_barrier.build(4, 0), 'all-arrive').steps.at(-1).waiting, [{ site: 5, threads: [0, 2] }, { site: 9, threads: [1, 3] }]), await qText(q));
q = await question('Which of these conditional barriers is legal');
await pick(q, 'if (i < n) __syncthreads();');
check('uniform question: the range test is marked wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('In the last block some threads pass the test'), await qText(q));
await pick(q, 'if (blockIdx.x % 2 == 0) __syncthreads();');
check('uniform question: the block-uniform condition is accepted, with the caveat about odd blocks', (await qText(q)).includes('Correct.') && (await qText(q)).includes('is a separate question'), await qText(q));
q = await question('What percentage of its block capacity is idle during the last wave');
const tq = tail(1000, 264);
await typeAnswer(q, 79);
check('tail question: the busy share is rejected and the model\'s answer shown', (await qText(q)).includes(`Not quite. The answer is ${Math.round(tq.idleShare * 100)} %.`) && Math.round(tq.idleShare * 100) === 21, await qText(q));
await typeAnswer(q, 21);
check('tail question: 21 is accepted, with 208 run and 56 idle', (await qText(q)).includes('Correct.') && (await qText(q)).includes(`leave ${tq.last} for the fourth`) && (await qText(q)).includes(`= ${tq.idle} slots idle`) && tq.last === 208 && tq.idle === 56, await qText(q));
q = await question('A grid-wide barrier limits how large the grid may be');
await pick(q, 'A grid-wide barrier is implemented in software');
check('grid barrier question: "too slow" is marked wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('the limit is not about speed'), await qText(q));
await pick(q, 'Every thread in a barrier\'s scope must be resident');
check('grid barrier question: residency is accepted', (await qText(q)).includes('Correct.') && (await qText(q)).includes('A thread that has not started cannot arrive'), await qText(q));
q = await question('Which habit breaks transparent scalability');
await pick(q, 'Launching more blocks than the device can run at once');
check('scalability question: "more blocks than fit" is marked wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('That is the normal case'), await qText(q));
await pick(q, 'Making one block wait for a result from another block');
check('scalability question: assumptions about other blocks is accepted', (await qText(q)).includes('Correct.') && (await qText(q)).includes('Source: ch. 4 §4.3, pp. 73–74'), await qText(q));

// ---------------------------------------------------------------- narrow screen
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((scopes) => ({
  page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1,
  benches: scopes.every((s) => document.querySelector(s).getBoundingClientRect().width <= document.documentElement.clientWidth),
}), [TL, HUNT, WAVES]);
check('at a narrow width nothing makes the page scroll sideways', narrow.page && narrow.benches, JSON.stringify(narrow));
await shot(TL, 'narrow-timeline');
await shot(HUNT, 'narrow-hunt');
await shot(WAVES, 'narrow-waves');

await pause(600);
check('answers were saved through PUT /api/progress, which this check intercepts', puts > 0, String(puts));
// Narration audio is generated centrally, after the chapter is built: until then its manifest and clips are absent.
const expected404 = (u) => /\/content\/ch04\/narration\/(manifest\.json|waves-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
