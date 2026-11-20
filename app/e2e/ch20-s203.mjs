// Browser check for section 20.3: the "step one row" stepper over the corrected Fig. 20.4
// and the outcome table (both against app/src/sim/masksoftmax.ts), the traffic explorer
// (unfusedTraffic), lab 20.1's listing, the corrections table, and questions.
// Usage: node e2e/ch20-s203.mjs [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto } from './browser.mjs';
import { causalSoftmaxProblem } from '../src/sim/attnref.ts';
import { CORRECT_SOFTMAX, SCHEDULES, VARIANTS, runAllRows, runRow, unfusedTraffic } from '../src/sim/masksoftmax.ts';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const LAB_FILES = ['kernel.cu', 'kernel.metal', 'launch.mm'].map((f) => 'ch20/lab1_causal_softmax/solution/' + f);
const labReady = LAB_FILES.every((f) => fs.existsSync(path.join(REPO, 'labs', f)));

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
const notFound = [];
page.on('response', (r) => r.status() === 404 && notFound.push(decodeURIComponent(r.url())));
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
const ST = 'section.bench[aria-label="Step one row"]';
const OC = 'section.bench[aria-label="Every thread order at once"]';
const TR = 'section.bench[aria-label="What the unfused pipeline moves"]';
const clickText = async (scope, label) => {
  const ok = await page.evaluate(
    (scope, label) => {
      const b = [...document.querySelector(scope).querySelectorAll('button')].find((x) => x.textContent.trim() === label);
      if (!b || b.disabled) return false;
      b.click();
      return true;
    },
    scope,
    label,
  );
  await pause(50);
  return ok;
};
const setSelect = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(80);
};
const setRange = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(80);
};
const toggle = async (label) => {
  await page.evaluate((ST, label) => [...document.querySelectorAll(ST + ' label.check')].find((l) => l.textContent.trim() === label).querySelector('input').click(), ST, label);
  await pause(80);
};
const narr = () => page.$eval(ST + ' [data-narrate]', (e) => e.textContent);
const hl = () => page.$$eval(ST + ' .code .ln.hl', (ls) => ls.map((l) => Number(l.dataset.n)));
const toEnd = async () => {
  for (let i = 0; i < 40 && (await clickText(ST, 'Run to next barrier')); i++);
};
const threadRow = (t) => page.$eval(`${ST} tr[data-thread="${t}"]`, (r) => [...r.querySelectorAll('td')].map((c) => c.textContent));
const prow = () => page.$$eval(ST + ' [data-prow] span', (s) => s.map((x) => x.dataset.v));
const cfgOf = (over = {}) => ({ ...CORRECT_SOFTMAX, n: 8, block: 4, ...over });

await goto(page, 'ch20/20.3');
await page.waitForSelector(ST);
await page.waitForSelector(ST + ' .code .ln');

// ---- the snippet and the first step
const lines = await page.$$eval(ST + ' .code .ln', (ls) => ls.length);
check('the stepper shows the corrected kernel, 28 lines', lines === 28, String(lines));
const input = causalSoftmaxProblem(8, 0);
let model = runRow(cfgOf(), input, 5, 'in-order', 1);
await clickText(ST, 'Step');
let n = await narr();
check('step 1: thread 0 reads S[5][0] at line 8, highlighted', n.startsWith('Thread 0, line 8: S[5][0]') && JSON.stringify(await hl()) === '[8]', n);
check('thread 1 owns columns 1 and 5 in row 5 (the small case q-cols points to)', (await threadRow(1))[1] === '1, 5');
// to the max reduction: release at 10, then BlockReduce at 11
await clickText(ST, 'Run to next barrier');
n = await narr();
check('the barrier at line 10 opens', n.startsWith('The barrier at line 10 opens'), n);
await clickText(ST, 'Run to next barrier');
n = await narr();
const red = model.steps.find((s) => s.thread === -1 && s.line === 11);
check('BlockReduce at line 11: the row max, in thread 0 only', n.includes('only thread 0 has it') && (await threadRow(0))[3] !== 'undefined' && (await threadRow(1))[3] === 'undefined' && n.includes('row max is'), n);
check('the reduction step is the model\'s', red !== undefined && Number.isFinite(red.value));
await toEnd();
const done = await page.$eval(ST + ' [data-result]', (e) => e.dataset.result).catch(() => 'none');
const shownP = await prow();
check('run to the end: row 5 matches the reference', done === 'correct' && model.correct);
check('every P cell is the model\'s value', JSON.stringify(shownP) === JSON.stringify(model.P.map(String)), JSON.stringify(shownP));
check('no failure to jump to in the corrected kernel', !(await page.evaluate((ST) => [...document.querySelectorAll(ST + ' button')].some((b) => b.textContent.includes('first failure')), ST)));
if (out) await page.screenshot({ path: out + '-stepper.png' });

// ---- idle threads bring the identity
await setSelect(ST + ' select[aria-label="row"]', 2);
check('row 2: thread 3 reads no column', (await threadRow(3))[1] === 'none');
await clickText(ST, 'Run to next barrier');
await clickText(ST, 'Run to next barrier');
check('thread 3 enters the reduction with -FLT_MAX and 0 (q-idle)', (await threadRow(3))[2] === '−FLT_MAX' && (await threadRow(3))[4] === '0', JSON.stringify(await threadRow(3)));

// ---- defects
const expectRun = async (over, row, schedule, label) => {
  const m = runRow(cfgOf(over), causalSoftmaxProblem(8, over.offset ?? 0), row, schedule, 1);
  const hasBtn = await clickText(ST, 'Go to the first failure');
  const nb = hasBtn ? await narr() : '';
  const bad = hasBtn ? await page.$eval(ST + ' [data-narrate]', (e) => e.classList.contains('bad')) : false;
  await toEnd();
  const res = await page.$eval(ST + ' [data-result]', (e) => e.dataset.result).catch(() => 'none');
  check(
    `${label}: ${m.violations.length ? 'first failure "' + m.violations[0].message.slice(0, 40) + '…", ' : 'no failure, '}result ${m.correct ? 'correct' : 'wrong'}, as the model`,
    hasBtn === m.violations.length > 0 && (!hasBtn || (bad && nb.includes(m.violations[0].message))) && res === (m.correct ? 'correct' : 'wrong'),
    `${hasBtn} ${nb} ${res}`,
  );
};
await setSelect(ST + ' select[aria-label="Thread order"]', 'reversed');
await setSelect(ST + ' select[aria-label="row"]', 5);
await toggle('line 13');
await expectRun({ barrier13: false }, 5, 'reversed', 'no barrier 13, last thread first');
const failN = await narr();
await toggle('line 13');
await toggle('line 19');
await expectRun({ barrier19: false }, 5, 'reversed', 'no barrier 19, last thread first');
await setSelect(ST + ' select[aria-label="Thread order"]', 'shuffled');
await expectRun({ barrier19: false }, 5, 'shuffled', 'no barrier 19, shuffled');
await toggle('line 19');
await toggle('line 22');
await expectRun({ barrier22: false }, 5, 'shuffled', 'no barrier 22, shuffled');
await setSelect(ST + ' select[aria-label="Thread order"]', 'lockstep');
await expectRun({ barrier22: false }, 5, 'lockstep', 'no barrier 22, lockstep');
await toggle('line 22');
await toggle('loop bound idx < N (mask lost)');
await expectRun({ bound: 'n' }, 5, 'lockstep', 'mask lost');
await toggle('loop bound idx < N (mask lost)');
await toggle('subtract the row max');
await expectRun({ subtractMax: false }, 5, 'lockstep', 'no max subtraction');
await toggle('logits + 90');
await expectRun({ subtractMax: false, offset: 90 }, 5, 'lockstep', 'no max subtraction, logits + 90');
if (out) await page.screenshot({ path: out + '-overflow.png' });
check('a failure narration is marked as a failure', failN.length > 0);

// ---- outcome table
const cells = await page.$$eval(OC + ' tbody tr', (rs) => rs.map((r) => ({ id: r.dataset.variant, cells: [...r.querySelectorAll('td[data-outcome]')].map((c) => c.dataset.outcome) })));
const want = VARIANTS.map((v) => ({
  id: v.id,
  cells: SCHEDULES.map((s) => (runAllRows(cfgOf(v.over), causalSoftmaxProblem(8, v.offset ?? 0), s, 1).correct ? 'correct' : 'wrong')),
}));
check('every cell of the outcome table is the model\'s', JSON.stringify(cells) === JSON.stringify(want), JSON.stringify(cells));
const row = (id) => cells.find((c) => c.id === id)?.cells.join(',');
check('lines 10 and 19 never matter; 13 and 22 fail out of lockstep and in-order', row('no10') === 'correct,correct,correct,correct' && row('no19') === 'correct,correct,correct,correct' && row('no13') === 'correct,correct,wrong,wrong' && row('no22') === 'correct,correct,wrong,wrong');

// ---- traffic explorer
const traffic = async () => ({
  sp: Number(await page.$eval(TR + ' [data-sp]', (e) => e.dataset.sp)),
  total: Number(await page.$eval(TR + ' [data-total]', (e) => e.dataset.total)),
  ratio: await page.$eval(TR + ' [data-ratio]', (e) => e.dataset.ratio),
});
let t = await traffic();
let m = unfusedTraffic(4096, 128, 32, 4);
check('default: 32 heads, N = 4096, fp32: S and P hold 4 GiB (q-sp-bytes)', t.sp === 4 * 2 ** 30 && t.total === m.total && t.ratio === (m.total / m.irreducible).toFixed(2), JSON.stringify(t));
await setRange(TR + ' input[aria-label="heads"]', 1);
await setRange(TR + ' input[aria-label="N (log scale)"]', 15);
await clickText(TR, '16-bit');
t = await traffic();
m = unfusedTraffic(32768, 128, 1, 2);
check('one head, N = 32,768, 16-bit: the model\'s bytes', t.sp === m.sp && t.total === m.total && t.ratio === (m.total / m.irreducible).toFixed(2), JSON.stringify(t));
await setSelect(TR + ' select[aria-label="head width"]', 64);
t = await traffic();
const m64 = unfusedTraffic(32768, 64, 1, 2);
check('halving d raises the ratio (it grows like N / d)', t.ratio === (m64.total / m64.irreducible).toFixed(2) && Number(t.ratio) > m.total / m.irreducible, t.ratio);

// ---- corrections and the lab's listing
const corr = await page.$$eval('table[data-table="corrections"] tbody tr', (rs) => rs.length);
check('the "as printed and as the lab writes it" table has six rows', corr === 6, String(corr));
const listing = await page.evaluate(() => {
  const figs = [...document.querySelectorAll('figure.code')].filter((f) => f.textContent.includes('lab 20.1'));
  return figs.map((f) => ({ tabs: [...f.querySelectorAll('.tabs button')].map((b) => b.textContent), text: f.querySelector('pre').textContent, status: f.querySelector('.status')?.textContent ?? '' }));
});
if (labReady) {
  check('lab 20.1\'s solution shows as CUDA, Metal kernel and Metal host tabs', listing.length === 1 && ['CUDA', 'Metal kernel', 'Metal host'].every((x) => listing[0].tabs.includes(x)), JSON.stringify(listing.map((l) => l.tabs)));
  check('the CUDA tab is the lab file, with a status in the fixed vocabulary', listing.length === 1 && !listing[0].text.startsWith('// could not load') && /emulator|unverified|verified on/.test(listing[0].status), listing[0]?.status);
} else {
  check('lab 20.1 is not written yet: the listing says it could not load the planned file', listing.length === 1 && listing[0].text.includes('could not load labs/ch20/lab1_causal_softmax/solution/kernel.cu'), JSON.stringify(listing));
  console.log('     (labs/ch20/lab1_causal_softmax/solution/ is missing; rerun once the lab exists)');
}

// ---- questions
const q = (part) => page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(part)), part);
const typeIn = async (part, v) => {
  const h = await q(part);
  await (await h.$('input[type=number]')).type(String(v));
  await (await h.$('button.primary')).click();
  await pause(150);
  return h.evaluate((e) => e.textContent);
};
const pick = async (part, label) => {
  const h = await q(part);
  await h.evaluate((e, label) => [...e.querySelectorAll('.opts label')].find((l) => l.textContent.trim().startsWith(label)).querySelector('input').click(), label);
  await (await h.$('button.primary')).click();
  await pause(150);
  return h.evaluate((e) => e.textContent);
};
let a = await typeIn('how many columns does thread 3 read', 250);
check('q-cols: 250 columns', a.includes('Correct.'));
a = await pick('thread 3 reads no column', 'The identities');
check('q-idle: the identities', a.includes('Correct.'));
a = await pick('removing the barrier at line 19', 'Thread 0 needs every private sum');
check('q-barrier: the reduction orders the write', a.includes('Correct.'));
a = await pick('Remove the <code>__syncthreads()</code>', 'A fast warp 0').catch(() => '');
if (!a) a = await pick('Lab 20.1 reuses your M4 reduction', 'A fast warp 0');
check('q-war-lab: warp 0 overwrites scratch[0]', a.includes('Correct.'));
a = await typeIn('how much memory do S and P together take', 4);
check('q-sp-bytes: 4 GiB', a.includes('Correct.'));
a = await pick('A kernel computes its block count', '0.');
check('q-griddim: 0 is wrong, with the reason', a.includes('Not quite.') && a.includes('slip'), a.slice(0, 100));
a = await pick('With nq = 1 and nk = 500', 'All 500');
check('q-decode-row: all 500', a.includes('Correct.'));

// Only the planned lab files may be missing.
const allowed = notFound.filter((u) => u.includes('/api/lab-file') && LAB_FILES.some((f) => u.includes(f)) && !labReady);
const other404 = notFound.filter((u) => !allowed.includes(u));
const errs = [...errors];
for (let i = 0; i < allowed.length; i++) {
  const k = errs.findIndex((e) => e.includes('status of 404'));
  if (k >= 0) errs.splice(k, 1);
}
check('no unexpected 404s', other404.length === 0, other404.join(' | '));
check('no console or page errors', errs.length === 0, errs.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
