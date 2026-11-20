// Browser check for section 10.3: the lab listing, the in-place stepper (barrier on and
// off, four thread orders, three data sets) and the questions. Expected values come from
// the tested model, src/sim/reduction.ts, imported here directly.
// Usage: APP_URL=http://localhost:5173 node e2e/ch10-s103.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { EXERCISE_6, FIG_VALUES, arraysAfter, firstIssueStep, runInPlace } from '../src/sim/reduction.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
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
const S = 'Step the simple kernel';
const bench = `section.bench[aria-label="${S}"]`;
const text = (sel) => page.evaluate((s) => document.querySelector(s)?.textContent.replace(/\s+/g, ' ').trim() ?? '', sel);
const click = async (label) => {
  const done = await page.evaluate((sel, label) => {
    const b = [...document.querySelectorAll(sel + ' button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, bench, label);
  await pause(50);
  return done;
};
const select = async (label, value) => {
  await page.select(`${bench} select[aria-label="${label}"]`, value);
  await pause(80);
};
const runToEnd = async () => {
  for (let i = 0; i < 80; i++) if (!(await click('Run to the barrier')) && !(await click('Run to the end'))) break;
};
const row = (r) => page.$$eval(`${bench} svg[data-part="history"] g[data-row="${r}"] g[data-cell]`, (gs) =>
  gs.map((g) => ({ v: g.dataset.value, w: g.dataset.written === '1', bad: g.dataset.bad === '1' })));
const narrate = () => text(`${bench} [data-readout="narrate"]`);

await goto(page, 'ch10/10.3');
await page.waitForSelector(bench);
await page.waitForSelector(`${bench} .code pre`);

// The lab listing, with the row offset marked, and its Metal tab with mem_device.
const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent));
check('listing has CUDA and Metal tabs from lab 10.1', tabs.includes('CUDA (lab 10.1)') && tabs.includes('Metal (lab 10.1)'), tabs.join('|'));
const first = await text('figure.code pre');
check('CUDA tab is the lab kernel with its row offset', first.includes('SimpleSumReductionKernel') && first.includes('input += 2 * blockDim.x * blockIdx.x'), first.slice(0, 120));
const marked = await page.$$eval('figure.code:first-of-type .ln.hl', (l) => l.map((x) => x.textContent));
check('the row offset and output[blockIdx.x] lines are marked', marked.length === 2 && marked.some((l) => l.includes('blockIdx')), JSON.stringify(marked));
await page.evaluate(() => [...document.querySelectorAll('.code .tabs button')].find((b) => b.textContent === 'Metal (lab 10.1)').click());
await pause(80);
const metal = await text('figure.code pre');
const metalMarked = await page.$$eval('figure.code .ln.hl', (l) => l.map((x) => x.textContent));
check('Metal tab: simple_sum with mem_device on the marked line', metal.includes('simple_sum') && metalMarked.some((l) => l.includes('mem_flags::mem_device')), JSON.stringify(metalMarked));

// The stepper, 16 values, thread 0 first, barrier on.
const model = runInPlace('simple', FIG_VALUES, { schedule: 'in-order' });
check('step counter matches the model', (await text(`${bench} [data-readout="step"]`)) === `step 0 of ${model.steps.length}`, await text(`${bench} [data-readout="step"]`));
await click('Step');
const s1 = model.steps[0];
let n = await narrate();
check('step 1: thread 0 adds input[1] into input[0]', n.includes(`Thread 0, iteration 1 (stride 1)`) && n.includes(`input[0] = ${s1.write.value}`), n);
check('the add line is highlighted', (await text(`${bench} .code .ln.hl`)).includes('input[i] += input[i + stride]'));
await click('Step');
n = await narrate();
check('step 2: thread 0 waits at the barrier', n.includes('Thread 0 reaches __syncthreads() at the end of iteration 1'), n);
check('thread table says thread 0 is waiting', (await text(`${bench} tr[data-thread="0"]`)).includes('waiting at the barrier'));
await click('Run to the barrier');
n = await narrate();
check('run to the barrier ends on the opening of iteration 1', n.startsWith('The barrier opens: all 8 threads have finished iteration 1'), n);
const r1 = await row(1);
const want1 = arraysAfter('simple', FIG_VALUES)[0];
check('row 1: 8 elements written, values as the model', r1.filter((c) => c.w).length === 8 && r1.every((c, i) => Number(c.v) === want1[i]), JSON.stringify(r1));
await runToEnd();
check('with the barrier, the output is the sum 69', (await text(`${bench} [data-readout="output"]`)).includes('the sum, 69'));
const r4 = await row(4);
const final = arraysAfter('simple', FIG_VALUES)[3];
check('last row is the final array', r4.every((c, i) => Number(c.v) === final[i]), JSON.stringify(r4.map((c) => c.v)));
check('no failure to jump to with the barrier', !(await text(bench)).includes('Go to the first failure'));
if (out) await page.screenshot({ path: out + '-s103-end.png' });

// Without the barrier: thread 0 reads input[2] too early.
await page.click(`${bench} label.check input`);
await pause(100);
const broken = runInPlace('simple', FIG_VALUES, { schedule: 'in-order', barrier: false });
check('without the barrier a failure is offered', await click('Go to the first failure'));
n = await narrate();
check('the failure is thread 0 reading input[2] before thread 1 wrote it', n.includes('Failure:') && n.includes(broken.issues[0].message), n);
check('the stepper is at the model\'s first failing step', (await text(`${bench} [data-readout="step"]`)).startsWith(`step ${firstIssueStep(broken)} of`));
await runToEnd();
check('without the barrier, thread 0 first: wrong answer', (await page.$(`${bench} [data-outcome="wrong"]`)) != null);
check('red elements mark writes from stale reads', (await row(2)).some((c) => c.bad));
if (out) await page.screenshot({ path: out + '-s103-fail.png' });
await select('Thread order', 'lockstep');
await runToEnd();
check('without the barrier, lockstep: still the right sum', (await page.$(`${bench} [data-outcome="correct"]`)) != null);
await select('Thread order', 'reversed');
await runToEnd();
check('without the barrier, last thread first: still the right sum', (await page.$(`${bench} [data-outcome="correct"]`)) != null);
await select('Thread order', 'shuffled');
await runToEnd();
check('without the barrier, shuffled: wrong, as the model', ((await page.$(`${bench} [data-outcome="wrong"]`)) != null) === (runInPlace('simple', FIG_VALUES, { schedule: 'shuffled', barrier: false }).outcome === 'wrong'));

// Exercise 6 with the barrier back on.
await page.click(`${bench} label.check input`);
await pause(80);
await select('Data', 'ex6');
await runToEnd();
const ex = await row(3);
check('exercise 6a: input[4] ends at 17, output 36', ex[4].v === '17' && (await text(`${bench} [data-readout="output"]`)).includes('36'), JSON.stringify(ex.map((c) => c.v)));
check('exercise 6a rows match the model', JSON.stringify((await row(1)).map((c) => Number(c.v))) === JSON.stringify(arraysAfter('simple', EXERCISE_6)[0]));

// Your own values: a bad list is refused, a good one runs.
await select('Data', 'own');
const own = `${bench} input[aria-label="your values"]`;
await page.$eval(own, (e) => { e.focus(); e.select(); });
await page.type(own, '1 2 3');
await pause(80);
check('three values are refused', (await text(bench)).includes('Give 2, 4, 8 or 16 numbers'));
await page.type(own, ' 4');
await pause(80);
await runToEnd();
check('four values run to their sum, 10', (await text(`${bench} [data-readout="output"]`)).includes('the sum, 10'));

// Questions.
const q = async (snippet) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(t)), snippet);
const qa = await q('exercise 6a');
await (await qa.$('input[type=number]')).type('17');
await (await qa.$('button.primary')).click();
await pause(100);
check('q-ex6a accepts 17', (await qa.evaluate((e) => e.textContent)).includes('Correct.'));
const qs = await q('Remove the barrier and run');
await qs.evaluate((e) => e.querySelectorAll('input[type=radio]')[1].click());
await (await qs.$('button.primary')).click();
await pause(100);
check('q-stale-read: the stride-2 read of input[2] is right', (await qs.evaluate((e) => e.textContent)).includes('Correct.'));
const qb = await q('3 blocks of 1024 threads');
await qb.evaluate((e) => e.querySelectorAll('input[type=radio]')[0].click());
await (await qb.$('button.primary')).click();
await pause(100);
check('q-three-blocks: "each block its own third" is marked wrong', (await qb.evaluate((e) => e.textContent)).includes('Not quite.'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
