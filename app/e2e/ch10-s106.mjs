// Browser check for section 10.6: the lab listing, the bug hunt "Where does the barrier
// go?" (predictions, the table over four thread orders, the replay stepper) and the
// global-traffic explorer. Expected values come from src/sim/reduction.ts.
// Usage: APP_URL=http://localhost:5173 node e2e/ch10-s106.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { checkLabStatuses } from './labstatus.mjs';
import { FIG_VALUES, VARIANTS, firstIssueStep, requests, runShared } from '../src/sim/reduction.ts';

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
const text = (sel) => page.evaluate((s) => document.querySelector(s)?.textContent.replace(/\s+/g, ' ').trim() ?? '', sel);
const HUNT = 'section.bench[aria-label="Where does the barrier go?"]';
const TRAF = 'section.bench[aria-label="Global traffic, three kernels"]';
const SCHEDULES = ['lockstep', 'in-order', 'reversed', 'shuffled'];
const press = async (root, label, group) => {
  const done = await page.evaluate((root, label, group) => {
    const scope = group ? document.querySelector(`${root} [aria-label="${group}"]`) : document.querySelector(root);
    const b = [...scope.querySelectorAll('button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, root, label, group);
  await pause(60);
  return done;
};
const outcomes = () => page.$$eval(`${HUNT} table[data-part="hunt-table"] tbody tr`, (rows) =>
  Object.fromEntries(rows.map((r) => [r.dataset.variant, [...r.querySelectorAll('td[data-outcome]')].map((c) => c.dataset.outcome)])));
const narrate = () => text(`${HUNT} [data-readout="hunt-narrate"]`);

await goto(page, 'ch10/10.6');
await page.waitForSelector(HUNT);
await page.waitForSelector('figure.code pre');

// The listing.
const listing = await text('figure.code pre');
check('listing is lab 10.1\'s shared-memory kernel, without the ruler comment', listing.includes('SharedMemorySumReductionKernel') && !listing.includes('-------'));
const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
check('CUDA and Metal tabs', tabs.includes('CUDA (lab 10.1)') && tabs.includes('Metal (lab 10.1)'), tabs.join('|'));

// Predictions, hidden outcomes, then the run.
let table = await outcomes();
check('outcomes are hidden before the run', Object.values(table).flat().every((o) => o === 'hidden'));
const guesses = { book: 'correct', bottom: 'some', none: 'some', 'inside-if': 'deadlock', le: 'oob', two: 'correct', 'two-race': 'correct' };
for (const v of VARIANTS) await page.select(`${HUNT} tr[data-variant="${v}"] select`, guesses[v]);
await press(HUNT, 'Run all four orders');
table = await outcomes();
const ones = new Array(16).fill(1);
const want = Object.fromEntries(VARIANTS.map((v) => [v, SCHEDULES.map((s) => runShared(v, ones, s).outcome)]));
check('all-ones: every cell is the model\'s outcome', JSON.stringify(table) === JSON.stringify(want), JSON.stringify(table));
check('the book\'s kernel and max-then-sum with the barrier are correct everywhere', table.book.every((o) => o === 'correct') && table.two.every((o) => o === 'correct'));
check('the barrier inside the if is stuck in every order', table['inside-if'].every((o) => o === 'deadlock'));
check('6 of 7 predictions scored right (the race guess was wrong)', (await text(`${HUNT} [data-readout="score"]`)).startsWith('6 of 7'), await text(`${HUNT} [data-readout="score"]`));
check('the wrong guess shows the actual class', (await text(`${HUNT} tr[data-variant="two-race"] [data-verdict]`)).includes('actually: wrong in some orders'));
if (out) await page.screenshot({ path: out + '-s106-table.png' });

// Replay a cell: barrier at the bottom, thread 0 first.
await page.evaluate((sel) => document.querySelector(`${sel} tr[data-variant="bottom"] td[data-outcome]:nth-of-type(4) button`).click(), HUNT);
await pause(100);
const bottom = runShared('bottom', ones, 'in-order');
check('clicking a cell selects its variant and order', (await page.$eval(`${HUNT} select[aria-label="Variant"]`, (s) => s.value)) === 'bottom' && (await page.$eval(`${HUNT} select[aria-label="Thread order"]`, (s) => s.value)) === 'in-order');
await press(HUNT, 'Go to the first failure');
let n = await narrate();
check('the first failure: thread 0 reads input_s[4] before anyone wrote it', n.includes('Failure:') && n.includes(bottom.issues[0].message), n);
check('the stepper is at the model\'s failing step', (await text(`${HUNT} [data-readout="hunt-step"]`)).startsWith(`step ${firstIssueStep(bottom)} of`));
check('the failing slot is marked', (await page.$$(`${HUNT} .mx.shared span.bad`)).length === 1);
check('the barrier line is not highlighted, the add is', (await text(`${HUNT} .code .ln.hl`)).includes('input_s[t] += input_s[t + stride]'));
await press(HUNT, 'Run to the end');
check('...and the output is garbage', (await text(`${HUNT} [data-readout="hunt-output"]`)).includes('wrong answer'));

// The book's kernel stepped barrier by barrier.
await page.select(`${HUNT} select[aria-label="Variant"]`, 'book');
await pause(80);
await press(HUNT, 'Step');
await press(HUNT, 'Step');
n = await narrate();
check('book: thread 0 loads, then waits on line 6', n.includes('Thread 0 reaches the barrier on line 6'), n);
await press(HUNT, 'Run to the next barrier');
n = await narrate();
check('book: the barrier on line 6 opens for all 8', n.startsWith('The barrier on line 6 opens: all 8 threads have arrived'), n);
check('all eight loads are in shared memory, each 2', JSON.stringify(await page.$$eval(`${HUNT} .mx.shared span`, (s) => s.map((x) => x.dataset.value))) === JSON.stringify(new Array(8).fill('2')));
await press(HUNT, 'Run to the end');
check('book: the output is 16', (await text(`${HUNT} [data-readout="hunt-output"]`)).includes('= 16'));

// Barrier inside the if: stuck, with the lines named.
await page.select(`${HUNT} select[aria-label="Variant"]`, 'inside-if');
await pause(80);
await press(HUNT, 'Run to the end');
n = await narrate();
check('inside the if: threads 0 to 3 wait on line 7, 4 to 7 ended', n.includes('threads 0 to 3 wait at the barrier on line 7') && n.includes('threads 4 to 7 ended without reaching it'), n);
check('the thread table shows the waiting line', (await text(`${HUNT} .panel table.threads`)).includes('waiting on line 7'));

// <=: out of bounds.
await page.select(`${HUNT} select[aria-label="Variant"]`, 'le');
await pause(80);
await press(HUNT, 'Go to the first failure');
n = await narrate();
check('<=: a read past the end of input_s', n.includes('input_s[8], past the end of the 8-element array'), n);

// Two reductions without the barrier: thread 1 finds the sum where the max was.
await page.select(`${HUNT} select[aria-label="Variant"]`, 'two-race');
await pause(80);
await press(HUNT, 'Go to the first failure');
n = await narrate();
check('max then sum without the barrier: write after read on buf_s[0]', n.includes('needed the max in buf_s[0]'), n);
check('the m column shows registers', (await text(`${HUNT} .panel table.threads thead`)).includes('m'));
await press(HUNT, 'Run to the end');
check('...and the output is 14, as on the emulator', (await text(`${HUNT} [data-readout="hunt-output"]`)).includes('= 14'));
const replayStatus = () => text(`${HUNT} .code .tabs .status`);
check('a broken variant is labelled wrong on purpose, not as checked code', (await replayStatus()) === 'front-end-checked; wrong on purpose (behaviour emulator-checked)', await replayStatus());
check('the replay tab is short and the caption names the kernel', (await text(`${HUNT} .code .tabs button`)) === 'the kernel' && (await text(`${HUNT} .code figcaption`)).includes('MaxThenSumNoBarrierKernel'));
await page.select(`${HUNT} select[aria-label="Variant"]`, 'two');
await pause(80);
check('a correct variant keeps the emulator-checked status', (await replayStatus()) === 'front-end-checked; emulator-checked; unverified on CUDA and HIP hardware', await replayStatus());
await page.select(`${HUNT} select[aria-label="Variant"]`, 'two-race');
await pause(80);
if (out) await page.screenshot({ path: out + '-s106-replay.png' });

// Mixed input: the table follows the model.
await press(HUNT, '3, 1, 7, 0, ...', 'Input');
table = await outcomes();
const wantMixed = Object.fromEntries(VARIANTS.map((v) => [v, SCHEDULES.map((s) => runShared(v, FIG_VALUES, s).outcome)]));
check('mixed input: every cell is the model\'s outcome', JSON.stringify(table) === JSON.stringify(wantMixed), JSON.stringify(table));

// Global traffic.
const traffic = () => page.$$eval(`${TRAF} [data-kind]`, (r) => r.map((x) => Number(x.dataset.total)));
check('traffic at 256: 141, 36, 9', JSON.stringify(await traffic()) === '[141,36,9]');
await page.$eval(`${TRAF} input[aria-label="N"]`, (e) => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(e, '5');
  e.dispatchEvent(new Event('input', { bubbles: true }));
});
await pause(80);
check('traffic at 2048: 1149, 204, 65', JSON.stringify(await traffic()) === JSON.stringify(['simple', 'convergent', 'shared'].map((k) => requests(k, 2048, 32).total)));
check('the formula is spelled out: 2048/32 + 1 = 65', (await text(`${TRAF} [data-readout="traffic-formula"]`)).includes('2048/32 + 1 = 64 coalesced loads'));

// Questions.
const q = async (snippet) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(t)), snippet);
const q1 = await q('The shared-memory kernel on 2048 values');
await (await q1.$('input[type=number]')).type('65');
await (await q1.$('button.primary')).click();
await pause(100);
check('q-req-shared accepts 65', (await q1.evaluate((e) => e.textContent)).includes('Correct.'));
const q2 = await q('A softmax kernel reduces the max');
await q2.evaluate((e) => e.querySelectorAll('input[type=radio]')[2].click());
await (await q2.$('button.primary')).click();
await pause(100);
check('q-two-reductions: a barrier for a write-after-read', (await q2.evaluate((e) => e.textContent)).includes('Correct.'));
const q3 = await q('at the top of the loop body');
await q3.evaluate((e) => e.querySelectorAll('input[type=radio]')[2].click());
await (await q3.$('button.primary')).click();
await pause(100);
check('q-barrier-top: "the bottom would work" is wrong', (await q3.evaluate((e) => e.textContent)).includes('Not quite.'));

// The listings' statuses: LabKit's words from this machine's reports, and "failed on this machine" when a run failed.
await checkLabStatuses(check, 'ch10/10.6', [
  { label: 'CUDA (lab 10.1)', lab: 'ch10/lab1_sum', file: 'kernel.cu' },
  { label: 'Metal (lab 10.1)', lab: 'ch10/lab1_sum', file: 'kernel.metal' },
]);

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
