// Browser check for section 14.2: the "Neighbouring pairs at once" bug hunt (three
// interleavings), the listing (snippet with its two marked lines, lab 14.1 tabs when present,
// the host loop), the "Step odd-even sort" stepper (Fig. 14.1 row by row, both stopping rules
// on the trap, the prediction mode, typed values) and two questions. Every expected number is
// recomputed here from src/sim/oddeven.ts. The narrated scene has its own check
// (ch14-scene-oddeven-walk.mjs).
// Usage: node e2e/ch14-s142.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { FIG_14_1, PRESETS, TRAP, overlapStep, runToSorted, walk } from '../src/sim/oddeven.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
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
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const clickText = async (scope, label) => {
  const done = await page.evaluate((sel, label) => {
    const btn = [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, bench(scope), label);
  await pause(70);
  return done;
};
const setSelect = async (scope, labelStart, value) => {
  await page.evaluate((sel, labelStart, value) => {
    const s = [...document.querySelectorAll(sel + ' label.field')].find((l) => l.textContent.trim().startsWith(labelStart)).querySelector('select');
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(s, String(value));
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, bench(scope), labelStart, value);
  await pause(90);
};
const values = (scope, row) => page.$eval(`${bench(scope)} [data-row="${row}"]`, (e) => e.dataset.values);
const shot = async (scope, name) => {
  if (!out) return;
  await page.$eval(bench(scope), (e) => e.scrollIntoView({ block: 'start' }));
  await pause(100);
  await page.screenshot({ path: `${out}-${name}.png` });
};

await goto(page, 'ch14/14.2');
const H = 'Neighbouring pairs at once';
const S = 'Step odd-even sort';
await page.waitForSelector(bench(S));
await page.waitForSelector('figure.code .tabs button');

// ---------------------------------------------------------------- the bug hunt
const reversed = PRESETS[1].values;
await clickText(H, 'Step');
let n = await text(bench(H) + ' .narrate');
check('first operation: thread 0 loads data[0]', n.startsWith(`Thread 0 (owns 0 and 1) loads data[0] = ${reversed[0]} into a.`), n);
await clickText(H, 'Run to the end');
const race = overlapStep(reversed, { kind: 'race', at: 4 });
n = await text(bench(H) + ' .narrate');
check('two threads racing on data[4]: not a permutation, with the model\'s lost and duplicated values',
  (await values(H, 'data')) === race.out.join(' ') && !race.permutation && n.includes(`lost: ${race.lost.join(', ')}; duplicated: ${race.duplicated.join(', ')}.`), n);
check('the duplicated value\'s cells are marked as failures', (await page.$$(bench(H) + ' [data-row="data"] span.bad')).length === race.out.filter((v) => race.duplicated.includes(v)).length);
await shot(H, 'hunt');
await setSelect(H, 'Interleaving', 'in-order');
await clickText(H, 'Run to the end');
const seq = overlapStep(reversed, { kind: 'in-order' });
check('threads one after another: a bubble-sort pass, still a permutation',
  (await values(H, 'data')) === seq.out.join(' ') && (await page.$eval(bench(H) + ' [data-result]', (e) => e.dataset.result)) === 'permutation');
await setSelect(H, 'Interleaving', 'race');
await setSelect(H, 'Values', 0); // Fig. 14.1's row
await setSelect(H, 'Shared element', 1);
await clickText(H, 'Run to the end');
check('a race where only one of the two threads swaps does no harm',
  (await page.$eval(bench(H) + ' [data-result]', (e) => e.dataset.result)) === 'permutation' && overlapStep(FIG_14_1, { kind: 'race', at: 1 }).permutation);

// ---------------------------------------------------------------- the listing
const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent.trim()));
check('the listing opens on the corrected Fig. 14.2', tabs[0] === 'Fig. 14.2, corrected', tabs.join(' | '));
const hl = await page.$$eval('figure.code', (fs) => [...fs[0].querySelectorAll('.ln.hl')].map((l) => `${l.dataset.n}:${l.textContent.trim()}`));
check('lines 5 and 10 are marked: the two differences from the page', hl.join(' | ') === '5:if(i + 1 < N) { | 10:atomicExch(hasChanged, 1u);', hl.join(' | '));
check('the kernel keeps the page\'s 13 lines', (await page.$$eval('figure.code', (fs) => fs[0].querySelectorAll('.ln').length)) === 13);
const labTabs = tabs.filter((t) => t.startsWith('Lab 14.1'));
check('lab 14.1\'s odd-even part appears as tabs when the lab exists (or none yet)', labTabs.length === 0 || labTabs.join() === 'Lab 14.1: CUDA and HIP,Lab 14.1: Metal kernel,Lab 14.1: Metal host', labTabs.join());
if (labTabs.length) {
  await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')].find((b) => b.textContent.trim() === 'Lab 14.1: CUDA and HIP').click());
  await pause(80);
  const lab = await page.$eval('figure.code pre', (e) => e.textContent);
  check('the lab tab shows only its odd-even kernel and host loop', lab.includes('oddEvenStep') && lab.includes('launchOddEvenSort') && !lab.includes('radixScatter'));
}
const host = await page.$$eval('figure.code', (fs) => fs[1]?.textContent ?? '');
check('the host loop from the same file stops after two quiet steps', host.includes('quiet = changed ? 0 : quiet + 1;') && host.includes('while (threads > 0 && quiet < 2)'));

// ---------------------------------------------------------------- the stepper
const pos = () => page.$eval(bench(S) + ' [data-pos]', (e) => Number(e.dataset.pos));
const counter = (k) => page.$eval(`${bench(S)} [data-counter="${k}"]`, (e) => e.textContent.trim());
const fig = walk(FIG_14_1, 7);
await clickText(S, 'Step');
n = await text(bench(S) + ' .narrate');
check('step 1 orders the even pairs: 4 threads compared, 2 swapped', n.includes(`Step 1 ordered the even pairs: ${fig[0].comparisons} threads compared, ${fig[0].swaps} swapped, so the flag is 1.`), n);
check('the row after step 1 is Fig. 14.1\'s second row', (await values(S, 'after')) === fig[0].after.join(' '));
const pairs = await page.$$eval(bench(S) + ' [data-row="before"] [data-pair]', (ps) => ps.map((p) => `${p.dataset.pair}:${p.dataset.swapped}`));
check('four brackets, the pairs (4,5) and (6,7) swapped', pairs.join() === '0-1:no,2-3:no,4-5:yes,6-7:yes', pairs.join());
const rowsDone = [];
for (let k = 1; k < 6; k++) {
  await clickText(S, 'Step');
  rowsDone.push(await values(S, 'after'));
}
check('steps 2 to 6 give Fig. 14.1\'s remaining rows', rowsDone.join(' / ') === fig.slice(1, 6).map((r) => r.after.join(' ')).join(' / '), rowsDone.join(' / '));
const threads = await page.$$eval(bench(S) + ' tr[data-thread]', (rs) => rs.map((r) => r.textContent));
check('on odd step 6 thread 3 has i = 7 and is out of range', threads[3].includes('7') && threads[3].includes('out of range'), threads[3]);
await shot(S, 'stepper-fig');
await clickText(S, 'Run');
const ours = runToSorted(FIG_14_1, 'ours');
check('our rule stops after 7 steps, sorted', (await pos()) === ours.launched && ours.launched === 7 && (await page.$eval(bench(S) + ' [data-stop]', (e) => e.dataset.stop)) === 'sorted');
check('the counters are the model\'s: comparisons, swaps, sorted after step 5',
  (await counter('comparisons')) === String(ours.comparisons) && (await counter('swaps')) === String(ours.swaps) && (await counter('sortedAfter')) === String(ours.sortedAfter),
  [await counter('comparisons'), await counter('swaps'), await counter('sortedAfter')].join());
check('the history lists the input and seven rows', (await page.$$(bench(S) + ' table[aria-label="Rows so far"] tr')).length === 8);

// The trap under both rules.
await setSelect(S, 'Values', 4);
await setSelect(S, 'Host stops', 'book');
await clickText(S, 'Run');
const trapBook = runToSorted(TRAP, 'book');
check('the trap under the book\'s literal rule: one step, unsorted', (await pos()) === trapBook.launched && trapBook.launched === 1 && (await page.$eval(bench(S) + ' [data-stop]', (e) => e.dataset.stop)) === 'unsorted');
await shot(S, 'stepper-trap');
await setSelect(S, 'Host stops', 'ours');
await clickText(S, 'Run');
const trapOurs = runToSorted(TRAP, 'ours');
check('the trap under our rule: sorted, after the model\'s number of steps', (await pos()) === trapOurs.launched && (await page.$eval(bench(S) + ' [data-stop]', (e) => e.dataset.stop)) === 'sorted', String(await pos()));

// Prediction mode on Fig. 14.1.
await setSelect(S, 'Values', 0);
await page.evaluate((sel) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('predict')).querySelector('input').click(), bench(S));
await pause(80);
const marks = await page.$$eval(bench(S) + ' [data-row="predict"] button', (bs) => bs.map((b) => b.textContent.trim()));
check('prediction mode offers one button per pair of the next step', marks.join() === 't0 swaps?,t1 swaps?,t2 swaps?,t3 swaps?', marks.join());
for (const t of [2, 3]) await clickText(S, `t${t} swaps?`);
await clickText(S, 'Check and step');
let v = await page.$eval(bench(S) + ' [data-verdict]', (e) => e.dataset.verdict);
check('marking threads 2 and 3 for step 1: 4 of 4 right', v === '4/4', v);
await clickText(S, 'Check and step'); // mark nothing on the odd step
v = await page.$eval(bench(S) + ' [data-verdict]', (e) => e.dataset.verdict);
const odd = fig[1].threads.filter((t) => t.active);
check('marking nothing on step 2: only the threads that did not swap count as right', v === `${odd.filter((t) => !t.swapped).length}/${odd.length}`, v);
await shot(S, 'stepper-predict');

// Typed values.
await page.evaluate((sel) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('predict')).querySelector('input').click(), bench(S));
const typeValues = (v) => page.$eval(bench(S) + ' input[aria-label="values"]', (e, v) => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(e, v);
  e.dispatchEvent(new Event('input', { bubbles: true }));
}, v);
await typeValues('5 1 4 2 3');
await pause(120);
await clickText(S, 'Run');
const typed = runToSorted([5, 1, 4, 2, 3], 'ours');
check('typed values run to the model\'s result', (await values(S, 'after')) === typed.out.join(' ') && (await pos()) === typed.launched, String(await pos()));
await typeValues('1');
await pause(120);
check('one value is refused with a message', (await text(bench(S))).includes('2 to 8 whole numbers below 100'));

// ---------------------------------------------------------------- questions
const q = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('n/2 = 5 threads')));
await (await q.$('input[type=number]')).type('4');
await (await q.$('button.primary')).click();
await pause(150);
check('numeric question: 4 threads on an odd step', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
const qs = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('stops at the first quiet step')));
await qs.evaluate((e) => [...e.querySelectorAll('.opts label')].find((l) => l.textContent.trim() === '[1, 3, 2, 4]').querySelector('input').click());
await pause(80);
await (await qs.$('button.primary')).click();
await pause(150);
check('choice question: the trap is [1, 3, 2, 4]', (await qs.evaluate((e) => e.textContent)).includes('Correct.'));

// A missing narration manifest (no audio generated yet) is the only 404 allowed.
const expected404 = (u) => /\/content\/ch14\/narration\/(manifest\.json|[\w-]+\.mp3)$/.test(u);
const realErrors = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', realErrors.length === 0 && missing.every(expected404), [...realErrors, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
