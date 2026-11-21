// Browser check for section 10.4: the 256-value table, the three-line diff, the efficiency
// prediction, the warp map (sizes, widths, kernels, iterations, quiz) and "Mirror it"
// (exercise 3). Expected values come from src/sim/reduction.ts, imported here directly.
// Usage: APP_URL=http://localhost:5173 node e2e/ch10-s104.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { checkLabStatuses } from './labstatus.mjs';
import { EXERCISE_6, MIRROR_SPEC, arraysAfter, iterations, runSpec, FIG_VALUES } from '../src/sim/reduction.ts';

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
const MAP = 'section.bench[aria-label="Warp map"]';
const MIR = 'section.bench[aria-label="Mirror it"]';
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
const setRange = async (sel, v) => {
  await page.$eval(sel, (e, v) => {
    const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
    set.call(e, String(v));
    e.dispatchEvent(new Event('input', { bubbles: true }));
    e.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(80);
};
const panel = (kind) => `${MAP} .panel[data-kernel="${kind}"]`;
const states = (kind) => page.$$eval(`${panel(kind)} svg[data-part="warps"] g[data-warp]`, (gs) => gs.map((g) => g.dataset.state));
const attr = (sel, a) => page.$eval(sel, (e, a) => e.getAttribute(a), a);

await goto(page, 'ch10/10.4');
await page.waitForSelector(MAP);
await page.waitForSelector('figure.code pre');

// The 256-value table and the diff.
check('book table: 864 slots for the simple kernel', (await text('[data-total="simple"]')).includes('864'));
check('book table: 384 slots for the convergent kernel', (await text('[data-total="convergent"]')).includes('384'));
const code = await text('figure.code pre');
const hl = await page.$eval('figure.code', (f) => [...f.querySelectorAll('.ln.hl')].map((x) => x.textContent));
check('listing is the lab\'s convergent kernel', code.includes('ConvergentSumReductionKernel'));
check('exactly three lines are marked as changed', hl.length === 3 && hl.some((l) => l.includes('threadIdx.x < stride')) && hl.some((l) => l.includes('stride = blockDim.x')), JSON.stringify(hl));

// The efficiency prediction.
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('what efficiency do you expect')));
await (await pred.$('input[aria-label="your prediction"]')).type('50');
await pred.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Commit prediction').click());
await pause(120);
const reveal = await pred.evaluate((e) => e.textContent);
check('the reveal gives 384 slots, 255 results, 66%', reveal.includes('384 slots for 255 results: 66%'), reveal);

// The warp map at 256, side by side.
check('256, both kernels: the model\'s totals', (await attr(`${panel('simple')} [data-readout="totals"]`, 'data-slots')) === '864' && (await attr(`${panel('convergent')} [data-readout="totals"]`, 'data-slots')) === '384');
await press(MAP, 'Step');
check('iteration 2: every simple warp diverges', JSON.stringify(await states('simple')) === JSON.stringify(['divergent', 'divergent', 'divergent', 'divergent']));
check('iteration 2: convergent warps full, full, idle, idle', JSON.stringify(await states('convergent')) === JSON.stringify(['full', 'full', 'idle', 'idle']));
await setRange(`${MAP} input[aria-label="Iteration"]`, 7);
check('iteration 7: simple warps 0 and 2 one lane, 1 and 3 idle', JSON.stringify(await states('simple')) === JSON.stringify(['divergent', 'idle', 'divergent', 'idle']));
check('the active lanes drawn match the model', (await page.$$eval(`${panel('simple')} svg[data-part="warps"] rect[fill="var(--ink-2)"]`, (r) => r.length)) === iterations('simple', 256, 32).iterations[6].active);
if (out) await page.screenshot({ path: out + '-s104-map.png' });
const erratum = await text(`${MAP} [data-readout="erratum"]`);
check('erratum line states the printed 736 and "ten", and the model\'s 864 and 7 against 5', erratum.includes('736 slots') && erratum.includes('from ten to five') && erratum.includes('gives 864, and counting from the code gives 7 divergent iterations against 5'), erratum);

// Exercises 1 and 2 on the map.
await press(MAP, '1024', 'N');
await setRange(`${MAP} input[aria-label="Iteration"]`, 5);
check('exercise 1: 16 divergent warps in iteration 5 (simple, 1024)', (await attr(`${panel('simple')} [data-divergent]`, 'data-divergent')) === '16');
check('exercise 2: none in the convergent kernel', (await attr(`${panel('convergent')} [data-divergent]`, 'data-divergent')) === '0');
check('1024 convergent: 1152 slots', (await attr(`${panel('convergent')} [data-readout="totals"]`, 'data-slots')) === '1152');

// 64-wide warps.
await press(MAP, '256', 'N');
await page.select(`${MAP} select[aria-label="Warp width"]`, 'w64');
await pause(80);
check('64-wide, 256: convergent 576 slots, simple 960', (await attr(`${panel('convergent')} [data-readout="totals"]`, 'data-slots')) === '576' && (await attr(`${panel('simple')} [data-readout="totals"]`, 'data-slots')) === '960');
check('64-wide: two warp rows', (await states('convergent')).length === 2);

// One kernel at a time.
await press(MAP, 'simple', 'Kernel');
check('"simple" shows one panel', (await page.$$(`${MAP} .panel[data-kernel]`)).length === 1);
await press(MAP, 'side by side', 'Kernel');

// Quiz me: draw, answer from the model, reveal, and the map follows.
await press(MAP, 'Draw a question');
const qtext = await text(`${MAP} [data-readout="quiz"]`);
const m = /The (simple|convergent) kernel on (\d+) values .* (\d+)-wide warps, iteration (\d+)/.exec(qtext);
check('quiz draws a question', !!m, qtext);
if (m) {
  const [, kind, N, w, it] = m;
  const answer = iterations(kind, Number(N), Number(w)).iterations[Number(it) - 1].divergentWarps;
  await page.type(`${MAP} input[aria-label="divergent warps"]`, String(answer));
  await press(MAP, 'Reveal');
  check('quiz reveal agrees with the model', (await attr(`${MAP} [data-readout="quiz-answer"]`, 'data-answer')) === String(answer) && (await text(`${MAP} [data-readout="quiz-answer"]`)).startsWith('right'));
  check('the map jumps to the quiz\'s kernel and iteration', (await page.$(`${MAP} .panel[data-kernel="${kind}"][data-iteration="${it}"]`)) != null && (await page.$$(`${MAP} .panel[data-kernel]`)).length === 1);
}

// Mirror it.
const verdict = async () => page.$eval(`${MIR} [data-readout="verdict"]`, (e) => ({ correct: e.dataset.correct, last: e.dataset.last, text: e.textContent }));
let v = await verdict();
check('Mirror it starts from Fig. 10.8: right sum, total in element 0', v.correct === '1' && v.last === '0', JSON.stringify(v));
for (const [label, value] of [['Owned element i =', MIRROR_SPEC.owned], ['Active when', MIRROR_SPEC.test], ['Partner', MIRROR_SPEC.partner], ['After the loop', MIRROR_SPEC.writer]]) {
  await page.select(`${MIR} select[aria-label="${label}"]`, value);
  await pause(60);
}
v = await verdict();
check('the mirrored parts: right sum, total in the last element', v.correct === '1' && v.last === '1' && v.text.includes("exercise 3's pattern"), JSON.stringify(v));
const lastRow = await page.$$eval(`${MIR} svg[data-part="spec-grid"] g[data-row="4"] g[data-cell]`, (gs) => gs.map((g) => g.dataset.value));
check('the grid\'s last row holds the total in element 15', lastRow[15] === String(runSpec(MIRROR_SPEC, FIG_VALUES).total), JSON.stringify(lastRow));
await page.select(`${MIR} select[aria-label="Partner"]`, 'plus');
await pause(60);
v = await verdict();
check('input[i + stride] in the mirror: out of bounds', v.text.includes('out of bounds') && v.correct === '0');
await page.select(`${MIR} select[aria-label="Partner"]`, 'minus');
await page.select(`${MIR} select[aria-label="Active when"]`, 'ge');
await pause(60);
v = await verdict();
check('a wrong active test is not accepted', v.correct === '0', JSON.stringify(v));
await press(MIR, 'Show our answer');
await page.waitForFunction((sel) => document.querySelector(sel)?.textContent.includes('MirroredSumReductionKernel'), {}, `${MIR} figure.code:last-of-type pre`);
check('our answer is the snippet file', (await text(`${MIR} figure.code:last-of-type pre`)).includes('threadIdx.x >= blockDim.x - stride'));
if (out) await page.screenshot({ path: out + '-s104-mirror.png' });
await press(MIR, 'Back to Fig. 10.8');
await page.select(`${MIR} select[aria-label="Data"]`, 'ex6');
await pause(80);
const ex6 = await page.$$eval(`${MIR} svg[data-part="spec-grid"] g[data-row="3"] g[data-cell]`, (gs) => gs.map((g) => Number(g.dataset.value)));
check('exercise 6b on Fig. 10.8: the model\'s final array, input[1] = 15', JSON.stringify(ex6) === JSON.stringify(arraysAfter('convergent', EXERCISE_6)[2]) && ex6[1] === 15, JSON.stringify(ex6));

// Questions.
const q = async (snippet) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(t)), snippet);
for (const [snippet, value] of [['Book exercise 1', '16'], ['Book exercise 2', '0'], ['exercise 6b', '15'], ['1024 values with 32-wide warps. What is its', '89'], ['64 wide', '576']]) {
  const h = await q(snippet);
  await (await h.$('input[type=number]')).type(value);
  await (await h.$('button.primary')).click();
  await pause(100);
  check(`question "${snippet}" accepts ${value}`, (await h.evaluate((e) => e.textContent)).includes('Correct.'));
}
const qc = await q('could the convergent kernel give a different answer');
await qc.evaluate((e) => e.querySelectorAll('input[type=radio]')[3].click());
await (await qc.$('button.primary')).click();
await pause(100);
check('q-noncommutative: string concatenation', (await qc.evaluate((e) => e.textContent)).includes('Correct.'));
const q3 = await q('Which active test is right?');
await q3.evaluate((e) => e.querySelectorAll('input[type=radio]')[2].click());
await (await q3.$('button.primary')).click();
await pause(100);
check('q-ex3: blockDim.x - stride with i - stride', (await q3.evaluate((e) => e.textContent)).includes('Correct.'));

// The listings' statuses: LabKit's words from this machine's reports, and "failed on this machine" when a run failed.
await checkLabStatuses(check, 'ch10/10.4', [
  { label: 'CUDA (lab 10.1)', lab: 'ch10/lab1_sum', file: 'kernel.cu' },
  { label: 'Metal (lab 10.1)', lab: 'ch10/lab1_sum', file: 'kernel.metal' },
]);

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
