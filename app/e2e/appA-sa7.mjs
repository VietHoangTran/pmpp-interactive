// Browser check for appendix A, section A.7: the narrated scene "One thread per row", the two
// elimination steppers (the book's systems, the 2 × 2 tiny-pivot system, every arithmetic,
// pivoting, both views), the Fig. A.11 erratum hunt and the questions. Every number it
// expects comes from the model the page runs on (src/sim/gauss.ts), imported here directly.
// Usage: node e2e/appA-sa7.mjs [screenshot-prefix]
import { registerHooks } from 'node:module';
import { open, goto, BASE } from './browser.mjs';

// The models import their siblings without a file extension (as the app's bundler allows).
registerHooks({
  resolve(spec, ctx, next) {
    try {
      return next(spec, ctx);
    } catch (e) {
      if (spec.startsWith('.') && !/\.\w+$/.test(spec)) return next(spec + '.ts', ctx);
      throw e;
    }
  },
});
const { bookPanels, busyFraction, diffPanels, equation, printedA11, system, trace } = await import('../src/sim/gauss.ts');

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
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
const B = (t) => `section.bench[aria-label="${t}"]`;
const S1 = B('Elimination, one phase at a time');
const S2 = B('Pivot or fail');
const E = B('Fig. A.11 as printed');
const SC = 'section.scene[aria-label="One thread per row"]';
const minus = (s) => s.replace(/-/g, '−');
const clickText = async (scope, label) => {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelector(scope).querySelectorAll('button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(80);
  return ok;
};
const setSelect = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, v);
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(120);
};
const setRange = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(120);
};
const setPivot = async (scope, on) => {
  await page.evaluate((scope, on) => {
    const i = [...document.querySelector(scope).querySelectorAll('label.check')].find((l) => l.textContent.includes('pivoting')).querySelector('input');
    if (i.checked !== on) i.click();
  }, scope, on);
  await pause(120);
};
const state = (scope) =>
  page.$eval(scope, (b) => ({
    note: b.querySelector('.narrate').textContent.replace(/\s+/g, ' ').trim(),
    bad: b.querySelector('.narrate').classList.contains('bad'),
    phase: b.querySelector('.narrate').dataset.phase,
    threads: [...b.querySelectorAll('td[data-state]')].map((t) => t.dataset.state),
    solution: b.querySelector('[data-solution]')?.dataset.solution ?? null,
    chip: b.querySelector('[data-solution] .chip')?.textContent ?? null,
    busy: b.querySelector('[data-busy]').textContent,
    barrier: !!b.querySelector('[data-barrier]'),
  }));
const cells = (scope) => page.$$eval(scope + ' [data-cell]', (c) => c.map((x) => ({ at: x.dataset.cell, text: x.textContent.trim(), bad: x.classList.contains('bad') })));

await goto(page, 'appA/A.7');
await page.waitForSelector(S1);

// ---------------------------------------------------------------- stepper 1: the book's first system
const t1 = trace(system('book1'), { pivot: false, arith: 'exact' });
let st = await state(S1);
check('stepper 1 starts at the book\'s first system, nothing run', st.phase === 'start' && st.threads.join() === 'ready,ready,ready' && st.note === t1.phases[0].note, JSON.stringify(st));
await clickText(S1, 'Step');
st = await state(S1);
check('phase 1 is the division, with the model\'s note and thread states', st.phase === 'divide' && st.note === t1.phases[1].note && st.threads.join() === t1.phases[1].threads.join(), JSON.stringify(st));
let c = await cells(S1);
check('after it, row 1 reads 1 5/3 2/3 19/3 (Fig. A.10, step 1)', c.slice(0, 4).map((x) => x.text).join(' ') === '1 5/3 2/3 19/3', JSON.stringify(c.slice(0, 4)));
await clickText(S1, 'Step');
st = await state(S1);
check('then a barrier, drawn across the threads', st.phase === 'barrier' && st.barrier && st.threads.every((s) => s === 'waiting'), JSON.stringify(st));
await clickText(S1, 'Step');
st = await state(S1);
check('then the subtraction: thread 0 idle, threads 1 and 2 working', st.phase === 'subtract' && st.threads.join() === 'idle,working,working', JSON.stringify(st));
await clickText(S1, 'equations');
const eq = await page.$$eval(S1 + ' [data-eq]', (e) => e.map((x) => x.textContent));
check('equation view shows Fig. A.9\'s step 2', eq.join(' | ') === t1.phases[3].m.map((r) => equation(r.map((x) => x.text), ['X', 'Y', 'Z'])).join(' | ') && eq[1] === '−1/6Y − 1/6Z = −5/6', eq.join(' | '));
await clickText(S1, 'matrix');
await clickText(S1, 'Run to the end');
st = await state(S1);
check('run to the end: X = 1, Y = 2, Z = 3, exact', st.solution === '1,2,3' && st.chip === 'exact solution', JSON.stringify(st));
check('busy thread slots: 60%, n / (2n − 1) for n = 3', st.busy.includes(`${Math.round(busyFraction(t1) * 100)}%`) && Math.round(busyFraction(t1) * 100) === 60, st.busy);
const lanes = await page.$$eval(S1 + ' svg [data-kind]', (g) => g.map((x) => x.dataset.kind));
check('the timeline has one column per phase, barriers included', lanes.length === t1.phases.length && lanes.filter((k) => k === 'barrier').length === t1.phases.filter((p) => p.kind === 'barrier').length, lanes.join(','));
if (out) await page.$eval(S1, (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-stepper-end.png' });
for (const a of ['fp32', 'fp16']) {
  await clickText(S1, a === 'fp32' ? 'FP32' : 'FP16');
  await clickText(S1, 'Run to the end');
  st = await state(S1);
  const t = trace(system('book1'), { pivot: false, arith: a });
  check(`${a.toUpperCase()}: the answer drifts as the model computes it`, st.solution === t.solution.map((x) => x.text).join(',') && st.solution !== '1,2,3' && st.chip.startsWith('close: largest error'), JSON.stringify(st));
}

// ---------------------------------------------------------------- stepper 2: pivot or fail
const tiny = (k, pivot, arith) => trace(system('tiny', k), { pivot, arith });
await clickText(S2, 'Run to the end');
st = await state(S2);
check('q-tiny: FP32, ε = 1e-8, no pivoting: x1 = 0, marked wrong', st.solution === tiny(8, false, 'fp32').solution.map((x) => x.text).join(',') && st.solution.startsWith('0,') && st.chip === 'wrong answer', JSON.stringify(st));
await setPivot(S2, true);
await clickText(S2, 'Run to the end');
st = await state(S2);
check('with pivoting: x1 = 1', st.solution === '1,1' && st.chip.startsWith('close'), JSON.stringify(st));
await setPivot(S2, false);
await clickText(S2, 'exact fractions');
await clickText(S2, 'Run to the end');
st = await state(S2);
check('exact fractions without pivoting: x1 = 10^8 / (10^8 − 1)', st.solution === tiny(8, false, 'exact').solution.map((x) => x.text).join(',') && st.chip === 'exact solution', JSON.stringify(st));
await clickText(S2, 'FP16');
await setRange(S2 + ' input[aria-label="epsilon exponent"]', 5);
await setPivot(S2, true);
await clickText(S2, 'Run to the end');
st = await state(S2);
check('FP16, ε = 1e-5, even with pivoting: 1/ε overflows and the answer is NaN', st.solution === tiny(5, true, 'fp16').solution.map((x) => x.text).join(',') && st.solution.includes('NaN') && st.chip === 'no answer: NaN or infinity', JSON.stringify(st));
await setRange(S2 + ' input[aria-label="epsilon exponent"]', 4);
await clickText(S2, 'Run to the end');
st = await state(S2);
check('FP16, ε = 1e-4 with pivoting: right', st.solution === tiny(4, true, 'fp16').solution.map((x) => x.text).join(',') && st.solution === '1,1', JSON.stringify(st));

// The book's second system, no pivoting: a zero divisor.
const t2 = trace(system('book2'), { pivot: false, arith: 'exact' });
await setSelect(S2 + ' select[aria-label="system"]', 'book2');
await setPivot(S2, false);
await clickText(S2, 'exact fractions');
await clickText(S2, 'Step');
st = await state(S2);
c = await cells(S2);
check('second system, step 1: row 1 divided by 0 is NaN, inf, inf, inf, shown as failures', st.bad && c.slice(0, 4).map((x) => x.text).join(' ') === 'NaN inf inf inf' && c.slice(0, 4).every((x) => x.bad) && st.note === t2.phases[1].note, JSON.stringify(c.slice(0, 4)));
await clickText(S2, 'Step');
await clickText(S2, 'Step');
c = await cells(S2);
check('after the subtraction every row holds a NaN (the model\'s matrix)', [0, 1, 2].every((r) => c.find((x) => x.at === `${r},0`).text === 'NaN') && c.map((x) => x.text).join(' ') === t2.phases[3].m.flat().map((x) => minus(x.text)).join(' '), c.map((x) => x.text).join(' '));
if (out) await page.$eval(S2, (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-nan.png' });
const t3 = trace(system('book2'), { pivot: true, arith: 'exact' });
await setPivot(S2, true);
await clickText(S2, 'Step');
st = await state(S2);
check('with pivoting the first phase is a max reduction down column X: |0| |2| |1| → |2| |1| → |2|', st.phase === 'pivot' && st.note.includes('Max reduction: |0| |2| |1| → |2| |1| → |2|') && st.threads.join() === 'searching,searching,searching', st.note);
await clickText(S2, 'Step');
st = await state(S2);
check('then row 2 swaps to the top', st.phase === 'swap' && st.threads.join() === t3.phases[2].threads.join() && (await cells(S2)).slice(0, 4).map((x) => x.text).join(' ') === '2 3 1 11', JSON.stringify(st));
await clickText(S2, 'Step');
await clickText(S2, 'Step');
st = await state(S2);
check('step 1 after the swap: row 2 lacks X and its thread skips (divergence)', st.phase === 'divide' && st.threads.join() === 'working,skipping,working', JSON.stringify(st));
await clickText(S2, 'Run to the end');
st = await state(S2);
check('and pivoting finds (1, 2, 3)', st.solution === '1,2,3', JSON.stringify(st));

// ---------------------------------------------------------------- the erratum hunt
const diffs = diffPanels(printedA11, bookPanels('A.11'));
const wrongKeys = diffs.map((d) => `${printedA11.findIndex((p) => p.label === d.panel)}:${d.what === 'caption' ? 'cap' : `${d.row}:${d.col}`}`);
check('the model finds 7 differences: step 1\'s caption and row 1 of steps 5 and 6', wrongKeys.join(' ') === '2:cap 6:0:1 6:0:2 6:0:3 7:0:1 7:0:2 7:0:3', wrongKeys.join(' '));
const clickKey = (k) => page.$eval(`${E} [data-key="${k}"]`, (e) => e.click());
for (const k of ['2:cap', '6:0:1', '6:0:2', '6:0:3', '7:0:1', '7:0:2', '3:1:1']) await clickKey(k);
await pause(150);
check('seven marks recorded', (await page.$eval(E + ' [data-marks]', (e) => e.dataset.marks)) === '7');
await clickText(E, 'Check my marks');
const result = await page.$eval(E + ' [data-result]', (e) => e.dataset.result);
check('check: 6 of 7 found, 1 false alarm', result === '6,7,1', result);
const badCells = await page.$$eval(E + ' .mx span.bad', (s) => s.map((x) => x.dataset.key));
check('all six wrong numbers are marked as wrong, the missed one included', badCells.join(' ') === wrongKeys.slice(1).join(' '), badCells.join(' '));
check('the caption is marked as found', (await page.$eval(`${E} [data-key="2:cap"]`, (e) => e.dataset.verdict)) === 'found');
await page.evaluate((E) => [...document.querySelector(E).querySelectorAll('label.check')][0].querySelector('input').click(), E);
await pause(150);
const fixed = await page.$eval(`${E} [data-key="6:0:1"]`, (e) => ({ text: e.textContent, good: e.classList.contains('good') }));
const cap = await page.$eval(`${E} [data-key="2:cap"]`, (e) => e.textContent);
check('the computed figure shows 3/2 in step 5 and "divide row 1 by 2"', fixed.text === '3/2' && fixed.good && cap === 'divide row 1 by 2', JSON.stringify([fixed, cap]));
if (out) await page.$eval(E, (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-erratum.png' });

// ---------------------------------------------------------------- the scene
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const res = await fetch(BASE + '/content/appA/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= 6; n++) scripts.push(strip(await (await fetch(`${BASE}/content/appA/narration/gauss-threads-${n}.txt`)).text()));
check('six scripts of 35 to 60 words', scripts.every((s) => { const w = s.split(' ').length; return w >= 35 && w <= 60; }), scripts.map((s) => s.split(' ').length).join(' '));
await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' }));
const box = (word) => page.evaluateHandle((sel, word) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes(word)).querySelector('input'), SC, word);
const setBox = async (word, on) => {
  const b = await box(word);
  if ((await b.evaluate((e) => e.checked)) !== on) await b.click();
  await pause(120);
};
const press = async (label) => {
  await page.evaluate((sel, label) => [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim().includes(label)).click(), SC, label);
  await pause(150);
};
const seek = (v) => setRange(SC + ' input[type=range]', v);
const drawn = () => page.$eval(SC + ' .stage svg', (s) => ({ ...s.dataset, text: [...s.querySelectorAll('text')].map((t) => t.textContent) }));
const labels = await page.$$eval(SC + ' .seg button', (b) => b.map((x) => x.textContent.trim()));
check('six parts with their names', labels.join('|') === '1. One row each|2. Divide, barrier|3. Subtract, barrier|4. The rest|5. A zero pivot|6. Pivoting', labels.join('|'));
check('captions switch, off by default', !(await (await box('captions')).evaluate((e) => e.checked)));
const caption = () => page.$eval(SC + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim());
if (!manifest?.clips?.['gauss-threads-1']) check('with no audio, part 1 shows its script anyway', (await caption()) === scripts[0]);
await setBox('captions', true);
await setBox('sound', false);
const T1 = trace(system('book1'), { pivot: false, arith: 'exact' });
const T2 = trace(system('book2'), { pivot: false, arith: 'exact' });
const T3 = trace(system('book2'), { pivot: true, arith: 'exact' });
const rhs = (t, i) => t.phases[i].m.map((r) => r[r.length - 1].text).join(',');
const nan = (t, i) => t.phases[i].m.flat().filter((x) => Number.isNaN(x.v)).length;
let d = await drawn();
check('part 1: the book\'s matrix, thread 0 holds 3 5 2 19', d.beat === '1' && d.phase === 'start' && d.rhs === '19,11,11' && d.text.includes('thread 0'), JSON.stringify(d));
await press('Play');
await pause(1500);
check('with sound off it plays on estimated timings', (await page.$eval(SC + ' input[type=range]', (e) => Number(e.value))) > 20);
await press('Pause');
await press('2. Divide, barrier');
check('jumping to part 2 shows its script', (await caption()) === scripts[1]);
await seek(1000);
d = await drawn();
check('part 2 ends on the barrier after the division (the model\'s phase 2)', d.phase === 'barrier' && d.index === '2' && d.rhs === rhs(T1, 2), JSON.stringify(d));
await press('3. Subtract, barrier');
await seek(1000);
d = await drawn();
check('part 3 ends after the subtraction, rows 2 and 3 without X', d.index === '4' && d.rhs === rhs(T1, 4) && d.rhs === '19/3,-5/6,14/3', JSON.stringify(d));
await press('4. The rest');
await seek(1000);
d = await drawn();
check('part 4 ends with X = 1, Y = 2, Z = 3', d.phase === 'done' && d.rhs === '1,2,3' && d.text.some((x) => x.includes('X = 1, Y = 2, Z = 3')), JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-scene-4.png' });
await press('5. A zero pivot');
await seek(1000);
d = await drawn();
check('part 5: the second system without pivoting; NaN in every row after the subtraction', d.system === 'book2' && d.pivot === '0' && d.index === '4' && Number(d.nan) === nan(T2, 4) && nan(T2, 4) >= 3, JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-scene-5.png' });
await press('6. Pivoting');
await seek(100);
d = await drawn();
const idx6 = Math.min(T3.phases.length - 1, 1 + Math.floor(0.1 * (T3.phases.length - 1)));
check('part 6 early: the phase the model has at that point', d.pivot === '1' && d.phase === T3.phases[idx6].kind, JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 6 ends: with pivoting, the same (1, 2, 3)', d.phase === 'done' && d.rhs === '1,2,3' && d.pivot === '1', JSON.stringify(d));
await setBox('captions', false);

// ---------------------------------------------------------------- questions
const question = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const answerChoice = async (needle, idx) => {
  const q = await question(needle);
  await (await q.$$('input[type=radio]'))[idx].click();
  await (await q.$('button.primary')).click();
  await pause(150);
  return (await q.evaluate((e) => e.textContent)).includes('Correct.');
};
const answerNum = async (needle, v) => {
  const q = await question(needle);
  await (await q.$('input[type=number]')).type(String(v));
  await (await q.$('button.primary')).click();
  await pause(150);
  return (await q.evaluate((e) => e.textContent)).includes('Correct.');
};
check('q-idle: 50% is accepted', await answerNum('what percentage of thread slots', 50));
check('q-tiny: 0 is right', await answerChoice('What does elimination give for x1', 0));
check('q-a11: 2 is right', await answerChoice('What should the divisor be', 1));
check('q-pivot-cost: a max seen by every thread is right', await answerChoice('cheap inside one block', 0));
check('q-partial: row interchanges within the column is right', await answerChoice('LU with partial pivoting', 0));
check('answers were saved through PUT /api/progress (intercepted)', puts > 0, String(puts));

// ---------------------------------------------------------------- dark theme, narrow screen
await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
await pause(250);
const hard = await page.$$eval(`${SC} svg [fill], ${SC} svg [stroke], ${S1} svg [fill], ${S1} svg [stroke]`, (els) =>
  els.flatMap((e) => ['fill', 'stroke'].map((a) => e.getAttribute(a))).filter((v) => v && v !== 'none' && !v.startsWith('var(')),
);
check('drawings use theme variables only', hard.length === 0, hard.join(','));
if (out) await page.$eval(S2, (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-dark.png' });
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
const narrow = await page.evaluate(() => ({
  page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1,
  benches: [...document.querySelectorAll('.bench > .body')].filter((b) => b.scrollWidth > b.clientWidth + 1).map((b) => b.parentElement.getAttribute('aria-label')),
}));
check('at 390 px the page does not scroll sideways and no bench scrolls inside itself', narrow.page && narrow.benches.length === 0, JSON.stringify(narrow));
if (out) {
  await page.$eval(S1, (e) => e.scrollIntoView());
  await page.screenshot({ path: out + '-narrow-stepper.png' });
  await page.$eval(E, (e) => e.scrollIntoView());
  await page.screenshot({ path: out + '-narrow-erratum.png' });
}

const expected404 = (u) => /\/content\/appA\/narration\/(manifest\.json|gauss-threads-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
