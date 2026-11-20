// Browser check for section A.6: the four-value stepper and its challenge, the thread-to-data
// stepper (Exercise 4), the half-loop prediction, the summation explorer, the "Kahan's
// correction" scene, the "Kahan, four ways" bug hunt and the section's questions. Every
// number it asserts comes from src/sim/fpsum.ts, imported through the app's module server.
// It passes with or without content/appA/narration/manifest.json: with no audio the scene
// runs on estimated timings and shows each script.
// Usage: APP_URL=http://localhost:5173 node e2e/appA-sa6.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);
const missing = [];
page.on('response', (r) => {
  if (r.status() === 404) missing.push(r.url());
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (title) => `section.bench[aria-label="${title}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const model = (fn, ...args) =>
  page.evaluate(async (src, args) => {
    const m = await import('/src/sim/fpsum.ts');
    return new Function('m', 'args', `return (${src})(m, ...args)`)(m, args);
  }, fn.toString(), args);
const click = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const btn = [...document.querySelectorAll(scope + ' button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(80);
  return done;
};
const select = async (scope, aria, value) => {
  await page.evaluate((scope, aria, value) => {
    const s = document.querySelector(`${scope} select[aria-label="${aria}"]`);
    s.value = String(value);
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, scope, aria, value);
  await pause(120);
};
const slide = async (sel, value) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(150);
};
const question = (t) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(t)), t);
const answerChoice = async (t, index) => {
  const q = await question(t);
  await (await q.$$('input[type=radio]'))[index].click();
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};

await goto(page, 'appA/A.6');
await page.waitForSelector(bench('Four values, two orders'));

// ================================================================ four values, two orders
const F = bench('Four values, two orders');
const toy = await model((m) => [m.toySum(m.BOOK_FOUR, 'sequential'), m.toySum(m.BOOK_FOUR, 'tree'), m.toySum(m.OUR_FOUR, 'sequential'), m.toySum(m.OUR_FOUR, 'tree')].map((r) => ({ rounded: r.adds.map((a) => a.rounded), result: r.result })));
const rounded = (order) => page.$$eval(`${F} [data-order="${order}"] li[data-rounded]`, (ls) => ls.map((l) => Number(l.dataset.rounded)));
check('nothing is shown before the first step', (await rounded('sequential')).length === 0);
for (let i = 0; i < 3; i++) await click(F, 'Step');
check("the book's values in order: 2, 2, 2 (the model's)", JSON.stringify(await rounded('sequential')) === JSON.stringify(toy[0].rounded) && toy[0].result === 2, JSON.stringify(await rounded('sequential')));
check("the book's values as a tree: 2, 0.5, 2.5", JSON.stringify(await rounded('tree')) === JSON.stringify(toy[1].rounded) && toy[1].result === 2.5, JSON.stringify(await rounded('tree')));
check('the summary gives 2.0 against 2.5, exact 2.5', (await text(`${F} [data-result=sequential]`)) === '2' && (await text(`${F} [data-result=tree]`)) === '2.5' && (await text(`${F} [data-summary]`)).includes('Exact sum 2.5'));
check('the lost 0.25 is marked', (await text(`${F} [data-order=sequential]`)).includes('0.25 lost'));
for (let i = 0; i < 3; i++) await click(F, 'Check: is the tree worse?');
check("the book's values do not satisfy the challenge", (await page.$(`${F} [data-challenge=no]`)) !== null);
check('after three tries a "Show one" button appears', await click(F, 'Show one'));
const vals = await page.$$eval(`${F} select`, (ss) => ss.map((s) => Number(s.value)));
check('"Show one" sets our values 0.25, 1, 0.25, 2 and the challenge holds', vals.join() === '0.25,1,0.25,2' && (await page.$(`${F} [data-challenge=yes]`)) !== null, vals.join());
check('ours: 3.5 in order, 3 as a tree (the model\'s)', (await text(`${F} [data-result=sequential]`)) === String(toy[2].result) && (await text(`${F} [data-result=tree]`)) === String(toy[3].result) && toy[3].result === 3);
// by hand: the reader's own values
await click(F, "The book's values");
for (const [i, v] of [[0, 0.25], [1, 1], [2, 0.25], [3, 2]]) await select(F, `value ${'abcd'[i]}`, v);
await click(F, 'Check: is the tree worse?');
check('the same values chosen by hand pass the check', (await page.$(`${F} [data-challenge=yes]`)) !== null);
if (out) await (await page.$(F)).screenshot({ path: out + '-fourvalues.png' });

// ================================================================ thread to data
const T = bench('Thread to data, with rounding');
const trees = await model((m) => {
  const v = m.ex4Order(m.EX4_VALUES.fp16, 'ascending');
  const c = m.treeTrace(v, 'convergent', 'fp16');
  const nb = m.treeTrace(v, 'neighbours', 'fp16');
  const b = m.ex4Order(m.EX4_VALUES.bf16, 'ascending');
  return { input: c.input, c: { result: c.result, error: c.error, s6: c.steps[5] }, nb: { result: nb.result, error: nb.error }, bf: { c: m.treeTrace(b, 'convergent', 'bf16').result, nb: m.treeTrace(b, 'neighbours', 'bf16').result }, bfvals: m.EX4_VALUES.bf16 };
});
const cells = () => page.$$eval(`${T} [data-cell]`, (cs) => cs.map((c) => Number(c.dataset.value)));
check('the stepper starts on the FP16 preset, ascending', JSON.stringify(await cells()) === JSON.stringify(trees.input), JSON.stringify(await cells()));
for (let i = 0; i < 6; i++) await click(T, 'Step');
let n = await text(`${T} .narrate`);
check('step 6: thread 5 adds cell 13 to cell 5 and loses 1 (the model\'s)', n.includes(`thread ${trees.c.s6.thread} adds cell ${trees.c.s6.src} to cell ${trees.c.s6.dst}`) && n.includes('1 lost') && trees.c.s6.lost === 1, n);
check('the cell that lost is marked', (await page.$eval(`${T} [data-cell="5"]`, (c) => c.className)) === 'bad');
const activeAt = () => page.$$eval(`${T} [data-thread]`, (ts) => ts.filter((t) => t.dataset.active === '1').map((t) => Number(t.dataset.thread)));
check('at level 1 all eight threads are active', (await activeAt()).length === 8);
await click(T, 'Finish this level');
await click(T, 'Step');
check('at level 2 only threads 0 to 3 are active (stride 4)', JSON.stringify(await activeAt()) === '[0,1,2,3]', JSON.stringify(await activeAt()));
await click(T, 'Run to the end');
const summary = async () => page.$$eval(`${T} [data-summary] tbody tr`, (trs) => Object.fromEntries(trs.map((tr) => [tr.dataset.pairing, { result: tr.querySelector('[data-k=result]').textContent, error: tr.querySelector('[data-k=error]').textContent }])));
let sm = await summary();
check('convergent loses 8 and neighbours is exact, as the model says', sm.convergent.result === String(trees.c.result) && sm.convergent.error === '8' && sm.neighbours.result === String(trees.nb.result) && sm.neighbours.error === '0', JSON.stringify(sm));
check('"lost so far" at the end is 8', (await page.$eval(`${T} [data-lost-so-far]`, (e) => Number(e.dataset.lostSoFar))) === 8);
if (out) await (await page.$(T)).screenshot({ path: out + '-ex4.png' });
await click(T, 'neighbours (2i, 2i + stride)');
await click(T, 'Step');
n = await text(`${T} .narrate`);
check('neighbours: thread 0 first adds cell 1 to cell 0', n.includes('thread 0 adds cell 1 to cell 0'), n);
check('neighbours: thread labels sit on even cells', JSON.stringify(await page.$$eval(`${T} [data-thread]`, (ts) => ts.map((t) => Number(t.closest('[data-cell]').dataset.cell)))) === '[0,2,4,6,8,10,12,14]');
await click(T, 'BF16');
await click(T, 'Run to the end');
sm = await summary();
check('BF16 preset: 1s, 8s and 256s, convergent below neighbours', (await page.$eval(`${T} input[aria-label=values]`, (e) => e.value)) === trees.bfvals.join(', ') && sm.convergent.result === String(trees.bf.c) && sm.neighbours.result === String(trees.bf.nb) && trees.bf.c < trees.bf.nb, JSON.stringify(sm));
await page.$eval(`${T} input[aria-label=values]`, (el) => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '1, 2, 3');
  el.dispatchEvent(new Event('input', { bubbles: true }));
});
await click(T, 'Use these values');
check('fewer than 16 values are refused with a reason', (await text(T)).includes('Give exactly 16 numbers (you gave 3)'));
await page.$eval(`${T} input[aria-label=values]`, (el) => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1');
  el.dispatchEvent(new Event('input', { bubbles: true }));
});
await click(T, 'Use these values');
check('own values are arranged by the Order menu (ascending)', JSON.stringify(await cells()) === JSON.stringify(Array.from({ length: 16 }, (_, i) => i + 1)));

// ================================================================ the prediction and the explorer
const pq = await question('What does the loop return?');
const pbox = await pq.$('input[aria-label="your prediction"]');
if (pbox) {
  check('the explorer\'s prediction comes first and shows nothing before a commit', !(await page.$('[data-reveal=half-loop]')));
  await pbox.type('30000');
  await (await pq.$('button.primary')).click();
  await pause(250);
}
const halfLoop = await model((m) => m.sumSequential(m.generate('uniform', 65536), 'fp16'));
check('the reveal gives the model\'s FP16 loop result, 2048', (await text('[data-reveal=half-loop]')) === String(halfLoop) && halfLoop === 2048);

const E = bench('Summation order and accumulator width');
const cell = (k) => page.$eval(`${E} [data-k="${k}"]`, (e) => e.dataset.v);
const sweepReady = () => page.waitForSelector(`${E} svg[data-sweep=ready]`, { timeout: 20000 });
const expect = (kind, logn, acc, method, c) => model((m, kind, n, acc, method, c) => {
  const x = m.generate(kind, n);
  const e = m.measure(x, { method, acc, chunk: c });
  return { result: String(e.result), reference: String(e.reference), ulps: String(e.ulps), last: m.runningTrace(x, acc).at(-1).acc };
}, kind, 2 ** logn, acc, method, c);
let want = await expect('uniform', 16, 'fp32', 'sequential', 64);
check('default: uniform, 65,536 values, FP32 loop: result and reference are the model\'s', (await cell('result')) === want.result && (await cell('reference')) === want.reference, `${await cell('result')} vs ${want.result}`);
await click(E, 'FP16');
want = await expect('uniform', 16, 'fp16', 'sequential', 64);
check('FP16: the loop returns 2048, and the running plot ends at 2048', (await cell('result')) === '2048' && want.result === '2048' && (await page.$eval(`${E} svg[data-final-acc]`, (s) => s.dataset.finalAcc)) === '2048');
await click(E, 'BF16');
check('BF16: the loop returns 256', (await cell('result')) === '256');
await click(E, 'FP16');
await slide(`${E} input[aria-label="n (log2)"]`, 18);
await select(E, 'Method', 'tree');
check('FP16 tree over 262,144 uniform values overflows', (await cell('result')) === 'Infinity' && (await text(`${E} [data-k=result]`)).includes('overflow'));
await select(E, 'Method', 'sequential');
check('...while the FP16 loop over the same values returns 2048', (await cell('result')) === '2048');
if (out) await (await page.$(E)).screenshot({ path: out + '-explorer-fp16.png' });
await slide(`${E} input[aria-label="n (log2)"]`, 16);
await click(E, 'FP32');
await select(E, 'Data', 'big-small');
await select(E, 'Method', 'kahan');
want = await expect('big-small', 16, 'fp32', 'kahan', 64);
check('one big, many small: the Kahan loop is the model\'s and within 1 ULP', (await cell('result')) === want.result && Number(await cell('ulps')) <= 1);
await select(E, 'Method', 'sequential');
check('...and the plain loop returns exactly 1', (await cell('result')) === '1');
await select(E, 'Method', 'chunked');
await slide(`${E} input[aria-label="c (log2)"]`, 10);
want = await expect('big-small', 16, 'fp32', 'chunked', 1024);
check('chunked with c = 1024 gives the model\'s result', (await cell('result')) === want.result, `${await cell('result')} vs ${want.result}`);
await sweepReady();
const pts = JSON.parse(await page.$eval(`${E} svg[data-sweep=ready]`, (s) => s.dataset.points));
const wantPts = await model((m) => m.chunkSweep(m.generate('big-small', 65536), 'fp32').map((p) => [p.c, p.plain.ulps, p.kahan.ulps]));
check('the error-against-c plot holds the model\'s sweep, c = 1 to 4096', JSON.stringify(pts) === JSON.stringify(wantPts) && pts.length === 13, JSON.stringify(pts.slice(0, 3)));
check('on this data the plain error grows with c and Kahan stays within 2 ULPs', pts.at(-1)[1] > 100 * pts[2][1] && pts.every((p) => p[2] <= 2));
check('the plot marks the chosen c', (await page.$eval(`${E} svg[data-sweep=ready] line[data-c]`, (l) => Number(l.dataset.c))) === 1024);
await select(E, 'Data', 'mixed-asc');
await select(E, 'Method', 'sequential');
const asc = Number(await cell('ulps'));
await select(E, 'Data', 'mixed-desc');
const desc = Number(await cell('ulps'));
check('mixed magnitudes: ascending beats descending for the loop', asc * 5 < desc, `${asc} vs ${desc}`);
if (out) await (await page.$(E)).screenshot({ path: out + '-explorer.png' });

// ================================================================ the scene
const S = 'section.scene[aria-label="Kahan\'s correction"]';
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const scripts = [];
for (let k = 1; k <= 6; k++) scripts.push(strip(await (await fetch(`${BASE}/content/appA/narration/kahan-${k}.txt`)).text()));
const words = scripts.map((s) => s.split(' ').length);
check('six scripts, each 35 to 60 words', words.length === 6 && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((s) => !/[<>=×%/_(){}+−#*^]/.test(s)), scripts.find((s) => /[<>=×%/_(){}+−#*^]/.test(s)));
const res = await fetch(BASE + '/content/appA/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
const labels = await page.$$eval(`${S} .seg button`, (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered', labels.length === 6, labels.join(' | '));
const capBox = () => page.evaluateHandle((S) => [...document.querySelectorAll(S + ' label.check')].find((l) => l.textContent.includes('captions')).querySelector('input'), S);
check('the captions switch is off by default', !(await (await capBox()).evaluate((e) => e.checked)));
if (!manifest?.clips?.['kahan-1']) check('a part with no audio shows its script', strip(await text(`${S} .caption`)) === scripts[0]);
const seekScene = (v) => slide(`${S} input[type=range]`, v);
const reg = (name) => page.$eval(`${S} [data-part="reg-${name}"]`, (g) => g.dataset.value).catch(() => null);
const sc = await model((m) => {
  const t = m.kahanTrace(m.sceneValues(), 'fp32');
  const k = (v) => Math.round(m.units25(v));
  return { c1: k(t[1].c), c2: k(t[2].c), c3: k(t[3].c), t3: k(t[3].t - 1), last: { sum: k(t.at(-1).sum - 1), plain: k(t.at(-1).plain - 1), exact: k(t.at(-1).exact - 1) } };
});
await click(S, '1. The plain sum stalls');
await seekScene(1000);
check('part 1: after 8 values the plain sum is still 1 and the exact mark is 8 steps of 2⁻²⁵ up', (await page.$eval(`${S} [data-part=plain]`, (g) => g.dataset.units)) === '0' && (await page.$eval(`${S} [data-part=exact]`, (g) => g.dataset.units)) === '8');
await click(S, '2. A second register');
await seekScene(1000);
check('part 2: c is 0, y is 2⁻²⁵, t rounds back to 1', (await reg('c')) === '0' && (await reg('y')) === '2⁻²⁵' && (await reg('t')) === '1');
await click(S, '3. Catch what was lost');
await seekScene(1000);
check(`part 3: c becomes minus 2⁻²⁵ (model: ${sc.c1})`, (await reg('c')) === '−2⁻²⁵' && sc.c1 === -1);
await click(S, '4. Put it back');
await seekScene(450);
check(`part 4, value 2: c is minus 2 × 2⁻²⁵ and t is still 1 (model: ${sc.c2})`, (await reg('c')) === '−2 × 2⁻²⁵' && (await reg('t')) === '1' && sc.c2 === -2);
await seekScene(1000);
check(`part 4, value 3: t moves up one ULP and c records the overshoot (model: ${sc.t3}, ${sc.c3})`, (await reg('t')) === '1 + 4 × 2⁻²⁵' && (await reg('c')) === '2⁻²⁵' && sc.t3 === 4 && sc.c3 === 1);
await click(S, '5. All 4,096 values');
await seekScene(1000);
const plot = await page.$eval(`${S} [data-part=plot]`, (g) => ({ kahan: Number(g.dataset.kahan), plain: Number(g.dataset.plain), exact: Number(g.dataset.exact) }));
check('part 5: Kahan ends on the exact total, the plain sum at 1 (the model\'s trace)', plot.kahan === sc.last.sum && plot.exact === sc.last.exact && plot.plain === sc.last.plain && plot.kahan === 4096 && plot.plain === 0, JSON.stringify(plot));
check('part 5: the headline gives 1024 ULPs against 0', (await text(`${S} [data-part=headline]`)).includes('1024 ULPs short; Kahan 0 ULPs off'));
if (out) await (await page.$(S)).screenshot({ path: out + '-scene-part5.png' });
await click(S, '6. Keep the compiler out');
await seekScene(1000);
const be = await text(`${S} [data-part=backends]`);
check('part 6: one way to keep c on each backend', be.includes('__fadd_rn') && be.includes('-ffast-math') && be.includes('#pragma METAL fp math_mode(safe)'), be);
check('part 6 ends on lab A.1', (await text(`${S} [data-part=headline]`)).includes('lab A.1'));
// it plays with the sound off and moves on by itself
await page.evaluate((S) => [...document.querySelectorAll(S + ' label.check')].find((l) => l.textContent.includes('sound')).querySelector('input').click(), S);
await click(S, '1. The plain sum stalls');
await seekScene(985);
check('the main button offers to resume part-way through a part', (await text(`${S} button.primary`)) === 'Resume');
await page.$eval(`${S} button.primary`, (b) => b.click());
await pause(2000);
check('with sound off, part 1 runs out and part 2 starts by itself', (await page.$$eval(`${S} .seg button`, (bs) => bs.findIndex((b) => b.getAttribute('aria-pressed') === 'true'))) === 1);
await page.$eval(`${S} button.primary`, (b) => b.click());
check('pause stops it', (await text(`${S} button.primary`)) === 'Resume');

// ================================================================ the bug hunt
const B = bench('Kahan, four ways');
await page.waitForFunction((B) => document.querySelectorAll(B + ' .code').length === 4, {}, B);
const codes = await page.$$eval(`${B} [data-version]`, (ps) => ps.map((p) => p.querySelector('.code pre').textContent));
check('each version shows its own function from kahan.h', codes.every((c, i) => c.includes(`version${i + 1}(`)) && codes[0].includes('T y = x[i] + c;') && codes[2].includes('sum = t;\n        c = (t - sum) - y;') && codes[3].includes('c = T(0);'), codes[2]);
check('running is disabled until all four are marked', (await page.$eval(`${B} button.primary`, (b) => b.disabled)) === true);
const hunt = await model((m) => m.bugHunt(m.bugHuntValues(), 'fp32').runs.map((r) => r.verdict));
const mark = async (v, verdict) => {
  const idx = ['kahan', 'plain', 'worse'].indexOf(verdict);
  await page.evaluate((B, v, idx) => document.querySelectorAll(`${B} input[name="kahan-v${v}"]`)[idx].click(), B, v, idx);
};
for (const v of [1, 2, 3, 4]) await mark(v, 'kahan');
await click(B, 'Run the four versions');
let verdicts = await page.$$eval(`${B} [data-verdict]`, (ps) => ps.map((p) => p.dataset.verdict));
check('the verdicts are the model\'s: plain, Kahan, worse, plain', JSON.stringify(verdicts) === JSON.stringify(hunt) && hunt.join() === 'plain,kahan,worse,plain', JSON.stringify(verdicts));
check('marking all four "like Kahan" scores 1 of 4', (await page.$eval(`${B} [data-score]`, (e) => e.dataset.score)) === '1');
for (const [v, verdict] of hunt.entries()) await mark(v + 1, verdict);
await click(B, 'Run the four versions');
check('the right marks score 4 of 4', (await page.$eval(`${B} [data-score]`, (e) => e.dataset.score)) === '4' && (await page.$$(`${B} [data-mark=right]`)).length === 4);
if (out) await (await page.$(B)).screenshot({ path: out + '-bughunt.png' });

// ================================================================ questions
let q = await answerChoice('always sorted from smallest to largest', 0);
check('q-ex4: the convergent pairing is the answer', q.includes('Correct.'));
q = await answerChoice('How should you sort it', 1);
check('q-sign: sorting by value is wrong, with the reason', q.includes('Not quite.') && q.includes('largest negative values come first'));
q = await answerChoice('262,144 values', 0);
check('q-half-tree: loop 2048, tree infinity', q.includes('Correct.'));
q = await answerChoice('which is more accurate, Kahan', 0);
check('q-kahan16: the plain FP32 loop', q.includes('Correct.'));
q = await answerChoice('but on Metal its result', 0);
check('q-fastmath: reassociation removed c', q.includes('Correct.') && q.includes('#pragma METAL fp math_mode(safe)'));
q = await answerChoice('coarsening factor', 0);
check('q-coarse: the error grows with c; Kahan brings it back', q.includes('Correct.'));

// ================================================================ notes, narrow, errors
const notes = await page.$$eval('aside.note', (as) => as.map((a) => a.textContent));
check('Book notes cite pp. 593, 594 and 600', ['593', '594', '600'].every((p) => notes.some((n) => n.startsWith('Book') && n.includes(p))));
check('Today notes cite the facts items D1 to D3 and A13, B5, C2, C3', notes.some((n) => n.includes('D1, D2, D3')) && notes.some((n) => n.includes('A13, B5, C2, C3')));
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
const fits = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
check('at 390 px the page does not scroll sideways', fits);
if (out) {
  await page.$eval(T, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-ex4.png' });
  await page.$eval(E, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-explorer.png' });
  await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-bughunt.png' });
}

const expected404 = (u) => /\/content\/appA\/narration\/(manifest\.json|kahan-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
