// Browser check for section 10.2: the tree builder (every operator and order, the
// book's two trees, the counters), the prediction and the float32 orders explorer, the
// tie explorer, and the questions. Expected values come from src/sim/reducetree.ts,
// imported here directly. The narrated scene has its own check, ch10-scene-tree.mjs.
// Usage: node e2e/ch10-s102.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import {
  FIG103A,
  FIG103B,
  SHAPES,
  builderResult,
  builderTree,
  compareOrders,
  f32hex,
  generate,
  intTree,
  surveyOrders,
  tieRow,
  treeStats,
  winners,
} from '../src/sim/reducetree.ts';

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
const bench = (t) => `section.bench[aria-label="${t}"]`;
const TB = bench('Build a reduction tree');
const OB = bench('Three orders, one float32 sum');
const WB = bench('Who wins a tie');
const press = async (scope, group, label) => {
  const ok = await page.evaluate((scope, group, label) => {
    const root = document.querySelector(scope);
    const where = group ? root.querySelector(`[aria-label="${group}"]`) : root;
    const btn = [...where.querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, group, label);
  await pause(80);
  return ok;
};
const setInput = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(120);
};
const minus = (s) => s.replace(/^-/, '−');

// ---------------------------------------------------------------- the tree builder
const builder = () => page.evaluate((TB) => {
  const root = document.querySelector(TB);
  const levels = [];
  for (const n of root.querySelectorAll('svg[data-part="builder-tree"] g[data-part="node"]')) (levels[Number(n.dataset.level)] ??= []).push(n.dataset.value);
  const res = root.querySelector('[data-readout="result"]');
  const c = (k) => Number(root.querySelector(`[data-counter="${k}"]`).textContent);
  const st = (k) => root.querySelector(`[data-stat="${k}"]`).textContent;
  return {
    levels,
    expr: root.querySelector('[data-part="expr"]').textContent,
    mine: res.querySelector('[data-value="mine"]').textContent,
    loop: res.querySelector('[data-value="loop"]').textContent,
    same: res.dataset.same,
    chip: res.querySelector('.chip').textContent,
    work: c('work'),
    span: c('span'),
    peak: c('peak'),
    bars: root.querySelector('[data-readout="counters"] [data-part="bars"]').dataset.ops,
    stats: { n: st('n'), work: st('work'), span: st('span'), peak: st('peak'), average: st('average') },
    scaleBars: root.querySelector('[data-readout="scale"] [data-part="bars"]').dataset.ops,
    // Leaf labels in the order they are drawn, left to right.
    under: [...root.querySelectorAll('svg[data-part="builder-tree"] text[data-part="under"]')]
      .sort((a, b) => Number(a.getAttribute('x')) - Number(b.getAttribute('x')))
      .map((t) => t.textContent),
  };
}, TB);
const valuesOf = (t) => t.levels.map((l) => l.map((n) => n.value));

await goto(page, 'ch10/10.2');
await page.waitForSelector(TB);

let b = await builder();
const fig3b = builderTree(FIG103B, 'sum', 'neighbours');
check('the builder starts on Fig. 10.3(b): 11 5 13 15, 16 28, 44', JSON.stringify(b.levels) === JSON.stringify(valuesOf(fig3b)) && b.mine === '44', JSON.stringify(b.levels));
check('with its expression, the loop\'s answer and a "same" chip', b.expr === '((4 + 7) + (2 + 3)) + ((8 + 5) + (9 + 6))' && b.loop === '44' && b.same === '1' && b.chip === 'same as the loop', b.expr);
check('counters: work 7, span 3, at most 4 at once; bars 4 2 1', b.work === 7 && b.span === 3 && b.peak === 4 && b.bars === '4 2 1', JSON.stringify(b));
check('every leaf is labelled with its index', b.under.join(' ') === 'x0 x1 x2 x3 x4 x5 x6 x7', b.under.join(' '));
if (out) await page.screenshot({ path: out + '-builder.png' });

await press(TB, 'Book examples', 'Fig. 10.3(a), max');
b = await builder();
check('Fig. 10.3(a): 3 7 4 6, 7 6, 7, and the parenthesisation printed on p. 225', JSON.stringify(b.levels.slice(1)) === JSON.stringify([['3', '7', '4', '6'], ['7', '6'], ['7']]) && b.expr === intTree(FIG103A, 'max', 'neighbours').expr && b.expr === '((3 max 1) max (7 max 0)) max ((4 max 1) max (6 max 3))', b.expr);
await press(TB, 'Order', 'the loop');
b = await builder();
check('the loop order: the other parenthesisation of p. 225, 7 steps of one operation', b.expr === '((((((3 max 1) max 7) max 0) max 4) max 1) max 6) max 3' && b.span === 7 && b.work === 7 && b.peak === 1 && b.bars === '1 1 1 1 1 1 1', b.expr);

// Every operator under every order, against the model.
await press(TB, 'Book examples', 'Fig. 10.3(b), sum');
const agree = [];
const differs = [];
for (const op of [['sum', 'sum'], ['max', 'max'], ['min', 'min'], ['product', 'product'], ['subtraction', 'sub'], ['join', 'join']]) {
  await press(TB, 'Operator', op[0]);
  for (const s of SHAPES) {
    await press(TB, 'Order', s === 'loop' ? 'the loop' : s);
    b = await builder();
    const want = minus(builderTree(FIG103B, op[1], s).root.value);
    const wantSame = builderResult(FIG103B, op[1], s) === builderResult(FIG103B, op[1], 'loop');
    if (b.mine !== want || (b.same === '1') !== wantSame) differs.push(`${op[0]}/${s}: ${b.mine} vs ${want}`);
    if (b.same === '1') agree.push(`${op[0]}/${s}`);
  }
}
check('for every operator and order, the result and the chip are the model\'s', differs.length === 0, differs.join(' | '));
check('sum, max, min and product agree with the loop in every order', ['sum', 'max', 'min', 'product'].every((o) => SHAPES.every((s) => agree.includes(`${o}/${s === 'loop' ? 'loop' : s}`))), agree.join(' '));
check('subtraction differs from the loop under both tree orders', !agree.includes('subtraction/neighbours') && !agree.includes('subtraction/halves'));
check('join agrees under neighbours and not under halves', agree.includes('join/neighbours') && !agree.includes('join/halves'));
// The current state is join / halves.
b = await builder();
const operandOrder = builderTree(FIG103B, 'join', 'halves').root.leaves.map((k) => 'x' + k).join(' ');
check('halves on join: 4 8 2 9 7 5 3 6, leaves drawn in combination order x0 x4 x2 x6 x1 x5 x3 x7', b.mine === '48297536' && b.loop === '47238596' && b.chip === 'differs from the loop' && b.under.join(' ') === operandOrder && operandOrder === 'x0 x4 x2 x6 x1 x5 x3 x7', JSON.stringify(b.under));
check('halves pairs x0 with x4, x1 with x5 ...: the first level reads 48 75 29 36', JSON.stringify(b.levels[1]) === JSON.stringify(['48', '75', '29', '36']), JSON.stringify(b.levels[1]));
if (out) await page.screenshot({ path: out + '-builder-join.png' });

// Sizes and editing.
await press(TB, 'Operator', 'sum');
await press(TB, 'Order', 'neighbours');
await press(TB, 'Leaves', '16');
b = await builder();
const pool = [...FIG103B, ...FIG103A];
check('16 leaves: work 15, span 4, 8 at once', b.work === 15 && b.span === 4 && b.peak === 8 && b.bars === '8 4 2 1' && b.mine === builderTree(pool, 'sum', 'neighbours').root.value, JSON.stringify([b.work, b.span, b.mine]));
await press(TB, 'Leaves', '4');
b = await builder();
check('4 leaves: work 3, span 2', b.work === 3 && b.span === 2 && b.mine === '16', JSON.stringify([b.work, b.span, b.mine]));
await setInput(TB + ' input[aria-label="leaf 0"]', 10);
b = await builder();
check('editing a leaf recomputes the tree: 10 + 7 + 2 + 3 = 22', b.mine === '22' && b.levels[0][0] === '10', JSON.stringify(b.levels));

// The counters slider.
let st = treeStats(1024);
check('the counters start at N = 1024: span 10, peak 512, average 102.3', b.stats.n === '1024' && b.stats.span === '10' && b.stats.peak === '512' && b.stats.average === '102.3' && b.stats.work === '1023' && b.scaleBars === st.opsPerStep.join(' '), JSON.stringify(b.stats));
await setInput(TB + ' input[aria-label="N for the counters"]', 3);
b = await builder();
st = treeStats(8);
check('N = 8: span 3, peak 4, average 2.33', b.stats.n === '8' && b.stats.span === '3' && b.stats.peak === '4' && b.stats.average === String(Number(st.average.toFixed(2))) && b.scaleBars === '4 2 1', JSON.stringify(b.stats));

// ---------------------------------------------------------------- the prediction and the float32 orders
const D = compareOrders(generate(1024, 'softmax', 1));
const DS = surveyOrders(1024, 'softmax', 100);
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Predict first') && e.textContent.includes('softmax denominator')));
await (await pred.$('input[type=number]')).type('1');
await (await pred.$('button.primary')).click();
await pause(150);
let t = await pred.evaluate((e) => e.textContent);
check('committing a prediction reveals the model\'s count and the survey', t.includes('You predicted 1') && t.includes(`${D.distinct} on the first data set below`) && t.includes(`at least two orders disagree in ${DS.disagree}`), t);

const orders = () => page.evaluate((OB) => {
  const root = document.querySelector(OB);
  return {
    n: root.querySelector('[data-value="n"]').textContent,
    rows: Object.fromEntries([...root.querySelectorAll('tr[data-order]')].map((r) => [r.dataset.order, { hex: r.dataset.hex, ulps: r.dataset.ulps, closest: !!r.querySelector('.chip') }])),
    distinct: root.querySelector('[data-readout="distinct"]').dataset.distinct,
    survey: root.querySelector('[data-readout="survey"]').textContent.replace(/\s+/g, ' '),
  };
}, OB);
const matches = (o, r) => SHAPES.every((s) => o.rows[s].hex === f32hex(r.sums[s]) && Number(o.rows[s].ulps) === r.ulps[s] && o.rows[s].closest === r.closest.includes(s)) && Number(o.distinct) === r.distinct;
let o = await orders();
check('the orders explorer starts on the prediction\'s data: 1024 softmax terms, seed 1', o.n === '1024' && matches(o, D), JSON.stringify(o.rows));
check('its survey is the model\'s', o.survey.includes(`disagree in ${DS.disagree}`) && o.survey.includes(`the loop in ${DS.closest.loop}, neighbours in ${DS.closest.neighbours}, halves in ${DS.closest.halves}`), o.survey);
if (out) await page.screenshot({ path: out + '-orders.png' });
await press(OB, null, 'New data');
o = await orders();
check('"New data" moves to seed 2', matches(o, compareOrders(generate(1024, 'softmax', 2))));
await press(OB, 'Data', 'wide-range mix');
o = await orders();
check('a wide-range mix, seed 2', matches(o, compareOrders(generate(1024, 'wide', 2))) && o.survey.includes(`disagree in ${surveyOrders(1024, 'wide', 100).disagree}`), JSON.stringify(o.rows));
await setInput(OB + ' input[aria-label="Number of values"]', 3);
o = await orders();
check('8 values: the model\'s sums and survey', o.n === '8' && matches(o, compareOrders(generate(8, 'wide', 2))) && o.survey.includes(`disagree in ${surveyOrders(8, 'wide', 100).disagree}`), JSON.stringify(o));
await setInput(OB + ' input[aria-label="Number of values"]', 16);
o = await orders();
check('65536 values: still the model\'s sums, surveyed over 10 data sets', o.n === '65536' && matches(o, compareOrders(generate(65536, 'wide', 2))) && o.survey.includes('Over 10 data sets'), o.survey);
await press(OB, 'Data', 'uniform in [−1, 1]');
o = await orders();
check('uniform data at 65536', matches(o, compareOrders(generate(65536, 'uniform', 2))));

// ---------------------------------------------------------------- who wins a tie
const ties = () => page.evaluate((WB) => {
  const root = document.querySelector(WB);
  const w = root.querySelector('[data-readout="winners"]');
  return {
    independent: w.dataset.independent,
    rows: Object.fromEntries([...w.querySelectorAll('tr[data-order]')].map((r) => [r.dataset.order, { i: Number(r.dataset.index), v: r.dataset.value, text: r.textContent }])),
    cells: [...root.querySelectorAll('[data-cell]')].map((c) => ({ max: c.dataset.max, label: c.getAttribute('aria-label') })),
    note: root.querySelector('[data-readout="rule-note"]').textContent,
  };
}, WB);
const rule = async (label) => {
  await page.evaluate((WB, label) => [...document.querySelectorAll(WB + ' label.check')].find((l) => l.textContent.trim() === label).querySelector('input').click(), WB, label);
  await pause(100);
};
const sameAsModel = (w, row, r) => SHAPES.every((s) => w.rows[s].i === winners(row, r)[s].i);
let row = tieRow(8, [1, 4]);
let w = await ties();
check('the tie explorer starts with keep-left: the loop and neighbours pick index 1, halves 4', w.independent === '0' && w.rows.loop.i === 1 && w.rows.neighbours.i === 1 && w.rows.halves.i === 4 && sameAsModel(w, row, 'left'), JSON.stringify(w.rows));
check('the two planted maxima are marked: cells 1 and 4', w.cells.filter((c) => c.max === '1').map((c) => c.label).join(' | ') === 'logit 1: 4 | logit 4: 4', JSON.stringify(w.cells));
if (out) await page.screenshot({ path: out + '-ties.png' });
await rule('larger value; ties to the lower index');
w = await ties();
check('ties to the lower index: index 1 in every order', w.independent === '1' && SHAPES.every((s) => w.rows[s].i === 1) && w.note.includes('hipCUB'), JSON.stringify(w.rows));
await rule('larger value; ties to the higher index');
w = await ties();
check('ties to the higher index: index 4 in every order', w.independent === '1' && SHAPES.every((s) => w.rows[s].i === 4));
await page.evaluate((WB) => document.querySelector(WB + ' [data-cell="6"]').click(), WB);
await pause(100);
w = await ties();
row = tieRow(8, [4, 6]);
check('clicking cell 6 moves the older maximum there: now 4 and 6, the winner index 6', w.cells.filter((c) => c.max === '1').length === 2 && w.cells[6].max === '1' && w.cells[1].max === '0' && sameAsModel(w, row, 'higher') && w.rows.loop.i === 6, JSON.stringify(w.rows));
await rule('larger value; keep the left operand on ties');
w = await ties();
check('keep-left on maxima 4 and 6, as the model says', sameAsModel(w, row, 'left') && w.independent === (new Set(SHAPES.map((s) => winners(row, 'left')[s].i)).size === 1 ? '1' : '0'));

// A NaN.
await page.evaluate((WB) => document.querySelector(WB + ' [data-cell="1"]').click(), WB); // maxima back to 6 and 1
await page.evaluate((WB) => document.querySelector(WB + ' [data-cell="4"]').click(), WB); // and to 1 and 4
await pause(100);
await page.evaluate((WB) => [...document.querySelectorAll(WB + ' label.check')].find((l) => l.textContent.includes('NaN at index')).querySelector('input').click(), WB);
await page.$eval(WB + ' select[aria-label="NaN index"]', (s) => { s.value = '6'; s.dispatchEvent(new Event('change', { bubbles: true })); });
await pause(120);
row = tieRow(8, [1, 4], 6);
await rule('a > b ? a : b');
w = await ties();
check('with a NaN at 6, the comparison max depends on the order (the loop ends on the last element)', w.independent === '0' && w.rows.loop.v === '−0.4' && w.cells[6].label === 'logit 6: NaN' && w.rows.loop.text.startsWith('the loop') && !w.rows.loop.text.includes('index'), JSON.stringify(w.rows));
await rule('fmaxf (NaN as missing)');
w = await ties();
check('fmaxf ignores the NaN: 4 in every order', w.independent === '1' && SHAPES.every((s) => w.rows[s].v === '4'), JSON.stringify(w.rows));
await rule('NaN wins, then larger value; ties to the lower index');
w = await ties();
check('PyTorch\'s policy: the NaN wins at index 6 in every order', w.independent === '1' && SHAPES.every((s) => w.rows[s].i === 6 && w.rows[s].v === 'NaN') && w.note.includes('PyTorch'), JSON.stringify(w.rows));
await press(WB, 'Logits', '16');
w = await ties();
row = tieRow(16, [1, 4], 6);
check('16 logits: the model\'s winners', w.cells.length === 16 && sameAsModel(w, row, 'nanfirst'), JSON.stringify(w.rows));

// ---------------------------------------------------------------- questions
const question = (text) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt').textContent.includes(t)), text);
async function numeric(text, value) {
  const q = await question(text);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
}
async function choose(text, option) {
  const q = await question(text);
  await q.evaluate((e, o) => [...e.querySelectorAll('.opts label')].find((l) => l.textContent.includes(o)).querySelector('input').click(), option);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
}
t = await numeric('4096 elements', 341.25);
check('q-avg-4096: 341.25', t.includes('Correct.') && t.includes('12 steps'));
t = await numeric('2 venues', 4);
check('q-venues: 4 days', t.includes('Correct.') && t.includes('against 3'));
t = await choose('x0 − x1 − x2', 'after rewriting');
check('q-subtract: rewrite as a sum', t.includes('Correct.'));
t = await choose('needs a commutative operator', 'halves');
check('q-commute: halves', t.includes('Correct.') && t.includes('x0, x4, x2, x6, x1, x5, x3, x7'));
t = await choose('Which combine rule makes its answer independent', 'keep the left operand');
check('q-argmax-rule: keep-left is marked wrong with its reason', t.includes('Not quite.') && t.includes('depends on the tree'));
t = await choose('Which combine rule makes its answer independent', 'equal values go to the lower index');
check('q-argmax-rule: lower index', t.includes('Correct.') && t.includes('first maximal index'));

// ---------------------------------------------------------------- notes and layout
const notes = await page.$$eval('.note', (n) => n.map((x) => x.className + ': ' + x.textContent));
check('the sidebar erratum (a span of four steps) is noted', notes.some((x) => x.includes('span of four steps')));
check('Today notes for NVIDIA\'s associativity example, fmaxf and PyTorch\'s tie rule', notes.some((x) => x.includes('today') && x.includes('5.6.1.4')) && notes.some((x) => x.includes('today') && x.includes('fmaxf')) && notes.some((x) => x.includes('today') && x.includes('first maximal value')));
check('each explorer states what it simplifies, and no margin note follows a bench', await page.evaluate(() => {
  const benches = [...document.querySelectorAll('.prose > section.bench')];
  return benches.every((b) => !b.nextElementSibling?.classList.contains('note')) && document.querySelectorAll('.prose .note.caveat').length >= 3;
}));
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
check('at a narrow width the page does not scroll sideways', await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
const clipped = await page.evaluate(() => [...document.querySelectorAll('.prose > section.bench:not(.scene)')].flatMap((b) => {
  const r = b.getBoundingClientRect();
  return [...b.querySelectorAll('button, input, select')].filter((x) => {
    const q = x.getBoundingClientRect();
    return q.width > 0 && (q.right > r.right + 1 || q.left < r.left - 1);
  }).map((x) => (x.textContent || x.getAttribute('aria-label')).trim());
}));
check('and every control sits inside its bench', clipped.length === 0, clipped.join(' | '));
if (out) {
  await page.$eval(TB, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-builder.png' });
  await page.$eval(WB, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-ties.png' });
}

const expected404 = (u) => /\/content\/ch10\/narration\/(manifest\.json|tree-fold-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
