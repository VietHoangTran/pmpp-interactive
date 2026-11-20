// Browser check for section 14.1: the "Ties in a candidate list" explorer (three presets,
// stable sort or bitonic network, the (value, index) option) and the "Two passes for a batch"
// explorer (prediction first, four combinations of passes), plus two questions. Every
// expected result is computed here from the models (src/sim/sortstable.ts, src/sim/sortnet.ts),
// not typed in.
// Usage: node e2e/ch14-s141.mjs [screenshot-prefix]
import { registerHooks } from 'node:module';
import { open, goto } from './browser.mjs';

// sortstable.ts imports its sibling without a file extension (as the app's bundler allows).
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
const { BATCH, CANDIDATE_PRESETS, cascade, rowChecks, sortCandidates } = await import('../src/sim/sortstable.ts');
const { tieGroups } = await import('../src/sim/sortnet.ts');

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
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
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const clickText = async (scope, label) => {
  const done = await page.evaluate((sel, label) => {
    const btn = [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, bench(scope), label);
  await pause(80);
  return done;
};
const setSelect = async (scope, labelStart, value) => {
  await page.evaluate((sel, labelStart, value) => {
    const s = [...document.querySelectorAll(sel + ' label.field')].find((l) => l.textContent.trim().startsWith(labelStart)).querySelector('select');
    const set = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set;
    set.call(s, value);
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, bench(scope), labelStart, value);
  await pause(80);
};
const setBox = async (scope, on) => {
  await page.evaluate((sel, on) => {
    const b = document.querySelector(sel + ' label.check input');
    if (b.checked !== on) b.click();
  }, bench(scope), on);
  await pause(80);
};
const fmtP = (p) => p.toFixed(3).replace(/0$/, '');

// ---------------------------------------------------------------- ties
const T = 'Ties in a candidate list';
await goto(page, 'ch14/14.1');
await page.waitForSelector(bench(T));

const shown = () => page.evaluate((sel) => ({
  output: [...document.querySelectorAll(sel + ' [data-row="output"] > span')].map((s) => `${s.querySelector('b').textContent}:${s.querySelector('small').textContent}`),
  groups: [...document.querySelectorAll(sel + ' tr[data-group]')].map((r) => `${r.dataset.group}:${r.dataset.ok}`),
  none: !!document.querySelector(sel + ' [data-group="none"]'),
  bad: document.querySelectorAll(sel + ' [data-row="output"] > span.bad').length,
}), bench(T));
const want = (presetId, kind, byIndex) => {
  const items = CANDIDATE_PRESETS.find((p) => p.id === presetId).items;
  const o = sortCandidates(items, kind, byIndex);
  return { output: o.map((x) => `${fmtP(x.key)}:t${x.id}`), groups: tieGroups(o).map((g) => `${fmtP(g.key)}:${g.increasing ? 'yes' : 'no'}`) };
};
let s = await shown();
let w = want('two', 'network', false);
check('default (ties at two values, network): output order is the model\'s', JSON.stringify(s.output) === JSON.stringify(w.output), s.output.join(' '));
check('the network reorders the 0.15 group and keeps the 0.20 pair', s.groups.join() === w.groups.join() && w.groups.join() === '0.20:yes,0.15:no' && s.bad === 3, s.groups.join());
if (out) await page.screenshot({ path: out + '-ties.png' });

await setBox(T, true);
s = await shown();
w = want('two', 'network', true);
check('with (value, index) the network gives the stable order', s.output.join() === w.output.join() && s.groups.every((g) => g.endsWith(':yes')) && s.bad === 0, s.groups.join());
await setBox(T, false);
await clickText(T, 'stable sort');
s = await shown();
w = want('two', 'stable', false);
check('the stable sort keeps tied tokens in token order without help', s.output.join() === w.output.join() && s.groups.join() === '0.20:yes,0.15:yes', s.groups.join());
await clickText(T, 'bitonic network (14.9)');
await clickText(T, 'all equal');
s = await shown();
check('all equal: the network keeps the input order, by luck', s.groups.join() === '0.125:yes' && s.output.join() === want('equal', 'network', false).output.join(), s.groups.join());
await clickText(T, 'no ties');
s = await shown();
check('no ties: nothing to check', s.none && s.groups.length === 0);

// ---------------------------------------------------------------- two passes
const P = 'Two passes for a batch';
const applyBtn = () => page.evaluate((sel) => [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim() === 'Apply both passes').disabled, bench(P));
check('cannot apply before predicting both rows', await applyBtn());
const rowsShown = () => page.$$eval(bench(P) + ' tr[data-rowcheck]', (rs) => rs.map((r) => ({ ok: r.dataset.ok, c: r.dataset.contiguous, s: r.dataset.sorted, g: r.querySelector('[data-guess]').dataset.guess })));
const modelRows = (first, stable2) => {
  const second = first === 'value' ? 'row' : 'value';
  return rowChecks(cascade(BATCH, [{ key: first, stable: true }, { key: second, stable: stable2 }])[2], 2, 6);
};
const yes = (b) => (b ? 'yes' : 'no');
const same = (shownRows, model) => shownRows.length === 2 && shownRows.every((r, k) => r.ok === yes(model[k].ok) && r.c === yes(model[k].contiguous) && r.s === yes(model[k].sorted));

// Default: rows first, unstable second pass. Predict "contiguous and sorted" for both: wrong.
await setSelect(P, 'Row 0', 'ok');
await setSelect(P, 'Row 1', 'ok');
check('both predictions made: apply is enabled', !(await applyBtn()));
await clickText(P, 'Apply both passes');
let r = await rowsShown();
let m = modelRows('row', false);
check('rows first, then value: rows interleaved, as the model says', same(r, m) && r.every((x) => x.c === 'no'), JSON.stringify(r));
check('the prediction "contiguous and sorted" is marked wrong for both rows', r.every((x) => x.g === 'wrong'), JSON.stringify(r));
const p2 = await page.$eval(bench(P) + ' [data-row="pass2"]', (e) => [...e.querySelectorAll('span small')].map((x) => x.textContent.split(' ')[0]).join(''));
check('after pass 2 by value the row labels alternate (r1 first: the 0.40)', p2.startsWith('r1r0'), p2);

// Value first, stable second: the cascade works.
await setSelect(P, 'Pass 1', 'value');
check('changing a pass hides the old result', !(await page.$(bench(P) + ' tr[data-rowcheck]')));
await clickText(P, 'a stable sort');
await clickText(P, 'Apply both passes');
r = await rowsShown();
m = modelRows('value', true);
check('value first, row last and stable: both rows contiguous and sorted', same(r, m) && r.every((x) => x.ok === 'yes' && x.g === 'right'), JSON.stringify(r));
if (out) await page.screenshot({ path: out + '-passes.png' });

// Value first, unstable second.
await clickText(P, 'an unstable sort');
await setSelect(P, 'Row 0', 'not');
await setSelect(P, 'Row 1', 'not');
await clickText(P, 'Apply both passes');
r = await rowsShown();
m = modelRows('value', false);
check('row pass unstable: rows contiguous, values out of order', same(r, m) && r.every((x) => x.c === 'yes' && x.s === 'no' && x.g === 'right'), JSON.stringify(r));

// ---------------------------------------------------------------- questions
const q = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('151,936 tokens')));
await (await q.$('input[type=number]')).type('135679');
await (await q.$('button.primary')).click();
await pause(150);
check('numeric question: 135,679 is accepted', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
const qc = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Sort it by age, ascending')));
await qc.evaluate((e) => [...e.querySelectorAll('.opts label')].find((l) => l.textContent.startsWith('(30,150), (29,80)')).querySelector('input').click());
await pause(80);
await (await qc.$('button.primary')).click();
await pause(150);
check('choice question: the cascade answer is accepted', (await qc.evaluate((e) => e.textContent)).includes('Correct.'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
