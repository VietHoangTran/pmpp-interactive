// Browser check for section X1.5: the crossing-token prediction and its bar chart, the
// "Sort what you need" explorer (vocabulary, shape, another row, p, top-k), the sort-share
// prediction and the questions. Every number asserted is recomputed here from the models
// (src/sim/nucleus.ts, src/sim/scan.ts).
// Usage: APP_URL=http://localhost:5173 node e2e/x1-sx15.mjs [--dark] [screenshot-prefix]
import { registerHooks } from 'node:module';
import { open, goto, BASE } from './browser.mjs';

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
const M = await import('../src/sim/nucleus.ts');
const SC = await import('../src/sim/scan.ts');

const dark = process.argv.includes('--dark');
const out = process.argv.slice(2).find((a) => !a.startsWith('--'));
const { browser, page, errors } = await open({ width: 1440, height: 1000, dark });
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
const B = 'section.bench[aria-label="Sort what you need"]';
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const seg = async (scope, group, label) => {
  await page.evaluate((scope, group, label) => {
    const g = document.querySelector(`${scope} [role=group][aria-label="${group}"]`);
    [...g.querySelectorAll('button')].find((b) => b.textContent.trim() === label).click();
  }, scope, group, label);
  await pause(250);
};
const setInput = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(250);
};
const press = async (scope, label) => {
  await page.evaluate((scope, label) => [...document.querySelectorAll(scope + ' button')].find((b) => b.textContent.trim() === label).click(), scope, label);
  await pause(250);
};
const kept = () => page.$eval(B + ' [data-readout=kept]', (e) => ({ kept: Number(e.dataset.kept), keptK: e.dataset.keptK }));
const strat = () => page.$$eval(B + ' tr[data-strategy]', (rs) => Object.fromEntries(rs.map((r) => [r.dataset.strategy, r.dataset.n === '' ? null : Number(r.dataset.n)])));
/** The explorer's state, recomputed from the model. */
const model = (V, s, seed, p, k) => {
  const row = M.rowView(V, s, seed);
  return M.sortPlan(row, p, k, 0.05);
};
const agree = async (name, V, s, seed, p, k) => {
  const plan = model(V, s, seed, p, k);
  const got = await kept();
  const st = await strat();
  const ok = got.kept === plan.keptFull && got.keptK === (plan.keptAfterTopK == null ? '' : String(plan.keptAfterTopK))
    && st.full === V && st.topk === plan.sizes.topK && st.adaptive === plan.sizes.adaptive && st.minp === plan.sizes.minPFirst;
  check(name, ok, JSON.stringify({ got, st, plan: { kept: plan.keptFull, k: plan.keptAfterTopK, sizes: plan.sizes } }));
  return plan;
};

await goto(page, 'x1/X1.5');
await page.waitForSelector(B);

// ---------------------------------------------------------------- the crossing prediction
const q = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const cross = await q('0.0625, 0.0625] and p = 0.875');
await (await cross.$('input[type=number]')).type('3');
await (await cross.$('button.primary')).click();
await pause(250);
const want = SC.topP(M.CROSSING.probs, M.CROSSING.p, 'exclusive');
const rev = await text('[data-reveal=crossing]');
check('p-crossing: the reveal says 3, as predicted', want.kept === 3 && rev.startsWith('3, as you said'), rev.slice(0, 80));
const bars = await page.$$eval('[data-part=crossing] g[data-token]', (gs) => gs.map((g) => [g.dataset.keep, Number(g.dataset.before)]));
check('p-crossing: the bars keep tokens 0 to 2 and show the mass before each', JSON.stringify(bars) === JSON.stringify(want.keep.map((k, i) => [k ? '1' : '0', want.before[i]])), JSON.stringify(bars));
check("p-crossing: token 2's inclusive sum is exactly p", want.upto[2] === M.CROSSING.p && want.before[2] < M.CROSSING.p);
if (out) await (await page.$('[data-part=crossing]')).screenshot({ path: out + '-crossing.png' });

// ---------------------------------------------------------------- the explorer
let V = 151936, s = 1.5, seed = 1, p = 0.95, k = 0;
await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));
let plan = await agree('explorer opens: 151,936 tokens, s = 1.5, p = 0.95, top-k off, as the model computes', V, s, seed, p, k);
check("peaked: llama.cpp's route sorts only 256 and the full sort is almost all waste", plan.sizes.adaptive === 256 && !plan.widened && plan.keptFull < 1000,
  JSON.stringify(plan.sizes));
check('peaked: the strategy note says the top 256 reach p', (await text(B + ' tr[data-strategy=adaptive]')).includes('the top 256 reach p'));
check('peaked: top-k off is shown as off', (await text(B + ' tr[data-strategy=topk]')).includes('off'));
if (out) await (await page.$(B)).screenshot({ path: out + '-explorer-peaked.png' });

s = 0.6;
await setInput(B + ' input[aria-label=shape]', s);
plan = await agree('flat row (s = 0.6): the model\'s kept count and sort sizes', V, s, seed, p, k);
check("flat: llama.cpp's route widens to the whole row; top-p keeps thousands", plan.widened && plan.sizes.adaptive === V && plan.keptFull > 1000
  && (await text(B + ' tr[data-strategy=adaptive]')).includes('widened to the row'), JSON.stringify(plan.sizes));
await seg(B, 'Top-k', '40');
k = 40;
plan = await agree('top-k 40 on the flat row: the sort sees 40 values; top-p over them as the model says', V, s, seed, p, k);
check('top-k 40: it is top-k that caps the sort', plan.sizes.topK === 40 && plan.keptAfterTopK <= 40 && (await text(B + ' [data-readout=kept]')).includes('After top-k = 40'));
if (out) await (await page.$(B)).screenshot({ path: out + '-explorer-flat-k40.png' });

await seg(B, 'Vocabulary', '50,257');
V = 50257;
await setInput(B + ' input[aria-label=p]', 80);
p = 0.8;
await agree('50,257 tokens, p = 0.80: the model\'s numbers', V, s, seed, p, k);
await press(B, 'another row');
seed = 2;
await agree('another row (seed 2): the model\'s numbers', V, s, seed, p, k);
await seg(B, 'Top-k', 'off');
k = 0;
s = 3;
await setInput(B + ' input[aria-label=shape]', s);
plan = await agree('very peaked (s = 3), top-k off: the model\'s numbers', V, s, seed, p, k);
check('very peaked: a handful of tokens', plan.keptFull <= 10, String(plan.keptFull));

// Measured share or the instruction to produce it.
const reports = await (await fetch(BASE + '/api/reports')).json();
const real = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const anyLab = real.some((b) => (reports[b].results ?? []).some((r) => r.lab.startsWith('x1/lab2') && r.variant === 'solution' && M.sortShare(r.bench ?? []).length));
check(anyLab ? 'lab X1.2 measurements are shown' : 'with no lab X1.2 run, the command to produce one is shown',
  anyLab ? !!(await page.$(B + ' [data-part=measured]')) : (await text(B + ' [data-part=measured-none]')).includes('pmpp.py run x1/lab2'));

// ---------------------------------------------------------------- prediction and questions
const share = await q("percentage of the composed chain's time is the sort");
await (await share.$('input[type=number]')).type('60');
await (await share.$('button.primary')).click();
await pause(250);
const rtext = await share.evaluate((e) => e.textContent);
check('p-sort-share is committed and the reveal reads this machine\'s reports', rtext.includes('You predicted 60') && (anyLab ? /of the\s+chain/.test(rtext) : rtext.includes('No measurement on this machine yet')), rtext.slice(0, 240));

const numeric = async (needle, value) => {
  const h = await q(needle);
  await (await h.$('input[type=number]')).type(String(value));
  await (await h.$('button.primary')).click();
  await pause(150);
  return (await h.evaluate((e) => e.textContent)).includes('Correct.');
};
const choice = async (needle, index) => {
  const h = await q(needle);
  await (await h.$$('input[type=radio]'))[index].click();
  await (await h.$('button.primary')).click();
  await pause(150);
  return (await h.evaluate((e) => e.textContent)).includes('Correct.');
};
check("q-mlx-ties accepts mlx-lm's 5", SC.topP(M.MLX_TIES.probs, M.MLX_TIES.p, 'mlx').kept === 5 && (await numeric("mlx-lm's", 5)));
check('q-inclusive-bug: the crossing token is dropped', SC.topP(M.INCLUSIVE_BUG.probs, M.INCLUSIVE_BUG.p, 'inclusive').kept === 1 && (await choice('inclusive sum up to i is below p', 0)));
check('q-why-scan: the exclusive scan gives the mass before', await choice('compute an exclusive scan rather than an inclusive', 0));
check('q-p1: float mass before can reach 1', await choice('Why must p ≥ 1 keep every token', 0));
check('q-adaptive: when the top 256 hold less than p', await choice('sort more than its first 256', 0));

check("the tie section links to chapter 11's explorer", (await page.$$eval('a', (as) => as.some((a) => a.getAttribute('href')?.includes('ch11') && a.textContent.includes('Where the nucleus ends')))));

// Phone width: nothing makes the page scroll sideways.
await page.setViewport({ width: 390, height: 800, deviceScaleFactor: 1 });
await pause(300);
const sw = await page.evaluate(() => document.documentElement.scrollWidth);
check('no sideways page scroll at 390 px', sw <= 390, String(sw));
if (out) {
  await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
