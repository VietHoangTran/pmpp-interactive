// Browser check for section 9.6: the "Run merging, one warp" stepper, the "Atomics per
// element" table, the listing, a prediction and questions. Expected values come from the
// model (src/sim/histruns.ts), imported directly.
// Usage: node e2e/ch09-s96.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { WORKED, makeImage, summary, trace, workedImage } from '../src/sim/histruns.ts';

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
  const done = await page.evaluate(
    (scope, label) => {
      const root = [...document.querySelectorAll('section.bench')].find((b) => b.getAttribute('aria-label') === scope);
      const btn = [...root.querySelectorAll('button')].find((b) => b.textContent.trim() === label);
      if (!btn || btn.disabled) return false;
      btn.click();
      return true;
    },
    scope,
    label,
  );
  await pause(80);
  return done;
};
const setField = async (scope, aria, value) => {
  const ok = await page.evaluate(
    (scope, aria, value) => {
      const root = [...document.querySelectorAll('section.bench')].find((b) => b.getAttribute('aria-label') === scope);
      const el = root.querySelector(`[aria-label="${aria}"]`);
      if (!el) return false;
      const proto = el.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, String(value));
      el.dispatchEvent(new Event(el.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true }));
      return true;
    },
    scope,
    aria,
    value,
  );
  await pause(150);
  return ok;
};

await goto(page, 'ch09/9.6');
const S = 'Run merging, one warp';
await page.waitForSelector(bench(S));
const lanes = () => page.$$eval(bench(S) + ' tr[data-lane]', (rs) => rs.map((r) => ({ atomics: Number(r.dataset.atomics), branch: r.dataset.branch })));
const totals = () => page.$eval(bench(S) + ' [data-totals]', (e) => ({ atomics: Number(e.dataset.atomics), elements: Number(e.dataset.elements), divergent: Number(e.dataset.divergent) }));
const narrate = () => page.$eval(bench(S) + ' .narrate', (e) => e.textContent);

// ---------------------------------------------------------------- the worked example
const B = 256;
const wcfg = { p: 'interleaved', n: B * 8, blockDim: B, coarse: 8, block: 0, firstThread: 0, lanes: 8 };
const wtr = trace(workedImage('interleaved', B), wcfg);
await clickText(S, 'Step');
check('step 1 loads each lane’s first pixel', (await narrate()).includes('Lines 11 to 13'));
let ls = await lanes();
check('after step 1 every lane holds its first value', ls.every((l) => l.branch === 'first' && l.atomics === 0), JSON.stringify(ls));
await clickText(S, 'Step');
check('iteration 1: lanes disagree and the warp splits', (await narrate()).includes('splits'), await narrate());
check('the split is marked', (await page.$eval(bench(S) + ' [data-splits]', (e) => Number(e.dataset.splits))) === 1);
await clickText(S, 'Run to the end');
ls = await lanes();
check('lane 0 (7, 7, 7, 3, 3, 7, 7, 9) issues 4 atomics', ls[0]?.atomics === 4, JSON.stringify(ls[0]));
check('every lane’s atomics match the model', JSON.stringify(ls.map((l) => l.atomics)) === JSON.stringify(wtr[wtr.length - 1].lanes.map((l) => l.atomics)), JSON.stringify(ls));
check('every lane ends with the final flush', ls.every((l) => l.branch === 'final'));
const wt = await totals();
check('totals: 32 atomics for 64 pixels', wt.atomics === 32 && wt.elements === 64, JSON.stringify(wt));
check('flushes add up to a plain count', (await page.$eval(bench(S) + ' [data-check]', (e) => e.dataset.check)) === 'ok');
check('the run column shows lane 0 as 7×3 3×2 7×2 9', (await page.$eval(bench(S) + ' tr[data-lane="0"]', (e) => e.textContent)).includes('7×3  3×2  7×2  9'));
if (out) await page.screenshot({ path: out + '-worked.png' });
check('model sanity: lane 0 is the question’s sequence', WORKED[0].join(',') === '7,7,7,3,3,7,7,9');

// ---------------------------------------------------------------- sky: one atomic per thread per row change
await setField(S, 'data', 'sky');
await setField(S, 'coarse factor', 32);
await clickText(S, 'Run to the end');
ls = await lanes();
check('sky, factor 32: a block spans two rows, so each lane flushes twice', ls.length === 8 && ls.every((l) => l.atomics === 2), JSON.stringify(ls));
await setField(S, 'coarse factor', 16);
await clickText(S, 'Run to the end');
ls = await lanes();
check('sky, factor 16: a block is one row, one atomic per lane', ls.every((l) => l.atomics === 1), JSON.stringify(ls));
check('sky: the warp never splits', (await totals()).divergent === 0);

// ---------------------------------------------------------------- stripes: the partitioning decides
await setField(S, 'data', 'stripes');
await clickText(S, 'Run to the end');
const stripes = makeImage('stripes');
const cfg = (p) => ({ p, n: stripes.length, blockDim: B, coarse: 16, block: 0, firstThread: 0, lanes: 8 });
const modelAtomics = (p) => { const t = trace(stripes, cfg(p)); return t[t.length - 1].atomics; };
const inter = (await totals()).atomics;
check('stripes, interleaved: matches the model', inter === modelAtomics('interleaved'), `${inter} vs ${modelAtomics('interleaved')}`);
await clickText(S, 'Contiguous (for comparison)');
await clickText(S, 'Run to the end');
const contig = (await totals()).atomics;
check('stripes, contiguous: two runs of 8 per lane, 16 atomics', contig === 16 && contig === modelAtomics('contiguous'), String(contig));
check('flipping the partitioning changes the totals', inter > 4 * contig, `${inter} vs ${contig}`);
if (out) await page.screenshot({ path: out + '-stripes.png' });

// ---------------------------------------------------------------- atomics per element
const T = 'Atomics per element';
const table = () => page.$$eval(bench(T) + ' tr[data-preset]', (rs) => Object.fromEntries(rs.map((r) => [r.dataset.preset, { inter: Number(r.dataset.inter), contig: Number(r.dataset.contig), split: r.dataset.split }])));
let t = await table();
check('noise: about one atomic per element', t.noise?.inter > 0.99 && t.noise.inter <= 1, JSON.stringify(t.noise));
check('constant: 1 / COARSE_FACTOR (16)', t.constant?.inter === 1 / 16 && Number(t.constant.split) === 0, JSON.stringify(t.constant));
check('sky row matches the model', t.sky?.inter === summary(makeImage('sky'), 'interleaved', 256, 16, 32).perElement, JSON.stringify(t.sky));
check('stripes merge only with contiguous elements', t.stripes?.inter > 0.99 && t.stripes.contig === 0.125, JSON.stringify(t.stripes));
check('token ids split more warps than noise', Number(t.tokens?.split) > Number(t.noise?.split), JSON.stringify([t.tokens, t.noise]));
await setField(T, 'coarse factor', 4);
t = await table();
check('constant at factor 4: 0.25', t.constant?.inter === 0.25, JSON.stringify(t.constant));
await setField(T, 'warp width', 64);
t = await table();
check('a 64-lane warp on noise splits as the model says', Number(t.noise.split) === summary(makeImage('noise'), 'interleaved', 256, 4, 64).divergentShare, t.noise.split);

// ---------------------------------------------------------------- prediction, listing, questions
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Predict first') && e.textContent.includes('sky preset')));
await (await pred.$('input[type=number]')).type('6.25');
await (await pred.$('button.primary')).click();
await page.waitForFunction((e) => e.querySelector('[data-fraction]'), {}, pred);
const frac = await pred.evaluate((e) => ({ v: Number(e.querySelector('[data-fraction]').dataset.fraction), t: e.textContent }));
check('p-runs-fraction reveals 6.25% from the model', frac.v === 6.25 && frac.t.includes('That is what you predicted'), frac.t.slice(0, 160));
check('the reveal links to the labs page', await pred.evaluate((e) => !!e.querySelector('a[href="#/ch09/labs"]')));

const missing = await page.$('[data-listing="missing"]');
if (missing) check('listing: lab 9.2 not in place yet (says so)', true);
else {
  const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent));
  check('listing shows CUDA and Metal tabs', tabs.includes('CUDA') && tabs.includes('Metal'), tabs.join('|'));
  const hl = await page.$$eval('.code .ln.hl', (e) => e.map((x) => x.textContent));
  check('the two flushes are highlighted', hl.length === 2 && hl.every((l) => l.includes('fetch_add(bin_r')), hl.join('|'));
}

const answer = async (needle, value) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
check('q-runs-count accepts 4', (await answer('How many atomic operations', 4)).includes('Correct.'));
const choose = async (needle, optionText) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await q.evaluate((e, o) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes(o)).querySelector('input').click(), optionText);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
check('q-final-flush: the lag answer is right', (await choose('Why is the flush after the loop', 'last run is always still in the registers')).includes('Correct.'));
check('q-spacing: the stripes answer is wrong and says why', (await choose('Which input gains most from run', 'Stripes of 8 identical')).includes('256 apart'));
const div = await choose('cost the most divergence', 'some lanes extend a run');
check('q-diverge: correct, and quotes the model’s noise figure', div.includes('Correct.') && /about [\d.]+%/.test(div), div.slice(0, 200));

// Predict, then read: the questions the next paragraph answers come before it.
const order = await page.evaluate(() => {
  const kids = [...document.querySelector('.prose').children];
  const at = (f) => kids.findIndex(f);
  const q = (t) => at((k) => k.classList.contains('q') && k.textContent.includes(t));
  const p = (t) => at((k) => k.tagName === 'P' && k.textContent.includes(t));
  return {
    fewQ: q('What does that make affordable'), fewP: p('With a handful of bins a thread can keep a whole private'),
    flushQ: q('Why is the flush after the loop'), flushP: p('The flush always lags one pixel behind'),
    tokQ: q('when counting the token ids of English prompts'), tokP: p('Equal tokens are rarely neighbours'),
  };
});
check('q-few-bins, q-final-flush and q-tokens each come before the paragraph that answers it',
  order.fewQ >= 0 && order.fewQ < order.fewP && order.flushQ >= 0 && order.flushQ < order.flushP && order.tokQ >= 0 && order.tokQ < order.tokP, JSON.stringify(order));
const prose96 = await page.$eval('.prose', (e) => e.textContent.replace(/\s+/g, ' '));
check('one bridge: count_equal has moved to the summary page', !prose96.includes('count_equal'));
check('warp-level combining is a related remedy, not the same idea', prose96.includes('merges across lanes rather than across one thread\'s elements') && prose96.includes('CUB\'s histogram ships the same idea'));

// ---------------------------------------------------------------- narrow screen
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
const wide = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
check('no horizontal page scroll at 390 px', wide <= 0, `${wide}px too wide`);

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
