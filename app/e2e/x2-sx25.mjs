// Browser check for section X2.5 "Dequantise, or multiply in integers": the three-path stepper,
// the two predictions, the activation-cost explorer, the narrated scene "The integer path, one
// block", the lab helper listing, the backend table and a question. Every number asserted is
// recomputed from src/sim/qdot.ts inside the page.
// Usage: APP_URL=http://localhost:5173 node e2e/x2-sx25.mjs [--dark] [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

const dark = process.argv.includes('--dark');
const out = process.argv.slice(2).find((a) => !a.startsWith('--'));
const { browser, page, errors } = await open({ width: 1440, height: 1000, dark });
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
const bench = (t) => `section.bench[aria-label="${t}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const attr = (sel, a) => page.$eval(sel, (e, a) => e.getAttribute(a), a);
const model = (fn, ...args) =>
  page.evaluate(async (src, args) => {
    const m = await import('/src/sim/qdot.ts');
    return new Function('m', 'args', `return (${src})(m, ...args)`)(m, args);
  }, fn.toString(), args);
const click = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
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
  await pause(250);
};
const close = (a, b, rel = 1e-9) => Math.abs(a - b) <= rel * Math.max(Math.abs(a), Math.abs(b), 1e-30);

await goto(page, 'x2/X2.5');
const T = bench('One block, three paths');
await page.waitForSelector(T);

// ---------------------------------------------------------------- the stepper
const ref = await model((m) => {
  const out = {};
  for (const o of [false, true]) {
    const c = m.makeCase(o);
    out[o ? 'big' : 'plain'] = Object.fromEntries(['A', 'B', 'C'].map((p) => {
      const r = m.tracePath(p, c);
      return [p, { result: r.result, steps: r.steps.map((s) => s.acc), perRow: r.perRow, perToken: r.perToken, amax: r.q8?.amax }];
    }));
    out[o ? 'bigExact' : 'plainExact'] = m.exactDequantDot(c.block, c.x);
  }
  return out;
});
check('path A starts at step 0 of 8', (await attr(T + ' [data-at]', 'data-at')) === '0');
await click(T, 'Step');
let n = await text(T + ' [data-narrate]');
check('step 1 of path A dequantises weights 0 to 3', n.startsWith('Weights 0, 1, 2, 3:') && n.includes('(exact in float32)'), n);
check('the running value is the model\'s', (await text(T + ' [data-acc]')).replace(/[−,]/g, (c) => (c === '−' ? '-' : '')) === String(Number(ref.plain.A.steps[0].toPrecision(8))), await text(T + ' [data-acc]'));
check('four lanes are shown', (await page.$$(T + ' [data-lanes] tbody tr')).length === 4);
await click(T, 'Run to the end');
const results = await page.$$eval(T + ' [data-results] tr[data-path]', (rs) => Object.fromEntries(rs.map((r) => [r.dataset.path, Number(r.dataset.result)])));
check('final results of A, B and C are the model\'s', ['A', 'B', 'C'].every((p) => results[p] === ref.plain[p].result), JSON.stringify(results));
check('A and B differ in float32 on this block, C differs more', results.A !== results.B && Math.abs(results.C - ref.plainExact) > Math.abs(results.B - ref.plainExact));
check('the exact value is shown', Number(await attr(T + ' [data-exact]', 'data-exact')) === ref.plainExact);
check('path A: 64 float multiplies per row', (await text(T + ' [data-count="mul"]')) === String(ref.plain.A.perRow.mul) && ref.plain.A.perRow.mul === 64);
if (out) await page.screenshot({ path: out + '-stepper-A.png' });

await click(T, 'B. float, offset folded out');
await click(T, 'Step');
n = await text(T + ' [data-narrate]');
check('path B starts with the per-token sum and pre-division', n.startsWith('Once per token for this block') && n.includes('1, 256, 16 or 4096'), n);
await click(T, 'Step');
n = await text(T + ' [data-narrate]');
check('path B masks a 16-bit word four ways without shifting', n.includes('0x000F, 0x0F00, 0x00F0, 0xF000') && n.includes('weights 0, 1, 16, 17'), n);
await click(T, 'Run to the end');
check('path B: 34 float multiplies per row', (await text(T + ' [data-count="mul"]')) === '34');

await click(T, 'C. integer, activations to int8');
await click(T, 'Step');
n = await text(T + ' [data-narrate]');
check('path C begins by finding amax', n.includes('amax = ') && n.includes('FP16'), n);
for (let i = 0; i < 2; i++) await click(T, 'Step');
n = await text(T + ' [data-narrate]');
check('then a dp4a on masked unsigned codes', n.includes('masked with 0x0F0F0F0F') && n.includes(`sumi = ${ref.plain.C.steps[2]}`), n);
check('the integer sum is shown as exact int32', (await text(T)).includes('(int32, exact)'));
await click(T, 'Run to the end');
check('path C: 20 integer operations and 3 float multiplies per row', (await text(T + ' [data-count="int"]')) === '20' && (await text(T + ' [data-count="mul"]')) === '3');

// the large activation
await page.evaluate((T) => document.querySelector(T + ' label.check input').click(), T);
await pause(200);
await click(T, 'Run to the end');
const big = await page.$$eval(T + ' [data-results] tr[data-path]', (rs) => Object.fromEntries(rs.map((r) => [r.dataset.path, Number(r.dataset.result)])));
check('with the large channel, C is the model\'s and its gap from B grows', big.C === ref.big.C.result &&
  Math.abs(big.C - big.B) > 5 * Math.abs(results.C - results.B), JSON.stringify(big));
if (out) await page.screenshot({ path: out + '-stepper-C.png' });

// ---------------------------------------------------------------- the predictions
const predict = async (needle, value) => {
  const q = await page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), needle);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(250);
  return q;
};
const agree = await model((m) => m.bitwiseAgreement(40));
await predict('in how many do they give the same bits', 40);
check('bitwise reveal shows the model\'s count', Number(await attr('[data-bitwise]', 'data-bitwise')) === agree.same && agree.same < 40, String(agree.same));
const med = await model((m) => m.rowErrors(4096, 'normal').median);
await predict('how large do you expect path C', 1);
check('activation-error reveal shows the model\'s median', (await text('[data-act-reveal]')).includes((med * 100).toPrecision(2) + '%'));

// ---------------------------------------------------------------- the explorer
const X = bench('What activation quantisation costs');
const summary = async () => ({ median: Number(await attr(X + ' [data-summary]', 'data-median')), p95: Number(await attr(X + ' [data-summary]', 'data-p95')) });
const counts = () => page.$$eval(X + ' rect[data-count]', (r) => r.reduce((t, x) => t + Number(x.dataset.count), 0));
let s = await summary();
const e4096 = await model((m) => [m.rowErrors(4096, 'normal'), m.rowErrors(4096, 'outliers')].map((e) => ({ median: e.median, p95: e.p95 })));
check('explorer: 4,096 normal is the model\'s median and 95th percentile', s.median === e4096[0].median && s.p95 === e4096[0].p95);
check('explorer: the histogram counts 200 rows', (await counts()) === 200);
await click(X, 'normal, with a few large channels');
await pause(300);
s = await summary();
check('explorer: large channels raise the median (model\'s value)', s.median === e4096[1].median && s.median > e4096[0].median);
await select(X, 'Row length', 256);
s = await summary();
const e256 = await model((m) => m.rowErrors(256, 'outliers').median);
check('explorer: row length 256 recomputes', s.median === e256);
if (out) await page.screenshot({ path: out + '-explorer.png' });

// ---------------------------------------------------------------- the scene
const S = 'section.scene[aria-label="The integer path, one block"]';
const sc = await model((m) => {
  const row = m.integerSceneRow();
  const r = m.tracePath('C', row.case0);
  return { amax: r.q8.amax, sumi: r.steps.filter((x) => x.kind === 'dp4a')[7].acc, outC: row.outC, outB: row.outB, result: r.result };
});
const scripts = [];
for (let i = 1; i <= 5; i++) scripts.push((await (await fetch(`${BASE}/content/x2/narration/integer-path-${i}.txt`)).text()).replace(/\[[^\]]*\]\s*/g, '').trim());
const words = scripts.map((x) => x.split(/\s+/).length);
check('scene: five scripts of 35 to 60 words', words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scene: five parts', (await page.$$(S + ' .seg button')).length === 5);
check('scene: captions are off by default', !(await page.$eval(S, (e) => [...e.querySelectorAll('label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').checked)));
const toEnd = async (k) => {
  await page.evaluate((S, k) => document.querySelectorAll(S + ' .seg button')[k].click(), S, k);
  await pause(150);
  await page.$eval(S + ' input[type=range]', (el) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '1000');
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await pause(150);
};
await toEnd(0);
const noAudio = !(await text(S + ' footer')).startsWith('Narration generated');
if (noAudio) check('scene: with no audio the script is shown', (await text(S + ' .caption')).startsWith(scripts[0].slice(0, 30)));
check('scene beat 1: the warp max is the model\'s', (await text(S + ' [data-amax]')).includes(`amax = ${Number(sc.amax.toPrecision(5))}`));
await toEnd(2);
check('scene beat 3: the eighth dp4a ends at the model\'s sumi', (await text(S + ' [data-sumi]')).includes(`sumi = ${sc.sumi.toLocaleString('en-US').replace('-', '−')}`));
await toEnd(3);
check('scene beat 4: the block result', (await text(S + ' [data-result]')).endsWith(String(Number(sc.result.toPrecision(6))).replace('-', '−')));
await toEnd(4);
check('scene beat 5: the row on the integer and the float path', (await text(S + ' [data-outc]')).includes(String(Number(sc.outC.toPrecision(6))).replace('-', '−')) &&
  (await text(S + ' [data-outb]')).includes(String(Number(sc.outB.toPrecision(6))).replace('-', '−')));
if (out) await page.$eval(S, (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-scene.png' });

// ---------------------------------------------------------------- the listing and the table
const code = await page.$$eval('.code pre', (p) => p.map((x) => x.textContent).join('\n'));
check('the dot4_i8 helper is cut from lab X2.3\'s solution', code.includes('dot4_i8(int a, int b, int c)') && code.includes('__dp4a(a, b, c)') && code.includes('(int8_t)'));
check('its status names both verification levels', (await text('.code .status')).includes('emulator-checked') && (await text('.code .status')).includes('unverified'));
check('the backend table has three rows', (await page.$$('[data-backends] tbody tr')).length === 3);

// ---------------------------------------------------------------- a question
const q = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('__builtin_amdgcn_sdot4') && e.textContent.includes('What happens')));
await (await q.$$('input[type=radio]'))[0].click();
await (await q.$('button.primary')).click();
await pause(150);
check('question q-hip-sdot4 accepts the compile error', (await q.evaluate((e) => e.textContent)).includes('Correct.'));

const expected404 = (u) => /\/content\/x2\/narration\/(manifest\.json|integer-path-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
