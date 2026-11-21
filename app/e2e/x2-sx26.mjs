// Browser check for section X2.6 "Where the accumulation must stay wide": the bound prediction,
// the accumulator ladder, the narrated scene "Where the sum stops fitting", the safe-or-unsafe
// bug spot and the questions. Every number asserted is recomputed from src/sim/qdot.ts inside
// the page.
// Usage: APP_URL=http://localhost:5173 node e2e/x2-sx26.mjs [--dark] [screenshot-prefix]
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
  await pause(100);
  return done;
};
const select = async (scope, aria, value) => {
  await page.evaluate((scope, aria, value) => {
    const s = document.querySelector(`${scope} select[aria-label="${aria}"]`);
    s.value = String(value);
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, scope, aria, value);
  await pause(200);
};
const range = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(200);
};
const question = (needle) => page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), needle);
const answerNumber = async (needle, v) => {
  const q = await question(needle);
  await (await q.$('input[type=number]')).type(String(v));
  await (await q.$('button.primary')).click();
  await pause(200);
  return q.evaluate((e) => e.textContent);
};

await goto(page, 'x2/X2.6');
const L = bench('The accumulator ladder');
await page.waitForSelector(L);

// ---------------------------------------------------------------- bounds
const B = await model((m) => ({ q8: m.BOUND_Q8Q8, terms: m.int32Terms(16129) }));
check('model bounds: 516,128 and 133,144 terms', B.q8 === 516128 && B.terms === 133144, JSON.stringify(B));
const pq = await question('The largest magnitude of one Q8_0');
await (await pq.$('input[type=number]')).type('500000');
await (await pq.$('button.primary')).click();
await pause(200);
check('prediction reveal states 516,128 and that FP16 cannot hold it', (await text('[data-bound-reveal]')).includes('516,128') && (await text('[data-bound-reveal]')).includes('65,504'));
check('numeric question: the block bound', (await answerNumber('can reach?', 516128)).includes('Correct.'));
check('numeric question: int32 terms', (await answerNumber('before the next one overflows', 133144)).includes('Correct.'));
check('numeric question: 133,145 is wrong', (await answerNumber('before the next one overflows', 133145)).includes('Not quite'));

// ---------------------------------------------------------------- the ladder
const ladder = async () => ({ status: await attr(L + ' [data-ladder]', 'data-status'), rel: Number(await attr(L + ' [data-ladder]', 'data-relerr')), b0: await text(L + ' [data-block0]') });
const want = (o) => model((m, o) => {
  const r = m.accumulate(o);
  return { rel: r.relErr, first: r.block0.first, end: r.block0.values[r.block0.values.length - 1] };
}, o);
const base = { outer: 'fp32', data: 'worst', n: 4096, pairing: 'q8q8', step: 1 };
let l = await ladder();
let w = await want({ ...base, inner: 'int16' });
check('ladder default (int16, worst case): wraps on the third product', l.status === 'wrapped' && l.b0.includes('wrapped at step 3') && w.first.wrap === 3, l.b0);
check('ladder: the relative error is the model\'s', l.rel === w.rel || (!Number.isFinite(l.rel) && !Number.isFinite(w.rel)));
check('ladder: wrapped points are drawn red', (await page.$$(L + ' circle[fill="var(--fail)"]')).length > 0);
if (out) await page.screenshot({ path: out + '-ladder-int16.png' });

await select(L, 'Inner accumulator', 'int32');
l = await ladder();
w = await want({ ...base, inner: 'int32' });
check('ladder int32: exact block sums, float-rounding error', l.status === 'exact' && l.b0.includes('ends at 516,128') && l.rel === w.rel && w.rel < 1e-6, JSON.stringify(l));

await select(L, 'Inner accumulator', 'fp16');
l = await ladder();
check('ladder FP16 inner: infinity at the fifth product', l.status === 'inf' && l.b0.includes('infinity at step 5') && (await text(L)).includes('overflow'), l.b0);

await select(L, 'Inner accumulator', 'fp32');
l = await ladder();
check('ladder FP32 inner: exact, 516,128 below 2^24', l.status === 'exact' && l.b0.includes('ends at 516,128'));

// FP16 on the outer sum, realistic data, drag the row length
await select(L, 'Inner accumulator', 'int32');
await select(L, 'Outer accumulator', 'fp16');
await click(L, 'realistic');
await range(L + ' input[type=range]', 2); // 128
const short = await ladder();
await range(L + ' input[type=range]', 9); // 16,384
const long = await ladder();
const wl = await want({ ...base, data: 'realistic', inner: 'int32', outer: 'fp16', n: 16384 });
const ws = await model((m) => [128, 16384].map((n) => m.accumulate({ inner: 'int32', outer: 'fp16', data: 'realistic', n, pairing: 'q8q8' }).absErr));
check('ladder FP16 outer: the model\'s error at 16,384', long.rel === wl.rel && (await text(L)).includes('16,384 (512 blocks)'));
check('ladder FP16 outer: absolute error grows with the row', ws[1] > 10 * ws[0] && short.rel > 1e-6, JSON.stringify(ws));
await select(L, 'Outer accumulator', 'fp32');
check('ladder int32 then FP32 at 16,384: float rounding', (await ladder()).rel < 1e-6);

// Q4_0 nibbles in a short, four at a time
await select(L, 'Inner accumulator', 'int16');
await select(L, 'Pairing', 'q4q8');
await click(L, 'worst case');
await select(L, 'Integer step', 4);
l = await ladder();
check('ladder Q4_0 nibbles in int16 with dp4a steps: wraps on step 5 of 8', l.status === 'wrapped' && l.b0.includes('wrapped at step 5') && (await page.$$(L + ' svg circle')).length === 8, l.b0);
await click(L, 'realistic');
check('ladder: the same short passes realistic Q4_0 data', (await ladder()).status !== 'wrapped');
if (out) await page.screenshot({ path: out + '-ladder-q4.png' });

// ---------------------------------------------------------------- the scene
const S = 'section.scene[aria-label="Where the sum stops fitting"]';
const scripts = [];
for (let i = 1; i <= 5; i++) scripts.push((await (await fetch(`${BASE}/content/x2/narration/narrow-accumulator-${i}.txt`)).text()).replace(/\[[^\]]*\]\s*/g, '').trim());
const words = scripts.map((x) => x.split(/\s+/).length);
check('scene: five scripts of 35 to 60 words', words.every((x) => x >= 35 && x <= 60), words.join(' '));
check('scene: captions are off by default', !(await page.$eval(S, (e) => [...e.querySelectorAll('label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').checked)));
const toEnd = async (k) => {
  await page.evaluate((S, k) => document.querySelectorAll(S + ' .seg button')[k].click(), S, k);
  await pause(150);
  await range(S + ' input[type=range]', 1000);
};
const col = (id) => page.$eval(`${S} [data-col="${id}"]`, (e) => ({ value: Number(e.dataset.value), status: e.dataset.status }));
const sm = await model((m) => {
  const s = m.narrowScene();
  return { i16: s.traces.int16.values[3], h: s.outerHalf[127], f: s.outerFloat[127] };
});
await toEnd(1);
let c = await col('int16');
check('scene beat 2: int16 has wrapped after four products', c.status === 'wrapped' && c.value === sm.i16 && c.value < 32767 * 2, JSON.stringify(c));
await toEnd(2);
c = await col('fp16');
check('scene beat 3: FP16 is infinity after six products', c.status === 'infinity' && (await attr(`${S} [data-col="fp16"]`, 'data-value')) === 'Infinity', JSON.stringify(c));
await toEnd(3);
const c32 = await col('int32');
const cf = await col('fp32');
check('scene beat 4: int32 and FP32 reach 516,128 exactly', c32.value === 516128 && cf.value === 516128 && c32.status === 'exact' && cf.status === 'exact');
await toEnd(4);
const drift = await text(S + ' [data-drift]');
check('scene beat 5: half is off by thousands of times what float is', drift.includes('after 128 blocks') && sm.h > 1000 * sm.f, drift);
if (out) await page.$eval(S, (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-scene.png' });

// ---------------------------------------------------------------- the bug spot
const G = bench('Safe or unsafe?');
const frags = await model((m) => Object.fromEntries(m.FRAGMENTS.map((id) => [id, m.runFragment(id)])));
await page.waitForFunction((G) => document.querySelectorAll(`${G} [data-frag-code]`).length === 4, {}, G).catch(() => {});
const shown = await page.$$eval(G + ' [data-frag]', (fs) => Object.fromEntries(fs.map((f) => [f.dataset.frag, f.querySelector('[data-frag-code]')?.textContent ?? ''])));
const fileFrags = await model(async (m) => m.parseFragments(await (await fetch('/api/lab-file?path=x2/snippets/accumulators.md')).text()));
check('bug spot: the four fragments are the snippet file\'s, each with the model\'s accumulator',
  fileFrags.length === 4 && fileFrags.every((f) => shown[f.id] === f.code) &&
  shown['short-sum'].startsWith('short sumi') && shown['int-then-float'].startsWith('int sumi') && shown['cuda-hfma2'].includes('__half2 acc') && shown['metal-half-dot'].includes('dot(w, half4('),
  JSON.stringify(Object.keys(shown)));
check('bug spot: Check waits for four marks', !(await click(G, 'Check')));
const mark = (id, safe) => page.evaluate((G, id, safe) => {
  const radios = document.querySelectorAll(`${G} [data-frag="${id}"] input[type=radio]`);
  radios[safe ? 0 : 1].click();
}, G, id, safe);
await mark('metal-half-dot', true); // a wrong mark first
for (const id of ['cuda-hfma2', 'short-sum', 'int-then-float']) await mark(id, frags[id].safe);
await pause(100);
await click(G, 'Check');
check('bug spot: 3 of 4 with one wrong mark', (await attr(G + ' [data-score]', 'data-score')) === '3');
await mark('metal-half-dot', false);
await pause(100);
await click(G, 'Check');
check('bug spot: 4 of 4 when all are right', (await attr(G + ' [data-score]', 'data-score')) === '4');
const v = await page.$$eval(G + ' [data-frag]', (fs) => Object.fromEntries(fs.map((f) => [f.dataset.frag, f.querySelector('[data-verdict]').textContent])));
check('bug spot: only the int-then-float fragment is safe', Object.entries(v).every(([id, t]) => t.includes(frags[id].safe ? ': safe' : ': unsafe')));
check('bug spot: the short wraps on dp4a step 5', v['short-sum'].includes('wraps on dp4a step 5 of 8') && frags['short-sum'].worst.at === 5);
check('bug spot: the __hfma2 lane overflows at the model\'s product', v['cuda-hfma2'].includes(`at product ${frags['cuda-hfma2'].worst.at}`));
check('bug spot: int then float is exact on the worst case', v['int-then-float'].includes((128 * 28448).toLocaleString('en-US')));
if (out) await page.$eval(G, (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-bugspot.png' });

// ---------------------------------------------------------------- choice questions
const choose = async (needle, idx) => {
  const q = await question(needle);
  await (await q.$$('input[type=radio]'))[idx].click();
  await (await q.$('button.primary')).click();
  await pause(150);
  return q.evaluate((e) => e.textContent);
};
check('question q-nibble-int16: the fifth step wraps', (await choose('in a short, four at a time', 0)).includes('Correct.'));
check('question q-half-prescale: rounds to 0', (await choose('An activation of 1e−4 meets', 0)).includes('Correct.'));
check('question q-half-prescale: "exact" is marked wrong', (await choose('An activation of 1e−4 meets', 1)).includes('Not quite.'));
const hp = await model((m) => m.halfPrescaled(1e-4, 12));
check('model: 1e-4 / 4096 is 0 in FP16', hp === 0);

const expected404 = (u) => /\/content\/x2\/narration\/(manifest\.json|narrow-accumulator-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
