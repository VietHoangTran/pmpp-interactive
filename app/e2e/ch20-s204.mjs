// Browser check for section 20.4 ("KV caching"): the which-cells-change grid, the
// recompute-or-cache explorer, the roofline explorer, the decode-stream scene and the
// questions. Every number asserted is computed from the tested model (src/sim/kvcache.ts).
// Usage: node e2e/ch20-s204.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

// The model the page runs on. Node strips the types itself; on a Node too old for that,
// take the same module as the dev server compiles it for the browser.
const load = async (path) => {
  try {
    return await import('../' + path);
  } catch {
    const code = await (await fetch(BASE + '/' + path)).text();
    return import('data:text/javascript;charset=utf-8,' + encodeURIComponent(code));
  }
};
const kv = await load('src/sim/kvcache.ts');

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file, and start from an empty one.
const puts = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
  if (r.method() === 'PUT') {
    puts.push(JSON.parse(r.postData()));
    return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  }
  return r.respond({ status: 200, contentType: 'application/json', body: '{"version":1}' });
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const attr = (sel, name) => page.$eval(sel, (e, n) => e.getAttribute(n), name);
const close = (a, b, rel = 1e-6) => Math.abs(a - b) <= Math.abs(b) * rel + 1e-9;
const setRange = async (sel, value) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(60);
};
const setSelect = async (sel, value) => {
  await page.$eval(sel, (el, v) => {
    el.value = String(v);
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await pause(60);
};
const clickIn = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const root = [...document.querySelectorAll('section.bench')].find((b) => b.getAttribute('aria-label') === scope);
    const btn = [...root.querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(60);
  return done;
};
const question = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const qText = (q) => q.evaluate((e) => e.textContent.replace(/\s+/g, ' '));

await goto(page, 'ch20/20.4');
await page.waitForSelector(bench('Which cells change?'));

// ---------------------------------------------------------------- which cells change
const C = 'Which cells change?';
const diff = kv.diffStep(6, 4, 1);
const need = kv.mustCompute(diff);
const cell = (name, i) => {
  const cols = diff.after[name].cols;
  return `${bench(C)} span[aria-label="${name} row ${Math.floor(i / cols)} column ${i % cols}"]`;
};
check('nothing is marked at first', (await text(bench(C))).includes('0 marked'));
const dashed = await page.$$eval(`${bench(C)} .mx > span.void`, (s) => s.length);
check('the new token’s slots are dashed before the reveal: 5 rows of 4 and 2 of 7 plus 2 columns of 6', dashed === 5 * 4 + 2 * (7 + 6), String(dashed));
for (const name of kv.MAT_NAMES) for (const i of need[name]) await page.click(cell(name, i));
const total = kv.MAT_NAMES.reduce((s, m) => s + need[m].size, 0);
check('the model says 34 cells need arithmetic', total === 34, String(total));
check('marking them counts 34', (await text(bench(C))).includes('34 marked'));
await clickIn(C, 'Reveal');
check('marking exactly the new row of every matrix scores all of them', (await attr(`${bench(C)} [data-score]`, 'data-score')) === '34/34/0/0/0');
check('the verdict says so', (await text(`${bench(C)} .narrate`)).includes('All 34 cells exactly'));
const kinds = await page.$$eval(`${bench(C)} [data-kind]`, (s) => s.map((x) => x.dataset.kind));
const count = (k) => kinds.filter((x) => x === k).length;
check('the reveal classifies 34 new cells, 12 masked, none changed', count('new') === 34 && count('masked') === 12 && count('changed') === 0, JSON.stringify({ n: count('new'), m: count('masked'), c: count('changed') }));
const sMasked = await page.$eval(cell('S', 0 * 7 + 6), (e) => e.textContent);
const pMasked = await page.$eval(cell('P', 0 * 7 + 6), (e) => e.textContent);
check('the masked new column shows −∞ in S and 0 in P', sMasked === '−∞' && pMasked === '0', `${sMasked} ${pMasked}`);
if (out) await page.screenshot({ path: out + '-cells.png' });
await clickIn(C, 'Clear');
await page.click(cell('X', 0));
await page.click(cell('S', 6));
await clickIn(C, 'Reveal');
check('one unchanged and one masked mark, nothing else: 0 hit, 34 missed, 1 extra, 1 masked', (await attr(`${bench(C)} [data-score]`, 'data-score')) === '0/34/34/1/1');
check('the verdict explains the masked mark', (await text(`${bench(C)} .narrate`)).includes('known without arithmetic'));
await clickIn(C, 'Another problem');
check('another problem clears the marks and the reveal', (await text(bench(C))).includes('0 marked') && !(await page.$(`${bench(C)} [data-score]`)));

// ---------------------------------------------------------------- recompute or cache
const R = 'Recompute or cache';
await setRange(`${bench(R)} input[aria-label="Context length"]`, 12);
await setSelect(`${bench(R)} select[aria-label="Width"]`, 4096);
await setRange(`${bench(R)} input[aria-label="Layers"]`, 32);
await setRange(`${bench(R)} input[aria-label="Tokens to generate"]`, 8);
await setSelect(`${bench(R)} select[aria-label="Cache element"]`, 2);
const base = Number(await attr(`${bench(R)} [data-base]`, 'data-base'));
const cached = Number(await attr(`${bench(R)} [data-cached]`, 'data-cached'));
check('per-token FLOPs at N = 4,096 match the model', base === kv.baselineStep(4097, 4096, 32).total && cached === kv.cachedStep(4097, 4096, 32).total, `${base} ${cached}`);
check('recomputing costs exactly n = 4,097 times the cached step', Number(await attr(`${bench(R)} [data-ratio]`, 'data-ratio')) === 4097);
check('the cache held after 256 more tokens matches the model', Number(await attr(`${bench(R)} [data-held]`, 'data-held')) === kv.cacheBytes(4096 + 256, 4096, 32, 2));
await setRange(`${bench(R)} input[aria-label="Context length"]`, 15);
check('at N = 32,768 the ratio is 32,769', Number(await attr(`${bench(R)} [data-ratio]`, 'data-ratio')) === 32769);
if (out) await page.screenshot({ path: out + '-recompute.png' });

// ---------------------------------------------------------------- roofline
const F = 'Prefill and decode on your roofline';
await setSelect(`${bench(F)} select[aria-label="Roof"]`, 'book');
check('the book’s example roof has its ridge at 66.9 / 3.35 = 19.97', close(Number(await attr(`${bench(F)} svg[data-ridge]`, 'data-ridge')), 66900 / 3350, 1e-4));
await setRange(`${bench(F)} input[aria-label="Prompt length"]`, 14);
await setSelect(`${bench(F)} select[aria-label="Element size"]`, 2);
await setSelect(`${bench(F)} select[aria-label="Head width"]`, 128);
const rows = await page.$$eval(`${bench(F)} tr[data-key]`, (r) => r.map((x) => ({ key: x.dataset.key, ai: Number(x.dataset.ai), bound: x.dataset.bound })));
const model = Object.fromEntries(kv.aiPoints(16384, 128, 2).map((x) => [x.key, x.ai]));
check('all four points at N = 16,384 match the model', rows.length === 4 && rows.every((r) => close(r.ai, model[r.key], 1e-3)), JSON.stringify(rows));
const ridge = 66900 / 3350;
check('prefill crosses the book’s ridge, decode does not', rows.every((r) => r.bound === (model[r.key] < ridge ? 'bandwidth' : 'compute')) && rows.find((r) => r.key === 'prefill-qk').bound === 'compute' && rows.find((r) => r.key === 'decode-qk').bound === 'bandwidth');
await setSelect(`${bench(F)} select[aria-label="Element size"]`, 4);
const dq = Number(await attr(`${bench(F)} tr[data-key="decode-qk"]`, 'data-ai'));
check('decode q′Kᵀ in fp32 sits near 2/p = 0.5', close(dq, kv.aiDecode(16384, 128, 4), 1e-3) && dq < 0.5 && dq > 0.49, String(dq));
await setRange(`${bench(F)} input[aria-label="Prompt length"]`, 0);
const pf1 = Number(await attr(`${bench(F)} tr[data-key="prefill-qk"]`, 'data-ai'));
check('a one-token prompt leaves prefill on the slope too', close(pf1, kv.aiPrefill(1, 128, 4), 1e-3) && (await attr(`${bench(F)} tr[data-key="prefill-qk"]`, 'data-bound')) === 'bandwidth');

// ---------------------------------------------------------------- the decode-stream scene
const S = 'section.scene[aria-label="One decode step streams the whole cache"]';
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
const part = async (label) => {
  await page.evaluate((sel, label) => [...document.querySelectorAll(sel + ' .seg button')].find((b) => b.textContent.trim() === label).click(), S, label);
  await pause(100);
};
const sceneAt = async (frac) => setRange(`${S} input[aria-label="Position in this part"]`, Math.round(frac * 1000));
check('six parts are offered', (await page.$$(`${S} .seg button`)).length === 6);
check('there is a captions option', (await text(S)).includes('captions'));
const PH = kv.decodePhases(4096, 32, 128, 2);
await part('1. Projections');
await sceneAt(1);
let g = await page.$eval(`${S} g[data-bytes]`, (e) => ({ b: Number(e.dataset.bytes), f: Number(e.dataset.flops) }));
check('part 1 ends with the projections’ bytes and FLOPs', g.b === Math.round(kv.phaseTotals(PH, 1).bytes) && g.f === Math.round(kv.phaseTotals(PH, 1).flops), JSON.stringify(g));
await part('3. q′ Kᵀ');
await sceneAt(1);
g = await page.$eval(`${S} g[data-bytes]`, (e) => ({ b: Number(e.dataset.bytes), f: Number(e.dataset.flops) }));
check('part 3 ends after the whole K cache has streamed', g.b === Math.round(kv.phaseTotals(PH, 3).bytes) && g.f === Math.round(kv.phaseTotals(PH, 3).flops), JSON.stringify(g));
await part('5. p′ V');
await sceneAt(1);
const ai = Number(await attr(`${S} [data-ai]`, 'data-ai'));
check('a whole decode step does about one FLOP per byte in 16-bit', close(ai, kv.phaseTotals(PH).ai, 1e-3) && ai > 0.99 && ai < 1, String(ai));
const script = (await (await fetch(`${BASE}/content/ch20/narration/decode-stream-5.txt`)).text()).replace(/\[[^\]]*\]\s*/g, '').trim();
await page.evaluate((sel) => {
  const box = [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('captions')).querySelector('input');
  if (!box.checked) box.click();
}, S);
await pause(100);
check('with captions on, part 5 shows its script', (await text(`${S} .caption`)) === script.replace(/\s+/g, ' '));
await part('6. Next token');
await sceneAt(1);
const bars = await page.$$eval(`${S} g[data-n]`, (gs) => gs.map((x) => ({ n: Number(x.dataset.n), b: Number(x.dataset.bytes) })));
check('part 6 draws six context lengths, bytes from the model, growing with N',
  bars.length === 6 && bars.every((x) => x.b === kv.phaseTotals(kv.decodePhases(x.n, 32, 128, 2)).bytes) && bars.every((x, i) => i === 0 || x.b > bars[i - 1].b), JSON.stringify(bars));
if (out) await page.screenshot({ path: out + '-scene.png' });

// ---------------------------------------------------------------- questions
let q = await question('how many FLOPs do q′Kᵀ and p′V take together');
await (await q.$('input[type=number]')).type('2097152');
await (await q.$('button.primary')).click();
await pause(150);
check('numeric question accepts 4Nd = 2,097,152', (await qText(q)).includes('Correct.'));
q = await question('why do rows 0 to N − 1 of P stay exactly the same');
await (await q.$$('input[type=radio]'))[0].click();
await (await q.$('button.primary')).click();
await pause(150);
check('choice question accepts the masked-entry answer', (await qText(q)).includes('Correct.'));
q = await question('is_causal=True');
await (await q.$$('input[type=radio]'))[1].click();
await (await q.$('button.primary')).click();
await pause(150);
check('the is_causal question rejects “all N keys”', (await qText(q)).includes('Not quite.'));
await pause(1200);
check('answers were saved through PUT /api/progress (intercepted)', puts.some((p) => p.answers && p.answers['ch20/20.4/q-new-row-cost']?.correct === true), String(puts.length));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
