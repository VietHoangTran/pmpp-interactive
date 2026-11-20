// Browser check for section 20.6 ("KV cache arithmetic intensity and memory requirement"):
// the batching explorer, the batch-share scene, the cache-size explorer, the speculative
// decoding stepper (both modes), the lab 20.4 prediction and the questions. Every number
// asserted is computed from the tested models (src/sim/kvmath.ts, src/sim/specdecode.ts).
// Usage: node e2e/ch20-s206.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

const load = async (path) => {
  try {
    return await import('../' + path);
  } catch {
    const code = await (await fetch(BASE + '/' + path)).text();
    return import('data:text/javascript;charset=utf-8,' + encodeURIComponent(code));
  }
};
const km = await load('src/sim/kvmath.ts');
const sd = await load('src/sim/specdecode.ts');

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
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
const setNumber = setRange;
const setSelect = async (sel, value) => {
  await page.$eval(sel, (el, v) => {
    el.value = String(v);
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await pause(80);
};
const clickIn = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const root = [...document.querySelectorAll('section.bench')].find((b) => b.getAttribute('aria-label') === scope);
    const btn = [...root.querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(80);
  return done;
};
const question = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const qText = (q) => q.evaluate((e) => e.textContent.replace(/\s+/g, ' '));

await goto(page, 'ch20/20.6');
await page.waitForSelector(bench('Batching raises one intensity, not the other'));

// ---------------------------------------------------------------- batching
const B = 'Batching raises one intensity, not the other';
const lin = () => attr(`${bench(B)} [data-linear]`, 'data-linear').then(Number);
const att = () => attr(`${bench(B)} [data-attn]`, 'data-attn').then(Number);
check('batch 1: the book’s linear layer does 0.9988 FLOP/B', close(await lin(), km.linearAI(1, 1024, 4096, 2), 1e-4) && close(await lin(), 0.9988, 1e-4), String(await lin()));
await setRange(`${bench(B)} input[aria-label="Batch size"]`, 9);
check('batch 512: 315.08 FLOP/B, the book’s 315', close(await lin(), 315.08, 1e-4), String(await lin()));
check('decode attention has not moved', close(await att(), km.batchedDecodeAttentionAI(512, 4096, 2), 1e-4) && close(await att(), km.batchedDecodeAttentionAI(1, 4096, 2), 1e-4), String(await att()));
const cross = await page.$$eval(`${bench(B)} tr[data-cross]`, (r) => r.map((x) => x.dataset.cross));
const bookCross = km.batchToReach(66900 / 3350, 1024, 4096, 2);
check('the linear layer crosses the book’s ridge at the batch the model computes', cross[cross.length - 1] === String(bookCross), `${cross} vs ${bookCross}`);
check('attention never crosses any ridge', (await page.$$eval(`${bench(B)} tr[data-cross] td:last-child`, (t) => t.map((x) => x.textContent))).every((x) => x === 'never'));
await setSelect(`${bench(B)} select[aria-label="Element size"]`, 4);
check('in fp32 decode attention sits at 0.5', close(await att(), 0.5, 1e-3), String(await att()));
await setSelect(`${bench(B)} select[aria-label="Linear layer"]`, 'qkv');
check('the 4,096 → 12,288 layer at batch 512 in fp32 matches the model', close(await lin(), km.linearAI(512, 4096, 12288, 4), 1e-4));
if (out) await page.screenshot({ path: out + '-batching.png' });

// ---------------------------------------------------------------- batch-share scene
const S = 'section.scene[aria-label="Batching shares weights, not caches"]';
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
const part = async (label) => {
  await page.evaluate((sel, label) => [...document.querySelectorAll(sel + ' .seg button')].find((b) => b.textContent.trim() === label).click(), S, label);
  await pause(100);
};
const sceneAt = (frac) => setRange(`${S} input[aria-label="Position in this part"]`, Math.round(frac * 1000));
check('five parts are offered', (await page.$$(`${S} .seg button`)).length === 5);
await part('1. One request');
let svg = await page.$eval(`${S} svg`, (e) => ({ b: e.dataset.batch, p: Number(e.dataset.projAi), a: Number(e.dataset.attnAi) }));
check('part 1: one request, both intensities near 1', svg.b === '1' && close(svg.p, km.linearAI(1, 4096, 12288, 2), 1e-3) && close(svg.a, km.batchedDecodeAttentionAI(1, 1024, 2), 1e-3), JSON.stringify(svg));
await part('2. Four share the weights');
await sceneAt(1);
svg = await page.$eval(`${S} svg`, (e) => ({ b: e.dataset.batch, p: Number(e.dataset.projAi), a: Number(e.dataset.attnAi) }));
check('part 2: four rows share the weights, projections about four times as intense', svg.b === '4' && close(svg.p, km.linearAI(4, 4096, 12288, 2), 1e-3) && svg.p > 3.9, JSON.stringify(svg));
await part('3. Four caches');
await sceneAt(1);
svg = await page.$eval(`${S} svg`, (e) => ({ b: e.dataset.batch, a: Number(e.dataset.attnAi) }));
check('part 3: four caches, attention still near 1', svg.b === '4' && close(svg.a, km.batchedDecodeAttentionAI(4, 1024, 2), 1e-3), JSON.stringify(svg));
await part('4. Grow the batch');
await sceneAt(1);
check('part 4 ends at batch 64', (await attr(`${S} svg`, 'data-batch')) === '64');
await part('5. The trade-off');
await sceneAt(1);
check('part 5: one GiB holds 2,048 tokens of Llama 2 7B cache', Number(await attr(`${S} svg`, 'data-tokens-per-gib')) === km.tokensThatFit(km.GiB, km.presetByKey('llama2-7b').shape, 2));
const rows5 = await page.$$eval(`${S} g[data-b]`, (g) => g.map((x) => Number(x.dataset.b) * Number(x.dataset.n)));
check('every row of part 5 has b × N = 2,048', rows5.length === 5 && rows5.every((v) => v === 2048), JSON.stringify(rows5));

// ---------------------------------------------------------------- size the cache
const Z = 'Size the cache';
await page.$eval(bench(Z), (e) => e.scrollIntoView({ block: 'start' }));
const num = async (a) => Number(await attr(`${bench(Z)} [${a}]`, a));
check('Llama 2 7B: 512 KiB per token, 2 GiB for 4,096 tokens', (await num('data-per-token')) === 512 * 1024 && (await num('data-total')) === 2 * km.GiB);
await setSelect(`${bench(Z)} select[aria-label="Model"]`, 'gpt3');
check('GPT-3 at its 2,048-token window: 9 GiB', (await num('data-total')) === 9 * km.GiB, String(await num('data-total')));
await clickIn(Z, 'the book’s 4,096');
check('GPT-3 at the book’s 4,096 tokens: 18 GiB', (await num('data-total')) === 18 * km.GiB);
await setSelect(`${bench(Z)} select[aria-label="Model"]`, 'palm2-printed');
check('the book’s “PaLM 2” row as printed: 8 GiB', (await num('data-total')) === 8 * km.GiB);
check('… and the explorer says the printed equation agrees for MHA', (await text(bench(Z))).includes('the book’s equation gives the same'));
await setSelect(`${bench(Z)} select[aria-label="Model"]`, 'palm-62b');
check('PaLM 62B with multi-query attention: 256 MiB', (await num('data-total')) === 256 * km.MiB);
await setSelect(`${bench(Z)} select[aria-label="Model"]`, 'deepseek-v2');
check('DeepSeek-V2: 69,120 bytes per token, and the printed Eq. 20.14 shown beside it', (await num('data-per-token')) === 69120 && (await text(bench(Z))).includes('printed equation gives 15 KiB'));
await setSelect(`${bench(Z)} select[aria-label="Model"]`, 'llama2-7b');
await setSelect(`${bench(Z)} select[aria-label="Device memory"]`, 'custom');
await setNumber(`${bench(Z)} input[type=number]:not([aria-label])`, 22);
const mb = await num('data-max-batch');
const expect = km.maxBatch(22 * km.GiB, 7e9 * 2, km.presetByKey('llama2-7b').shape, 4096, 2);
check('22 GiB with 7B 16-bit weights fits the model’s number of 4,096-token sequences', mb === expect && mb === 4, `${mb} vs ${expect}`);
await setRange(`${bench(Z)} input[aria-label="Batch"]`, 3);
check('a batch of 8 does not fit there', (await attr(`${bench(Z)} [data-fits]`, 'data-fits')) === 'no');
await setSelect(`${bench(Z)} select[aria-label="Cache element"]`, 'q4_0');
check('q4_0 cache: 18 bytes per 32 values', (await num('data-per-token')) === km.bytesPerToken(km.presetByKey('llama2-7b').shape, 18 / 32));
if (out) await page.screenshot({ path: out + '-size.png' });

// ---------------------------------------------------------------- speculative decoding
const P = 'Speculative decoding';
await page.$eval(bench(P), (e) => e.scrollIntoView({ block: 'start' }));
for (let i = 0; i < 3; i++) await clickIn(P, 'Step');
check('round 1 accepts Programming Massively', (await attr(`${bench(P)} [data-round="1"] [data-accepted]`, 'data-accepted')) === sd.replayFigure()[0].accepted.join(' '));
const verdicts = await page.$$eval(`${bench(P)} [data-round="1"] [data-verify]`, (v) => v.map((x) => x.dataset.verify));
check('round 1 shows two ticks and a cross', JSON.stringify(verdicts) === JSON.stringify(['ok', 'ok', 'bad']), JSON.stringify(verdicts));
for (let i = 0; i < 5; i++) await clickIn(P, 'Step');
const fc = sd.figCounters(6);
check('the replay ends with 2 target passes for 5 tokens', (await attr(`${bench(P)} [data-counters]`, 'data-counters')) === `${fc.targetPasses}/${fc.tokens.length}` && fc.tokens.length === 5);
await clickIn(P, 'Your own');
await setRange(`${bench(P)} input[aria-label="Speculation depth"]`, 4);
await setRange(`${bench(P)} input[aria-label="Acceptance probability"]`, 0.7);
const sim = sd.simulate({ k: 4, alpha: 0.7, seed: 1, tokens: 64, keepTarget: false });
check('your own run matches the simulator (target passes)', Number(await attr(`${bench(P)} [data-passes]`, 'data-passes')) === sim.targetPasses, `${await attr(`${bench(P)} [data-passes]`, 'data-passes')} vs ${sim.targetPasses}`);
check('the closed form is shown beside it', close(Number(await attr(`${bench(P)} [data-expected]`, 'data-expected')), sd.expectedPerPass(4, 0.7, false), 1e-3));
await page.evaluate((sel) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('keep the target'))?.querySelector('input').click(), bench(P));
await pause(80);
check('keeping the target’s token changes the expectation to (1 − α^k)/(1 − α)', close(Number(await attr(`${bench(P)} [data-expected]`, 'data-expected')), sd.expectedPerPass(4, 0.7, true), 1e-3));
await setRange(`${bench(P)} input[aria-label="Acceptance probability"]`, 1);
check('with α = 1 every pass keeps k = 4 tokens', Number(await attr(`${bench(P)} [data-per-pass]`, 'data-per-pass')) === 4);
if (out) await page.screenshot({ path: out + '-spec.png' });

// ---------------------------------------------------------------- the lab 20.4 prediction
const pq = await question('Lab 20.4’s decode kernel reads an fp32 cache');
await (await pq.$('input[type=number]')).type('60');
await (await pq.$('button.primary')).click();
await pause(400);
const pt = await qText(pq);
check('committing a prediction reveals the measurement or how to make one', pt.includes('You predicted 60') && (pt.includes('No measurement on this machine yet') || pt.includes('GB/s')), pt.slice(0, 200));

// ---------------------------------------------------------------- questions
let q = await question('on a batch of 512 rows');
await (await q.$('input[type=number]')).type('315.08');
await (await q.$('button.primary')).click();
await pause(150);
check('numeric question accepts 315.08', (await qText(q)).includes('Correct.'));
q = await question('at its real context window of 2,048 tokens');
await (await q.$('input[type=number]')).type('18');
await (await q.$('button.primary')).click();
await pause(150);
check('numeric question rejects the book’s 18', (await qText(q)).includes('Not quite. The answer is 9 GiB.'));
q = await question('What is wrong with it?');
await (await q.$$('input[type=radio]'))[0].click();
await (await q.$('button.primary')).click();
await pause(150);
check('choice question accepts the PaLM 62B correction', (await qText(q)).includes('Correct.'));
await pause(1200);
check('answers were saved through PUT /api/progress (intercepted)', puts.some((p) => p.answers && p.answers['ch20/20.6/q-linear-ai']?.correct === true && p.answers['ch20/20.6/p-decode-bw']?.value === 60), String(puts.length));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
