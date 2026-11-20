// Browser check for section 20.7 ("Alleviating the memory requirements of the attention
// mechanism"): the head-sharing explorer, the MHA/GQA/MQA/MLA comparison, the contiguous-
// or-paged stepper, the paged-blocks scene and the questions. Every number asserted is
// computed from the tested models (src/sim/kvmath.ts, src/sim/paged.ts).
// Usage: node e2e/ch20-s207.mjs [screenshot-prefix]
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
const pg = await load('src/sim/paged.ts');

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

await goto(page, 'ch20/20.7');
await page.waitForSelector(bench('Who shares a K/V head'));

// ---------------------------------------------------------------- who shares a K/V head
const H = 'Who shares a K/V head';
const svgH = `${bench(H)} svg[data-hq]`;
const read = (a) => attr(`${bench(H)} [${a}]`, a);
check('starts as Fig. 20.19(c): 6 query heads, 3 K/V heads', (await attr(svgH, 'data-hq')) === '6' && (await attr(svgH, 'data-hkv')) === '3' && (await attr(svgH, 'data-variant')) === 'GQA');
check('query head 4 reads K/V head 2 with 3 K/V heads', (await read('data-reads')) === String(km.kvHeadOf(4, 6, 3)));
await setSelect(`${bench(H)} select[aria-label="K/V heads"]`, 2);
check('FlashAttention’s example: with 2 K/V heads, query head 4 reads K/V head 1', (await read('data-reads')) === '1' && km.kvHeadOf(4, 6, 2) === 1);
await page.click(`${bench(H)} g[data-qhead="2"]`);
await pause(60);
check('clicking query head 2 shows K/V head 0', (await read('data-reads')) === '0');
check('g_q = 3 and G = 2 side by side', (await read('data-gq')) === '3' && (await read('data-groups')) === '2');
const lines = await page.$$eval(`${bench(H)} g[data-kvhead]`, (g) => g.length);
check('two K/V heads are drawn', lines === 2, String(lines));
await clickIn(H, 'MQA');
check('MQA: one K/V head', (await attr(svgH, 'data-hkv')) === '1' && (await read('data-reads')) === '0');
await clickIn(H, 'MLA');
check('MLA: one latent of 512 + 64 values, intensity 2N h_q / (2 h_q + N)', (await read('data-values')) === '576' && close(Number(await read('data-ai')), km.aiMLA(4096, 6), 1e-3));
await clickIn(H, 'MHA');
check('MHA: every head its own K and V', (await attr(svgH, 'data-hkv')) === '6' && (await read('data-values')) === String(2 * 6 * 128));
await setRange(`${bench(H)} input[aria-label="Query heads"]`, 32);
await clickIn(H, 'GQA');
await setSelect(`${bench(H)} select[aria-label="K/V heads"]`, 8);
check('Mistral’s shape: g_q = 4, G = 8, Eq. 20.13 gives 3.996', (await read('data-gq')) === '4' && (await read('data-groups')) === '8' && (await read('data-ai')) === km.aiGQA(4096, 32, 4).toFixed(3) && (await read('data-ai')) === '3.996');
await setRange(`${bench(H)} input[aria-label="Query heads"]`, 7);
check('a prime number of heads has no GQA: the button is disabled', await page.$eval(`${bench(H)} .seg button:nth-child(2)`, (b) => b.disabled));
if (out) await page.screenshot({ path: out + '-heads.png' });

// ---------------------------------------------------------------- compare
const C = 'Four ways to cache K and V';
await page.$eval(bench(C), (e) => e.scrollIntoView({ block: 'start' }));
const row = (v) => page.$eval(`${bench(C)} tr[data-variant="${v}"]`, (r) => ({ ...r.dataset }));
let mha = await row('MHA');
let mla = await row('MLA');
check('DeepSeek-V2: MHA equivalent 3.75 MiB per token, MLA 69,120 bytes', Number(mha.bytes) === 3.75 * km.MiB && Number(mla.bytes) === 69120, JSON.stringify([mha.bytes, mla.bytes]));
check('… the book’s Eq. 20.14 gives 15,360', Number(mla.printed) === 15360);
check('… MLA intensity per [13] beside the printed Eq. 20.15', mla.ai === km.aiMLA(4096, 128).toFixed(3) && mla.aiPrinted === km.aiMLAPrinted(4096, 128).toFixed(3) && mla.ai !== mla.aiPrinted);
check('… and the bar says about 57 times smaller', (await text(`${bench(C)} tr[data-variant="MLA"]`)).includes('56.9× smaller than MHA'));
await setSelect(`${bench(C)} select[aria-label="Model to compare"]`, 'mistral-7b');
await setSelect(`${bench(C)} select[aria-label="GQA K/V heads"]`, 8);
const gqa = await row('GQA');
check('Mistral 7B with 8 K/V heads: 128 KiB per token, intensity 3.996 at 4,096 tokens', Number(gqa.bytes) === 128 * 1024 && gqa.ai === '3.996', JSON.stringify(gqa));
await setRange(`${bench(C)} input[aria-label="Cached tokens"]`, 12);
const mqa = await row('MQA');
check('MQA for the same model: Eq. 20.11 at N = 4,096', mqa.ai === km.aiMQA(4096, 32).toFixed(3));

// ---------------------------------------------------------------- contiguous or paged
const P = 'Contiguous or paged';
await page.$eval(bench(P), (e) => e.scrollIntoView({ block: 'start' }));
const cfg = { pool: 1024, block: 16, maxSeq: 256, requests: 16, seed: 1 };
let run = pg.simulate(cfg);
const side = async (name) => page.$eval(`${bench(P)} [data-pool="${name}"]`, (d) => {
  const v = (a) => Number(d.querySelector(`[data-${a}]`).getAttribute(`data-${a}`));
  return { reserved: v('reserved'), used: v('used'), wasted: v('wasted'), running: v('running'), waiting: v('waiting'), done: v('done'), admitted: v('admitted'), evictions: v('evictions') };
});
const expectSide = (f) => ({ reserved: f.reserved, used: f.used, wasted: f.wasted, running: f.live.length, waiting: f.waiting.length, done: f.done, admitted: f.admitted, evictions: f.evictions });
check('nothing has arrived at step 0', (await text(`${bench(P)} [data-events]`)).includes('Nothing has arrived yet'));
for (let i = 0; i < 5; i++) await clickIn(P, 'Step');
check('after 5 steps the contiguous counters match the model', JSON.stringify(await side('contiguous')) === JSON.stringify(expectSide(run.frames[5].contiguous)), JSON.stringify([await side('contiguous'), expectSide(run.frames[5].contiguous)]));
check('after 5 steps the paged counters match the model', JSON.stringify(await side('paged')) === JSON.stringify(expectSide(run.frames[5].paged)), JSON.stringify([await side('paged'), expectSide(run.frames[5].paged)]));
const owners = await page.$$eval(`${bench(P)} [data-pool="paged"] g[data-block]`, (g) => g.length);
check('the paged pool draws one block per allocated block', owners === run.frames[5].paged.live.reduce((s, r) => s + r.blocks.length, 0), String(owners));
await clickIn(P, 'Next event');
const stepNow = Number(await attr(`${bench(P)} [data-step]`, 'data-step'));
const nextEv = run.frames.findIndex((f, i) => i > 5 && (f.paged.events.length || f.contiguous.events.length));
check('next event jumps to the model’s next step with an event', stepNow === run.frames[nextEv].step, `${stepNow} vs ${nextEv}`);
const evText = await text(`${bench(P)} [data-events]`);
check('and names what happened', run.frames[nextEv].paged.events.every((e) => evText.includes(e)), evText);
const lastIdx = run.frames.length - 1;
await setRange(`${bench(P)} input[aria-label="Step"]`, lastIdx);
check('at the end both sides have finished all 16 requests', (await side('contiguous')).done === 16 && (await side('paged')).done === 16);
check('the finish steps are the model’s, paged well before contiguous',
  Number(await attr(`${bench(P)} [data-finished-paged]`, 'data-finished-paged')) === run.finishedAt.paged &&
  Number(await attr(`${bench(P)} [data-finished-contiguous]`, 'data-finished-contiguous')) === run.finishedAt.contiguous &&
  run.finishedAt.paged < run.finishedAt.contiguous, JSON.stringify(run.finishedAt));
const mid = Math.floor(lastIdx / 4);
await setRange(`${bench(P)} input[aria-label="Step"]`, mid);
const pm = await side('paged');
check('paged waste stays below one block per running request', pm.wasted <= pm.running * (cfg.block - 1) && JSON.stringify(pm) === JSON.stringify(expectSide(run.frames[mid].paged)), JSON.stringify(pm));
await setSelect(`${bench(P)} select[aria-label="Block size"]`, 64);
run = pg.simulate({ ...cfg, block: 64 });
check('changing the block size restarts the run at step 0', Number(await attr(`${bench(P)} [data-step]`, 'data-step')) === 0);
for (let i = 0; i < 8; i++) await clickIn(P, 'Step');
check('with 64-token blocks the paged counters still match the model', JSON.stringify(await side('paged')) === JSON.stringify(expectSide(run.frames[8].paged)));
if (out) await page.screenshot({ path: out + '-paged.png' });

// ---------------------------------------------------------------- paged-blocks scene
const S = 'section.scene[aria-label="Paging the KV cache"]';
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
const part = async (label) => {
  await page.evaluate((sel, label) => [...document.querySelectorAll(sel + ' .seg button')].find((b) => b.textContent.trim() === label).click(), S, label);
  await pause(100);
};
const sceneAt = (frac) => setRange(`${S} input[aria-label="Position in this part"]`, Math.round(frac * 1000));
const SC = { pool: 256, block: 32, maxSeq: 128, requests: 10, seed: 12 };
const srun = pg.simulate(SC);
const reuse = pg.firstReuse(srun);
check('five parts are offered', (await page.$$(`${S} .seg button`)).length === 5);
await part('1. Contiguous');
await sceneAt(1);
const c1 = await page.$eval(`${S} g[data-side="contiguous"]`, (g) => ({ live: Number(g.dataset.live), reserved: Number(g.dataset.reserved) }));
check('part 1: two max_seq slots, both taken', c1.live === 2 && c1.reserved === 2 * SC.maxSeq, JSON.stringify(c1));
await part('3. Finish and reuse');
await sceneAt(1);
check('part 3 shows the model’s first finish-and-admit step', Number(await attr(`${S} svg[data-frame]`, 'data-frame')) === srun.frames[reuse].step && reuse > 0);
const ev = await text(`${S} [data-events]`);
check('… and names the finished and the admitted request', srun.frames[reuse].paged.events.every((e) => ev.includes(e)), ev);
await sceneAt(0);
check('just before, the admitted request is in the paged queue', (await text(`${S} svg`)).includes(`paged: r${srun.frames[reuse].paged.events.find((e) => e.endsWith(' admitted')).slice(1).split(' ')[0]}`));
await part('4. Growing');
await sceneAt(1);
const f4 = Number(await attr(`${S} svg[data-frame]`, 'data-frame'));
const scat = srun.frames[f4].paged.live.filter((q) => pg.isScattered(q.blocks));
const tables = await text(`${S} [data-tables]`);
check('part 4 ends on block tables that are scattered, as drawn', scat.length >= 2 && scat.every((q) => tables.includes(`r${q.id}: [${q.blocks.join(', ')}]`)), tables);
await part('5. Side by side');
await sceneAt(1);
check('part 5: the finish steps and peaks are the model’s',
  Number(await attr(`${S} svg[data-finished-paged]`, 'data-finished-paged')) === srun.finishedAt.paged &&
  Number(await attr(`${S} svg[data-finished-contiguous]`, 'data-finished-contiguous')) === srun.finishedAt.contiguous &&
  Number(await attr(`${S} svg[data-peak-paged]`, 'data-peak-paged')) === pg.peakLive(srun).paged);
check('… and paging finishes in under half the steps, as the narration says', srun.finishedAt.paged < srun.finishedAt.contiguous / 2);
if (out) await page.screenshot({ path: out + '-scene.png' });

// ---------------------------------------------------------------- questions
let q = await question('What is g_q as Eqs. 20.12 and 20.13 use it?');
await (await q.$('input[type=number]')).type('4');
await (await q.$('button.primary')).click();
await pause(150);
check('q-gq accepts 4', (await qText(q)).includes('Correct.'));
q = await question('Which K/V head does query head 4 read?');
await (await q.$('input[type=number]')).type('1');
await (await q.$('button.primary')).click();
await pause(150);
check('q-head-map accepts 1', (await qText(q)).includes('Correct.'));
q = await question('How many bytes per token?');
await (await q.$('input[type=number]')).type('15360');
await (await q.$('button.primary')).click();
await pause(150);
check('q-mla-bytes rejects the printed 15,360', (await qText(q)).includes('Not quite. The answer is 69120 bytes.'));
q = await question('where are PagedAttention’s KV blocks');
await (await q.$$('input[type=radio]'))[0].click();
await (await q.$('button.primary')).click();
await pause(150);
check('q-paged accepts “in GPU memory, anywhere”', (await qText(q)).includes('Correct.'));
await pause(1200);
check('answers were saved through PUT /api/progress (intercepted)', puts.some((p) => p.answers && p.answers['ch20/20.7/q-gq']?.correct === true), String(puts.length));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
