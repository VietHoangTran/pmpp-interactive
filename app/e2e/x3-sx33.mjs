// Browser check for section X3.3, "Row reductions: RMSNorm and softmax": the quoted
// excerpts and lab 10.3's functions beside them, the stepper "One row, two softmaxes" on both
// paths with its predictions, the classification "Sort the kernels", the narrated scene "One
// framework, two softmaxes" (captions option, every part's drawing against the model) and
// the questions. Expected values come from src/sim/rowreduce.ts, run inside the page through
// the app's module server.
// Usage: APP_URL=http://localhost:5173 node e2e/x3-sx33.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
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
const ST = 'section.bench[aria-label="One row, two softmaxes"]';
const SK = 'section.bench[aria-label="Sort the kernels"]';
const SC = 'section.scene[aria-label="One framework, two softmaxes"]';
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const press = async (scope, label) => {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(80);
  return ok;
};
const disabled = (scope, label) =>
  page.evaluate((scope, label) => [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === label)?.disabled ?? null, scope, label);
const f4 = (v) => String(Number(v.toFixed(4)));
const cell = (k) => page.$eval(`${ST} td[data-k="${k}"]`, (e) => e.textContent.trim());
const setNumber = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(60);
};

await goto(page, 'x3/X3.3');
await page.waitForSelector(ST);
await page.waitForFunction(() => document.body.textContent.includes('lab 10.3, blockReduceSum'), { timeout: 15000 });

const M = await page.evaluate(async () => {
  const m = await import('/src/sim/rowreduce.ts');
  const tp = m.twoPass(m.ROW16);
  const on = m.online(m.ROW16);
  const asks = m.predictionSteps(on).map((i) => ({ step: i, sum: on.steps[i].sum, before: on.steps[i].before, chunk: on.steps[i].chunk }));
  return {
    tp: { steps: tp.steps.length, sum: tp.sum, out: tp.out, reads: tp.rowReads, exps: tp.counters.exps, max: tp.max },
    on: { steps: on.steps.length, sum: on.sum, out: on.out, reads: on.rowReads, exps: on.counters.exps, rises: on.rises },
    asks,
    ulps: m.maxUlps(tp.out, on.out),
    kernels: m.KERNELS.map((k) => ({ id: k.id, structure: k.structure, exp: k.exp })),
  };
});

// ---------------------------------------------------------------- excerpts
const codes = await page.$$eval('.code', (cs) => cs.map((c) => c.textContent));
check('llama.cpp rms_norm_f32 is quoted', codes.some((c) => c.includes('block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum)')));
check('MLX online normaliser is quoted', codes.some((c) => c.includes('normalizer * softmax_exp(prevmax - maxval)')));
check("lab 10.3's blockReduceSum is loaded from the lab file", codes.some((c) => c.includes('__device__ float blockReduceSum') && c.includes('laneIdx() < warpsInBlock() ? scratch[laneIdx()] : 0.0f')));
check("lab 10.3's softmaxRowsKernel is loaded with its barrier", codes.some((c) => c.includes('softmaxRowsKernel') && c.includes('__syncthreads();')));
const caps = await page.$$eval('figure.x3-excerpt', (f) => f.map((x) => x.querySelector('figcaption').textContent));
check('each excerpt names its licence and copyright, with a permalink', caps.length === 2 && caps[0].includes('Copyright (c) 2023-2026 The ggml authors') && caps[1].includes('Copyright © 2023 Apple Inc.') && (await page.$$('figure.x3-excerpt a[data-permalink]')).length === 2, JSON.stringify(caps));

// ---------------------------------------------------------------- stepper: max, then sum
await page.$eval(ST, (e) => e.scrollIntoView({ block: 'start' }));
for (let i = 0; i < 4; i++) await press(ST, 'Step');
check('after pass 1 the running max is the row max', (await cell('max')) === String(M.tp.max), await cell('max'));
check('after pass 1 the row has been read once', (await cell('reads')) === '1', await cell('reads'));
check('pass 1 computes no exponential', (await cell('exps')) === '0');
await press(ST, 'Step');
check('pass 2 narrates the sum against the max', (await text(`${ST} [data-part="narrate"]`)).includes('exp(x − 4)'));
if (out) await page.screenshot({ path: out + '-stepper-twopass.png' });
await press(ST, 'Run to the end');
check('two-pass ends after its steps', (await text(ST)).includes(`step ${M.tp.steps} of ${M.tp.steps}`));
check('two-pass read the row three times', (await cell('reads')) === '3', await cell('reads'));
check('two-pass computed 32 exponentials', (await cell('exps')) === String(M.tp.exps));
check('two-pass wrote 64 bytes', (await cell('bytes')).endsWith('/ 64'), await cell('bytes'));
const outs = await page.$$eval(`${ST} .mx.global span[data-col]`, (s) => s.slice(16).map((x) => x.textContent));
check('the written outputs are the model\'s', outs.every((v, i) => v === M.tp.out[i].toFixed(4)), outs.join(' '));
check('the comparison reports the largest difference in ulps', (await page.$eval(`${ST} [data-part="verdict"]`, (e) => Number(e.dataset.maxulps))) === M.ulps);
check('the comparison table has a row per element', (await page.$$(`${ST} [data-part="compare"] tbody tr`)).length === 16);

// ---------------------------------------------------------------- stepper: online, with predictions
await press(ST, 'online max and sum (MLX CUDA)');
check('switching path restarts', (await text(ST)).includes(`step 0 of ${M.on.steps}`));
await press(ST, 'Run to the next prediction');
check('the stepper stops before the first rescale after chunk 0', (await text(ST)).includes(`step ${M.asks[0].step} of ${M.on.steps}`));
check('a prediction is asked for', !!(await page.$(`${ST} [data-part="predict"]`)));
check('Step is disabled until you predict or skip', (await disabled(ST, 'Step')) === true);
check('the prompt shows the normaliser before the rescale', (await text(`${ST} [data-part="predict"]`)).includes(f4(M.asks[0].before)));
await setNumber(`${ST} input[aria-label="predicted normaliser"]`, 1.83);
await press(ST, 'Check and step');
const n1 = await text(`${ST} [data-part="narrate"]`);
check('a wrong prediction is marked wrong, with the model value', n1.includes('You predicted 1.83') && n1.includes(f4(M.asks[0].sum)), n1);
check('the rescale is shown', n1.includes('normalizer × exp('));
await press(ST, 'Run to the next prediction');
await setNumber(`${ST} input[aria-label="predicted normaliser"]`, Number(M.asks[1].sum.toFixed(3)));
await press(ST, 'Check and step');
const n2 = await text(`${ST} [data-part="narrate"]`);
check('a right prediction is marked right', !!(await page.$(`${ST} [data-verdict="right"]`)), n2);
check('a factor of exactly 1 is pointed out', n2.includes('exactly 1'), n2);
await press(ST, 'Run to the next prediction');
await press(ST, 'Skip');
check('a skipped prediction says so', (await text(`${ST} [data-part="narrate"]`)).includes('You skipped this prediction.'));
if (out) await page.screenshot({ path: out + '-stepper-online.png' });
await press(ST, 'Run to the next prediction');
check('online ends after its steps', (await text(ST)).includes(`step ${M.on.steps} of ${M.on.steps}`));
check('online read the row twice', (await cell('reads')) === '2', await cell('reads'));
check('online computed 36 exponentials', (await cell('exps')) === String(M.on.exps));
check('the end shows both paths\' traffic', (await text(`${ST} [data-part="compare"]`)).includes('online: 2 reads, 1 write, 36 exponentials'));
const sumCell = await cell('sum');
check('the final normaliser is the model\'s', sumCell === f4(M.on.sum), sumCell);
if (out) await page.screenshot({ path: out + '-stepper-compare.png' });

// ---------------------------------------------------------------- classification
await page.$eval(SK, (e) => e.scrollIntoView({ block: 'start' }));
for (const k of M.kernels) {
  await page.$eval(`${SK} [data-kernel="${k.id}"] select[aria-label^="reductions"]`, (el, v) => {
    el.value = v;
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, k.structure);
  await page.$eval(`${SK} [data-kernel="${k.id}"] select[aria-label^="exponential"]`, (el, v) => {
    el.value = v;
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, k.exp);
}
await press(SK, 'Check');
check('all fourteen answers from the model score fourteen', (await text(`${SK} [data-part="score"]`)) === '14 of 14 right.', await text(`${SK} [data-part="score"]`));
check('checking shows why, with the facts item', (await text(SK)).includes('softmax_exp is __expf'));
await page.$eval(`${SK} [data-kernel="mlx-metal-softmax"] select[aria-label^="exponential"]`, (el) => {
  el.value = 'precise::exp';
  el.dispatchEvent(new Event('change', { bubbles: true }));
});
await press(SK, 'Check');
check('a wrong exponential for MLX Metal is marked with the right one', (await text(`${SK} [data-kernel="mlx-metal-softmax"]`)).includes('fast::exp') && (await text(`${SK} [data-part="score"]`)) === '13 of 14 right.');

// ---------------------------------------------------------------- the scene
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const scripts = [];
for (let n = 1; n <= 5; n++) scripts.push(strip(await (await fetch(`${BASE}/content/x3/narration/two-softmaxes-${n}.txt`)).text()));
check('every script is 35 to 60 words', scripts.every((s) => s.split(' ').length >= 35 && s.split(' ').length <= 60), scripts.map((s) => s.split(' ').length).join(' '));
const res = await fetch(BASE + '/content/x3/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const anyAudio = manifest && Object.keys(manifest.clips ?? {}).some((k) => k.startsWith('two-softmaxes-'));
await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' }));
const LABELS = ['Find the max', 'Sum, then write', 'Online', 'Write', 'Compared'];
check('five parts with the planned labels', JSON.stringify(await page.$$eval(SC + ' .seg button', (b) => b.map((x) => x.textContent.replace(/^\d+\.\s*/, '').trim()))) === JSON.stringify(LABELS));
const capBox = () => page.$eval(SC, (e) => [...e.querySelectorAll('label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').checked);
const caption = () => page.$eval(SC + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim()).catch(() => null);
check('the captions option exists and is off by default', (await capBox()) === false);
if (anyAudio) check('captions off: no caption shown', (await caption()) === null);
else check("no audio yet: part 1's script is shown regardless", (await caption()) === scripts[0], String(await caption()));
await page.evaluate((SC) => [...document.querySelectorAll(SC + ' label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').click(), SC);
await pause(120);
check('captions on: the script is shown', (await caption()) === scripts[0]);
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(SC);
await pause(500);
check('the captions choice is remembered after a reload', await capBox());
await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' }));
const seek = async (v) => {
  await setNumber(SC + ' input[type=range]', v);
  await pause(80);
};
const part = async (label) => {
  await press(SC, label);
  await pause(80);
};
const state = () => page.$eval(SC + ' [data-part="state"]', (e) => ({ max: Number(e.dataset.max), sum: Number(e.dataset.sum), reads: Number(e.dataset.reads), writes: Number(e.dataset.writes) }));

await part('1. Find the max');
await seek(1000);
let s = await state();
check('part 1 ends with the row max after one read', s.max === M.tp.max && s.reads === 1 && s.writes === 0, JSON.stringify(s));
await seek(0);
check('part 1 starts on chunk 0', (await page.$$eval(SC + ' [data-hot="1"]', (g) => g.map((x) => Number(x.dataset.col)))).join() === '0,1,2,3');
await part('2. Sum, then write');
await seek(1000);
s = await state();
check('part 2 ends after three reads and one write', s.reads === 3 && s.writes === 1, JSON.stringify(s));
check('part 2 ends with the model sum', Math.abs(s.sum - M.tp.sum) < 1e-6, String(s.sum));
const drawn = await page.$$eval(SC + ' [data-out]', (g) => g.map((x) => x.dataset.out));
check("part 2's outputs are the model's", drawn.every((v, i) => v === M.tp.out[i].toFixed(4)), drawn.join(' '));
check('part 2 caption follows the part', (await caption()) === scripts[1]);
await part('3. Online');
await seek(1000);
s = await state();
check('part 3 ends with the online normaliser after one read', Math.abs(s.sum - M.on.sum) < 1e-6 && s.reads === 1, JSON.stringify(s));
await part('4. Write');
await seek(1000);
s = await state();
check('part 4 ends after two reads and one write', s.reads === 2 && s.writes === 1, JSON.stringify(s));
check('part 4 shows the ulps between the methods', (await page.$eval(SC + ' [data-part="compare"]', (e) => Number(e.dataset.ulps))) === M.ulps);
if (out) await page.screenshot({ path: out + '-scene-write.png' });
await part('5. Compared');
await seek(1000);
const rows = await page.$$eval(SC + ' [data-row]', (g) => Object.fromEntries(g.map((x) => [x.dataset.row, [x.dataset.a, x.dataset.b]])));
check('part 5: reads 3 against 2', JSON.stringify(rows['reads of the row']) === JSON.stringify([String(M.tp.reads), String(M.on.reads)]), JSON.stringify(rows));
check('part 5: exponentials 32 against 36', JSON.stringify(rows['exponentials']) === JSON.stringify([String(M.tp.exps), String(M.on.exps)]));
check('part 5: fast::exp on Metal, __expf on CUDA', JSON.stringify(rows['its exponential']) === JSON.stringify(['fast::exp', '__expf']));
// It plays by itself.
await part('1. Find the max');
await page.evaluate((SC) => [...document.querySelectorAll(SC + ' label.check')].find((l) => l.textContent.includes('sound')).querySelector('input').click(), SC);
await page.$eval(SC + ' button.primary', (b) => b.click());
await pause(1500);
const pos = await page.$eval(SC + ' input[type=range]', (e) => Number(e.value));
check('pressing Play moves the animation on', pos > 0, String(pos));
await page.$eval(SC + ' button.primary', (b) => b.click());

// ---------------------------------------------------------------- questions
const q = async (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const qb = await q('as a fraction of lab 10.3');
await (await qb.$('input[type=number]')).type('0.75');
await (await qb.$('button.primary')).click();
await pause(120);
check('q-bytes accepts 0.75', (await qb.evaluate((e) => e.textContent)).includes('Correct.'));
const qr = await q('sync is needed as we reuse buf_iw');
await (await qr.$$('input[type=radio]'))[0].click();
await (await qr.$('button.primary')).click();
await pause(120);
check('q-26385 accepts the shared-buffer answer', (await qr.evaluate((e) => e.textContent)).includes('Correct.'));
const qf = await q('576 floats');
await (await qf.$$('input[type=radio]'))[1].click();
await (await qf.$('button.primary')).click();
await pause(120);
check('q-float4-576 rejects the 1024 distractor with its reason', (await qf.evaluate((e) => e.textContent)).includes('Not quite.') && (await qf.evaluate((e) => e.textContent)).includes("block size"));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
