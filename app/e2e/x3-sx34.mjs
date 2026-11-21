// Browser check for section X3.4, "Matrix products: the shape picks the kernel": the quoted
// MLX GEMV beside lab 22.1's functions, the explorer "Which kernel runs" in every project and
// at the token counts where the chains change, the strip of code constants and the reader's
// own knees (from /api/reports), and the questions. Expected values come from
// src/sim/matmulroute.ts, run inside the page through the app's module server.
// Usage: APP_URL=http://localhost:5173 node e2e/x3-sx34.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';

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
const B = 'section.bench[aria-label="Which kernel runs"]';
const TOKENS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 32, 64, 128, 256, 512];
const select = async (label, value) => {
  await page.$eval(`${B} select[aria-label="${label}"]`, (el, v) => {
    el.value = String(v);
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await pause(80);
};
const tokens = async (M) => {
  await page.$eval(`${B} input[aria-label="Tokens"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, TOKENS.indexOf(M));
  await pause(80);
};
const box = async (needle, on) => {
  await page.evaluate((B, needle, on) => {
    const i = [...document.querySelectorAll(B + ' label.check')].find((l) => l.textContent.includes(needle)).querySelector('input');
    if (i.checked !== on) i.click();
  }, B, needle, on);
  await pause(80);
};
const result = () => page.$eval(`${B} [data-part="result"]`, (e) => ({ candidate: e.dataset.candidate, kind: e.dataset.kind, certain: e.dataset.certain === '1', text: e.textContent }));
const verdicts = () => page.$$eval(`${B} [data-part="trace"] li`, (l) => l.map((x) => [x.dataset.route, x.dataset.verdict]));
/** The model's route for a project and shape, computed in the page. */
const model = (project, over) =>
  page.evaluate(async (project, over) => {
    const m = await import('/src/sim/matmulroute.ts');
    const r = m.route(project, { ...m.DEFAULT_SHAPE, ...over });
    return { candidate: r.candidate, kind: r.kind, certain: r.certain, checks: r.checks.map((c) => [c.route, c.verdict]) };
  }, project, over);
const same = async (name, project, over) => {
  const [a, b, v] = [await result(), await model(project, over), await verdicts()];
  check(name, a.candidate === b.candidate && a.certain === b.certain && JSON.stringify(v) === JSON.stringify(b.checks), JSON.stringify({ page: a.candidate, model: b.candidate, v, m: b.checks }));
  return a;
};

await goto(page, 'x3/X3.4');
await page.waitForSelector(B);
await page.waitForFunction(() => document.body.textContent.includes('lab 22.1, warpRowDots'), { timeout: 15000 });

// ---------------------------------------------------------------- excerpts
const codes = await page.$$eval('.code', (cs) => cs.map((c) => c.textContent));
check('MLX gemv_impl is quoted', codes.some((c) => c.includes('sum = cg::reduce(warp, sum, cg::plus<Acc>{});')));
const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent));
check("lab 22.1's three functions are tabs", ['lab 22.1, warpRowDots', 'gemvGatherKernel', 'warpSum'].every((t) => tabs.includes(t)), tabs.join('|'));
check('lab 22.1 is loaded from the lab file', codes.some((c) => c.includes('__device__ void warpRowDots') && c.includes('for (int k = k0 + lane; k < k1; k += warpSize)')));

// ---------------------------------------------------------------- explorer: llama.cpp CUDA
await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));
let r = await same('llama.cpp CUDA, one token, F16: matches the model', 'llama-cuda', { M: 1 });
check('one token: mmvf is the first candidate, not certain', r.candidate === 'mmvf' && !r.certain && r.text.includes('First candidate'), r.text);
check('for a float weight, mmf and FP8 stay undecided and mmq is ruled out by its name', JSON.stringify((await verdicts()).filter(([k]) => ['mmf', 'mmq', 'fp8'].includes(k))) === JSON.stringify([['mmf', 'not-modelled'], ['mmq', 'false'], ['fp8', 'not-modelled']]));
if (out) await page.screenshot({ path: out + '-explorer.png' });
await tokens(8);
r = await same('8 tokens: matches the model', 'llama-cuda', { M: 8 });
check('8 tokens keep mmvf a candidate', r.candidate === 'mmvf');
await tokens(9);
r = await same('9 tokens: matches the model', 'llama-cuda', { M: 9 });
check('9 tokens rule out mmvf and mmvq', (await verdicts()).filter(([k]) => k === 'mmvf' || k === 'mmvq').every(([, v]) => v === 'false') && r.candidate === 'mmf');
await box('activations (src1) are F32', false);
r = await same('a non-F32 input: matches the model', 'llama-cuda', { M: 9, inputF32: false });
check('a non-F32 input goes straight to cuBLAS, certainly', r.candidate === 'cublas' && r.certain);
await box('activations (src1) are F32', true);
await tokens(4);
await select('Weight type', 'quant');
r = await same('a quantised weight at 4 tokens: matches the model', 'llama-cuda', { M: 4, weight: 'quant' });

// ---------------------------------------------------------------- one token everywhere
await select('Weight type', 'f16');
await tokens(1);
const all = await page.$$eval(`${B} [data-part="all"] tbody tr`, (rs) => Object.fromEntries(rs.map((x) => [x.dataset.project, x.dataset.kind])));
const read = await page.evaluate(async () => (await import('/src/sim/matmulroute.ts')).SHAPE_RULE_READ);
check('one token: a GEMV in every project whose rule was read', read.every((p) => all[p] === 'gemv'), JSON.stringify(all));
check('one token: PyTorch CUDA hands it to a library', all['torch-cuda'] === 'library');

// ---------------------------------------------------------------- MLX and llama.cpp Metal
await select('Project and backend', 'mlx-cuda');
r = await same('MLX CUDA, x @ W.T, one row: matches the model', 'mlx-cuda', { M: 1 });
check('MLX CUDA one row with b transposed: GEMV, certainly', r.candidate === 'mlx-gemv' && r.certain);
await box('b transposed', false);
r = await same('MLX CUDA without the transposed view: matches the model', 'mlx-cuda', { M: 1, bTransposed: false });
check('without it: cuBLASLt', r.candidate === 'cublaslt');
await box('b transposed', true);
await box('fused bias', true);
r = await result();
check('a fused bias sends one row to cuBLASLt', r.candidate === 'cublaslt');
await box('fused bias', false);
await select('Project and backend', 'mlx-metal');
await tokens(4);
r = await same('MLX Metal, 4 vectors: matches the model', 'mlx-metal', { M: 4 });
check('MLX Metal, 4 vectors: the wide GEMV', r.candidate === 'wide');
await tokens(6);
r = await result();
check('MLX Metal, 6 vectors: steel', r.candidate === 'steel');
await select('Project and backend', 'llama-metal');
await select('Weight type', 'kquant');
await tokens(3);
r = await same('llama.cpp Metal, k-quant, 3 tokens: matches the model', 'llama-metal', { M: 3, weight: 'kquant' });
check('k-quants need 4 tokens for mul_mv_ext', r.candidate === 'mul_mm');
await tokens(1);
r = await result();
check('llama.cpp Metal, one token: mul_mv, certainly', r.candidate === 'mul_mv' && r.certain);
await select('Project and backend', 'torch-mps');
r = await same('PyTorch MPS, one token: matches the model', 'torch-mps', { M: 1, weight: 'kquant' });
check('PyTorch MPS rank-1: the GEMV kernels, with the Linear.mm note', r.candidate === 'mps-gemv' && r.text.includes('MPSNDArrayMatrixMultiplication'));

// ---------------------------------------------------------------- the strip and the knees
const marks = await page.$$eval(`${B} [data-part="strip"] [data-mark]`, (g) => g.map((x) => [x.dataset.mark, Number(x.dataset.tokens)]));
check('the strip shows the code constants 8 and 5', marks.filter(([k]) => k === 'code').map(([, t]) => t).join() === '8,5', JSON.stringify(marks));
const knees = await page.evaluate(async () => {
  const m = await import('/src/sim/matmulroute.ts');
  const reports = await (await fetch('/api/reports')).json();
  return m.readerKnees(reports).map((k) => k.knee);
});
const shown = marks.filter(([k]) => k === 'knee').map(([, t]) => t);
check("the strip shows the reader's knees from the reports", JSON.stringify(shown) === JSON.stringify(knees.filter((k) => k != null)), JSON.stringify({ shown, knees }));
const listed = await page.$$eval(`${B} [data-part="knees"] li`, (l) => l.map((x) => x.dataset.knee));
check('every measured series is listed, with or without a knee', listed.length === knees.length && (knees.length > 0 || (await page.$eval(`${B} [data-part="knees"]`, (e) => e.textContent)).includes('No lab 22.2 measurement')), JSON.stringify(listed));
check('the token line follows the slider', (await page.$eval(`${B} [data-part="tokens"]`, (e) => e.dataset.m)) === '1');
if (out) await page.screenshot({ path: out + '-strip.png' });

// ---------------------------------------------------------------- questions
const q = async (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const answer = async (needle, i) => {
  const h = await q(needle);
  await (await h.$$('input[type=radio]'))[i].click();
  await (await h.$('button.primary')).click();
  await pause(120);
  return h.evaluate((e) => e.textContent);
};
check('q-mlx-transposed accepts GEMV', (await answer('GEMV or cuBLASLt?', 0)).includes('Correct.'));
check('q-nine accepts mmvf and mmvq ruled out', (await answer('a batch of 9 tokens', 0)).includes('Correct.'));
const t = await answer('Does this matrix product reach cuBLAS?', 1);
check('q-one-token rejects "the weights are not F32" with its reason', t.includes('Not quite.') && t.includes('activations'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
