// Browser check for section X3.5, "Attention: own kernels, libraries and submodules": the
// quoted MLX decode-kernel lines beside lab 20.4's kernel, the explorer "Which attention
// kernel" in every project at the shapes where the rules change, the reveal of the start
// page's prediction (with and without a committed value), and the questions. Expected values
// come from src/sim/attnroute.ts, run inside the page through the app's module server.
// Usage: APP_URL=http://localhost:5173 node e2e/x3-sx35.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
await page.setRequestInterception(true);
// Leave no trace in the real progress file; for the second visit, serve a committed start
// page prediction.
let served = null;
page.on('request', (r) => {
  if (r.method() === 'PUT' && r.url().includes('/api/progress'))
    return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  if (served && r.method() === 'GET' && r.url().endsWith('/api/progress'))
    return r.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(served) });
  return r.continue();
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const B = 'section.bench[aria-label="Which attention kernel"]';
const KEYS = [1, 16, 64, 100, 128, 255, 256, 300, 512, 1000, 1023, 1024, 2048, 4095, 4096, 5000, 8192];
const project = async (v) => {
  await page.$eval(`${B} select[aria-label="Project and backend"]`, (el, v) => {
    el.value = v;
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(80);
};
const keys = async (k) => {
  await page.$eval(`${B} input[aria-label="Keys"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, KEYS.indexOf(k));
  await pause(80);
};
const seg = async (group, label) => {
  await page.evaluate((B, group, label) => [...document.querySelectorAll(`${B} [aria-label="${group}"] button`)].find((b) => b.textContent.trim() === label).click(), B, group, label);
  await pause(80);
};
const box = async (needle, on) => {
  await page.evaluate((B, needle, on) => {
    const i = [...document.querySelectorAll(B + ' label.check')].find((l) => l.textContent.includes(needle)).querySelector('input');
    if (i.checked !== on) i.click();
  }, B, needle, on);
  await pause(80);
};
const result = () => page.$eval(`${B} [data-part="result"]`, (e) => ({ home: e.dataset.home, text: e.textContent }));
const model = (p, over) =>
  page.evaluate(async (p, over) => {
    const m = await import('/src/sim/attnroute.ts');
    const r = m.attnRoute(p, { ...m.DEFAULT_INPUT, ...over, use: { ...m.DEFAULT_INPUT.use, ...(over.use ?? {}) } });
    return { name: r.kernel.name, home: r.kernel.home, checks: r.checks.map((c) => c.verdict), order: r.order ?? null };
  }, p, over);
const same = async (name, p, over) => {
  const a = await result();
  const b = await model(p, over);
  const v = await page.$$eval(`${B} [data-part="trace"] li`, (l) => l.map((x) => x.dataset.verdict));
  check(name, a.home === b.home && a.text.includes(b.name) && JSON.stringify(v) === JSON.stringify(b.checks), JSON.stringify({ a, b, v }));
  return a;
};

await goto(page, 'x3/X3.5');
await page.waitForSelector(B);
await page.waitForFunction(() => document.body.textContent.includes('lab 20.4, decodeAttentionKernel'), { timeout: 15000 });

// ---------------------------------------------------------------- excerpts
const codes = await page.$$eval('.code', (cs) => cs.map((c) => c.textContent));
check('MLX kernel_sdpav lines are quoted', codes.some((c) => c.includes('const U scale_log2 = params.scale * M_LOG2E;') && c.includes('exp2f(max_score - new_max)')));
check('lab 20.4 is loaded from the lab file', codes.some((c) => c.includes('__global__ void decodeAttentionKernel') && c.includes('const float alpha = expf(m - mNew);')));
const hl = await page.$$eval('.code .ln.hl', (l) => l.map((x) => x.textContent));
check("lab 20.4's online update lines are highlighted", hl.some((t) => t.includes('const float alpha = expf(m - mNew);')) && hl.some((t) => t.includes('expf(s - mNew)')), hl.join(' | '));

// ---------------------------------------------------------------- explorer: MLX CUDA
await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));
let r = await same('MLX CUDA, one query, 100 keys: matches the model', 'mlx-cuda', { queries: 1, keys: 100 });
check('100 keys: its own sdpa_vector kernel', r.home === 'own' && r.text.includes('sdpa_vector'));
await keys(255);
r = await same('255 keys: matches the model', 'mlx-cuda', { keys: 255 });
check('255 keys: still its own kernel', r.home === 'own');
await keys(256);
r = await same('256 keys: matches the model', 'mlx-cuda', { keys: 256 });
check('256 keys: cuDNN', r.home === 'library' && r.text.includes('cuDNN'));
await seg('Mask', 'array');
r = await same('an array mask: matches the model', 'mlx-cuda', { keys: 256, mask: 'array' });
check('an array mask keeps the decode on its own kernel', r.home === 'own');
await seg('Mask', 'none');
await box('supports_sdpa_cudnn', false);
await keys(300);
r = await result();
check('cuDNN declined at 300 keys: its own kernel', r.home === 'own');
await box('supports_sdpa_cudnn', true);
await seg('Queries', 'more (prefill)');
r = await same('prefill: matches the model', 'mlx-cuda', { queries: 128, keys: 300 });
check('prefill on MLX CUDA: cuDNN', r.home === 'library');
if (out) await page.screenshot({ path: out + '-explorer.png' });
const homes = await page.$$eval(`${B} [data-part="all"] tbody tr`, (rs) => Object.fromEntries(rs.map((x) => [x.dataset.project, x.dataset.home])));
check('prefill everywhere: llama.cpp and MLX Metal own, MLX CUDA a library, PyTorch CUDA a submodule',
  homes.llama === 'own' && homes['mlx-metal'] === 'own' && homes['mlx-cuda'] === 'library' && homes['torch-cuda'] === 'submodule', JSON.stringify(homes));

// ---------------------------------------------------------------- MLX Metal
await project('mlx-metal');
await seg('Queries', '1');
await box('GQA', true);
await keys(5000);
r = await same('MLX Metal, GQA, one query, 5000 keys: matches the model', 'mlx-metal', { queries: 1, keys: 5000, gqa: true });
check('GQA with 5000 keys: the 2-pass kernel', r.text.includes('2pass'));
await keys(4095);
r = await result();
check('4095 keys: one pass', !r.text.includes('2pass'));
await box('GQA', false);
await box("devc == 'd'", true);
await keys(1024);
r = await same('a d or s device at 1024 keys: matches the model', 'mlx-metal', { keys: 1024, deviceDS: true });
check('a d or s device at 1024 keys: 2-pass', r.text.includes('2pass'));
await seg('Queries', 'more (prefill)');
r = await result();
check('prefill on MLX Metal: steel attention', r.text.includes('steel'));

// ---------------------------------------------------------------- PyTorch CUDA
await project('torch-cuda');
await seg('Queries', '1');
r = await same('PyTorch CUDA defaults: matches the model', 'torch-cuda', { queries: 1, keys: 1024 });
check('the order walked is flash, efficient, math, cuDNN', (await page.$eval(`${B} [data-part="order"]`, (e) => e.textContent)).includes('flash_attention, efficient_attention, math, cudnn_attention'));
check('flash runs from the submodule', r.home === 'submodule' && r.text.includes('third_party/flash-attention'));
await box('use_flash_attention', false);
await box('use_efficient_attention', false);
r = await same('flash and efficient declined: matches the model', 'torch-cuda', { queries: 1, keys: 1024, use: { flash: false, efficient: false } });
check('flash and efficient declined: math before cuDNN', r.text.includes('math'));
await box('check_prefer_cudnn_attention', true);
r = await result();
check('cuDNN preferred: cuDNN first', r.home === 'library' && (await page.$eval(`${B} [data-part="order"]`, (e) => e.textContent)).includes('Order walked: cudnn_attention'));
await project('llama');
r = await result();
check('llama.cpp: its own kernels', r.home === 'own' && r.text.includes('fattn'));

// ---------------------------------------------------------------- the start page reveal
const M = await page.evaluate(async () => {
  const m = await import('/src/sim/attnroute.ts');
  return { own: m.OWN_EVERYWHERE, homes: { 'llama.cpp': m.cudaHome('llama'), mlx: m.cudaHome('mlx-cuda'), pytorch: m.cudaHome('torch-cuda') } };
});
const rows = () => page.$$eval('[data-part="cuda-homes"] tbody tr', (r) => r.map((x) => ({ p: x.dataset.project, home: x.dataset.home, right: x.dataset.right })));
let rv = await page.$eval('[data-part="reveal"]', (e) => ({ text: e.textContent, predicted: e.querySelector('[data-predicted]').dataset.predicted }));
check('with no prediction, the reveal says so and links to the start page', rv.predicted === 'none' && rv.text.includes('start page'), rv.text);
let rr = await rows();
check("the reveal gives each project's CUDA home from the model", rr.length === 3 && rr.every((r) => r.home === M.homes[r.p]), JSON.stringify(rr));
check('the reveal gives the all-backends count from the model', rv.text.includes(`${M.own.length} of the three projects, ${M.own.join(', ')}`));
check('the all-backends table lists each backend', (await page.$$('[data-part="homes"] tbody tr')).length === 7);
served = { version: 1, sections: {}, answers: { 'x3/start/p-attention-home': { value: { 'llama.cpp': 'own', mlx: 'own', pytorch: 'submodule' }, attempts: 1, at: '2026-11-20T00:00:00Z' } }, review: {} };
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector('.page[data-ready] [data-part="reveal"]', { timeout: 30000 });
await pause(400);
rv = await page.$eval('[data-part="reveal"]', (e) => ({ text: e.textContent, predicted: e.querySelector('[data-predicted]').dataset.predicted }));
rr = await rows();
check('a committed prediction is shown back, marked per project', rv.predicted === 'yes' && JSON.stringify(rr.map((r) => r.right)) === JSON.stringify(['1', '0', '1']), JSON.stringify(rr));
served = null;
if (out) {
  await page.$eval('[data-part="reveal"]', (e) => e.scrollIntoView({ block: 'center' }));
  await page.screenshot({ path: out + '-reveal.png' });
}

// ---------------------------------------------------------------- questions
const answer = async (needle, i) => {
  const h = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await (await h.$$('input[type=radio]'))[i].click();
  await (await h.$('button.primary')).click();
  await pause(120);
  return h.evaluate((e) => e.textContent);
};
check('q-decode-100 accepts its own sdpa_vector', (await answer('100 cached keys', 0)).includes('Correct.'));
check('q-order accepts math first', (await answer('flash and memory-efficient attention both reject', 0)).includes('Correct.'));
const t = await answer('what changes in the result', 1);
check('q-exp2 rejects "divide by log2 e" with its reason', t.includes('Not quite.') && t.includes('Nothing needs undoing'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
