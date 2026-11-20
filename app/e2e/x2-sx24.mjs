// Browser check for section X2.4: the stepper "One warp across a quantised row" (layouts,
// lanes per block, load width, load order, warp width, row length, segment model, stepping),
// the Predict on lab X2.1's mapping and its "Show it in the stepper", the explorer "Lanes per
// block against row length", lab X2.2's reference solution with its stepper preset, and
// three questions. Expected values come from src/sim/qlayout.ts, imported inside the page.
// Usage: APP_URL=http://localhost:5173 node e2e/x2-sx24.mjs [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto } from './browser.mjs';

const out = process.argv[2];
const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const hasLab = fs.existsSync(path.join(REPO, 'labs/x2/lab2_q4gemv/solution/kernel.cu'));
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const S = 'section.bench[aria-label="One warp across a quantised row"]';
const T = 'section.bench[aria-label="Lanes per block against row length"]';
const seg = async (scope, control, label) => {
  const ok = await page.evaluate((scope, control, label) => {
    const b = [...document.querySelectorAll(`${scope} [data-control="${control}"] button`)].find((x) => x.textContent.trim() === String(label));
    if (!b) return false;
    b.click();
    return true;
  }, scope, control, label);
  await pause(120);
  return ok;
};
const setSelect = async (scope, label, value) => {
  await page.evaluate((scope, label, value) => {
    const s = document.querySelector(`${scope} select[aria-label="${label}"]`);
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(s, String(value));
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, scope, label, value);
  await pause(120);
};
const button = async (scope, text) => {
  const ok = await page.evaluate((scope, text) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === text);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, text);
  await pause(120);
  return ok;
};
const num = (sel, a) => page.$eval(sel, (e, a) => Number(e.getAttribute(a)), a);
const txt = (sel) => page.$eval(sel, (e) => Number(e.textContent.trim()));
const segs = () => txt(S + ' [data-code-segments]');
const narrate = () => page.$eval(S + ' [data-narrate]', (e) => e.textContent.replace(/\s+/g, ' '));
const qHandle = (text) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(t)), text);
async function answerChoice(text, index) {
  const q = await qHandle(text);
  if (!(await q.evaluate((e) => !!e))) return 'missing';
  await q.evaluate((e, i) => e.querySelectorAll('input[type=radio]')[i].click(), index);
  await pause(60);
  await q.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Check').click());
  await pause(120);
  return q.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
}
async function answerNumber(text, value) {
  const q = await qHandle(text);
  if (!(await q.evaluate((e) => !!e))) return 'missing';
  const input = await q.$('input[type=number]');
  await input.click({ clickCount: 3 });
  await input.type(String(value));
  await q.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Check').click());
  await pause(120);
  return q.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
}

await goto(page, 'x2/X2.4');
await page.waitForSelector(S);
const M = await page.evaluate(async () => {
  const m = await import('/src/sim/qlayout.ts');
  const q4 = (o) => ({ layout: 'q4_0-blocks', lanesPerBlock: 1, loadBytes: 2, warpWidth: 32, inW: 4096, ...o });
  const st = (mm, it = 0, k = 0, sb = 32) => m.loadStep(mm, it, k, sb);
  return {
    lpb1: st(q4()).code.transactions,
    lpb4: st(q4({ lanesPerBlock: 4 })).code.transactions,
    mis4: st(q4({ loadBytes: 4 })).misaligned,
    mlx4: st({ layout: 'mlx4-split', lanesPerBlock: 1, loadBytes: 4, warpWidth: 32, inW: 4096 }).misaligned,
    piece: st(q4({ lanesPerBlock: 4, order: 'piece' })).code.transactions,
    seg64: st(q4(), 0, 0, 64).code.transactions,
    x21: st({ layout: 'q8_0-split', lanesPerBlock: 8, loadBytes: 4, warpWidth: 32, inW: 4096 }),
    tail32: m.iterationPlan(1056, 'q4_0-blocks', 1, 32),
    f1024: [32, 64].map((w) => m.iterationPlan(1024, 'q4_0-blocks', 1, w).idleFraction.toFixed(4)),
    f1024l2: m.iterationPlan(1024, 'q4_0-blocks', 2, 64).idleFraction.toFixed(4),
    p1056: [32, 64].map((w) => m.iterationPlan(1056, 'q4_0-blocks', 1, w).idleLast),
  };
});

// ---------------------------------------------------------------- the stepper
check('default: one lane per Q4_0 block, 2-byte loads, 32-byte segments: 18 segments', (await segs()) === M.lpb1 && M.lpb1 === 18, String(await segs()));
check('64 useful bytes in that load', (await txt(S + ' [data-code-needed]')) === 64);
check('the code strip marks 18 touched segments', (await page.$$eval(S + ' [data-strip="code"] [data-touched="1"]', (e) => e.length)) === 18);
if (out) { await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' })); await page.screenshot({ path: out + '-stepper-lpb1.png' }); }
await seg(S, 'lanes', 4);
check('four lanes per block: 5 segments', (await segs()) === M.lpb4 && M.lpb4 === 5, String(await segs()));
await button(S, 'Step');
check('Step moves to the next load of the iteration', (await narrate()).includes('load 2 of 2'), await narrate());
await button(S, 'Next iteration');
check('Next iteration moves on', (await narrate()).startsWith('Iteration 2 of'), await narrate());
await seg(S, 'order', 'its own piece');
check("one piece per lane (lab X2.2's order): still 5 segments", (await segs()) === M.piece, String(await segs()));
await seg(S, 'order', 'interleaved with its neighbours');
await seg(S, 'lanes', 1);
await seg(S, 'width', 4);
check('4-byte loads on Q4_0 blocks: the even blocks are misaligned (16 lanes)', (await num(S + ' [data-misaligned]', 'data-misaligned')) === M.mis4 && M.mis4 === 16);
const red = await page.$$eval(S + ' [data-strip="code"] rect[data-lane]', (e) => e.filter((x) => x.getAttribute('fill') === 'var(--fail)').map((x) => Number(x.dataset.lane)));
check('the misaligned loads are drawn in the failure colour, lanes 0, 2, 4, ...', red.length === 16 && red.every((l) => l % 2 === 0), JSON.stringify(red));
if (out) { await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' })); await page.screenshot({ path: out + '-stepper-misaligned.png' }); }
await setSelect(S, 'Layout', 'mlx4-split');
check('separate arrays: the same 4-byte loads are all aligned', (await num(S + ' [data-misaligned]', 'data-misaligned')) === M.mlx4 && M.mlx4 === 0);
await setSelect(S, 'Layout', 'q4_0-blocks');
await seg(S, 'lanes', 8);
check('eight lanes with 4-byte loads cannot split a 16-byte Q4_0 block, and the stepper says so', (await page.$eval(S + ' [data-problem]', (e) => e.textContent)).includes('16 code bytes'));
await seg(S, 'lanes', 1);
await seg(S, 'width', 2);
await setSelect(S, 'Row length', 1056);
await button(S, 'Last iteration');
check('in = 1056: the last iteration has 31 idle lanes', (await num(S + ' [data-idle]', 'data-idle')) === M.tail32.idleLast && (await narrate()).includes('31 lanes fail the bounds test'), await narrate());
const lanes = await page.$$eval(S + ' [data-strip="lanes"] [data-lane]', (e) => e.map((x) => x.dataset.active));
check('the lane strip shows one active lane and 31 empty ones', lanes.filter((a) => a === '1').length === 1 && lanes.length === 32);
await seg(S, 'warp', 64);
check('a 64-lane wavefront draws 64 lanes', (await page.$$eval(S + ' [data-strip="lanes"] [data-lane]', (e) => e.length)) === 64);
await seg(S, 'warp', 32);
await setSelect(S, 'Row length', 4096);
await setSelect(S, 'Segments', 'metal');
check('Apple: the stepper asks for an assumed segment size', !!(await page.$(S + ' select[aria-label="Assumed segment"]')) && (await page.$eval(S, (e) => e.textContent)).includes('your assumption'));
await setSelect(S, 'Assumed segment', 64);
check('with 64-byte segments the lane-per-block load touches the model count', (await segs()) === M.seg64, String(await segs()));
await setSelect(S, 'Segments', 'nvidia');

// ---------------------------------------------------------------- lab X2.1's mapping
const pq = await qHandle('how many segments does one of those code loads touch');
if (await pq.evaluate((e) => !!e.querySelector('input[type=number]'))) {
  const input = await pq.$('input[type=number]');
  await input.type('4');
  await pq.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.includes('Commit')).click());
  await pause(150);
}
const rev = await page.$eval('[data-reveal="x21"]', (e) => e.textContent).catch(() => '');
check("the reveal gives the model's segments for lab X2.1's code and scale loads", rev.includes(`${M.x21.code.transactions} segments`) && rev.includes(`${M.x21.scale.bytesNeeded} bytes in`), rev.slice(0, 200));
await page.$eval('[data-action="load-x21"]', (b) => b.click());
await pause(150);
check('"Show it in the stepper" loads Q8_0 split arrays, 8 lanes per block, 4-byte loads',
  (await page.$eval(S + ' select[aria-label="Layout"]', (e) => e.value)) === 'q8_0-split' && (await segs()) === 4 && (await txt(S + ' [data-scale-segments]')) === 1);
if (out) { await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' })); await page.screenshot({ path: out + '-stepper-x21.png' }); }

// ---------------------------------------------------------------- the tail explorer
const setRange = async (value) => {
  await page.$eval(T + ' input[type=range]', (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(120);
};
const tailRows = () => page.$$eval(T + ' tr[data-width]', (e) => e.map((x) => [x.dataset.width, x.dataset.iterations, x.dataset.idleLast, x.dataset.idleFrac]));
await setRange(32);
let tr = await tailRows();
check('in = 1024, one lane per block: no idle lanes on 32, half idle on 64', tr[0][3] === M.f1024[0] && tr[1][3] === M.f1024[1] && M.f1024[1] === '0.5000', JSON.stringify(tr));
await seg(T, 'tail-lanes', 2);
tr = await tailRows();
check('two lanes per block fill the 64-lane wavefront again', tr[1][3] === M.f1024l2, JSON.stringify(tr));
await seg(T, 'tail-lanes', 1);
await setRange(33);
tr = await tailRows();
check('in = 1056: 31 idle lanes in the last iteration on both widths', tr[0][2] === String(M.p1056[0]) && tr[1][2] === String(M.p1056[1]), JSON.stringify(tr));
if (out) { await page.$eval(T, (e) => e.scrollIntoView({ block: 'start' })); await page.screenshot({ path: out + '-tail.png' }); }

// ---------------------------------------------------------------- the code
if (hasLab) {
  await page.waitForSelector('.code .tabs button', { timeout: 5000 }).catch(() => null);
  const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent));
  check("lab X2.2's solution is shown with CUDA and Metal tabs", tabs.includes('CUDA and HIP') && tabs.includes('Metal'), tabs.join('|'));
  check('the CUDA tab is the lab file', (await page.$eval('.code pre', (e) => e.textContent)).includes('LANES_PER_BLOCK'));
} else check('with no lab X2.2 files yet, the page says so', !!(await page.$('[data-code="missing"]')));
await page.$eval('[data-action="load-x22"]', (b) => b.click());
await pause(150);
check("\"Show lab X2.2's mapping\" sets four lanes, 2-byte loads, one piece per lane",
  (await page.$eval(S + ' [data-control="order"] button[aria-pressed="true"]', (e) => e.textContent)) === 'its own piece' &&
  (await page.$eval(S + ' [data-control="lanes"] button[aria-pressed="true"]', (e) => e.textContent)) === '4' && (await segs()) === M.piece);

// ---------------------------------------------------------------- questions
check('q-lane-per-block accepts 18', (await answerNumber('How many segments does that one load instruction touch', 18)).includes('Correct.'));
check('q-tail accepts 31', (await answerNumber('How many lanes are idle in the last iteration', 31)).includes('Correct.'));
check('q-wave64: 16 blocks, nothing to change', (await answerChoice('how many blocks does one iteration of a 64-lane wavefront cover', 0)).includes('Correct.'));
check('q-four-lanes: a wrong answer explains itself', (await answerChoice('The same row with four lanes per block', 1)).includes('Not quite.'));

const expected404 = (u) => !hasLab && /\/api\/lab-file\?path=x2%2Flab2_q4gemv/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
