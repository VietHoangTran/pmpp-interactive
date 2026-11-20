// Browser check for section 10.8: the listing, the cost explorer, the read-past-the-array
// explorer, the prediction and two questions. Every number is compared with the model the
// page runs on (src/sim/shuffle.ts) or, for the measurement, with /api/reports.
// Usage: node e2e/ch10-s108.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { KERNELS, blockCosts, ones, twoStage } from '../src/sim/shuffle.ts';

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
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const bench = (t) => `section.bench[aria-label="${t}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const clickText = async (scope, label) => {
  const done = await page.evaluate((sel, label) => {
    const btn = [...document.querySelector(sel).querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, bench(scope), label);
  await pause(80);
  return done;
};
const setSelect = async (scope, aria, value) => {
  await page.$eval(`${bench(scope)} select[aria-label="${aria}"]`, (s, v) => {
    s.value = String(v);
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await pause(100);
};

await goto(page, 'ch10/10.8');
await page.waitForSelector(bench('Hybrid or two-stage: count the costs'));

// ---- the listing
await page.waitForFunction(() => [...document.querySelectorAll('.code .tabs button')].some((b) => b.textContent === 'CUDA'), { timeout: 10000 }).catch(() => {});
const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent));
check('the two-stage kernel is shown for CUDA, and for Metal', tabs.includes('CUDA') && tabs.includes('Metal'), tabs.join('|'));
check('the CUDA tab\'s status is the chapter\'s fixed wording, with HIP unverified', (await page.$eval('.code .tabs .status', (e) => e.textContent)) === 'front-end-checked; emulator-checked; unverified on CUDA and HIP hardware');
const code = await text('.code pre');
check('the CUDA tab is the lab kernel, with the guarded stage-2 read', code.includes('TwoStageWarpLevelSumReductionKernel') && code.includes('laneIdx() < warps ? partialSums_s[laneIdx()] : 0.0f') && code.includes('MAX_BLOCK_DIM / MIN_WARP_SIZE'));
await page.evaluate(() => [...document.querySelectorAll('.code .tabs button')].find((b) => b.textContent === 'Metal').click());
await pause(100);
const mcode = await text('.code pre');
check('the Metal tab is two_stage_warp_sum, its lanes from attributes', mcode.includes('two_stage_warp_sum') && mcode.includes('[[thread_index_in_simdgroup]]') && mcode.includes('lane < groups'));

// ---- cost explorer
const C = 'Hybrid or two-stage: count the costs';
const table = () => page.$$eval(`${bench(C)} table[data-costs] tbody tr`, (rs) => Object.fromEntries(rs.map((r) => [r.dataset.row, [...r.querySelectorAll('td[data-kernel]')].map((d) => Number(d.textContent))])));
const fromModel = (B, W) => {
  const cs = KERNELS.map((k) => blockCosts(k, B, W));
  return {
    barriers: cs.map((c) => c.barriers), writes: cs.map((c) => c.sharedWrites), reads: cs.map((c) => c.sharedReads),
    rounds: cs.map((c) => c.shuffleRounds), shuffles: cs.map((c) => c.warpShuffles), steps: cs.map((c) => c.steps.length),
    slots: cs.map((c) => c.slots), useful: cs.map((c) => c.useful), wasted: cs.map((c) => c.wasted),
  };
};
const timeline = () => page.$$eval(`${bench(C)} g[data-kernel]`, (gs) => gs.map((g) => ({
  kernel: g.dataset.kernel, barriers: Number(g.dataset.barriers),
  steps: [...g.querySelectorAll('g[data-step]')].map((s) => [Number(s.dataset.warps), Number(s.dataset.useful), Number(s.dataset.barrier)]),
})));
let t = await table();
check('1024 threads, width 32: barriers 10, 6 and 1', same(t.barriers, [10, 6, 1]), JSON.stringify(t.barriers));
check('every row of the table is the model\'s', same(t, fromModel(1024, 32)), JSON.stringify(t));
check('all three do 2047 useful additions; the two-stage kernel wastes 4257 lane slots against 129', same(t.useful, [2047, 2047, 2047]) && same(t.wasted, [129, 129, 4257]));
let tl = await timeline();
const modelTl = (B, W) => KERNELS.map((k) => blockCosts(k, B, W)).map((c) => ({ kernel: c.kernel, barriers: c.barriers, steps: c.steps.map((s) => [s.warps, s.useful, s.barrierBefore ? 1 : 0]) }));
check('the timeline draws every step with the model\'s warps, useful additions and barriers', same(tl, modelTl(1024, 32)));
const says = await text(`${bench(C)} [data-costs-says]`);
check('the sentence uses the numbers: 5 of 6 barriers, 3968 accesses, 4128 more lane slots', says.includes('saves 5 of 6 barriers and 3968 shared-memory accesses, and spends 4128 more lane slots'), says);
if (out) await page.$eval(bench(C), (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-costs-1024.png' });
await setSelect(C, 'block size', 256);
t = await table();
check('256 threads: the table follows the model (barriers 8, 4, 1)', same(t, fromModel(256, 32)) && same(t.barriers, [8, 4, 1]), JSON.stringify(t.barriers));
await setSelect(C, 'warp width', 'w64');
await setSelect(C, 'block size', 1024);
t = await table();
tl = await timeline();
check('width 64 at 1024 threads: table and timeline follow the model (barriers 10, 5, 1)', same(t, fromModel(1024, 64)) && same(t.barriers, [10, 5, 1]) && same(tl, modelTl(1024, 64)), JSON.stringify(t.barriers));
await setSelect(C, 'warp width', 'book');

// ---- read past the array
const P = 'Read past the array';
const slots = () => page.$$eval(`${bench(P)} g[data-slot]`, (gs) => gs.map((g) => g.dataset.state));
const laneVals = () => page.$$eval(`${bench(P)} g[data-lane]`, (gs) => gs.map((g) => g.dataset.value));
const result = () => page.$eval(`${bench(P)} [data-result]`, (e) => [e.dataset.result, e.textContent.trim()]);
const fromTwo = (r) => ({ slots: r.slots.map((s) => s.state), lanes: r.lanes.map((l) => (l.value == null ? '?' : String(l.value))) });
let r = twoStage(ones(512), 256, 32, false);
check('as printed at 256 threads: 8 slots written, 24 past the end, every lane as the model says', same({ slots: await slots(), lanes: await laneVals() }, fromTwo(r)) && (await slots()).filter((s) => s === 'past').length === 24);
check('and the sum is undefined', same(await result(), ['fail', 'undefined']));
check('the code line shows 256/32 = 8 entries', (await text(`${bench(P)} [data-read-code]`)).includes('256/32 = 8 entries'));
if (out) await page.$eval(bench(P), (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-past-256.png' });
await setSelect(P, 'block size', 1024);
check('as printed at 1024 threads of width 32: the one shape it fits, 2048', same(await result(), ['ok', '2048']) && (await slots()).every((s) => s === 'written'));
await setSelect(P, 'warp width', 'w64');
r = twoStage(ones(2048), 1024, 64, false);
check('as printed at 1024 threads of width 64: 16 written, 48 past, undefined', same({ slots: await slots(), lanes: await laneVals() }, fromTwo(r)) && (await slots()).filter((s) => s === 'past').length === 48 && (await result())[0] === 'fail');
if (out) await page.screenshot({ path: out + '-past-w64.png' });
await clickText(P, 'guarded, padded with 0');
r = twoStage(ones(2048), 1024, 64, true);
check('guarded: lanes 16 to 63 start with 0 and the sum is right', same({ slots: await slots(), lanes: await laneVals() }, fromTwo(r)) && (await laneVals()).slice(16).every((v) => v === '0') && same(await result(), ['ok', '2048']));
await setSelect(P, 'block size', 64);
check('guarded at 64 threads of width 64: one warp, right', (await result())[0] === 'ok');
if (out) await page.screenshot({ path: out + '-past-guarded.png' });

// ---- prediction and the measured comparison
const reports = await (await fetch(BASE + '/api/reports')).json();
const found = Object.keys(reports).filter((b) => b !== 'emu').flatMap((b) => {
  const rows = (reports[b].results ?? []).filter((x) => x.lab === 'ch10/lab1_sum' && x.variant === 'solution').flatMap((x) => x.bench).filter((x) => x.name === 'rows');
  const blocks = [...new Set(rows.map((x) => x.block))];
  const block = blocks.includes(256) ? 256 : blocks[0];
  const s = rows.find((x) => x.kernel === 'shared' && x.block === block)?.seconds;
  const w = rows.find((x) => x.kernel === 'warp' && x.block === block)?.seconds;
  return s && w ? [{ b, speedup: s / w }] : [];
});
const pq = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Predict first') && e.textContent.includes('rows of 512 values')));
await (await pq.$('input[type=number]')).type('1.5');
await (await pq.$('button.primary')).click();
await pause(400);
const reveal = await pq.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
if (found.length) {
  const fmt = (v) => String(Number(v.toPrecision(3)));
  check('the reveal shows this machine\'s measured ratio', reveal.includes('You predicted 1.5') && found.every((f) => reveal.includes(`${fmt(f.speedup)}×`)), reveal.slice(0, 300));
} else {
  check('with no measurement the reveal says what to run', reveal.includes('You predicted 1.5') && reveal.includes('pmpp.py run 10.1 --solution --bench'), reveal.slice(0, 300));
}

// ---- questions
const answer = async (needle, value) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  if (typeof value === 'number') await (await q.$('input[type=number]')).type(String(value));
  else await q.evaluate((e, v) => [...e.querySelectorAll('.opts label')].find((l) => l.textContent.includes(v)).querySelector('input').click(), value);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
check('q-barriers accepts 6', (await answer('How many times does the hybrid kernel', 6)).includes('Correct.'));
check('q-guard-256 accepts "8 sums and 24 reads past the end"', (await answer('launched with 256 threads per block', '24 reads past the end')).includes('Correct.'));
check('q-w64-warps rejects 32 and gives 16', (await answer('How many per-warp partial sums', 32)).includes('The answer is 16'));
check('q-metal-lanes: "simd_sum hands the sum to every thread" is wrong, for the SIMD-group reason', (await answer('before thread 100', 'Nothing: simd_sum hands the sum to every thread')).includes('its own SIMD-group'));

// ---- a narrow screen
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
check('at a narrow width nothing makes the page scroll sideways', overflow <= 1, String(overflow));
if (out) {
  await page.$eval(bench(C), (e) => e.scrollIntoView());
  await page.screenshot({ path: out + '-narrow.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
