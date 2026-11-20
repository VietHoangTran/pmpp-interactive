// Browser check for section 11.4: the scan-scan-add explorer, the shuffle-up stepper, the
// "Port it" hunt with its 64-lane sub-view, the listings, and two questions. Expected values
// come from the model (src/sim/warpscan.ts, imported directly).
// Usage: APP_URL=http://localhost:5173 node e2e/ch11-s114.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { FIG_11_6, HALF_CASES, halfCase, laneValues, portIssues, scanScanAdd, warpScanTrace } from '../src/sim/warpscan.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);
// A lab file another builder has not written yet answers 404, which Chrome logs as a console
// error; the page is designed to show nothing for it. Count those and forgive exactly as many.
let labMisses = 0;
page.on('response', (r) => { if (r.status() === 404 && r.url().includes('/api/lab-file')) labMisses++; });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (t) => `section.bench[aria-label="${t}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const clickText = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const root = document.querySelector(scope);
    const btn = [...root.querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(80);
  return done;
};
const select = async (scope, aria, value) => {
  await page.evaluate((scope, aria, value) => {
    const s = document.querySelector(`${scope} select[aria-label="${aria}"]`);
    s.value = value;
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, scope, aria, value);
  await pause(120);
};

await goto(page, 'ch11/11.4');
await page.waitForSelector(bench('Scan, scan, add'));

// ---------------------------------------------------------------- Scan, scan, add
const A = bench('Scan, scan, add');
const row = (k) => text(`${A} [data-row="${k}"]`);
const r0 = scanScanAdd(FIG_11_6, 4);
check('starts on Fig. 11.6: stage 1 as printed', (await row('stage1')) === r0.stage1.join(' '), await row('stage1'));
check('segment totals 7 7 6 11', (await row('totals')) === '7 7 6 11', await row('totals'));
check('scanned totals 7 14 20 31', (await row('stage2')) === '7 14 20 31', await row('stage2'));
check('final row is the inclusive scan', (await row('final')) === r0.final.join(' '), await row('final'));
check('the explanation names the segment ends', (await text(`${A} [data-ssa-says]`)).includes('positions 3, 7, 11, 15'));
// edit input 4 (0 -> 5)
const in4 = await page.$(`${A} input[data-input="4"]`);
await in4.click({ clickCount: 3 });
await in4.type('5');
await pause(150);
const edited = FIG_11_6.map((v, i) => (i === 4 ? 5 : v));
const r1 = scanScanAdd(edited, 4);
check('editing an input recomputes all three stages', (await row('final')) === r1.final.join(' ') && (await row('stage2')) === r1.stage2.join(' '), await row('final'));
await clickText(A, 'segments of 8');
const r2 = scanScanAdd(edited, 8);
check('segments of 8: two totals, same final row', (await row('totals')) === r2.totals.join(' ') && (await row('final')) === r1.final.join(' '), await row('totals'));
await select(A, 'what a segment is', 'block');
check('a segment can be a block (Fig. 11.14)', (await text(`${A} [data-ssa-says]`)).includes('Fig. 11.14'));
await clickText(A, "Fig. 11.6's numbers");
await clickText(A, 'segments of 4');
check('preset restores Fig. 11.6', (await row('final')) === r0.final.join(' '));
if (out) await page.screenshot({ path: out + '-ssa.png' });

// ---------------------------------------------------------------- Shuffle up, lane by lane
const S = bench('Shuffle up, lane by lane');
const cells = (r) => page.$$eval(`${S} [data-cell^="${r}-"]`, (els) => els.map((e) => [Number(e.dataset.value), e.dataset.kind]));
const t8 = warpScanTrace(laneValues(8));
await clickText(S, 'Step');
let n = await text(`${S} [data-narrate]`);
check('step 1 narrates stride 1 and lane 0 keeping its own value', n.startsWith('Stride 1') && n.includes('Lane 0 has no lane 1 to the left'), n);
const row1 = await cells(1);
check('row after stride 1 is the model\'s', same(row1.map((c) => c[0]), t8.steps[0].lanes.map((l) => l.after)), JSON.stringify(row1));
check('lane 0 is kept, the others added', row1[0][1] === 'kept' && row1.slice(1).every((c) => c[1] === 'added'));
await clickText(S, 'Run to the end');
check('the book\'s test: correct', (await page.$eval(`${S} [data-result]`, (e) => e.dataset.result)) === 'ok');
check('three strides on 8 lanes', (await page.$$(`${S} [data-cell^="3-"]`)).length === 8 && (await page.$$(`${S} [data-cell^="4-"]`)).length === 0);
await select(S, 'participation test', 'none');
await clickText(S, 'Run to the end');
const tn = warpScanTrace(laneValues(8), { test: 'none' });
check('no test: wrong, lane 0 at 8 times its input', (await page.$eval(`${S} [data-result]`, (e) => e.dataset.result)) === 'wrong'
  && (await text(`${S} [data-result]`)).includes(`Lane 0 ends at ${tn.final[0]}, 8 times its input`), await text(`${S} [data-result]`));
const doubled = await page.$$eval(`${S} [data-kind="doubled"]`, (els) => els.length);
check('red cells are the model\'s doubled lanes', doubled === tn.steps.reduce((a, s) => a + s.lanes.filter((l) => l.doubled).length, 0), String(doubled));
await select(S, 'participation test', 'threadIdx');
await clickText(S, 'Run to the end');
const tt = warpScanTrace(laneValues(8, 8), { test: 'threadIdx', warp: 1 });
const last = await cells(3);
check('threadIdx.x in warp 1: the model\'s values, wrong', same(last.map((c) => c[0]), tt.final) && (await text(`${S} [data-result]`)).includes('wrong'));
await clickText(S, '32 lanes');
await select(S, 'participation test', 'none');
await clickText(S, 'Run to the end');
check('32 lanes, no test: lane 0 at 32 times its input; only lane 31 right', (await text(`${S} [data-result]`)).includes('32 times its input') && (await text(`${S} [data-result]`)).includes('Only lane 31 survives'), await text(`${S} [data-result]`));
if (out) await page.screenshot({ path: out + '-stepper.png' });

// ---------------------------------------------------------------- Port it
const P = bench('Port it');
await page.waitForSelector(`${P} [data-port-listing] [data-line="20"]`);
const lineState = (l) => page.$eval(`${P} [data-line="${l}"]`, (e) => e.dataset.state);
await select(P, 'target', 'hip32');
for (const l of [20, 34]) await page.click(`${P} [data-line="${l}"]`);
check('two lines marked', (await lineState(20)) === 'marked' && (await lineState(34)) === 'marked');
await clickText(P, 'Check');
check('HIP on gfx10: both issues found', (await text(`${P} [data-port-result]`)).includes('You found 2 of 2'), await text(`${P} [data-port-result]`));
check('the mask line is reported as a compile error', (await text(`${P} [data-issue="mask"]`)).includes('compile error'));
check('issue rows are the model\'s', same(await page.$$eval(`${P} [data-issue]`, (els) => els.map((e) => e.dataset.issue)), portIssues('hip32').map((p) => p.id)));
await clickText(P, 'Clear');
await select(P, 'target', 'metal');
for (const l of [34, 59]) await page.click(`${P} [data-line="${l}"]`);
await clickText(P, 'Check');
check('Metal: 1 of 5 found, one extra line struck', (await text(`${P} [data-port-result]`)).includes('You found 1 of 5') && (await lineState(59)) === 'extra', await text(`${P} [data-port-result]`));
check('Metal: missed lines are marked missed', (await lineState(20)) === 'missed' && (await lineState(7)) === 'missed' && (await lineState(34)) === 'found');
check('Metal: the threadgroup array is a compile error', (await text(`${P} [data-issue="shared-in-helper"]`)).includes('compile error'));
if (out) await page.screenshot({ path: out + '-port.png' });
const v64 = laneValues(64);
for (const c of HALF_CASES) {
  await clickText(P, c.label);
  const got = await page.$$eval(`${P} [data-lane64]`, (els) => els.map((e) => e.dataset.ok));
  const model = halfCase(v64, c.key);
  const want = v64.map((_, l) => (model.wrong.includes(l) ? 'wrong' : 'right'));
  check(`64 lanes, ${c.key}: right and wrong lanes are the model's (${model.wrong.length} wrong)`, same(got, want));
}
check('mixed counting: lanes 32 to 63 wrong', halfCase(v64, 'mixed').wrong.length === 32 && halfCase(v64, 'mixed').wrong[0] === 32);

// ---------------------------------------------------------------- listings and table
const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent));
check('as printed tab', tabs.includes('as printed (CUDA only)'), tabs.join('|'));
check('lab 11.1 tabs: portable and two Metal', ['portable (lab 11.1)', 'Metal, shuffle loop (lab 11.1)', 'Metal, simd_prefix_inclusive_sum (lab 11.1)'].every((t) => tabs.includes(t)), tabs.join('|'));
check('as printed shows the 32-bit mask', (await text('.code pre')).includes('__shfl_up_sync(0xffffffff, val, stride)'));
await page.evaluate(() => [...document.querySelectorAll('.code .tabs button')].find((b) => b.textContent === 'portable (lab 11.1)').click());
await pause(100);
check('portable tab: typed mask, warpSize loop', (await text('.code pre')).includes('lane_mask_t') && (await text('.code pre')).includes('stride < (unsigned int)warpSize'));
check('backends table present', (await page.$$('table[data-backends] tbody tr')).length === 6);

// ---------------------------------------------------------------- questions
const q = async (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const sss = await q('What is the final value at position 9');
await (await sss.$('input[type=number]')).type('15');
await (await sss.$('button.primary')).click();
await pause(120);
check('q-sss accepts 15', (await sss.evaluate((e) => e.textContent)).includes('Correct.'));
const below = await q('What happens to lane 0 of a 32-lane warp?');
await (await below.$$('input[type=radio]'))[0].click();
await (await below.$('button.primary')).click();
await pause(120);
check('q-below-stride: doubling five times is correct', (await below.evaluate((e) => e.textContent)).includes('Correct.'));

const forgiven = errors.filter((e) => !(e.includes('404') && labMisses-- > 0));
check('no console or page errors', forgiven.length === 0, forgiven.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
