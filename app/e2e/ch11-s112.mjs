// Browser check for section 11.2: the Fig. 11.3 listing, the stride-by-stride stepper
// (levels, final positions, additions, the y15 tree), the warp explorer (exercise 2 at
// widths 32 and 64), the race hunt (the book's race, other orders, each barrier), the
// outcome table, the narrated scene "Doubling strides" (scripts, parts, captions, every
// part's drawing) and questions. Expected values come from src/sim/ksblock.ts, imported
// directly. Passes with or without content/ch11/narration/manifest.json.
// Usage: node e2e/ch11-s112.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import {
  EXERCISE_1,
  FIG_11_3,
  SCHEDULES,
  divergence,
  kernelSource,
  levels,
  rangeShort,
  run,
  totalAdditions,
  treeOf,
  warpStates,
} from '../src/sim/ksblock.ts';

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
const click = async (scope, label) => {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(80);
  return ok;
};
const select = async (scope, aria, value) => {
  await page.$eval(`${scope} select[aria-label="${aria}"]`, (s, v) => {
    s.value = String(v);
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await pause(100);
};
const setRange = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(100);
};
const tick = async (scope, label, on) => {
  await page.evaluate((scope, label, on) => {
    const i = [...document.querySelectorAll(scope + ' label.check')].find((l) => l.textContent.includes(label)).querySelector('input');
    if (i.checked !== on) i.click();
  }, scope, label, on);
  await pause(100);
};
/** Cells of a KsGrid inside `scope`: row -> pos -> {value, range, state}. */
const cells = (scope) =>
  page.$$eval(scope + ' svg[data-part="ks-grid"] g[data-part="cell"]', (gs) =>
    gs.map((g) => ({ row: Number(g.dataset.row), pos: Number(g.dataset.pos), value: g.dataset.value, range: g.dataset.range, state: g.dataset.state })),
  );
const rowValues = (cs, row) => cs.filter((c) => c.row === row).sort((a, b) => a.pos - b.pos).map((c) => c.value);

await goto(page, 'ch11/11.2');
await page.waitForSelector(bench('Kogge-Stone, stride by stride'));

// ---- the listing
await page.waitForFunction(() => [...document.querySelectorAll('.code pre')].some((p) => p.textContent.includes('kogge_stone_scan_kernel')));
const fig = await page.$$eval('.code', (cs) => cs.map((c) => ({ text: c.textContent, n: c.querySelectorAll('.ln').length, hl: [...c.querySelectorAll('.ln.hl')].map((l) => l.dataset.n + ':' + l.textContent.trim()) })).find((c) => c.text.includes('kogge_stone_scan_kernel') && c.text.includes('Fig. 11.3')));
check('Fig. 11.3 is shown from the snippet file, 21 lines numbered as on the page', !!fig && fig.n === 21, JSON.stringify(fig?.n));
check('lines 10 and 14, the two barriers, are marked', !!fig && same(fig.hl, ['10:__syncthreads();', '14:__syncthreads();']), JSON.stringify(fig?.hl));
check('the printed typo is fixed and said so', !!fig && !fig.text.replace('__ syncthreads();</code>', '').includes('__ syncthreads();\n') && fig.text.includes('with a space that would not compile'));

// ---- stride by stride
const St = bench('Kogge-Stone, stride by stride');
const L = levels(EXERCISE_1);
let cs = await cells(St);
check('the stepper starts on exercise 1\'s input, one row', same(rowValues(cs, 0), EXERCISE_1.map(String)) && !cs.some((c) => c.row > 0));
for (let k = 1; k < L.length; k++) {
  await click(St, 'Step');
  cs = await cells(St);
  const rd = await page.$eval(St + ' [data-readout="stride"]', (e) => ({ fin: Number(e.dataset.final), add: Number(e.dataset.additions), text: e.textContent }));
  check(`after stride ${L[k].stride}: the row is the model's (${L[k].values.join(' ')}), ranges included`, same(rowValues(cs, k), L[k].values.map(String)) && same(cs.filter((c) => c.row === k).sort((a, b) => a.pos - b.pos).map((c) => c.range), L[k].coef.map(rangeShort)), JSON.stringify(rowValues(cs, k)));
  check(`after stride ${L[k].stride}: the first ${2 ** k} positions are final and shaded`, rd.fin === Math.min(8, 2 ** k) && cs.filter((c) => c.row === k && c.state === 'final').length === Math.min(8, 2 ** k), JSON.stringify(rd));
}
let rd = await page.$eval(St + ' [data-readout="stride"]', (e) => e.textContent);
check('done: 17 additions = 8·3 − 7 in 3 steps, against 7 for the loop', rd.includes(`${totalAdditions(8)} additions = 8·3 − 7`) && rd.includes('against 7 for the loop'), rd);
check('Step is disabled at the end; Back steps back', !(await click(St, 'Step')) && (await click(St, 'Back')) && (await page.$eval(St + ' [data-readout="level"]', (e) => e.textContent)).includes('stride 2'));
await click(St, 'sixteen values');
for (let i = 0; i < 4; i++) await click(St, 'Step');
rd = await page.$eval(St + ' [data-readout="stride"]', (e) => e.textContent);
check('sixteen values: 49 additions, and the tree behind y15 uses 15 of them', rd.includes('49 additions = 16·4 − 15') && rd.includes('y15 uses 15'), rd);
const marks = await page.$$eval(St + ' rect[data-part="mark"][data-tone="tree"]', (ms) => ms.map((m) => `${m.dataset.row}:${m.dataset.pos}`).sort());
const want = treeOf(15, 16).nodes.filter((n) => n.level > 0).map((n) => `${n.level}:${n.pos}`).sort();
check('the y15 tree marks are the heavy boxes of Fig. 11.2 (8, 4, 2, 1)', same(marks, want), JSON.stringify(marks));
if (out) await page.screenshot({ path: out + '-stride.png' });

// ---- warps
const W = bench('Which warps diverge?');
let loopRd = await page.$eval(W + ' [data-readout="loop"]', (e) => ({ it: Number(e.dataset.iterations), text: e.textContent }));
check('256 threads, width 32: 5 divergent iterations, strides 1 to 16', loopRd.it === divergence(256, 32).iterations && loopRd.it === 5 && loopRd.text.includes('1, 2, 4, 8, 16'), JSON.stringify(loopRd));
await click(W, '64: AMD gfx9');
loopRd = await page.$eval(W + ' [data-readout="loop"]', (e) => ({ it: Number(e.dataset.iterations), text: e.textContent }));
check('width 64: 6 iterations, strides 1 to 32', loopRd.it === 6 && loopRd.text.includes('1, 2, 4, 8, 16, 32'), JSON.stringify(loopRd));
await click(W, '32: the book, NVIDIA');
await setRange(W + ' input[type=range]', 2);
const warps = await page.$$eval(W + ' g[data-part="warp"]', (gs) => gs.map((g) => ({ state: g.dataset.state, active: Number(g.dataset.active) })));
check('at stride 4, warp 0 has 28 of 32 lanes active and is divergent; the rest are full', same(warps.map((w) => w.state), warpStates(256, 32, 4).map((w) => w.state)) && warps[0].active === 28 && warps[0].state === 'divergent', JSON.stringify(warps));
await setRange(W + ' input[type=range]', 6);
const w64 = await page.$$eval(W + ' g[data-part="warp"]', (gs) => gs.map((g) => g.dataset.state));
check('at stride 64, warps 0 and 1 are idle, none divergent', same(w64, warpStates(256, 32, 64).map((w) => w.state)) && w64[0] === 'idle' && w64[1] === 'idle' && !w64.includes('divergent'), JSON.stringify(w64));
await select(W, 'Threads per block', 1024);
check('1024 threads draw 32 warps', (await page.$$(W + ' g[data-part="warp"]')).length === 32);

// ---- the race hunt
const R = bench('The race behind barrier 2');
const race = run({ ...FIG_11_3, threads: 8, barrier2: false }, EXERCISE_1, 'book-race');
check('the hunt starts with barrier 2 removed and the book\'s order, and offers a failure', await click(R, 'Go to the first failure'));
let n = await page.$eval(R + ' [data-readout="narrate"]', (e) => e.textContent);
check('the first failure is thread 6 reading slot 4 after thread 4 overwrote it', n.includes('Thread 6, stride 2') && n.includes('overwritten before read') && n.includes(race.violations[0].message), n);
cs = await cells(R);
check('the slot thread 6 read is drawn red: x1..x4 in the stride-2 row', cs.some((c) => c.row === 2 && c.pos === 4 && c.range === 'x1..x4') && (await page.$$(R + ' rect[data-part="mark"][data-tone="bad"]')).length === 1);
const hl = await page.$$eval(R + ' .code .ln.hl', (ls) => ls.map((l) => l.dataset.n));
check('the code marks the read, line 13', same(hl, kernelSource({ ...FIG_11_3, barrier2: false }).line.read.map(String)), JSON.stringify(hl));
await click(R, 'Run to the end');
let o = await page.$eval(R + ' [data-readout="outcome"]', (e) => ({ ok: e.dataset.correct, text: e.textContent }));
check('the end: only position 6 is wrong, x0 + 2x1 + 2x2 + x3 + x4 + x5 + x6 = 46 instead of 33', o.ok === '0' && o.text.includes('1 of 8 outputs wrong') && o.text.includes('46 (x0 + 2x1 + 2x2 + x3 + x4 + x5 + x6) instead of 33'), o.text);
if (out) await page.screenshot({ path: out + '-race.png' });
const outcome = async () => {
  await click(R, 'Run to the end');
  return page.$eval(R + ' [data-readout="outcome"]', (e) => e.dataset.correct === '1');
};
await select(R, 'Thread order', 'reversed');
check('last thread first: the same kernel happens to be right', (await outcome()) === run({ ...FIG_11_3, threads: 8, barrier2: false }, EXERCISE_1, 'reversed').correct && (await outcome()));
await select(R, 'Thread order', 'lockstep');
check('lockstep hides the bug', await outcome());
await select(R, 'Thread order', 'in-order');
check('thread 0 first: wrong', !(await outcome()));
await tick(R, 'barrier 2', true);
check('with barrier 2 back, thread 0 first is right', await outcome());
await tick(R, 'barrier 1', false);
await select(R, 'Thread order', 'reversed');
check('without barrier 1, last thread first reads unloaded slots and is wrong', !(await outcome()) && (await page.$eval(R + ' [data-readout="narrate"]', (e) => e.textContent)).length > 0);
await click(R, 'Go to the first failure');
n = await page.$eval(R + ' [data-readout="narrate"]', (e) => e.textContent);
check('and the first failure says read before write', n.includes('read before write'), n);
await tick(R, 'barrier 1', true);
await tick(R, 'add straight into', true);
await select(R, 'Thread order', 'book-race');
const directSrc = await page.$eval(R + ' .code pre', (e) => e.textContent);
check('the direct variant prints direct_add_scan_kernel with +=', directSrc.includes('direct_add_scan_kernel') && directSrc.includes('+= buffer_s'));
check('straight into buffer_s, the book\'s order is wrong even with both barriers', !(await outcome()));

// ---- the outcome table
const O = bench('Every thread order at once');
const table = await page.$$eval(O + ' tbody tr', (rs) => rs.map((r) => [...r.querySelectorAll('td[data-outcome]')].map((c) => c.dataset.outcome)));
const variants = [{}, { barrier2: false }, { barrier1: false }, { temp: false }];
const expect = variants.map((v) => SCHEDULES.map((s) => (run({ ...FIG_11_3, threads: 8, ...v }, EXERCISE_1, s).correct ? 'correct' : 'wrong')));
check('the outcome table is the model\'s, cell by cell', same(table, expect), JSON.stringify(table));
check('as printed, every order is correct; lockstep is correct for every kernel', table[0].every((c) => c === 'correct') && table.every((r) => r[0] === 'correct'));

// ---- the scene
const SC = 'section.scene[aria-label="Doubling strides"]';
await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const scripts = [];
for (let i = 1; i <= 6; i++) {
  const res = await fetch(`${BASE}/content/ch11/narration/ks-doubling-${i}.txt`);
  scripts.push(res.ok ? strip(await res.text()) : '');
}
const words = scripts.map((s) => s.split(' ').length);
check('six scripts, each 35 to 60 words', scripts.every(Boolean) && words.every((w) => w >= 35 && w <= 60), words.join(' '));
check('scripts read no symbols aloud', scripts.every((s) => !/[<>=×%/_(){}+−#]/.test(s)), scripts.find((s) => /[<>=×%/_(){}+−#]/.test(s)));
check('the counting script uses the model\'s numbers: 7, 6, 4, 17 against 7', scripts[4].includes(`${L[1].added.filter(Boolean).length}, then ${L[2].added.filter(Boolean).length}, then ${L[3].added.filter(Boolean).length} additions: ${totalAdditions(8)} in all`) && scripts[4].includes('only 7'), scripts[4]);
const labels = await page.$$eval(SC + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts with their names', same(labels, ['1. Eight values', '2. Stride 1', '3. Stride 2', '4. Stride 4', '5. The count', '6. The race']), labels.join(' | '));
const captionsBox = () => page.evaluateHandle((sc) => [...document.querySelectorAll(sc + ' label.check')].find((l) => l.textContent.includes('captions')).querySelector('input'), SC);
check('captions are off by default', !(await (await captionsBox()).evaluate((e) => e.checked)));
await (await captionsBox()).click();
await pause(120);
const caption = () => page.$eval(SC + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim());
check('captions on: part 1\'s script is shown', (await caption()) === scripts[0], await caption());
const seek = (v) => setRange(SC + ' input[type=range]', v);
const sceneHead = () => page.$eval(SC + ' [data-part="headline"]', (e) => e.textContent);
await page.evaluate((sc) => [...document.querySelectorAll(sc + ' label.check')].find((l) => l.textContent.includes('sound')).querySelector('input').click(), SC);
await click(SC, 'Play');
await pause(1500);
const moved = await page.$eval(SC + ' input[type=range]', (e) => Number(e.value));
check('with the sound off it plays on estimated timings', moved > 20, String(moved));
await click(SC, 'Pause');
await seek(990);
await click(SC, 'Resume');
await pause(1800);
const cur = await page.$$eval(SC + ' .seg button', (bs) => bs.findIndex((b) => b.getAttribute('aria-pressed') === 'true') + 1);
check('at the end of part 1, part 2 starts by itself, with its caption', cur === 2 && (await caption()) === scripts[1], `part ${cur}`);
await click(SC, 'Pause');
for (let k = 1; k <= 3; k++) {
  await click(SC, labels[k]);
  await seek(1000);
  cs = await cells(SC);
  check(`part ${k + 1}: rows 0 to ${k} are the model's levels`, [0, 1, 2, 3].slice(0, k + 1).every((r) => same(rowValues(cs, r), L[r].values.map(String))) && !cs.some((c) => c.row > k), JSON.stringify(rowValues(cs, k)));
}
check('part 4 ends with every position final', cs.filter((c) => c.row === 3 && c.state === 'final').length === 8);
await click(SC, '5. The count');
await seek(1000);
const bars = await page.$eval(SC + ' [data-part="bars"]', (e) => [Number(e.dataset.ks), Number(e.dataset.loop)]);
check('part 5: the bars count 17 and 7', same(bars, [17, 7]) && (await sceneHead()).includes('17 additions against 7'));
await click(SC, '6. The race');
await seek(200);
cs = await cells(SC);
check('part 6 begins with thread 4\'s early store, x1..x4, and nothing else in the stride-2 row', cs.filter((c) => c.row === 2 && c.state !== 'carried' && c.state !== 'empty').map((c) => c.pos + ':' + c.range).join() === '4:x1..x4');
await seek(1000);
cs = await cells(SC);
const six = cs.find((c) => c.row === 3 && c.pos === 6);
check('part 6 ends with position 6 red: 46, x0..x6 +x1+x2, not 33', six.state === 'bad' && six.value === '46' && six.range === rangeShort(race.outputCoef[6]) && (await sceneHead()).includes('= 46, not 33'), JSON.stringify(six));
if (out) await page.screenshot({ path: out + '-scene.png' });

// ---- questions
const ask = async (text, value) => {
  const q = await page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(t)), text);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
check('q-ex1 accepts 18', (await ask('Which value is at position 5', 18)).includes('Correct.'));
check('q-ex2-64 accepts 6', (await ask('64-wide wavefronts', 6)).includes('Correct.'));
check('q-ex3 accepts 20,481 and gives the two-block reading', (await ask('Exercise 3', 20481)).includes('18,434'));
const qr = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('count twice at the end')));
await qr.evaluate((e) => [...e.querySelectorAll('label')].find((l) => l.textContent.trim() === 'x1 and x2').querySelector('input').click());
await (await qr.$('button.primary')).click();
await pause(120);
check('q-race accepts x1 and x2, and names the missing x0', (await qr.evaluate((e) => e.textContent)).includes('Correct.') && (await qr.evaluate((e) => e.textContent)).includes('without x0'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
