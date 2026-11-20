// Browser check for section 19.3: both narrated scenes, the unfold explorer, the Eq. 19.5
// trainer, the Fig. 19.11 erratum hunt, the Exercise 4 warp explorer, the layer-shape
// explorer, the lab listings, both predictions and a handful of questions. Every number it
// asserts is one the models in sim/unfold.ts and sim/implicitgemm.ts produce.
// Usage: node e2e/ch19-s193.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
let puts = 0;
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.method() === 'PUT' && r.url().includes('/api/progress')) {
    puts++;
    return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  }
  return r.continue();
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (t) => `section.bench[aria-label="${t}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const clickText = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const btn = [...document.querySelector(scope).querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(80);
  return done;
};
const setSelect = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    el.value = String(v);
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(120);
};
const setInput = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(120);
};
const click = async (sel) => {
  await page.$eval(sel, (e) => e.dispatchEvent(new MouseEvent('click', { bubbles: true })));
  await pause(100);
};
const q = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const numeric = async (needle, v) => {
  const h = await q(needle);
  await (await h.$('input[type=number]')).type(String(v));
  await (await h.$('button.primary')).click();
  await pause(150);
  return h.evaluate((e) => e.textContent);
};

await goto(page, 'ch19/19.3');
await page.waitForSelector(bench('Unfold explorer'));

// ---------------------------------------------------------------- page structure
const misplaced = await page.$$eval('.prose > aside.note', (notes) =>
  notes.filter((n) => {
    const p = n.previousElementSibling;
    return p && (p.matches('section.bench, figure.code, .wide') || p.classList.contains('q'));
  }).length,
);
check('every margin note follows a paragraph, never a bench, code block or question', misplaced === 0, `${misplaced} misplaced`);
const h2 = await page.$$eval('.prose > h2', (h) => h.map((x) => x.textContent));
check('section headings are sentence case', h2.every((t) => !/\s[A-Z][a-z]+\s[A-Z][a-z]/.test(t.replace(/Eq\. 19\.5|B\[u, v\]|Exercise 4|GEMM/g, ''))), h2.join(' | '));

// ---------------------------------------------------------------- the two scenes
const SCENES = [
  ['Unfolding the input into a matrix', 'unfold'],
  ['Implicit unfolding, one tile at a time', 'implicit-tiles'],
];
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').trim();
// Choose the part first and let it render (its position resets to 0), then move the slider:
// done in one go, the slider can still hold the old value, and setting it again fires nothing.
const seek = async (scene, part, at) => {
  const sel = `section.scene[aria-label="${scene}"]`;
  await page.evaluate((sel, part) => document.querySelector(sel).querySelectorAll('.seg button')[part].click(), sel, part);
  await pause(120);
  await page.evaluate((sel, at) => {
    const r = document.querySelector(sel).querySelector('input[type=range]');
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(r, String(Math.round(at * 1000)));
    r.dispatchEvent(new Event('input', { bubbles: true }));
  }, sel, at);
  await pause(150);
  return page.$eval(`section.scene[aria-label="${scene}"] svg`, (e) => e.textContent.replace(/\s+/g, ' '));
};
for (const [title, clip] of SCENES) {
  const S = `section.scene[aria-label="${title}"]`;
  check(`scene "${title}" has six parts`, (await page.$$(S + ' .seg button')).length === 6);
  const box = await page.$eval(S, (e) => [...e.querySelectorAll('label.check')].some((l) => l.textContent.includes('captions')));
  check(`scene "${title}" offers captions`, box);
  // Captions are the scripts themselves (shown regardless while a part has no audio).
  await page.evaluate((sel) => {
    const b = [...document.querySelector(sel).querySelectorAll('label.check')].find((l) => l.textContent.includes('captions')).querySelector('input');
    if (!b.checked) b.click();
  }, S);
  await pause(120);
  for (const n of [1, 6]) {
    await page.evaluate((sel, n) => document.querySelector(sel).querySelectorAll('.seg button')[n - 1].click(), S, n);
    await pause(120);
    const script = strip(await (await fetch(`${BASE}/content/ch19/narration/${clip}-${n}.txt`)).text());
    check(`scene "${title}" part ${n} caption is its script`, (await text(S + ' .caption')) === script, await text(S + ' .caption'));
  }
}
let svg = await seek(SCENES[0][0], 0, 0.95);
check('unfold part 1: the three partial sums make 14', svg.includes('7 + 5 + 2 = 14'), svg.slice(0, 300));
svg = await seek(SCENES[0][0], 1, 0.9);
check('unfold part 2: F is the same array read as 2 × 12', svg.includes('F: 2 × 12, the same array read row by row') && svg.includes('no data moves'));
svg = await seek(SCENES[0][0], 2, 0.97);
check('unfold part 3: column 0 complete, 12 of 12 entries', svg.includes('12 of 12 entries'), svg.slice(-200));
svg = await seek(SCENES[0][0], 3, 0.97);
check('unfold part 4: 48 entries from 27 values, copies 1 2 1 / 2 4 2 / 1 2 1', svg.includes('48 entries from 27 values') && svg.includes('copies of each input pixel in B so far'));
if (out) await page.screenshot({ path: out + '-unfold-part4.png' });
svg = await seek(SCENES[0][0], 4, 0.97);
check('unfold part 5: Y = F · B is 14 20 15 24 / 12 24 17 26, and its rows are the maps, no reshuffle', svg.includes('Y = F · B: 2 × 41420152412241726') && svg.includes('no reshuffle') && svg.includes('the output maps'), svg.slice(-400));
svg = await seek(SCENES[0][0], 5, 0.97);
check('unfold part 6: LeNet C1 at 19.14 against a limit of 25', svg.includes('19.14') && svg.includes('K = 5: limit 25'));
svg = await seek(SCENES[1][0], 0, 0.5);
check('implicit part 1: the block owns 14, 20 (map 0) and 12, 24 (map 1)', svg.includes('block (0, 0) owns 14, 20 (map 0) and 12, 24 (map 1)'));
svg = await seek(SCENES[1][0], 1, 0.97);
check('implicit part 2: three pixels fill four slots', svg.includes('3 input pixels fill 4 slots'));
svg = await seek(SCENES[1][0], 2, 0.9);
check('implicit part 3: phase 0 sums 3, 2, 1, 2 in registers', svg.includes('running sums (registers)') && svg.includes('3212'));
svg = await seek(SCENES[1][0], 4, 0.97);
check('implicit part 5: after phase 5 the sums are 14, 20, 12, 24', svg.includes('after phase 5: 14, 20, 12, 24, stored to Y'));
svg = await seek(SCENES[1][0], 5, 0.97);
check('implicit part 6: 96 direct, 48 written, 48 read back, 48 gathered', svg.includes('96') && (svg.match(/48/g) ?? []).length >= 4);
if (out) await page.screenshot({ path: out + '-implicit-part6.png' });
// Play runs on the clock without audio.
await page.evaluate((sel) => [...document.querySelector(sel).querySelectorAll('.seg button')][0].click(), `section.scene[aria-label="${SCENES[1][0]}"]`);
await page.evaluate((sel) => [...document.querySelector(sel).querySelectorAll('button')].find((b) => /Play|Resume/.test(b.textContent)).click(), `section.scene[aria-label="${SCENES[1][0]}"]`);
await pause(1500);
const pos = await page.$eval(`section.scene[aria-label="${SCENES[1][0]}"] input[type=range]`, (e) => Number(e.value));
check('playing moves the animation forward', pos > 30, String(pos));
await page.evaluate((sel) => [...document.querySelector(sel).querySelectorAll('button')].find((b) => b.textContent.trim() === 'Pause')?.click(), `section.scene[aria-label="${SCENES[1][0]}"]`);

// ---------------------------------------------------------------- unfold explorer
const E = bench('Unfold explorer');
const ev = (k) => text(`${E} [data-value="${k}"]`);
check("explorer starts on the book's example: ratio 1.78 against K² = 4", (await ev('ratio')) === '1.78' && (await text(E + ' [data-panel="unfold"]')).includes('against K² = 4'));
check('explorer: 27 inputs, 48 entries of B', (await ev('elements')) === 'X 27; B 48');
check('explorer: hovering Y[0, 0] shows the dot product 14', (await text(E + ' [data-readout="dot"]')).includes('= 14'));
await click(`${E} [data-y="1,2"]`);
check('explorer: Y[n, 1, 1, 0] = 17 from row 1 of F and column 2 of B', /Y\[n, 1, 1, 0\] = row 1 of F · column 2 of B = .* = 17/.test(await text(E + ' [data-readout="dot"]')), await text(E + ' [data-readout="dot"]'));
check('explorer: the hovered column of B is lit (12 cells)', (await page.$$(`${E} [data-part="b"] span.on`)).length === 12);
check('explorer: the patches in X are lit (12 cells)', (await page.$$(`${E} [data-part="x"] span.hot`)).length === 12);
const heat = await page.$$eval(`${E} [data-part="x"] [data-x^="0,"] sub`, (s) => s.map((x) => x.textContent).join(' '));
check('explorer: copies per pixel of map 0 are 1 2 1 / 2 4 2 / 1 2 1', heat === '×1 ×2 ×1 ×2 ×4 ×2 ×1 ×2 ×1', heat);
await click(`${E} [data-x="0,1,1"]`);
check('explorer: the centre pixel of map 0 is copied into B[3, 0], B[2, 1], B[1, 2], B[0, 3]', (await text(E + ' [data-readout="copies"]')).includes('4 entries of B: B[3, 0], B[2, 1], B[1, 2], B[0, 3]'), await text(E + ' [data-readout="copies"]'));
// Predict before LeNet C1.
const pr = await q('LeNet C1 (1 × 32 × 32 input, K = 5)');
await (await pr.$('input[type=number]')).type('20');
await pr.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.includes('Commit')).click());
await pause(150);
check('prediction p-ratio-c1 reveals 19.14', (await pr.evaluate((e) => e.textContent)).includes('19.14'));
await setSelect(`${E} select[aria-label="Layer"]`, 'c1');
check('explorer: LeNet C1 gives 19.14 against K² = 25, drawn to scale', (await ev('ratio')) === '19.14' && !!(await page.$(`${E} svg[aria-label="The three matrices drawn to scale"]`)));
await setSelect(`${E} select[aria-label="Layer"]`, 'c3');
check('explorer: LeNet C3 holds 15,000 entries for 1,176 inputs (12.76)', (await ev('elements')) === 'X 1,176; B 15,000' && (await ev('ratio')) === '12.76');
// A 64-channel 224 × 224 layer with K = 3 and a batch of 8: past Metal's documented floor for the batch only.
for (const [k, v] of [['C', 64], ['H', 224], ['W', 224], ['K', 3], ['M', 64], ['N', 8]]) await setInput(`${E} input[aria-label="${k}"]`, v);
check('explorer: 108.3 MiB per sample, 866.3 MiB for 8', (await ev('bytes')).startsWith('108.3 MiB per sample; 866.3 MiB for the batch of 8'), await ev('bytes'));
check('explorer: only the batch passes Metal\'s documented 256 MB', (await page.$eval(`${E} [data-value="flags"]`, (e) => e.dataset.flags)) === 'metal:batch');
check('explorer: the Metal flag says to check your own device', (await ev('flags')).includes("Check your own device's value"));
for (const [k, v] of [['C', 512], ['H', 512], ['W', 512], ['K', 11]]) await setInput(`${E} input[aria-label="${k}"]`, v);
check('explorer: a huge layer passes all three limits for one sample', (await page.$eval(`${E} [data-value="flags"]`, (e) => e.dataset.flags)) === 'int32:sample cudnn:sample metal:sample');
check('explorer: the ratio tends to K² (above 113 of 121)', Number(await ev('ratio')) > 113 && Number(await ev('ratio')) < 121, await ev('ratio'));
for (const [k, v] of [['C', 3], ['H', 224], ['W', 224], ['K', 8], ['N', 1]]) await setInput(`${E} input[aria-label="${k}"]`, v);
await page.evaluate((sel) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('stride = K')).querySelector('input').click(), E);
await pause(120);
check('explorer, Beyond: stride = K on 224 with K = 8 gives a ratio of 1', (await ev('ratio')) === '1', await ev('ratio'));
if (out) await page.screenshot({ path: out + '-explorer.png' });
await setSelect(`${E} select[aria-label="Layer"]`, 'book');

// ---------------------------------------------------------------- trainer
const T = bench('Where does B[u, v] come from?');
const answerStep = async (v) => {
  await setInput(`${T} [data-steps] input[type=number]`, v);
  await page.evaluate((sel) => [...document.querySelector(sel).querySelectorAll('[data-steps] button')].find((b) => b.textContent === 'Check').click(), T);
  await pause(100);
};
for (const v of [2, 1, 0, 1, 0, 24]) await answerStep(v);
const res = await text(`${T} [data-result]`);
check('trainer: B[10, 2] ← X[n, 2, 2, 0] = 3, all six right', res.includes('B[10, 2] ← X[n, 2, 2, 0] = 3') && res.includes('all six right'), res);
check('trainer: the answer is drawn as an arrow from B to X', !!(await page.$(`${T} line[marker-end]`)));
// A wrong answer is corrected with the arithmetic.
await setSelect(`${T} select[aria-label="Trainer layer"]`, 1);
check('trainer: the 4 × 4 layer starts at B[7, 5]', (await page.$eval(`${T} input[aria-label="u"]`, (e) => e.value)) === '7' && (await page.$eval(`${T} input[aria-label="v"]`, (e) => e.value)) === '5');
await answerStep(3);
check('trainer: a wrong channel is shown with its working (7 / 4 = 1)', (await text(`${T} [data-step="c"]`)).includes('You said 3. 7 / 4 = 1'), await text(`${T} [data-step="c"]`));
for (const v of [1, 1, 1, 2, 27]) await answerStep(v);
check('trainer: B[7, 5] ← X[n, 1, 2, 3], flat 27', (await text(`${T} [data-result]`)).includes('B[7, 5] ← X[n, 1, 2, 3]') && (await text(`${T} [data-step="flat"]`)).includes('= 27'));
// Reverse: X[n, 1, 1, 2] on the 4 × 4 layer is copied four times.
await clickText(T, 'X to every B[u, v]');
await click(`${T} [data-xcell="1,1,2"]`);
await setInput(`${T} input[aria-label="your count"]`, 4);
await clickText(T, 'Check');
const rev = await text(`${T} [data-reverse-result]`);
check('trainer, reverse: X[n, 1, 1, 2] is copied 4 times', rev.startsWith('Right') && (rev.match(/B\[/g) ?? []).length === 4, rev);
if (out) await page.screenshot({ path: out + '-trainer.png' });

// ---------------------------------------------------------------- erratum hunt
const H = bench('Fig. 19.11 as printed and corrected');
const pt = await q('how many of the 25 terms');
await (await pt.$('input[type=number]')).type('16');
await pt.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.includes('Commit')).click());
await pause(150);
check('prediction p-terms-c1 reveals 16', (await pt.evaluate((e) => e.textContent)).includes('16. 25 / 16 truncates to 1 phase'));
const hv = (k) => text(`${H} [data-value="${k}"]`);
check('hunt: LeNet C1 as printed runs 1 phase, 16 of 25 terms', (await hv('phases')).startsWith('1 (') && (await hv('terms')).startsWith('16 of 25'));
check('hunt: every one of the 4,704 outputs is wrong', (await page.$eval(`${H} [data-wrong]`, (e) => e.dataset.wrong)) === '4704');
check('hunt: the Y grid draws every output red', (await page.$$eval(`${H} svg rect`, (r) => r.filter((x) => x.getAttribute('fill') === 'var(--fail)').length)) === 4704);
await clickText(H, 'phases: ceiling');
check('hunt: the ceiling alone gathers past the end of X (5,488 reads)', (await hv('oobx')).startsWith('5,488'), await hv('oobx'));
await clickText(H, 'bounds checks');
check('hunt: ceiling plus bounds checks matches the loop nest', (await page.$eval(`${H} [data-verdict]`, (e) => e.dataset.verdict)) === 'correct');
await page.evaluate((sel) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('planted bug')).querySelector('input').click(), H);
await pause(150);
check('hunt: the planted bug (W for W_out) breaks it', (await page.$eval(`${H} [data-verdict]`, (e) => e.dataset.verdict)) === 'wrong');
await setSelect(`${H} select[aria-label="Shape"]`, 'c3');
await page.evaluate((sel) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('planted bug')).querySelector('input').click(), H);
await clickText(H, 'phases: (C·K·K) / T, as printed');
await clickText(H, 'no bounds checks');
check('hunt: LeNet C3 as printed keeps 144 of 150 terms', (await hv('terms')).startsWith('144 of 150'));
check('hunt: and 180 slots are hit by a stray store', (await hv('wrong')).includes('(180 hit by a stray store)'), await hv('wrong'));
if (out) await page.screenshot({ path: out + '-hunt.png' });
await setSelect(`${H} select[aria-label="Shape"]`, 'book');
check("hunt: the book's example is right as printed", (await page.$eval(`${H} [data-verdict]`, (e) => e.dataset.verdict)) === 'correct');
const table = await page.$$eval(`${H} [data-table="outcomes"] tbody tr`, (rows) => rows.map((r) => [...r.querySelectorAll('td[data-outcome]')].map((c) => (c.dataset.outcome === 'correct' ? 1 : 0)).join('')));
check('hunt: the outcome table says which fix each shape needs', JSON.stringify(table) === JSON.stringify(['1111', '0001', '0001', '1111', '0011']), JSON.stringify(table));
check('question q-phases: 6 terms dropped', /Correct\./.test(await numeric("How many of each output's 150 terms are dropped", 6)));

// ---------------------------------------------------------------- warp explorer
const Wp = bench("One warp's accesses in Fig. 19.11");
const label = async (i) => page.$$eval(`${Wp} .mx-label.t-global`, (l, i) => l[i].textContent, i);
check('warp: C3 F load is 2 runs of 16', (await label(0)).includes('2 runs (16 + 16 floats)'), await label(0));
check('warp: C3 gather is 4 runs, 10 + 6 + 10 + 6', (await label(1)).includes('4 runs (10 + 6 + 10 + 6 floats)'), await label(1));
check('warp: each row of threads gathers in 2 runs with jumps of K = 5', (await text(`${Wp} [data-readout="gather"]`)).includes('ty 0 in 2 runs, ty 1 in 2 runs; the jumps between runs are 5 (K = 5)'));
check('warp: the jump of 5 is drawn between the runs', (await page.$$eval(`${Wp} [data-jump]`, (j) => j.map((x) => x.textContent))).filter((t) => t === '+5').length === 2);
const passes = await page.$$eval(`${Wp} [data-table="banks"] [data-passes]`, (c) => c.map((x) => x.dataset.passes).join(''));
check('warp: no bank conflict on the 32-bank model for any of the four shared accesses', passes === '1111', passes);
await setSelect(`${Wp} select[aria-label="Block column"]`, 6);
check('warp: edge block 6 of C3 stores 4 + 4 outputs', (await label(2)).includes('2 runs (4 + 4 floats)'), await label(2));
await setSelect(`${Wp} select[aria-label="Inner step"]`, 9);
check('warp: still conflict-free at inner step 9', (await page.$$eval(`${Wp} [data-table="banks"] [data-passes]`, (c) => c.map((x) => x.dataset.passes).join(''))) === '1111');
check('question q-ex4: 2 runs', /Correct\./.test(await numeric('How many runs of consecutive addresses', 2)));

// ---------------------------------------------------------------- shapes explorer
const Sh = bench('Which layers suit a GEMM');
const pad = (id) => page.$eval(`${Sh} tr[data-layer="${id}"] [data-padding]`, (e) => Number(e.dataset.padding));
const tiles = (id) => text(`${Sh} tr[data-layer="${id}"] [data-value="tiles"]`);
check('shapes: LeNet C5 at batch 1 leaves 94% of the tile slots empty', Math.abs((await pad('c5')) - 0.9414) < 1e-4, String(await pad('c5')));
check('shapes: LeNet C3 needs 10 phases, 300 / 20 loads per output', (await text(`${Sh} tr[data-layer="c3"]`)).includes('300 / 20'));
await setInput(`${Sh} input[aria-label="N (batch)"]`, 16);
check('shapes: one GEMM per sample, batch 16: C5 still 94% padding, 128 blocks', Math.abs((await pad('c5')) - 0.9414) < 1e-4 && (await tiles('c5')) === '128');
await clickText(Sh, 'whole batch in one GEMM (needs a permute)');
check('shapes: the whole batch in one GEMM: C5 down to 6% padding in 8 blocks, and a permute', Math.abs((await pad('c5')) - 0.0625) < 1e-4 && (await tiles('c5')) === '8' && (await text(`${Sh} tr[data-layer="c5"]`)).includes('comes out M × N × P'));
check('shapes: the lab\'s 1 × 1 layer is 256 × 256 × 3,136 for 16 samples', (await text(`${Sh} tr[data-layer="pointwise"]`)).includes('256 × 256 × 3,136'));

// ---------------------------------------------------------------- code from the lab
const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
check('Fig. 19.11 is shown from the lab in CUDA and Metal', tabs.includes('CUDA (also HIP, emulator)') && tabs.includes('Metal'), tabs.join('|'));
const listing = await page.$eval('figure.code pre', (e) => e.textContent);
check('the CUDA tab is the provided, corrected kernel', listing.includes('ConvLayer_MM_Kernel') && listing.includes('(CKK + TILE_WIDTH - 1) / TILE_WIDTH'));
check('its status comes from this machine\'s lab runs', (await page.$eval('figure.code .status', (e) => e.textContent)).startsWith('tested with lab 19.1'));
const R = '[data-part="solution-regions"]';
await clickText(R, "Show the reference solution's unroll and GEMM call");
await pause(300);
const rtabs = await page.$$eval(`${R} .tabs button`, (b) => b.map((x) => x.textContent));
check('the solution pieces: unroll in CUDA and Metal, the GEMM path in CUDA and Metal host code', JSON.stringify(rtabs) === JSON.stringify(['unroll, kernel.cu', 'unroll, kernel.metal', 'gemm-path, kernel.cu', 'gemm-path, launch.mm']), JSON.stringify(rtabs));
check('the unroll piece is the kernel that writes Bt', (await page.$eval(`${R} pre`, (e) => e.textContent)).includes('__global__ void unrollKernel'));
await page.evaluate((sel) => [...document.querySelectorAll(sel + ' .tabs button')].find((b) => b.textContent === 'gemm-path, kernel.cu').click(), R);
await pause(100);
check('the GEMM path passes F as X and Bt as W', (await page.$eval(`${R} pre`, (e) => e.textContent)).includes('launchLinear(F, Bt, Y + (size_t)n * M * P, M, C * K * K, P);'));

// ---------------------------------------------------------------- questions
check('question q-belems: 15000', /Correct\./.test(await numeric('How many elements does B hold for one sample', 15000)));
check('question q-map: 27', /Correct\./.test(await numeric('copies which flat element', 27)));
const roles = await q('without copying F, what do you pass');
await roles.evaluate((e) => e.querySelectorAll('.opts input')[0].click());
await roles.evaluate((e) => e.querySelector('button.primary').click());
await pause(150);
check('question q-roles: X = F, W = Bᵀ is right', (await roles.evaluate((e) => e.textContent)).includes('Correct.'));
const wb = await q('run a single GEMM with F');
await wb.evaluate((e) => e.querySelectorAll('.opts input')[1].click());
await wb.evaluate((e) => e.querySelector('button.primary').click());
await pause(150);
check('question q-wholebatch: claiming NCHW for free is marked wrong, with the reason', (await wb.evaluate((e) => e.textContent)).includes('Not quite.') && (await wb.evaluate((e) => e.textContent)).includes('the sample index sits inside the map index'));
check('answers were sent to the (intercepted) progress endpoint', puts > 0, String(puts));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
