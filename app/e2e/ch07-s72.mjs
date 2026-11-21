// Browser check for section 7.2: the erratum hunt on Fig. 7.7 and the corrected kernel
// from lab 7.1, the thread-to-taps stepper, the divergence explorer, the narrated
// animation "One thread per output", and the questions. Expected values come from
// src/sim/convolve.ts. It passes with or without content/ch07/narration/manifest.json:
// with no audio the player runs on estimated timings and shows each script regardless.
// Usage: node e2e/ch07-s72.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { checkLabStatuses } from './labstatus.mjs';
import { basicTaps, laneActivity, mapThread, pyramidFilter, stripsOf, toyImage, warpDivergence } from '../src/sim/convolve.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
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
const norm = (s) => s.replace(/\s+/g, ' ').trim();
const bench = (t) => `section.bench[aria-label="${t}"]`;
const HUNT = bench('Fig. 7.7 as printed');
const ST = bench('From thread to taps');
const WX = bench('Which warps diverge');
const SC = 'section.scene[aria-label="One thread per output"]';

async function setRange(sel, v) {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(100);
}
async function setSelect(sel, v) {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(100);
}
async function press(scope, label, exact = true) {
  const ok = await page.evaluate((scope, label, exact) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => (exact ? x.textContent.trim() === label : x.textContent.trim().includes(label)));
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label, exact);
  await pause(100);
  return ok;
}
const question = (text) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.replace(/\s+/g, ' ').includes(t)), text);
async function answerChoice(text, optionStart) {
  const q = await question(text);
  const found = await q.evaluate((e, o) => {
    const label = [...e.querySelectorAll('.opts label')].find((l) => l.textContent.replace(/\s+/g, ' ').trim().startsWith(o));
    if (!label) return false;
    label.querySelector('input').click();
    return true;
  }, optionStart);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return { found, text: norm(await q.evaluate((e) => e.textContent)) };
}
async function answerNumber(text, value) {
  const q = await question(text);
  const input = await q.$('input[type=number]');
  await input.click();
  const len = await input.evaluate((e) => e.value.length);
  for (let i = 0; i < len; i++) await page.keyboard.press('Backspace');
  await input.type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return norm(await q.evaluate((e) => e.textContent));
}

await goto(page, 'ch07/7.2');
await page.waitForSelector(HUNT);

// ---- erratum hunt on the printed lines
check('the hunt quotes printed lines 02, 07, 08, 09, 10 and 14', JSON.stringify(await page.$$eval(HUNT + ' tr[data-line]', (t) => t.map((x) => x.dataset.line))) === JSON.stringify(['02', '07', '08', '09', '10', '14']));
check('line 10 is quoted as printed, F[fRow][fCol]', (await page.$eval(HUNT + ' tr[data-line="10"]', (e) => e.textContent)).includes('F[fRow][fCol]'));
check('the corrected kernel is hidden before checking', (await page.$(ST + ' .code')) !== null && !(await page.evaluate(() => [...document.querySelectorAll('.code .tabs button')].some((b) => b.textContent === 'Metal host'))));
const picks = { '02': 'ok', '07': 'undeclared', '08': 'undeclared', '09': 'undeclared', '10': 'subscript', '14': 'subscript' };
for (const [n, v] of Object.entries(picks)) await setSelect(`${HUNT} select[aria-label="Line ${n}"]`, v);
check('Check is enabled once every line has a verdict', await press(HUNT, 'Check'));
const res = await page.$$eval(HUNT + ' tr[data-line]', (t) => Object.fromEntries(t.map((x) => [x.dataset.line, x.dataset.result])));
check('line 09 is fine as printed: marked wrong when called undeclared', res['09'] === 'wrong' && res['07'] === 'right' && res['10'] === 'right', JSON.stringify(res));
check('the verdict counts 5 of 6', (await page.$eval(HUNT + ' [data-out="hunt"]', (e) => e.dataset.right)) === '5');
const why09 = norm(await page.$eval(HUNT + ' tr[data-line="09"]', (e) => e.textContent));
check('and explains that 09 only fails because 07 and 08 never declared the names', why09.includes('only because 07 and 08 never declared'), why09);
await press(HUNT, 'Try again');
for (const [n, v] of Object.entries({ ...picks, '09': 'ok' })) await setSelect(`${HUNT} select[aria-label="Line ${n}"]`, v);
await press(HUNT, 'Check');
check('all six right on the second try', (await page.$eval(HUNT + ' [data-out="hunt"]', (e) => e.dataset.right)) === '6');
await page.waitForFunction(() => [...document.querySelectorAll('.code .tabs button')].some((b) => b.textContent === 'Metal host'), { timeout: 8000 });
const tabs = await page.$$eval('.code .tabs', (ts) => ts.map((t) => [...t.querySelectorAll('button')].map((b) => b.textContent)).find((l) => l.includes('Metal host')));
check('checking reveals the corrected kernel: CUDA, Metal kernel, Metal host', JSON.stringify(tabs) === JSON.stringify(['CUDA', 'Metal kernel', 'Metal host']), JSON.stringify(tabs));
const listing = await page.evaluate(() => {
  const fig = [...document.querySelectorAll('figure.code')].find((f) => [...f.querySelectorAll('.tabs button')].some((b) => b.textContent === 'Metal host'));
  return { code: fig.querySelector('pre').textContent, status: fig.querySelector('.tabs .status')?.textContent ?? '' };
});
check('the CUDA tab is lab 7.1\'s basic part only', listing.code.includes('convBasicKernel') && listing.code.includes('if (outRow >= height || outCol >= width) return;') && !listing.code.includes('@part') && !listing.code.includes('convConstKernel'), listing.code.slice(0, 200));
check('its status uses the fixed vocabulary', /emulator-checked|unverified/.test(listing.status) && !/verified on CUDA/.test(listing.status) === !listing.status.includes('verified on CUDA'), listing.status);
await page.evaluate(() => {
  const fig = [...document.querySelectorAll('figure.code')].find((f) => [...f.querySelectorAll('.tabs button')].some((b) => b.textContent === 'Metal host'));
  [...fig.querySelectorAll('.tabs button')].find((b) => b.textContent === 'Metal kernel').click();
});
await pause(100);
const metal = await page.evaluate(() => [...document.querySelectorAll('figure.code')].find((f) => [...f.querySelectorAll('.tabs button')].some((b) => b.textContent === 'Metal host')).querySelector('pre').textContent);
check('the Metal tab converts the unsigned position to int', metal.includes('int outCol = int(gid.x);') && metal.includes('kernel void conv_basic'), metal.slice(0, 200));
if (out) await page.$eval(HUNT, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-hunt.png' });

let r = await answerChoice('The printed kernel has no output guard', 'Threads past the right edge');
check('q-guard: stray stores into the next row and past the end', r.text.includes('Correct.') && r.text.includes('P[601]'), r.text.slice(-300));

// ---- stepper
const N = toyImage(16, 16);
const step = () => page.$eval(ST + ' [data-out="step"]', (e) => ({ ...e.dataset, text: e.textContent.replace(/\s+/g, ' ') }));
const hl = () => page.$$eval(ST + ' .code .ln.hl', (ls) => ls.map((l) => l.textContent.trim()));
await page.waitForSelector(ST + ' .code pre');
let s = await step();
check('default: thread (1,1) of block (1,1) computes P[5][5], corner (3, 3)', s.outrow === '5' && s.outcol === '5' && s.corner === '3,3', JSON.stringify(s));
let h = await hl();
check('step 0 highlights the two index lines of the lab kernel', h.length === 2 && h[0].startsWith('int outCol') && h[1].startsWith('int outRow'), JSON.stringify(h));
await press(ST, 'Step');
s = await step();
const run55 = basicTaps(16, 16, 2, 5, 5, N.data, pyramidFilter(2).data);
check('step 1 is tap 1: inRow 3, inCol 3, N[51], with its product', s.text.includes('Tap 1 of 25') && s.text.includes('N[3·16 + 3]') && s.text.includes(`= ${run55.taps[0].weight} × ${run55.taps[0].value} = ${run55.taps[0].product}`), s.text);
h = await hl();
check('a passing tap highlights inRow, inCol, the ghost test and the multiply-add', h.length === 4 && h[0].startsWith('int inRow') && h[2].startsWith('if (inRow >= 0') && h[3].startsWith('Pvalue +='), JSON.stringify(h));
check('the current input cell and weight are lit', (await page.$$(ST + ' [data-grid="N"] span.hot')).length === 1);
await press(ST, 'Run to the store');
s = await step();
check(`at the store Pvalue = ${run55.pvalue}, the model and the reference`, Number(s.pvalue) === run55.pvalue && s.pvalue === s.ref, JSON.stringify(s));
check('the store step names P[85] and the match chip is shown', s.text.includes('P[85]') && (await page.$eval(ST + ' [data-out="match"]', (e) => e.className)).includes('ok'));
check('the store line is highlighted', (await hl()).join('|').includes('= Pvalue;'));
// q-map's thread
await setSelect(ST + ' select[aria-label="blockIdx"]', '2,1');
await setSelect(ST + ' select[aria-label="threadIdx"]', '3,0');
s = await step();
const m = mapThread({ bx: 2, by: 1, tx: 3, ty: 0, bdx: 4, bdy: 4 }, 2);
check(`block (2,1), thread (3,0): output (4, 11), corner (2, 9), the model's`, s.outrow === String(m.outRow) && s.outcol === String(m.outCol) && s.corner === `${m.cornerRow},${m.cornerCol}`, JSON.stringify(s));
// click an output: P[0][1]
await page.click(ST + ' [data-grid="P"] [data-rc="0,1"]');
await pause(100);
s = await step();
check('clicking P[0][1] selects its thread', s.outrow === '0' && s.outcol === '1' && s.text.includes('blockIdx.x·blockDim.x + threadIdx.x = 0·4 + 1'), s.text);
await press(ST, 'Step');
s = await step();
check('its first tap is skipped as a ghost cell', s.text.includes('row -2 is above the image') && s.text.includes('skipped'), s.text);
h = await hl();
check('a skipped tap does not highlight the multiply-add', h.length === 3 && !h.some((l) => l.startsWith('Pvalue +=')), JSON.stringify(h));
await press(ST, 'Run to the store');
s = await step();
check('P[0][1] skips 13 of 25 taps (q-skip)', s.text.includes(`${basicTaps(16, 16, 2, 0, 1).skipped} of 25 taps were skipped`) && s.pvalue === s.ref, s.text);
if (out) await page.$eval(ST, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-stepper.png' });
// other radius and block size
await press(ST, 'r = 3');
await press(ST, '8×8 blocks');
s = await step();
check('8×8 blocks keep the same output (0, 1) and r = 3 moves the corner to (−3, −2)', s.outrow === '0' && s.outcol === '1' && s.corner === '-3,-2', JSON.stringify(s));
await press(ST, 'Run to the store');
s = await step();
check('r = 3: Pvalue equals the reference', s.pvalue === s.ref && Number(s.ref) === basicTaps(16, 16, 3, 0, 1, N.data, pyramidFilter(3).data).pvalue, JSON.stringify(s));
await press(ST, 'Restart');
await press(ST, 'Play');
await pause(1300);
s = await step();
check('Play advances by itself', Number(s.at) >= 2, s.at);
await press(ST, 'Pause');

r = await answerNumber('What is the flat input index of its patch', 41);
check('q-map: 41', r.includes('Correct.'), r.slice(-200));
r = await answerNumber('the thread computing P[0][1] skip', 13);
check('q-skip: 13', r.includes('Correct.'), r.slice(-200));

// ---- which warps diverge
const wx = () => page.$eval(WX + ' [data-out="warps"]', (e) => Object.fromEntries(Object.entries(e.dataset).map(([k, v]) => [k, Number(v)])));
const expectW = async (name, w, h, bx, by, rr, ws) => {
  const d = warpDivergence(w, h, bx, by, rr, ws);
  const got = await wx();
  check(`${name}: ${d.warps} warps, ${d.ghost} at the ghost test, ${d.guard} at the guard (model)`, got.warps === d.warps && got.ghost === d.ghost && got.guard === d.guard && got.idle === d.idle, JSON.stringify(got));
  return d;
};
await expectW('default 256×256, 16×16 blocks, r = 2, warps of 32', 256, 256, 16, 16, 2, 32);
const strips = await page.$eval(WX + ' svg[data-strips]', (e) => Number(e.dataset.strips));
check('the overview draws the model\'s merged strips', strips === stripsOf(warpDivergence(256, 256, 16, 16, 2, 32).marked).length, String(strips));
await press(WX, '1D: 4096 outputs, r = 3');
const stored = await page.evaluate(() => fetch('/api/progress').then((x) => x.json()));
if (!stored.answers?.['ch07/7.2/q-warps1d']) {
  const lock = await page.$eval(WX + ' [data-out="warps"]', (e) => ({ locked: e.dataset.locked ?? '', ghost: e.dataset.ghost ?? null, text: e.textContent.replace(/\s+/g, ' ') }));
  const drawn = await page.$eval(WX + ' svg[data-strips]', (e) => Number(e.dataset.strips));
  const boxes = (await page.$$(WX + ' g[data-cls]')).length;
  check('before "One dimension" is answered, the 1D case hides its counts and pictures',
    lock.locked === 'q-warps1d' && lock.ghost === null && lock.text.includes('One dimension') && drawn === 0 && boxes === 0, JSON.stringify({ lock, drawn, boxes }));
}
r = await answerNumber('4096 outputs, r = 3, warps of 32', 2);
check('q-warps1d: 2', r.includes('Correct.'), r.slice(-200));
const d1 = await expectW('the 1D preset (q-warps1d), once answered', 4096, 1, 32, 1, 3, 32);
check('1D: 2 of 128 warps diverge', d1.ghost === 2 && d1.warps === 128);
await setSelect(WX + ' select[aria-label="Image width"]', 1024);
await setSelect(WX + ' select[aria-label="Image height"]', 1024);
await press(WX, '32×32');
await setRange(WX + ' input[aria-label="Radius"]', 2);
const d32 = await expectW('1024×1024, 32×32 blocks, r = 2', 1024, 1024, 32, 32, 2, 32);
const top = await page.$$eval(WX + ' g[data-cls]', (gs) => gs.map((g) => ({ cls: g.dataset.cls, x: Number(g.querySelector('rect').getAttribute('x')) })));
check('with 32×32 blocks the close-up\'s divergent warps are all in the left block', top.some((t) => t.cls === 'ghost') && top.filter((t) => t.cls === 'ghost').every((t) => t.x < 32 * (300 / 64)), JSON.stringify(top.filter((t) => t.cls === 'ghost').slice(0, 3)));
await setRange(WX + ' input[aria-label="Radius"]', 7);
await expectW('r = 7 does not change it with 32×32 blocks', 1024, 1024, 32, 32, 7, 32);
check('the share with 32×32 blocks is 2 of 32 whatever r', d32.ghost === 2048 && (await wx()).ghost === 2048);
await press(WX, '16×16');
await expectW('16×16 blocks, r = 7: the top and bottom split too', 1024, 1024, 16, 16, 7, 32);
await press(WX, '32×32');
await setSelect(WX + ' select[aria-label="Warp width"]', 'w64');
await expectW('64-wide warps bring the top and bottom edges back', 1024, 1024, 32, 32, 7, 64);
await setSelect(WX + ' select[aria-label="Image width"]', 100);
await setSelect(WX + ' select[aria-label="Image height"]', 100);
await setSelect(WX + ' select[aria-label="Warp width"]', 'book');
await expectW('100×100 with 32×32 blocks: the guard splits the right edge', 100, 100, 32, 32, 7, 32);
// The paragraph's qualification: at width 100 the count holds up to r = 4 and grows at r = 5.
await setRange(WX + ' input[aria-label="Radius"]', 4);
const w4 = (await wx()).divergent;
await setRange(WX + ' input[aria-label="Radius"]', 5);
const w5 = (await wx()).divergent;
check('width 100: the divergent count at r = 4 is the one at r = 1, and it grows at r = 5 (model)',
  w4 === warpDivergence(100, 100, 32, 32, 1, 32).divergent && w5 === warpDivergence(100, 100, 32, 32, 5, 32).divergent && w5 > w4, `${w4} ${w5}`);
check('the paragraph after the explorer qualifies the claim',
  norm(await page.$eval('.prose', (e) => e.textContent)).includes('as long as the borders of width r stay inside the first and last column of blocks'));
const reports = await page.evaluate(() => fetch('/api/reports').then((x) => x.json()));
const opts = await page.$$eval(WX + ' select[aria-label="Warp width"] option', (o) => o.map((x) => x.value));
const expectedKeys = ['book', 'w64', ...Object.keys(reports).sort().filter((b) => (b === 'cuda' || b === 'hip') && Number(reports[b].device.warpSize) > 0)];
check('warp widths: the book\'s 32, 64, and only CUDA or HIP devices that reported one', expectedKeys.every((k) => opts.some((o) => o === k || o.startsWith(k))) && opts.length === expectedKeys.length, JSON.stringify(opts));
if (out) await page.$eval(WX, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-warps.png' });

r = await answerChoice('do warps diverge at the ghost test along the top edge', 'Not with 32×32 blocks; yes with 16×16 blocks.');
check('q-blockshape', r.text.includes('Correct.'), r.text.slice(-200));

// ---- the narrated animation
const strip = (x) => x.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const resM = await fetch(BASE + '/content/ch07/narration/manifest.json');
const manifest = resM.ok ? await resM.json() : null;
const scripts = [];
for (let n = 1; n <= 6; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch07/narration/one-thread-per-output-${n}.txt`)).text()));
const counts = scripts.map((x) => x.split(' ').length);
check('six scripts, each 35 to 60 words', counts.every((c) => c >= 35 && c <= 60), counts.join(' '));
check('scripts are written for the ear: no symbols read aloud', scripts.every((x) => !/[<>=×%/_(){}[\]]/.test(x)));
const big = warpDivergence(1024, 1024, 32, 32, 2, 32);
check('the numbers spoken are the model\'s (P 5, 5 from row 3, column 3; 16 of 25 ghost taps; about 6 percent)',
  scripts[1].includes('row 5 and column 5') && scripts[1].includes('row 3, column 3') && scripts[3].includes(`${basicTaps(16, 16, 2, 0, 0).skipped} of its 25 taps`) && scripts[5].includes(`about ${Math.round((big.ghost / big.warps) * 100)} percent`));
await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);
const labels = await page.$$eval(SC + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts are offered', JSON.stringify(labels) === JSON.stringify(['1. Sixteen blocks', '2. One thread', '3. 25 taps', '4. A corner thread', '5. Idle lanes', '6. Zoom out']), labels.join(' | '));
const capBox = () => page.evaluateHandle((SC) => [...document.querySelectorAll(SC + ' label.check')].find((l) => l.textContent.includes('captions')).querySelector('input'), SC);
check('there is a captions switch, off by default', !(await (await capBox()).evaluate((e) => e.checked)));
const caption = () => page.$eval(SC + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim()).catch(() => null);
if (!manifest?.clips?.['one-thread-per-output-1']) {
  check('a part with no audio shows its script even with captions off', (await caption()) === scripts[0], await caption());
  check('with no manifest the player says it runs on estimated timings', (await page.$eval(SC + ' footer', (e) => e.textContent)).includes('No narration manifest found'));
} else {
  check('manifest names the agreed model and voice', manifest.model === 'eleven_v4' && manifest.voice === 'tnSpp4vdxKPjI9w0GnoV');
  check('with audio and captions off, no caption is shown', (await caption()) === null);
}
await (await capBox()).click();
await pause(120);
check('switching captions on shows part 1\'s script', (await caption()) === scripts[0]);
const svgData = () => page.$eval(SC + ' .stage svg', (e) => ({ ...e.dataset, text: [...e.querySelectorAll('text')].map((t) => t.textContent).join(' | ') }));
const go = async (k, at) => {
  await page.evaluate((SC, k) => [...document.querySelectorAll(SC + ' .seg button')][k - 1].click(), SC, k);
  await pause(120);
  await setRange(SC + ' input[type=range]', at);
};
await go(1, 1000);
check('part 1 ends with all 16 blocks lit', (await svgData()).lit === '16');
await go(2, 1000);
let sv = await svgData();
const pm = mapThread({ bx: 1, by: 1, tx: 1, ty: 1, bdx: 4, bdy: 4 }, 2);
check('part 2 draws P[5][5] and the corner (3, 3) from the model', sv.outrow === String(pm.outRow) && sv.outcol === String(pm.outCol) && sv.cornerrow === '3' && sv.cornercol === '3' && sv.text.includes('= (3, 3)'), JSON.stringify(sv));
check('the caption follows the part', (await caption()) === scripts[1]);
await go(3, 500);
sv = await svgData();
const midTap = Number(sv.tap);
check('halfway through part 3 the register holds the running sum after that tap', midTap > 0 && midTap < 25 && Number(sv.pvalue) === run55.taps[midTap - 1].pvalue, JSON.stringify(sv));
await go(3, 1000);
sv = await svgData();
check(`part 3 ends with one store of ${run55.pvalue} to P[85]`, sv.tap === '25' && Number(sv.stored) === run55.pvalue && sv.text.includes(`P[85] = ${run55.pvalue}`), JSON.stringify(sv));
await go(4, 1000);
sv = await svgData();
check('part 4: the corner thread skips 16 of 25 taps', sv.skipped === '16' && sv.total === '16', JSON.stringify(sv));
await go(5, 1000);
sv = await svgData();
const lanes = Array.from({ length: 16 }, (_, t) => ({ row: Math.floor(t / 4), col: t % 4 }));
const act = laneActivity(16, 16, 2, lanes);
const split = act[0].filter((_, k) => act.some((l) => l[k]) && act.some((l) => !l[k])).length;
check(`part 5: ${split} of the 25 taps split the corner block's lanes (model)`, Number(sv.split) === split && Number(sv.splittotal) === split, JSON.stringify(sv));
check('the lanes x taps chart has 16 × 25 cells', (await page.$$eval(SC + ' .stage svg rect', (rs) => rs.length)) >= 400);
await go(6, 1000);
sv = await svgData();
check(`part 6: ${big.ghost} of ${big.warps} warps, in 2 strips, from the model`, Number(sv.ghost) === big.ghost && Number(sv.warps) === big.warps && sv.strips === '2', JSON.stringify(sv));
if (out) await page.screenshot({ path: out + '-scene6.png' });
// it plays by itself, muted, and moves on
await go(1, 0);
await page.evaluate((SC) => { const s = [...document.querySelectorAll(SC + ' label.check')].find((l) => l.textContent.includes('sound')).querySelector('input'); if (s.checked) s.click(); }, SC);
await press(SC, 'Play', false);
await pause(1500);
const pos = await page.$eval(SC + ' input[type=range]', (e) => Number(e.value));
check('Play runs the animation on its clock', pos > 30, String(pos));
await press(SC, 'Pause');
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(SC);
await pause(500);
check('the captions choice is remembered after a reload', await (await capBox()).evaluate((e) => e.checked));
await (await capBox()).click();

// ---- dark theme and a narrow width
await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
if (out) await page.$eval(WX, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-dark.png' });
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
const over = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', over <= 1, `overflow ${over}px`);
if (out) await page.$eval(ST, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-narrow.png' });

const expected404 = (u) => /\/content\/ch07\/narration\/(manifest\.json|one-thread-per-output-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
// The listings' statuses: LabKit's words from this machine's reports, and "failed on this machine" when a run failed.
await checkLabStatuses(check, 'ch07/7.2', [
  { label: 'CUDA', lab: 'ch07/lab1_constconv', file: 'kernel.cu' },
  { label: 'Metal kernel', lab: 'ch07/lab1_constconv', file: 'kernel.metal' },
  { label: 'Metal host', lab: 'ch07/lab1_constconv', file: 'launch.mm' },
], { reveal: ['Show the corrected kernel now'] });

await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
