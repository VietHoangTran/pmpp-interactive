// Browser check for section 3.3: the blur kernel listing, the loop stepper, the boundary
// explorer, the defect bench, the narrated scene, the three-orders bench, the "name the
// defect" challenge and the questions. Expected numbers are the ones pinned by
// src/sim/blur.test.ts.
// Usage: node e2e/ch03-s33.mjs [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto, BASE } from './browser.mjs';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);
const notFound = [];
page.on('response', (r) => { if (r.status() === 404) notFound.push(new URL(r.url()).pathname); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent);
const attr = (sel, name) => page.$eval(sel, (e, name) => e.getAttribute(name), name);
const count = (sel) => page.$$eval(sel, (e) => e.length);
const clickText = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const root = [...document.querySelectorAll('section.bench')].find((b) => b.getAttribute('aria-label') === scope);
    const btn = [...root.querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(60);
  return done;
};
// A select or a range input, set the way a person would change it.
const setValue = async (sel, value) => {
  await page.$eval(sel, (el, value) => {
    const proto = el.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, String(value));
    el.dispatchEvent(new Event(el.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true }));
  }, value);
  await pause(80);
};
// A checkbox found by a piece of its label.
const setBox = async (scope, label, on) => {
  await page.evaluate((sel, label, on) => {
    const box = [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes(label)).querySelector('input');
    if (box.checked !== on) box.click();
  }, scope, label, on);
  await pause(100);
};
const centre = async (sel) => {
  const b = await (await page.$(sel)).boundingBox();
  return [b.x + b.width / 2, b.y + b.height / 2];
};
const shot = async (name, sel) => {
  if (!out) return;
  if (sel) await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};

await goto(page, 'ch03/3.3');
await page.waitForSelector(bench('Step through one thread'));

// ---------------------------------------------------------------- the listing
const line = (n) => text(`figure.code:not(.bench figure) pre .ln[data-n="${n}"]`);
check('listing shows the #define as line 0', (await line(0)).includes('#define BLUR_SIZE 1'));
check("line 05 is the outer guard, as in the book's numbering", (await line(5)).includes('if (col < w && row < h)'));
check('line 08 is the blank line the book skips', (await line(8)).trim() === '');
check('line 15 is the inner test', (await line(15)).includes('curRow >= 0 && curRow < h && curCol >= 0 && curCol < w'));
check('line 22 is the store with the cast', (await line(22)).includes('(unsigned char)((float)pixVal/pixels)'));
const marked = await page.$$eval('figure.code pre .ln.hl', (l) => l.slice(0, 3).map((x) => x.dataset.n));
check('the two guards and the store are marked', marked.join() === '5,15,22', marked.join());

// ---------------------------------------------------------------- predict first
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('how many pass the test on line 15')));
await (await pred.$('input[type=number]')).type('4');
await (await pred.$('button.primary')).click();
await pause(150);
const revealed = await pred.evaluate((e) => e.textContent);
check('prediction reveals the nine pairs', revealed.includes('Yes, 4.') && revealed.includes('(-1, -1)') && revealed.includes('(1, 1)'), revealed.slice(0, 200));

// ---------------------------------------------------------------- loop stepper
const S = 'Step through one thread';
const narr = () => text(bench(S) + ' .narrate');
const clause = (c) => attr(`${bench(S)} tr[data-clause="${c}"]`, 'data-value');
check('stepper starts at step 0 of 10', (await text(bench(S))).includes('step 0 of 10'));
await clickText(S, 'Step');
let n = await narr();
check('step 1: (-1, -1) fails the first clause and the rest is not evaluated',
  n.includes('curRow >= 0 is false') && n.includes('position (-1, -1) is skipped') && n.includes('3 clauses after it are never evaluated'), n);
check('clause table shows the short circuit', (await clause('rowLo')) === 'false' && (await clause('rowHi')) === 'null' && (await clause('colHi')) === 'null');
for (let i = 0; i < 3; i++) await clickText(S, 'Step');
n = await narr();
check('step 4: (0, -1) fails on the column', n.includes('curCol >= 0 is false') && n.includes('position (0, -1) is skipped'), n);
check('its row clauses were true', (await clause('rowLo')) === 'true' && (await clause('rowHi')) === 'true' && (await clause('colLo')) === 'false');
await clickText(S, 'Step');
n = await narr();
check('step 5: (0, 0) is read with its flat index', n.includes('in[0·7 + 0] = in[0] = 1') && n.includes('pixels = 1'), n);
const hl = await page.$$eval(bench(S) + ' .code .ln.hl', (l) => l.map((x) => x.dataset.n).join());
check('lines 12, 13, 15, 16, 17 are highlighted for a kept position', hl === '12,13,15,16,17', hl);
await shot('stepper', bench(S));
await clickText(S, 'Run to the end');
n = await narr();
check('corner (0,0): 4 kept, 5 skipped, out[0] = 4', n.includes('4 of 9 positions were kept and 5 skipped') && n.includes('(float)16/4') && (await text(bench(S))).includes('out[0] = 4'), n);
check('registers end at pixVal 16, pixels 4', (await text(bench(S) + ' td[data-reg="pixVal"]')) === '16' && (await text(bench(S) + ' td[data-reg="pixels"]')) === '4');
check('five positions are hatched', (await count(bench(S) + ' svg rect[fill^="url("]')) === 5);
const kept = [];
for (let k = 0; k < 6; k++) {
  await setValue(bench(S) + ' select', k);
  await clickText(S, 'Run to the end');
  kept.push(Number(await text(bench(S) + ' td[data-reg="pixels"]')));
  if (k === 4) {
    n = await narr();
    check('bottom-right corner: 19/4 = 4.750 is stored as 4', n.includes('(float)19/4') && n.includes('4.750') && n.includes('The cast drops the fraction') && (await text(bench(S))).includes('out[41] = 4'), n);
  }
}
check('the six cases keep 4, 6, 6, 6, 4 and 9 values', kept.join() === '4,6,6,6,4,9', kept.join());

// ---------------------------------------------------------------- boundary explorer
const E = 'Boundary explorer';
const k = (name) => text(`${bench(E)} [data-k="${name}"]`);
const cell = (r, c) => `${bench(E)} svg g[data-rc="${r},${c}"]`;
await page.$eval(bench(E), (e) => e.scrollIntoView({ block: 'start' }));
await pause(200);
check('explorer starts on an interior pixel: 9 of 9', (await k('pixel')) === '(2, 3)' && (await k('count')) === '9 of 9');
let [x, y] = await centre(cell(0, 0));
await page.mouse.click(x, y);
await pause(120);
check('clicking the corner: 4 of 9, as 2 rows × 2 columns', (await k('count')) === '4 of 9' && (await k('product')) === '2 rows × 2 columns = 4', await k('product'));
check('the corner has five hatched positions', (await count(bench(E) + ' svg rect[fill^="url("]')) === 5);
const [x3, y3] = await centre(cell(0, 3));
await page.mouse.move(x, y);
await page.mouse.down();
await page.mouse.move((x + x3) / 2, y, { steps: 4 });
await page.mouse.move(x3, y3, { steps: 4 });
await page.mouse.up();
await pause(120);
check('dragging along the top edge: pixel (0, 3) sums 6 of 9', (await k('pixel')) === '(0, 3)' && (await k('count')) === '6 of 9', await k('pixel'));
await page.focus(bench(E) + ' [data-explorer-grid]');
await page.keyboard.press('ArrowDown');
await page.keyboard.press('ArrowDown');
await page.keyboard.press('ArrowRight');
await pause(120);
check('arrow keys move the pixel: (2, 4) sums 9 of 9', (await k('pixel')) === '(2, 4)' && (await k('count')) === '9 of 9', await k('pixel'));
check('sum and stored byte come from the model', (await k('sum')) === '48' && (await k('byte')) === '5', `${await k('sum')} ${await k('byte')}`);
check('whole-picture summary for 8 by 6, radius 1', (await k('summary')).includes('24 threads sum 9, 20 threads sum 6, 4 threads sum 4') && (await k('summary')).includes('352 reads in all; 432'), await k('summary'));
await setValue(bench(E) + ' input[aria-label="BLUR_SIZE"]', 3);
[x, y] = await centre(cell(0, 0));
await page.mouse.click(x, y);
await pause(120);
check('BLUR_SIZE 3 at the corner: 16 of 49', (await k('count')) === '16 of 49' && (await k('rows')) === '0 to 3: 4 of 7', await k('count'));
await setBox(bench(E), 'count map', true);
check('count map prints each thread\'s count', (await text(cell(0, 0))) === '16' && (await text(cell(2, 3))) === '42', `${await text(cell(0, 0))} ${await text(cell(2, 3))}`);
await shot('explorer', bench(E));
await setValue(bench(E) + ' input[aria-label="BLUR_SIZE"]', 1);
await setValue(bench(E) + ' input[aria-label="Width"]', 3);
await setValue(bench(E) + ' input[aria-label="Height"]', 3);
check('3 by 3 picture: one thread sums 9, four sum 6, four sum 4', (await k('summary')).includes('1 thread sums 9, 4 threads sum 6, 4 threads sum 4'), await k('summary'));
check('the selected pixel is clamped into the smaller picture', (await k('pixel')) === '(0, 0)', await k('pixel'));

// ---------------------------------------------------------------- break the kernel
const B = 'Break the kernel';
const verdict = () => attr(bench(B) + ' [data-verdict]', 'data-verdict');
const access = async (cls) => ((await page.$(`${bench(B)} li[data-access="${cls}"]`)) ? Number(await attr(`${bench(B)} li[data-access="${cls}"]`, 'data-n')) : 0);
const wrongCells = () => count(bench(B) + ' svg rect[stroke="var(--fail)"]');
await page.$eval(bench(B), (e) => e.scrollIntoView({ block: 'start' }));
check('correct kernel: out matches the reference under every order', (await verdict()) === 'correct' && (await attr(bench(B) + ' [data-orders]', 'data-orders')) === 'same');
await setBox(bench(B), 'curCol < w', false);
let t = await text(bench(B));
check('without curCol < w: 7 of 42 pixels wrong, all marked', (await verdict()) === 'wrong' && t.includes('7 of 42 pixels wrong') && (await wrongCells()) === 7, String(await wrongCells()));
check('17 reads wrapped into another row, 2 left the buffer', (await access('read-wrapped')) === 17 && (await access('read-outside')) === 2);
check('the example names the wrapped read', t.includes('thread (0, 5) read in[6]') && t.includes('got pixel (1, 0)'), t.slice(-600));
await shot('break-col', bench(B));
await setValue(bench(B) + ' select[aria-label="Picture"]', 'grey');
check('a uniform picture hides all but 2 of them', (await text(bench(B))).includes('2 of 42 pixels wrong') && (await access('read-wrapped')) === 17);
await setValue(bench(B) + ' select[aria-label="Picture"]', 'texture');
await clickText(B, 'Restore the correct kernel');
await setBox(bench(B), 'line 22', false);
t = await text(bench(B));
check('dividing by the area: 22 border pixels wrong and no bad access', t.includes('22 of 42 pixels wrong') && t.includes('only the arithmetic is wrong') && (await count(bench(B) + ' li[data-access]')) === 0);
await clickText(B, 'Restore the correct kernel');
await setBox(bench(B), 'line 05', false);
check('without the outer guard: 12 stores on other pixels, 10 outside, 8 threads divide 0 by 0',
  (await access('write-wrapped')) === 12 && (await access('write-outside')) === 10 && (await access('zero-div')) === 8 && (await text(bench(B))).includes('10 of 42 pixels wrong'));
check('undefined pixels are shown as question marks', (await page.$$eval(bench(B) + ' svg text', (e) => e.filter((x) => x.textContent === '?').length)) === 5);
await shot('break-outer', bench(B));
await clickText(B, 'Restore the correct kernel');
await setBox(bench(B), 'a separate out buffer', false);
t = await text(bench(B));
check('in place: 34 of 42 wrong with every access inside the buffer', t.includes('34 of 42 pixels wrong') && (await access('overwritten')) === 131 && (await access('read-wrapped')) === 0 && (await access('read-outside')) === 0);
const byOrder = await page.$$eval(bench(B) + ' [data-order]', (e) => e.map((x) => x.dataset.wrong).join());
check('in place the result changes with the thread order', (await attr(bench(B) + ' [data-orders]', 'data-orders')) === 'differ' && byOrder === '34,36,30', byOrder);
await shot('break-inplace', bench(B));
await setValue(bench(B) + ' select[aria-label="Thread order"]', 'reversed');
check('choosing the reversed order shows its picture', (await text(bench(B))).includes('36 of 42 pixels wrong'));
await clickText(B, 'Restore the correct kernel');
check('restoring gives the correct kernel again', (await verdict()) === 'correct' && !(await clickText(B, 'Restore the correct kernel')));

// ---------------------------------------------------------------- narrated scene
const SC = 'section.scene';
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').trim();
const script = async (i) => strip(await (await fetch(`${BASE}/content/ch03/narration/in-place-blur-${i}.txt`)).text());
const part = async (label) => {
  await page.evaluate((sel, label) => [...document.querySelectorAll(sel + ' .seg button')].find((b) => b.textContent.includes(label)).click(), SC, label);
  await pause(120);
};
const seek = (v) => setValue(SC + ' input[type=range]', v);
const stage = () => text(SC + ' .stage svg');
await page.$eval(SC, (e) => e.scrollIntoView({ block: 'start' }));
await pause(300);
check('scene offers five parts and a captions switch', (await count(SC + ' .seg button')) === 5 && (await text(SC)).includes('captions'));
const hasManifest = fs.existsSync(path.join(repo, 'content/ch03/narration/manifest.json'));
if (!(await page.$(SC + ' .caption'))) await setBox(SC, 'captions', true);
check("the caption is part 1's script", (await text(SC + ' .caption')).trim() === (await script(1)), await text(SC + ' .caption'));
for (let i = 1; i <= 5; i++) {
  const words = (await script(i)).split(/\s+/).length;
  check(`script ${i} is 35 to 60 words`, words >= 35 && words <= 60, String(words));
}
await seek(1000);
check('part 1 ends with every pixel matching the reference', (await stage()).includes('36 of 36 match the reference'), await stage());
await part('2. One buffer');
await seek(1000);
let sv = await stage();
check('part 2: thread (1, 1) stores 28 and its neighbour reads it', sv.includes('thread (1, 1) stores 28') && sv.includes('it reads 28 where in held 0'), sv.slice(-200));
check("part 2's caption is its script", (await text(SC + ' .caption')).trim() === (await script(2)));
await part('3. In order');
await seek(1000);
sv = await stage();
check('part 3: 24 of 36 pixels wrong, nothing outside the buffer', sv.includes('24 of 36 pixels are wrong') && sv.includes('accesses outside the buffer: 0'), sv.slice(-200));
check('part 3 marks the wrong pixels', (await count(SC + ' .stage svg rect[stroke="var(--fail)"]')) === 24);
await shot('scene-3', SC);
await part('4. Reversed');
await seek(1000);
sv = await stage();
check('part 4: the reversed run differs from the in-order run in 31 pixels', sv.includes('the two runs differ in 31 pixels') && sv.includes('wrong so far: 19'), sv.slice(-200));
await shot('scene-4', SC);
await part('5. The rule');
await seek(1000);
check('part 5 states the rule', (await stage()).includes('Each thread writes only its own element,') && (await stage()).includes('and reads only data that no thread writes.'));
await part('1. Two buffers');
await clickText('Why the blur writes somewhere else', 'Play');
await pause(2200);
const pos = await page.$eval(SC + ' input[type=range]', (e) => Number(e.value));
check('pressing Play moves the animation', pos > 30, String(pos));
check('the drawing counts threads as they finish', /\d+ of 36 threads done/.test(await stage()));
await clickText('Why the blur writes somewhere else', 'Pause');

// ---------------------------------------------------------------- one buffer, three orders
const T = 'One buffer, three thread orders';
const orders = () => page.$$eval(bench(T) + ' [data-order]', (e) => e.map((x) => x.dataset.wrong).join());
check('in place on the spot picture: 24, 19 and 17 pixels wrong', (await orders()) === '24,19,17' && (await attr(bench(T) + ' [data-same]', 'data-same')) === 'no', await orders());
await shot('three-orders', bench(T));
await setBox(bench(T), 'is the same buffer', false);
check('with a separate out buffer all three orders match the reference', (await orders()) === '0,0,0' && (await attr(bench(T) + ' [data-same]', 'data-same')) === 'yes', await orders());
await setBox(bench(T), 'is the same buffer', true);
await setValue(bench(T) + ' select[aria-label="Picture"]', 'texture');
check('on the textured picture the three orders differ as well', (await attr(bench(T) + ' [data-same]', 'data-same')) === 'no');

// ---------------------------------------------------------------- name the defect
const N = 'Name the defect';
const pickDefects = async (list) => {
  for (let i = 0; i < 3; i++) await setValue(`${bench(N)} select[aria-label="Defect behind picture ${i + 1}"]`, list[i]);
};
await page.$eval(bench(N), (e) => e.scrollIntoView({ block: 'start' }));
check('Check is disabled until all three are chosen', !(await clickText(N, 'Check')));
await pickDefects(['inPlace', 'rowLo', 'colHi']); // challenge(1), pinned in blur.test.ts
await clickText(N, 'Check');
check('the first set is named correctly', (await attr(bench(N) + ' [data-score]', 'data-score')) === '3' && (await text(bench(N))).includes('3 of 3 named correctly'));
await shot('challenge', bench(N));
await clickText(N, 'New set');
await pickDefects(['area', 'area', 'area']); // challenge(2) is rowLo, colHi, rowHi
await clickText(N, 'Check');
t = await text(bench(N));
check('wrong answers are corrected with the reason', (await attr(bench(N) + ' [data-score]', 'data-score')) === '0' && t.includes('it was: curRow >= 0 is missing') && t.includes('Only the top row is wrong'), t.slice(-300));
check('both benches give the picture rows first: 7 by 6, 7 rows of 6',
  (await text(bench(N) + ' > footer')).includes('7 by 6 picture (7 rows of 6)') && (await text(bench('Break the kernel') + ' > footer')).includes('7 by 6 picture (7 rows of 6)'),
  await text(bench(N) + ' > footer'));
// A missing outer guard: the explanation names the rows the model marks wrong, row 4 spared.
{
  const { challenge, blurImage, textured, withDefect, wrongSpan } = await import('../src/sim/blur.ts');
  let seed = 2;
  while (!challenge(seed).includes('outer')) seed++;
  for (let s = 2; s < seed; s++) await clickText(N, 'New set');
  const set = challenge(seed);
  await pickDefects(set.map((d) => (d === 'outer' ? 'area' : d)));
  await clickText(N, 'Check');
  t = await text(bench(N));
  const span = wrongSpan(blurImage(textured(6, 7), withDefect('outer')).wrong, 6);
  check(`the outer-guard explanation lists the wrong rows from the model (set ${seed})`,
    span.rows.join() === '1,2,3,5,6' && span.spared.join() === '4' &&
    t.includes('Columns 0 and 1 are wrong in rows 1, 2, 3, 5 and 6.') && t.includes('Row 4 survives only because its own block runs after the block that wrote over it.') && !t.includes('from row 1 down'),
    t.slice(-500));
}

// ---------------------------------------------------------------- questions
const q = (needle) => page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), needle);
let h = await q('BLUR_SIZE = 3 on a 100 by 100 picture');
await (await h.$('input[type=number]')).type('16');
await (await h.$('button.primary')).click();
await pause(120);
check('numeric question accepts 16', (await h.evaluate((e) => e.textContent)).includes('Correct.'));
h = await q('8 columns wide and 6 rows high');
await (await h.$('input[type=number]')).type('9');
await (await h.$('button.primary')).click();
await pause(120);
check('numeric question rejects 9 and gives 15', (await h.evaluate((e) => e.textContent)).includes('The answer is 15'));
const choose = async (needle, option) => {
  const hq = await q(needle);
  await hq.evaluate((e, option) => [...e.querySelectorAll('.opts label')].find((l) => l.textContent.includes(option)).querySelector('input').click(), option);
  await (await hq.$('button.primary')).click();
  await pause(120);
  return hq.evaluate((e) => e.textContent);
};
check('divisor question: border darker, corners darkest', (await choose('uniformly grey picture', 'the border is darker')).includes('Correct.'));
check('outer-guard question: the inner test does not cover the store', (await choose('The test on line 05', 'Nothing. The inner test')).includes('Not quite.'));
check('in-place question: the order decides', (await choose('same buffer as', 'depends on the order')).includes('Correct.'));
check('bridge question: masked mean pooling', (await choose('same shape as lines 10 to 22', 'mean pooling over the unmasked tokens')).includes('Correct.'));

// ---------------------------------------------------------------- the lab's kernels, when the lab is there
const labThere = ['kernel.cu', 'kernel.metal', 'launch.mm'].every((f) => fs.existsSync(path.join(repo, 'labs/ch03/lab2_blur/solution', f)));
const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent));
if (labThere) {
  check('lab 3.2 solution is shown with CUDA, Metal kernel and Metal host tabs', ['CUDA (lab 3.2)', 'Metal kernel', 'Metal host'].every((x) => tabs.includes(x)), tabs.join('|'));
  await page.evaluate(() => [...document.querySelectorAll('.code .tabs button')].find((b) => b.textContent === 'Metal kernel').click());
  await pause(100);
  const metal = await page.evaluate(() => [...document.querySelectorAll('figure.code')].find((f) => f.textContent.includes('Metal host')).querySelector('pre').textContent);
  check('the Metal tab is the lab file', metal.includes('thread_position_in_grid') && metal.includes('curRow >= 0 && curRow < h && curCol >= 0 && curCol < w'));
  // The status beside the tabs is the fixed vocabulary, computed from this machine's lab runs.
  const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
  const passed = (b) => {
    const rs = (reports[b]?.results ?? []).filter((x) => x.lab === 'ch03/lab2_blur' && x.variant === 'solution' && x.tests.length);
    return rs.length > 0 && rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass));
  };
  // The labs page's wording (LabKit's solutionTabStatus).
  const st = (b) => {
    const rs = (reports[b]?.results ?? []).filter((x) => x.lab === 'ch03/lab2_blur' && x.variant === 'solution' && x.tests.length);
    return !rs.length ? 'none' : passed(b) ? 'pass' : 'fail';
  };
  const NAME = { cuda: 'CUDA', hip: 'HIP', emu: 'emulator' };
  const WHERE = { cuda: 'CUDA', hip: 'HIP', emu: 'the emulator' };
  const and = (xs) => (xs.length < 3 ? xs.join(' and ') : xs.slice(0, -1).join(', ') + ' and ' + xs.at(-1));
  const fam = ['cuda', 'hip', 'emu'];
  const ok = fam.filter((b) => st(b) === 'pass').map((b) => (b === 'emu' ? 'emulator-checked' : `verified on ${NAME[b]}`));
  const bad = fam.filter((b) => st(b) === 'fail').map((b) => WHERE[b]);
  const none = ['cuda', 'hip'].filter((b) => st(b) === 'none').map((b) => NAME[b]);
  const wantMetal = st('metal') === 'pass' ? 'verified on Metal' : `unverified: ${st('metal') === 'fail' ? 'failed on this machine' : 'no run on this machine'}`;
  const wantCuda = !ok.length && !bad.length ? 'unverified: no run on this machine'
    : [ok.join(', '), bad.length ? `unverified on ${and(bad)}: failed on this machine` : '', none.length ? `unverified on ${none.join(' and ')}` : ''].filter(Boolean).join('; ');
  const fig = '[...document.querySelectorAll("figure.code")].find((f) => f.textContent.includes("Metal host"))';
  const statusMetal = await page.evaluate(`${fig}.querySelector('.tabs .status').textContent`);
  await page.evaluate(`[...${fig}.querySelectorAll('.tabs button')].find((b) => b.textContent === 'CUDA (lab 3.2)').click()`);
  await pause(100);
  const statusCuda = await page.evaluate(`${fig}.querySelector('.tabs .status').textContent`);
  check(`the lab 3.2 tabs carry statuses computed from the reports ("${wantCuda}" / "${wantMetal}")`,
    statusMetal === wantMetal && statusCuda === wantCuda, `${statusCuda} / ${statusMetal}`);
} else check('lab 3.2 files did not load, and the page does not show a broken listing', !tabs.includes('CUDA (lab 3.2)'));

// ---------------------------------------------------------------- narrow screen
await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 1 });
await pause(400);
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('narrow-explorer', bench(E));

// Until the narration audio is generated there is no manifest, and asking for it is a 404.
const expected404 = (p) => (!hasManifest && p === '/content/ch03/narration/manifest.json') || (!labThere && p === '/api/lab-file');
const stray = notFound.filter((p) => !expected404(p));
const other = errors.filter((e) => !e.includes('status of 404'));
check('no console or page errors', other.length === 0 && stray.length === 0, [...other, ...stray].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
