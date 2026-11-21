// Browser check for section 9.1: the "Paint and bin" explorer agrees with the model
// (src/sim/histogram.ts, imported here directly), the sequential loop is the checked
// snippet, and the questions answer.
// Usage: APP_URL=http://localhost:5173 node e2e/ch09-s91.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { binOf, histogram, hottest, preset } from '../src/sim/histogram.ts';

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
const B = 'section.bench[aria-label="Paint and bin"]';
const press = async (group, label) => {
  const ok = await page.evaluate((B, group, label) => {
    const g = document.querySelector(`${B} [role=group][aria-label="${group}"]`);
    const b = [...g.querySelectorAll('button')].find((x) => x.textContent.trim().startsWith(label));
    if (!b) return false;
    b.click();
    return true;
  }, B, group, label);
  await pause(100);
  return ok;
};
const bars = () => page.$$eval(B + ' rect[data-bin]', (rs) => rs.map((r) => Number(r.dataset.count)));
const readout = () => page.$eval(B + ' [data-part="readout"]', (e) => e.textContent.replace(/\s+/g, ' '));
const setPaint = async (v) => {
  await page.$eval(B + ' input[aria-label="Paint value"]', (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(80);
};
const pixel = (i) => page.$(`${B} [data-part="image"] button[data-index="${i}"]`);

await goto(page, 'ch09/9.1');
await page.waitForSelector(B);
await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));

// ---- starting state: our sky preset, intervals of 64
const sky = preset('sky').values;
let got = await bars();
check('starts on the sky preset in 4 bins, matching the model', JSON.stringify(got) === JSON.stringify(histogram(sky, 64)), JSON.stringify(got));
check('the bars sum to 64', (await page.$eval(B + ' [data-sum]', (e) => e.textContent)) === '64');
check('the busiest bin is reported as a count of colliding threads', (await page.$eval(B + ' [data-part="hot"]', (e) => e.textContent)).includes(`${hottest(histogram(sky, 64)).count} threads`));

// ---- point at a pixel: flat index and bin
await (await pixel(42)).hover();
await pause(100);
let r = await readout();
check('hovering row 5, column 2 shows flat index 5 × 8 + 2 = 42', r.includes('row 5, column 2') && r.includes('= 42'), r);
check('and its bin, value / 64', r.includes(`Value ${sky[42]}`) && r.includes(`bin ${binOf(sky[42], 64)}`), r);

// ---- paint: set pixel 63 (value 40, bin 0) to 255 (bin 3)
await setPaint(255);
await (await pixel(63)).click();
await pause(100);
const painted = sky.slice();
painted[63] = 255;
got = await bars();
check('painting pixel 63 white moves one count from bin 0 to bin 3', JSON.stringify(got) === JSON.stringify(histogram(painted, 64)) && got[0] === 14 && got[3] === 36, JSON.stringify(got));
r = await readout();
check('the readout follows the painted pixel: index 63, bin 3', r.includes('= 63') && r.includes('bin 3'), r);
// Drag-paint three more pixels black.
await setPaint(0);
const box = async (i) => (await pixel(i)).boundingBox();
const b0 = await box(0);
const b2 = await box(2);
await page.mouse.move(b0.x + b0.width / 2, b0.y + b0.height / 2);
await page.mouse.down();
await page.mouse.move(b2.x + b2.width / 2, b2.y + b2.height / 2, { steps: 8 });
await page.mouse.up();
await pause(120);
for (const i of [0, 1, 2]) painted[i] = 0;
got = await bars();
check('dragging across pixels 0 to 2 paints all three', JSON.stringify(got) === JSON.stringify(histogram(painted, 64)), JSON.stringify(got));

// ---- interval widths
for (const [w, label, n] of [[32, '32', 8], [16, '16', 16], [1, '1 ', 256]]) {
  await press('Interval width', label);
  got = await bars();
  check(`width ${w}: ${n} bars, equal to histogram(image, ${w})`, got.length === n && JSON.stringify(got) === JSON.stringify(histogram(painted, w)), JSON.stringify(got.slice(0, 16)));
  check(`width ${w}: the bars still sum to 64`, (await page.$eval(B + ' [data-sum]', (e) => e.textContent)) === '64');
}
await (await pixel(10)).hover();
await pause(80);
r = await readout();
check('at one bin per value, the bin is the pixel value itself', r.includes(`bin ${painted[10]} = ⌊${painted[10]} / 1⌋`), r);
const hot1 = hottest(histogram(painted, 1));
check('at 256 bins the sky\'s white pixels pile into value 255', hot1.bin === 255 && (await page.$eval(B + ' [data-part="hot"]', (e) => e.dataset.hotCount)) === String(hot1.count), String(hot1.count));
if (out) await page.screenshot({ path: out + '-paint.png' });

// ---- presets
await press('Image', 'flat grey');
got = await bars();
check('flat grey: one bar of 64 at value 128', got[128] === 64 && got.reduce((s, c) => s + c, 0) === 64);
await press('Interval width', '64');
await press('Image', 'night scene');
check('night scene at width 64 matches the model', JSON.stringify(await bars()) === JSON.stringify(histogram(preset('night').values, 64)));
await press('Image', 'noise');
check('noise at width 64 matches the model', JSON.stringify(await bars()) === JSON.stringify(histogram(preset('noise').values, 64)));

// ---- the listing is the checked snippet
await page.waitForFunction(() => [...document.querySelectorAll('.code pre')].some((p) => p.textContent.includes('histogram_sequential')));
const code = await page.$$eval('.code pre', (ps) => ps.map((p) => p.textContent).join('\n'));
check('the sequential loop is loaded from labs/ch09/snippets', code.includes('const unsigned char* image') && code.includes('++bins[b];'));

// ---- questions
const question = (text) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt').textContent.includes(t)), text);
const qText = (q) => q.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
async function choose(text, option) {
  const q = await question(text);
  await q.evaluate((e, o) => [...e.querySelectorAll('.opts label')].find((l) => l.textContent.includes(o)).querySelector('input').click(), option);
  await (await q.$('button.primary')).click();
  await pause(120);
  return qText(q);
}
async function numeric(text, value) {
  const q = await question(text);
  await (await q.$('input[type=number]')).evaluate((el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return qText(q);
}
let t = await numeric('how many threads update bin 255', 40);
check('contention: 40 is wrong and reveals 400000', t.includes('Not quite. The answer is 400000'), t.slice(0, 200));
t = await numeric('how many threads update bin 255', 400000);
check('contention: 400,000 accepted', t.includes('Correct.') && t.includes('section 9.3'), t.slice(0, 200));
t = await choose('breaks the owner-computes rule', 'softmax');
check('owner: the softmax row is wrong, with the reduction pointer', t.includes('Not quite.') && t.includes('chapter 10'), t.slice(0, 300));
t = await choose('breaks the owner-computes rule', 'token id');
check('owner: counting token ids is accepted', t.includes('Correct.'));
t = await choose('limited by memory on a CPU', 'bins sit in L1');
check('cpu: accepted with the book\'s argument', t.includes('Correct.') && t.includes('§9.1, p. 203'));
const cpuOrder = await page.evaluate(() => {
  const kids = [...document.querySelector('.prose').children];
  return {
    q: kids.findIndex((k) => k.classList.contains('q') && k.textContent.includes('limited by memory on a CPU')),
    p: kids.findIndex((k) => k.tagName === 'P' && k.textContent.includes('On a CPU this loop is limited by memory')),
  };
});
check('q-cpu comes before the paragraph that answers it', cpuOrder.q >= 0 && cpuOrder.q < cpuOrder.p, JSON.stringify(cpuOrder));

const body = await page.$eval('.prose', (e) => e.textContent.replace(/\s+/g, ' '));
check('the bincount bridge and the MPSImageHistogram note are there', body.includes('torch.bincount') && (await page.$$eval('.note.today', (n) => n.some((x) => x.textContent.includes('MPSImageHistogram') && x.textContent.includes('C6')))));
check('the explorer states what it simplifies', await page.$$eval('.note.caveat', (n) => n.some((x) => x.textContent.includes('8-bit grayscale only'))));

// ---- dark theme and narrow screen
const dark = await open({ width: 400, height: 900, dark: true });
await dark.page.setRequestInterception(true);
dark.page.on('request', (r) => (r.method() === 'PUT' ? r.respond({ status: 200, body: '{}' }) : r.continue()));
await goto(dark.page, 'ch09/9.1');
await dark.page.waitForSelector(B);
const wide = await dark.page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
check('no sideways scrolling at 400 px', wide <= 1, String(wide));
if (out) {
  await dark.page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));
  await dark.page.screenshot({ path: out + '-narrow-dark.png', fullPage: false });
}
check('no errors in the narrow dark view', dark.errors.length === 0, dark.errors.join(' | '));
await dark.browser.close();

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
