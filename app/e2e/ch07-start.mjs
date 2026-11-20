// Browser check for chapter 7, "Before you start": the map of the four kernels and its
// links, all five prerequisite questions (right and wrong answers), and the page at a
// narrow width.
// Usage: node e2e/ch07-start.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';

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
const norm = (s) => s.replace(/\s+/g, ' ').trim();

/** The question box whose prompt contains `text`. */
const question = (text) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.replace(/\s+/g, ' ').includes(t)), text);
async function answerChoice(text, optionStart) {
  const q = await question(text);
  const ok = await q.evaluate((e, o) => {
    const label = [...e.querySelectorAll('.opts label')].find((l) => l.textContent.replace(/\s+/g, ' ').trim().startsWith(o));
    if (!label) return false;
    label.querySelector('input').click();
    return true;
  }, optionStart);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return { found: ok, text: norm(await q.evaluate((e) => e.textContent)) };
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

await goto(page, 'ch07/start');
await page.waitForSelector('table[data-map]');

// ---- the map of the four kernels
const rows = await page.$$eval('table[data-map] tbody tr', (trs) =>
  trs.map((t) => ({ section: t.dataset.section, href: t.querySelector('a')?.getAttribute('href'), text: t.textContent.replace(/\s+/g, ' ') })),
);
check('the map has four kernels', rows.length === 4, JSON.stringify(rows.map((r) => r.section)));
check('each kernel links to its section (7.2, 7.4, 7.5, 7.6)',
  JSON.stringify(rows.map((r) => r.href)) === JSON.stringify(['#/ch07/7.2', '#/ch07/7.4', '#/ch07/7.5', '#/ch07/7.6']), JSON.stringify(rows.map((r) => r.href)));
check('the book figures are named (Figs. 7.7, 7.9, 7.12, 7.15)', ['7.7', '7.9', '7.12', '7.15'].every((f, i) => rows[i].text.includes('Fig. ' + f)));
check('FLOP/B as the book computes it: 0.25, 0.5, the §7.5 formula with 5.42, none for Fig. 7.15',
  rows[0].text.includes('0.25') && rows[1].text.includes('0.5') && rows[2].text.includes('5.42') && rows[3].text.includes('no figure'));
const body = norm(await page.$eval('.prose', (e) => e.textContent));
check('the bridge names PyTorch cross-correlation and im2col in llama.cpp, and points to section 7.5 for the causal convolution',
  body.includes('valid 2D cross-correlation operator') && body.includes('ggml_im2col') && body.includes('section 7.5 follows into llama.cpp') && !body.includes('ggml_ssm_conv'));
check('the Gaussian is called a smoothing filter, not a sharpener', body.includes('This course calls it a smoothing filter'));
check('links to the labs page', (await page.$$eval('a', (as) => as.map((a) => a.getAttribute('href')))).includes('#/ch07/labs'));
if (out) await page.screenshot({ path: out + '-top.png' });

// ---- the five prerequisite questions
let r = await answerChoice('blur kernel with BLUR_SIZE = 1', 'Yes: the blur also counts');
check('q-blur-corner: the tempting answer is marked wrong and explains the divisor', r.found && r.text.includes('Not quite.') && r.text.includes('divide by different numbers'), r.text.slice(0, 200));
r = await answerChoice('blur kernel with BLUR_SIZE = 1', 'No: the blur divides');
check('q-blur-corner: the right answer is accepted', r.found && r.text.includes('Correct.') && r.text.includes('ch. 3 §3.3'), r.text.slice(-200));
let t = await answerNumber('3×3 patch centred on (row 3, column 5)', 36);
check('q-rowmajor-patch: 36', t.includes('Correct.') && t.includes('2·16 + 4 = 36'), t.slice(-200));
t = await answerNumber('How many warps diverge at', 2);
check('q-divergence: 2 is wrong, and the answer 1 is given', t.includes('The answer is 1'), t.slice(-200));
t = await answerNumber('How many warps diverge at', 1);
check('q-divergence: 1', t.includes('Correct.') && t.includes('992 to 999'), t.slice(-200));
t = await answerNumber('acc += w_s[k] * x[i + k]', 0.5);
check('q-shared-flopb: 0.5 FLOP/B', t.includes('Correct.') && t.includes('ch. 5 §5.1–5.2'), t.slice(-200));
r = await answerChoice('How many copies of table', 'One.');
check('q-constant-copies: one per device', r.found && r.text.includes('Correct.') && r.text.includes('a distinct object per device'), r.text.slice(-200));
check('every question names its source page', (await page.$$eval('.q .verdict', (v) => v.map((e) => e.textContent))).every((s) => /Source: .*p/.test(s)));
if (out) await page.screenshot({ path: out + '-questions.png', fullPage: true });

// ---- narrow width: nothing scrolls the page sideways
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
const over = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', over <= 1, `overflow ${over}px`);
if (out) await page.screenshot({ path: out + '-narrow.png' });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
