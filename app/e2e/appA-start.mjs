// Browser check for appendix A's start page: the teaser (commit a prediction, then four sums
// of one softmax denominator, each checked against src/sim/fpsum.ts imported through the
// app's module server), the map's links, the margin notes, the four prerequisite questions,
// no console errors, and the page at a narrow width.
// Usage: APP_URL=http://localhost:5173 node e2e/appA-start.mjs [screenshot-prefix]
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
const model = (fn, ...args) =>
  page.evaluate(async (src, args) => {
    const m = await import('/src/sim/fpsum.ts');
    return new Function('m', 'args', `return (${src})(m, ...args)`)(m, args);
  }, fn.toString(), args);
const question = (text) =>
  page.evaluateHandle((text) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(text)), text);
const answerChoice = async (text, index) => {
  const q = await question(text);
  const radios = await q.$$('input[type=radio]');
  await radios[index].click();
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
const answerNumber = async (text, value) => {
  const q = await question(text);
  const input = await q.$('input[type=number]');
  await input.click({ clickCount: 3 });
  await input.type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};

await goto(page, 'appA/start');
await page.waitForSelector('.q');

// ---- the teaser: commit first, then the four sums
const teaser = await question('half-precision');
const box = await teaser.$('input[aria-label="your prediction"]');
if (box) {
  check('no result is shown before a prediction is committed', !(await page.$('table[aria-label="Four ways to sum one softmax denominator"]')));
  await box.type('5');
  await (await teaser.$('button.primary')).click();
  await pause(300);
}
await page.waitForSelector('table[aria-label="Four ways to sum one softmax denominator"]');
const rows = await page.$$eval('table[aria-label="Four ways to sum one softmax denominator"] tbody tr', (trs) =>
  trs.map((tr) => ({ id: tr.dataset.row, result: tr.querySelector('[data-k=result]').textContent, rel: tr.querySelector('[data-k=rel]').textContent, ulps: tr.querySelector('[data-k=ulps]').textContent })),
);
const want = await model((m) => {
  const t = m.teaser();
  return { reference: m.show(t.reference, 10), rows: t.rows.map((r) => ({ id: r.id, result: m.show(r.e.result, 9), rel: m.pct(r.e.rel), ulps: m.ulpText(r.e.ulps), relRaw: r.e.rel })) };
});
check('the reveal shows four rows: float loop, float tree, half loop, half tree', rows.map((r) => r.id).join() === 'fp32-loop,fp32-tree,fp16-loop,fp16-tree', rows.map((r) => r.id).join());
check('every result, relative error and ULP count is the model\'s', JSON.stringify(rows) === JSON.stringify(want.rows.map(({ relRaw, ...r }) => r)), JSON.stringify({ rows, want: want.rows }));
check('the reference shown is the model\'s compensated sum', (await page.$eval('[data-k=reference]', (e) => e.textContent)) === want.reference);
const half = want.rows.find((r) => r.id === 'fp16-loop');
const tree = want.rows.find((r) => r.id === 'fp16-tree');
check('the half loop is off by more than three quarters; the half tree is within one ULP', half.relRaw > 0.75 && Number(tree.ulps) <= 1, JSON.stringify([half, tree]));
const reveal = await teaser.evaluate((e) => e.textContent);
check('the reveal names sections A.4 and A.6', reveal.includes('A.4') && reveal.includes('A.6'));
if (out) await page.screenshot({ path: out + '-teaser.png' });

// ---- notes and the map
const notes = await page.$$eval('aside.note', (as) => as.map((a) => a.textContent));
check('the page cites the book (pp. 583, 593, 594)', notes.some((n) => n.includes('583')));
check('a Today note gives MX v1.1 and IEEE 754-2019 (facts A1, A4)', notes.some((n) => n.startsWith('Today') && n.includes('v1.1') && n.includes('UE8M0') && n.includes('754-2019')));
check('a Not-in-the-book note names PyTorch, MLX and llama.cpp', notes.some((n) => n.includes('Not in the book') && n.includes('PyTorch') && n.includes('MLX') && n.includes('ggml_prec_set_acc')));
const links = await page.$$eval('.prose a', (as) => as.map((a) => a.getAttribute('href')));
check('the map links to A.1 to A.7 and to the labs', ['#/appA/A.1', '#/appA/A.3', '#/appA/A.4', '#/appA/A.5', '#/appA/A.6', '#/appA/A.7', '#/appA/labs'].every((h) => links.includes(h)), links.join(' '));

// ---- the prerequisite questions
let t = await answerChoice('using', 0);
check('q-tolerance: comparing with a tolerance is right', t.includes('Correct.'), t.slice(0, 120));
t = await answerChoice('using', 3);
check('q-tolerance: comparing bit patterns is wrong, with its reason', t.includes('Not quite.') && t.includes('The bit patterns differ'), t.slice(0, 160));
t = await answerNumber('which element does thread 2 add', 10);
check('q-convergent: thread 2 adds element 10', t.includes('Correct.'));
t = await answerNumber('per byte of weight traffic', 2);
check('q-bytes: 2 FLOP/B', t.includes('Correct.'));
t = await answerChoice('atomic add', 0);
check('q-atomic: yes, the order differs from run to run', t.includes('Correct.'));
const labels = await page.$$eval('.q .kind', (ks) => ks.map((k) => k.textContent));
check('each prerequisite question names its chapter', ['From chapter 10', 'From chapters 5 and 20'].every((l) => labels.includes(l)), labels.join('|'));
if (out) await page.screenshot({ path: out + '-questions.png', fullPage: true });

// ---- narrow
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
const fits = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
check('at 390 px the page does not scroll sideways', fits);
if (out) await page.screenshot({ path: out + '-narrow.png', fullPage: true });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
