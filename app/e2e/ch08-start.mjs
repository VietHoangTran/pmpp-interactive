// Browser check for chapter 8, "Before you start": the four prerequisite questions and
// the halo prediction, whose reveal is compared with the model it runs on
// (src/sim/stencil3d.ts, imported here directly).
// Usage: node e2e/ch08-start.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { haloCount, haloFraction, haloFraction2D, threadsCube } from '../src/sim/stencil3d.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });

// Leave no trace in the real progress file, and start from an empty one so the prediction
// is uncommitted whatever the reader has already done.
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
const saved = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.url().includes('/api/progress')) {
    if (r.method() === 'PUT') saved.push(r.postData() ?? '');
    return r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : JSON.stringify(EMPTY) });
  }
  return r.continue();
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const qText = (needle) =>
  page.evaluate((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n))?.textContent ?? '', needle);
const qHandle = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const answerNumeric = async (needle, value) => {
  const q = await qHandle(needle);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
const answerChoice = async (needle, index) => {
  const q = await qHandle(needle);
  const radios = await q.$$('input[type=radio]');
  await radios[index].click();
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
const commitPrediction = async (value) => {
  const q = await qHandle('Commit a percentage');
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(150);
  return q.evaluate((e) => e.textContent);
};

await goto(page, 'ch08/start');
await page.waitForSelector('.q');

// The four prerequisite questions, labelled by chapter.
const labels = await page.$$eval('.q .kind', (k) => k.map((x) => x.textContent));
for (const n of [4, 5, 6, 7]) check(`question labelled "From chapter ${n}"`, labels.includes(`From chapter ${n}`), labels.join('|'));
check('the prediction comes after the questions', labels[labels.length - 1] === 'Predict first', labels.join('|'));

// Margin notes: the book's pages, the Today item and the Not-in-the-book note.
const notes = await page.$$eval('aside.note', (a) => a.map((x) => x.textContent));
check('the page cites the book, pp. 183–188', notes.some((t) => t.includes('pp. 183–188')), notes.join(' | '));
check('the llama.cpp bridge is marked as not in the book', notes.some((t) => t.startsWith('Not in the book') && t.includes('D1')));
check('the llama.cpp fact carries its check date', notes.some((t) => t.startsWith('Today') && t.includes('2026-11-20')));

// Questions: a numeric one right, a choice right, a choice wrong.
const halo2d = Math.round(1024 * haloFraction2D(32));
let t = await answerNumeric('32 × 32 input tile for a 3 × 3 convolution', halo2d);
check(`ch. 7 halo count: ${halo2d} is accepted`, t.includes('Correct.'), t.slice(0, 200));
t = await answerChoice('which threads make up its first warp of 32', 0);
check('ch. 4 warp question: four rows of eight is correct', t.includes('Correct.') && t.includes('four separate stretches'), t.slice(0, 300));
t = await answerChoice('which barrier could you drop', 0);
check('ch. 5 question: the barrier after the writes is wrong, with the reason', t.includes('Not quite.') && t.includes('whatever buffer it is in'), t.slice(0, 300));
t = await answerChoice('fully unrolled', 1);
check('ch. 6 question: "removes a barrier" is wrong', t.includes('Not quite.'), t.slice(0, 200));

// The prediction: a reader who carries the 2D figure over.
const want = `${haloCount(8)} of the ${threadsCube(8)} loaded elements are halo: ${Math.round(haloFraction(8) * 1000) / 10}%`;
t = await commitPrediction(12);
check('reveal gives the model\'s count and share', t.includes(want), t.slice(0, 300));
check('reveal answers the 2D intuition', t.includes('The 2D figure does not carry over'), t.slice(0, 400));
await pause(700); // the app saves progress after a short delay
check('the commit was saved through PUT /api/progress (intercepted)', saved.some((s) => s.includes('ch08/start/p-halo3d')));
const link = await page.evaluate(() => [...document.querySelectorAll('.q a')].map((a) => a.getAttribute('href')));
check('reveal links to page 8.4', link.includes('#/ch08/8.4'), link.join('|'));
if (out) await page.screenshot({ path: out + '-reveal.png' });

// A fresh page (a reload, so the empty progress file is read again): the 2D share at this
// width gets its own explanation.
const fresh = async () => {
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.waitForSelector('.page[data-ready]', { timeout: 30000 });
  await page.waitForSelector('.q');
  await pause(300);
};
await fresh();
t = await commitPrediction(43.75);
check('reveal for 43.75% explains the third axis', t.includes('third axis multiplies'), t.slice(0, 400));

// A fresh page: the right answer.
await fresh();
t = await commitPrediction(58);
check('reveal for 58% confirms it', t.includes('That is what you predicted'), t.slice(0, 400));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
