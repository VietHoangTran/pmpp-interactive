// Browser check for chapter 21, "Before you start": the five prerequisite questions and the
// matrix-size prediction, whose reveal is checked for each common slip. The sector count is
// compared with the model the page computes it from (src/sim/coalesce.ts, imported here).
// Usage: node e2e/ch21-start.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { SEGMENT_MODELS, warpAccess } from '../src/sim/coalesce.ts';

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
const qHandle = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const answerNumeric = async (needle, value) => {
  const q = await qHandle(needle);
  const input = await q.$('input[type=number]');
  await input.click({ clickCount: 3 });
  await input.press('Backspace');
  await input.type(String(value));
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
const PREDICT = 'Commit a size in GB';
const commitPrediction = async (value) => {
  const q = await qHandle(PREDICT);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(150);
  return q.evaluate((e) => e.textContent);
};
const fresh = async () => {
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.waitForSelector('.page[data-ready]', { timeout: 30000 });
  await page.waitForSelector('.q');
  await pause(300);
};

await goto(page, 'ch21/start');
await page.waitForSelector('.q');

// Structure: the prediction sits in the PyTorch section, the five questions come last, in
// the order the chapter uses them.
const labels = await page.$$eval('.q .kind', (k) => k.map((x) => x.textContent));
const want = ['Predict first', 'From chapter 9 and appendix A', 'From chapter 7', 'From chapter 6', 'From chapter 6', 'From chapter 4'];
check('one prediction, then five questions labelled by chapter, in order', JSON.stringify(labels) === JSON.stringify(want), labels.join('|'));
const heads = await page.$$eval('.prose h2', (h) => h.map((x) => x.textContent));
check('sentence-case headings in the planned order', JSON.stringify(heads) === JSON.stringify(['The problem in one paragraph', 'In PyTorch terms', 'The chapter in this course', 'Before you start']), heads.join('|'));

// Margin notes: the book's pages, the opening's slip, the VMD fact with its date, the bridge as ours.
const notes = await page.$$eval('aside.note', (a) => a.map((x) => x.textContent));
check('the page cites §21.1, pp. 513–514', notes.some((t) => t.includes('§21.1') && t.includes('pp. 513–514')), notes.join(' | '));
check('the opening\'s "gather-to-scatter" slip is flagged', notes.some((t) => t.includes('gather-to-scatter') && t.includes('scatter to gather')));
check('the VMD release carries its check date and facts item', notes.some((t) => t.startsWith('Today') && t.includes('2.0.0') && t.includes('2026-11-20') && t.includes('A1')));
check('the PyTorch bridge is marked as not in the book, with the book\'s pages', notes.some((t) => t.startsWith('Not in the book') && t.includes('p. 536') && t.includes('p. 497')));

// Links: the map points at every page of the chapter and at chapter 22's decomposition section.
const links = await page.$$eval('.prose a', (a) => a.map((x) => x.getAttribute('href')));
for (const l of ['#/ch21/21.2', '#/ch21/21.5', '#/ch21/labs', '#/ch21/wrap', '#/ch22/22.2', '#/review'])
  check(`links to ${l}`, links.includes(l), links.join('|'));

// Questions. The sector count comes from the coalescing model.
const lanes = Array.from({ length: 32 }, (_, t) => t);
const seg = SEGMENT_MODELS.nvidia.segmentBytes;
const s4 = warpAccess(lanes.map((t) => 16 * t), 4, seg).transactions;
const s1 = warpAccess(lanes.map((t) => 4 * t), 4, seg).transactions;
check('model: stride 4 touches 16 sectors, stride 1 touches 4', s4 === 16 && s1 === 4, `${s4} ${s1}`);
let t = await answerNumeric('How many 32-byte sectors', s1);
check(`sector question: ${s1} (the coalesced count) is wrong, with the answer ${s4}`, t.includes(`Not quite. The answer is ${s4} sectors.`), t.slice(0, 300));
t = await answerNumeric('How many 32-byte sectors', s4);
check(`sector question: ${s4} is accepted, explanation gives ${4 * s4} against ${4 * s1}`, t.includes('Correct.') && t.includes(`${4 * s4} sectors for the four`) && t.includes(`needs ${4 * s1}`), t.slice(0, 900));
check('sector explanation points to page 21.2', t.includes('On page 21.2'));

t = await answerChoice('differ from each other in the last few digits', 0);
check('float-order question: "no, order and rounding" is correct', t.includes('Correct.') && t.includes('not bit-for-bit reproducible'), t.slice(0, 400));
t = await answerChoice('What does NVIDIA document about the two reads', 1);
check('constant-memory question: "both are broadcast" is wrong, with the reason', t.includes('Not quite.') && t.includes('one access per distinct address in the warp'), t.slice(0, 500));
check('constant-memory explanation names case A and case B on page 21.2', t.includes('case A') && t.includes('case B'));
t = await answerChoice('What can still make the coarsened', 0);
check('coarsening question: registers and occupancy is correct', t.includes('Correct.') && t.includes('--resource-usage'), t.slice(0, 400));
t = await answerChoice('What does the warp execute', 1);
check('divergence question: "the body 8 times" is wrong, with the reason', t.includes('Not quite.') && t.includes('per path, not per lane'), t.slice(0, 400));
check('divergence explanation points to page 21.5', t.includes('On page 21.5'));
await pause(700); // the app saves progress after a short delay
check('answers were saved through PUT /api/progress (intercepted)', saved.some((s) => s.includes('ch21/start/q-sectors')));

// The prediction. Every reveal states the model's size: 10^6 × 10^5 × 4 bytes.
const GB = (1e6 * 1e5 * 4) / 1e9;
const GIB = Math.round(((1e6 * 1e5 * 4) / 2 ** 30) * 10) / 10;
const headline = `${GB} GB.`;
t = await commitPrediction(100);
check(`reveal gives ${GB} GB and ${GIB} GiB`, t.includes(headline) && t.includes(`${GIB} GiB`), t.slice(0, 300));
check('reveal for 100 explains the missing 4 bytes', t.includes('You counted entries, not bytes'), t.slice(0, 500));
check('reveal gives the result size and the minimum traffic', t.includes('4 MB') && t.includes(`at least ${(2 * GB).toLocaleString('en-US')} GB of memory traffic`), t.slice(0, 900));
await pause(700);
check('the commit was saved through PUT /api/progress (intercepted)', saved.some((s) => s.includes('ch21/start/p-materialise')));
if (out) await page.screenshot({ path: out + '-reveal.png' });

const cases = [
  [40, 'The grid has ten times more points', 'reveal for 40 points at the 10^6 grid points'],
  [800, 'You counted two matrices', 'reveal for 800 recognises distances plus quotients'],
  [0.4, 'That is the size in terabytes', 'reveal for 0.4 recognises terabytes'],
  [4000, 'off by a power of ten', 'reveal for 4000 says it is a power of ten off'],
  [372, 'That is what you predicted, in GiB', 'reveal for 372 accepts GiB'],
  [400, 'That is what you predicted.', 'reveal for 400 confirms it'],
  [7, 'one entry per pair', 'reveal for an unrelated guess shows the count'],
];
for (const [v, needle, name] of cases) {
  await fresh();
  t = await commitPrediction(v);
  check(name, t.includes(needle) && t.includes(headline), t.slice(0, 500));
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
