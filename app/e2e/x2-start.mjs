// Browser check for module X2's start page: the "what runs at decode" table, the
// classification "Which kernel serves this call?" (all right, then one wrong with its
// feedback), the Q4_0 size Predict and its reveal table against qtraffic.ts, the links of
// the plan, and all six prerequisite questions.
// Usage: APP_URL=http://localhost:5173 node e2e/x2-start.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';

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
const C = 'section.bench[aria-label="Which kernel serves this call?"]';
const qHandle = (text) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(t)), text);
async function answerChoice(text, index) {
  const q = await qHandle(text);
  if (!(await q.evaluate((e) => !!e))) return 'missing';
  await q.evaluate((e, i) => e.querySelectorAll('input[type=radio]')[i].click(), index);
  await pause(60);
  await q.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Check').click());
  await pause(120);
  return q.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
}
async function answerNumber(text, value) {
  const q = await qHandle(text);
  if (!(await q.evaluate((e) => !!e))) return 'missing';
  const input = await q.$('input[type=number]');
  await input.click({ clickCount: 3 });
  await input.type(String(value));
  await q.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Check').click());
  await pause(120);
  return q.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
}
const pick = async (call, family) => {
  await page.evaluate((C, call, family) => {
    const s = document.querySelector(`${C} tr[data-call="${call}"] select`);
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(s, family);
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, C, call, family);
  await pause(60);
};
const checkBtn = async () => {
  await page.evaluate((C) => [...document.querySelectorAll(C + ' button')].find((b) => b.textContent.trim() === 'Check').click(), C);
  await pause(120);
};

await goto(page, 'x2/start');
await page.waitForSelector(C);
const M = await page.evaluate(async () => {
  const m = await import('/src/sim/qtraffic.ts');
  return Object.fromEntries(['f32', 'f16', 'q8_0', 'q4_0'].map((f) => [f, { bits: m.bitsPerWeight(f), gb: (m.modelBytes(7e9, f) / 1e9).toFixed(2) }]));
});

// ---------------------------------------------------------------- the today table
const today = await page.$eval('table[data-table="today"]', (e) => e.textContent);
check('the decode table names the three routes and their kernels', ['Int4WeightOnlyConfig', '_weight_int4pack_mm', 'QuantizedLinear', 'affine_qmv_fast', 'mmvq', 'mul_mv'].every((t) => today.includes(t)));

// ---------------------------------------------------------------- classification
const ANSWERS = { 'torch-int4': 'int4pack', 'int-mm': 'refused', mlx: 'mlx-qmv', 'llama-cuda': 'mmvq', 'llama-metal': 'mulmv' };
const disabled = await page.evaluate((C) => [...document.querySelectorAll(C + ' button')].find((b) => b.textContent.trim() === 'Check').disabled, C);
check('Check is disabled until every call has an answer', disabled);
for (const [call, fam] of Object.entries(ANSWERS)) await pick(call, fam);
await checkBtn();
let verdicts = await page.$$eval(C + ' tr[data-call]', (r) => r.map((x) => x.dataset.verdict));
check('all five right: five "right" verdicts and a score of 5', verdicts.every((v) => v === 'right') && (await page.$eval(C + ' [data-score]', (e) => e.dataset.score)) === '5', JSON.stringify(verdicts));
await pick('int-mm', 'mmvq');
await checkBtn();
verdicts = await page.$$eval(C + ' tr[data-call]', (r) => r.map((x) => [x.dataset.call, x.dataset.verdict]));
const why = await page.$eval(C + ' tr[data-why="int-mm"]', (e) => e.textContent);
check('a wrong pick is marked, and its feedback names the right family and the facts item', verdicts.find(([c]) => c === 'int-mm')[1] === 'wrong' && why.includes('refused') && why.includes('A6'), why);
if (out) { await page.$eval(C, (e) => e.scrollIntoView({ block: 'start' })); await page.screenshot({ path: out + '-classify.png' }); }

// ---------------------------------------------------------------- the size Predict
const pq = await qHandle('How many GB?');
if (await pq.evaluate((e) => !!e.querySelector('input[type=number]'))) {
  const input = await pq.$('input[type=number]');
  await input.type('3.5');
  await pq.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.includes('Commit')).click());
  await pause(150);
}
const rev = await page.$eval('[data-reveal="sizes"]', (e) => e.textContent).catch(() => '');
check('the reveal gives the Q4_0 size from the model and says 3.5 GB is the codes alone', rev.includes(`${M.q4_0.gb} GB`) && rev.includes('3.5 GB is the codes alone'), rev.slice(0, 160));
const sizes = await page.$$eval('table[data-table="sizes"] tr[data-format]', (r) => r.map((x) => [x.dataset.format, Number(x.querySelector('[data-bits]').textContent), x.querySelector('[data-gb]').textContent]));
check('the size table: bits and GB per format from the model; Q8_0 is 8.5 bits', sizes.length === 4 && sizes.every(([f, b, g]) => b === M[f].bits && g === M[f].gb) && M.q8_0.bits === 8.5, JSON.stringify(sizes));
if (out) { await page.$eval('[data-reveal="sizes"]', (e) => e.scrollIntoView({ block: 'center' })); await page.screenshot({ path: out + '-sizes.png' }); }

// ---------------------------------------------------------------- the plan and the links
const links = await page.$$eval('.prose a', (as) => as.map((a) => a.getAttribute('href')));
check('the plan links all seven pages and the labs page', ['X2.1', 'X2.2', 'X2.3', 'X2.4', 'X2.5', 'X2.6', 'X2.7', 'labs'].every((s) => links.includes(`#/x2/${s}`)), links.join(' '));
check('the prerequisite questions link back to 5.1, 22 labs, 6.1, A.2, A.6 and 10.7', ['#/ch05/5.1', '#/ch22/labs', '#/ch06/6.1', '#/appA/A.2', '#/appA/A.6', '#/ch10/10.7'].every((h) => links.includes(h)));

// ---------------------------------------------------------------- the six questions
check('q-gemv-intensity', (await answerChoice('What is it with float32 weights and activations', 0)).includes('Correct.'));
check('q-gather', (await answerChoice('which thread writes Y[b][j]', 0)).includes('Correct.'));
check('q-coalesce', (await answerChoice('Are those loads coalesced?', 0)).includes('Correct.'));
check('q-fp16-int accepts 2048', (await answerNumber('Up to what n is every integer', 2048)).includes('Correct.'));
check('q-acc-order', (await answerChoice('lose more in FP16 than in float32', 0)).includes('Correct.'));
check('q-warp-reduce: a wrong answer explains itself', (await answerChoice('which lanes hold the warp', 1)).includes('Not quite.'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
