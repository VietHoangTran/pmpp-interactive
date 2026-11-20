// Browser check for module X1's start page: the chain map drawn from chain.ts's STAGES
// (every box and every lab link), the bytes Predict and its reveal, and all six
// prerequisite questions with their links back to the sections they come from.
// Expected values come from src/sim/chain.ts, imported inside the page through the app's
// module server (chain.ts imports its neighbours without file extensions).
// Usage: APP_URL=http://localhost:5173 node e2e/x1-start.mjs [screenshot-prefix]
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

/** The question block whose prompt contains `text`. */
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

await goto(page, 'x1/start');
await page.waitForSelector('ol[data-part="chain-map"]');
const M = await page.evaluate(async () => {
  const m = await import('/src/sim/chain.ts');
  return {
    order: m.M5_ORDER,
    labs: Object.fromEntries(m.M5_ORDER.map((id) => [id, m.stage(id).kernels.map((k) => [k.lab.num, k.lab.chapter])])),
    draw: [m.DRAW.lab.num, m.DRAW.lab.chapter],
    greedy: [m.GREEDY.lab.num, m.GREEDY.lab.chapter],
    rowBytes: m.rowBytes(m.VOCAB.qwen3),
    vocab: m.VOCAB.qwen3,
  };
});

// ---------------------------------------------------------------- the chain map
const boxes = await page.$$eval('ol[data-part="chain-map"] > li', (ls) => ls.map((l) => l.dataset.stage));
check('the map has one box per M5 stage in M5 order, then the draw', JSON.stringify(boxes) === JSON.stringify([...M.order, 'draw']), JSON.stringify(boxes));
for (const id of M.order) {
  const links = await page.$$eval(`ol[data-part="chain-map"] li[data-stage="${id}"] li[data-lab]`, (ls) =>
    ls.map((l) => [l.dataset.lab, l.querySelector('a').getAttribute('href')]));
  const want = M.labs[id].map(([num, ch]) => [num, `#/${ch}/labs`]);
  check(`map box "${id}" links each kernel to its lab's page`, JSON.stringify(links) === JSON.stringify(want), JSON.stringify(links));
}
const drawLinks = await page.$$eval('ol[data-part="chain-map"] li[data-stage="draw"] li[data-lab]', (ls) => ls.map((l) => [l.dataset.lab, l.querySelector('a').getAttribute('href')]));
check('the draw box links lab X1.3, and greedy links lab 10.3',
  JSON.stringify(drawLinks) === JSON.stringify([[M.draw[0], `#/${M.draw[1]}/labs`], [M.greedy[0], `#/${M.greedy[1]}/labs`]]), JSON.stringify(drawLinks));
const allLabHrefs = await page.$$eval('ol[data-part="chain-map"] a', (as) => as.map((a) => a.getAttribute('href')));
check("every map link goes to a chapter's labs page", allLabHrefs.every((h) => /^#\/(ch09|ch10|ch11|ch12|ch14|x1)\/labs$/.test(h)), allLabHrefs.join(' '));
if (out) await page.screenshot({ path: out + '-map.png' });

// ---------------------------------------------------------------- the Predict
const pq = await qHandle('least number of bytes');
const hasInput = await pq.evaluate((e) => !!e.querySelector('input[type=number]'));
if (hasInput) {
  const input = await pq.$('input[type=number]');
  await input.type('600000');
  await pq.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.includes('Commit')).click());
  await pause(150);
}
const reveal = await page.$eval('[data-reveal="bytes"]', (e) => e.textContent.replace(/\s+/g, ' '));
check('the reveal gives 8 bytes per token and the row total from the model',
  reveal.includes(`${M.vocab.toLocaleString('en-US')} × 8 = ${M.rowBytes.toLocaleString('en-US')} bytes`) && M.rowBytes === 1215488, reveal.slice(0, 160));
check('the reveal gives batch 64: 77,791,232 bytes, 77.8 MB', reveal.includes((M.rowBytes * 64).toLocaleString('en-US') + ' bytes') && reveal.includes('77.8 MB'));
check('the reveal says every stage is memory-bound and links the roofline', reveal.includes('memory-bound') && (await page.$eval('[data-reveal="bytes"] a', (a) => a.getAttribute('href'))) === '#/ch05/5.1');

// ---------------------------------------------------------------- the six questions and their links
const LINKS = [
  ['Lab 10.3\'s softmax subtracts', '#/ch10/labs'],
  ['Sorted probabilities 0.5, 0.25, 0.125, 0.125', '#/ch11/11.1'],
  ['Lab 12.4 builds each row', '#/ch12/12.5'],
  ['private copy of the counts', '#/ch09/9.4'],
  ['MSD radix select', '#/ch14/14.9'],
  ['one block per row. At batch 1', '#/ch10/10.10'],
];
for (const [t, h] of LINKS) {
  const q = await qHandle(t);
  const hrefs = await q.evaluate((e) => (e ? [...e.querySelectorAll('.prompt a')].map((a) => a.getAttribute('href')) : []));
  check(`question "${t.slice(0, 28)}…" links to ${h}`, hrefs.includes(h), JSON.stringify(hrefs));
}
let v = await answerChoice("Lab 10.3's softmax subtracts", 0);
check('q-max-subtract: overflow to infinity is right', v.startsWith('Correct.'), v.slice(0, 80));
v = await answerChoice("Lab 10.3's softmax subtracts", 1);
check('q-max-subtract: "nothing goes wrong" is rejected with exp(95)\'s size', v.startsWith('Not quite.') && (await (await qHandle("Lab 10.3's softmax subtracts")).evaluate((e) => e.textContent)).includes('88.7'));
v = await answerNumber('Sorted probabilities 0.5, 0.25, 0.125, 0.125', 2);
check('q-exclusive: 2 tokens', v.startsWith('Correct.'), v.slice(0, 80));
v = await answerChoice('Lab 12.4 builds each row', 0);
check('q-stable: the order an atomic counter gives is not repeatable', v.startsWith('Correct.'), v.slice(0, 80));
v = await answerNumber('private copy of the counts', 593.5);
check('q-vocab-kib: 593.5 KiB', v.startsWith('Correct.'), v.slice(0, 80));
v = await answerNumber('MSD radix select', 13);
check('q-msd-bin: 13 keys', v.startsWith('Correct.'), v.slice(0, 80));
v = await answerNumber('MSD radix select', 20);
check('q-msd-bin: 20 is rejected', v.startsWith('Not quite. The answer is 13'), v.slice(0, 80));
v = await answerChoice('one block per row. At batch 1', 0);
check('q-one-block: 1/S', v.startsWith('Correct.'), v.slice(0, 80));
if (out) await page.screenshot({ path: out + '-questions.png', fullPage: true });

// ---------------------------------------------------------------- notes
const todays = await page.$$eval('aside.note.today', (a) => a.map((x) => x.textContent));
check('every Today note says it was checked in the facts pass on 2026-11-20', todays.length >= 3 && todays.every((t) => t.includes('Checked 2026-11-20 in the facts pass')), String(todays.length));
check('an "Ours, not the book\'s" note opens the page', (await page.$eval('aside.note.beyond', (a) => a.textContent)).startsWith("Ours, not the book's"));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
