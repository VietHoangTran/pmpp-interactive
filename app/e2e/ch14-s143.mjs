// Browser check for section 14.3: the "Merge tree" explorer (the book's eight blocks, the stage
// slider, fewer blocks than segments, other sizes) and the "Sixteen keys, merged for real"
// panel (every stage, both tie rules), plus two questions. Every expected number is recomputed
// here from src/sim/mergestages.ts.
// Usage: node e2e/ch14-s143.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { PANEL_KEYS, PANEL_SEG, blocksOf, mergePlan, mergeSortStages } from '../src/sim/mergestages.ts';

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
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const clickText = async (scope, label) => {
  const done = await page.evaluate((sel, label) => {
    const btn = [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, bench(scope), label);
  await pause(80);
  return done;
};
const setSelect = async (scope, labelStart, value) => {
  await page.evaluate((sel, labelStart, value) => {
    const s = [...document.querySelectorAll(sel + ' label.field')].find((l) => l.textContent.trim().startsWith(labelStart)).querySelector('select');
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(s, String(value));
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, bench(scope), labelStart, value);
  await pause(90);
};
const setStage = async (v) => {
  await page.$eval(bench(M) + ' input[type=range]', (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(90);
};
const shot = async (scope, name) => {
  if (!out) return;
  await page.$eval(bench(scope), (e) => e.scrollIntoView({ block: 'start' }));
  await pause(100);
  await page.screenshot({ path: `${out}-${name}.png` });
};

const M = 'Merge tree';
const K = 'Sixteen keys, merged for real';
await goto(page, 'ch14/14.3');
await page.waitForSelector(bench(M));

const table = () => page.$$eval(bench(M) + ' tr[data-stage-row]', (rs) => rs.map((r) => ({ stage: r.dataset.stageRow, share: r.dataset.share, searches: r.dataset.searches, cur: r.classList.contains('cur') })));
const assigned = () => page.$$eval(bench(M) + ' [data-merge]', (ts) => ts.map((t) => t.textContent.trim()));

// The book's eight blocks (the default).
let plan = mergePlan({ n: 2 ** 20, segLen: 2 ** 17, blocks: 8 });
let t = await table();
check('the book\'s example: sort, then 2, 4 and 8 blocks per merge',
  t.map((r) => r.share).join(' | ') === '1 block per segment | 2 blocks per merge | 4 blocks per merge | 8 blocks per merge'
  && plan.rows.slice(1).map((r) => r.blocksPerMerge).join() === '2,4,8', t.map((r) => r.share).join(' | '));
check('split searches per stage are the model\'s (4, 6, 7)', t.slice(1).map((r) => r.searches).join() === plan.rows.slice(1).map((r) => r.splitSearches).join() && t.slice(1).map((r) => r.searches).join() === '4,6,7', t.map((r) => r.searches).join());
check('stage 0 is drawn: one block per segment', (await assigned()).join() === 'block 0,block 1,block 2,block 3,block 4,block 5,block 6,block 7');
await setStage(1);
check('stage 1: blocks 0-1, 2-3, 4-5, 6-7 on the four merges', (await assigned()).join() === [0, 1, 2, 3].map((m) => blocksOf(plan.rows[1], m)).map((b) => `blocks ${b[0]}–${b[b.length - 1]}`).join() && (await table())[1].cur);
await setStage(3);
check('stage 3: one merge, all eight blocks', (await assigned()).join() === 'blocks 0–7' && (await page.$eval(bench(M) + ' svg', (e) => e.dataset.segments)) === '1');
await shot(M, 'tree-book');

// q-stages and fewer blocks than segments: 2^20 keys, segments of 1,024, 64 blocks.
await setSelect(M, 'Initial segment length', 1024);
await setSelect(M, 'Blocks in the grid', 64);
plan = mergePlan({ n: 2 ** 20, segLen: 1024, blocks: 64 });
t = await table();
check('2^20 keys in segments of 1,024: 10 merge stages', t.length === 11 && plan.stages === 10 && (await page.$eval(bench(M) + ' [data-summary]', (e) => e.textContent)).includes('1,024 segments, so 10 merge stages'));
check('64 blocks: whole merges per block until merges drop below 64, then blocks share',
  t.map((r) => r.share).join(' | ') === plan.rows.map((r) => (r.blocksPerMerge ? `${r.blocksPerMerge} block${r.blocksPerMerge > 1 ? 's' : ''} per ${r.stage ? 'merge' : 'segment'}` : `${r.mergesPerBlock} ${r.stage ? 'merge' : 'segment'}s per block`)).join(' | ')
  && t[1].share === '8 merges per block' && t[5].share === '2 blocks per merge' && t[10].share === '64 blocks per merge', t.map((r) => r.share).join(' | '));
check('no split search while blocks take whole merges', t.slice(1, 5).every((r) => r.searches === '0') && t[10].searches === '63');
await setStage(1);
check('stage 1 has too many merges to label one by one: a summary instead', (await page.$eval(bench(M) + ' [data-assign]', (e) => e.textContent)).includes('512 merges: 8 merges per block'));
await shot(M, 'tree-64');

// q-blocks: 64 blocks, 64 segments, stage 4.
await setSelect(M, 'Initial segment length', 2 ** 14);
await setStage(4);
plan = mergePlan({ n: 2 ** 20, segLen: 2 ** 14, blocks: 64 });
t = await table();
check('64 segments on 64 blocks: 16 blocks per merge in stage 4', t[4].share === '16 blocks per merge' && plan.rows[4].blocksPerMerge === 16 && t[4].cur);
check('stage 4 labels its four merges with sixteen blocks each', (await assigned()).join() === 'blocks 0–15,blocks 16–31,blocks 32–47,blocks 48–63');

// The preset button restores the book's example.
await clickText(M, 'The book\'s eight blocks');
check('the preset restores the book\'s eight blocks', (await table()).map((r) => r.share).join() === '1 block per segment,2 blocks per merge,4 blocks per merge,8 blocks per merge');

// ---------------------------------------------------------------- the sixteen keys
const panel = () => page.$eval(bench(K) + ' [data-row="panel"]', (e) => e.dataset.values);
const snaps = mergeSortStages(PANEL_KEYS, PANEL_SEG);
const enc = (items) => items.map((x) => `${x.key}:${x.id}`).join(' ');
check('stage 0: each segment of two sorted', (await panel()) === enc(snaps[0].items));
await clickText(K, 'merge 1');
check('merge 1: runs of four, as the model', (await panel()) === enc(snaps[1].items));
await clickText(K, 'merge 2');
check('merge 2: runs of eight', (await panel()) === enc(snaps[2].items));
await clickText(K, 'merge 3');
check('merge 3: sorted, every repeated key in input order', (await panel()) === enc(snaps[3].items) && (await page.$eval(bench(K) + ' [data-ties]', (e) => e.dataset.ties)) === 'stable');
await clickText(K, 'ties: take the right run first');
const right = mergeSortStages(PANEL_KEYS, PANEL_SEG, 'right');
const chip = await page.$eval(bench(K) + ' .chip', (e) => e.textContent.trim());
check('taking the right run first: still sorted, ties reordered', (await panel()) === enc(right[3].items) && (await page.$eval(bench(K) + ' [data-ties]', (e) => e.dataset.ties)) === 'reordered' && chip === 'repeated keys reordered: 2, 4, 5, 7, 9', chip);
await shot(K, 'panel');

// ---------------------------------------------------------------- questions
const q = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('In merge stage 4 (counting from 1)')));
await (await q.$('input[type=number]')).type('16');
await (await q.$('button.primary')).click();
await pause(150);
check('numeric question: 16 blocks per merge', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
const qm = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('How many search steps does the stage make')));
await (await qm.$('input[type=number]')).type('2621440');
await (await qm.$('button.primary')).click();
await pause(150);
check('numeric question: 2,621,440 search steps in the last stage', (await qm.evaluate((e) => e.textContent)).includes('Correct.'));
const qs = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('What does it output?')));
await qs.evaluate((e) => [...e.querySelectorAll('.opts label')].find((l) => l.textContent.trim().startsWith('2b, 2a, 5b, 5a')).querySelector('input').click());
await pause(80);
await (await qs.$('button.primary')).click();
await pause(150);
check('choice question: with <, every tie takes the right run first', (await qs.evaluate((e) => e.textContent)).includes('Correct.'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
