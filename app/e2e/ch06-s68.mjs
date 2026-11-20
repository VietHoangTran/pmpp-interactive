// Browser check for section 6.8, the checklist of optimisations: the sorting drill
// (one card misfiled, then all twelve, the 3 + 6 + 2 + 1 tally), the decision tool
// (every symptom raises the rows the model ranks first, and marks the rows that spend
// what is short), a row's details with its facts-file "seen in" entries, the scenario
// cards, three questions, and the page at a narrow width. Expected values come from the
// model, src/sim/checklist.ts, imported here directly.
// Usage: node e2e/ch06-s68.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { ROWS, SCENARIOS, SYMPTOMS, categoryCounts, rankRows } from '../src/sim/checklist.ts';

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
const bench = (t) => `section.bench[aria-label="${t}"]`;
const S = bench('Sort the twelve');
const D = bench('Checklist as a decision tool');
const C = bench('Scenario cards');
const norm = (s) => s.replace(/\s+/g, ' ').trim();
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const click = async (sel) => {
  await page.$eval(sel, (e) => e.click());
  await pause(60);
};

await goto(page, 'ch06/6.8');
await page.waitForSelector(S);

// ---- sorting drill
check('twelve cards, in the book order', JSON.stringify(await page.$$eval(S + ' [data-card]', (c) => c.map((x) => x.dataset.card))) === JSON.stringify(ROWS.map((r) => r.id)));
check('the book column of the tally is hidden until all twelve are placed', (await page.$$eval(S + ' tr[data-tally] td:last-child', (t) => t.map((x) => x.textContent))).every((x) => x === '?'));
await click(`${S} [data-card="coarsen"] button[data-cat="memory"]`);
check('a misfiled card says so and gives the book’s category', (await page.$eval(`${S} [data-card="coarsen"]`, (e) => e.dataset.verdict)) === 'wrong' &&
  (await text(`${S} [data-card="coarsen"]`)).includes('The book files it under general.'));
check('status counts one placed', (await text(S + ' [data-sort-result]')).startsWith('1 of 12 placed.'));
for (const r of ROWS) await click(`${S} [data-card="${r.id}"] button[data-cat="${r.category}"]`);
const verdicts = await page.$$eval(S + ' [data-card]', (c) => c.map((x) => x.dataset.verdict));
check('every card filed as the book does shows "Same as the book"', verdicts.every((v) => v === 'right'), verdicts.join(','));
const book = categoryCounts();
const tally = await page.$$eval(S + ' tr[data-tally]', (t) => t.map((r) => [...r.querySelectorAll('td.num')].map((c) => Number(c.textContent))));
check('tally: yours equals the book’s, 3 / 6 / 2 / 1', JSON.stringify(tally) === JSON.stringify([[3, 3], [6, 6], [2, 2], [1, 1]]) && book.compute === 3, JSON.stringify(tally));
const sortResult = await text(S + ' [data-sort-result]');
check('the drill reports 12 of 12 and the count 3 + 6 + 2 + 1 = 12', sortResult.includes('12 of 12 where the book puts them') && sortResult.includes('3 + 6 + 2 + 1 = 12'), sortResult);
if (out) await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-sort.png' });

// ---- decision tool: every symptom
for (const s of SYMPTOMS) {
  await click(`${D} button[data-symptom="${s.id}"]`);
  const shown = await page.$$eval(D + ' li[data-row]', (l) => l.map((x) => [x.dataset.row, x.dataset.role]));
  const want = rankRows(s.id).map((x) => [x.row.id, x.role]);
  check(`symptom "${s.label}": rows in the model’s order, ${s.relief.length} relieving first${s.spends.length ? `, ${s.spends.length} marked as spending` : ''}`,
    JSON.stringify(shown) === JSON.stringify(want), JSON.stringify(shown.slice(0, 6)));
  const first = await text(`${D} li[data-row="${s.relief[0].row}"] [data-how]`);
  check(`symptom "${s.label}": the top row says how it relieves it, with a page`, first.includes(s.relief[0].how) && first.includes(s.relief[0].page), first);
}
await click(`${D} button[data-symptom="none"]`);
check('book order lists the twelve rows as the table does', JSON.stringify(await page.$$eval(D + ' li[data-row]', (l) => l.map((x) => x.dataset.row))) === JSON.stringify(ROWS.map((r) => r.id)));
const looks = await text(D + ' [data-looks]');
check('book order describes the 3 / 6 / 2 / 1 split', looks.includes('three rows for compute, six for memory, two for synchronisation, one general'), looks);

// a row's details: costs, returns, seen in
await click(`${D} button[data-symptom="global-stall"]`);
await page.evaluate((D) => [...document.querySelectorAll(D + ' li[data-row="vector"] button')].find((b) => b.textContent === 'More').click(), D);
await pause(80);
const seen = await page.$$eval(`${D} [data-details="vector"] [data-seen]`, (d) => d.map((x) => x.dataset.seen));
check('vector loads: details show the two facts entries, D4 then D1', JSON.stringify(seen) === JSON.stringify(['D4', 'D1']), seen.join(','));
const vec = await text(`${D} li[data-row="vector"]`);
check('vector loads: cost cites pp. 138–139 and facts A11; returns in ch. 15', vec.includes('facts A11') && vec.includes('pp. 138–139') && vec.includes('ch. 15'), vec);
check('in this symptom, tiling is marked as spending what is short', (await page.$eval(`${D} li[data-row="smem-tiling"]`, (e) => e.dataset.role)) === 'spends' &&
  (await text(`${D} li[data-row="smem-tiling"]`)).includes('spends what is short'));
await page.evaluate((D) => [...document.querySelectorAll(D + ' button')].find((b) => b.textContent === 'Open all rows').click(), D);
await pause(80);
check('"Open all rows" opens twelve', (await page.$$(D + ' [data-details]')).length === 12);
const notChecked = await page.$$eval(D + ' [data-details]', (d) => d.filter((x) => x.textContent.includes('not checked')).map((x) => x.dataset.details));
check('rows without facts entries say "not checked"', JSON.stringify(notChecked.sort()) === JSON.stringify(ROWS.filter((r) => !r.seenIn.length).map((r) => r.id).sort()), notChecked.join(','));
const db = await text(`${D} [data-details="double-buffer"]`);
check('double buffering: the MLX absence is shown as "Not in"', db.includes('Not in mlx') || db.includes('Not in MLX'), db);
if (out) await page.$eval(D, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-tool.png' });

// ---- scenario cards
const picks = [];
for (const [i, s] of SCENARIOS.entries()) {
  const sel = `${C} [data-scenario="${s.id}"]`;
  const already = await page.$eval(sel, (e) => e.dataset.picked);
  // Predict right on the odd cards and wrong on the even ones (if this browser has not answered them already).
  const pick = already !== 'none' ? already : i % 2 === 0 ? s.expect : s.expect === 'faster' ? 'slower' : 'faster';
  if (already === 'none') await click(`${sel} button[data-outcome="${pick}"]`);
  picks.push(pick);
  const reveal = await text(sel + ' [data-reveal]');
  const right = pick === s.expect;
  check(`card ${i + 1} (${s.id}): reveal says ${right ? 'as you predicted' : 'what the rule says'}, with the row and ${s.page}`,
    (right ? reveal.includes('as you predicted') : reveal.includes('the rule says')) && reveal.includes(s.page) && reveal.includes(s.why.slice(0, 40)), reveal);
  check(`card ${i + 1}: the buttons lock after a prediction`, await page.$eval(`${sel} button[data-outcome="faster"]`, (b) => b.disabled));
}
const nRight = SCENARIOS.filter((s, i) => picks[i] === s.expect).length;
check(`scenario summary: ${nRight} of ${SCENARIOS.length}`, (await text(C + ' [data-scenario-result]')) === `${nRight} of ${SCENARIOS.length} predicted as the rule says.`);
if (out) await page.$eval(C, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-scenarios.png' });

// ---- questions
const answerChoice = async (promptPart, optionPart) => {
  const q = await page.evaluateHandle((p) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(p)), promptPart);
  await q.evaluate((e, o) => [...e.querySelectorAll('.opts label')].find((x) => x.textContent.includes(o)).querySelector('input').click(), optionPart);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(100);
  // the verdict, plus the reason shown beside a wrong pick
  return q.evaluate((e) => (e.querySelector('.verdict')?.textContent ?? '') + ' ' + (e.querySelector('.opts label.wrong .muted')?.textContent ?? ''));
};
let v = await answerChoice('compute a 2T × 2T tile of the output', 'Faster: tiling always');
check('q-tiling-occ: "faster" is wrong, with its reason', v.includes('Not quite.') && v.includes('removes resident blocks'), v);
v = await answerChoice('compute a 2T × 2T tile of the output', 'No change:');
check('q-tiling-occ: "no change" is wrong: twice the shared memory per block at the same thread count', v.includes('Not quite.') && v.includes('twice the shared memory per block at the same thread count'), v);
v = await answerChoice('compute a 2T × 2T tile of the output', 'Slower:');
const occExplain = await page.evaluate(() => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes('compute a 2T × 2T tile of the output')).textContent.replace(/\s+/g, ' '));
check('q-tiling-occ: "slower" is correct, cites p. 153, and says why doubling TILE_WIDTH alone keeps 8 bytes per thread',
  v.includes('Correct.') && v.includes('p. 153') && occExplain.includes('four times the shared memory and four times the threads') && occExplain.includes('two floats, 8 bytes'), v + ' / ' + occExplain);
const tb = await text(`${C} [data-scenario="tiling-bandwidth"]`);
check('scenario "tiling-bandwidth" says a tile twice as wide is four times the shared memory and the threads', tb.includes('four times the shared memory and four times the threads') && !tb.includes('doubling its shared memory'), tb);
v = await answerChoice('32 sectors per request', 'Coalescing:');
check('q-sectors: the coalescing row is correct', v.includes('Correct.') && v.includes('A17'), v);
v = await answerChoice('under “general”', 'The overhead it removes depends on the kernel');
check('q-general: correct', v.includes('Correct.'), v);

// ---- narrow screen
await page.setViewport({ width: 390, height: 900 });
await pause(300);
const fits = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
check('at 390 px nothing overflows the page', fits);
if (out) await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-narrow.png' });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
