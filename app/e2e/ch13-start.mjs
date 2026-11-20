// Browser check for chapter 13, "Before you start": the redrawn running example (compared
// with the model it is computed from, src/sim/mergetile.ts, imported here directly), the
// hook and tie questions, the five prerequisite questions, the margin notes, the links of
// the chapter map, and the page at phone width.
// Usage: node e2e/ch13-start.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { RUNNING, coRankByMerge, mergeStable, mergeWith, tag } from '../src/sim/mergetile.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });

// Leave no trace in the real progress file, and start from an empty one so that no
// question is prefilled whatever the reader has already done.
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

await goto(page, 'ch13/start');
await page.waitForSelector('.q');

// ---------------------------------------------------------------- the figure against the model
const A = [...RUNNING.A];
const B = [...RUNNING.B];
const C = mergeStable(A, B);
const rowTags = (name) => page.$$eval(`figure svg g[data-row="${name}"] g[data-tag]`, (g) => g.map((x) => x.dataset.tag));
const rowKeys = (name) =>
  page.$$eval(`figure svg g[data-row="${name}"] g[data-tag]`, (g) => g.map((x) => Number(x.querySelectorAll('text')[1].textContent)));
check('figure row A is the book\'s A', JSON.stringify(await rowKeys('A')) === JSON.stringify(A), JSON.stringify(await rowKeys('A')));
check('figure row B is the book\'s B', JSON.stringify(await rowKeys('B')) === JSON.stringify(B), JSON.stringify(await rowKeys('B')));
check('figure row C has the merged keys', JSON.stringify(await rowKeys('C')) === JSON.stringify(C.map((x) => x.key)), JSON.stringify(await rowKeys('C')));
check('figure row C carries the stable merge\'s origin tags', JSON.stringify(await rowTags('C')) === JSON.stringify(C.map(tag)), JSON.stringify(await rowTags('C')));
const dashed = await page.$$eval('figure svg g[data-row="C"] g[data-tag] rect', (r) => r.map((x) => x.getAttribute('stroke-dasharray') !== null));
check('elements from B, and only those, have a dashed outline', JSON.stringify(dashed) === JSON.stringify(C.map((x) => x.src === 'B')), JSON.stringify(dashed));
const ties = await page.$$eval('figure svg g[data-tie] text', (t) => t.map((x) => x.textContent));
check('the two ties are bracketed with A first', JSON.stringify(ties) === JSON.stringify(['A1 before B0', 'A4 before B1 before B2']), JSON.stringify(ties));
const fill = await page.$eval('figure svg g[data-row="C"] rect', (r) => r.getAttribute('fill'));
check('cells use the global-memory colour variable', fill === 'var(--global-wash)', fill);

// ---------------------------------------------------------------- margin notes and labels
const notes = await page.$$eval('aside.note', (a) => a.map((x) => x.textContent));
check('the page cites the chapter opening, pp. 303, 308', notes.some((t) => t.includes('pp. 303, 308')), notes.join(' | '));
check('the running example cites §13.1–13.2 and Figs. 13.1, 13.2', notes.some((t) => t.includes('pp. 303–306') && t.includes('Figs. 13.1, 13.2')));
check('the MLX bridge is marked as not in the book', notes.some((t) => t.startsWith('Not in the book') && t.includes('D1')));
check('the library facts carry their check date', notes.some((t) => t.startsWith('Today') && t.includes('2026-11-20') && t.includes('merge_partition')));
const labels = await page.$$eval('.q .kind', (k) => k.map((x) => x.textContent));
check('the hook comes first, before any theory', labels[0] === 'Before any theory', labels.join('|'));
for (const n of [2, 5, 6]) check(`a question labelled "From chapter ${n}"`, labels.includes(`From chapter ${n}`), labels.join('|'));
check('seven questions on the page', labels.length === 7, String(labels.length));
const links = await page.$$eval('.prose a', (a) => a.map((x) => x.getAttribute('href')));
for (const l of ['#/ch13/13.3', '#/ch13/labs', '#/ch13/wrap', '#/review']) check(`the page links to ${l}`, links.includes(l), links.join('|'));

// ---------------------------------------------------------------- the hook
let t = await answerChoice('Which input elements does thread 1 need', 1);
check('hook: B[1] instead of A[4] is wrong, and the tie rule is the reason', t.includes('Not quite.') && t.includes('A[4] = 10 goes before B[1] = 10'), t.slice(0, 400));
t = await answerChoice('Which input elements does thread 1 need', 2);
const [i3, i6] = [coRankByMerge(3, A, B), coRankByMerge(6, A, B)];
check('hook: A[2..4] and nothing from B is correct', t.includes('Correct.'), t.slice(0, 300));
check('hook: the explanation gives the model\'s co-ranks of 3 and 6', t.includes(`(${i3}, ${3 - i3}) and (${i6}, ${6 - i6})`), t.slice(0, 600));
if (out) await page.screenshot({ path: out + '-hook.png' });

// ---------------------------------------------------------------- the tie rule
t = await answerChoice('compared with A[i] < B[j]', 1);
const diff = C.filter((x, k) => tag(x) !== tag(mergeWith(A, B, '<')[k])).length;
check(`tie: "nothing changes" is wrong, C differs in ${diff} places`, t.includes('Not quite.') && t.includes(`differs in ${diff} of its 9 places`), t.slice(0, 400));
t = await answerChoice('compared with A[i] < B[j]', 3);
const ltSeq = mergeWith(A, B, '<').map(tag).join(' ');
check('tie: the right answer shows the model\'s < order', t.includes('Correct.') && t.includes(ltSeq), t.slice(0, 500));

// ---------------------------------------------------------------- the prerequisite questions
t = await answerNumeric('int per = ceil', 9);
check('ceil: 9 is wrong, the answer is 8', t.includes('Not quite. The answer is 8 outputs.'), t.slice(0, 300));
t = await answerNumeric('int per = ceil', 8);
check('ceil: 8 is correct, C[8] is never written', t.includes('Correct.') && t.includes('is never written'), t.slice(0, 400));
t = await answerNumeric('32-byte segments', 8);
check('stride: 8 segments is correct', t.includes('Correct.') && t.includes('50%'), t.slice(0, 300));
t = await answerChoice('sorted 1,024-element array', 0);
check('strategy: remapping threads to keys is wrong', t.includes('Not quite.') && t.includes('still depends on the data'), t.slice(0, 300));
t = await answerChoice('sorted 1,024-element array', 1);
check('strategy: staging through shared memory is correct', t.includes('Correct.') && t.includes('names this very chapter'), t.slice(0, 400));
t = await answerChoice('calls does each iteration need', 3);
check('barriers: "none within one warp" is wrong, with the reason', t.includes('Not quite.') && t.includes('not guaranteed to run in lockstep'), t.slice(0, 400));
t = await answerChoice('calls does each iteration need', 2);
check('barriers: two is correct', t.includes('Correct.') && t.includes('write after read'), t.slice(0, 400));
t = await answerNumeric('40 global reads of setup', 6);
check('coarsen: 6 million is correct', t.includes('Correct.') && t.includes('41 million'), t.slice(0, 300));
// The right option is not always first: the chapter spreads it over all four positions.
const rightAt = await page.$$eval('.q', (qs) => qs.filter((q) => q.querySelector('.opts')).map((q) => [...q.querySelectorAll('.opts label')].findIndex((l) => l.classList.contains('right'))));
check('the four choice questions have their right option in positions 3, 4, 2 and 3', JSON.stringify(rightAt) === '[2,3,1,2]', JSON.stringify(rightAt));

await pause(700); // the app saves progress after a short delay
const last = saved.at(-1) ?? '';
check('answers were saved through PUT /api/progress (intercepted)', ['q-hook', 'q-tie', 'q-ceil', 'q-stride', 'q-strategy', 'q-barriers', 'q-coarsen'].every((id) => last.includes('ch13/start/' + id)), last.slice(0, 300));
if (out) await page.screenshot({ path: out + '-page.png', fullPage: true });

// ---------------------------------------------------------------- phone width
await page.setViewport({ width: 390, height: 860, deviceScaleFactor: 1 });
await pause(300);
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('no horizontal scroll at 390 px', overflow <= 0, String(overflow));
const figW = await page.$eval('figure svg', (s) => s.getBoundingClientRect().width);
check('the figure shrinks to fit at 390 px', figW > 250 && figW <= 390, String(figW));
if (out) {
  const fig = await page.$('figure');
  await fig.screenshot({ path: out + '-narrow-figure.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
