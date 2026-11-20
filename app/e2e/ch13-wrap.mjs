// Browser check for chapter 13, "Summary and exercises": the carry-forward list, the mixed
// check (five questions, each answered wrong then right), the map of the book's four
// exercises with links that open written pages, the errata list against the notes' "Unclear
// in the PDF", what was left out, the project status computed from the reports, and the
// link to chapter 14.
// Usage: APP_URL=http://localhost:<port> node e2e/ch13-wrap.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file, and start from an empty one so every
// question is unanswered whatever the reader has already done.
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
const saved = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
  if (r.method() === 'PUT') saved.push(r.postData() ?? '');
  return r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : JSON.stringify(EMPTY) });
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const shot = async (name, sel) => {
  if (!out) return;
  await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};
const question = (part) => page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(part)), part);
const qText = (q) => q.evaluate((e) => e.textContent);
const typeAnswer = async (q, v) => {
  const input = await q.$('input[type=number]');
  await input.focus();
  await input.evaluate((e) => e.select());
  await input.type(String(v));
  await (await q.$('button.primary')).click();
  await pause(120);
};
const pick = async (q, index) => {
  await (await q.$$('input[type=radio]'))[index].click();
  await (await q.$('button.primary')).click();
  await pause(120);
};
const n = (v) => v.toLocaleString('en-US');

await goto(page, 'ch13/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary and exercises", numbered 13.9', h1.includes('Summary and exercises') && (await page.$eval('.page', (e) => e.textContent)).includes('13.9'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', "The book's exercises", 'Where the PDF looks wrong', 'What this chapter left out', 'The project so far'];
check('the page has its six parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
check('the carry-forward list names the four tiled merges and says llama.cpp\'s Metal merge does not stage',
  (await page.$$eval('.prose > ul', (uls) => uls[0].children[4].textContent)).includes("CUB, rocPRIM, MLX, PyTorch MPS; llama.cpp's Metal merge does not stage at all"));
check('the carry-forward list has six points and cites the book section',
  (await page.$$eval('.prose > ul', (uls) => uls[0].children.length)) === 6 && (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('§13.9, p. 327'));
const text0 = await body();
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('no kernel listing on the page (no snippet needed)', (await page.$$('figure.code')).length === 0);

// ---- the five questions, each answered wrong then right; expected numbers recomputed here
check(`no question is answered at the start (${(await page.$$('.q')).length} questions)`, (await page.$$('.q')).length === 5 && (await page.$$('.q .verdict')).length === 0);

let q = await question('splits a merge by input');
await typeAnswer(q, 500);
check('input split: 500, the even share, is not the busiest thread', (await qText(q)).includes('Not quite.'), await qText(q));
await typeAnswer(q, 1000 / 4 + 1000);
check('input split: 1,250 is right, against 500 for an output split', (await qText(q)).includes('Correct.') && (await qText(q)).includes('1,250 outputs') && (await qText(q)).includes('gives every thread 500'));

q = await question('sorted largest first');
await pick(q, 2);
check('descending: reversing in and out is wrong; each list keeps its order (reversed twice) but B now wins ties',
  (await qText(q)).includes('Not quite.') && (await qText(q)).includes('each list keeps its own order, since it is reversed twice') &&
  (await qText(q)).includes("B's element now comes first") && !(await qText(q)).includes('the later element now comes first'));
await pick(q, 0);
check('descending: flipping only the merge is wrong, the search still looks for an ascending split', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('ascending merge'));
await pick(q, 1);
check('descending: every comparison flips and keeps its strictness, operators spelled out; llama.cpp named',
  (await qText(q)).includes('Correct.') && (await qText(q)).includes('A[i-1] < B[j] and B[j-1] <= A[i] in the search') && (await qText(q)).includes('llama.cpp'));

const topk = { k: 50, per: 8, block: 128 };
q = await question('you want only the first');
await typeAnswer(q, topk.block);
check('top-k: all 128 threads is wrong', (await qText(q)).includes('Not quite.'));
await typeAnswer(q, Math.ceil(topk.k / topk.per));
check(`top-k: ${Math.ceil(topk.k / topk.per)} threads search; thread 6 owns outputs 48 and 49; the rest return`,
  (await qText(q)).includes('Correct.') && (await qText(q)).includes('outputs 48 and 49') && (await qText(q)).includes('Threads 7 to 127'));

const ex4 = { total: 1030400 + 608000, per: 8, block: 1024 };
const threads = ex4.total / ex4.per;
q = await question("exercise 4");
await pick(q, 1);
check('exercise 4: every thread searching global memory in the tiled kernel is wrong (only thread 0)', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('only thread 0'));
await pick(q, 3);
check('exercise 4: only 200 threads searching shared memory is wrong (lines 41, 43 and 48)', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('Lines 41 and 43'));
await pick(q, 0);
check(`exercise 4: ${n(threads)}; ${n(threads / ex4.block)} and ${n(threads)}`,
  (await qText(q)).includes('Correct.') && (await qText(q)).includes(`${n(threads)}; then ${n(threads / ex4.block)} and ${n(threads)}`) && (await qText(q)).includes(`${n(ex4.total)} outputs`));

q = await question('runs about as fast at 8 outputs per thread');
await pick(q, 1);
check('cache: "the count is wrong" is wrong, lane t stores to C[8t + s]', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('C[8t + s]'));
await pick(q, 3);
check('cache: coalescing across warps is wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes("one warp's instruction"));
await pick(q, 0);
check('cache: each lane fills its own segment over eight steps; predict, then measure', (await qText(q)).includes('Correct.') && (await qText(q)).includes('Predict from the model'));

await pause(700); // the app saves 400 ms after the last change
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
const ids = ['q-input-split', 'q-descending', 'q-topk-threads', 'q-ex4', 'q-cache'].map((s) => 'ch13/wrap/' + s);
check('all five answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 5);
check('the running count of answers appears after answering', (await body()).includes('So far in this chapter you have answered'));
await shot('questions', '.q');

// ---- the book's exercises
const rows = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) =>
  trs.map((tr) => ({ ex: tr.children[0].textContent.trim(), where: tr.children[2].textContent.trim(), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('the exercise table covers all four exercises', rows.map((r) => r.ex).join(' | ') === '1 | 2 | 3 | 4a, 4b, 4c', rows.map((r) => r.ex).join(' | '));
check('exercises 1 to 3 point at page 13.3, exercise 3 also at the lab, exercise 4 at the question above',
  ['1', '2', '3'].every((e) => rows.find((r) => r.ex === e).links.includes('#/ch13/13.3')) && rows.find((r) => r.ex === '3').links.includes('#/ch13/labs') &&
  rows.find((r) => r.ex === '4a, 4b, 4c').where.includes('"Book exercise 4" question above'), JSON.stringify(rows.map((r) => [r.ex, r.links])));
check('the exercise 3 sketch for the book\'s loop is on the page', text0.includes('Exercise 3 in the book\'s own loop') && text0.includes('so line 48 goes'));
// What the table points at exists: the explorer and the animation on 13.3, the lab's part (b).
await goto(page, 'ch13/13.3');
const on133 = await page.evaluate(() => ({
  explorer: !!document.querySelector('section.bench[aria-label="Co-rank explorer"]'),
  scene: !!document.querySelector('section.scene[aria-label="Searching for the split"]'),
  exact: [...document.querySelectorAll('.prose h2')].some((h) => h.textContent.trim() === 'Load exactly what a tile uses'),
  part5: [...document.querySelectorAll('section.scene[aria-label="Searching for the split"] .seg button')].some((b) => b.textContent.includes('5. Every thread at once')),
}));
check('page 13.3 has the co-rank explorer, "Searching for the split" with its part 5, and "Load exactly what a tile uses"', on133.explorer && on133.scene && on133.exact && on133.part5, JSON.stringify(on133));
await goto(page, 'ch13/wrap');

// ---- errata
const errata = await page.$$eval('ul[data-list="errata"] > li', (lis) => lis.map((li) => li.textContent));
const want = ['on the right Fig. 13.2', 'i = 4, j = 2', 'A[i_t] to A[i_t + 1]', 'i_low = max(0,k-m)', 'In Fig. 13.7, for iteration 0', 'thread 0',
  'A[0], A[2], and B[0]', 'much smaller', 'ceil of an integer quotient', 'line 48', 'line 16', 'start at 0', 'line 37', 'no __syncthreads()',
  'refill reads', 'j_m_1_cir', 'merge_sequetial_circular'];
check(`the errata list has ${want.length} items covering the notes' 18 "Unclear in the PDF" items`, errata.length === want.length &&
  want.every((w, i) => errata[i].includes(w)), JSON.stringify(errata.map((e) => e.slice(0, 40))));
check('our own inferences are marked as ours (line 48, the refill index)', errata[9].includes('Our own emulation') && errata[14].includes('our own emulation'));
const gaps = await page.$$eval('ul[data-list="gaps"] > li', (lis) => lis.map((li) => li.textContent));
check('the gaps list names __device__, k_curr, the shared-memory size and integer min', gaps.length === 4 &&
  ['__device__', 'k_curr', 'third launch argument', 'min'].every((w, i) => gaps[i].includes(w)), JSON.stringify(gaps.map((g) => g.slice(0, 40))));
check('the errata are framed as readings of one PDF, checked against vendor documentation', text0.includes('trust your copy') && text0.includes('ch13-facts.md'));
check('what was left out names §13.7 with its pages and says why', text0.includes('pp. 321–327') && text0.includes('no library or project read for this course uses one'));
check('what was left out names the tiled walkthrough, Observation 1, the trace, Hadoop and set operations',
  text0.includes('pp. 316–320') && text0.includes('Observation 1') && text0.includes('(p. 306)') && text0.includes('Hadoop (p. 305)') && text0.includes('set intersection and union (p. 303)'));

// ---- project status from the reports
const results = (run, id, variant) => (run ? run.results.filter((x) => x.lab === id && x.variant === variant) : []);
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = results(run, id, variant).filter((x) => x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const where = (b) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
const passing = [...Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race'), 'emu'].filter((b) => labStatus(reports[b], 'ch13/lab1_merge', 'work') === 'pass');
const status = await page.$eval('[data-lab131]', (e) => ({ v: e.dataset.lab131, text: e.textContent }));
check(`project status: lab 13.1 work passes on ${passing.join(', ') || 'nothing'} here`, status.v === (passing.join(',') || 'none') &&
  (passing.length ? status.text.includes(passing.map(where).join(' and ')) : status.text.includes('has not passed on this machine yet') || status.text.includes('No lab results')), JSON.stringify(status));
check('the page says this chapter adds no milestone, why, and what the sampling module can reuse',
  text0.includes('adds no milestone') && text0.includes('merges two sorted lists') && text0.includes('total mode'));
check('the page points at chapter 14', await page.$$eval('.prose a', (as) => as.some((a) => a.getAttribute('href') === '#/ch14/start')));

// ---- every link on the page goes to a section that exists
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
for (const l of links) {
  await goto(page, l.slice(2));
  const ok = await page.evaluate(() => !!document.querySelector('.page h1') && !document.querySelector('.page').textContent.includes('This section is being written'));
  check(`link ${l} opens a written page`, ok);
}

// ---- margin notes sit after paragraphs, never after a bench or code block
await goto(page, 'ch13/wrap');
const misplaced = await page.$$eval('.prose > aside.note', (as) => as.filter((a) => {
  const prev = a.previousElementSibling;
  return prev && !['P', 'UL', 'OL', 'ASIDE'].includes(prev.tagName);
}).map((a) => `${a.previousElementSibling.tagName}.${a.previousElementSibling.className}`));
check('every margin note follows a paragraph, a list or another note', misplaced.length === 0, misplaced.join(', '));

await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
check('no sideways page scroll at 390 px', await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
