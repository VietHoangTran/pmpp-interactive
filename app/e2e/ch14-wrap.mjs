// Browser check for chapter 14, "Summary and exercises": the carry-forward list, the kernel of
// the first question (the snippet file itself), the mixed check (five questions, each answered
// wrong then right, with the numbers recomputed here), the map of the book's four exercises with
// links that open written pages, the errata against the notes' "Unclear in the PDF", what was
// left out, and the project status computed from the reports.
// Usage: APP_URL=http://localhost:<port> node e2e/ch14-wrap.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file, and start from an empty one so every question is
// unanswered whatever the reader has already done.
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
const question = async (part) => (await page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(part)), part)).asElement();
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

await goto(page, 'ch14/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('[data-lab143="loading"]'), { timeout: 30000 });
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary and exercises", numbered 14.10', h1.includes('Summary and exercises') && (await page.$eval('.page', (e) => e.textContent)).includes('14.10'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', "The book's exercises", 'Where the PDF looks wrong', 'What this chapter left out', 'The project so far'];
check('the page has its six parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
const carry = await page.$$eval('.prose > ul', (uls) => [...uls[0].children].map((li) => li.textContent));
check('the carry-forward list has seven points, from sort guarantees to "top-k is a selection"', carry.length === 7 &&
  carry[0].includes('an order and a permutation') && carry[3].includes('stable partition') && carry[4].includes('no portable form') && carry[6].includes('Top-k is a selection'), JSON.stringify(carry.map((c) => c.slice(0, 30))));
check('the carry-forward list cites the book\'s summary', (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('§14.10, pp. 346–347'));
const text0 = await body();
check('nothing on the page says it is being written', !/being written/i.test(text0));

// ---- the kernel of the first question is the snippet, function only, with its status
const file = await page.evaluate(() => fetch('/api/lab-file?path=' + encodeURIComponent('ch14/snippets/wrap_blocksort.cu')).then((r) => (r.ok ? r.text() : 'MISSING')));
check('the snippet exists in the repository', file !== 'MISSING' && file.includes('__global__ void blockOddEvenSort'));
await page.waitForSelector('figure.code pre');
const code = await page.$eval('figure.code pre', (e) => e.textContent);
const start = file.indexOf('__global__ void blockOddEvenSort');
const fn = file.slice(start, file.indexOf('\n}', start) + 2);
check('the listing is the kernel from the file, without its header comment', code.trim() === fn.trim() && !code.includes('Chapter 14, "Summary'), code.slice(0, 80));
check('the listing has two barriers, neither inside the step loop (the defect is on the page)', (code.match(/__syncthreads\(\)/g) ?? []).length === 2 &&
  !code.slice(code.indexOf('for (unsigned int step'), code.indexOf('}\n    __syncthreads();\n    for (unsigned int i = t; i < n; i += blockDim.x)\n        keys[i]')).includes('__syncthreads'));
check('the listing\'s status uses the project vocabulary', (await page.$eval('figure.code .tabs .status', (e) => e.textContent)) === 'front-end-checked; emulator-checked; unverified on CUDA and HIP hardware');
check('tier colours: keys global, s shared, indices in registers',
  (await page.$$('figure.code .v-global')).length > 0 && (await page.$$('figure.code .v-shared')).length > 0 && (await page.$$('figure.code .v-reg')).length > 0);
await shot('code', 'figure.code');

// ---- the five questions, each answered wrong then right; expected numbers recomputed here
check('five questions, none answered at the start', (await page.$$('.q')).length === 5 && (await page.$$('.q .verdict')).length === 0);

let q = await question('A test on a GPU sorts 64 random keys');
await pick(q, 0);
check('blocksort: "nothing is wrong" is wrong, element 2t + 2 changes hands between steps', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('element 2t + 2 belongs to thread t + 1'));
await pick(q, 2);
check('blocksort: atomics are wrong, the missing piece is an order between steps', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('an order between steps'));
await pick(q, 3);
check('blocksort: a flag is not needed, n steps sort every test case', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('all 256 lists of eight 0s and 1s'));
await pick(q, 1);
check('blocksort: the barrier per step; the emulator\'s evidence and the warp caveat are given',
  (await qText(q)).includes('Correct.') && (await qText(q)).includes('leaves 64, 129 and 256 keys unsorted in all three thread orders') && (await qText(q)).includes('the language does not promise it'));

q = await question('lets each thread take its key');
await pick(q, 2);
check('atomic scatter: "keys lost" is wrong, atomicAdd gives every thread its own slot', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('every slot is taken exactly once'));
await pick(q, 0);
check('atomic scatter: "sorted, ties unordered" is wrong, later passes undo earlier ones', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('undo the earlier ones'));
await pick(q, 1);
check('atomic scatter: a permutation, not sorted; stability carries each pass into the next', (await qText(q)).includes('Correct.') && (await qText(q)).includes('carries each pass\'s work into the next'));

// 64 rows need ceil(log2 64) = 6 bits of row id; 32 + 6 bits in 8-bit digits.
const rows = 64;
const bits = 32 + Math.ceil(Math.log2(rows));
const passes = Math.ceil(bits / 8);
q = await question('one ordinary LSD radix sort');
await typeAnswer(q, 4);
check('packed rows: 4 passes (one row\'s count) is wrong', (await qText(q)).includes('Not quite.'));
await typeAnswer(q, passes);
check(`packed rows: ${passes} passes over ${bits} bits; the cascade of 14.1; CUB's bit range`, (await qText(q)).includes('Correct.') &&
  (await qText(q)).includes(`ceil(${bits} / 8) = ${passes} passes`) && (await qText(q)).includes("section 14.1's cascade") && (await qText(q)).includes(`begin_bit and end_bit, so a packed 64-bit key like this one can be sorted on its low ${bits} bits`));

const bins = [0, 3, 12, 40, 97];
const k = 50;
let above = 0;
let bin = 0;
while (above + bins[bin] < k) above += bins[bin++];
q = await question('A radix select looks for the 50 largest');
await typeAnswer(q, bins[bin]);
check(`select: ${bins[bin]} (the whole bin) is wrong`, (await qText(q)).includes('Not quite.'));
await typeAnswer(q, k - above);
const totals = bins.slice(0, bin + 1).map((_, b) => bins.slice(0, b + 1).reduce((x, y) => x + y, 0)).join(', ');
check(`select: ${k - above} keys still to pick from bin ${bin + 1}; running totals ${totals}; llama.cpp named`, (await qText(q)).includes('Correct.') &&
  (await qText(q)).includes(`Running totals from the top: ${totals}`) && (await qText(q)).includes(`${above} of them`) && (await qText(q)).includes('top_k_radix_cuda'));

q = await question('already hold 0.95 of the probability');
await pick(q, 2);
check('top-p: token order is wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('first tokens by id'));
await pick(q, 3);
check('top-p: skipping the check is wrong on a flat row', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('flat row'));
await pick(q, 1);
check('top-p: select, sort the candidates, check the sum; llama.cpp\'s host top-p named', (await qText(q)).includes('Correct.') && (await qText(q)).includes('partially sorts the top 256'));

await pause(700); // the app saves 400 ms after the last change
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
const ids = ['q-blocksort', 'q-atomic-scatter', 'q-packed-rows', 'q-select-bin', 'q-topp-head'].map((s) => 'ch14/wrap/' + s);
check('all five answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 5);
check('the running count of answers appears after answering', (await body()).includes('So far in this chapter you have answered'));
await shot('questions', '.q');

// ---- the book's exercises
const exRows = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) =>
  trs.map((tr) => ({ ex: tr.children[0].textContent.trim(), where: tr.children[2].textContent.trim(), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('the exercise table covers all four exercises', exRows.map((r) => r.ex).join(' | ') === '1 | 2 | 3 | 4', exRows.map((r) => r.ex).join(' | '));
check('exercises 1 to 3 point at 14.6, 14.7, 14.8 and lab 14.2; exercise 4 at 14.3 and chapter 13\'s labs',
  exRows[0].links.includes('#/ch14/14.6') && exRows[0].links.includes('#/ch14/labs') && exRows[1].links.includes('#/ch14/14.7') && exRows[1].where.includes('lab 14.2') &&
  exRows[2].links.includes('#/ch14/14.8') && exRows[2].where.includes('lab 14.2') && exRows[3].links.includes('#/ch14/14.3') && exRows[3].links.includes('#/ch13/labs'), JSON.stringify(exRows));
// What the table points at exists.
await goto(page, 'ch14/14.3');
const on143 = await page.evaluate(() => [...document.querySelectorAll('.prose h2')].some((h) => h.textContent.trim() === 'Exercise 4, in outline'));
check('page 14.3 has "Exercise 4, in outline"', on143);
await goto(page, 'ch14/14.6');
const on146 = await page.evaluate(() => ({
  warps: !!document.querySelector('section.bench[aria-label="Where a warp\'s writes land"]'),
  table: !!document.querySelector('section.bench[aria-label="The bucket table"]'),
}));
check('page 14.6 has the warp explorer and the bucket table', on146.warps && on146.table, JSON.stringify(on146));
await goto(page, 'ch14/14.7');
check('page 14.7 has the radix-width explorer', await page.evaluate(() => !!document.querySelector('section.bench[aria-label="Radix width"]')));
await goto(page, 'ch14/14.8');
check('page 14.8 has Fig. 14.13 and the keys-per-thread explorer', await page.evaluate(() => !!document.querySelector('section.bench[aria-label="Fig. 14.13, two ways"]') && !!document.querySelector('section.bench[aria-label="Keys per thread"]')));
await goto(page, 'ch14/wrap');
await page.waitForFunction(() => !document.querySelector('[data-lab143="loading"]'), { timeout: 30000 });

// ---- errata: every "Unclear in the PDF" item of notes/ch14.md, in order
const errata = await page.$$eval('ul[data-list="errata"] > li', (lis) => lis.map((li) => li.textContent));
const want = ['will be discussed further in Chapter 12', 'transposition sort and sampling sort', 'for keys mapping to the zero bucket', 'N + 1 entries',
  'indeed to right destination indices', 'gridExclusiveScan', 'until no elements are swapped', 'Fig. 14.13', 'as seen in Chapter 13'];
check(`the errata list has the notes' ${want.length} items, in order`, errata.length === want.length && want.every((w, i) => errata[i].includes(w)), JSON.stringify(errata.map((e) => e.slice(0, 40))));
check('the inferred items (6 to 9) are marked as our reading', errata.slice(5).every((e) => e.startsWith('Our reading')) && errata.slice(0, 5).every((e) => !e.startsWith('Our reading')));
const gaps = await page.$$eval('ul[data-list="gaps"] > li', (lis) => lis.map((li) => li.textContent));
check('the gaps list names the unsigned wrap and the plain flag store', gaps.length === 2 && gaps[0].includes('i < N - 1') && gaps[1].includes('hasChanged'), JSON.stringify(gaps));
check('the errata are framed as readings of one PDF', text0.includes('trust your copy'));
const left = await page.$$eval('ul[data-list="left-out"] > li', (lis) => lis.map((li) => li.textContent));
check('what was left out names Cole, exercise 4, Batcher, sample sort, MSD, Thrust, the one-kernel pass and the references',
  left.length === 8 && ['Cole', 'exercise 4', 'Batcher', 'Sample sort', 'MSD radix sort', 'Thrust', 'cooperative launch', 'references'].every((w, i) => left[i].includes(w)), JSON.stringify(left.map((l) => l.slice(0, 30))));

// ---- project status from the reports
const results = (run, id, variant) => (run ? run.results.filter((x) => x.lab === id && x.variant === variant) : []);
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = results(run, id, variant).filter((x) => x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const where = (b) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
const passingOn = (lab) => [...Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race'), 'emu'].filter((b) => labStatus(reports[b], lab, 'work') === 'pass');
const own = passingOn('ch14/lab3_sortrows');
const status = await page.$eval('[data-lab143]', (e) => ({ v: e.dataset.lab143, text: e.textContent }));
check(`project status: lab 14.3 work passes on ${own.join(', ') || 'nothing'} here`, status.v === (own.join(',') || 'none') &&
  (own.length ? status.text.includes(own.map(where).join(' and ')) : status.text.includes('does not pass on any backend') || status.text.includes('No lab results')), JSON.stringify(status));
const m5 = await page.$$eval('table[data-table="m5"] tbody tr', (trs) => trs.map((tr) => ({ num: tr.dataset.m5, pass: tr.dataset.pass, href: tr.querySelector('a')?.getAttribute('href') })));
const M5 = [['9.3', 'ch09/lab3_tokencounts', '#/ch09/labs'], ['11.6', 'ch11/lab6_topp', '#/ch11/labs'], ['12.4', 'ch12/lab4_candidates', '#/ch12/labs'], ['14.3', 'ch14/lab3_sortrows', '#/ch14/labs']];
check('the M5 table has its four pieces, each linked to its labs page, with the reader\'s status from the reports', m5.length === 4 &&
  M5.every(([num, lab, href], i) => m5[i].num === num && m5[i].href === href && m5[i].pass === (passingOn(lab).join(',') || 'none')), JSON.stringify(m5));
check('the page says what is left of M5 and points at chapter 15', text0.includes('module X1 joins these pieces') && (await page.$$eval('.prose a', (as) => as.some((a) => a.getAttribute('href') === '#/ch15'))));

// ---- every link on the page goes to a page that exists
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
for (const l of links) {
  await goto(page, l.slice(2));
  const ok = await page.evaluate(() => !!document.querySelector('.page h1') && !document.querySelector('.page').textContent.includes('This section is being written'));
  check(`link ${l} opens a written page`, ok);
}

// ---- margin notes sit after paragraphs, never after a bench, a table or code
await goto(page, 'ch14/wrap');
const misplaced = await page.$$eval('.prose > aside.note', (as) => as.filter((a) => {
  const prev = a.previousElementSibling;
  return prev && !['P', 'UL', 'OL', 'ASIDE'].includes(prev.tagName);
}).map((a) => `${a.previousElementSibling.tagName}.${a.previousElementSibling.className}`));
check('every margin note follows a paragraph, a list or another note', misplaced.length === 0, misplaced.join(', '));

await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
check('no sideways page scroll at 390 px', await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
if (out) await page.screenshot({ path: `${out}-narrow.png` });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
