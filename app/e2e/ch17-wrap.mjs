// Browser check for chapter 17, "Summary and exercises": the carry-forward list, exercise 5's
// JDS kernel loaded from labs/ch17/snippets/spmv_jds.cu with its status and highlighted
// lines, the mixed check (five questions, each answered wrong then right), the map of the
// book's five exercises with links that open written pages, the errata list against the
// notes' "Unclear in the PDF", what was left out, the project status computed from the
// reports, and the pointer to chapter 18. Expected numbers come from src/sim/sparse.ts,
// imported inside the page.
// Usage: APP_URL=http://localhost:<port> node e2e/ch17-wrap.mjs [screenshot-prefix] [--dark]
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
const list = (xs) => '[' + xs.join(', ') + ']';

await goto(page, 'ch17/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const model = await page.evaluate(async () => {
  const m = await import('/src/sim/sparse.ts');
  const j = m.toJDS(m.FIG17_14);
  const e = m.toJDS(m.EXERCISE1);
  const sums = m.spmvRef(m.FIG17_14, Array(6).fill(1), Array(6).fill(0));
  return { row: j.row, iterPtr: j.iterPtr, exRow: e.row, exIter: e.iterPtr, sums, unperm: j.row.map((o) => sums[o]) };
});
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary and exercises", numbered 17.8', h1.includes('Summary and exercises') && (await page.$eval('.page', (e) => e.textContent)).includes('17.8'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'Exercise 5, answered', 'A mixed check', "The book's exercises", 'Where the PDF looks wrong', 'What this chapter left out', 'The project so far'];
check('the page has its seven parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
const carry = await page.$$eval('ul[data-list="carry"] > li', (lis) => lis.map((l) => l.textContent));
check('the carry-forward list has six points: mapping and atomics, coalescing, divergence, amortising, accumulate and -1 pads, data dependence',
  carry.length === 6 && carry[0].includes('decides whether you need atomics') && carry[1].includes('t·stride + r') && carry[2].includes('index array') &&
  carry[3].includes('reuse') && carry[4].includes('column −1') && carry[5].includes('Performance depends on the data'), JSON.stringify(carry.map((c) => c.slice(0, 40))));
check('the carry-forward list cites the book section', (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('§17.8, pp. 421–422'));
const text0 = await body();
check('nothing on the page says it is being written', !/being written/i.test(text0));

// ---- exercise 5's kernel, from the checked file
{
  const file = await page.evaluate(() => fetch('/api/lab-file?path=' + encodeURIComponent('ch17/snippets/spmv_jds.cu')).then((r) => (r.ok ? r.text() : 'MISSING')));
  const lines = file.split('\n');
  let k = 0;
  while (k < lines.length && (lines[k].startsWith('//') || lines[k].trim() === '')) k++;
  const want = lines.slice(k).join('\n');
  await page.waitForFunction(() => !document.querySelector('figure.code pre')?.textContent.startsWith('// loading'), { timeout: 10000 });
  const fig = await page.$eval('figure.code', (f) => ({
    code: f.querySelector('pre').textContent, status: f.querySelector('.tabs .status')?.textContent ?? '', tab: f.querySelector('.tabs button')?.textContent ?? '',
    caption: f.querySelector('figcaption')?.textContent ?? '',
    hl: [...f.querySelectorAll('pre code > span.ln.hl')].map((s) => s.textContent.trim()),
  }));
  check('the JDS kernel is labs/ch17/snippets/spmv_jds.cu without its header, verbatim', file !== 'MISSING' && fig.code.replace(/\s+$/, '') === want.replace(/\s+$/, ''), fig.code.slice(0, 80));
  check('its tab says it is ours, its status uses the fixed vocabulary', fig.tab.includes('(ours)') && fig.status === 'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware', fig.status);
  check('the caption says the book shows no JDS kernel and names the file', fig.caption.includes('The book shows no JDS kernel') && fig.caption.includes('labs/ch17/snippets/spmv_jds.cu'));
  check('three lines are highlighted: the loop test, the index, the store into the original row',
    fig.hl.length === 3 && fig.hl[0].startsWith('for (unsigned int t = 0; t < jdsMatrix.numIters && r <') && fig.hl[1] === 'unsigned int i = jdsMatrix.iterPtr[t] + r;' && fig.hl[2] === 'y[jdsMatrix.row[r]] += sum;', JSON.stringify(fig.hl));
  const per = model.iterPtr.slice(1).map((p, t) => p - model.iterPtr[t]);
  check(`the explanation reads iterPtr ${list(model.iterPtr)} from the model and the threads per iteration ${per.join(', ')}`,
    text0.includes(`iterPtr = ${list(model.iterPtr)}`) && text0.includes(`number ${per.slice(0, -1).join(', ')} and ${per.at(-1)}`));
  await shot('kernel', 'figure.code');
}

// ---- the five questions, each answered wrong then right
check(`no question is answered at the start (${(await page.$$('.q')).length} questions)`, (await page.$$('.q')).length === 5 && (await page.$$('.q .verdict')).length === 0);

const right = model.unperm.filter((v, r) => v === model.sums[r]).length;
let q = await question('the sorted position');
await typeAnswer(q, 0);
check('unpermuted store: 0 is wrong', (await qText(q)).includes('Not quite.'));
await typeAnswer(q, right);
check(`unpermuted store: ${right} is right, with row ${list(model.row)}, the sums ${list(model.sums)} and the defect's ${list(model.unperm)}`,
  right === 1 && (await qText(q)).includes('Correct.') && (await qText(q)).includes(`row = ${list(model.row)}`) &&
  (await qText(q)).includes(list(model.sums)) && (await qText(q)).includes(list(model.unperm)) && (await qText(q)).includes('5 of 6'));

q = await question('a backward pass needs');
await pick(q, 2);
check('transpose: swapping rowPtrs and colIdx is wrong, different lengths and meanings', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('different lengths and different meanings'));
await pick(q, 3);
check('transpose: ELL is not the transpose, slot t·stride + r still belongs to row r', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('still belongs to row r'));
await pick(q, 0);
check('transpose: CSR of A is CSC of Aᵀ; keep a column-oriented copy if it repeats; FlexAttention named',
  (await qText(q)).includes('Correct.') && (await qText(q)).includes('exactly the CSC of Aᵀ') && (await qText(q)).includes('q_indices'));

q = await question('distinct');
await typeAnswer(q, 512);
check('sectors: 512, the request count, is not the distinct count', (await qText(q)).includes('Not quite.'));
await typeAnswer(q, 64);
check('sectors: 64 distinct; ELL also 64; CSR asks 8 times as much', (await qText(q)).includes('Correct.') && (await qText(q)).includes('32 × 2 = 64') && (await qText(q)).includes('asks for 8') && (await qText(q)).includes('lab 17.1'));

q = await question('decode batch');
await typeAnswer(q, 50);
check('ragged batch: 50% is wrong', (await qText(q)).includes('Not quite.'));
await typeAnswer(q, 6.5);
check('ragged batch: 6.5% is right; 8,300 in 128,000; the four fixes named',
  (await qText(q)).includes('Correct.') && (await qText(q)).includes('8,300 real tokens in 64 × 2,000 = 128,000 slots') && (await qText(q)).includes('the hybrid'));

q = await question('top 2 experts');
await pick(q, 1);
check('MoE combine: seq_cst is wrong, the order of the adds stays undefined', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('undefined order'));
await pick(q, 2);
check('MoE combine: sorting the inputs is wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('not the moment each atomic lands'));
await pick(q, 3);
check('MoE combine: double atomics are wrong, rarer is not reproducible', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('Rarer is not reproducible'));
await pick(q, 0);
check('MoE combine: one owner per row, as CSR; the residual is why order matters', (await qText(q)).includes('Correct.') && (await qText(q)).includes('COO against CSR') && (await qText(q)).includes('The residual matters'));

await pause(700); // the app saves 400 ms after the last change
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
const ids = ['q-jds-unpermuted', 'q-transpose', 'q-csr-sectors', 'q-ragged', 'q-moe-combine'].map((s) => 'ch17/wrap/' + s);
check('all five answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 5);
check('the running count of answers appears after answering', (await body()).includes('So far in this chapter you have answered'));
check('the mixed check is labelled as ours', await page.$$eval('.prose > aside.note', (as) => as.some((a) => a.textContent.includes('Not in the book') && a.textContent.includes('expert combine are ours'))));
await shot('questions', '.q');

// ---- the book's exercises
const rows = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) =>
  trs.map((tr) => ({ ex: tr.children[0].textContent.trim(), where: tr.children[2].textContent.trim(), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('the exercise table covers all five exercises', rows.map((r) => r.ex).join(' | ') === '1 | 2 | 3 | 4 | 5', rows.map((r) => r.ex).join(' | '));
const has = (ex, l) => rows.find((r) => r.ex === ex)?.links.includes(l);
check('exercises 1 and 2 point at 17.2, 3 at the start page and 17.5, 4 at the lab and 17.2, 5 at 17.5',
  has('1', '#/ch17/17.2') && has('2', '#/ch17/17.2') && has('3', '#/ch17/start') && has('3', '#/ch17/17.5') && has('4', '#/ch17/labs') && has('4', '#/ch17/17.2') && has('5', '#/ch17/17.5'),
  JSON.stringify(rows.map((r) => [r.ex, r.links])));
check(`exercise 1's JDS arrays are read from the model: row ${list(model.exRow)}, iterPtr ${list(model.exIter)}`,
  rows[0].where.includes(`row = ${list(model.exRow)}`) && rows[0].where.includes(`iterPtr = ${list(model.exIter)}`));
check('the exercise 3 and 4 sketch is on the page', text0.includes('Exercise 3 in three passes') && text0.includes('with += either order works'));
// What the table points at exists on the pages it names.
await goto(page, 'ch17/17.2');
const on172 = await page.evaluate(() => ({
  explorer: !!document.querySelector('section.bench[aria-label="Format explorer"]'),
  ex1: !!document.querySelector('button[data-preset="ex1"]'),
  count: document.querySelector('[data-storage]')?.textContent.includes("Exercise 2's count") ?? false,
  orderQ: [...document.querySelectorAll('.q')].some((q) => q.textContent.includes('launch the COO part first')),
}));
check('page 17.2 has the format explorer, its Exercise 1 matrix, the Exercise 2 count and the order question', on172.explorer && on172.ex1 && on172.count && on172.orderQ, JSON.stringify(on172));
await goto(page, 'ch17/17.5');
const on175 = await page.evaluate(() => ({
  scene: !!document.querySelector('section.scene[aria-label="Routing tokens to experts"]'),
  offset: [...document.querySelectorAll('.q')].some((q) => q.textContent.includes('stable scatter')),
  loop: [...document.querySelectorAll('.q')].some((q) => q.textContent.includes('Exercise 5 asks for a JDS kernel')),
}));
check('page 17.5 has the animation "Routing tokens to experts", the offset question and the loop-test question', on175.scene && on175.offset && on175.loop, JSON.stringify(on175));
await goto(page, 'ch17/start');
check('the start page has the scan question', await page.evaluate(() => [...document.querySelectorAll('.q')].some((q) => q.textContent.includes('exclusive scan'))));
await goto(page, 'ch17/wrap');

// ---- errata
const errata = await page.$$eval('ul[data-list="errata"] > li', (lis) => lis.map((li) => li.textContent));
const want = ['y[row] = sum', 'was calculated in Fig. 17.9', '"row 2"', 'fewer than 10', 'in increasing order', 'Now we examine the ELL format',
  "solution's x elements", 'A parallel SpMV/CSR kernel', 'rowIdxs', 'disadvantages of CSR over COO', "pad slot's column"];
check(`the errata list has ${want.length} items: the notes' nine "Unclear in the PDF" items, the 1% example and the pad column`,
  errata.length === want.length && want.every((w, i) => errata[i].includes(w)), JSON.stringify(errata.map((e) => e.slice(0, 40))));
const gaps = await page.$$eval('ul[data-list="gaps"] > li', (lis) => lis.map((li) => li.textContent));
check('the gaps list names the structs, the include, the 64-byte unit and the missing JDS kernel', gaps.length === 4 &&
  ['CSRMatrix', '#include <cuda/atomic>', 'such as 64-bytes', 'No JDS kernel'].every((w, i) => gaps[i].includes(w)), JSON.stringify(gaps.map((g) => g.slice(0, 40))));
check('the errata are framed as readings of one PDF, checked against vendor documentation', text0.includes('trust your copy') && text0.includes('ch17-facts.md'));
check('what was left out names the solver background, the names, sectioned JDS, CSC, SpMSpV, the COO kernel, exercises 3 and 4, Bell and Garland and the two claims',
  ['pp. 402–403', 'pp. 410, 416', 'p. 417', 'pp. 419–420', 'p. 421', 'Fig. 17.5', 'Exercise 3 is worked', 'reference 3', '(p. 409)', '(p. 422)'].every((s) => text0.includes(s)));

// ---- project status from the reports
const results = (run, id, variant) => (run ? run.results.filter((x) => x.lab === id && x.variant === variant) : []);
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = results(run, id, variant).filter((x) => x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const where = (b) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
const passing = [...Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race'), 'emu'].filter((b) => labStatus(reports[b], 'ch17/lab1_spmv', 'work') === 'pass');
const status = await page.$eval('[data-lab171]', (e) => ({ v: e.dataset.lab171, text: e.textContent }));
check(`project status: lab 17.1 work passes on ${passing.join(', ') || 'nothing'} here`, status.v === (passing.join(',') || 'none') &&
  (passing.length ? status.text.includes(passing.map(where).join(' and ')) : status.text.includes('has not passed on this machine yet') || status.text.includes('No lab results')), JSON.stringify(status));
check('the page says this chapter adds no milestone and why', text0.includes('adds no milestone') && text0.includes('nothing on its main path runs an SpMV'));
check('the page points at chapter 18', await page.$$eval('.prose a', (as) => as.some((a) => a.getAttribute('href') === '#/ch18/start')));

// ---- every link on the page goes to a section that exists (chapter 18 may still be planned)
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
for (const l of links) {
  await goto(page, l.slice(2));
  const st = await page.evaluate(() => ({
    h1: !!document.querySelector('.page h1'),
    writing: document.querySelector('.page').textContent.includes('This section is being written'),
    planned: document.querySelector('.page').textContent.includes('This chapter is planned and not built yet'),
  }));
  if (l.startsWith('#/ch18/') && st.planned) console.log(`note  ${l}: chapter 18 is still planned on this server; the link will open once it is built`);
  else check(`link ${l} opens a written page`, st.h1 && !st.writing && !st.planned, JSON.stringify(st));
}

// ---- margin notes sit after paragraphs, never after a bench or code block
await goto(page, 'ch17/wrap');
const misplaced = await page.$$eval('.prose > aside.note', (as) => as.filter((a) => {
  const prev = a.previousElementSibling;
  return prev && !['P', 'UL', 'OL', 'ASIDE'].includes(prev.tagName);
}).map((a) => `${a.previousElementSibling.tagName}.${a.previousElementSibling.className}`));
check('every margin note follows a paragraph, a list or another note', misplaced.length === 0, misplaced.join(', '));
const stacked = await page.$$eval('.prose > aside.note', (as) => as.filter((a) => a.previousElementSibling?.tagName === 'ASIDE').length);
check('no two margin notes stack after one paragraph', stacked === 0, String(stacked));

await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
check('no sideways page scroll at 390 px', await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
