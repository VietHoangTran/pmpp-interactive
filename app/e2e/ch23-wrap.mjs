// Browser check for chapter 23, "Summary and exercises": the carry-forward list, the mixed
// check (one host sequence shown from a checked snippet, five questions, the pipeline number
// computed from the stepper's model), the map of the book's exercises, the errata and dated
// lists, what was left out, the project status computed from the reports, and where next.
// Usage: APP_URL=http://localhost:<port> node e2e/ch23-wrap.mjs [screenshot-prefix] [--dark]
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
const question = (part) => page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.replace(/\s+/g, ' ').includes(part)), part);
const qText = async (q) => (await q.evaluate((e) => e.textContent)).replace(/\s+/g, ' ');
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

await goto(page, 'ch23/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
// The pipeline question's numbers, from the model the stepper on page 23.3 runs on.
const M = await page.evaluate(async () => {
  const { DEFAULTS, simulate } = await import('/src/sim/chunkpipe.ts');
  const s = simulate({ ...DEFAULTS, a: 32, b: 8, c: 32, n: 8, streams: 8, engines: 2, order: 'depth', overhead: 0, host: 'pinned', queues: Infinity, view: 'cuda' });
  return { total: s.total, serial: s.serial, bubble: s.bubble };
});
console.log('note  from the model: ' + JSON.stringify(M));
const body = async () => (await page.$eval('.prose', (e) => e.textContent)).replace(/\s+/g, ' ');

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary and exercises", numbered 23.6', h1.includes('Summary and exercises') && (await page.$eval('.page', (e) => e.textContent)).includes('23.6'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', 'The book’s exercises', 'Where the PDF looks wrong', 'What this chapter left out', 'The project so far', 'Where next'];
check('the page has its seven parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
check('the carry-forward list has seven points and cites §23.6, pp. 574–575',
  (await page.$$eval('ul[data-carry] > li', (l) => l.length)) === 7 &&
  (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('§23.6') && (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('574–575'));
const text0 = await body();
check('nothing on the page says it is being written', !/being written/i.test(text0));

// ---- the sequence for the event question is the checked snippet
const file = await page.evaluate(() => fetch('/api/lab-file?path=' + encodeURIComponent('ch23/snippets/rerecord.cu')).then((r) => (r.ok ? r.text() : 'MISSING')));
const m = /\/\/ \[shown\]\n([\s\S]*?)\n[ \t]*\/\/ \[\/shown\]/.exec(file);
const region = m ? m[1].split('\n').map((l) => l.replace(/^ {4}/, '')).join('\n').trim() : '';
const shownCode = await page.$eval('figure.code pre', (e) => e.textContent);
check('the listing is labs/ch23/snippets/rerecord.cu between its markers, line for line', file !== 'MISSING' && region && shownCode.trim() === region, shownCode.slice(0, 80));
check('the listing is labelled as not a lab file and front-end-checked, never run',
  (await page.$eval('figure.code .tabs', (e) => e.textContent)).includes('not a lab file') &&
  (await page.$eval('figure.code .tabs .status', (e) => e.textContent)) === 'front-end-checked; never run');
check('x and y are coloured as global memory', (await page.$$('figure.code .v-global')).length >= 4);
check('the Beyond note says the sequence, the decode loop and the pipeline are ours, and the sequence never ran',
  await page.$$eval('.prose > aside.note', (as) => as.some((a) => a.textContent.includes('are ours') && a.textContent.includes('never been run'))));
await shot('code', 'figure.code');

// ---- the five questions, each answered wrong then right
check(`five questions, none answered at the start`, (await page.$$('.q')).length === 5 && (await page.$$('.q .verdict')).length === 0);

let q = await question('consumeC is meant to');
await pick(q, 0);
check('re-record: "done was last recorded after produceB" is wrong: the wait took done as it stood', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('The wait was issued on line 3'));
await pick(q, 3);
check('re-record: "recording twice is an error" is wrong: a new record overwrites', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('a new record overwrites'));
await pick(q, 1);
check('re-record: the wait took done after produceA is right, and the fix is to wait after the record', (await qText(q)).includes('Correct.') && (await qText(q)).includes('Issue the wait after the record it needs'));

q = await question('A decode loop makes one token per step');
await pick(q, 0);
check('decode: a higher priority is wrong: nothing is queued while the host waits', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('has nothing to prefer'));
await pick(q, 1);
check('decode: queue step n + 1 first is right, and mlx-lm is named', (await qText(q)).includes('Correct.') && (await qText(q)).includes('async_eval'));

q = await question('Two GPUs each hold half');
await typeAnswer(q, M.serial);
check(`pipeline: ${M.serial} (one after another) is wrong and the answer ${M.total} is given`, (await qText(q)).includes(`Not quite. The answer is ${M.total} units.`), await qText(q));
await typeAnswer(q, M.total);
check(`pipeline: ${M.total} is right, against ${M.serial}, with a bubble of ${M.bubble}`,
  (await qText(q)).includes('Correct.') && (await qText(q)).includes(`against ${M.serial} one after another`) && (await qText(q)).includes(`idles ${M.bubble} units`));
check('the pipeline prompt states the numbers the model was given (8 micro-batches, 4, 1, 4)', (await qText(q)).includes('cut into 8 micro-batches') && (await qText(q)).includes('takes 4 time units on GPU 0, 1 unit to cross the link'));

q = await question('moved 4,000 bytes');
await typeAnswer(q, 2);
check('exercise 2: 2 is wrong and 4 bytes is given', (await qText(q)).includes('Not quite. The answer is 4 bytes.'), await qText(q));
await typeAnswer(q, 4);
check('exercise 2: 4 is right; count, rank and tag are named', (await qText(q)).includes('Correct.') && (await qText(q)).includes('2,000 is the destination rank and 4 the tag'));

q = await question('must be at least 128 bytes');
await pick(q, 2);
check('exercise 3: the 128-byte minimum is wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('single 4-byte float'));
await pick(q, 3);
check('exercise 3: both (a) and (b) is right, and the page says it reads two right answers', (await qText(q)).includes('Correct.') && (await qText(q)).includes('two right answers'));

await pause(700); // the app saves 400 ms after the last change
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
const ids = ['q-rerecord', 'q-decode', 'q-pp', 'q-ex2', 'q-ex3'].map((s) => 'ch23/wrap/' + s);
check('all five answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 5);
check('the running count of answers appears after answering', (await body()).includes('So far in this chapter you have answered'));
await shot('questions', '.q');

// ---- the book's exercises
const rows = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) =>
  trs.map((tr) => ({ ex: tr.children[0].textContent.trim(), where: tr.children[2].textContent.replace(/\s+/g, ' '), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('the exercise table covers all three exercises, exercise 1 in its parts', rows.map((r) => r.ex).join(' | ') === '1a–1d | 1e | 2 | 3', rows.map((r) => r.ex).join(' | '));
check('exercise 1 points at the start page; 2 and 3 are on this page',
  rows[0].links.includes('#/ch23/start') && rows[1].links.includes('#/ch23/start') && rows[2].where.includes('Book exercise 2') && rows[3].where.includes('Book exercise 3'), JSON.stringify(rows));
await goto(page, 'ch23/start');
{
  const t = await page.$eval('.prose', (e) => e.textContent.replace(/\s+/g, ' '));
  check('the start page has the explorer preset "Exercise 1 (p. 576)" and the question "Book exercise 1e"',
    !!(await page.$('[data-preset="exercise1"]')) && t.includes('Exercise 1 (p. 576)') && t.includes('Book exercise 1e'));
}
await goto(page, 'ch23/wrap');

// ---- errata and dated statements
const errata = await page.$$eval('ul[data-list="errata"] > li', (lis) => lis.map((li) => li.textContent.replace(/\s+/g, ' ')));
const want = ['five essential MPI functions', 'line 19', 'line 32', 'cudaMemset operation', 'l2Norm', 'launch_jacobi_kernel', 'thick dotted', 'periodic boundaries',
  '5-point stencil', 'squared residues', 'starting value is never shown', 'Fig. 23.13(b)', 'edge columns', 'responsible these rows', 'lower number is a higher priority'];
check(`the errata list has ${want.length} items covering the notes' "Unclear in the PDF"`, errata.length === want.length &&
  want.every((w, i) => errata[i].includes(w)), JSON.stringify(errata.map((e) => e.slice(0, 40))));
const dated = await page.$$eval('ul[data-list="dated"] > li', (lis) => lis.map((li) => li.textContent.replace(/\s+/g, ' ')));
const wantDated = ['cudaMemset is blocking', 'requires', 'does not preempt', 'MVAPICH-Plus', '2.29', 'NVSHMEMX_INIT_ATTR_INITIALIZER', 'CUB link'];
check('the dated list: memset, "requires" pinned, priorities, CUDA-aware MPI, NCCL one-sided, setup, references',
  dated.length === wantDated.length && wantDated.every((w, i) => dated[i].includes(w)), JSON.stringify(dated.map((e) => e.slice(0, 40))));
check('the errata are framed as readings of one PDF, and the dated list as checked against vendor documentation',
  text0.includes('trust your copy') && text0.includes('ch23-facts.md') && text0.includes('ignores them by default'));
check('what was left out names sections 23.2, 23.4 and 23.5 with pages, and the repository of the book\'s own code',
  text0.includes('section 23.2, pp. 545–553') && text0.includes('section 23.4, pp. 563–568') && text0.includes('section 23.5, pp. 568–574') && text0.includes('Multi GPU programming models'));

// ---- project status from the reports
const results = (run, id, variant) => (run ? run.results.filter((x) => x.lab === id && x.variant === variant) : []);
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = results(run, id, variant).filter((x) => x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const where = (b) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
const passing = [...Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race'), ...(reports.emu ? ['emu'] : [])].filter((b) => labStatus(reports[b], 'ch23/lab1_overlap', 'work') === 'pass');
const status = await page.$eval('[data-project]', (e) => ({ v: e.dataset.project, text: e.textContent.replace(/\s+/g, ' ') }));
check(`project status: lab 23.1 work passes on ${passing.join(', ') || 'nothing'} here`, status.v === (passing.join(',') || 'none') &&
  (passing.length ? status.text.includes(passing.map(where).join(' and ')) : status.text.includes('has not passed on this machine yet') || status.text.includes('No lab results')), JSON.stringify(status));
check('the page says this chapter adds no milestone and points at chapter 24',
  status.text.includes('adds no milestone') && (await page.$$eval('.prose a', (as) => as.some((a) => a.getAttribute('href') === '#/ch24/start'))));

// ---- every link on the page goes to a section that exists
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
for (const l of links) {
  await goto(page, l.slice(2));
  const ok = await page.evaluate(() => !!document.querySelector('.page h1') && !document.querySelector('.page').textContent.includes('being written'));
  check(`link ${l} opens a written page`, ok);
}

// ---- margin notes sit after paragraphs, never after a bench or code block
await goto(page, 'ch23/wrap');
const misplaced = await page.$$eval('.prose > aside.note', (as) => as.filter((a) => {
  const prev = a.previousElementSibling;
  return prev && !['P', 'UL', 'OL', 'ASIDE'].includes(prev.tagName);
}).map((a) => `${a.previousElementSibling.tagName}.${a.previousElementSibling.className}`));
check('every margin note follows a paragraph, a list or another note', misplaced.length === 0, misplaced.join(', '));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
