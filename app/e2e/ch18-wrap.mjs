// Browser check for chapter 18, "Summary and exercises": the carry-forward list, the mixed
// check (five questions, each answered wrong then right), the map of the book's three
// exercises with links that open written pages and the interactives they name, Exercise 1's
// answers against the models, the errata list against the notes' "Unclear in the PDF", what
// was left out, the project status computed from the reports, and the pointer to chapter 19.
// Expected numbers come from src/sim/bfs.ts and src/sim/gridbfs.ts, imported inside the page.
// Usage: APP_URL=http://localhost:<port> node e2e/ch18-wrap.mjs [screenshot-prefix] [--dark]
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
// Choice options may be shuffled: pick by the text of the option's label.
const pickText = async (q, part) => {
  const radios = await q.$$('label');
  for (const l of radios) {
    if ((await l.evaluate((e) => e.textContent)).includes(part)) {
      await (await l.$('input[type=radio]')).click();
      await (await q.$('button.primary')).click();
      await pause(120);
      return true;
    }
  }
  return false;
};
const list = (xs) => '[' + xs.join(', ') + ']';

await goto(page, 'ch18/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const model = await page.evaluate(async () => {
  const b = await import('/src/sim/bfs.ts');
  const g = await import('/src/sim/gridbfs.ts');
  const csr = b.toCSR(b.EX1);
  const rows = Array.from({ length: b.EX1.n }, (_, v) => b.outNeighbours(csr, v));
  const run = (k) => b.runBFS(b.EX1, { variant: k, root: 0, blockSize: 4, warpWidth: 4 }).calls;
  const tree = g.treeSummary(16, 5);
  const grid = g.gridSummary(1000);
  const stride = g.strideAssignment(2500, 1024);
  const max = Math.max(...stride.map((s) => s.length));
  return {
    csr, rows, tree, grid: { vertices: grid.vertices, launches: grid.launches, maxFrontier: grid.maxFrontier },
    pushLoop: run('push').map((c) => c.pastCheck), pullLoop: run('pull').map((c) => c.pastCheck), pullLabel: run('pull').map((c) => c.labelsMax),
    edgeSrc: run('edge').map((c) => c.pastCheck), edgeLabel: run('edge').map((c) => c.labelsMax), frontier: run('frontier').map((c) => c.items),
    strideMax: max, strideThree: stride.filter((s) => s.length === max).length,
  };
});
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary and exercises", numbered 18.9', h1.includes('Summary and exercises') && (await page.$eval('.page', (e) => e.textContent)).includes('18.9'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', "The book's exercises", 'Where the PDF looks wrong', 'What this chapter left out', 'The project so far'];
check('the page has its six parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
const carry = await page.$$eval('ul[data-list="carry"] > li', (lis) => lis.map((l) => l.textContent));
check('the carry-forward list has six points: push and pull, wasted threads, work lists and CAS, privatisation, launches, uneven work',
  carry.length === 6 && carry[0].includes('who does the scanning') && carry[1].includes('wasted threads') && carry[2].includes('compare-and-swap') &&
  carry[3].includes('Privatise') && carry[4].includes('resident') && carry[5].includes('bucket'), JSON.stringify(carry.map((c) => c.slice(0, 40))));
check('the carry-forward list cites the book section', (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('§18.9, p. 449'));
const text0 = await body();
check('nothing on the page says it is being written', !/being written/i.test(text0));

// ---- the five questions, each answered wrong then right
check(`no question is answered at the start (${(await page.$$('.q')).length} questions)`, (await page.$$('.q')).length === 5 && (await page.$$('.q .verdict')).length === 0);

let q = await question('children, 5 levels deep');
await typeAnswer(q, 5);
check('tree launches: 5 is wrong (the last launch finds nothing)', (await qText(q)).includes('Not quite.'));
await typeAnswer(q, model.tree.launches);
check(`tree launches: ${model.tree.launches} is right, against the grid's ${model.grid.launches.toLocaleString('en-US')} launches`,
  model.tree.launches === 6 && (await qText(q)).includes('Correct.') && (await qText(q)).includes(`takes ${model.grid.launches.toLocaleString('en-US')}`) &&
  (await qText(q)).includes(model.tree.maxFrontier.toLocaleString('en-US')) && (await qText(q)).includes(model.tree.vertices.toLocaleString('en-US')));

q = await question('repetition-penalty kernel');
check('exchange: wrong answer "not atomic" explained', (await pickText(q, 'is not atomic on device memory')) && (await qText(q)).includes('Not quite.') && (await qText(q)).includes('atomic read-modify-write'));
check('exchange: wrong answer "both get UNVISITED" explained', (await pickText(q, 'could both get UNVISITED back')) && (await qText(q)).includes('Only the first sees UNVISITED'));
check('exchange: wrong answer "only faster" explained with a back edge', (await pickText(q, 'compare-and-swap is chosen only because it is faster')) && (await qText(q)).includes('7 → 0'));
check('exchange: the right answer is the unconditional write', (await pickText(q, 'An exchange always writes')) && (await qText(q)).includes('Correct.') && (await qText(q)).includes('the condition and the write in one step'));

q = await question('top-p sampler');
check('sampler: "drops a token" is wrong', (await pickText(q, 'drops a token now and then')) && (await qText(q)).includes('Not quite.') && (await qText(q)).includes('same tokens in different orders'));
check('sampler: "twice" is wrong', (await pickText(q, 'public list twice')) && (await qText(q)).includes('committed once'));
check('sampler: "seq_cst" is wrong', (await pickText(q, 'memory_order_seq_cst')) && (await qText(q)).includes('which thread reaches the counter first'));
check('sampler: the order of the list is the cause; sort or a stable filter fixes it', (await pickText(q, 'The list order depends on the schedule')) && (await qText(q)).includes('Correct.') && (await qText(q)).includes('stable filter') && (await qText(q)).includes('Fig. 18.16'));

q = await question('commit after every frontier element');
check('stride barrier: "only threads still in the loop" is wrong', (await pickText(q, 'waits only for the threads still inside')) && (await qText(q)).includes('Not quite.'));
check('stride barrier: "block barriers cannot deadlock" is wrong', (await pickText(q, 'Only the grid barrier can deadlock')) && (await qText(q)).includes('Fig. 4.4'));
check('stride barrier: "only coalescing" is wrong', (await pickText(q, 'lose coalescing')) && (await qText(q)).includes('Correctness goes first'));
check('stride barrier: different trip counts reach the barrier a different number of times', (await pickText(q, 'different numbers of trips')) && (await qText(q)).includes('Correct.') && (await qText(q)).includes('chapter 4'));

q = await question('handle 3 vertices in that level');
await typeAnswer(q, 2500 - 2 * 1024 + 1);
check('stride count: an off-by-one is wrong', (await qText(q)).includes('Not quite.'));
await typeAnswer(q, model.strideThree);
check(`stride count: ${model.strideThree} threads take ${model.strideMax}`, model.strideThree === 452 && (await qText(q)).includes('Correct.') && (await qText(q)).includes('ranks 0 to 451'));

await pause(700); // the app saves 400 ms after the last change
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
const ids = ['q-tree-launches', 'q-exchange', 'q-sampler-order', 'q-stride-barrier', 'q-stride-count'].map((s) => 'ch18/wrap/' + s);
check('all five answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 5);
check('the running count of answers appears after answering', (await body()).includes('So far in this chapter you have answered'));
check('the mixed check is labelled as ours', await page.$$eval('.prose > aside.note', (as) => as.some((a) => a.textContent.includes('Not in the book') && a.textContent.includes('stride count are ours'))));
await shot('questions', '.q');

// ---- the book's exercises
const rows = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) =>
  trs.map((tr) => ({ ex: tr.children[0].textContent.trim(), where: tr.children[2].textContent.trim(), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('the exercise table covers 1a, 1b, 1c, 2 and 3', rows.map((r) => r.ex).join(' | ') === '1a | 1b | 1c | 2 | 3', rows.map((r) => r.ex).join(' | '));
const has = (ex, l) => rows.find((r) => r.ex === ex)?.links.includes(l);
check('1a and 1b point at the start page, 1c and 2 at 18.3, 3 at 18.6',
  has('1a', '#/ch18/start') && has('1b', '#/ch18/start') && has('1c', '#/ch18/18.3') && has('2', '#/ch18/18.3') && has('3', '#/ch18/18.6'), JSON.stringify(rows.map((r) => [r.ex, r.links])));
check(`1b's CSR is read from the model: srcPtrs ${list(model.csr.srcPtrs)}`,
  rows[1].where.includes(`srcPtrs = ${list(model.csr.srcPtrs)}`) && rows[1].where.includes(`dst = ${list(model.csr.dst)}`));
check('1a row by row from the model', text0.includes(model.rows.map((r, v) => `${v} → {${r.join(', ')}}`).join('; ')));
const ex1c = await page.$$eval('table[data-table="ex1c"] tbody tr', (trs) => trs.map((tr) => ({ k: tr.dataset.kernel, what: tr.children[1].textContent, v: [...tr.children].slice(2).map((td) => Number(td.textContent)) })));
const find = (k, w) => ex1c.find((r) => r.k === k && r.what.startsWith(w))?.v.join(',');
check('1c table: push 8 per call and looping from the model', find('push', 'threads launched') === '8,8,8,8' && find('push', 'threads that loop') === model.pushLoop.join(','));
check('1c table: pull looping and labelling from the model', find('pull', 'threads that loop') === model.pullLoop.join(',') && find('pull', 'threads that label') === model.pullLabel.join(','));
check('1c table: edge-centric 15 per call, source check and may-label from the model',
  find('edge', 'threads launched') === '15,15,15,15' && find('edge', 'threads whose source') === model.edgeSrc.join(',') && find('edge', 'threads that may label') === model.edgeLabel.join(','));
check('1c table: frontier launches from the model', find('frontier', 'threads launched') === model.frontier.join(','));
check('1c is labelled ours (edge directions read from the drawing)', await page.$$eval('.prose > aside.note', (as) => as.some((a) => a.textContent.includes('our reading of p. 450'))));
await shot('exercises', 'table[data-table="exercises"]');

// What the table points at exists on the pages it names.
await goto(page, 'ch18/start');
const onStart = await page.evaluate(() => ({
  lens: !!document.querySelector('section.bench[aria-label="Format lens"]'),
  ex1: [...document.querySelectorAll('section.bench[aria-label="Format lens"] button')].some((b) => b.textContent.includes('Exercise 1 graph')),
  csrQ: [...document.querySelectorAll('.q')].some((q) => q.textContent.includes('Which edges leave vertex 4')),
}));
check('the start page has the format lens with the Exercise 1 graph, and the CSR reading question', onStart.lens && onStart.ex1 && onStart.csrQ, JSON.stringify(onStart));
await goto(page, 'ch18/18.3');
const on183 = await page.evaluate(() => ({
  explorer: !!document.querySelector('section.bench[aria-label="One graph, five ways to run a level"]'),
  preds: [...document.querySelectorAll('.q')].filter((q) => q.textContent.includes('Exercise 1 graph') || q.textContent.includes('Same graph, edge-centric')).length,
  ex2: [...document.querySelectorAll('.q')].some((q) => q.textContent.includes('Exercise 2')),
}));
check('page 18.3 has the explorer, the two Exercise 1 predictions and the Exercise 2 question', on183.explorer && on183.preds >= 2 && on183.ex2, JSON.stringify(on183));
await goto(page, 'ch18/18.6');
const on186 = await page.evaluate(() => ({
  stepper: !!document.querySelector('section.bench[aria-label="Three grid barriers"]'),
  ex3btn: [...document.querySelectorAll('button')].some((b) => b.textContent.includes('Exercise 3: a counter per level')),
  ex3tab: [...document.querySelectorAll('figure.code .tabs button')].some((b) => b.textContent === 'Exercise 3 kernel'),
  ex3q: [...document.querySelectorAll('.q')].some((q) => q.textContent.includes('Exercise 3')),
}));
check('page 18.6 has the stepper "Three grid barriers" with its Exercise 3 mode, the Exercise 3 kernel tab and question', on186.stepper && on186.ex3btn && on186.ex3tab && on186.ex3q, JSON.stringify(on186));
await goto(page, 'ch18/wrap');

// ---- errata
const text1 = await body();
const errata = await page.$$eval('ul[data-list="errata"] > li', (lis) => lis.map((li) => li.textContent));
const want = ['p. 441', 'p. 439: "the threads', 'p. 433', 'unbalanced parenthesis', 'breath-first', 'not schedule', 'Figs. 18.7, 18.9, 18.11 and 18.13',
  'line 29', 'three net terminals', 'undirected'];
check(`the errata list has ${want.length} items: the notes' six numbered slips and their other items`,
  errata.length === want.length && want.every((w, i) => errata[i].includes(w)), JSON.stringify(errata.map((e) => e.slice(0, 40))));
const gaps = await page.$$eval('ul[data-list="gaps"] > li', (lis) => lis.map((li) => li.textContent));
check('the gaps list names capacity, host loop, constants, structs and include, 64-bit ranks, the Exercise 1 reading, the four-argument launch, the __shared__ placement', gaps.length === 8 &&
  ['capacity', 'No host loop', 'PRIVATE_FRONTIER_CAPACITY', 'CSRGraph', '64-bit', 'Exercise 1 graph', 'four arguments', 'inside the level loop'].every((w, i) => gaps[i].includes(w)), JSON.stringify(gaps.map((g) => g.slice(0, 40))));
check('the errata are framed as readings of one PDF, checked against vendor documentation', text1.includes('trust your copy') && text1.includes('ch18-facts.md'));
check('what was left out names §18.1, §18.2, p. 436, p. 438, §18.8, p. 449, the pull and edge kernels, Exercise 2, references 1 and 3',
  ['pp. 425–429', 'pp. 430–431', '(p. 436)', '(p. 438)', 'pp. 448–449', '(p. 449)', 'never lab work', 'Exercise 2 is worked', 'References 1'].every((s) => text1.includes(s)));

// ---- project status from the reports
const results = (run, id, variant) => (run ? run.results.filter((x) => x.lab === id && x.variant === variant) : []);
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = results(run, id, variant).filter((x) => x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const where = (b) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
const passing = [...Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race'), 'emu'].filter((b) => labStatus(reports[b], 'ch18/lab1_frontier', 'work') === 'pass');
const status = await page.$eval('[data-lab181]', (e) => ({ v: e.dataset.lab181, text: e.textContent }));
check(`project status: lab 18.1 work passes on ${passing.join(', ') || 'nothing'} here`, status.v === (passing.join(',') || 'none') &&
  (passing.length ? status.text.includes(passing.map(where).join(' and ')) : status.text.includes('has not passed on this machine yet') || status.text.includes('No lab results')), JSON.stringify(status));
check('the page says this chapter adds no milestone and why', text1.includes('adds no milestone') && text1.includes('traverses no graph'));
check('the page points at chapter 19', await page.$$eval('.prose a', (as) => as.some((a) => a.getAttribute('href') === '#/ch19/start')));

// ---- every link on the page goes to a written page
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
for (const l of links) {
  await goto(page, l.slice(2));
  const st = await page.evaluate(() => ({
    h1: !!document.querySelector('.page h1'),
    writing: document.querySelector('.page').textContent.includes('This section is being written'),
    planned: document.querySelector('.page').textContent.includes('This chapter is planned and not built yet'),
  }));
  check(`link ${l} opens a written page`, st.h1 && !st.writing && !st.planned, JSON.stringify(st));
}

// ---- margin notes sit after paragraphs, never after a bench or code block
await goto(page, 'ch18/wrap');
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
