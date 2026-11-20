// Browser check for chapter 21, "Summary and exercises": the carry-forward list, the mixed
// check (one kernel shown from a checked snippet, five questions), this machine's numbers
// beside the intensity question, the map of the book's exercises, the errata lists, what was
// left out, and the project status computed from the reports.
// Usage: APP_URL=http://localhost:<port> node e2e/ch21-wrap.mjs [screenshot-prefix] [--dark]
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
const fmt = (v, digits = 3) =>
  !isFinite(v) ? 'n/a' : v === 0 ? '0' : Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : Number(v.toPrecision(digits)).toString();
const n = (v) => v.toLocaleString('en-US');

await goto(page, 'ch21/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
// The models the page uses, imported in the page (cutoff.ts imports dcs.ts without an extension).
const M = await page.evaluate(async () => {
  const { warpAccess } = await import('/src/sim/coalesce.ts');
  const { acceptedShare, counts, superRadius } = await import('/src/sim/cutoff.ts');
  const { arithmetic, opsGather } = await import('/src/sim/dcs.ts');
  const lanes = [...Array(32).keys()];
  const sectors = (stride) => warpAccess(lanes.map((t) => stride * t * 4), 4, 32).transactions;
  const list = (b) => counts(superRadius(12, b, 3, 'half') / b, 3).total;
  const share = (b) => Math.round(100 * acceptedShare(12, b, 3));
  return { ops: arithmetic(opsGather(1)), adj8: sectors(8), adj4: sectors(4), str: sectors(1), list4: list(4), list2: list(2), share4: share(4), share2: share(2) };
});
console.log('note  from the models: ' + JSON.stringify(M));
const body = async () => (await page.$eval('.prose', (e) => e.textContent)).replace(/\s+/g, ' ');

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary and exercises", numbered 21.6', h1.includes('Summary and exercises') && (await page.$eval('.page', (e) => e.textContent)).includes('21.6'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', "The book's exercises", 'Where the PDF looks wrong', 'What this chapter left out', 'The project so far'];
check('the page has its six parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
check('the carry-forward list has eight points and cites the book section',
  (await page.$$eval('.prose > ul', (uls) => uls[0].children.length)) === 8 &&
  (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('§21.6') && (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('527–528'));
const text0 = await body();
check('nothing on the page says it is being written', !/being written/i.test(text0));

// ---- the kernel for the bias question is the checked snippet
const file = await page.evaluate(() => fetch('/api/lab-file?path=' + encodeURIComponent('ch21/snippets/bias_gather.cu')).then((r) => (r.ok ? r.text() : 'MISSING')));
const below = file.split('// ---- shown in the lesson ----\n')[1] ?? '';
const shownCode = await page.$eval('figure.code pre', (e) => e.textContent);
check('the listing is labs/ch21/snippets/bias_gather.cu below its marker, line for line', file !== 'MISSING' && below && shownCode.trim() === below.trim(), shownCode.slice(0, 80));
check('the listing is labelled as not a lab file and front-end-checked, never run',
  (await page.$eval('figure.code .tabs', (e) => e.textContent)).includes('not a lab file') &&
  (await page.$eval('figure.code .tabs .status', (e) => e.textContent)) === 'front-end-checked; never run');
check('logits, ids and bias are coloured global, the thread\'s values registers',
  (await page.$$('figure.code .v-global')).length >= 4 && (await page.$$('figure.code .v-reg')).length >= 4);
check('the Beyond note says the kernel, the factor of 8 and the halved bins are ours, and the kernel never ran',
  await page.$$eval('.prose > aside.note', (as) => as.some((a) => a.textContent.includes('are ours') && a.textContent.includes('never been run'))));
await shot('code', 'figure.code');

// ---- the five questions, each answered wrong then right; numbers from the models the page uses
check(`no question is answered at the start (${(await page.$$('.q')).length} questions)`, (await page.$$('.q')).length === 5 && (await page.$$('.q .verdict')).length === 0);

let q = await question('128,000 logits');
await pick(q, 0);
check('bias: "nothing to change" is wrong: 6,400,000 comparisons for 50 additions', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('6,400,000 comparisons'));
await pick(q, 2);
check('bias: "two threads race" is wrong: a duplicate is summed in one thread', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('summed inside that one thread'));
await pick(q, 1);
check('bias: a thread per pair with an atomic, or sort and one writer per id, is right; binning and PyTorch are named',
  (await qText(q)).includes('Correct.') && (await qText(q)).includes('binning with one bin per token') && (await qText(q)).includes('get_rows_back'));

const ops = M.ops;
const intensity = (64 * ops) / 8;
q = await question('per byte of global-memory traffic');
await typeAnswer(q, intensity / 2);
check(`intensity: ${intensity / 2} is wrong and the answer ${intensity} is given`, (await qText(q)).includes(`Not quite. The answer is ${intensity} operations per byte.`), await qText(q));
await typeAnswer(q, intensity);
check(`intensity: ${intensity} is right, ${n((1024 * ops) / 8)} at chunk 1,024, and the book's example threshold is labelled`,
  (await qText(q)).includes('Correct.') && (await qText(q)).includes(n((1024 * ops) / 8)) && (await qText(q)).includes("the book's H100") && (await qText(q)).includes('(p. 94)'));

const { adj8, str, adj4 } = M;
q = await question('how many 32-byte sectors do the');
await typeAnswer(q, 8 * str);
check(`factor 8: the strided ${8 * str} is not the answer`, (await qText(q)).includes(`The answer is ${8 * adj8} sectors.`), await qText(q));
await typeAnswer(q, 8 * adj8);
check(`factor 8: ${8 * adj8} is right (${adj8} per statement against ${adj4} at factor 4, strided ${str} and ${8 * str})`,
  (await qText(q)).includes('Correct.') && (await qText(q)).includes(`${adj8} sectors per statement`) && (await qText(q)).includes(`it was ${adj4} per statement`) &&
  (await qText(q)).includes(`${8 * str} in all`));

const list = (b) => (b === 4 ? M.list4 : M.list2);
const share = (b) => (b === 4 ? M.share4 : M.share2);
q = await question('You halve the bin side');
check(`small bins: the prompt states the model's ${list(4)} bins and ${share(4)} %`, (await qText(q)).includes(`list has ${n(list(4))} bins`) && (await qText(q)).includes(`about ${share(4)} %`));
await pick(q, 3);
check(`small bins: "the list gets shorter" is wrong: ${n(list(2))} against ${list(4)}`, (await qText(q)).includes('Not quite.') && (await qText(q)).includes(`${n(list(2))} against ${list(4)}`));
await pick(q, 0);
check(`small bins: about ${share(2)} % pass, ${n(list(2))} bins, blocks of 64 threads, is right`,
  (await qText(q)).includes('Correct.') && (await qText(q)).includes(`About ${share(2)} %`) && (await qText(q)).includes('about four times as many'));

q = await question('within a tolerance, not bit for bit');
await pick(q, 1);
check('bitwise: "different atom order" is wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('from n = 0 upwards'));
await pick(q, 0);
check('bitwise: the arithmetic differs and each step rounds, is right', (await qText(q)).includes('Correct.') && (await qText(q)).includes('compares a kernel with itself'));

await pause(700); // the app saves 400 ms after the last change
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
const ids = ['q-sparse-bias', 'q-intensity', 'q-factor8', 'q-small-bins', 'q-bitwise'].map((s) => 'ch21/wrap/' + s);
check('all five answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 5);
check('the running count of answers appears after answering', (await body()).includes('So far in this chapter you have answered'));
await shot('questions', '.q');

// ---- this machine's numbers beside the intensity question
const results = (run, id, variant) => (run ? run.results.filter((x) => x.lab === id && x.variant === variant) : []);
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = results(run, id, variant).filter((x) => x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const rowsOf = (run, id) => {
  const mine = labStatus(run, id, 'work') === 'pass' ? results(run, id, 'work').flatMap((x) => x.bench) : [];
  return mine.length ? mine : results(run, id, 'solution').flatMap((x) => x.bench);
};
const machine = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race').flatMap((b) => {
  const rows = rowsOf(reports[b], 'ch21/lab1_dcs');
  const sec = (name, chunk) => rows.find((r) => r.name === name && r.nx === 1024 && r.chunk === chunk)?.seconds;
  const r = (chunk) => (sec('coarse_adjacent', chunk) && sec('coarse_strided', chunk) ? sec('coarse_adjacent', chunk) / sec('coarse_strided', chunk) : null);
  return r(64) && r(1024) ? [{ b, r64: r(64), r1024: r(1024) }] : [];
});
const shownMachine = await page.$$eval('[data-machine]', (ps) => ps.map((p) => ({ b: p.dataset.machine, r64: p.dataset.r64 ?? '', r1024: p.dataset.r1024 ?? '', text: p.textContent })));
if (machine.length)
  check(`your own numbers: ${machine.map((m) => `${m.b} strided/adjacent ${fmt(m.r64)} at chunk 64, ${fmt(m.r1024)} at 1,024`).join('; ')}`,
    machine.every((m) => shownMachine.some((s) => s.b === m.b && s.r64 === fmt(m.r64) && s.r1024 === fmt(m.r1024) && s.text.includes(fmt(m.r64)))), JSON.stringify(shownMachine));
else check('no lab 21.1 rows, and the page says what to run', shownMachine.length === 1 && shownMachine[0].b === 'none' && shownMachine[0].text.includes('run 21.1 --bench'));

// ---- the book's exercises
const rows = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) =>
  trs.map((tr) => ({ ex: tr.children[0].textContent.trim(), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('the exercise table covers all five exercises', rows.map((r) => r.ex).join(' | ') === '1 | 2 | 3 | 4 | 5', rows.map((r) => r.ex).join(' | '));
const sectionOf = { 1: '#/ch21/labs', 2: '#/ch21/21.2', 3: '#/ch21/21.2', 4: '#/ch21/21.2', 5: '#/ch21/21.5' };
check('each exercise points at the page that works it', rows.every((r) => r.links.includes(sectionOf[r.ex])), JSON.stringify(rows.map((r) => [r.ex, r.links])));
const row3 = await page.$eval('table[data-table="exercises"] tbody tr:nth-child(3)', (tr) => tr.textContent.replace(/\s+/g, ' '));
check('exercise 3: the coalescing cost is worked by the 21.2 section and its stepper', row3.includes('Coalescing: who owns which four') && row3.includes('write-statement stepper'), row3);
// The questions those rows point at exist on those pages.
for (const [route, needles] of [['21.2', ['Exercise 2.', 'Exercise 3, made concrete', 'Exercise 4.', 'Coalescing: who owns which four']], ['21.5', ['Exercise 5.']], ['labs', ['launchDcsGather', 'exercise 1']]]) {
  await goto(page, 'ch21/' + route);
  const t = await page.$eval('.prose', (e) => e.textContent.replace(/\s+/g, ' '));
  check(`page ${route} has what the table names (${needles.join(', ')})`, needles.every((x) => t.includes(x)));
}
await goto(page, 'ch21/wrap');

// ---- errata and dated statements
const errata = await page.$$eval('ul[data-list="errata"] > li', (lis) => lis.map((li) => li.textContent.replace(/\s+/g, ' ')));
const want = ["line 05", 'do not add up', 'two problems', 'half-warps', 'Fig. 21.13 labels', 'gather-to-scatter', 'indexed three ways', 'No kernel tests a bound', 'atom[]'];
check(`the errata list has ${want.length} items covering the notes' "Unclear in the PDF"`, errata.length === want.length &&
  want.every((w, i) => errata[i].includes(w)), JSON.stringify(errata.map((e) => e.slice(0, 40))));
const dated = await page.$$eval('ul[data-list="dated"] > li', (lis) => lis.map((li) => li.textContent.replace(/\s+/g, ' ')));
check('the dated list names half-warps (2007), the legacy atomic, unquantified speed, registers and the constant cache',
  dated.length === 5 && ['2007', 'legacy', 'not quantified', 'stays within the allowed limit', 'scalar cache'].every((w, i) => dated[i].includes(w)), JSON.stringify(dated.map((e) => e.slice(0, 40))));
check('the errata are framed as readings of one PDF, checked against vendor documentation', text0.includes('trust your copy') && text0.includes('ch21-facts.md'));
check('what was left out names §21.1, Figs. 21.3 and 21.4, Fig. 21.13 and the binned kernel',
  text0.includes('§21.1, pp. 513–514') && text0.includes('Figs. 21.3 and 21.4') && text0.includes('512 circles') && text0.includes('not built in a lab'));

// ---- project status from the reports
const where = (b) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
const passing = [...Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race'), 'emu'].filter((b) => labStatus(reports[b], 'ch21/lab1_dcs', 'work') === 'pass');
const status = await page.$eval('[data-lab211]', (e) => ({ v: e.dataset.lab211, text: e.textContent }));
check(`project status: lab 21.1 work passes on ${passing.join(', ') || 'nothing'} here`, status.v === (passing.join(',') || 'none') &&
  (passing.length ? status.text.includes(passing.map(where).join(' and ')) : status.text.includes('has not passed on this machine yet') || status.text.includes('No lab results')), JSON.stringify(status));
check('the page says this chapter adds no milestone and points at chapter 22 and M8',
  text0.includes('adds no milestone') && text0.includes('milestone M8') && (await page.$$eval('.prose a', (as) => as.some((a) => a.getAttribute('href') === '#/ch22/start'))));

// ---- every link on the page goes to a section that exists
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
for (const l of links) {
  await goto(page, l.slice(2));
  const ok = await page.evaluate(() => !!document.querySelector('.page h1') && !document.querySelector('.page').textContent.includes('This section is being written'));
  check(`link ${l} opens a written page`, ok);
}

// ---- margin notes sit after paragraphs, never after a bench or code block
await goto(page, 'ch21/wrap');
const misplaced = await page.$$eval('.prose > aside.note', (as) => as.filter((a) => {
  const prev = a.previousElementSibling;
  return prev && !['P', 'UL', 'OL', 'ASIDE'].includes(prev.tagName);
}).map((a) => `${a.previousElementSibling.tagName}.${a.previousElementSibling.className}`));
check('every margin note follows a paragraph, a list or another note', misplaced.length === 0, misplaced.join(', '));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
