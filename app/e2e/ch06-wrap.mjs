// Browser check for chapter 6, "Summary and exercises": the carry-forward list, the mixed
// check (one kernel shown from a checked snippet, six questions), the map of the book's
// exercises, the errata list, what was shortened, and the project status from the reports.
// Usage: APP_URL=http://localhost:<port> node e2e/ch06-wrap.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';
import { warpAccess } from '../src/sim/coalesce.ts';

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
const lastAnswers = () => (saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {});

// ---- the numbers, from the same model the page uses (NVIDIA's 32-byte sectors)
const H = 32, D = 128, h = 5, d = 7;
const lanes = (f) => Array.from({ length: 32 }, (_, l) => 4 * f(l));
const K = warpAccess(lanes((t) => (t * H + h) * D + d), 4, 32);
const Q = warpAccess(lanes(() => h * D + d), 4, 32);
const K4 = warpAccess(lanes((t) => (t * H + h) * D + 4), 16, 32);
const REMAP = warpAccess(lanes((l) => (3 * H + h) * D + l), 4, 32);

await goto(page, 'ch06/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary and exercises", numbered 6.10', h1.includes('Summary and exercises') && (await page.$eval('.page', (e) => e.textContent)).includes('6.10'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((x) => x.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', 'The book’s exercises', 'Where the PDF looks wrong', 'What this course shortened', 'The project so far'];
check('the page has its six parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
check('the carry-forward list has six points and cites the book section',
  (await page.$$eval('.prose > ul', (uls) => uls[0].children.length)) === 6 && (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('§6.10, p. 154'));
const text0 = await body();
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('the carry-forward list names each vendor\'s unit and says Apple documents none', text0.includes('32-byte sectors on NVIDIA, 64- or 128-byte lines on AMD, none documented') );
const misplaced = await page.$$eval('.prose > .note', (ns) => ns.filter((n) => n.previousElementSibling?.tagName !== 'P').map((n) => n.textContent.slice(0, 60)));
check('every margin note follows a paragraph', misplaced.length === 0, misplaced.join(' | '));

// ---- the kernel for the first two questions is the checked snippet
const file = await page.evaluate(() => fetch('/api/lab-file?path=' + encodeURIComponent('ch06/snippets/wrap_scores.cu')).then((r) => (r.ok ? r.text() : 'MISSING')));
const shownCode = await page.$eval('figure.code pre', (e) => e.textContent);
check('the listing is labs/ch06/snippets/wrap_scores.cu, line for line', file !== 'MISSING' && shownCode.trim() === file.trim(), shownCode.slice(0, 80));
check('the listing is labelled as not a lab file and front-end-checked, never run',
  (await page.$eval('figure.code .tabs', (e) => e.textContent)).includes('not a lab file') &&
  (await page.$eval('figure.code .tabs .status', (e) => e.textContent)) === 'front-end-checked; never run');
check('q, K and s are coloured as global memory, acc and the indices as registers',
  (await page.$$('figure.code .v-global')).length >= 3 && (await page.$$('figure.code .v-reg')).length >= 4);
check('the kernels and shapes are labelled as ours', text0.includes('The kernels and shapes in these questions are ours'));
await shot('code', 'figure.code');

// ---- the six questions, each answered wrong then right
check(`no question is answered at the start (${(await page.$$('.q')).length} questions)`, (await page.$$('.q')).length === 6 && (await page.$$('.q .verdict')).length === 0);

let q = await question('how many 32-byte sectors does one 32-lane warp');
await typeAnswer(q, 1);
check(`scores, sectors: 1 is wrong, the answer is ${K.transactions}`, (await qText(q)).includes(`Not quite. The answer is ${K.transactions} sectors.`));
await typeAnswer(q, K.transactions);
check(`scores, sectors: ${K.transactions} is right, with ${K.efficiency * 100}% used and ${Q.transactions} sector for q`,
  (await qText(q)).includes('Correct.') && (await qText(q)).includes(`${K.efficiency * 100}%`) && (await qText(q)).includes(`${Q.transactions} sector for the whole warp`) && (await qText(q)).includes(`${4 * H * D} bytes apart`));

q = await question('makes its reads of');
await pick(q, 1);
check(`scores, fix: float4 loads are wrong, still ${K4.transactions} sectors`, (await qText(q)).includes('Not quite.') && (await qText(q)).includes(`${K4.transactions} sectors per warp`));
await pick(q, 2);
check('scores, fix: padding K is wrong, padding is about shared-memory banks', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('shared memory'));
await pick(q, 0);
check(`scores, fix: lanes along d is right, ${REMAP.transactions} sectors`, (await qText(q)).includes('Correct.') && (await qText(q)).includes(`${REMAP.transactions} sectors, all bytes`));

q = await question('Splitting heads');
await pick(q, 3);
check('permute: "neither, it is a transpose" is wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('innermost dimension'));
await pick(q, 0);
check('permute: both sides coalesce, 4 sectors each', (await qText(q)).includes('Correct.') && (await qText(q)).includes('writes 4 sectors and reads 4'));

q = await question('For which K does every row start');
await pick(q, 1);
check('row alignment: "any K" is wrong because only the first row starts at the allocation', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('Only the first row'));
await pick(q, 0);
check('row alignment: K a multiple of 4 is right, with 2 of 8 rows at K = 4097', (await qText(q)).includes('Correct.') && (await qText(q)).includes('(2 of the first 8)') && (await qText(q)).includes('(8 of 8)'));

q = await question('chosen by');
await pick(q, 1);
check('three buffers: "yes, no overwrite" is wrong, that was the false dependence', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('false dependence'));
await pick(q, 0);
check('three buffers: no, the remaining barrier is a true dependence', (await qText(q)).includes('Correct.') && (await qText(q)).includes('read-after-write'));

q = await question('What do you change first?');
await pick(q, 1);
check('bottleneck: padding first is wrong, citing p. 153', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('p. 153'));
await pick(q, 0);
check('bottleneck: fewer bytes first is right, judged by duration', (await qText(q)).includes('Correct.') && (await qText(q)).includes('duration'));

await pause(700); // the app saves 400 ms after the last change
const ans = lastAnswers();
const ids = ['q-scores-sectors', 'q-scores-fix', 'q-permute', 'q-rowalign', 'q-threebuf', 'q-bottleneck'].map((s) => 'ch06/wrap/' + s);
check('all six answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 6);
check('the running count of answers appears after answering', (await body()).includes('So far in this chapter you have answered'));
await shot('questions', '.q');

// ---- the book's exercises
const rows = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) =>
  trs.map((tr) => ({ ex: tr.children[0].textContent.trim(), where: tr.children[2].textContent.trim(), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('the exercise table covers all five exercises', rows.map((r) => r.ex).join(' | ') === '1 | 2 | 3 | 4 | 5', rows.map((r) => r.ex).join(' | '));
const sectionOf = { 1: '#/ch06/6.1', 2: '#/ch06/6.1', 3: '#/ch06/6.1', 4: '#/ch06/6.3', 5: '#/ch06/6.4' };
check('each exercise points at the section that works it', rows.every((r) => r.links.includes(sectionOf[r.ex])), JSON.stringify(rows.map((r) => [r.ex, r.links])));
check('exercises 2 and 4 point at their labs as well', rows[1].links.includes('#/ch06/labs') && rows[3].links.includes('#/ch06/labs') && rows[1].where.includes('lab 6.4') && rows[3].where.includes('lab 6.2'));

// ---- errata
const errata = await page.$$eval('ul[data-list="errata"] > li', (lis) => lis.map((li) => li.textContent));
check(`the errata list has ${errata.length} items, covering the notes' list`, errata.length === 15, String(errata.length));
for (const [name, frag] of [
  ['the shift, pp. 140–141', '(i << 5) + i'],
  ['"second" barrier, p. 147', '“second” barrier'],
  ['HBM3E, p. 133', '1254.4 GB/s'],
  ['local arrays, p. 146', 'local memory'],
  ['Fig. 6.12', 'never uses its parameter N'],
  ['the 8 × 8 claim', 'use all four channels'],
  ['Fig. 6.8', 'idle time'],
  ['naming drift', 'BLOCK_SIZE'],
  ['Exercise 3 readings', 'two defensible answers'],
  ['summary omission', 'leaves out vector loads and bank conflicts'],
]) check(`errata include ${name}`, errata.some((e) => e.includes(frag)));
check('readings of the PDF are marked as ours', text0.includes('our readings of one PDF'));
check('dated statements checked against today\'s documents are named: one contradicted, two unconfirmed',
  text0.includes('contradicted today: “one to eight channels” (p. 133) understates current GPUs') && text0.includes('Two could not be confirmed either way')
  && text0.includes('NVIDIA gives no shared-memory latency in cycles') && !text0.includes('did not survive') && text0.includes('hundreds of clock cycles'));
check('the page says what was shortened, with pages', ['(p. 125)', 'Figs. 6.5 and 6.6, pp. 131–132', '(pp. 149–153)', 'Asynchronous copies'].every((s) => text0.includes(s)));

// ---- project status, computed from the reports
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = (run?.results ?? []).filter((x) => x.lab === id && x.variant === variant && x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
const m2 = backends.filter((b) => labStatus(reports[b], 'ch06/lab4_linear', 'work') === 'pass');
const shown = await page.$eval('[data-m2]', (e) => ({ v: e.dataset.m2, text: e.textContent }));
check(`milestone M2 status is read from the reports (${m2.join(',') || 'none'})`, shown.v === (m2.join(',') || 'none') &&
  (backends.length === 0 ? shown.text.includes('No lab results') : m2.length ? shown.text.includes('passes on') : shown.text.includes('has not passed')), JSON.stringify(shown));
check('the project paragraph names M2, lab 6.4 and chapter 15', text0.includes('Lab 6.4 is milestone M2') && text0.includes('milestone M3'));
check('the next chapter is named and linked', text0.includes('Chapter 7 opens Part 2') && (await page.$$eval('.prose a[href="#/ch07/start"]', (as) => as.length)) === 1);

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch06/6.1', '#/ch06/6.2', '#/ch06/6.3', '#/ch06/6.4', '#/ch06/6.6', '#/ch06/6.7', '#/ch06/labs', '#/ch15/labs', '#/ch07/start']);
check(`links go to routes that exist (${links.join(', ')})`, links.length >= 6 && links.every((l) => known.has(l)), links.join(', '));
await page.evaluate(() => [...document.querySelectorAll('.prose a')].find((a) => a.getAttribute('href') === '#/ch06/labs').click());
await page.waitForFunction(() => document.querySelector('.page h1')?.textContent.includes('Labs'), { timeout: 10000 });
check('the link to the labs page opens it', (await page.$eval('.page h1', (e) => e.textContent)).includes('Labs'));

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch06/wrap');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'table[data-table="exercises"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
