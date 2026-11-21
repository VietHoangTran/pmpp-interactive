// Browser check for appendix A, "Summary and exercises": the carry-forward list, the
// mixed check (five questions whose answers come from the same models the sections use),
// the map of the book's exercises to where they are worked, the errata list, and the
// project status computed from the reports.
// Usage: node e2e/appA-wrap.mjs [screenshot-prefix] [--dark]
// Environment: APP_URL (default http://localhost:5173).
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
const lastAnswers = () => (saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {});

await goto(page, 'appA/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', "The book's exercises", 'Where the PDF looks wrong', 'The project so far'];
check('the page has its five parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
check('the page title is "Summary and exercises"', (await page.$eval('.page h1', (e) => e.textContent)).includes('Summary and exercises'));
check('the carry-forward list has seven points and cites §A.8, p. 599',
  (await page.$$eval('.prose > ul', (uls) => uls[0].children.length)) === 7 && (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('599'));
const text0 = await body();
check('nothing on the page says it is being written, and there is no em dash', !/being written|—/i.test(text0));

// ---- the five questions; every number comes from the models the sections use
// The models are imported through the app's module server, as the sections import them.
const M = await page.evaluate(async () => {
  const fs = await import('/src/sim/fpsum.ts');
  const mf = await import('/src/sim/minifloat.ts');
  const squares = Array(4096).fill(16);
  return {
    loop: fs.sumSequential(squares, 'fp16'),
    tree: String(fs.sumTree(squares, 'fp16')),
    spacing: mf.ulp(mf.BF16, 10),
    a: mf.round(mf.BF16, 10.01),
    b: mf.round(mf.BF16, 10.03),
    minNormal: mf.minNormal(mf.FP32),
    e4m3: mf.round(mf.E4M3, 0.0006),
  };
});
const loop = M.loop;
const tree = Number(M.tree);
check(`model: the FP16 loop gives ${loop} and the tree ${tree}`, loop === 32768 && tree === Infinity);
check(`no question is answered at the start (${(await page.$$('.q')).length} questions)`, (await page.$$('.q')).length === 5 && (await page.$$('.q .verdict')).length === 0);

let q = await question('A norm layer sums the squares');
await pick(q, 2);
check('rms: "both infinity" is wrong, because the loop stalls on a tie', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('exactly half a gap'));
await pick(q, 0);
check(`rms: loop ${loop.toLocaleString('en-US')} and tree infinity is right, with the 2.83 and 1.41 consequences`,
  (await qText(q)).includes('Correct.') && (await qText(q)).includes('32,768') && (await qText(q)).includes('2.83') && (await qText(q)).includes('1.41'));

const spacing = M.spacing;
q = await question('What is the gap between neighbouring BF16 values near 10?');
await typeAnswer(q, 0.125);
check('bf16 argmax: 0.125 is not the gap', (await qText(q)).includes('Not quite.'));
await typeAnswer(q, spacing);
check(`bf16 argmax: ${spacing} is right, and both logits round to ${M.a}`,
  (await qText(q)).includes('Correct.') && M.a === 10 && M.b === 10 && (await qText(q)).includes('Both logits round to 10.0'));

q = await question('Which way of computing its log-probability');
await pick(q, 0);
check('log-prob: the log of the flushed probability is minus infinity', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('log 0 is minus infinity'));
await pick(q, 1);
check(`log-prob: the shifted form is right; exp(-90) = ${Math.exp(-90).toExponential(2)} is below FP32's smallest normal ${M.minNormal.toExponential(2)}`,
  (await qText(q)).includes('Correct.') && Math.exp(-90) < M.minNormal);

q = await question('One activation is 0.0006');
await typeAnswer(q, 0.0006);
check('e4m3: 0.0006 is not what is stored', (await qText(q)).includes('Not quite.'));
await typeAnswer(q, M.e4m3);
check(`e4m3: ${M.e4m3} is right, below half of 2^-9`, (await qText(q)).includes('Correct.') && M.e4m3 === 0);

q = await question('in a batch of 8');
await pick(q, 1);
check('batch order: a race is wrong, because each configuration is repeatable', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('repeatable'));
await pick(q, 0);
check('batch order: a different summation order is right', (await qText(q)).includes('Correct.'));

await pause(700); // the app saves 400 ms after the last change
const ans = lastAnswers();
const ids = ['q-rms-half', 'q-bf16-argmax', 'q-logprob-ftz', 'q-e4m3-small', 'q-batch-order'].map((s) => 'appA/wrap/' + s);
check('all five answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 5);
check('the running count of answers appears after answering', (await body()).includes('So far in this appendix you have answered'));
await shot('questions', '.q');

// ---- the book's exercises and where they are worked
const rows = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) =>
  trs.map((tr) => ({ ex: tr.children[0].textContent.trim(), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
const where = { 1: '#/appA/A.2', 2: '#/appA/A.2', 3: '#/appA/A.4', 4: '#/appA/A.6', 5: '#/appA/A.4' };
check('the table has the five exercises, in order', rows.map((r) => r.ex).join(',') === '1,2,3,4,5', rows.map((r) => r.ex).join(','));
check('every row links to the section that works it', rows.every((r) => r.links.includes(where[r.ex])), JSON.stringify(rows));
check('the references paragraph names MX v1.1 and section 5.6 of the CUDA Programming Guide', text0.includes('version 1.1') && text0.includes('section 5.6 of the CUDA Programming Guide'));

// ---- errata
const errata = await page.$$eval('.prose > ul', (uls) => [...uls[1].children].map((li) => li.textContent));
check(`the errata list has ${errata.length} items`, errata.length === 10, String(errata.length));
check('it covers ftz, the 256 claim, the NaN count, TF32, Fig. A.11, partial pivoting and Exercise 4',
  ['--ftz=true', '256 times', '24 − 2', 'TF32', 'Fig. A.11', 'complete pivoting', 'Exercise 4'].every((s) => errata.some((e) => e.includes(s))), errata.map((e) => e.slice(0, 30)).join(' | '));
check('it ends with the typo pages', errata.at(-1).includes('pp. 583, 585, 586, 590, 591, 592 and 597'));
check('readings of the PDF are marked as ours', text0.includes('our readings of one PDF'));

// ---- project status, computed from the reports
const LAB = 'appA/lab1_sumerror';
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = (run?.results ?? []).filter((x) => x.lab === id && x.variant === variant && x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const devices = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const passing = devices.filter((b) => labStatus(reports[b], LAB, 'work') === 'pass');
const want = devices.length === 0
  ? 'No lab results are recorded on this machine yet.'
  : passing.length
    ? `Your own version passes on ${passing.join(' and ')} on this machine.`
    : 'Your own version has not passed on this machine yet.';
check(`lab A.1 status is read from the reports: "${want}"`, text0.includes(want));
check('the project paragraph says no milestone and names X2, M9 and M4', text0.includes('This appendix adds no milestone') && text0.includes('Module X2 picks this up') && text0.includes('M9') && text0.includes('M4'));

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/appA/start', ...['A.1', 'A.2', 'A.3', 'A.4', 'A.5', 'A.6', 'A.7'].map((s) => '#/appA/' + s), '#/appA/labs', '#/appA/wrap', '#/x2']);
check(`links go to pages that exist (${links.length} distinct)`, links.length >= 6 && links.every((l) => known.has(l)), links.join(', '));
await page.evaluate(() => [...document.querySelectorAll('.prose a')].find((a) => a.getAttribute('href') === '#/appA/labs').click());
await page.waitForFunction(() => document.querySelector('.page h1')?.textContent.includes('Labs'), { timeout: 10000 });
check('the link to the labs page opens it', (await page.$eval('.page h1', (e) => e.textContent)).includes('Labs'));

// ---- margin notes follow text
await goto(page, 'appA/wrap');
const after = await page.$$eval('.prose > aside.note', (as) => as.map((a) => a.previousElementSibling?.tagName ?? ''));
check('every margin note directly follows a paragraph or a list', after.every((t) => ['P', 'UL', 'OL'].includes(t)), after.join(','));

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'appA/wrap');
await pause(300);
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'table[data-table="exercises"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
