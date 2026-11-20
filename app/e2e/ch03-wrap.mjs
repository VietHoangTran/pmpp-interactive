// Browser check for chapter 3, "Summary and exercises": the carry-forward list, the mixed
// check (one kernel shown from a checked snippet, five questions), the map of the book's
// exercises, the errata list and the project status computed from the reports.
// Usage: node e2e/ch03-wrap.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';
import { span } from '../src/sim/blur.ts';
import { layout, strides } from '../src/sim/mapping.ts';

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

await goto(page, 'ch03/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', "The book's exercises", 'Where the PDF looks wrong', 'What this chapter left out', 'The project so far'];
check('the page has its six parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
check('the carry-forward list has eight points and cites the book section',
  (await page.$$eval('.prose > ul', (uls) => uls[0].children.length)) === 8 && (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('§3.5, p. 64'));
const text0 = await body();
check('it states the mapping, the flat index and the storage convention',
  text0.includes('row = blockIdx.y*blockDim.y + threadIdx.y') && text0.includes('plane*m*n + row*m + col') && text0.includes('holds the weight as (in, out)'));
check('nothing on the page says it is being written', !/being written/i.test(text0));

// ---- the kernel for the first question is the checked snippet
const file = await page.evaluate(() => fetch('/api/lab-file?path=' + encodeURIComponent('ch03/snippets/scale_rows_defect.cu')).then((r) => (r.ok ? r.text() : 'MISSING')));
const shownCode = await page.$eval('figure.code pre', (e) => e.textContent);
check('the listing is labs/ch03/snippets/scale_rows_defect.cu, line for line', file !== 'MISSING' && shownCode.trim() === file.trim(), shownCode.slice(0, 80));
check('the listing is labelled as not a lab file, front-end-checked, and defective on purpose',
  (await page.$eval('figure.code .tabs', (e) => e.textContent)).includes('not a lab file') &&
  (await page.$eval('figure.code .tabs .status', (e) => e.textContent)) === 'front-end-checked; has a defect on purpose');
check('the buffer is coloured as global memory and the indices as registers',
  (await page.$$('figure.code .v-global')).length >= 4 && (await page.$$('figure.code .v-reg')).length >= 6);
await shot('code', 'figure.code');

// ---- the five questions, each answered; numbers come from the models the page uses
const decode = layout({ height: 1, width: 4096, blockW: 16, blockH: 16, style: 'blocks' });
const corner = span(0, 64, 2) ** 3;
check(`no question is answered at the start (${(await page.$$('.q')).length} questions)`, (await page.$$('.q')).length === 5 && (await page.$$('.q .verdict')).length === 0);

let q = await question('guaranteed to hold 1.0 afterwards');
await pick(q, 0);
check('in-place rows: "all 200" is wrong, and the reason given is the read of another thread\'s element',
  (await qText(q)).includes('Not quite.') && (await qText(q)).includes('196 threads read an element that a different thread stores to'));
await pick(q, 1);
check('in-place rows: "only the four in column 0" is right', (await qText(q)).includes('Correct.'));
check('  the explanation gives the emulator-checked counts and the fix', (await qText(q)).includes('4, 64 and 37 of the 200') && (await qText(q)).includes('emulator-checked') && (await qText(q)).includes('second buffer'));

q = await question('Which product does the thread add at step');
await pick(q, 1);
check('batched product: skipping j elements per batch entry is wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('whole matrices of j*k elements'));
await pick(q, 0);
const st = strides([8, 3, 5]);
check(`batched product: (b*j + row)*k + i is right, and the explanation shows the strides (${st.join(', ')})`,
  (await qText(q)).includes('Correct.') && (await qText(q)).includes(`(${st.join(', ')})`));

q = await question('How many of the launched threads');
await typeAnswer(q, decode.working);
check(`decode shape: ${decode.working} (the threads with work) is not the answer`, (await qText(q)).includes(`Not quite. The answer is ${decode.surplus} threads.`));
await typeAnswer(q, decode.surplus);
check(`decode shape: ${decode.surplus} surplus threads of ${decode.threads} is right, with the grid (${decode.gridX}, ${decode.gridY})`,
  (await qText(q)).includes('Correct.') && (await qText(q)).includes(`(${decode.gridX}, ${decode.gridY})`) && (await qText(q)).includes(decode.threads.toLocaleString('en-US')));

q = await question('thread of a corner element add up');
await typeAnswer(q, 125);
check('volume corner: 125 (the whole cube) is not the answer', (await qText(q)).includes(`Not quite. The answer is ${corner} values.`));
await typeAnswer(q, corner);
check(`volume corner: ${corner} is right, with the edge and face counts`, (await qText(q)).includes('Correct.') && (await qText(q)).includes('45') && (await qText(q)).includes('75'));

q = await question('does it pass?');
await pick(q, 3);
check('square test: (33, 20, 1) does not pass, and the reason is a read past the end of B', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('past its end'));
await pick(q, 2);
check('square test: (20, 1, 35) passes, because k = 1', (await qText(q)).includes('Correct.') && (await qText(q)).includes('k = 1'));

await pause(700); // the app saves 400 ms after the last change
const ans = lastAnswers();
const ids = ['q-inplace-rows', 'q-batched', 'q-decode-surplus', 'q-corner3d', 'q-squaretest'].map((s) => 'ch03/wrap/' + s);
check('all five answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 5);
check('the running count of answers appears after answering', (await body()).includes('So far in this chapter you have answered'));
await shot('questions', '.q');

// ---- the book's exercises
const rows = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) =>
  trs.map((tr) => ({ ex: tr.children[0].textContent.trim(), where: tr.children[2].textContent.trim(), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('the exercise table covers 1a, 1b, 1c, 2, 3a to 3d, 4a, 4b and 5',
  rows.map((r) => r.ex).join(' | ') === '1a, 1b | 1c | 2 | 3a, 3b, 3c | 3d | 4a, 4b | 5', rows.map((r) => r.ex).join(' | '));
check('every exercise is tied to a section or lab by a link', rows.every((r) => r.links.length >= 1 && r.where.length > 10), JSON.stringify(rows));
check('exercises 1a and 1b point at lab 3.4, exercise 2 at lab 3.3',
  rows[0].where.includes('lab 3.4') && rows[0].links.includes('#/ch03/labs') && rows[2].where.includes('lab 3.3'));

// ---- errata and omissions
const errata = await page.$$eval('ul[data-list="errata"] > li', (lis) => lis.map((li) => li.textContent));
check(`the errata list has ${errata.length} items, the four that change the code first`,
  errata.length === 12 && errata[0].includes('line 19') && errata[1].includes('0.99') && errata[2].includes('(Pout, Pin, width, height)') && errata[3].includes('colortoGrayscaleConvertion'),
  String(errata.length));
check('it includes the Fig. 3.8 numbering, the missing matmul launch and the p. 63 arithmetic',
  errata.some((e) => e.includes('07 to 09')) && errata.some((e) => e.includes('never prints the launch')) && errata.some((e) => e.includes('0 ∗ 0 + 1 = 1')));
check('readings of the PDF are marked as ours, and the Apple slip is kept apart from the book\'s',
  (await body()).includes('our readings of one PDF') && (await body()).includes('One slip is not the book\'s') && (await body()).includes('64 by 24'));
const leftOut = await page.$$eval('.prose > ul', (uls) => uls.at(-1).textContent);
check('what was left out is listed with where to find it', ['chapter 7', 'chapter 8', 'chapter 4', 'chapter 5', 'chapter 6'].every((c) => leftOut.includes(c)), leftOut.slice(0, 120));

// ---- project status, computed from the reports
const M1 = 'ch03/lab3_naive_gemm';
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = (run?.results ?? []).filter((x) => x.lab === id && x.variant === variant && x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const devices = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const passing = devices.filter((b) => labStatus(reports[b], M1, 'work') === 'pass');
const wantM1 = devices.length === 0 && !reports.emu
  ? 'No lab results are recorded on this machine yet.'
  : passing.length
    ? `Your own version passes on ${passing.join(' and ')} on this machine.`
    : labStatus(reports.emu, M1, 'work') === 'pass'
      ? 'Your own version passes on the emulator; it has not passed on a device on this machine yet.'
      : 'Your own version has not passed on this machine yet.';
const m1 = await page.$eval('[data-m1]', (e) => e.textContent);
check(`milestone M1 status is read from the reports: "${wantM1}"`, m1 === wantM1, m1);
check('the project paragraph names M1, its second half in chapter 4, and M2', (await body()).includes('first half of milestone M1') && (await body()).includes('The second half is chapter 4') && (await body()).includes('milestone M2'));

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch03/start', '#/ch03/3.1', '#/ch03/3.2', '#/ch03/3.3', '#/ch03/3.4', '#/ch03/labs', '#/ch03/wrap']);
check(`links go to sections of this chapter that exist (${links.join(', ')})`, links.length >= 4 && links.every((l) => known.has(l)), links.join(', '));
await page.evaluate(() => [...document.querySelectorAll('.prose a')].find((a) => a.getAttribute('href') === '#/ch03/labs').click());
await page.waitForFunction(() => document.querySelector('.page h1')?.textContent.includes('Labs'), { timeout: 10000 });
check('the link to the labs page opens it', (await page.$eval('.page h1', (e) => e.textContent)).includes('Labs'));

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch03/wrap');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'table[data-table="exercises"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
