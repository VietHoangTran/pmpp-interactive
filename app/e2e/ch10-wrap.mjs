// Browser check for chapter 10, "Summary and exercises": the carry-forward list, the mixed
// check (one kernel shown from a checked snippet, five questions), the map of the book's
// exercises, the errata list and the project status computed from the reports.
// Usage: APP_URL=http://localhost:<port> node e2e/ch10-wrap.mjs [screenshot-prefix] [--dark]
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

// Fig. 10.10's tree on any block size (the model the page states; Wrap.snippets.test.ts pins it too).
const covers = (B) => {
  const s = Array.from({ length: B }, () => 2);
  for (let st = Math.floor(B / 2); st >= 1; st = Math.floor(st / 2)) for (let t = 0; t < st; t++) s[t] += s[t + st];
  return s[0];
};

await goto(page, 'ch10/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary and exercises", numbered 10.11', h1.includes('Summary and exercises') && (await page.$eval('.page', (e) => e.textContent)).includes('10.11'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', "The book's exercises", 'Where the PDF looks wrong', 'The project so far'];
check('the page has its five parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
check('the carry-forward list has six points and cites the book section',
  (await page.$$eval('.prose > ul', (uls) => uls[0].children.length)) === 6 && (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('§10.11, p. 249'));
const text0 = await body();
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('the carry-forward list says what makes warp code port', text0.includes('a typed lane mask, the width read at run time'));

// ---- the kernel for the first question is the checked snippet's centreRows
await page.waitForFunction(() => !document.querySelector('figure.code pre')?.textContent.startsWith('// loading'), { timeout: 10000 });
const file = await page.evaluate(() => fetch('/api/lab-file?path=' + encodeURIComponent('ch10/snippets/wrap_centre.cu')).then((r) => (r.ok ? r.text() : 'MISSING')));
const shownCode = (await page.$eval('figure.code pre', (e) => e.textContent)).trim();
const start = file.indexOf('__global__ void centreRows');
const kernel = start >= 0 ? file.slice(start).trim() : 'MISSING';
check('the listing is centreRows from labs/ch10/snippets/wrap_centre.cu, line for line', file !== 'MISSING' && shownCode === kernel, shownCode.slice(0, 80));
check('the listing is labelled as not a lab file, front-end-checked and wrong on purpose',
  (await page.$eval('figure.code .tabs', (e) => e.textContent)).includes('not a lab file') &&
  (await page.$eval('figure.code .tabs .status', (e) => e.textContent)) === 'front-end-checked; wrong on purpose');
check('the buffers are coloured as global memory, the array as shared memory, the sums as registers',
  (await page.$$('figure.code .v-global')).length >= 4 && (await page.$$('figure.code .v-shared')).length >= 2 && (await page.$$('figure.code .v-reg')).length >= 6);
await shot('code', 'figure.code');

// ---- the five questions, each answered wrong then right
check(`no question is answered at the start (${(await page.$$('.q')).length} questions)`, (await page.$$('.q')).length === 5 && (await page.$$('.q .verdict')).length === 0);

let q = await question('Which outputs are right?');
await pick(q, 0);
check('centre: "all of them" is wrong, because nothing hands the total back', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('nothing hands it back'));
await pick(q, 2);
check('centre: "warp 0\'s columns" is wrong, because shuffle-down leaves the sum in lane 0', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('lane 0 only'));
await pick(q, 1);
check('centre: thread 0\'s columns 0, 256, 512, 768 are right, and the fix is a hand-back through shared memory',
  (await qText(q)).includes('Correct.') && (await qText(q)).includes('0, 256, 512, 768') && (await qText(q)).includes('the other 996 of each row are wrong') && (await qText(q)).includes('simd_sum'));

q = await question('How many of the 192 values');
await typeAnswer(q, 192);
check(`odd block: 192 is not the answer (the model keeps ${covers(96)})`, (await qText(q)).includes(`Not quite. The answer is ${covers(96)} values.`) && covers(96) === 128);
await typeAnswer(q, covers(96));
check('odd block: 128 is right, and the explanation names the stranded slot', (await qText(q)).includes('Correct.') && (await qText(q)).includes('slot 2 is never read again'));
check('odd block: a power of two keeps every value (model)', [64, 128, 256, 1024].every((B) => covers(B) === 2 * B));

q = await question('one result per column');
await pick(q, 0);
check('columns: a block per column is wrong, because its loads are 4,096 floats apart', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('4,096 floats apart'));
await pick(q, 1);
check('columns: one thread per column is right, coalesced along a row', (await qText(q)).includes('Correct.') && (await qText(q)).includes('coalesced'));

q = await question('Which can give different bits the second time?');
await pick(q, 2);
check('bits: "sum and max" is wrong, because a max never rounds', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('never rounds'));
await pick(q, 1);
check('bits: only the float sum, citing p. 245', (await qText(q)).includes('Correct.') && (await qText(q)).includes('p. 245'));

q = await question('Which reductions over the row does it actually need?');
await pick(q, 2);
check('greedy: a max is wrong, because the token is the index', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('the token is its index'));
await pick(q, 1);
check('greedy: one argmax over the logits is right', (await qText(q)).includes('Correct.') && (await qText(q)).includes('keep the order'));

await pause(700); // the app saves 400 ms after the last change
const ans = lastAnswers();
const ids = ['q-centre', 'q-oddblock', 'q-columns', 'q-bits', 'q-greedy'].map((s) => 'ch10/wrap/' + s);
check('all five answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 5);
check('the running count of answers appears after answering', (await body()).includes('So far in this chapter you have answered'));
await shot('questions', '.q');

// ---- the book's exercises
const rows = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) =>
  trs.map((tr) => ({ ex: tr.children[0].textContent.trim(), where: tr.children[2].textContent.trim(), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('the exercise table covers all six exercises', rows.map((r) => r.ex).join(' | ') === '1 | 2 | 3 | 4 | 5 | 6a | 6b', rows.map((r) => r.ex).join(' | '));
check('every exercise is tied to a section or lab by a link', rows.every((r) => r.links.length >= 1 && r.where.length > 10), JSON.stringify(rows));
const sectionOf = { 1: '#/ch10/10.4', 2: '#/ch10/10.4', 3: '#/ch10/10.4', 4: '#/ch10/10.9', 5: '#/ch10/10.9', '6a': '#/ch10/10.3', '6b': '#/ch10/10.4' };
check('each exercise points at the section that works it', rows.every((r) => r.links.includes(sectionOf[r.ex])), JSON.stringify(rows.map((r) => [r.ex, r.links])));
check('exercises 4 and 5 point at the labs as well', rows[3].links.includes('#/ch10/labs') && rows[4].links.includes('#/ch10/labs'));

// ---- errata
const errata = await page.$$eval('ul[data-list="errata"] > li', (lis) => lis.map((li) => li.textContent));
check(`the errata list has ${errata.length} items, covering the notes' list and the facts file`, errata.length === 13, String(errata.length));
for (const [name, frag] of [
  ['Fig. 10.10 output parameter', 'has no output parameter'],
  ['partialSums_s for small blocks', 'right only for 1,024 threads'],
  ['Fig. 10.18 N and *output', 'never uses its N'],
  ['the HIP mask', '0xffffffff'],
  ['736 against 864', '736'],
  ['ten to five', 'seven to five'],
  ['blockIdx.x on p. 229', 'blockIdx.x'],
  ['section 10.10 on p. 228', 'section 10.9 does it'],
  ['minimum (minimum)', 'the second is the maximum'],
  ['span of four steps', 'a span of four steps'],
  ['figure and line references', 'lines 17–19'],
  ['"more control divergence"', 'more control divergence'],
  ['dead CCCL links', 'nvidia.github.io/cccl/unstable/'],
]) check(`errata include ${name}`, errata.some((e) => e.includes(frag)));
check('readings of the PDF are marked as ours', text0.includes('our readings of one PDF'));
check('the wrap names what of the book is left out on purpose, with pages', ['World Cup', '(pp. 226–227)', 'p. 221', 'Thrust and CUB', 'chapter 12', '§20.5'].every((s) => text0.includes(s)));

// ---- project status, computed from the reports
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = (run?.results ?? []).filter((x) => x.lab === id && x.variant === variant && x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
const done = backends.filter((b) => labStatus(reports[b], 'ch10/lab3_softmax_argmax', 'work') === 'pass');
const m4 = await page.$eval('[data-m4]', (e) => ({ v: e.dataset.m4, text: e.textContent }));
check(`milestone M4 status is read from the reports (${done.join(',') || 'none'})`, m4.v === (done.join(',') || 'none') &&
  (backends.length === 0 ? m4.text.includes('No lab results') : done.length ? m4.text.includes('passes on') : m4.text.includes('does not pass')), JSON.stringify(m4));
check('the project paragraph names M4 and lab 10.3', text0.includes('milestone M4') && text0.includes("Lab 10.3's row-wise max, softmax and argmax"));
check('the next chapter is named and linked', text0.includes('Chapter 11 keeps the tree') && (await page.$$eval('.prose a[href="#/ch11"]', (as) => as.length)) === 1);

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch10/start', '#/ch10/10.2', '#/ch10/10.3', '#/ch10/10.4', '#/ch10/10.7', '#/ch10/10.8', '#/ch10/10.9', '#/ch10/10.10', '#/ch10/labs', '#/ch11']);
check(`links go to routes that exist (${links.join(', ')})`, links.length >= 6 && links.every((l) => known.has(l)), links.join(', '));
await page.evaluate(() => [...document.querySelectorAll('.prose a')].find((a) => a.getAttribute('href') === '#/ch10/labs').click());
await page.waitForFunction(() => document.querySelector('.page h1')?.textContent.includes('Labs'), { timeout: 10000 });
check('the link to the labs page opens it', (await page.$eval('.page h1', (e) => e.textContent)).includes('Labs'));

// ---- the whole chapter: two margin notes in a row open a blank gap in the text column
// (each note is its own grid row); notes that share a paragraph go in one NoteGroup.
{
  const routes = ['start', '10.1', '10.2', '10.3', '10.4', '10.5', '10.6', '10.7', '10.8', '10.9', '10.10', 'labs', 'wrap'];
  const bad = [];
  for (const r of routes) {
    await goto(page, 'ch10/' + r);
    const n = await page.evaluate(() => document.querySelectorAll('.prose > .note + .note').length);
    if (n) bad.push(`${r}: ${n}`);
  }
  check('no page of the chapter stacks two margin notes after one paragraph', bad.length === 0, bad.join(', '));
}

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch10/wrap');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'table[data-table="exercises"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
