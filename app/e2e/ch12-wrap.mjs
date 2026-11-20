// Browser check for chapter 12, "Summary and exercises": the carry-forward list, the mixed
// check (one kernel shown from a checked snippet, five questions), the map of the book's
// exercises, the errata list and the project status computed from the reports. It ends by
// sweeping every page of the chapter for stacked margin notes, notes after a bench or a
// listing, sideways scroll at 1440 and 390 px, and console errors.
// Usage: APP_URL=http://localhost:<port> node e2e/ch12-wrap.mjs [screenshot-prefix] [--dark]
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

await goto(page, 'ch12/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary and exercises", numbered 12.9', h1.includes('Summary and exercises') && (await page.$eval('.page', (e) => e.textContent)).includes('12.9'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'In code you will read', 'A mixed check', "The book's exercises", 'Where the PDF looks wrong', 'The project so far'];
check('the page has its six parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
check('the carry-forward list has six points and cites the book section',
  (await page.$$eval('.prose > ul', (uls) => uls[0].children.length)) === 6 && (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('§12.9, p. 302'));
const text0 = await body();
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('the carry-forward list says what makes warp code port, and what in place needs',
  text0.includes('a ballot that every lane takes in uniform control flow') && text0.includes('a release, an acquire and a barrier after the reads'));
check('the byte counts in the carry-forward list are section 12.5\'s (30, 10, 6 at half kept)', text0.includes('30 bytes per key, a count-then-scatter design 10, one pass 6'));

// ---- the kernel for the first question is the checked snippet's candidatePairs
await page.waitForFunction(() => !document.querySelector('figure.code pre')?.textContent.startsWith('// loading'), { timeout: 10000 });
const file = await page.evaluate(() => fetch('/api/lab-file?path=' + encodeURIComponent('ch12/snippets/wrap_pairs.cu')).then((r) => (r.ok ? r.text() : 'MISSING')));
const shownCode = (await page.$eval('figure.code pre', (e) => e.textContent)).trim();
const start = file.indexOf('__global__ void candidatePairs');
const kernel = start >= 0 ? file.slice(start).trim() : 'MISSING';
check('the listing is candidatePairs from labs/ch12/snippets/wrap_pairs.cu, line for line', file !== 'MISSING' && shownCode === kernel, shownCode.slice(0, 80));
check('the listing is labelled as not a lab file, front-end-checked and wrong on purpose',
  (await page.$eval('figure.code .tabs', (e) => e.textContent)).includes('not a lab file') &&
  (await page.$eval('figure.code .tabs .status', (e) => e.textContent)) === 'front-end-checked; wrong on purpose');
check('the buffers and counters are coloured as global memory, the logit and column as registers',
  (await page.$$('figure.code .v-global')).length >= 6 && (await page.$$('figure.code .v-reg')).length >= 4);
await shot('code', 'figure.code');

// ---- the five questions, each answered wrong then right
check(`no question is answered at the start (${(await page.$$('.q')).length} questions)`, (await page.$$('.q')).length === 5 && (await page.$$('.q .verdict')).length === 0);

let q = await question('Where can that test fail?');
await pick(q, 0);
check('pairs: "nowhere" is wrong, because slot k comes from two counters', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('two different counters'));
await pick(q, 2);
check('pairs: "emu, reversed" is wrong, because emu runs each thread to its end', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('runs each thread to its end'));
await pick(q, 1);
check('pairs: interleaving between the two atomics is right; emu never shows it; the fix is one slot per pair',
  (await qText(q)).includes('Correct.') && (await qText(q)).includes('in all three thread orders') && (await qText(q)).includes('one slot per pair'));

q = await question('compact the keys in place?');
await pick(q, 0);
check('unstable in place: "never moves up" is wrong, because a counter hands slots out in any order', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('get the highest slot'));
await pick(q, 2);
check('unstable in place: a slot above its own index is right; grid barriers only under cooperative launch', (await qText(q)).includes('Correct.') && (await qText(q)).includes('cooperative launch'));

q = await question('In what order are the final candidates?');
await pick(q, 0);
check('chain: "column order, the last filter is stable" is wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('whatever order the atomic filter produced'));
await pick(q, 3);
check('chain: an order that can change is right; one stable filter on the AND gives column order', (await qText(q)).includes('Correct.') && (await qText(q)).includes('ANDed together'));

q = await question('Which way must the chain between blocks run?');
await pick(q, 1);
check('right-aligned: first to last is wrong, because keys move toward the end', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('slots of later blocks'));
await pick(q, 0);
check('right-aligned: last to first is right, the case of adding rows and columns', (await qText(q)).includes('Correct.') && (await qText(q)).includes('adding rows and columns'));

// ---- real code
const real = await page.$$eval('ul[data-list="real-code"] > li', (lis) => lis.map((li) => li.textContent));
check('five places in real code, each from the facts file', real.length === 5 && real.some((t) => t.includes('torch.nonzero')) && real.some((t) => t.includes('nonzero_static')) && real.some((t) => t.includes('lane-major')));
check('the stability rule is stated as ranking by kept keys before, not as staging', real.some((t) => t.includes('ranking every kept key by the number of kept keys before it in input order')) && !real.some((t) => /Staging the kept keys .* is what makes/.test(t)));
const carry = await page.$$eval('h2 + ul > li', (lis) => lis.map((li) => li.textContent).join(' '));
check('carry forward: in place on HIP and Metal is one block walking its range; several dispatches only out of place', carry.includes('compact in place with one block (threadgroup) that walks its range') && carry.includes('several dispatches are the documented-safe design out of place'), carry.slice(0, 200));

q = await question('carrying pairs');
await typeAnswer(q, 26);
check('pair bytes: 26 is not the answer (30 is)', (await qText(q)).includes('Not quite. The answer is 30 bytes.'));
await typeAnswer(q, 30);
check('pair bytes: 30 is right, and one pass moves 6', (await qText(q)).includes('Correct.') && (await qText(q)).includes('= 6, a fifth as much'));

await pause(700); // the app saves 400 ms after the last change
const ans = lastAnswers();
const ids = ['q-pairs', 'q-unstable-inplace', 'q-chain', 'q-rightalign', 'q-pairbytes'].map((s) => 'ch12/wrap/' + s);
check('all five answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 5);
check('the running count of answers appears after answering', (await body()).includes('So far in this chapter you have answered'));
await shot('questions', '.q');

// ---- the book's exercises
const rows = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) =>
  trs.map((tr) => ({ ex: tr.children[0].textContent.trim(), where: tr.children[2].textContent.trim(), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('the exercise table covers all four exercises', rows.map((r) => r.ex).join(' | ') === '1 | 2 | 3 | 4', rows.map((r) => r.ex).join(' | '));
const sectionOf = { 1: '#/ch12/12.5', 2: '#/ch12/12.7', 3: '#/ch12/12.6', 4: '#/ch12/12.6' };
check('each exercise points at the section that works it and at the labs', rows.every((r) => r.links.includes(sectionOf[r.ex]) && r.links.includes('#/ch12/labs') && r.where.length > 20), JSON.stringify(rows.map((r) => [r.ex, r.links])));

// ---- errata
const errata = await page.$$eval('ul[data-list="errata"] > li', (lis) => lis.map((li) => li.textContent));
check(`the errata list has ${errata.length} items, covering the notes' list and the facts file`, errata.length === 13, String(errata.length));
for (const [name, frag] of [
  ['Fig. 12.1 captions', 'both panels "(a)"'],
  ['Chapter 5 for coalescing', '"Chapter 5"'],
  ['uninitialised j', 'unsigned int j;'],
  ['__activemask snapshot', 'cannot be used that way'],
  ['HIP mask', '64-lane wavefronts'],
  ['Fig. 12.4 header and size()', 'legacy alias'],
  ['compiler claim', 'blog post last updated in 2017'],
  ['__ballot_sync parentheses', 'without parentheses'],
  ['Fig. 12.6 j and zero atomic', 'kept nothing'],
  ['counter zeroing', 'only for Fig. 12.2'],
  ['Fig. 12.5 order', 'green, orange, yellow, blue'],
  ['gridExclusiveScan', 'never defined'],
  ['k0 for k1', 'it is k1'],
  ['earlier for later', 'should be "later"'],
  ['dead CUB link', 'nvidia.github.io/cccl/unstable/'],
]) check(`errata include ${name}`, errata.some((e) => e.includes(frag)));
check('readings of the PDF are marked as ours', text0.includes('our readings of one PDF'));
check('the wrap names what of the book is left out on purpose, with pages',
  ['heap compaction', '(pp. 289–290)', 'In-place transposition', 'reference [2]', 'Fig. 12.4', 'Thrust and CUB'].every((s) => text0.includes(s)));

// ---- project status, computed from the reports
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = (run?.results ?? []).filter((x) => x.lab === id && x.variant === variant && x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
const done = backends.filter((b) => labStatus(reports[b], 'ch12/lab4_candidates', 'work') === 'pass');
const m5 = await page.$eval('[data-m5]', (e) => ({ v: e.dataset.m5, text: e.textContent }));
check(`milestone M5 status is read from the reports (${done.join(',') || 'none'})`, m5.v === (done.join(',') || 'none') &&
  (backends.length === 0 ? m5.text.includes('No lab results') : done.length ? m5.text.includes('passes on') : m5.text.includes('does not pass')), JSON.stringify(m5));
check('the project paragraph names M5 and lab 12.4', text0.includes('milestone M5') && text0.includes("Lab 12.4's candidate compaction"));
check('the next chapter is named and linked', text0.includes('Chapter 13 runs the other way') && (await page.$$eval('.prose a[href="#/ch13"]', (as) => as.length)) === 1);

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch12/12.3', '#/ch12/12.5', '#/ch12/12.6', '#/ch12/12.7', '#/ch12/labs', '#/ch06/6.1', '#/ch13']);
check(`links go to routes that exist (${links.join(', ')})`, links.length >= 6 && links.every((l) => known.has(l)), links.join(', '));
await page.evaluate(() => [...document.querySelectorAll('.prose a')].find((a) => a.getAttribute('href') === '#/ch12/labs').click());
await page.waitForFunction(() => document.querySelector('.page h1')?.textContent.includes('Labs'), { timeout: 10000 });
check('the link to the labs page opens it', (await page.$eval('.page h1', (e) => e.textContent)).includes('Labs'));

// ---- the whole chapter: margin notes follow paragraphs (two in a row, or one straight after a
// bench or a listing, opens a blank gap), every route has content, nothing scrolls sideways.
const routes = ['start', '12.1', '12.2', '12.3', '12.4', '12.5', '12.6', '12.7', '12.8', 'labs', 'wrap'];
{
  const stacked = [];
  const afterBlock = [];
  const placeholder = [];
  for (const r of routes) {
    await goto(page, 'ch12/' + r);
    const info = await page.evaluate(() => ({
      stacked: document.querySelectorAll('.prose > .note + .note').length,
      after: [...document.querySelectorAll('.prose > .note')].filter((n) => {
        const prev = n.previousElementSibling;
        return prev && (prev.matches('section.bench, figure.code, .scene, table, .wide') || prev.classList.contains('bench'));
      }).length,
      placeholder: document.querySelector('.page').textContent.includes('being written'),
    }));
    if (info.stacked) stacked.push(`${r}: ${info.stacked}`);
    if (info.after) afterBlock.push(`${r}: ${info.after}`);
    if (info.placeholder) placeholder.push(r);
  }
  check('no page of the chapter stacks two margin notes after one paragraph', stacked.length === 0, stacked.join(', '));
  check('no margin note comes straight after a bench, a listing, a scene or a table', afterBlock.length === 0, afterBlock.join(', '));
  check('no page of the chapter says it is being written', placeholder.length === 0, placeholder.join(', '));
}
for (const width of [1440, 390]) {
  await page.setViewport({ width, height: 900, deviceScaleFactor: 1 });
  const bad = [];
  for (const r of routes) {
    await goto(page, 'ch12/' + r);
    const info = await page.evaluate(() => ({ overflow: document.documentElement.scrollWidth - window.innerWidth, h1: document.querySelector('.page h1')?.textContent ?? '' }));
    if (info.overflow > 1 || !info.h1) bad.push(`${r}: ${JSON.stringify(info)}`);
  }
  check(`every page of the chapter at ${width} px has a heading and no sideways page scroll`, bad.length === 0, bad.join(' | '));
}
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch12/wrap');
await shot('phone', 'table[data-table="exercises"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
