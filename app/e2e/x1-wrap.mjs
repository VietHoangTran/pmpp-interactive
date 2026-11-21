// Browser check for module X1's summary: the carry-forward list, the mixed check (five
// questions, each answered wrong then right, with the numbers recomputed here from the
// models), the map of the module's tasks with links that open written pages, where the sources
// disagree, and the project status computed from the reports.
// Usage: APP_URL=http://localhost:<port> node e2e/x1-wrap.mjs [screenshot-prefix] [--dark]
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
const question = async (part) => (await page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(part)), part)).asElement();
const qText = (q) => q.evaluate((e) => e.textContent);
const press = (q, label) => q.evaluate((e, label) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === label).click(), label);
const typeAnswer = async (q, v) => {
  const input = await q.$('input[type=number]');
  await input.click({ clickCount: 3 });
  await input.type(String(v));
  await press(q, 'Check');
  await pause(120);
};
const pick = async (q, index) => {
  await q.evaluate((e, i) => e.querySelectorAll('input[type=radio]')[i].click(), index);
  await pause(40);
  await press(q, 'Check');
  await pause(120);
};

await goto(page, 'x1/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('[data-m5-own="loading"]'), { timeout: 30000 });
const body = () => page.$eval('.prose', (e) => e.textContent);
const M = await page.evaluate(async () => {
  const t = await import('/src/sim/topkselect.ts');
  const p = await import('/src/sim/penalties.ts');
  const c = await import('/src/sim/chain.ts');
  return { passes: t.MAX_PASSES, select: t.SELECT_PER_KEY, pass: p.bytesPerLogit(1), labs: Object.values(c.LAB).map((l) => [l.num, l.chapter, l.title]) };
});

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary"', h1.trim().endsWith('Summary'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', "The module's exercises", 'Where the sources disagree', 'The project so far'];
check('the page has its five parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
const carry = await page.$$eval('.prose > ul', (uls) => [...uls[0].children].map((li) => li.textContent));
check('the carry-forward list has six points, from the chain to one tie rule', carry.length === 6 &&
  carry[0].includes('chain of row-wise stages') && carry[2].includes('Top-k is a selection') && carry[3].includes('shorter prefix') && carry[5].includes('One tie rule'), JSON.stringify(carry.map((c) => c.slice(0, 30))));
const notes = await page.$$eval('.prose aside.note', (as) => as.map((a) => a.textContent));
check('the module is labelled as ours, with the book\'s anchor §20.1, p. 481', notes.some((t) => t.includes("Ours, not the book's")) && notes.some((t) => t.includes('§20.1, p. 481')));
const text0 = await body();
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('no device figure is typed into the page', !/\d+(\.\d+)? (GB\/s|KiB)\b|\d+\.\d+ ms\b/.test(text0));

// ---- the five questions: the choices answered wrong then right, the number once
check('five questions, none answered at the start', (await page.$$('.q')).length === 5 && (await page.$$('.q .verdict')).length === 0);
let q = await question('lab X1.1\'s select applies the penalties on the fly');
const fused = (M.passes + 1) * (4 + 4);
// A numeric question takes one answer and then shows the right one, so it is answered once, right.
await typeAnswer(q, fused);
check(`fuse-select: ${fused} bytes, against ${M.pass} + ${M.select} = ${M.pass + M.select} apart`, (await qText(q)).includes('Correct.') &&
  (await qText(q)).includes(`5 × 8 = ${fused} bytes`) && (await qText(q)).includes(`move ${M.pass} for the penalty`) && (await qText(q)).includes(`${M.select} for the select, ${M.pass + M.select} in all`), await qText(q));
q = await question('You log every token of both runs');
await pick(q, 1);
check('near-boundary: "different Philox libraries" is wrong, M5 uses its own', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('M5 uses no library generator'));
await pick(q, 2);
check('near-boundary: "a defect" is wrong, decisive draws must match', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('decisive draws, which must match exactly'));
await pick(q, 0);
check('near-boundary: nothing is wrong; exp, order and rounding; digest covers decisive draws', (await qText(q)).includes('Correct.') && (await qText(q)).includes('round toward zero') && (await qText(q)).includes('the digest covers those'));
q = await question('A model\'s output projection is padded');
await pick(q, 2);
check('padding: minus infinity is wrong under Metal fast math', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('ch10-facts C3'));
await pick(q, 3);
check('padding: −FLT_MAX is wrong, it overflows at T below 1', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('overflows to minus infinity'));
await pick(q, 4);
check('padding: a smaller vocab is wrong, it is also the row stride', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('row stride'));
await pick(q, 0);
check('padding: a large finite negative logit; the lab plants such logits', (await qText(q)).includes('Correct.') && (await qText(q)).includes('plant exactly such logits'));
q = await question('four completions of one prompt');
await pick(q, 1);
check('n samples: the block index is not an input to u', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('(seed, stream, step) alone'));
await pick(q, 0);
check('n samples: the stream id; one stream per row as in lab X1.3', (await qText(q)).includes('Correct.') && (await qText(q)).includes('one stream per row'));
q = await question('You port lab X1.1\'s gather to HIP');
await pick(q, 1);
check('popc: "does not compile" is wrong, the cut is silent', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('The cut is silent'));
await pick(q, 0);
check('popc: lanes 32 to 63 collide; __popcll is the fix', (await qText(q)).includes('Correct.') && (await qText(q)).includes('__popcll'));
await pause(700); // the app saves 400 ms after the last change
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
const ids = ['q-fuse-select', 'q-near-boundary', 'q-pad-vocab', 'q-n-samples', 'q-hip-popc'].map((s) => 'x1/wrap/' + s);
check('all five answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 5);
check('the running count of answers appears after answering', (await body()).includes('So far in this module you have answered'));
await shot('questions', '.q');

// ---- the module's tasks
const exRows = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) =>
  trs.map((tr) => ({ task: tr.children[0].textContent.trim(), lab: tr.children[2].textContent.trim(), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('the task table says plainly the book has none, and has eleven rows', text0.includes('The book has no exercises for this module') && exRows.length === 11, String(exRows.length));
const sections = ['X1.1', 'X1.2', 'X1.3', 'X1.4', 'X1.5', 'X1.6', 'X1.7', 'X1.8'];
check('every section of the module is linked from the task table', sections.every((s) => exRows.some((r) => r.links.includes(`#/x1/${s}`))), JSON.stringify(exRows.map((r) => r.links)));
check('every lab of the module is named in the task table', ['lab X1.1', 'lab X1.2', 'lab X1.3', 'lab X1.4'].every((l) => exRows.some((r) => r.lab.includes(l))));

// ---- where the sources disagree
const disagree = await page.$$eval('.prose > ul', (uls) => [...uls[1].children].map((li) => li.querySelector('b')?.textContent));
check('the disagreements are grouped: order, ties, repetition penalty, the draw, random numbers',
  JSON.stringify(disagree) === JSON.stringify(['Order.', 'Ties.', 'Repetition penalty.', 'The draw.', 'Random numbers.']), JSON.stringify(disagree));
check('the page says the module has no PDF errata, and where its own facts file and plan were wrong', text0.includes('This module has no PDF to be wrong') && text0.includes('where our own facts file and plan were wrong') && text0.includes('1 input in 256 (our count)'));
check('the disagreements are dated and sourced to the facts file', notes.some((t) => t.includes('Checked 2026-11-20 in the facts pass (notes/today/x1-facts.md')));

// ---- the project status, from the reports
const NAME = (b) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = run ? run.results.filter((x) => x.lab === id && x.variant === variant && x.tests.length) : [];
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const realB = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const passingOn = (id) => [...realB, 'emu'].filter((b) => labStatus(reports[b], id, 'work') === 'pass');
const own = passingOn('x1/lab3_sampler');
const ownText = await page.$eval('[data-m5-own]', (e) => ({ v: e.dataset.m5Own, t: e.textContent }));
check(`the sampler line reads the reports: ${own.join(',') || 'none'}`, ownText.v === (own.join(',') || 'none') &&
  (Object.keys(reports).length === 0 ? ownText.t.includes('No lab results') : own.length ? ownText.t.includes(own.map(NAME).join(' and ')) : ownText.t.includes('does not pass on any backend')), JSON.stringify(ownText));
const DIRS = { '9.3': 'ch09/lab3_tokencounts', '10.3': 'ch10/lab3_softmax_argmax', '11.6': 'ch11/lab6_topp', '12.4': 'ch12/lab4_candidates', '14.3': 'ch14/lab3_sortrows', 'X1.1': 'x1/lab1_topk', 'X1.2': 'x1/lab2_filters', 'X1.3': 'x1/lab3_sampler' };
const m5 = await page.$$eval('table[data-table="m5"] tbody tr', (trs) => trs.map((tr) => ({ num: tr.dataset.m5, pass: tr.dataset.pass, href: tr.querySelector('a').getAttribute('href'), title: tr.children[0].textContent })));
check(`the M5 table has chain.ts's ${M.labs.length} labs, with their titles and links`, m5.length === M.labs.length &&
  M.labs.every(([num, ch, title]) => m5.some((r) => r.num === num && r.href === `#/${ch}/labs` && r.title === title)), JSON.stringify(m5));
check('each M5 row reads the reader\'s own status from the reports', m5.every((r) => r.pass === (passingOn(DIRS[r.num]).join(',') || 'none')), JSON.stringify(m5.map((r) => [r.num, r.pass])));
check('the next chapter is named with its pages, not linked', text0.includes('chapter 20 of the book, on large language models (pp. 477–512)') &&
  !(await page.$$eval('.prose a', (as) => as.some((a) => a.getAttribute('href')?.startsWith('#/ch20')))));
await shot('project', 'table[data-table="m5"]');

// ---- margin notes, links, narrow screen
const misplaced = await page.$$eval('.prose > aside.note', (as) => as.filter((a) => {
  const prev = a.previousElementSibling;
  return prev && !['P', 'UL', 'OL', 'ASIDE'].includes(prev.tagName);
}).map((a) => `${a.previousElementSibling.tagName}.${a.previousElementSibling.className}`));
check('every margin note follows a paragraph, a list or another note', misplaced.length === 0, misplaced.join(', '));
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
check('no sideways page scroll at 390 px', await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
if (out) await page.screenshot({ path: `${out}-narrow.png` });
await page.setViewport({ width: 1440, height: 1000, deviceScaleFactor: 1 });
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
for (const l of links) {
  await goto(page, l.slice(2));
  const ok = await page.evaluate(() => !!document.querySelector('.page h1') && !document.querySelector('.page').textContent.includes('This section is being written'));
  check(`link ${l} opens a written page`, ok);
}

// ---- the whole module: two margin notes in a row open a blank gap in the text column;
// notes that share a paragraph go in one NoteGroup.
{
  const stacked = [];
  for (const r of ['start', 'X1.1', 'X1.2', 'X1.3', 'X1.4', 'X1.5', 'X1.6', 'X1.7', 'X1.8', 'labs', 'wrap']) {
    await goto(page, 'x1/' + r);
    const n = await page.evaluate(() => document.querySelectorAll('.prose > .note + .note').length);
    if (n) stacked.push(`${r}: ${n}`);
  }
  check('no page of the module stacks two margin notes after one paragraph', stacked.length === 0, stacked.join(', '));
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
