// Browser check for chapter 7, "Summary and exercises": the carry-forward list, the mixed
// check (one kernel shown from a checked snippet, six questions), the map of the book's
// exercises, the errata list and the project status computed from the reports.
// Usage: APP_URL=http://localhost:<port> node e2e/ch07-wrap.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';

// Exercises 7 and 10, recomputed here from the book's counts (Wrap.test.ts pins the page's own constants).
const EX7_BLOCKS = Math.ceil(1000 / 28) ** 2;
const EX7_THREADS_712 = (28 + 5 - 1) ** 2;
const EX7_THREADS_715 = 28 ** 2;
const EX7_EXTRA = EX7_BLOCKS * (EX7_THREADS_712 - EX7_THREADS_715);
const EX10_OUT = 10 - (5 - 1);
const EX10_AI = (2 * EX10_OUT ** 3 * 5 ** 3) / (4 * (10 ** 3 + EX10_OUT ** 3));

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
const n = (v) => v.toLocaleString('en-US');

await goto(page, 'ch07/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary and exercises", numbered 7.7', h1.includes('Summary and exercises') && (await page.$eval('.page', (e) => e.textContent)).includes('7.7'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', 'The book’s exercises', 'Where the PDF looks wrong', 'The project so far'];
check('the page has its five parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
check('the carry-forward list has seven points and cites the book section',
  (await page.$$eval('.prose > ul', (uls) => uls[0].children.length)) === 7 && (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('§7.7, pp. 179–180'));
const text0 = await body();
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('the carry-forward list tells the reader to read the warp width, not write 32', text0.includes('read the width, never write 32'));
check('what was left out is named, with pages', text0.includes('pp. 170 to 171') && text0.includes('Fig. 7.8') && text0.includes('no L2 latency figure could be checked'));
check('every margin note follows a paragraph or list, never a bench, table or code block',
  await page.$$eval('.prose > aside.note', (as) => as.every((a) => { const p = a.previousElementSibling; return p && !p.matches('section.bench, figure.code, .wide, table'); })));

// ---- the kernel for the first question is the checked snippet
const file = await page.evaluate(() => fetch('/api/lab-file?path=' + encodeURIComponent('ch07/snippets/wrap_causal_conv1d.cu')).then((r) => (r.ok ? r.text() : 'MISSING')));
const shownCode = await page.$eval('figure.code pre', (e) => e.textContent);
check('the listing is labs/ch07/snippets/wrap_causal_conv1d.cu, line for line', file !== 'MISSING' && shownCode.trim() === file.trim(), shownCode.slice(0, 80));
check('the listing is labelled as not a lab file, front-end-checked and emulator-checked',
  (await page.$eval('figure.code .tabs', (e) => e.textContent)).includes('not a lab file') &&
  (await page.$eval('figure.code .tabs .status', (e) => e.textContent)) === 'front-end-checked; emulator-checked');
check('x and y are coloured as global memory, tile as shared, w as constant, the indices as registers',
  (await page.$$('figure.code .v-global')).length >= 3 && (await page.$$('figure.code .v-shared')).length >= 4 &&
  (await page.$$('figure.code .v-const')).length >= 2 && (await page.$$('figure.code .v-reg')).length >= 6);
await shot('code', 'figure.code');

// ---- the six questions, each answered wrong then right; the numbers come from the page's own constants (pinned by Wrap.test.ts)
check(`no question is answered at the start (${(await page.$$('.q')).length} questions)`, (await page.$$('.q')).length === 6 && (await page.$$('.q .verdict')).length === 0);

let q = await question('For which lengths');
await pick(q, 1);
check('early return: "whenever n is not a multiple of 128" is wrong, and the reason names threads 0, 1 and 2', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('threads 0, 1 and 2 all exist'));
await pick(q, 3);
check('early return: "the barrier hangs" is wrong, citing the exited-thread rule', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('ch04-facts A6, C3'));
await pick(q, 2);
check('early return: 1 or 2 outputs in the last block is right, and the fix keeps every thread to the barrier',
  (await qText(q)).includes('Correct.') && (await qText(q)).includes('n mod 128') && (await qText(q)).includes('reaches the barrier'));

q = await question("padding_mode='replicate'");
await pick(q, 1);
check('replicate: adding the edge pixel per skipped tap is wrong (weights differ)', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('its own weight'));
await pick(q, 0);
check('replicate: the clamp is right, and labelled as ours', (await qText(q)).includes('Correct.') && (await qText(q)).includes('the clamp is ours'));

q = await question('threadIdx.x / 32');
await pick(q, 0);
check('warp index: "1 and 1" is wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('two values'));
await pick(q, 1);
check('warp index: "1 and 2" is right, and the fix uses warpSize', (await qText(q)).includes('Correct.') && (await qText(q)).includes('threadIdx.x / warpSize'));

q = await question('Exercise 7 on one case');
await typeAnswer(q, EX7_BLOCKS * EX7_THREADS_712);
check(`exercise 7: ${n(EX7_BLOCKS * EX7_THREADS_712)} (all of Fig. 7.12's threads) is not the answer`, (await qText(q)).includes(`Not quite. The answer is ${EX7_EXTRA}`));
await typeAnswer(q, EX7_EXTRA);
check(`exercise 7: ${n(EX7_EXTRA)} is right, with ${n(EX7_BLOCKS)} blocks of ${n(EX7_THREADS_712)} and ${EX7_THREADS_715} threads`,
  (await qText(q)).includes('Correct.') && (await qText(q)).includes(`${n(EX7_BLOCKS)} blocks`) && (await qText(q)).includes(`${n(EX7_THREADS_712)} threads`) && (await qText(q)).includes(`${n(EX7_THREADS_715)} threads`));

q = await question('Exercise 10 asks');
await typeAnswer(q, 5.42);
check('exercise 10: the 2D figure 5.42 is not the answer', (await qText(q)).includes('Not quite.'));
await typeAnswer(q, Number(EX10_AI.toFixed(2)));
check(`exercise 10: ${EX10_AI.toFixed(2)} FLOP/B is right, with a ${EX10_OUT}-wide output tile and the ideal 31.25`,
  (await qText(q)).includes('Correct.') && (await qText(q)).includes(`output tile ${EX10_OUT}`) && (await qText(q)).includes('31.25'));

q = await question('within a few percent of the basic kernel');
await pick(q, 2);
check('same speed: "both compute-bound" is wrong, pointing to lab 5.1', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('Lab 5.1 measures yours'));
await pick(q, 1);
check('same speed: the filter was already cached is right', (await qText(q)).includes('Correct.') && (await qText(q)).includes('p. 168'));

await pause(700); // the app saves 400 ms after the last change
const ans = lastAnswers();
const ids = ['q-early-return', 'q-replicate', 'q-warp-index', 'q-ex7', 'q-3d', 'q-same-speed'].map((s) => 'ch07/wrap/' + s);
check('all six answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 6);
check('the running count of answers appears after answering', (await body()).includes('So far in this chapter you have answered'));
await shot('questions', '.q');

// ---- the book's exercises
const rows = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) =>
  trs.map((tr) => ({ ex: tr.children[0].textContent.trim(), where: tr.children[2].textContent.trim(), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('the exercise table covers all eleven exercises', rows.map((r) => r.ex).join(' | ') === '1 | 2 | 3 | 4, 5, 6 | 7 | 8, 9 | 10 | 11', rows.map((r) => r.ex).join(' | '));
const sectionOf = { 1: '#/ch07/7.1', 2: '#/ch07/7.1', 3: '#/ch07/7.1', '4, 5, 6': '#/ch07/7.1', 7: '#/ch07/7.5', 11: '#/ch07/7.5' };
check('each exercise worked in a section points at it', rows.every((r) => !sectionOf[r.ex] || r.links.includes(sectionOf[r.ex])), JSON.stringify(rows.map((r) => [r.ex, r.links])));
check('exercises 8 to 10 say plainly what is and is not done', rows[5].where.startsWith('not coded') && rows[6].where.includes('sizing and intensity only'));
check('the 3D port is described with the facts it runs into', text0.includes('13,500 bytes') && text0.includes('8 KB working set') && text0.includes('The emulator runs 3D blocks'));

// ---- errata
const errata = await page.$$eval('ul[data-list="errata"] > li', (lis) => lis.map((li) => li.textContent));
check(`the errata list has ${errata.length} items`, errata.length === 17, String(errata.length));
for (const [name, frag] of [
  ['the Gaussian', 'GaussianBlur'],
  ['x[3] for y[3]', '“for x[3]”'],
  ['M in the expansion', '321'],
  ['five arguments', 'five arguments'],
  ['Fig. 7.7', 'Fig. 7.7 does not compile'],
  ['Fig. 7.9 radius', 'Fig. 7.9'],
  ['7 cubed', '7³ = 343'],
  ['extern', '-rdc=true'],
  ['L1 size', '96 to'],
  ['F_c and F', 'declare the filter as F_c'],
  ['thread (6,6)', 'thread (6,6)'],
  ['stores in the denominator', 'denominator'],
  ['line 21', 'line 21'],
  ['chapter 20', 'chapter 20'],
  ['Exercise 1', 'Exercise 1 asks for P[0]'],
]) check(`errata include ${name}`, errata.some((e) => e.includes(frag)));
check('readings of the PDF are marked as ours', text0.includes('our readings of one PDF'));

// ---- project status, computed from the reports
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = (run?.results ?? []).filter((x) => x.lab === id && x.variant === variant && x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const backends = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const on = (id) => backends.filter((b) => labStatus(reports[b], id, 'work') === 'pass');
const l1 = on('ch07/lab1_constconv');
const l2 = on('ch07/lab2_tiledconv');
const proj = await page.$eval('[data-project]', (e) => ({ v: e.dataset.project, text: e.textContent }));
const wantV = [l1.length ? 'lab1' : '', l2.length ? 'lab2' : ''].filter(Boolean).join(',') || 'none';
const phrase = (x) => (x.length ? `passes on ${x.join(' and ')}` : 'has not passed on a device here yet');
check(`project status is read from the reports (${wantV})`, proj.v === wantV &&
  (backends.length === 0 ? proj.text.includes('No lab results') : proj.text.includes(`your lab 7.1 ${phrase(l1)}, and your lab 7.2 ${phrase(l2)}`)), JSON.stringify(proj));
check('the project paragraph says chapter 7 adds no milestone and the project stands at M2', proj.text.includes('adds no milestone') && proj.text.includes('M2'));
check('the next chapter is named and linked', text0.includes('picks up where this one stops') && (await page.$$eval('.prose a[href="#/ch08/start"]', (as) => as.length)) === 1);

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch07/7.1', '#/ch07/7.2', '#/ch07/7.5', '#/ch07/7.6', '#/ch07/labs', '#/ch08/start']);
check(`links go to routes that exist (${links.join(', ')})`, links.length >= 6 && links.every((l) => known.has(l)), links.join(', '));
await page.evaluate(() => [...document.querySelectorAll('.prose a')].find((a) => a.getAttribute('href') === '#/ch07/labs').click());
await page.waitForFunction(() => document.querySelector('.page h1')?.textContent.includes('Labs'), { timeout: 10000 });
check('the link to the labs page opens it', (await page.$eval('.page h1', (e) => e.textContent)).includes('Labs'));

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch07/wrap');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'table[data-table="exercises"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
