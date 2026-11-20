// Browser check for chapter 4, "Summary and exercises": the carry-forward list, the mixed
// check (one kernel shown from a checked snippet, six questions), the map of the book's
// exercises, the errata list and the project status computed from the reports.
// Usage: APP_URL=http://localhost:<port> node e2e/ch04-wrap.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';
import { capacity, waves } from '../src/sim/blocksched.ts';
import { PRESETS, warpsNeeded } from '../src/sim/latency.ts';
import { BOOK, occupancy, pct } from '../src/sim/occupancy.ts';
import { classifyBlock, dims } from '../src/sim/warps.ts';

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

await goto(page, 'ch04/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary and exercises", numbered 4.9', h1.includes('Summary and exercises') && (await page.$eval('.page', (e) => e.textContent)).includes('4.9'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', "The book's exercises", 'Where the PDF looks wrong', 'The project so far'];
check('the page has its five parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
check('the carry-forward list has seven points and cites the book section',
  (await page.$$eval('.prose > ul', (uls) => uls[0].children.length)) === 7 && (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('§4.9, pp. 89–90'));
const text0 = await body();
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('the carry-forward list tells the reader to read the width, not write 32', text0.includes('read it, never write 32'));

// ---- the kernel for the first question is the checked snippet
const file = await page.evaluate(() => fetch('/api/lab-file?path=' + encodeURIComponent('ch04/snippets/warp_tail_sum.cu')).then((r) => (r.ok ? r.text() : 'MISSING')));
const shownCode = await page.$eval('figure.code pre', (e) => e.textContent);
check('the listing is labs/ch04/snippets/warp_tail_sum.cu, line for line', file !== 'MISSING' && shownCode.trim() === file.trim(), shownCode.slice(0, 80));
check('the listing is labelled as not a lab file and front-end-checked, never run',
  (await page.$eval('figure.code .tabs', (e) => e.textContent)).includes('not a lab file') &&
  (await page.$eval('figure.code .tabs .status', (e) => e.textContent)) === 'front-end-checked; never run');
check('the buffers are coloured as global memory, the array as shared memory, the indices as registers',
  (await page.$$('figure.code .v-global')).length >= 2 && (await page.$$('figure.code .v-shared')).length >= 6 && (await page.$$('figure.code .v-reg')).length >= 6);
await shot('code', 'figure.code');

// ---- the six questions, each answered wrong then right; numbers come from the models the page uses
check(`no question is answered at the start (${(await page.$$('.q')).length} questions)`, (await page.$$('.q')).length === 6 && (await page.$$('.q .verdict')).length === 0);

let q = await question('What does this chapter say about those six lines');
await pick(q, 0);
check('warp sync: "correct, a warp runs in step" is wrong, and the reason cites p. 81', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('p. 81'));
await pick(q, 1);
check('warp sync: "fails on 64-wide warps" is wrong, and the reason is AMD\'s lockstep', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('lockstep'));
await pick(q, 2);
check('warp sync: the lockstep assumption is the answer, with each vendor\'s position', (await qText(q)).includes('Correct.') && (await qText(q)).includes('__syncwarp()') && (await qText(q)).includes('A10, B5, C2'));

q = await question('Which is true?');
await pick(q, 0);
check('warp barrier: "legal, no warp diverges" is wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('about the block'));
await pick(q, 2);
check('warp barrier: uniform per warp and still illegal is right, quoting NVIDIA\'s rule', (await qText(q)).includes('Correct.') && (await qText(q)).includes('evaluates uniformly across the entire thread block'));

const TILE = dims(16, 8);
const div = (k, w) => classifyBlock(TILE, w, (_x, y) => y < k).states.filter((s) => s === 'divergent').length;
q = await question('For which K does no warp diverge');
await pick(q, 1);
check(`row mask: K = 2 is wrong, with its divergent warp at 64 (${div(2, 64)})`, (await qText(q)).includes('Not quite.') && (await qText(q)).includes(`${div(2, 64)} divergent warp`));
await pick(q, 2);
check(`row mask: K = 4 is right (${div(4, 32)} and ${div(4, 64)} divergent warps)`, (await qText(q)).includes('Correct.') && (await qText(q)).includes(`${div(4, 32)} divergent warps at 32 and ${div(4, 64)}`) && div(4, 32) === 0 && div(4, 64) === 0);

const before = warpsNeeded(PRESETS.loadHeavy.work, PRESETS.loadHeavy.latency);
const after = warpsNeeded(4 * PRESETS.loadHeavy.work, PRESETS.loadHeavy.latency);
q = await question('How many warps keep the slot busy now');
check(`packing: the prompt states the model's ${before} warps before packing`, (await qText(q)).includes(`so ${before} warps keep one issue slot busy`));
await typeAnswer(q, before / 4);
check(`packing: ${before / 4} (a quarter of ${before}) is not the answer`, (await qText(q)).includes(`Not quite. The answer is ${after} warps.`));
await typeAnswer(q, after);
check(`packing: ${after} warps is right`, (await qText(q)).includes('Correct.') && (await qText(q)).includes(`down from ${before}`));

q = await question('Will that help?');
await pick(q, 0);
check(`small grid: "yes, twice the warps" is wrong, because ${capacity(40, 2)} blocks could already run`, (await qText(q)).includes('Not quite.') && (await qText(q)).includes(`${capacity(40, 2)} could already run at once`));
await pick(q, 1);
check(`small grid: no, with ${waves(64, capacity(40, 2))} and ${waves(64, capacity(40, 4))} of a wave`, (await qText(q)).includes('Correct.') && (await qText(q)).includes(`is ${waves(64, 80)} of`) && (await qText(q)).includes(`it is ${waves(64, 160)}`));

const k = { threads: 256, regs: 40, shared: 0 };
const a = occupancy({ ...BOOK, shared: null }, k);
const b = occupancy({ ...BOOK, threadSlots: 1536, blockSlots: 16, shared: null }, k);
q = await question('What are its occupancies?');
await pick(q, 1);
check(`two devices: "the other keeps more warps" is wrong (${a.residentWarps} on each)`, (await qText(q)).includes('Not quite.') && (await qText(q)).includes(`${a.residentWarps} on each`) && a.residentWarps === b.residentWarps);
await pick(q, 2);
check(`two devices: ${pct(a.byWarps)}% and ${pct(b.byWarps)}% with the same warps is right`, (await qText(q)).includes('Correct.') && (await qText(q)).includes(`${a.residentWarps} of ${a.warpSlots} warp slots, ${pct(a.byWarps)}%`) && (await qText(q)).includes(`${b.residentWarps} of ${b.warpSlots}, ${pct(b.byWarps)}%`));

await pause(700); // the app saves 400 ms after the last change
const ans = lastAnswers();
const ids = ['q-warpsync', 'q-warpbarrier', 'q-rowmask', 'q-pack', 'q-smallgrid', 'q-twodevices'].map((s) => 'ch04/wrap/' + s);
check('all six answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 6);
check('the running count of answers appears after answering', (await body()).includes('So far in this chapter you have answered'));
await shot('questions', '.q');

// ---- the book's exercises
const rows = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) =>
  trs.map((tr) => ({ ex: tr.children[0].textContent.trim(), where: tr.children[2].textContent.trim(), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('the exercise table covers all nine exercises', rows.map((r) => r.ex).join(' | ') === '1 (a, b) | 1 (c, d, e) | 2, 3 | 4 | 5 | 6 | 7, 8 | 9', rows.map((r) => r.ex).join(' | '));
check('every exercise is tied to a section or lab by a link', rows.every((r) => r.links.length >= 1 && r.where.length > 10), JSON.stringify(rows));
const sectionOf = { '1 (a, b)': '#/ch04/4.4', '1 (c, d, e)': '#/ch04/4.5', '2, 3': '#/ch04/4.5', 4: '#/ch04/4.3', 6: '#/ch04/4.7', '7, 8': '#/ch04/4.7', 9: '#/ch04/4.8' };
check('each exercise points at the section that works it', rows.every((r) => !sectionOf[r.ex] || r.links.includes(sectionOf[r.ex])), JSON.stringify(rows.map((r) => [r.ex, r.links])));
check('exercise 5 points at lab 4.3 as well', rows[4].links.includes('#/ch04/labs'));

// ---- errata
const errata = await page.$$eval('ul[data-list="errata"] > li', (lis) => lis.map((li) => li.textContent));
check(`the errata list has ${errata.length} items, covering the notes' list and the facts file`, errata.length === 13, String(errata.length));
for (const [name, frag] of [
  ['the section pointer on p. 69', '"Section 4.6"'],
  ['Fig. 4.4', '__global__'],
  ['Fig. 4.10', 'threads 1 and 7'],
  ['1 of 314', 'one of 314 launched'],
  ['the 200 × 150 picture', '75 or 100'],
  ['two definitions of occupancy', 'with threads on p. 90'],
  ['regsPerBlock', 'regsPerMultiprocessor'],
  ['clockRate', 'CUDA 13.0'],
  ['integrated GPUs', 'integrated GPUs'],
]) check(`errata include ${name}`, errata.some((e) => e.includes(frag)));
check('readings of the PDF are marked as ours', text0.includes('our readings of one PDF'));
check('the unconfirmed device figure is presented as the book\'s, and the GPC split as given by nobody', text0.includes('four processing blocks of 32') && text0.includes('the book gives no split of its 132 SMs among 8 GPCs, and NVIDIA does not publish one'));
check('the wrap names what of the book is left out on purpose, with pages', ['intrinsic functions (p. 70)', 'Flynn\'s taxonomy', '(pp. 78–79)', 'post-office analogy for latency tolerance (p. 83)', 'hotel analogy for querying a device (p. 87)'].every((s) => text0.includes(s)));

// ---- project status, computed from the reports
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = (run?.results ?? []).filter((x) => x.lab === id && x.variant === variant && x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
const both = backends.filter((b) => labStatus(reports[b], 'ch03/lab3_naive_gemm', 'work') === 'pass' && labStatus(reports[b], 'ch04/lab1_launchconfig', 'work') === 'pass');
const m1 = await page.$eval('[data-m1]', (e) => ({ v: e.dataset.m1, text: e.textContent }));
check(`milestone M1 status is read from the reports (${both.join(',') || 'none'})`, m1.v === (both.join(',') || 'none') &&
  (backends.length === 0 ? m1.text.includes('No lab results') : both.length ? m1.text.includes('both pass on') : m1.text.includes('do not both pass')), JSON.stringify(m1));
check('the project paragraph names M1 and labs 3.3 and 4.1', text0.includes('milestone M1') && text0.includes("Lab 3.3's naive GEMM and lab 4.1's launch"));
check('the next chapter is named and linked', text0.includes('Chapter 5 turns to the memory on the chip') && (await page.$$eval('.prose a[href="#/ch05/start"]', (as) => as.length)) === 1);

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch04/start', '#/ch04/4.1', '#/ch04/4.2', '#/ch04/4.3', '#/ch04/4.4', '#/ch04/4.5', '#/ch04/4.6', '#/ch04/4.7', '#/ch04/4.8', '#/ch04/labs', '#/ch05/start']);
check(`links go to routes that exist (${links.join(', ')})`, links.length >= 6 && links.every((l) => known.has(l)), links.join(', '));
await page.evaluate(() => [...document.querySelectorAll('.prose a')].find((a) => a.getAttribute('href') === '#/ch04/labs').click());
await page.waitForFunction(() => document.querySelector('.page h1')?.textContent.includes('Labs'), { timeout: 10000 });
check('the link to the labs page opens it', (await page.$eval('.page h1', (e) => e.textContent)).includes('Labs'));

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch04/wrap');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'table[data-table="exercises"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
