// Browser check for chapter 2, "Summary and exercises": the carry-forward list and the
// table of the book's four groups, the mixed check (two kernels shown from checked
// snippets, five questions), the map of the book's exercises, the errata and today's
// differences, the project status computed from the reports, and links. It ends with a
// sweep of every route of the chapter at 1440 and 390 px: content present, no sideways
// page scroll, no lab file that failed to load, nothing "being written".
// Usage: node e2e/ch02-wrap.mjs [screenshot-prefix] [--dark]
// Environment: APP_URL (default http://localhost:5173).
import { open, goto } from './browser.mjs';
import { ceilDiv, launch } from '../src/sim/launch.ts';

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
// Narration audio is generated centrally; until it exists the scenes of 2.1, 2.3, 2.5 and
// 2.6 ask for a manifest and get a 404. That, and only that, is allowed.
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });
const expected404 = (u) => /\/content\/ch02\/narration\/(manifest\.json|[a-z-]+-\d\.mp3)$/.test(u);

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

await goto(page, 'ch02/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure
check('the page title is the summary, numbered 2.8', (await page.$eval('.page h1', (e) => e.textContent)).includes('Summary and exercises'));
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', "The book's exercises", 'Where the PDF looks wrong',
  "Where today's documentation differs from the page", 'What this chapter left out', 'The project so far'];
check('the page has its seven parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
check('the carry-forward list has seven points and cites the book section',
  (await page.$$eval('.prose > ul', (uls) => uls[0].children.length)) === 7 && (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('§2.8, p. 40'));
const text0 = await body();
check('it states the index, the guard, the integer block count and the check after a launch',
  text0.includes('i = blockIdx.x * blockDim.x + threadIdx.x') && text0.includes('i < n') && text0.includes('(n + B - 1) / B') && text0.includes('read the error state right after it'));
const groups = await page.$$eval('table[data-table="four-groups"] tbody tr', (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
check("the book's four groups are set against HIP and Metal",
  groups.length === 4 && groups[0][0] === 'Function declarations' && groups[1][1].includes('<<<') && groups[2][3].includes('[[thread_position_in_grid]]') && groups[3][2].includes('hipMalloc'),
  JSON.stringify(groups));
check('nothing on the page says it is being written, and no lab file failed to load', !/being written|could not load/i.test(text0));

// ---- the two kernels are the checked snippets
const codes = await page.$$eval('figure.code pre', (ps) => ps.map((p) => p.textContent));
for (const [i, name] of [[0, 'saxpy_stub_defect.cu'], [1, 'count_above_race.cu']]) {
  const file = await page.evaluate((p) => fetch('/api/lab-file?path=' + encodeURIComponent(p)).then((r) => (r.ok ? r.text() : 'MISSING')), 'ch02/snippets/' + name);
  check(`listing ${i + 1} is labs/ch02/snippets/${name}, line for line`, file !== 'MISSING' && codes[i]?.trim() === file.trim(), (codes[i] ?? '').slice(0, 80));
}
const statuses = await page.$$eval('figure.code .tabs .status', (s) => s.map((x) => x.textContent));
check('both listings are labelled as not lab files, front-end-checked, and defective on purpose',
  statuses.length === 2 && statuses.every((s) => s === 'front-end-checked; has a defect on purpose') &&
  (await page.$$eval('figure.code .tabs', (t) => t.every((x) => x.textContent.includes('not a lab file')))));
check('device arrays are coloured as global memory and the index as a register',
  (await page.$$('figure.code .v-global')).length >= 6 && (await page.$$('figure.code .v-reg')).length >= 4);
await shot('code', 'figure.code');

// ---- the five questions, each answered once wrong and once right
const logits = launch(50257, 256, ceilDiv(50257, 256));
check(`no question is answered at the start (${(await page.$$('.q')).length} questions)`, (await page.$$('.q')).length === 5 && (await page.$$('.q .verdict')).length === 0);

let q = await question('What does y_h hold afterwards?');
await pick(q, 2);
check('saxpy: "exactly 3, because cudaMalloc zeroes" is wrong, and the reason quotes NVIDIA', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('The memory is not cleared.'));
await pick(q, 1);
check('saxpy: "3 plus whatever y_d held" is right', (await qText(q)).includes('Correct.'));
check('  the explanation names the missing copy, the emulator-checked run and the Metal zero fill',
  (await qText(q)).includes('cudaMemcpy(y_d, y_h, size, cudaMemcpyHostToDevice)') && (await qText(q)).includes('emulator-checked') && (await qText(q)).includes('newBufferWithLength'));

q = await question('stop at the guard?');
await typeAnswer(q, logits.blocks);
check(`logits: ${logits.blocks} (the block count) is not the answer`, (await qText(q)).includes(`Not quite. The answer is ${logits.surplus} threads.`));
await typeAnswer(q, logits.surplus);
check(`logits: ${logits.surplus} of ${logits.threads} threads stop at the guard, from the model`,
  (await qText(q)).includes('Correct.') && (await qText(q)).includes(logits.threads.toLocaleString('en-US')) && (await qText(q)).includes(`${logits.blocks} blocks`));

q = await question('What can you expect of');
await pick(q, 0);
check('count: "it is 300" is wrong, and the reason is a lost update', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('one of the additions is lost'));
await pick(q, 2);
check('count: "less than 300, and it can change" is right', (await qText(q)).includes('Correct.') && (await qText(q)).includes('It can be less than 300'));
check('count: the source cites chapter 4 §4.4 for threads of a block in any order', (await qText(q)).includes('ch. 4 §4.4, p. 74'));
check('  the explanation gives both emulator flavours\' one-off results and points to atomics and reduction',
  (await qText(q)).includes('gave 300 every time') && (await qText(q)).includes('188, 191 and 202') && (await qText(q)).includes('chapter 9') && (await qText(q)).includes('chapter 10'));

q = await question('What goes wrong?');
await pick(q, 2);
check('table bytes: "the kernel\'s int i overflows" is wrong; the largest index fits', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('614,399,999'));
await pick(q, 1);
check('table bytes: the byte count does not fit in an int, and the wrapped value is shown',
  (await qText(q)).includes('Correct.') && (await qText(q)).includes('2,457,600,000') && (await qText(q)).includes('-1,837,367,296') && (await qText(q)).includes('2,147,483,647'));

q = await question('does cudaMemcpy return');
await pick(q, 0);
check('overlap: "about 5 ms" is wrong: the launch returns at once', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('returns before the kernel has run'));
await pick(q, 1);
check('overlap: "about 3 ms" is right, and the wait is worded as our reading of NVIDIA\'s statements',
  (await qText(q)).includes('Correct.') && (await qText(q)).includes('our reading of those three statements') && (await qText(q)).includes('waitUntilCompleted'));

await pause(700); // the app saves 400 ms after the last change
const ans = lastAnswers();
const ids = ['q-inout', 'q-logits', 'q-count', 'q-bytes-int', 'q-overlap'].map((s) => 'ch02/wrap/' + s);
check('all five answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 5);
check('the running count of answers appears after answering', (await body()).includes('So far in this chapter you have answered'));
await shot('questions', '.q');

// ---- the book's exercises
const rows = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) =>
  trs.map((tr) => ({ ex: tr.children[0].textContent.trim(), where: tr.children[2].textContent.trim(), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('the exercise table covers all ten exercises', rows.map((r) => r.ex).join(' | ') === '1 | 2, 3 | 4 | 5, 6 | 7 | 8 | 9 | 10', rows.map((r) => r.ex).join(' | '));
check('every exercise is tied to a section by a link', rows.every((r) => r.links.length >= 1 && r.where.length > 10), JSON.stringify(rows));
const where = Object.fromEntries(rows.map((r) => [r.ex, r.links]));
check('exercises 1 to 3 point at 2.5, 4 and 9 at 2.6, 5 to 8 at 2.4, 10 at 2.7, 8 also at the labs',
  where['1'].includes('#/ch02/2.5') && where['2, 3'].includes('#/ch02/2.5') && where['4'].includes('#/ch02/2.6') && where['9'].includes('#/ch02/2.6') &&
  where['5, 6'].includes('#/ch02/2.4') && where['7'].includes('#/ch02/2.4') && where['8'].includes('#/ch02/2.4') && where['8'].includes('#/ch02/labs') && where['10'].includes('#/ch02/2.7'));

// ---- errata, today's differences, omissions
const errata = await page.$$eval('ul[data-list="errata"] > li', (lis) => lis.map((li) => li.textContent.replace(/\s+/g, ' ')));
check(`the errata list has ${errata.length} items, the three in code first`,
  errata.length === 10 && errata[0].includes('float *d_A *d_B, *d_C;') && errata[1].includes('tests error') && errata[2].includes('vectAdd') && errata[2].includes('ceil(n/256)'),
  String(errata.length));
check('it lists the swapped captions, the naming of Figs. 2.5, 2.12 and 2.13, the guard of Fig. 2.9 and the p. 37 slips',
  errata.some((e) => e.includes('swapped')) && errata.some((e) => e.includes('Figs. 2.5, 2.12 and 2.13')) && errata.some((e) => e.includes('Fig. 2.9 (p. 33) has no')) && errata.some((e) => e.includes('vecAddVecKernel')));
check('readings of the PDF are marked as ours, and the two non-errors are set apart',
  (await body()).includes('our readings of one PDF') && (await body()).includes('Two things look like slips and are not'));
const today = await page.$$eval('ul[data-list="today"] > li', (lis) => lis.map((li) => li.textContent.replace(/\s+/g, ' ')));
check(`today's differences: ${today.length} items, asynchronous launch first, each with the section that takes it up`,
  today.length === 9 && today[0].includes('asynchronous') && today.some((t) => t.includes('cudaMemcpyDefault')) && today.some((t) => t.includes('__tile__')) && today.some((t) => t.includes('not documented')));
const notes = await page.$$eval('.prose aside.note', (as) => as.map((a) => a.textContent));
check('the differences are checked against the facts file, with its date', notes.some((n) => n.includes('Checked 2026-11-20 (notes/today/ch02-facts.md, A1, A2')));
const leftOut = await page.$$eval('.prose > ul', (uls) => uls.at(-1).textContent);
check('what was left out is listed with where to find it', ['chapter 3', 'chapter 23', 'appendix C', 'chapters 6 and 15', 'nvrtc'].every((c) => leftOut.includes(c)), leftOut.slice(0, 120));

// ---- project status, computed from the reports
const M0 = 'ch02/lab3_elementwise';
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = (run?.results ?? []).filter((x) => x.lab === id && x.variant === variant && x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const devices = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const passing = devices.filter((b) => labStatus(reports[b], M0, 'work') === 'pass');
const wantM0 = devices.length === 0 && !reports.emu
  ? 'No lab results are recorded on this machine yet.'
  : passing.length
    ? `Your own version passes on ${passing.join(' and ')} on this machine.`
    : labStatus(reports.emu, M0, 'work') === 'pass'
      ? 'Your own version passes on the emulator; it has not passed on a device on this machine yet.'
      : 'Your own version has not passed on this machine yet.';
const m0 = await page.$eval('[data-m0]', (e) => e.textContent);
check(`milestone M0 status is read from the reports: "${wantM0}"`, m0 === wantM0, m0);
check('the project paragraph names M0, its three operations and the next milestone', (await body()).includes('milestone M0') && (await body()).includes('residual add') && (await body()).includes('M1'));

// ---- margin notes follow text
const after = await page.$$eval('.prose > aside.note', (as) => as.map((a) => a.previousElementSibling?.tagName ?? ''));
check('every margin note directly follows a paragraph or a list', after.every((t) => ['P', 'UL', 'OL'].includes(t)), after.join(','));

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch02/start', '#/ch02/2.1', '#/ch02/2.2', '#/ch02/2.3', '#/ch02/2.4', '#/ch02/2.5', '#/ch02/2.6', '#/ch02/2.7', '#/ch02/labs', '#/ch02/wrap', '#/ch03/start']);
check(`links go to sections that exist (${links.join(', ')})`, links.length >= 8 && links.every((l) => known.has(l)), links.join(', '));
await page.evaluate(() => [...document.querySelectorAll('.prose a')].find((a) => a.getAttribute('href') === '#/ch02/labs').click());
await page.waitForFunction(() => document.querySelector('.page h1')?.textContent.includes('Labs'), { timeout: 10000 });
check('the link to the labs page opens it', (await page.$eval('.page h1', (e) => e.textContent)).includes('Labs'));

// ---- every route of the chapter, desktop and phone
const routes = ['start', '2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7', 'labs', 'wrap'].map((s) => 'ch02/' + s);
for (const width of [1440, 390]) {
  await page.setViewport({ width, height: 900, deviceScaleFactor: 1 });
  for (const route of routes) {
    await goto(page, route);
    await pause(600); // lab files and reports arrive after the first render
    const info = await page.evaluate(() => ({
      text: document.querySelector('.page').textContent,
      overflow: document.documentElement.scrollWidth - window.innerWidth,
      h1: document.querySelector('.page h1')?.textContent,
    }));
    check(`route "${route}" at ${width}px: has content, no sideways page scroll, every lab file loaded`,
      !!info.h1 && !/being written|could not load/i.test(info.text) && info.overflow <= 1, JSON.stringify({ h1: info.h1, overflow: info.overflow }));
  }
}
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch02/wrap');
await shot('phone', 'table[data-table="exercises"]');

const real = errors.filter((x) => !/Failed to load resource: the server responded with a status of 404/.test(x));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
