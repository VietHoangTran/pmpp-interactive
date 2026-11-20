// Browser check for chapter 8, "Summary and exercises": the carry-forward list, the mixed
// check (one kernel shown from a checked snippet, five questions), this machine's numbers
// for the last question, the map of the book's exercises, the errata lists, what was left
// out, and the project status computed from the reports.
// Usage: APP_URL=http://localhost:<port> node e2e/ch08-wrap.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';
import { cubeAI, stencilIdealAI } from '../src/sim/stencil3d.ts';
import { sharedBytes, slabAI } from '../src/sim/stencilslab.ts';

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
const fmt = (v, digits = 3) =>
  !isFinite(v) ? 'n/a' : v === 0 ? '0' : Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : Number(v.toPrecision(digits)).toString();

await goto(page, 'ch08/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary and exercises", numbered 8.7', h1.includes('Summary and exercises') && (await page.$eval('.page', (e) => e.textContent)).includes('8.7'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', "The book's exercises", 'Where the PDF looks wrong', 'What this chapter left out', 'The project so far'];
check('the page has its six parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
check('the carry-forward list has seven points and cites the book section',
  (await page.$$eval('.prose > ul', (uls) => uls[0].children.length)) === 7 && (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('§8.7, p. 198'));
const text0 = await body();
check('nothing on the page says it is being written', !/being written/i.test(text0));

// ---- the kernel for the double-buffering question is the checked snippet
const file = await page.evaluate(() => fetch('/api/lab-file?path=' + encodeURIComponent('ch08/snippets/regtile_two_planes.cu')).then((r) => (r.ok ? r.text() : 'MISSING')));
const shownCode = await page.$eval('figure.code pre', (e) => e.textContent);
check('the listing is labs/ch08/snippets/regtile_two_planes.cu, line for line', file !== 'MISSING' && shownCode.trim() === file.trim(), shownCode.slice(0, 80));
check('the listing is labelled as not a lab file and front-end-checked, never run',
  (await page.$eval('figure.code .tabs', (e) => e.textContent)).includes('not a lab file') &&
  (await page.$eval('figure.code .tabs .status', (e) => e.textContent)) === 'front-end-checked; never run');
check('in and out are coloured global, inCurr_s shared, the window registers',
  (await page.$$('figure.code .v-global')).length >= 2 && (await page.$$('figure.code .v-shared')).length >= 6 && (await page.$$('figure.code .v-reg')).length >= 6);
await shot('code', 'figure.code');

// ---- the five questions, each answered wrong then right; numbers from the models the page uses
check(`no question is answered at the start (${(await page.$$('.q')).length} questions)`, (await page.$$('.q')).length === 5 && (await page.$$('.q .verdict')).length === 0);

const share2D = Math.round(1000 * (2 * 30 ** 2) / (32 ** 2 + 30 ** 2)) / 10;
const cubePct = Math.round((100 * cubeAI(8)) / stencilIdealAI(7));
const slabPct = Math.round((100 * slabAI(32, 30)) / stencilIdealAI(7));
let q = await question('What share of the 2D stencil');
await typeAnswer(q, 59);
check(`2D tile: the 3D cube's ${cubePct}% is not the answer`, (await qText(q)).includes(`Not quite. The answer is ${share2D} %.`) || (await qText(q)).includes(`The answer is ${share2D}`), await qText(q));
await typeAnswer(q, share2D);
check(`2D tile: ${share2D}% is right, compared with the cube's ${cubePct}% and the slab's ${slabPct}%`,
  (await qText(q)).includes('Correct.') && (await qText(q)).includes(`${cubePct}% of its ideal`) && (await qText(q)).includes(`as coded, ${slabPct}%`));

q = await question('27-point stencil');
await pick(q, 0);
check('27-point: "both moves" is wrong, because each thread reads nine points of each plane', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('nine points of each'));
await pick(q, 2);
check('27-point: "neither" is wrong: marching does not depend on the shape', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('does not depend on the stencil'));
await pick(q, 1);
check('27-point: marching works, the outer planes stay shared; the book\'s axis-only condition is cited', (await qText(q)).includes('Correct.') && (await qText(q)).includes('p. 196'));

const one = sharedBytes('registers', 32), three = sharedBytes('planes', 32);
q = await question('shared plane doubled');
await pick(q, 0);
check('two planes: "a fast thread overwrites the plane being read" is wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('It writes the other buffer'));
await pick(q, 1);
check('two planes: "free, the barrier was never needed" is wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('same inCurr_s'));
await pick(q, 2);
check(`two planes: correct at ${(2 * one).toLocaleString('en-US')} bytes instead of ${one.toLocaleString('en-US')}, below ${three.toLocaleString('en-US')}`,
  (await qText(q)).includes('Correct.') && (await qText(q)).includes(`${(2 * one).toLocaleString('en-US')} bytes per 32 × 32 block instead of ${one.toLocaleString('en-US')}`) &&
  (await qText(q)).includes(`(${three.toLocaleString('en-US')} bytes)`) && (await qText(q)).includes('§6.7'));

q = await question('decode-step attention kernel');
await pick(q, 2);
check('decode: the query in every thread\'s registers is wrong, with the 64-register budget', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('64 registers per thread'));
await pick(q, 1);
check('decode: query shared, running max and sum in registers, keys from global is right', (await qText(q)).includes('Correct.') && (await qText(q)).includes('Ask who reads each value'));
check('decode: the explanation names the uncoalesced key loads and chapter 6\'s two ways out, and makes no claim about every kernel',
  (await qText(q)).includes('uncoalesced') && (await qText(q)).includes('corner turning') && !(await qText(q)).includes('every attention kernel') && !(await qText(q)).includes('gains nothing'));

q = await question('GB/s at 8 B/point');
await pick(q, 2);
check('cache: "3.6 times faster" is wrong, because that needs about 2 bytes per point', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('about 2 bytes'));
await pick(q, 1);
check('cache: most requested bytes never reach DRAM is right', (await qText(q)).includes('Correct.') && (await qText(q)).includes('Register tiling cuts requests'));

await pause(700); // the app saves 400 ms after the last change
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
const ids = ['q-2d-tile', 'q-box27', 'q-two-planes', 'q-decode-placement', 'q-cache-basic'].map((s) => 'ch08/wrap/' + s);
check('all five answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 5);
check('the running count of answers appears after answering', (await body()).includes('So far in this chapter you have answered'));
await shot('questions', '.q');

// ---- this machine's numbers for the last question
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
const machine = Object.keys(reports).filter((b) => b !== 'emu').flatMap((b) => {
  const a = rowsOf(reports[b], 'ch08/lab1_stencil3d').find((r) => r.name === 'basic')?.gbps_compulsory;
  const bw = rowsOf(reports[b], 'ch05/lab1_roofline').find((r) => r.name === 'roofline')?.bandwidth_gb_per_s;
  return a && bw ? [{ b, a, bw }] : [];
});
const shownMachine = await page.$$eval('[data-machine]', (ps) => ps.map((p) => ({ b: p.dataset.machine, ratio: p.dataset.ratio ?? '', text: p.textContent })));
if (machine.length)
  check(`your own numbers: ${machine.map((m) => `${m.b} ${fmt(m.a)} against ${fmt(m.bw)} GB/s`).join('; ')}`,
    machine.every((m) => shownMachine.some((s) => s.b === m.b && s.ratio === fmt(m.a / m.bw) && s.text.includes(fmt(m.a)) && s.text.includes(fmt(m.bw)))), JSON.stringify(shownMachine));
else check('no rows for labs 5.1 and 8.1 on one backend, and the page says what to run', shownMachine.length === 1 && shownMachine[0].b === 'none' && shownMachine[0].text.includes('run 8.1 --bench'));

// ---- the book's exercises
const rows = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) =>
  trs.map((tr) => ({ ex: tr.children[0].textContent.trim(), where: tr.children[2].textContent.trim(), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('the exercise table covers every part of both exercises', rows.map((r) => r.ex).join(' | ') === '1a | 1b | 1c | 1d | 2a, 2b | 2c | 2d, 2e', rows.map((r) => r.ex).join(' | '));
const sectionOf = { '1a': '#/ch08/8.4', '1b': '#/ch08/8.4', '1c': '#/ch08/8.4', '1d': '#/ch08/8.5', '2a, 2b': '#/ch08/8.5', '2c': '#/ch08/8.5', '2d, 2e': '#/ch08/8.5' };
check('each exercise points at the page that works it', rows.every((r) => r.links.includes(sectionOf[r.ex])), JSON.stringify(rows.map((r) => [r.ex, r.links])));
// The questions those rows point at exist on those pages (their ids are in the section's questions).
for (const [route, needles] of [['8.4', ['Book exercise 1a', 'Book exercise 1b', 'Book exercise 1c']], ['8.5', ['Exercise 1d', 'Exercise 2a', 'Exercise 2b', 'Exercise 2c', 'Exercises 2d and 2e']]]) {
  await goto(page, 'ch08/' + route);
  const kinds = await page.$$eval('.q .kind', (k) => k.map((x) => x.textContent));
  check(`page ${route} has the questions the table names (${needles.join(', ')})`, needles.every((n) => kinds.includes(n)), JSON.stringify(kinds));
}
await goto(page, 'ch08/wrap');

// ---- errata and dated statements
const errata = await page.$$eval('ul[data-list="errata"] > li', (lis) => lis.map((li) => li.textContent));
const want = ['c3', 'inNext_s', '1.47', 'Fig. 8.10', 'IN_TILE_DIM', 'Fig. 8.2 is meant', 'one on each side', '2π/6', 'compared 6.25', 'corner cells'];
check(`the errata list has ${want.length} items covering the notes' "Unclear in the PDF"`, errata.length === want.length &&
  want.every((w, i) => errata[i].includes(w)), JSON.stringify(errata.map((e) => e.slice(0, 40))));
const dated = await page.$$eval('ul[data-list="dated"] > li', (lis) => lis.map((li) => li.textContent));
check('the dated list names t = 8, coalescing, precision, registers and the launch failure',
  dated.length === 5 && ['1000 threads', '80%', '64:1', 'no figures', '__launch_bounds__'].every((w, i) => dated[i].includes(w)), JSON.stringify(dated.map((e) => e.slice(0, 40))));
check('the errata are framed as readings of one PDF, checked against vendor documentation', text0.includes('trust your copy') && text0.includes('ch08-facts.md'));
check('what was left out names §8.1 and says where the provided kernels are', text0.includes('pp. 184–187') && text0.includes('lab 8.1 provides both'));
check('what was left out lists the opening\'s differences as the book gives them, and puts the 3D-grid point in §8.7',
  text0.includes('Most of the opening\'s list of differences from convolution (p. 183)') && text0.includes('§8.7 summary\'s (p. 198)') && !text0.includes('except the two that shape the kernels'));

// ---- project status from the reports
const where = (b) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
const passing = [...Object.keys(reports).filter((b) => b !== 'emu'), 'emu'].filter((b) => labStatus(reports[b], 'ch08/lab1_stencil3d', 'work') === 'pass');
const status = await page.$eval('[data-lab81]', (e) => ({ v: e.dataset.lab81, text: e.textContent }));
check(`project status: lab 8.1 work passes on ${passing.join(', ') || 'nothing'} here`, status.v === (passing.join(',') || 'none') &&
  (passing.length ? status.text.includes(passing.map(where).join(' and ')) : status.text.includes('has not passed on this machine yet') || status.text.includes('No lab results')), JSON.stringify(status));
check('the page says this chapter adds no milestone and points at M3 and at chapter 9',
  text0.includes('adds no milestone') && text0.includes('milestone M3') && (await page.$$eval('.prose a', (as) => as.some((a) => a.getAttribute('href') === '#/ch09/start'))));

// ---- every link on the page goes to a section that exists
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
for (const l of links) {
  await goto(page, l.slice(2));
  const ok = await page.evaluate(() => !!document.querySelector('.page h1') && !document.querySelector('.page').textContent.includes('This section is being written'));
  check(`link ${l} opens a written page`, ok);
}

// ---- margin notes sit after paragraphs, never after a bench or code block
await goto(page, 'ch08/wrap');
const misplaced = await page.$$eval('.prose > aside.note', (as) => as.filter((a) => {
  const prev = a.previousElementSibling;
  return prev && !['P', 'UL', 'OL', 'ASIDE'].includes(prev.tagName);
}).map((a) => `${a.previousElementSibling.tagName}.${a.previousElementSibling.className}`));
check('every margin note follows a paragraph, a list or another note', misplaced.length === 0, misplaced.join(', '));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
