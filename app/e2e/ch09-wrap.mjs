// Browser check for chapter 9, "Summary and exercises": the carry-forward list, the mixed
// check (one kernel shown from a checked snippet, five questions, each answered wrong and
// then right), the map of the book's exercises, the errata list, what was shortened, and the
// project status from the reports. The progress file is replaced by an empty one and never
// written.
// Usage: APP_URL=http://localhost:<port> node e2e/ch09-wrap.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
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

// ---- the numbers, from the same model the page uses (Wrap.tsx; the unit test runs the kernel)
const N = 1000, BLOCK = 256;
const image = Array.from({ length: N }, (_, i) => (37 * i + 11) % 256);
const live = N % BLOCK;
const lost = [...new Set(image.slice(N - live).filter((v) => v % BLOCK >= live))].sort((a, b) => a - b);
const appendMs = (100000 / (1e9 / 400)) * 1e3;

await goto(page, 'ch09/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary and exercises", numbered 9.7', h1.includes('Summary and exercises') && (await page.$eval('.page', (e) => e.textContent)).includes('9.7'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((x) => x.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', "The book's exercises", 'Where the PDF looks wrong', 'What this course shortened', 'Where you will meet it in real code', 'The project so far'];
check('the page has its seven parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
check('the carry-forward list has six points and cites §9.7',
  (await page.$$eval('.prose > ul', (uls) => uls[0].children.length)) === 6 && (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('§9.7, pp. 219–220'));
const text0 = await body();
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('no em dash used as an aside in the prose', !/ — /.test(text0));
const misplaced = await page.$$eval('.prose > .note', (ns) => ns.filter((n) => n.previousElementSibling?.tagName !== 'P').map((n) => n.textContent.slice(0, 60)));
check('every margin note follows a paragraph', misplaced.length === 0, misplaced.join(' | '));

// ---- the kernel for the first question is the checked snippet
const file = await page.evaluate(() => fetch('/api/lab-file?path=' + encodeURIComponent('ch09/snippets/wrap_early_exit.cu')).then((r) => (r.ok ? r.text() : 'MISSING')));
await page.waitForFunction(() => !document.querySelector('figure.code pre')?.textContent.startsWith('// loading'), { timeout: 10000 });
const shownCode = await page.$eval('figure.code pre', (e) => e.textContent);
const fnStart = file.indexOf('__global__ void histogramEarlyExit');
check('the listing is histogramEarlyExit from labs/ch09/snippets/wrap_early_exit.cu, line for line',
  file !== 'MISSING' && fnStart > 0 && shownCode.trim() === file.slice(fnStart).trim(), shownCode.slice(0, 80));
check('the listing is labelled as not a lab file, emulator-checked and wrong on purpose',
  (await page.$eval('figure.code .tabs', (e) => e.textContent)).includes('not a lab file') &&
  (await page.$eval('figure.code .tabs .status', (e) => e.textContent)) === 'emulator-checked; wrong on purpose');
check('bins_s is coloured as shared memory, image and bins as global',
  (await page.$$('figure.code .v-shared')).length >= 3 && (await page.$$('figure.code .v-global')).length >= 2);
check('the kernels and settings are labelled as ours', text0.includes('The kernel, the router, the sampler and the narrow counters are ours'));
await shot('code', 'figure.code');

// ---- the five questions, each answered wrong then right
check(`five questions, none answered at the start`, (await page.$$('.q')).length === 5 && (await page.$$('.q .verdict')).length === 0);

let q = await question('Which counts come out wrong?');
await pick(q, 0);
check('early exit: "none" is wrong, the barrier is fine but the work is not', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('jobs besides their pixel'));
await pick(q, 3);
check('early exit: "all at random" is wrong, NVIDIA\'s barrier waits for non-exited threads', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('have not exited'));
await pick(q, 1);
check(`early exit: bins ${lost[0]} to ${lost.at(-1)} is right, ${live} threads stay in the last block`,
  (await qText(q)).includes('Correct.') && (await qText(q)).includes(`bins ${lost[0]} to ${lost.at(-1)}`) && (await qText(q)).includes(`= ${live} pixels`) && (await qText(q)).includes(`${lost.length} bins one short`));

q = await question('weights=w');
await pick(q, 0);
check('float weights: "float atomics lose updates" is wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('every update arrives'));
await pick(q, 1);
check('float weights: order plus rounding is right, citing PyTorch\'s reason', (await qText(q)).includes('Correct.') && (await qText(q)).includes('because of floating point atomicAdd usage'));

q = await question('mixture-of-experts');
await pick(q, 3);
check('experts: a plain increment is wrong, a lost update undersizes a buffer', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('written past its end'));
await pick(q, 0);
check('experts: one global atomic per choice is wrong on uneven routing', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('32,768 atomics'));
await pick(q, 1);
check('experts: a private copy per block is right, at most 8,192 commits', (await qText(q)).includes('Correct.') && (await qText(q)).includes('128 × 64 = 8,192'));

q = await question('unsigned char bins_s[256]');
await pick(q, 0);
check('narrow counts: "nothing" is wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('range of a count'));
await pick(q, 1);
check('narrow counts: wrap past 255 and emulated 8-bit atomics is right', (await qText(q)).includes('Correct.') && (await qText(q)).includes('drastically slower') && (await qText(q)).includes('1,024'));

q = await question('surviving candidates');
await typeAnswer(q, 4);
check(`append: 4 ms is wrong, the answer is ${appendMs} ms`, (await qText(q)).includes(`The answer is ${appendMs} ms`));
await typeAnswer(q, appendMs);
check('append: 40 ms is right, and the return value\'s use is tied to p. 209', (await qText(q)).includes('Correct.') && (await qText(q)).includes('p. 209'));

await pause(700); // the app saves 400 ms after the last change
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
const ids = ['q-early-exit', 'q-float-weights', 'q-moe', 'q-narrow-counts', 'q-append'].map((s) => 'ch09/wrap/' + s);
check('all five answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 5);
check('the running count of answers appears after answering', (await body()).includes('So far in this chapter you have answered'));
await shot('questions', '.q');

// ---- the book's exercises
const rows = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) =>
  trs.map((tr) => ({ ex: tr.children[0].textContent.trim(), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('the exercise table covers 1 to 4 and 5a to 5c', rows.map((r) => r.ex).join(' | ') === '1 | 2 | 3 | 4 | 5a | 5b | 5c', rows.map((r) => r.ex).join(' | '));
const sectionOf = { 1: '#/ch09/9.3', 2: '#/ch09/9.3', 3: '#/ch09/9.3', 4: '#/ch09/9.3', '5a': '#/ch09/9.4', '5b': '#/ch09/9.4', '5c': '#/ch09/9.5' };
check('each exercise points at the section that works it', rows.every((r) => r.links.includes(sectionOf[r.ex])), JSON.stringify(rows));

// ---- real code: the pointers moved here from the sections
const real = await page.$$eval('ul[data-list="real-code"] > li', (lis) => lis.map((li) => li.textContent.replace(/\s+/g, ' ')));
check('real code: CUB, llama.cpp top-k and count_equal, each described', real.length === 3 && real[0].includes('DeviceHistogram') && real[0].includes('bins_pool') && real[1].includes('256-bin histogram') && real[2].includes('count_equal') && real[2].includes('atomicAdd per warp'), JSON.stringify(real));
check('real code: labelled not in the book, with its facts items', await page.$$eval('.note.beyond', (n) => n.some((x) => x.textContent.includes('None of these is in the book') && x.textContent.includes('D3, D4'))));

// ---- errata
const errata = await page.$$eval('ul[data-list="errata"] > li', (lis) => lis.map((li) => li.textContent));
check(`the errata list has ${errata.length} items`, errata.length === 9, String(errata.length));
for (const [name, frag] of [
  ['text-histogram leftovers, pp. 211 and 216', 'In our text histogram example'],
  ['Fig. 9.11 for Fig. 9.12, p. 215', 'describes the code of Fig. 9.12'],
  ['"line 14", p. 213', '"line 14"'],
  ['the defaults the declaration omits', 'system scope and seq_cst'],
  ['both namespace spellings', 'both spellings are valid'],
  ['nobody zeroes the bins', 'No listing zeroes'],
  ['exercise 4', 'Exercise 4 gives'],
  ['private to each SM', 'private to each SM'],
  ['b and bNext', 'promotes bNext'],
]) check(`errata include ${name}`, errata.some((e) => e.includes(frag)));
check('readings of the PDF are marked as ours, with the facts file', text0.includes('our readings of one PDF') && text0.includes('notes/today/ch09-facts.md'));
check('dated statements are named as the book\'s example', text0.includes('no vendor publishes latencies in cycles today') && text0.includes('1,024 threads per block still holds'));
check('the page says what was shortened, with pages', ['(pp. 204–205)', '(pp. 202–203)', '(p. 206)', '(p. 211)', '(p. 213)', '(p. 216)', '(p. 217)', 'sign-aware repetition penalty'].every((s) => text0.includes(s)));

// ---- project status, computed from the reports
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = (run?.results ?? []).filter((x) => x.lab === id && x.variant === variant && x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
const m5 = backends.filter((b) => labStatus(reports[b], 'ch09/lab3_tokencounts', 'work') === 'pass');
const shown = await page.$eval('[data-m5]', (e) => ({ v: e.dataset.m5, text: e.textContent }));
check(`milestone M5 status is read from the reports (${m5.join(',') || 'none'})`, shown.v === (m5.join(',') || 'none') &&
  (backends.length === 0 ? shown.text.includes('No lab results') : m5.length ? shown.text.includes('passes on') : shown.text.includes('does not pass')), JSON.stringify(shown));
check('the project paragraph names M5, lab 9.3 and module X1', text0.includes('Lab 9.3 is the counting piece of milestone M5') && text0.includes('module X1 assembles M5'));
check('the next chapter is named and linked', text0.includes('Chapter 10 keeps this chapter') && (await page.$$eval('.prose a[href="#/ch10"]', (as) => as.length)) === 1);

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch09/9.1', '#/ch09/9.2', '#/ch09/9.3', '#/ch09/9.4', '#/ch09/9.5', '#/ch09/9.6', '#/ch09/labs', '#/ch05/wrap', '#/ch10', '#/ch12/12.4', '#/x1/X1.3', '#/x1/labs']);
check(`links go to routes that exist (${links.join(', ')})`, links.length >= 7 && links.every((l) => known.has(l)), links.join(', '));
await page.evaluate(() => [...document.querySelectorAll('.prose a')].find((a) => a.getAttribute('href') === '#/ch10').click());
await page.waitForFunction(() => document.querySelector('.page h1') && !document.querySelector('.page h1').textContent.includes('Summary and exercises'), { timeout: 10000 });
const next = await page.$eval('.page', (e) => e.textContent.slice(0, 300));
check('the link to chapter 10 opens it', /Reduction|Before you start/.test(next), next.slice(0, 120));

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch09/wrap');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'table[data-table="exercises"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
