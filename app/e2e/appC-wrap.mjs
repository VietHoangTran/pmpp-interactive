// Browser check for appendix C, "Summary": the carry-forward list and the compact map,
// the mixed check (one program shown from a checked snippet, five questions), the map
// of the book's parts to where they are worked, the two lists of errata and
// corrections, what was left out, and the project status computed from the reports.
// Usage: node e2e/appC-wrap.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';
import { ROWS, SITES, VERDICT_LABEL, grid } from '../src/sim/derefmap.ts';
import { INDEX } from '../src/sim/addrwidth.ts';
import { breakEven, BOOK_RULE } from '../src/sim/zerocopy.ts';

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

await goto(page, 'appC/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', "The book's exercises", 'Where the PDF looks wrong', 'What this appendix left out', 'The project so far'];
check('the page has its six parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
check('the page title is "Summary"', (await page.$eval('.page h1', (e) => e.textContent)).includes('Summary'));
check('the carry-forward list has eight points and cites the appendix\'s pages',
  (await page.$$eval('.prose > ul', (uls) => uls[0].children.length)) === 8 && (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('pp. 623–629'));
const text0 = await body();
check('it keeps the two unified memories apart and says nothing orders anything',
  text0.includes('Unified Memory, with a capital U') && text0.includes("Apple's unified memory is one physical memory") && text0.includes('None of this orders anything'));
check('nothing on the page says it is being written, and there is no em dash', !/being written|—/i.test(text0));

// ---- the compact map: opened beyond the book, every cell from the model, on all three platforms
const MAP = 'section.bench[aria-label="Who can dereference what, era by era"]';
const cells = () =>
  page.$$eval(MAP + ' td[data-row]', (tds) => tds.map((t) => ({ row: t.dataset.row, site: t.dataset.site, v: t.dataset.verdict, chip: t.querySelector('[data-chip]').textContent })));
const sameAsModel = (shown, g) => shown.length === ROWS.length * SITES.length && shown.every((c) => g[c.row][c.site].verdict === c.v && VERDICT_LABEL[c.v] === c.chip);
check('the compact map opens at the step beyond the book', (await page.$eval(MAP + ' button[data-era][aria-pressed="true"]', (b) => b.dataset.era)) === 'hmm');
check('  compact: the era\'s description and the "what changed" list are hidden', (await page.$(MAP + ' [data-changes]')) === null && !(await page.$eval(MAP, (e) => e.textContent)).includes('Plain malloc memory works in kernels through the host'));
check('  every CUDA cell matches the model', sameAsModel(await cells(), grid('cuda', 'hmm')), JSON.stringify((await cells()).slice(0, 3)));
await page.evaluate((m) => document.querySelector(m + ' button[data-platform="hip"]').click(), MAP);
await pause(80);
check('  HIP view matches the model (XNACK off)', sameAsModel(await cells(), grid('hip', 'hmm', { xnack: false })));
await page.evaluate((m) => document.querySelector(m + ' button[data-platform="metal"]').click(), MAP);
await pause(80);
check('  Metal view matches the model', sameAsModel(await cells(), grid('metal', 'hmm')));
await page.evaluate((m) => document.querySelector(m + ' button[data-platform="cuda"]').click(), MAP);
await pause(80);
await page.evaluate((m) => document.querySelector(m + ' td[data-row="malloc"][data-site="host"] button')?.click(), MAP);
await pause(80);
check('  clicking a cell shows its rule and its source', (await page.$eval(MAP + ' [data-source]', (e) => e.textContent)).startsWith('Source:'));
await shot('map', MAP);

// ---- the program for the first question is the checked snippet
const file = await page.evaluate(() => fetch('/api/lab-file?path=' + encodeURIComponent('appC/snippets/wrap_layer_table.cu')).then((r) => (r.ok ? r.text() : 'MISSING')));
const shownCode = await page.$eval('figure.code pre', (e) => e.textContent);
check('the listing is labs/appC/snippets/wrap_layer_table.cu, line for line', file !== 'MISSING' && shownCode.trim() === file.trim(), shownCode.slice(0, 80));
check('the listing is labelled as not a lab file, front-end-checked, and defective on purpose',
  (await page.$eval('figure.code .tabs', (e) => e.textContent)).includes('not a lab file') &&
  (await page.$eval('figure.code .tabs .status', (e) => e.textContent)) === 'front-end-checked; has a defect on purpose');
check('the table and the output are coloured as global memory', (await page.$$('figure.code .v-global')).length >= 4);
await shot('code', 'figure.code');

// ---- the five questions; numbers come from the models the page uses
const perLayer = 32768 * 32 * 128;
const firstBad = Math.ceil(2 ** INDEX.int32.log2Count / perLayer);
const embed = Math.round(breakEven(1 / 128000, BOOK_RULE));
check(`no question is answered at the start (${(await page.$$('.q')).length} questions)`, (await page.$$('.q')).length === 5 && (await page.$$('.q .verdict')).length === 0);

let q = await question('What happens at lines A and B?');
await pick(q, 0);
check('layer table: "both print 0" is wrong, and the reason is the host dereference of cudaMalloc memory',
  (await qText(q)).includes('Not quite.') && (await qText(q)).includes('That pointer came from cudaMalloc'));
await pick(q, 1);
check('layer table: line B is the defect', (await qText(q)).includes('Correct.'));
check('  the explanation names HIP\'s different rule and the emulator\'s blind spot', (await qText(q)).includes('hipMalloc') && (await qText(q)).includes('emulator'));

q = await question('for which the pointer comes out wrong?');
await typeAnswer(q, 32);
check(`KV offset: 32 is not the answer, ${firstBad} is`, (await qText(q)).includes(`Not quite. The answer is ${firstBad}.`));
await typeAnswer(q, firstBad);
check(`KV offset: layer ${firstBad} is right, and the fix casts before multiplying`, (await qText(q)).includes('Correct.') && (await qText(q)).includes('(size_t)layer * tokens * heads * dim'));

q = await question('to read the logits after');
await pick(q, 0);
check('private logits: reading contents is wrong, because contents is NULL', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('contents is NULL'));
await pick(q, 1);
check('private logits: a blit into a shared buffer is right', (await qText(q)).includes('Correct.') && (await qText(q)).includes('feedPrivateCopy'));

q = await question('What does NVIDIA\'s documentation say?');
await pick(q, 2);
check('mapped atomics: pinning does not make it safe', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('Mapped memory is pinned already'));
await pick(q, 1);
check('mapped atomics: updates can be lost', (await qText(q)).includes('Correct.'));

q = await question('after how many generated tokens');
await typeAnswer(q, 128000);
check(`embedding: 128,000 is outside the tolerance of ${embed}`, (await qText(q)).includes(`Not quite. The answer is ${embed} tokens.`));
await typeAnswer(q, embed);
check(`embedding: ${embed} tokens is right, from the model's break-even`, (await qText(q)).includes('Correct.') && (await qText(q)).includes(embed.toLocaleString('en-US')));

await pause(700); // the app saves 400 ms after the last change
const ans = lastAnswers();
const ids = ['q-layer-table', 'q-kv-offset', 'q-private-logits', 'q-mapped-atomics', 'q-embed-mapped'].map((s) => 'appC/wrap/' + s);
check('all five answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 5);
check('the running count of answers appears after answering', (await body()).includes('So far in this appendix you have answered'));
await shot('questions', '.q');

// ---- the book's parts and where they are worked
const rows = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) =>
  trs.map((tr) => ({ part: tr.children[0].textContent.trim(), where: tr.children[2].textContent.trim(), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('the page says the appendix has no exercises', (await body()).includes('The appendix has none'));
check('the table covers the opening and C.1 to C.9', rows.length === 10 && rows.slice(1).every((r, i) => r.part.startsWith('C.' + (i + 1))), rows.map((r) => r.part).join(' | '));
check('every row links to the section that works it', rows.every((r, i) => r.links.includes(i === 0 ? '#/appC/start' : '#/appC/C.' + i)), JSON.stringify(rows.map((r) => r.links)));

// ---- errata and corrections
const errata = await page.$$eval('ul[data-list="errata"] > li', (lis) => lis.map((li) => li.textContent));
check(`the errata list has ${errata.length} items, Fig. C.1 first`, errata.length === 6 && errata[0].includes('Fig. C.1') && errata[0].includes('no semicolon'), String(errata.length));
check('it includes the dangling flag reference, the UVAS/UVA names and the reference\'s author order',
  errata.some((e) => e.includes('We found no earlier mention')) && errata.some((e) => e.includes('UVAS')) && errata.some((e) => e.includes('Gelado, Cabezas, Navarro')));
const corr = await page.$$eval('ul[data-list="corrections"] > li', (lis) => lis.map((li) => li.textContent));
check(`the corrections list has ${corr.length} items, each naming its facts item`, corr.length === 10 && corr.every((c) => /\(A\d+\)/.test(c)), corr.map((c) => c.slice(0, 30)).join(' | '));
check('it gives the corrected dates: March 2010, March 2012, April 2014, CUDA 10.2',
  ['March 2010', 'March 2012', 'April 2014', 'CUDA 10.2'].every((s) => corr.some((c) => c.includes(s))));
check('readings of the PDF are marked as ours', (await body()).includes('our readings of one PDF'));
const leftOut = await page.$$eval('.prose > ul', (uls) => uls.at(-1).textContent);
check('what was left out is listed with where to find it', ['p. 517', 'chapter 23', 'p. 629', 'reference [1]'].every((c) => leftOut.includes(c)), leftOut.slice(0, 120));

// ---- project status, computed from the reports
const LAB = 'appC/lab1_feeding';
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = (run?.results ?? []).filter((x) => x.lab === id && x.variant === variant && x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const devices = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const passing = devices.filter((b) => labStatus(reports[b], LAB, 'work') === 'pass');
const want = devices.length === 0 && !reports.emu
  ? 'No lab results are recorded on this machine yet.'
  : passing.length
    ? `Your own feed functions pass on ${passing.join(' and ')} on this machine.`
    : labStatus(reports.emu, LAB, 'work') === 'pass'
      ? 'Your own feed functions pass on the emulator; they have not passed on a device on this machine yet.'
      : 'Your own feed functions have not passed on this machine yet.';
const status = await page.$eval('[data-c1]', (e) => e.textContent);
check(`lab C.1 status is read from the reports: "${want}"`, status === want, status);
check('the project paragraph says no milestone, names what comes next (X1) and where the mechanisms reappear (X3)', (await body()).includes('The appendix adds no milestone') && (await body()).includes('chapter X1, sampling kernels, comes next') && (await body()).includes('Chapter X3'));

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/appC/start', ...['C.1', 'C.2', 'C.3', 'C.4', 'C.5', 'C.6', 'C.7', 'C.8', 'C.9'].map((s) => '#/appC/' + s), '#/appC/labs', '#/appC/wrap', '#/x1', '#/x3']);
check(`links go to pages that exist (${links.length} distinct)`, links.length >= 10 && links.every((l) => known.has(l)), links.join(', '));
await page.evaluate(() => [...document.querySelectorAll('.prose a')].find((a) => a.getAttribute('href') === '#/appC/labs').click());
await page.waitForFunction(() => document.querySelector('.page h1')?.textContent.includes('Labs'), { timeout: 10000 });
check('the link to the labs page opens it', (await page.$eval('.page h1', (e) => e.textContent)).includes('Labs'));

// ---- margin notes follow text
await goto(page, 'appC/wrap');
const after = await page.$$eval('.prose > aside.note', (as) => as.map((a) => a.previousElementSibling?.tagName ?? ''));
check('every margin note directly follows a paragraph or a list', after.every((t) => ['P', 'UL', 'OL'].includes(t)), after.join(','));

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'appC/wrap');
await pause(300);
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
const mapFit = await page.$eval('table[data-derefmap]', (t) => {
  const box = t.getBoundingClientRect();
  const last = [...t.querySelectorAll('thead th')].at(-1).getBoundingClientRect();
  return { sw: t.scrollWidth, cw: t.clientWidth, lastRight: Math.round(last.right), boxRight: Math.round(box.right) };
});
check('at 390 px the compact map shows its last column without scrolling', mapFit.sw <= mapFit.cw + 1 && mapFit.lastRight <= mapFit.boxRight + 1, JSON.stringify(mapFit));
await shot('phone', 'table[data-table="exercises"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
