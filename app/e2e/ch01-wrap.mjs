// Browser check for chapter 1, "Summary and check": the carry-forward list, the mixed
// check (five questions, numbers compared with the models the sections run on), the table
// of the book's worked numbers, the aged statements, the errata, what was left out, and the
// project paragraph computed from the reports. It ends by visiting every route of the
// chapter at desktop and phone width.
// Usage: node e2e/ch01-wrap.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';
import { BOOK_CASES, ceiling, remaining, speedup, untouchedShare } from '../src/sim/amdahl.ts';
import { chaptersBefore, goal, leadsTo, prerequisites } from '../src/sim/bookmap.ts';
import { design, finish } from '../src/sim/designs.ts';

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
// The start page's narrated animation may have no audio yet; those 404s are expected.
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

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
const n = (v) => v.toLocaleString('en-US', { maximumFractionDigits: 2 });
const p3 = (v) => String(Number(v.toPrecision(3))); // the page's fmt() for values below 1,000
const and = (xs) => (xs.length <= 1 ? xs.join('') : xs.slice(0, -1).join(', ') + ' and ' + xs.at(-1));
const question = (part) => page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(part)), part);
const qText = (q) => q.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
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
const body = () => page.$eval('.prose', (e) => e.textContent.replace(/\s+/g, ' '));

await goto(page, 'ch01/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));

// ---- structure
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', "The book's exercises", 'Where the book has aged', 'Where the PDF looks wrong', 'What this chapter left out', 'The project so far'];
check('the page has its seven parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
check('the page title and lede are the design\'s',
  (await page.$eval('.page h1', (e) => e.textContent)).includes('Summary and check') &&
  (await page.$eval('.page', (e) => e.textContent)).includes("What to carry into chapter 2, a mixed check on the whole chapter"));
const carry = await page.$$eval('.prose > ul', (uls) => [...uls[0].children].map((li) => li.textContent.replace(/\s+/g, ' ')));
check(`the carry-forward list has eight points (${carry.length})`, carry.length === 8);
check('it cites the pages it summarises', (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('pp. xvii–xxvi, 1–18'));
const before15 = chaptersBefore([15]);
const before20 = chaptersBefore([20]);
check(`its reading-path figures are the model's: matmul needs ${and(before15)}, attention ${before20.length} chapters`,
  carry[6].includes(`fast matmul needs chapters ${and(before15)} and nothing else`) && carry[6].includes(`attention needs ${before20.length} chapters`), carry[6]);
check('it gives the law with its ceiling, and the rule of thumb as the book\'s',
  carry[3].includes('1 / ((1 − p) + p / s)') && carry[3].includes('never more than 1 / (1 − p)') && carry[4].includes('rule of thumb'));
check('it says HIP is CUDA\'s model on AMD GPUs and Metal a second code base', carry[7].includes("HIP is CUDA's programming model on AMD GPUs") && carry[7].includes('Metal is a second code base'));
const text0 = await body();
check('nothing on the page says it is being written', !/being written/i.test(text0));

// ---- the five questions, each answered wrong then right; numbers come from the models
check(`five questions, none answered at the start (${(await page.$$('.q')).length})`, (await page.$$('.q')).length === 5 && (await page.$$('.q .verdict')).length === 0);

const left = remaining(0.9, 9);
const share = 100 * untouchedShare(0.9, 9);
let q = await question('What share of the run time does the host part take now?');
await typeAnswer(q, 10);
check(`re-profile: 10 (the old share) is not accepted; the answer is ${p3(share)} %`, (await qText(q)).includes(`Not quite. The answer is ${p3(share)} %.`), await qText(q));
await typeAnswer(q, 90);
check('re-profile: 90 is not accepted either', (await qText(q)).includes('Not quite.'));
await typeAnswer(q, share);
const tq = await qText(q);
check(`re-profile: ${p3(share)} is right, and the explanation walks the time (${p3(left.total * 100)}%, ${p3(speedup(0.9, 9))}×, ceiling ${p3(ceiling(0.9))}×)`,
  tq.includes('Correct.') && tq.includes(`takes ${p3(left.total * 100)}% of what it did (${p3(speedup(0.9, 9))}× overall)`) && tq.includes(`The ceiling for this step was ${p3(ceiling(0.9))}×`) && tq.includes('take another before you choose the next'), tq);

const A = design(64, 2);
const B = design(64, 0);
const last = (d, chains, len) => finish(d, chains, len).last;
const chainA = last(A, 1, 1200), chainB = last(B, 1, 1200);
const crowdA = last(A, 1024, 100), crowdB = last(B, 1024, 100);
q = await question('which arrangement finishes the job first?');
check(`two phases: the prompt states the start page's designs (${A.units} of speed ${A.speed}, ${B.units} of speed ${B.speed})`,
  (await qText(q)).includes(`A with ${A.units} units of speed ${A.speed}, and B with ${B.units} units of speed ${B.speed}`) && (await qText(q)).includes('Then a crowd: 1,024 independent chains'));
await pick(q, 1);
check(`two phases: both on B is wrong, ${n(chainB + crowdB)} in all`, (await qText(q)).includes('Not quite.') && (await qText(q)).includes(`${n(chainB + crowdB)} in all`), await qText(q));
await pick(q, 0);
check(`two phases: both on A is wrong, ${n(chainA + crowdA)} in all`, (await qText(q)).includes(`${n(chainA + crowdA)} in all`));
await pick(q, 3);
check(`two phases: the wrong way round costs ${n(chainB + crowdA)}`, (await qText(q)).includes(`= ${n(chainB + crowdA)}`));
await pick(q, 2);
const tb = await qText(q);
check(`two phases: the chain on A, then the crowd on B is right: ${n(chainA + crowdB)}, the least of the four`,
  tb.includes('Correct.') && tb.includes(`${n(chainA + crowdB)}, against ${n(chainB + crowdB)} for B alone and ${n(chainA + crowdA)} for A alone`) &&
  chainA + crowdB < Math.min(chainA + crowdA, chainB + crowdB, chainB + crowdA), tb);

q = await question('What does a throughput-oriented design expect you to change?');
await pick(q, 0);
check('more threads: "nothing" is wrong, and the reason is that a waiting thread is not executing', (await qText(q)).includes('a waiting thread is not executing'));
await pick(q, 1);
check('more threads: many more threads than units is right, and saturation is kept apart from latency',
  (await qText(q)).includes('Correct.') && (await qText(q)).includes('more threads cure only the first') && (await qText(q)).includes('Chapter 4'));

const behind9 = leadsTo([9]);
const sampling = goal('sampling').chapters;
const blocked = sampling.filter((c) => prerequisites([c]).includes(9));
q = await question('skip chapter 9, histogram');
await pick(q, 3);
check(`skip 9: "none" is wrong, the chart puts ${behind9.length} chapters behind it (${and(behind9)})`,
  (await qText(q)).includes(`The chart puts ${behind9.length} chapters behind chapter 9: ${and(behind9)}.`), await qText(q));
await pick(q, 0);
check(`skip 9: matmul is wrong, it needs ${and(before15)} only`, (await qText(q)).includes(`Chapter 15 needs chapters ${and(before15)} and nothing else.`));
await pick(q, 2);
check(`skip 9: sampling is right, ${blocked.length} of its ${sampling.length} chapters sit behind 9 (${and(blocked)})`,
  (await qText(q)).includes('Correct.') && (await qText(q)).includes(`${blocked.length} of the ${sampling.length} chapters that the sampling goal ends at sit behind it: ${and(blocked)}`) && (await qText(q)).includes('That link is our reading'), await qText(q));

q = await question('which steps lose the reason they were there for?');
await pick(q, 0);
check('one memory: "allocation and launch go" is wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('its threads still have to be started'));
await pick(q, 1);
check('one memory: the two copies are right, Apple\'s wording is quoted and ordering still matters',
  (await qText(q)).includes('Correct.') && (await qText(q)).includes('"allocated in system memory"') && (await qText(q)).includes('must not read a result before the GPU work'));

await pause(700); // the app saves 400 ms after the last change
const ans = lastAnswers();
const ids = ['q-reprofile', 'q-both', 'q-morethreads', 'q-skip9', 'q-sharedmemory'].map((s) => 'ch01/wrap/' + s);
check('all five answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 5);
check('the running count of answers appears after answering', (await body()).includes('So far in this chapter you have answered'));
await shot('questions', '.q');

// ---- the book's worked numbers (there are no exercises)
check('the page says the chapter has no exercises', (await body()).includes('There are none. Chapter 1 has no exercise section'));
const rows = await page.$$eval('table[data-table="worked"] tbody tr', (trs) =>
  trs.map((tr) => ({ what: tr.children[0].textContent.replace(/\s+/g, ' ').trim(), page: tr.children[1].textContent.trim(), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check(`the worked-number table has eight rows (${rows.length})`, rows.length === 8);
check("the three Amdahl rows quote the book's figures as the model records them",
  BOOK_CASES.every((c, i) => rows[3 + i].what.includes(c.book + '×') && rows[3 + i].page === '9'), JSON.stringify(rows.slice(3, 6)));
check("the H100 trio appears only as the book's example", rows[1].what === "34, 67 and 1,979 TFLOPS on the book's example GPU");
check('rows point at the start page and at 1.3', rows[0].links.includes('#/ch01/start') && rows[2].links.includes('#/ch01/1.3') && rows[7].links.includes('#/ch01/1.3'));

// ---- aged statements and errata
const aged = await page.$$eval('table[data-table="aged"] tbody tr', (trs) => trs.map((tr) => tr.textContent.replace(/\s+/g, ' ')));
check(`the table of aged statements has nine rows (${aged.length})`, aged.length === 9);
check('the CUDA date is the book\'s 2007 against NVIDIA\'s 2006, with toolkit 1.0 in June 2007',
  aged[0].includes('2007') && aged[0].includes('2006') && aged[0].includes('June 2007') && aged[0].includes("book's statement only"));
check('OpenCL: 9 December 2008, as the book\'s own reference [9] gives', aged[6].includes('9 December 2008') && aged[6].includes('reference [9]'));
check('exascale is not read as "every machine pairs CPUs with GPUs"', aged[4].includes('some of the five and not for all'));
check('the hyper-threading row quotes the book (Intel and AMD) and answers it with the E-core Xeon and Ampere, without device numbers',
  aged[2].includes('Intel and AMD server processors give each core two hardware threads (hyper-threading) (p. 3)') && aged[2].includes('Intel E-core Xeon we checked, whose threads equal its cores') &&
  aged[2].includes("Ampere, the book's ARM example, calls its cores single-threaded") && !/\d+\s+cores/.test(aged[2]), aged[2]);
check('the Today note names the facts items it rests on', (await page.$$eval('.prose aside.note', (as) => as.map((a) => a.textContent))).some((t) => t.includes('A6, A7') && t.includes('ch05-facts.md, A9')));
const errata = await page.$$eval('ul[data-list="errata"] > li', (lis) => lis.map((li) => li.textContent.replace(/\s+/g, ' ')));
check(`the errata list has nine items (${errata.length})`, errata.length === 9);
check('it covers the four parts, scan as chapter 14, Fig. 1.1(a)/(b), the junction under 14, the titles, the 14 chapters, the OpenCL date and the typos',
  errata[0].includes('"into four major parts" (p. xxii)') && errata[1].includes('"relies on the scan pattern covered in Chapter 14" (p. 16)') &&
  errata[2].includes('"Fig. 1.1(a)"') && errata[3].includes('under the box of chapter 14') && errata[4].includes('Chapter titles differ') &&
  errata[5].includes('"consists of 14 chapters"') && errata[6].includes('OpenCL to 2009 (p. 13)') && errata[6].includes('reference [9]') &&
  errata[7].includes('"coverts"'));
check('the 1,979 figure is marked "not a slip" and sent to the start page', errata[8].startsWith('Not a slip') && errata[8].includes('1,979'));
check('readings of the PDF are marked as ours', (await body()).includes('our readings of one PDF'));
const omitted = await page.$$eval('ul[data-list="omitted"] > li', (lis) => lis.map((li) => li.textContent.replace(/\s+/g, ' ')));
check(`what was left out is listed with pages, ${omitted.length} items`, omitted.length === 8 && omitted.every((o) => /pp?\. /.test(o)), JSON.stringify(omitted.filter((o) => !/pp?\. /.test(o))));
check('the first left-out item separates what the start page covers from what is not taught',
  omitted[0].includes('The end of single-CPU scaling (pp. 1–3): a few sentences on the start page.') && omitted[0].includes('von Neumann model and the program counter') && omitted[0].includes('not taught'), omitted[0]);
check('the carry-forward list calls the arrows prerequisites that this course treats as advice', carry[6].includes('a chart of prerequisites, which this course treats as advice'), carry[6]);

// ---- the project paragraph, computed from the reports
const NAME = { metal: 'Metal', cuda: 'CUDA', hip: 'HIP' };
const devices = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const wantDevices = devices.length
  ? `Lab results recorded on this machine so far come from ${and(devices.map((b) => `${NAME[b] ?? b} (${String(reports[b].device?.name ?? 'unnamed device')})`))}.`
  : 'No lab results from a GPU are recorded on this machine yet. The first will come from chapter 2.';
check(`the devices sentence is read from the reports: "${wantDevices}"`, (await page.$eval('[data-devices]', (e) => e.textContent)) === wantDevices);
const um = reports.metal?.device?.hasUnifiedMemory;
const unifiedShown = await page.$$eval('[data-unified]', (es) => es.map((e) => e.textContent.replace(/\s+/g, ' ')));
const wantUnified = um === undefined ? null : um === true || um === '1' || um === 'true' ? 'as true: one memory for CPU and GPU' : 'as false';
check(`the unified-memory sentence follows the Metal report (${String(um)})`,
  wantUnified === null ? unifiedShown.length === 0 : unifiedShown.length === 1 && unifiedShown[0].includes(wantUnified), JSON.stringify(unifiedShown));
check('the project paragraph names M0 as chapter 2\'s and asks for the doctor command',
  (await body()).includes('The first milestone, M0, is chapter 2\'s') && (await page.$$eval('.prose code', (cs) => cs.map((c) => c.textContent))).includes('python3 labs/pmpp.py doctor'));

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch01/start', '#/ch01/1.3', '#/ch01/1.7', '#/ch23/23.4', '#/ch02/start']);
check(`links go to sections that exist (${links.join(', ')})`, links.length === 5 && links.every((l) => known.has(l)), links.join(', '));
await page.evaluate(() => [...document.querySelectorAll('.prose a')].find((a) => a.getAttribute('href') === '#/ch02/start').click());
await page.waitForFunction(() => location.hash === '#/ch02/start' && document.querySelector('.page[data-ready] h1'), { timeout: 15000 });
check('the link to chapter 2 opens its start page', (await page.$eval('.page h1', (e) => e.textContent)).length > 0);

// ---- the places the tables point at exist on the other pages
await goto(page, 'ch01/start');
check('the start page has the "Spend a power budget" explorer and the 16-bit question',
  !!(await page.$('section.bench[aria-label="Spend a power budget"]')) && (await body()).includes('What explains the step of about 30 times from 32 to 16 bits?'));
await goto(page, 'ch01/1.3');
const h13 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
check('1.3 has the "Speed-up" heading, the Amdahl explorer and "What caps the kernel"',
  h13.includes('Speed-up') && !!(await page.$('section.bench[aria-label="Amdahl\'s law"]')) && !!(await page.$('section.bench[aria-label="What caps the kernel"]')));
await goto(page, 'ch01/1.7');
check('1.7 has the reading-path explorer', !!(await page.$('section.bench[aria-label="Reading path"]')));

// ---- every route of the chapter, at desktop and phone width
const routes = ['start', '1.3', '1.7', 'wrap'].map((s) => 'ch01/' + s);
const rail = [];
for (const width of [1440, 390]) {
  await page.setViewport({ width, height: 900, deviceScaleFactor: 1 });
  for (const route of routes) {
    await goto(page, route);
    const info = await page.evaluate(() => ({
      placeholder: /being written/i.test(document.querySelector('.page').textContent),
      overflow: document.documentElement.scrollWidth - window.innerWidth,
      h1: document.querySelector('.page h1')?.textContent,
    }));
    if (width === 1440 && route === 'ch01/start') rail.push(...(await page.$$eval('.rail nav a', (as) => as.map((a) => a.getAttribute('href')))));
    check(`route ${route} at ${width}px: has content, no sideways page scroll`, !info.placeholder && !!info.h1 && info.overflow <= 1, JSON.stringify(info));
  }
}
check('the rail lists the four sections of chapter 1 in order', JSON.stringify(rail.filter((h) => h.startsWith('#/ch01/'))) === JSON.stringify(routes.map((r) => '#/' + r)), rail.join(', '));
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch01/wrap');
await shot('phone', 'table[data-table="worked"]');

const expected404 = (u) => /\/content\/ch01\/narration\/(manifest\.json|two-designs-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
