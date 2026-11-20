// Browser check for chapter 24, "Summary and where next": the three habits, the mixed
// check (four questions, numbers compared with the model the start page runs on), the
// table of the book's numbers, the errata, what was left out, the project paragraph
// computed from the reports, and the hand-off table whose links degrade to plain text
// for chapters not built yet. It ends by visiting both routes of the chapter at desktop
// and phone width, in both themes.
// Usage: node e2e/ch24-wrap.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { BOOK, DEVICES, linearLayerRatio, pcieRate, threshold, tileFor, VENDOR } from '../src/sim/hwtable.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// A known progress file (two sections done, nothing answered); nothing reaches the real one.
const START = {
  version: 1,
  sections: { 'ch05/5.1': { done: true, visited: '2026-11-20T00:00:00.000Z' }, 'ch24/start': { done: true, visited: '2026-11-20T00:00:00.000Z' } },
  answers: {},
  review: {},
};
const puts = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
  if (r.method() === 'PUT') puts.push(JSON.parse(r.postData()));
  return r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : JSON.stringify(START) });
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const squash = (s) => s.replace(/\s+/g, ' ');
const p3 = (v) => (Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : String(Number(v.toPrecision(3)))); // the page's fmt()
const question = (part) => page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(part)), part);
const qText = async (q) => squash(await q.evaluate((e) => e.textContent));
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
const body = async () => squash(await page.$eval('.prose', (e) => e.textContent));

await goto(page, 'ch24/wrap');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));

// ---- structure
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', "The book's exercises", 'Where the PDF looks wrong', 'What this chapter left out', 'The project so far', 'Where next'];
check('the page has its seven parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
check("the page title and lede are the design's",
  (await page.$eval('.page h1', (e) => e.textContent)).includes('Summary and where next') &&
    (await page.$eval('.page', (e) => e.textContent)).includes("Carry three habits out of the book's last chapter"));
const carry = await page.$$eval('.prose ul[data-carry] > li', (ls) => ls.map((l) => l.textContent.replace(/\s+/g, ' ')));
check(`three habits to carry forward (${carry.length})`, carry.length === 3 && carry[0].startsWith('Recompute before you quote') && carry[1].startsWith('Ask what a figure counts') && carry[2].startsWith('Read the rows as limits'));
const thr = (k) => { const d = DEVICES.find((x) => x.key === k); return threshold(d.peak, d.bw); };
check(`the thresholds in the third habit are the model's: about ${Number(thr('book2007').toPrecision(2))}, ${p3(thr('book2025'))} and ${p3(thr('book2025tc'))}`,
  carry[2].includes(`from about ${Number(thr('book2007').toPrecision(2))} FLOP/B to ${p3(thr('book2025'))} in FP32 and to ${p3(thr('book2025tc'))}`) && carry[2].includes('our reading'), carry[2]);
check('nothing on the page says it is being written', !/being written/i.test(await body()));
check('the first habit counts one arithmetic slip and one unit slip', carry[0].includes('fails once on arithmetic and once on units'), carry[0]);
check('the mixed check says it asks new questions, and the appendices are promised, not claimed', (await body()).includes('on new numbers and new questions about the table') && (await body()).includes('Appendices A and C get their own pages in this course'));

// ---- the mixed check: each question wrong first, then right
check(`four questions, none answered at the start (${(await page.$$('.q')).length})`, (await page.$$('.q')).length === 4 && (await page.$$('.q .verdict')).length === 0);

const row = (id) => BOOK.rows.find((r) => r.id === id);
const gap = row('membw').to / row('pcie').to;
let q = await question('streaming its weights from host memory');
await typeAnswer(q, 93);
check('q-gap: 93 (the growth of bandwidth) is not accepted', (await qText(q)).includes(`Not quite. The answer is ${gap} times.`), await qText(q));
await typeAnswer(q, gap);
let t = await qText(q);
const vendor2007 = VENDOR.g80.membw.value / pcieRate(VENDOR.g80.pcieGen, 16);
const vendor2025 = VENDOR.b200.membw.value / pcieRate(VENDOR.b200.pcieGen, 16);
check(`q-gap: ${gap} is right; 2007 was ${p3(row('membw').from / row('pcie').from)}, NVIDIA's boards ${p3(vendor2007)} to ${p3(vendor2025)}`,
  t.includes('Correct.') && t.includes(`8,000 / 64 = ${gap}`) && t.includes(`86.4 / 8 = ${p3(row('membw').from / row('pcie').from)}`) && t.includes(`from ${p3(vendor2007)} to ${p3(vendor2025)}`), t);

q = await question('A colleague writes');
await pick(q, 0);
check('q-perboard: "the units check out" is wrong and says why', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('Units say nothing about whether both numbers describe the same hardware'));
await pick(q, 1);
t = await qText(q);
check('q-perboard: the board figure is for eight GPUs; per GPU 75,000 / 8,000 = 9.38 FLOP/B', t.includes('Correct.') && t.includes('75,000 GFLOP/s / 8,000 GB/s = 9.38 FLOP/B') && t.includes('32 FLOP/B kernel is compute-bound'), t);

q = await question('Instinct MI355X');
const amd = threshold(157300, 8000);
await typeAnswer(q, 312.5);
check('q-mi355x: the FP16 matrix threshold (312.5) is not accepted', (await qText(q)).includes('Not quite. The answer is 19.66 FLOP/B.'), await qText(q));
await typeAnswer(q, 19.7);
t = await qText(q);
check(`q-mi355x: 19.7 is right; T = ${tileFor(amd)}; labelled as AMD's peaks, not your device`, t.includes('Correct.') && t.includes(`= ${p3(amd)} FLOP/B`) && t.includes(`T = ${tileFor(amd)}`) && t.includes('not your device'), t);

q = await question('Which next step in this course raises the speed that ceiling allows it');
await pick(q, 1);
check('q-next: TF32 is wrong, and not in matmul2d', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('matmul2d'));
await pick(q, 2);
t = await qText(q);
const l = (b) => p3(linearLayerRatio(1, 1024, 4096, b));
check(`q-next: module X2; the ratio goes ${Number(linearLayerRatio(1, 1024, 4096, 2).toPrecision(2))}, ${Number(linearLayerRatio(1, 1024, 4096, 1).toPrecision(2))}, ${Number(linearLayerRatio(1, 1024, 4096, 0.5).toPrecision(2))}`,
  t.includes('Correct.') && t.includes(`at 8 bits it is ${Number(linearLayerRatio(1, 1024, 4096, 1).toPrecision(2))}, at 4 bits ${Number(linearLayerRatio(1, 1024, 4096, 0.5).toPrecision(2))}`) && t.includes('M9'), t + ' ' + l(2));

// ---- the book's numbers
const worked = await page.$$eval('table[data-table="worked"] tbody tr', (rs) => rs.map((r) => [...r.cells].map((c) => c.textContent.trim())));
check('the exercises part says there are none and maps the six numbers of the chapter', (await body()).includes('There are none: chapter 24 has no code and no exercise section') && worked.length === 6, String(worked.length));
check('every number in that table is printed on pp. 577–578', worked.every((r) => ['577–578', '578'].includes(r[1])), JSON.stringify(worked.map((r) => r[1])));
check('the table points at the start page', !!(await page.$('.prose a[href="#/ch24/start"]')));

// ---- errata
const errata = await page.$$eval('.prose ul[data-errata] > li', (ls) => ls.map((l) => l.textContent.replace(/\s+/g, ' ')));
const all = errata.join(' | ');
check(`the errata list (${errata.length} items) covers the notes and the facts file`,
  errata.length === 10 && ['638x', '6,377', '6,366', '1,800 TB/s', 'Tensor Core FP32', 'TF32', '768 MB', 'generation 1', '16x', '75 TFLOPS', '180 GB', 'November 2006', 'a 18-year comparison', 'massively computing devices', 'a definitely no', 'chapters 22 and 23'].every((s) => all.includes(s)), all);
const tc = (await body()).split('Tensor Core FP32').length - 1;
check('"Tensor Core FP32" appears only inside the errata list', tc === 1 && all.includes('Tensor Core FP32'), String(tc));
check('the open 86.4 GB/s source and PCIe generation are worded as such', all.includes("NVIDIA's pages we read do not print 86.4 GB/s") && all.includes('a 2007 forum post quoting') && all.includes("No NVIDIA page we read gives the 8800 GTX's generation") && (await page.$$eval('.prose aside.note', (as) => as.map((a) => a.textContent))).some((x) => x.includes('Two points stay open')));
// One count across both pages: two contradictions, then two corrections against NVIDIA's documents.
const intro = squash(await page.$eval('.prose ul[data-errata]', (u) => u.previousElementSibling.previousElementSibling.textContent));
check('the errata intro counts as the start page does: two contradictions, then one label and the host-link generation', intro.includes("Two cells contradict the table's own arithmetic or units") && intro.includes("One label and the 2007 host-link generation are wrong against NVIDIA's documents") && intro.includes('the last two are wording and a loose grouping'), intro);
check('the intro says where the PCIe rates come from', intro.includes("PCIe rates from the Linux kernel's table"), intro);
check('the order matches the intro: 638x, the unit, the label, Gen2', errata[0].includes('638x') && errata[1].includes('1,800 TB/s') && errata[2].includes('Tensor Core FP32') && errata[3].includes('Gen2 x16'), errata.slice(0, 4).join(' | '));
check('the B200 dates follow the facts: announced March 2024, first quarter of volume sales ended January 2025', all.includes('first quarter of Blackwell volume sales ended in January 2025') && !all.includes('not launch years') && !all.includes('early 2025'), all);
check('80 TFLOPS and 192 GB are said to match NVIDIA figures, not to be them', all.includes("80 TFLOPS matches NVIDIA's GB200 NVL72 rack figure divided by 72") && all.includes('192 GB matches'), all);

// ---- what was left out
const left = await page.$$eval('.prose ul[data-left] > li', (ls) => ls.map((l) => l.textContent.replace(/\s+/g, ' ')));
check(`each left-out item names its page (${left.length})`, left.length === 6 && left.every((x) => /\(pp?\. \d|p\. 58|Page 581/.test(x)), JSON.stringify(left));
check('the scale-out claim is attributed and its unverified parts said', left[1].includes('The authors write') && left[1].includes('could not verify'));

// ---- the project paragraph, computed from the reports and the progress file
const proj = squash(await page.$eval('.prose [data-project]', (e) => e.textContent));
const gpu = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const labsRun = new Set(Object.values(reports).flatMap((r) => r.results.map((x) => x.lab))).size;
check('the project paragraph counts the two sections marked done', proj.includes('you have marked 2 of '), proj);
check(gpu.length ? `it counts the ${labsRun} labs run, from the reports` : 'with no GPU report it says so',
  gpu.length ? proj.includes(`Of the ${labsRun} labs run on this machine`) : proj.includes('No lab results from a GPU'), proj);
check('the emulators are not named as GPUs', !/ on (emu|race)\b/.test(proj), proj);

// ---- where next: links only where a chapter has pages
const next = await page.$$eval('table[data-table="next"] tbody tr', (rs) => rs.map((r) => ({ id: r.dataset.next, a: r.querySelector('a')?.getAttribute('href') ?? null, goals: r.querySelector('[data-goals]')?.textContent.replace(/^for: /, '') ?? '' })));
check('the hand-off table lists appendices A and C and modules X1 to X3', next.map((n) => n.id).join() === 'appA,appC,x1,x2,x3', next.map((n) => n.id).join());
check('every row names the goal it serves', next.every((n) => n.goals.trim().length > 0), JSON.stringify(next));
for (const n of next) {
  await goto(page, `${n.id}/start`);
  const planned = (await page.$eval('.page', (e) => e.textContent)).includes('planned and not built yet');
  check(`${n.id}: ${planned ? 'planned, so plain text' : 'built, so a link'}`, planned ? n.a === null : !!n.a?.startsWith(`#/${n.id}/`), JSON.stringify(n));
}
await goto(page, 'ch24/wrap');
const ch23 = await page.$$eval('.prose ul[data-left] a', (as) => as.map((a) => a.getAttribute('href')));
await goto(page, 'ch23/start');
const ch23planned = (await page.$eval('.page', (e) => e.textContent)).includes('planned and not built yet');
check('the chapter 23 mention is plain text while that chapter is planned', ch23planned ? !ch23.some((h) => h?.startsWith('#/ch23')) : ch23.some((h) => h?.startsWith('#/ch23')), JSON.stringify(ch23));
await goto(page, 'ch24/wrap');
check('the bridge is the PyTorch source layout, with its unchecked part said', (await body()).includes('aten/src/ATen/native/cuda/') && (await body()).includes('tools/amd_build/build_amd.py') && (await body()).includes("that is module X3's work"));

// ---- page rules
const misplaced = await page.$$eval('.prose > .note', (ns) => ns.filter((n) => n.previousElementSibling?.tagName !== 'P').map((n) => n.textContent.slice(0, 40)));
check('every margin note directly follows a paragraph', misplaced.length === 0, misplaced.join(' | '));
check('the page has no code block and no narrated scene', !(await page.$('.prose .code')) && !(await page.$('.prose .scene')));
check('the start page links here', await (async () => { await goto(page, 'ch24/start'); return !!(await page.$('.prose a[href="#/ch24/wrap"]')); })());

// ---- progress
const last = puts.at(-1) ?? { answers: {}, review: {} };
const A = last.answers;
check('all four answers recorded under ch24/wrap, first tries wrong where they were', ['q-gap', 'q-perboard', 'q-mi355x', 'q-next'].every((id) => A['ch24/wrap/' + id]?.correct === true) && last.review['ch24/wrap/q-gap']?.lapses === 1, JSON.stringify(A));
check('nothing was written outside ch24/wrap', Object.keys(A).every((k) => k.startsWith('ch24/wrap/')), Object.keys(A).join());

// ---- both routes, both widths, both themes: content, no sideways page scroll
for (const width of [1440, 390]) {
  await page.setViewport({ width, height: 900, deviceScaleFactor: 1 });
  for (const theme of ['light', 'dark']) {
    for (const route of ['ch24/start', 'ch24/wrap']) {
      await goto(page, route);
      await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, theme);
      await pause(60);
      const info = await page.evaluate(() => ({
        placeholder: /being written/i.test(document.querySelector('.page').textContent),
        overflow: document.documentElement.scrollWidth - window.innerWidth,
        h1: document.querySelector('.page h1')?.textContent,
      }));
      check(`${route} at ${width}px, ${theme}: has content, no sideways page scroll`, !info.placeholder && !!info.h1 && info.overflow <= 1, JSON.stringify(info));
      if (out) await page.screenshot({ path: `${out}-${route.replace('/', '-')}-${width}-${theme}.png`, fullPage: true });
    }
  }
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
