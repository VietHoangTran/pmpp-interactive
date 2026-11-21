// Browser check for chapter 11, "Summary and exercises": the carry-forward list, the mixed
// check (one kernel shown from a checked snippet, five questions), the map of the book's
// exercises, the errata list and the project status computed from the reports. It also
// walks every route of the chapter for stacked margin notes, sideways scroll at 1440 and
// 390 px, "being written" placeholders and console errors.
// Usage: APP_URL=http://localhost:<port> node e2e/ch11-wrap.mjs [screenshot-prefix] [--dark]
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
const pick = async (q, index) => {
  await (await q.$$('input[type=radio]'))[index].click();
  await (await q.$('button.primary')).click();
  await pause(120);
};
const lastAnswers = () => (saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {});

// The models the page states (Wrap.snippets.test.ts pins them too, against the emulator and src/sim/scan.ts).
const warpStarts = (block, cols, width) => Array.from({ length: Math.ceil(Math.min(block, cols) / width) }, (_, k) => k * width).filter((c) => c > 0);
const ksMax = (xs, B, pad) => {
  let cur = Array.from({ length: B }, (_, i) => (i < xs.length ? xs[i] : pad));
  for (let s = 1; s < B; s *= 2) { const p = cur; cur = p.map((v, i) => (i >= s ? Math.max(p[i - s], v) : v)); }
  return cur;
};

await goto(page, 'ch11/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary and exercises", numbered 11.11', h1.includes('Summary and exercises') && (await page.$eval('.page', (e) => e.textContent)).includes('11.11'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', "The book's exercises", 'Where the PDF looks wrong', 'The project so far'];
check('the page has its five parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
check('the carry-forward list has seven points and cites the book section',
  (await page.$$eval('.prose > ul', (uls) => uls[0].children.length)) === 7 && (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('§11.11, pp. 285–286'));
const text0 = await body();
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('the carry-forward list names the top-p rule and the three byte counts', text0.includes('top-p keeps a token while that mass is') && text0.includes('16, 12 or 8'));

// ---- the kernel for the first question is the checked snippet's massBeforeRows
await page.waitForFunction(() => !document.querySelector('figure.code pre')?.textContent.startsWith('// loading'), { timeout: 10000 });
const file = await page.evaluate(() => fetch('/api/lab-file?path=' + encodeURIComponent('ch11/snippets/wrap_mass_before.cu')).then((r) => (r.ok ? r.text() : 'MISSING')));
const shownCode = (await page.$eval('figure.code pre', (e) => e.textContent)).trim();
const start = file.indexOf('__global__ void massBeforeRows');
const kernel = start >= 0 ? file.slice(start).trim() : 'MISSING';
check('the listing is massBeforeRows from labs/ch11/snippets/wrap_mass_before.cu, line for line', file !== 'MISSING' && shownCode === kernel, shownCode.slice(0, 80));
check('the listing is labelled as not a lab file, checked, and wrong on purpose',
  (await page.$eval('figure.code .tabs', (e) => e.textContent)).includes('not a lab file') &&
  (await page.$eval('figure.code .tabs .status', (e) => e.textContent)) === 'front-end-checked and emulator-checked; wrong on purpose');
check('the buffers are coloured as global memory, the warp totals as shared memory, the masses as registers',
  (await page.$$('figure.code .v-global')).length >= 4 && (await page.$$('figure.code .v-shared')).length >= 2 && (await page.$$('figure.code .v-reg')).length >= 5);
await shot('code', 'figure.code');

// ---- the five questions, each answered wrong then right
check(`no question is answered at the start (${(await page.$$('.q')).length} questions)`, (await page.$$('.q')).length === 5 && (await page.$$('.q .verdict')).length === 0);

const starts = warpStarts(256, 256, 32);
let q = await question('which outputs are wrong?');
await pick(q, 0);
check('mass before: "none" is wrong, because a shuffle stays inside the warp', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('stays inside the warp'));
await pick(q, 3);
check('mass before: "every token of warps 1 to 7" is wrong, because blockScan already added the earlier warps', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('Only one lane per warp'));
await pick(q, 1);
check(`mass before: tokens ${starts.join(', ')} are wrong, and the fix is a lane test with the earlier warps' total`,
  (await qText(q)).includes('Correct.') && (await qText(q)).includes(starts.join(', ')) && (await qText(q)).includes('warpSums_s[w − 1]') && (await qText(q)).includes('simd_prefix_exclusive_sum') && (await qText(q)).includes('64, 128 and 192'));

q = await question('Which of the stored outputs are wrong?');
await pick(q, 0);
check('max pad: "all of them" is wrong: which direction does each step read', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('Which direction'));
await pick(q, 1);
const maxOut = ksMax([-3.5, -1.25, -2, -0.5, -4, -1], 8, 0);
check(`max pad: none, and the outputs ${maxOut.slice(0, 6).join(', ')} come from the model`, (await qText(q)).includes('Correct.') && (await qText(q)).includes(maxOut.slice(0, 6).join(', ')) && maxOut[6] === 0 && maxOut[7] === 0);

q = await question('What does each give for the last token?');
await pick(q, 2);
check('tail: the reversed answer is wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('The other way round'));
await pick(q, 1);
check('tail: the loop gives 0.9999, Kogge-Stone 1, and the labs run the loop in double', (await qText(q)).includes('Correct.') && (await qText(q)).includes('The loop 0.9999, Kogge-Stone 1.') && (await qText(q)).includes('run it in double'));

q = await question('What change lets the scan use the whole GPU?');
await pick(q, 0);
check('batch 1: more threads per block is wrong, still one block', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('Still one block'));
await pick(q, 1);
check('batch 1: split the row across blocks, citing what llama.cpp and PyTorch do', (await qText(q)).includes('Correct.') && (await qText(q)).includes('CUB') && (await qText(q)).includes('D1, D4'));

q = await question("which gives sequence k's own inclusive scan");
await pick(q, 1);
check('packed: G[i] − G[s] is wrong, it drops the first element', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('drops it'));
await pick(q, 0);
check('packed: G[i] − G[s − 1], and the price in float', (await qText(q)).includes('Correct.') && (await qText(q)).includes('segmented'));

await pause(700); // the app saves 400 ms after the last change
const ans = lastAnswers();
const ids = ['q-mass-before', 'q-maxpad', 'q-tail', 'q-batch1', 'q-subtract'].map((s) => 'ch11/wrap/' + s);
check('all five answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 5);
check('the running count of answers appears after answering', (await body()).includes('So far in this chapter you have answered'));
await shot('questions', '.q');

// ---- the book's exercises
const rows = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) =>
  trs.map((tr) => ({ ex: tr.children[0].textContent.trim(), where: tr.children[2].textContent.trim(), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('the exercise table covers all seven exercises', rows.map((r) => r.ex).join(' | ') === '1 | 2 | 3 | 4 | 5 | 6 | 7', rows.map((r) => r.ex).join(' | '));
const sectionOf = { 1: '#/ch11/11.2', 2: '#/ch11/11.2', 3: '#/ch11/11.2', 4: '#/ch11/11.7', 5: '#/ch11/11.9', 6: '#/ch11/11.10', 7: '#/ch11/11.10' };
check('each exercise points at the section that works it', rows.every((r) => r.links.includes(sectionOf[r.ex])), JSON.stringify(rows.map((r) => [r.ex, r.links])));
check('exercises 4 and 5 point at the labs as well, as untested items', rows[3].links.includes('#/ch11/labs') && rows[4].links.includes('#/ch11/labs') && rows[3].where.includes('untested') && rows[4].where.includes('untested'));
check('exercise 3 gives both readings', text0.includes('20,481') && text0.includes('18,434'));

// ---- errata
const errata = await page.$$eval('ul[data-list="errata"] > li', (lis) => lis.map((li) => li.textContent));
check(`the errata list has ${errata.length} items, covering the notes' list and the facts file`, errata.length === 17, String(errata.length));
for (const [name, frag] of [
  ['N unused in Figs. 11.12, 11.13, 11.18', 'take N and never use it'],
  ['the missing comma', 'has no comma'],
  ['blockDim.x against BLOCK_DIM', 'BLOCK_DIM'],
  ['zeroed flags and counter', 'must be zero before every launch'],
  ['fetch_add(0) against load', 'load(acquire)'],
  ['the HIP mask and __shared__ in a device function', '0xffffffff'],
  ['__ syncthreads', '__ syncthreads();'],
  ['the race value drops x0', 'x₀ + 2x₁ + 2x₂'],
  ['the p. 259 slip', 'p. 259'],
  ['second Fig. 11.5', 'the second is Fig. 11.9'],
  ['blockSum', 'blockSum'],
  ['bottom portion and Fig. 11.20 shading', 'columns 9 and 11'],
  ['exercise 3 ambiguity', 'two blocks of 1,024'],
  ['iteration numbering', 'counted from 1'],
  ['typos', 'wap-level'],
  ['66.9 against 67', '66.9'],
  ['the copy claim is unverified', 'unverified'],
  ['what lanes below the stride receive', 'lanes below the stride'],
  ['deadlock named only in the summary', 'prevents deadlocks'],
]) check(`errata include ${name}`, errata.some((e) => e.includes(frag)));
check('readings of the PDF are marked as ours', text0.includes('our readings of one PDF'));
check('the wrap names what of the book is left out on purpose', ['Exercise 4 and exercise 5', 'Fig. 11.16(c)', 'chapter 18', 'p. 251', 'Thrust and CUB are named, not used'].every((s) => text0.includes(s)));

// ---- project status, computed from the reports
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = (run?.results ?? []).filter((x) => x.lab === id && x.variant === variant && x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
const done = backends.filter((b) => labStatus(reports[b], 'ch11/lab6_topp', 'work') === 'pass');
const m5 = await page.$eval('[data-m5]', (e) => ({ v: e.dataset.m5, text: e.textContent }));
check(`milestone M5 status is read from the reports (${done.join(',') || 'none'})`, m5.v === (done.join(',') || 'none') &&
  (backends.length === 0 ? m5.text.includes('No lab results') : done.length ? m5.text.includes('passes on') : m5.text.includes('does not pass')), JSON.stringify(m5));
check('the project paragraph names M5 and lab 11.6', text0.includes('milestone M5') && text0.includes("Lab 11.6's top-p keep mask"));
check('the next chapter is named and linked', text0.includes('Chapter 12 uses the scan') && (await page.$$eval('.prose a[href="#/ch12"]', (as) => as.length)) === 1);

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
check(`links go to routes of this chapter or the next (${links.join(', ')})`, links.length >= 6 && links.every((l) => /^#\/(ch11\/(start|labs|wrap|11\.\d+)|ch12)$/.test(l)), links.join(', '));
await page.evaluate(() => [...document.querySelectorAll('.prose a')].find((a) => a.getAttribute('href') === '#/ch11/labs').click());
await page.waitForFunction(() => document.querySelector('.page h1')?.textContent.includes('Labs'), { timeout: 10000 });
check('the link to the labs page opens it', (await page.$eval('.page h1', (e) => e.textContent)).includes('Labs'));

// ---- the whole chapter: every route at 1440 and 390 px
{
  const chapterIndex = await page.evaluate(() => [...document.querySelectorAll('nav a, aside a, .rail a')].map((a) => a.getAttribute('href')).filter((h) => h && h.startsWith('#/ch11/')));
  const routes = ['start', '11.1', '11.2', '11.3', '11.4', '11.5', '11.6', '11.7', '11.8', '11.9', '11.10', 'labs', 'wrap'];
  check('the chapter rail lists every route', routes.every((r) => chapterIndex.includes('#/ch11/' + r)), chapterIndex.join(' '));
  for (const width of [1440, 390]) {
    await page.setViewport({ width, height: 900, deviceScaleFactor: 1 });
    const bad = [];
    for (const r of routes) {
      const before = errors.length;
      await goto(page, 'ch11/' + r);
      const res = await page.evaluate(() => ({
        stacked: document.querySelectorAll('.prose > .note + .note').length,
        overflow: document.documentElement.scrollWidth - window.innerWidth,
        placeholder: /being written|being finished/i.test(document.querySelector('.prose')?.textContent ?? ''),
      }));
      // A narration clip that has not been generated yet is a 404 the scenes expect.
      const errs = errors.slice(before).filter((e) => !/\.mp3|narration/.test(e));
      errors.splice(before, errors.length - before, ...errs);
      if (res.overflow > 1 || res.placeholder || (width === 1440 && res.stacked) || errs.length) bad.push(`${r}: ${JSON.stringify(res)} ${errs.join(' | ')}`);
    }
    check(`every route at ${width} px: no sideways scroll, no "being written", no stacked margin notes, no errors`, bad.length === 0, bad.join('; '));
  }
}
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch11/wrap');
await shot('phone', 'table[data-table="exercises"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
