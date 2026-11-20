// Browser check for chapter 22, "Summary": the carry-forward list, the mixed check (five
// questions, each answered wrong then right), this machine's lab 22.1 number beside the
// router question, the map of the book's examples, the errata list, what was left out, and
// the project status computed from the reports.
// Usage: APP_URL=http://localhost:<port> node e2e/ch22-wrap.mjs [screenshot-prefix] [--dark]
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
const qText = async (q) => (await q.evaluate((e) => e.textContent)).replace(/\s+/g, ' ');
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

await goto(page, 'ch22/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const body = async () => (await page.$eval('.prose', (e) => e.textContent)).replace(/\s+/g, ' ');

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary", numbered 22.6', h1.includes('Summary') && (await page.$eval('.page', (e) => e.textContent)).includes('22.6'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', "The book's exercises", 'Where the PDF looks wrong', 'What this course left out', 'The project so far'];
check('the page has its six parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
check('the carry-forward list has seven points and cites §22.6, p. 540',
  (await page.$$eval('.prose > ul', (uls) => uls[0].children.length)) === 7 &&
  (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('§22.6') && (await page.$eval('.prose aside.note', (e) => e.textContent)).includes('540'));
const text0 = await body();
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('the exercises section says the chapter has none and maps the book\'s examples to pages',
  text0.includes('This chapter has none') && (await page.$$eval('table[data-table="examples"] tbody tr', (t) => t.length)) === 7);
const errata = await page.$$eval('ul[data-list="errata"] li', (li) => li.map((x) => x.textContent.replace(/\s+/g, ' ')));
check('the errata list has the nine slips from the notes',
  errata.length === 9 && ['Chapter 12', 'benefit over the gather', '16.8×', 'grid points', 'overflow atoms', 'warp-level shuffling register tiling', 'matrix-vector', 'attention heads', 'well-define'].every((s, i) => errata[i].includes(s)),
  JSON.stringify(errata.map((e) => e.slice(0, 40))));
check('the Beyond note says the mixed check\'s numbers and examples are ours',
  await page.$$eval('.prose > aside.note', (as) => as.some((a) => a.textContent.includes('in these five questions is ours'))));

// ---- the five questions, each answered wrong then right
check(`no question is answered at the start (${(await page.$$('.q')).length} questions)`, (await page.$$('.q')).length === 5 && (await page.$$('.q .verdict')).length === 0);

let q = await question('launching them, whatever the batch');
await typeAnswer(q, 22.9);
check('host amortisation: the kernels-only 22.9 is wrong and 24 is given', (await qText(q)).includes('Not quite. The answer is 24'), await qText(q));
await typeAnswer(q, 24);
check('host amortisation: 24 is right; kernels alone give 22.9; each user waits 16 ms against 12',
  (await qText(q)).includes('Correct.') && (await qText(q)).includes('= 22.9') && (await qText(q)).includes('16 ms against 12'));

q = await question('rotary position embedding');
await pick(q, 0);
check('rope: "atomics decide" is wrong: nothing to collide', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('nothing to collide'));
await pick(q, 3);
check('rope: "parallelism decides" is wrong: the split changes the grain', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('changes the grain'));
await pick(q, 2);
check('rope: the two views coincide, as with odd-even sort', (await qText(q)).includes('Correct.') && (await qText(q)).includes('odd-even sort'));

q = await question('greedy decoder computes a softmax');
await pick(q, 1);
check('greedy: 16-bit softmax is wrong: rounding can create a tie', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('same value'));
await pick(q, 0);
check('greedy: argmax of the logits is right; this is problem formulation; lab 22.2 keeps probs for M5',
  (await qText(q)).includes('Correct.') && (await qText(q)).includes('problem formulation') && (await qText(q)).includes('M5'));

q = await question('at least every 40 ms');
await typeAnswer(q, 1164);
check('latency limit: batch 64\'s 1,164 tokens/s is wrong, 1,067 is given', (await qText(q)).includes('Not quite. The answer is 1067 tokens'), await qText(q));
await typeAnswer(q, 1067);
check('latency limit: 1,067 at batch 32 is right; batch 64 would wait 55 ms', (await qText(q)).includes('Correct.') && (await qText(q)).includes('wait 55 ms'));

q = await question('split-K to gain the most');
await pick(q, 3);
check('router: "none more than another" is wrong: bytes the same, loads in flight', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('loads in flight'));
await pick(q, 1);
check('router: the 4096 → 8 router is right; the book\'s trade is named', (await qText(q)).includes('Correct.') && (await qText(q)).includes('exposed parallelism and atomics'));

await pause(700); // the app saves 400 ms after the last change
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
const ids = ['q-host-amortise', 'q-rope', 'q-greedy', 'q-slo', 'q-router'].map((s) => 'ch22/wrap/' + s);
check('all five answers were sent to /api/progress under their ids, and none reached the real file', ids.every((id) => ans[id] !== undefined), Object.keys(ans).join(', '));
check('every question names where its answer comes from', (await page.$$eval('.q .verdict', (vs) => vs.filter((v) => v.textContent.includes('Source:')).length)) === 5);
check('the running count of answers appears after answering', (await body()).includes('So far in this chapter you have answered'));
await shot('questions', '.q');

// ---- this machine's lab 22.1 number beside the router question
const results = (run, id, variant) => (run ? run.results.filter((x) => x.lab === id && x.variant === variant) : []);
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = results(run, id, variant).filter((x) => x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const real = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
{
  const want = real.flatMap((b) => {
    const mine = labStatus(reports[b], 'ch22/lab1_gemv', 'work') === 'pass' ? results(reports[b], 'ch22/lab1_gemv', 'work').flatMap((x) => x.bench) : [];
    const rows = (mine.length ? mine : results(reports[b], 'ch22/lab1_gemv', 'solution').flatMap((x) => x.bench))
      .filter((r) => r.name === 'gemv' && r.in === 4096 && r.out === 16 && r.batch === 1 && r.seconds);
    const g = rows.find((r) => r.kernel === 'gather');
    const s = rows.filter((r) => r.kernel === 'splitk').sort((a, b) => a.seconds - b.seconds)[0];
    return g && s ? [{ b, ratio: fmt(g.seconds / s.seconds) }] : [];
  });
  const shown = await page.$$eval('[data-machine]', (ps) => ps.map((p) => ({ b: p.dataset.machine, ratio: p.dataset.ratio ?? '' })));
  check(want.length ? `this machine's narrow-layer ratio is shown (${want.map((w) => `${w.b} ${w.ratio}`).join(', ')})` : 'no lab 22.1 rows, and the line says what to run',
    want.length ? JSON.stringify(shown) === JSON.stringify(want) : shown.length === 1 && shown[0].b === 'none', JSON.stringify(shown));
}

// ---- project status
{
  const where = (b) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  const passing = (id) => [...real, 'emu'].filter((b) => labStatus(reports[b], id, 'work') === 'pass');
  const m8 = passing('ch22/lab2_batching');
  const shown = await page.$eval('[data-m8]', (e) => ({ v: e.dataset.m8, text: e.textContent }));
  const none = real.length === 0 && !reports.emu;
  check(`project: M8 status from the reports (${m8.join(',') || 'none'})`, shown.v === (m8.join(',') || 'none') &&
    (none ? shown.text.includes('No lab results') : m8.length ? shown.text.includes(m8.map(where).join(' and ')) : shown.text.includes('has not passed on this machine yet')), JSON.stringify(shown));
  check('project: the next chapter and X2 are named', text0.includes('Chapter 23 takes up') && text0.includes('module X2'));
}

// ---- links and margin notes
const known = await page.evaluate(async () => {
  const reg = await import('/src/chapters/registry.ts');
  const out = [];
  for (const id of reg.CHAPTER_IDS) {
    const ch = await reg.loadChapter(id);
    out.push(`#/${id}`, ...ch.sections.map((s) => `#/${id}/${s.id}`));
  }
  return out;
});
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
check('the page links to every section of the chapter and its labs', ['22.1', '22.2', '22.3', '22.4', '22.5', 'labs'].every((s) => links.includes(`#/ch22/${s}`)), links.join(' '));
check('every link goes to a known route', links.every((l) => known.includes(l)), links.filter((l) => !known.includes(l)).join(' '));
const misplaced = await page.$$eval('.prose > aside.note', (as) => as.filter((a) => {
  const prev = a.previousElementSibling;
  return prev && !['P', 'UL', 'OL', 'ASIDE'].includes(prev.tagName);
}).map((a) => `${a.previousElementSibling.tagName}.${a.previousElementSibling.className}`));
check('every margin note follows a paragraph or another note', misplaced.length === 0, misplaced.join(', '));
await shot('errata', 'ul[data-list="errata"]');

await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await page.goto('about:blank');
await goto(page, 'ch22/wrap');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
