// Browser check for chapter 15, "Before you start": the prediction of how far lab 5.3's
// kernel is from the vendor library and its reveal (computed here from /api/reports, the
// rows lab 15.1's benchmark records), the table of every shape, the chapter map and the
// labs table, the four prerequisite questions, margin notes, links, the page at 390 px.
// Usage: APP_URL=http://localhost:<port> node e2e/ch15-start.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file, and start from an empty one.
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
  return r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : JSON.stringify(EMPTY) });
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const norm = (s) => s.replace(/\s+/g, ' ').trim();
const shot = async (name, sel) => {
  if (!out) return;
  await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};
const fmt = (v, digits = 3) =>
  !isFinite(v) ? 'n/a' : v === 0 ? '0' : Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : Number(v.toPrecision(digits)).toString();
const compare = (predicted, measured) => {
  const r = predicted / measured;
  if (Math.abs(r - 1) <= 0.1) return 'Your prediction was within 10% of the measurement.';
  return r < 1 ? `The measurement is ${fmt(1 / r)} times your prediction.` : `Your prediction is ${fmt(r)} times the measurement.`;
};

await goto(page, 'ch15/start');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Before you start" of chapter 15', h1.includes('Before you start'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['Underneath your framework', 'Your starting point', 'The chapter at a glance', 'Before you start'];
check('sections in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
const text0 = norm(await page.$eval('.prose', (e) => e.textContent));
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('the bridge names the libraries and the batch-8 switch, labelled as not in the book', text0.includes('cuBLAS or cuBLASLt') && text0.includes('8 tokens or fewer') && text0.includes('Not in the book'));
const misplaced = await page.$$eval('.prose > .note', (ns) => ns.filter((n) => n.previousElementSibling?.tagName !== 'P').map((n) => n.textContent.slice(0, 60)));
check('every margin note follows a paragraph', misplaced.length === 0, misplaced.join(' | '));

// ---- the prediction and its reveal, against the reports
const LAB = 'ch15/lab1_coarse';
const labStatus = (run, v) => {
  if (v === 'work' && run?.work_state?.[LAB] === 'untouched') return 'none';
  const rs = (run?.results ?? []).filter((x) => x.lab === LAB && x.variant === v && x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const rows = (run, v) => run.results.filter((x) => x.lab === LAB && x.variant === v).flatMap((x) => x.bench);
const sets = Object.keys(reports)
  .filter((b) => b !== 'emu' && b !== 'race')
  .flatMap((b) => {
    const mine = labStatus(reports[b], 'work') === 'pass' ? rows(reports[b], 'work') : [];
    const set = mine.length ? mine : rows(reports[b], 'solution');
    return set.length ? [{ b, rows: set }] : [];
  });
const at = (set, name, m, k, n) => set.rows.find((r) => r.name === name && r.m === m && r.k === k && r.n === n);
const want = sets.flatMap((s) => {
  const a = at(s, 'tiled_5_3', 2048, 2048, 2048);
  const v = at(s, 'vendor', 2048, 2048, 2048);
  return a && v ? [{ b: s.b, value: (100 * a.gflops) / v.gflops, a: a.gflops, v: v.gflops }] : [];
});
const P = 'section.bench[aria-label="Your starting point"]';
check('nothing is revealed before the prediction is committed', (await page.$$(`${P} [data-measured]`)).length === 0 && (await page.$(`${P} [data-gap-table]`)) === null);
const GUESS = 20;
await page.type(`${P} input[aria-label="your prediction"]`, String(GUESS));
await page.evaluate((P) => [...document.querySelectorAll(`${P} button`)].find((b) => b.textContent.trim() === 'Commit prediction').click(), P);
await page.waitForFunction((P) => document.querySelector(`${P} [data-measured]`) && document.querySelector(`${P} [data-measured]`).dataset.measured !== 'loading', {}, P);
const shown = await page.$$eval(`${P} [data-measured]`, (ps) => ps.map((x) => ({ m: x.dataset.measured, b: x.dataset.backend ?? '', text: x.textContent })));
if (want.length) {
  check(`the reveal shows lab 5.3 as a percentage of the library at 2048 cubed (${want.map((w) => `${w.b} ${fmt(w.value)}`).join(', ')})`,
    shown.length === want.length && want.every((w) => { const s = shown.find((x) => x.b === w.b); return s && s.m === fmt(w.value) && s.text.includes(compare(GUESS, w.value)) && s.text.includes(`${fmt(w.a)} against ${fmt(w.v)} GFLOP/s`); }),
    JSON.stringify(shown));
  check('  and says the figures are lab 5.3\'s reference kernel and the library, from lab 15.1\'s benchmark', shown.every((s) => s.text.includes("lab 15.1's benchmark (lab 5.3's reference kernel, T = 16, and the vendor library)")), shown[0]?.text);
  for (const set of sets) {
    const tableRows = await page.$$eval(`${P} [data-gap-table="${set.b}"] tbody tr`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    const all = set.rows.filter((r) => r.name === 'tiled_5_3').map((r) => ({ r, lib: at(set, 'vendor', r.m, r.k, r.n) }));
    check(`${set.b}: the table lists every shape lab 15.1 timed (${all.length} rows), with lab 5.3, the library and the percentage`,
      all.length > 0 && tableRows.length === all.length && all.every(({ r, lib }, i) => tableRows[i][0] === `${r.m} × ${r.k} × ${r.n}` && tableRows[i][1] === fmt(r.gflops) && (!lib || (tableRows[i][2] === fmt(lib.gflops) && tableRows[i][3] === fmt((100 * r.gflops) / lib.gflops)))),
      JSON.stringify(tableRows.slice(0, 2)));
  }
} else {
  check('no lab 15.1 benchmark: the reveal says so and names the command', shown.length === 1 && shown[0].m === 'none' && shown[0].text.includes('python3 labs/pmpp.py run 15.1 --solution --bench'), JSON.stringify(shown));
}
check('the committed prediction is echoed', norm(await page.$eval(P, (e) => e.textContent)).includes(`You predicted ${GUESS} % of the library`));
await shot('gap', P);

// ---- the chapter map and the labs
const mapLinks = await page.$$eval('table[data-map] a', (a) => a.map((x) => x.getAttribute('href')));
const sections = ['15.1', '15.2', '15.3', '15.4', '15.5', '15.6', '15.7', '15.8', '15.9'];
check('the map links all nine sections 15.1 to 15.9', JSON.stringify(mapLinks) === JSON.stringify(sections.map((s) => '#/ch15/' + s)), mapLinks.join(' '));
const labs = await page.$$eval('table[data-labs] tr[data-lab]', (t) => t.map((r) => r.textContent));
check('seven labs listed, lab 15.6 as milestone M3', labs.length === 7 && labs[5].includes('milestone M3'), labs.join(' | '));

// ---- the four questions
const q = (part) => page.evaluateHandle((p) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(p)), part);
const answerNumber = async (part, value) => {
  const h = await q(part);
  await h.evaluate((e, v) => {
    const el = e.querySelector('input[type=number]');
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(60);
  await (await h.$('button.primary')).click();
  await pause(100);
  return h.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
};
const answerChoice = async (part, option) => {
  const h = await q(part);
  const ok = await h.evaluate((e, o) => {
    const l = [...e.querySelectorAll('.opts label')].find((x) => x.textContent.includes(o));
    if (!l) return false;
    l.querySelector('input').click();
    return true;
  }, option);
  await pause(60);
  await (await h.$('button.primary')).click();
  await pause(100);
  // The verdict, then the reason shown beside the picked option (Ask shows an option's `why` there).
  return ok && (await h.evaluate((e) => (e.querySelector('.verdict')?.textContent ?? '') + ' | ' + (e.querySelector('.opts label.wrong .muted')?.textContent ?? '')));
};
let v = await answerNumber('Lab 5.3’s kernel uses 16 × 16 tiles', 16);
check('q-tile-ratio: 16 is marked wrong and the answer given', v.includes('Not quite.') && v.includes('4'), v);
v = await answerNumber('Lab 5.3’s kernel uses 16 × 16 tiles', 4);
check('q-tile-ratio: 4 FLOP/B is correct, and the explanation points to 15.2', v.includes('Correct.') && v.includes('15.2') && v.includes('ch. 5 §5.4'), v);
v = await answerChoice('Where does acc live', 'In registers, like any automatic variable.');
check('q-local-array: registers is wrong, with its reason', v && v.includes('Not quite.') && v.includes('needs an address'), v);
v = await answerChoice('Where does acc live', 'In local memory');
check('q-local-array: local memory is correct, and the explanation points to 15.3', v && v.includes('Correct.') && v.includes('15.3'), v);
v = await answerNumber('how many lanes land in the busiest bank', 16);
check('q-bank-stride: 16 lanes is correct, and the explanation points to 15.6', v.includes('Correct.') && v.includes('15.6'), v);
v = await answerNumber('128 registers per thread', 25);
check('q-registers: 25% is correct, and the explanation points to 15.7', v.includes('Correct.') && v.includes('15.7') && v.includes('12.5%'), v);

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set([...sections.map((s) => '#/ch15/' + s), '#/ch15/labs', '#/review', '#/ch05/5.2', '#/ch05/5.4', '#/ch06/6.4', '#/ch06/6.6', '#/ch04/4.7']);
check(`every link goes to a known route (${links.length})`, links.every((l) => known.has(l)), links.filter((l) => !known.has(l)).join(', '));

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch15/start');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', P);

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
