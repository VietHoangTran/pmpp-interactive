// Browser check for chapter 22, "Before you start": the chapter map, the labs paragraph, the
// bridge to inference (labelled as ours), the five prerequisite questions with their
// explanations, margin notes, links, and the page at 390 px.
// Usage: APP_URL=http://localhost:<port> node e2e/ch22-start.mjs [screenshot-prefix] [--dark]
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

await goto(page, 'ch22/start');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Before you start" of chapter 22', h1.includes('Before you start'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['A chapter without code', 'Where you will meet these decisions', 'Before you start'];
check('sections in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
const text0 = norm(await page.$eval('.prose', (e) => e.textContent));
check('the chapter says plainly it has no code listings, exercises or device figures', text0.includes('no code listings in this chapter, no exercises and no device figures'));
check('both labs are named, the second as milestone M8', text0.includes('Lab 22.1 cuts a decode-time matrix-vector product two ways') && text0.includes('Lab 22.2 is milestone M8'));
check('the inference bridge is labelled as not in the book', text0.includes('Not in the book') && text0.includes('These four mappings onto inference are ours'));
check("Wing's paper is cited with volume, issue and pages", text0.includes('Computational thinking') && text0.includes('49(3), 2006, pp. 33–35'));
check('no Today note on this page', (await page.$$('.prose .note.today')).length === 0);
const misplaced = await page.$$eval('.prose > .note', (ns) => ns.filter((n) => n.previousElementSibling?.tagName !== 'P').map((n) => n.textContent.slice(0, 60)));
check('every margin note follows a paragraph', misplaced.length === 0, misplaced.join(' | '));
const sources = await page.$$eval('.prose > .note', (ns) => ns.map((n) => n.textContent.trim()).filter((t) => t.startsWith('Book')));
check('book pages are cited (p. 529 and §22.6, p. 540)', sources.includes('Book p. 529') && sources.includes('Book §22.6, p. 540'), sources.join(' | '));

// ---- the map
const map = await page.$$eval('table[data-map] tbody tr', (rows) => rows.map((r) => [...r.children].map((c) => c.textContent.trim())));
check('the map has the four steps in the summary’s order, then 22.3 and 22.5',
  JSON.stringify(map.map((r) => r[0])) === JSON.stringify(['Problem formulation', 'Algorithm selection', 'Problem decomposition', 'Optimisation', "Amdahl's law for a whole application", 'Batching']),
  JSON.stringify(map.map((r) => r[0])));
check('each step names its section', JSON.stringify(map.map((r) => r[1])) === JSON.stringify(['22.4', '22.1', '22.2', 'chapters 2 to 21', '22.3', '22.5']), JSON.stringify(map.map((r) => r[1])));
const mapLinks = await page.$$eval('table[data-map] a', (a) => a.map((x) => x.getAttribute('href')));
check('the map links sections 22.1 to 22.5', ['22.1', '22.2', '22.3', '22.4', '22.5'].every((s) => mapLinks.includes('#/ch22/' + s)), mapLinks.join(' '));
await shot('map', 'table[data-map]');

// ---- the five questions
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
  return norm(await h.evaluate((e) => e.querySelector('.verdict')?.textContent ?? ''));
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
  return ok && norm(await h.evaluate((e) => (e.querySelector('.verdict')?.textContent ?? '') + ' | ' + (e.querySelector('.opts label.wrong .muted')?.textContent ?? '')));
};
check('five questions on the page', (await page.$$('.prose .q')).length === 5);

let v = await answerNumber('You make every kernel 4 times faster', 4);
check('q-decode-amdahl: 4× is marked wrong and the answer 2.5 given', v.includes('Not quite.') && v.includes('The answer is 2.5 ×'), v);
v = await answerNumber('You make every kernel 4 times faster', 2.5);
check('q-decode-amdahl: 2.5 is correct; the explanation works it and points to 22.3', v.includes('Correct.') && v.includes('1 / 0.4 = 2.5') && v.includes('22.3') && v.includes('ch. 1 §1.3'), v);
v = await answerNumber('at most what percentage of the peak arithmetic', 0.5);
check('q-gemv-ceiling: 0.5 is marked wrong', v.includes('Not quite.') && v.includes('The answer is 2.5 %'), v);
v = await answerNumber('at most what percentage of the peak arithmetic', 2.5);
check('q-gemv-ceiling: 2.5% is correct; the explanation names batching (22.5) and quantisation', v.includes('Correct.') && v.includes('0.5 / 20 = 2.5%') && v.includes('22.5') && v.includes('quantisation'), v);
v = await answerChoice('passes every test on the emu', 'gives every thread a private copy');
check('q-emu-blind: the private-copy option is wrong, with its reason', v && v.includes('Not quite.') && v.includes('AddressSanitizer'), v);
v = await answerChoice('passes every test on the emu', 'runs each thread until it reaches a synchronisation point');
check('q-emu-blind: the right option; the explanation names race, ThreadSanitizer and scatter in 22.2', v && v.includes('Correct.') && v.includes('ThreadSanitizer') && v.includes('scatter') && v.includes('22.2'), v);
v = await answerChoice('how many threads write each partial-sum slot', 'Two, one for each operand');
check('q-tree-writers: "two" is wrong, with its reason', v && v.includes('Not quite.') && v.includes('only the thread that owns the slot writes it'), v);
v = await answerChoice('how many threads write each partial-sum slot', 'Exactly one');
check('q-tree-writers: exactly one is correct; the explanation says output-centric, judged per iteration', v && v.includes('Correct.') && v.includes('output-centric') && v.includes('one iteration at a time'), v);
v = await answerNumber('At batch 3, what percentage of the rows', 3);
check('q-m3-rows: 3% is marked wrong', v.includes('Not quite.') && v.includes('The answer is 2.34 %'), v);
v = await answerNumber('At batch 3, what percentage of the rows', 3 / 1.28);
check('q-m3-rows: 3/128 is correct; the explanation says 125 padded rows and points to 22.5', v.includes('Correct.') && v.includes('125 rows are zero padding') && v.includes('22.5'), v);
await shot('questions', '.prose .q');

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set([
  ...['start', '22.1', '22.2', '22.3', '22.4', '22.5', 'labs', 'wrap'].map((s) => '#/ch22/' + s),
  '#/review', '#/ch01/1.3', '#/ch05/5.1', '#/ch09/9.2', '#/ch10/10.3', '#/ch15/labs',
]);
check(`every link goes to a known route (${links.length})`, links.every((l) => known.has(l)), links.filter((l) => !known.has(l)).join(', '));

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch22/start');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'table[data-map]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
