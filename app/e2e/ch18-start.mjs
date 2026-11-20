// Browser check for chapter 18's "Before you start": the format lens (CSR, CSC and COO slices
// for a clicked vertex or edge, storage counts) and the "Move the root" explorer behind the
// root prediction, compared with src/sim/bfs.ts, plus one question.
// Usage: node e2e/ch18-start.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { BOOK, EX1, backtrace, levels, showLevel, storage, toCSC, toCSR } from '../src/sim/bfs.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (title) => `section.bench[aria-label="${title}"]`;
const L = bench('Format lens');
const R = bench('Move the root');
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const clickSvg = async (sel) => {
  const ok = await page.evaluate((sel) => {
    const el = document.querySelector(sel);
    if (!el) return false;
    el.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    return true;
  }, sel);
  await pause(120);
  return ok;
};
const cells = (array, cls) =>
  page.$$eval(`${L} [data-array="${array}"] span.${cls}`, (s) => s.map((x) => ({ i: Number(x.dataset.i), v: Number(x.firstChild.textContent) })));
const pressIn = async (scope, label) => {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim().startsWith(label));
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(120);
  return ok;
};

await goto(page, 'ch18/start');
await page.waitForSelector(L);

// ---- the format lens, the book's graph
const csr = toCSR(BOOK);
const csc = toCSC(BOOK);
check('the lens draws the book\'s 9 vertices and 15 edges', (await page.$$(`${L} svg [data-vertex]`)).length === 9 && (await page.$$(`${L} svg [data-edge]`)).length === 15);
check('the arrays are Fig. 18.3\'s', (await page.$$eval(`${L} [data-array="csr-srcPtrs"] span`, (s) => s.map((x) => x.firstChild.textContent).join(','))) === csr.srcPtrs.join(','));
await clickSvg(`${L} svg [data-vertex="3"]`);
let dst = await cells('csr-dst', 'on');
check('vertex 3: the CSR slice is dst[7..8] = 4, 8', JSON.stringify(dst) === JSON.stringify([{ i: 7, v: 4 }, { i: 8, v: 8 }]), JSON.stringify(dst));
let ptr = await cells('csr-srcPtrs', 'hot');
check('vertex 3: the pointer pair is srcPtrs[3] = 7, srcPtrs[4] = 9', JSON.stringify(ptr) === JSON.stringify([{ i: 3, v: 7 }, { i: 4, v: 9 }]), JSON.stringify(ptr));
let n = await text(`${L} .narrate`);
check('vertex 3: the narration names edges 3 → 4 and 3 → 8', n.includes('3 → 4, 3 → 8') && n.includes('srcPtrs[3] = 7'), n);
check('vertex 3: its out-edges are drawn bold, its in-edge dashed', (await page.$eval(`${L} svg [data-edge="3-8"]`, (e) => e.dataset.look)) === 'out' && (await page.$eval(`${L} svg [data-edge="1-3"]`, (e) => e.dataset.look)) === 'in');
if (out) await page.screenshot({ path: out + '-lens-3.png' });

await clickSvg(`${L} svg [data-vertex="6"]`);
const src = await cells('csc-src', 'on');
check('vertex 6: the CSC slice is src[8..10] = 2, 5, 7', JSON.stringify(src.map((x) => x.v)) === '[2,5,7]' && JSON.stringify(src.map((x) => x.i)) === '[8,9,10]', JSON.stringify(src));
check('vertex 6 matches the model\'s CSC', JSON.stringify(src.map((x) => x.v)) === JSON.stringify(csc.src.slice(csc.dstPtrs[6], csc.dstPtrs[7])));
await clickSvg(`${L} svg [data-vertex="8"]`);
n = await text(`${L} .narrate`);
check('vertex 8: an empty CSR slice, no out-edges', n.includes('no edges leave 8') && (await cells('csr-dst', 'on')).length === 0, n);

await clickSvg(`${L} svg [data-edge="3-8"]`);
const coo = await cells('coo-src', 'hot');
n = await text(`${L} .narrate`);
check('edge 3 → 8 is COO entry 8, and src[12] in CSC', JSON.stringify(coo) === JSON.stringify([{ i: 8, v: 3 }]) && n.includes('src[12]'), JSON.stringify(coo) + ' ' + n);
let st = await text(`${L} [data-storage]`);
const sb = storage(BOOK);
check('storage: dense 81, CSR 25, COO 30', st.includes(`dense matrix ${sb.dense}`) && st.includes(`CSR ${sb.csr}`) && st.includes(`COO ${sb.coo}`) && sb.csr === 25, st);

// The Exercise 1 graph.
await pressIn(L, 'Exercise 1 graph');
check('the Exercise 1 preset: 8 vertices and 15 edges', (await page.$$(`${L} svg [data-vertex]`)).length === 8 && (await page.$$(`${L} svg [data-edge]`)).length === 15);
const se = storage(EX1);
st = await text(`${L} [data-storage]`);
check('the Exercise 1 preset: storage from the model', st.includes(`dense matrix ${se.dense}`) && st.includes(`CSR ${se.csr}`), st);
await clickSvg(`${L} svg [data-vertex="1"]`);
dst = await cells('csr-dst', 'on');
check('Exercise 1, vertex 1: dst[2..4] = 0, 4, 7 (Exercise 1b)', JSON.stringify(dst.map((x) => x.v)) === '[0,4,7]' && dst[0].i === 2, JSON.stringify(dst));
await pressIn(L, "the book's graph");

// ---- the root prediction and the explorer behind it
const pq = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Predict first') && e.textContent.includes('At what level is vertex 4')));
const committed = await pq.evaluate((e) => !e.querySelector('input'));
if (!committed) {
  check('before a prediction the explorer is closed', !(await page.$(R)));
  await (await pq.$('input')).type('3');
  await pq.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.includes('Commit')).click());
  await pause(200);
}
const reveal = await pq.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
check('the reveal gives level 4 by 2 → 7 → 0 → 1 → 4', reveal.includes('level 4, by 2 → 7 → 0 → 1 → 4'), reveal);
await page.waitForSelector(R);
const lv2 = levels(BOOK, 2);
const shownLevels = async () => page.$$eval(`${R} td[data-level]`, (t) => t.map((x) => x.textContent.trim()));
check('root 2: the level row is Fig. 18.4(b)', (await shownLevels()).join(',') === lv2.map(showLevel).join(','), (await shownLevels()).join(','));
const tag4 = await page.$eval(`${R} svg [data-vertex="4"] [data-tag]`, (e) => e.textContent);
check('root 2: vertex 4 is labelled 4 on the graph', tag4 === '4', tag4);
check('root 2: the walk back from 4 is 2 → 7 → 0 → 1 → 4', (await page.$eval(`${R} [data-path]`, (e) => e.dataset.path)) === backtrace(BOOK, 2, 4).join(' → '));
check('root 2: every one of the 9 vertices has a different level than with root 0', (await page.$eval(`${R} [data-changed]`, (e) => e.dataset.changed)) === '9');
const pathEdges = await page.$$eval(`${R} svg [data-edge][data-look="path"]`, (e) => e.map((x) => x.dataset.edge).sort().join(' '));
check('the path is drawn on edges 2-7, 7-0, 0-1, 1-4', pathEdges === ['0-1', '1-4', '2-7', '7-0'].sort().join(' '), pathEdges);
if (out) await page.screenshot({ path: out + '-root2.png' });
await clickSvg(`${R} svg [data-vertex="0"]`);
check('clicking vertex 0 makes it the root: levels of Fig. 18.4(a)', (await shownLevels()).join(',') === levels(BOOK, 0).map(showLevel).join(','));
check('root 0: vertex 4 back at level 2, path 0 → 1 → 4', (await page.$eval(`${R} svg [data-vertex="4"] [data-tag]`, (e) => e.textContent)) === '2' && (await page.$eval(`${R} [data-path]`, (e) => e.dataset.path)) === '0 → 1 → 4');
await page.evaluate((R) => {
  const s = document.querySelector(R + ' select');
  s.value = '8';
  s.dispatchEvent(new Event('change', { bubbles: true }));
}, R);
await pause(150);
n = await text(`${R} [data-path]`);
check('walking back from 8 reads its in-edges 3, 4, 6 and takes 3', n.includes('in-edges from 3 (level 2), 4 (level 2), 6 (level 2); step to 3') && (await page.$eval(`${R} [data-path]`, (e) => e.dataset.path)) === '0 → 1 → 3 → 8', n);
await clickSvg(`${R} svg [data-vertex="8"]`);
n = await text(`${R} [data-path]`);
check('root 8 reaches nothing: 8 has no out-edges', (await shownLevels()).filter((x) => x === '·').length === 8, (await shownLevels()).join(','));

// ---- a question
const q = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Which edges leave vertex 4')));
await q.evaluate((e) => e.querySelectorAll('input[type=radio]')[0].click());
await q.evaluate((e) => e.querySelector('button.primary').click());
await pause(150);
check('the CSR reading question accepts 4 → 5 and 4 → 8', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
const qn = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('lane 0 runs a loop 100 times')));
await (await qn.$('input[type=number]')).type('100');
await qn.evaluate((e) => e.querySelector('button.primary').click());
await pause(150);
check('the divergence question accepts 100 passes', (await qn.evaluate((e) => e.textContent)).includes('Correct.'));

// ---- dark theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
await page.$eval(L, (e) => e.scrollIntoView({ block: 'start' }));
if (out) await page.screenshot({ path: out + '-lens-dark.png' });
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
check('at a narrow width the page does not scroll sideways', narrow);
if (out) {
  await page.$eval(L, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
