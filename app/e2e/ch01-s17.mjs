// Browser check for section 1.7: the prediction about chapter 20, the "Reading path"
// explorer, the edition map and the questions. Everything shown is compared with
// the tested model in src/sim/bookmap.ts.
// Usage: node e2e/ch01-s17.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import * as bm from '../src/sim/bookmap.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file, and start from an empty one so that the
// prediction and the questions are unanswered whatever the reader has already done.
const puts = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.url().includes('/api/progress')) {
    if (r.method() === 'PUT') {
      puts.push(JSON.parse(r.postData()));
      return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
    }
    return r.respond({ status: 200, contentType: 'application/json', body: '{"version":1,"sections":{},"answers":{},"review":{}}' });
  }
  return r.continue();
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent);
const question = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const clickText = async (scope, label) => {
  const done = await page.evaluate(
    (scope, label) => {
      const root = [...document.querySelectorAll('section.bench')].find((b) => b.getAttribute('aria-label') === scope);
      const btn = [...root.querySelectorAll('button')].find((b) => b.textContent.trim() === label);
      if (!btn || btn.disabled) return false;
      btn.click();
      return true;
    },
    scope,
    label,
  );
  await pause(80);
  return done;
};
const and = (xs) => (xs.length <= 1 ? xs.join('') : xs.slice(0, -1).join(', ') + ' and ' + xs[xs.length - 1]);

await goto(page, 'ch01/1.7');
const P = 'Reading path';
await page.waitForSelector(bench(P));

// ---- the prediction ----
const before20 = bm.chaptersBefore([20]);
let q = await question('how many chapters must you have read');
check('prediction hides its answer until committed', !(await q.evaluate((e) => e.textContent)).includes(and(before20)));
await (await q.$('input[type=number]')).type('18');
await (await q.$('button.primary')).click();
await pause(150);
const pred = await q.evaluate((e) => e.textContent);
// A line would put chapters 2 to 19 before chapter 20: eighteen of them.
const line = bm.CHAPTERS.filter((c) => c.n > 1 && c.n < 20).length;
check(
  'committing reveals 7 and the chapters, and reacts to a guess of 18, the count for a line',
  line === 18 && before20.length === 7 && pred.includes('You predicted 18') && pred.includes(`${before20.length}: chapters ${and(before20)}`) &&
    pred.includes('Eighteen would be right if the chart were a line: chapters 2 to 19.'),
  pred,
);

// ---- reading path ----
const stat = async (name) => Number(await text(`${bench(P)} [data-stat="${name}"]`));
const listed = () => page.$$eval(`${bench(P)} [data-role=order] li`, (li) => li.map((x) => x.dataset.item));
const laterListed = () => page.$$eval(`${bench(P)} [data-role=later] li`, (li) => li.map((x) => Number(x.dataset.item)));
const states = () =>
  page.$$eval(`${bench(P)} svg g[data-ch]`, (g) => Object.fromEntries(g.map((x) => [x.dataset.ch, x.dataset.state])));
const chaptersIn = (st, ...kinds) => Object.keys(st).map(Number).filter((n) => kinds.includes(st[n])).sort((a, b) => a - b);
const items = (xs) => xs.map(String);

check('the chart draws 24 chapters, 5 extra pages and every arrow of the model', (await page.$$(`${bench(P)} svg g[data-ch]`)).length === bm.CHAPTERS.length && (await page.$$(`${bench(P)} svg g[data-extra]`)).length === bm.EXTRAS.length && (await page.$$(`${bench(P)} svg path[data-edge]`)).length === bm.EDGES.length);
const kinds = await page.$$eval(`${bench(P)} svg path[data-edge]`, (p) => Object.fromEntries(p.map((x) => [x.dataset.edge, x.dataset.kind])));
check(
  'the dashed arrow and the two uncertain arrows are drawn differently, with a legend',
  kinds['9>10'] === 'dashed' && kinds['9>17'] === 'uncertain' && kinds['9>21'] === 'uncertain' && Object.values(kinds).filter((k) => k !== 'solid').length === 3 &&
    (await text(`${bench(P)} [data-role=legend]`)).includes('drawn dashed in the book') && (await text(`${bench(P)} [data-role=legend]`)).includes('ambiguous junction'),
  JSON.stringify(kinds),
);

// Default goal: fast matmul.
let st = await states();
check('fast matmul: five chapters before chapter 15, six with it', (await stat('before')) === 5 && (await stat('total')) === 6 && bm.chaptersBefore([15]).length === 5);
check('fast matmul: the list is the model\'s reading order without chapter 1', (await listed()).join() === items(bm.readingOrder([15]).filter((n) => n !== 1)).join(), (await listed()).join());
check('fast matmul: chapter 15 is the selection, Part I is kept, all of Part II is dimmed', st[15] === 'target' && chaptersIn(st, 'pre').join() === '1,2,3,4,5,6' && [7, 9, 10, 13].every((n) => st[n] === 'dim'), JSON.stringify(st));
check('fast matmul: nothing builds on chapter 15', (await text(`${bench(P)} [data-role=leads]`)).includes('Nothing in the chart builds on chapter 15'));
const onEdges = () => page.$$eval(`${bench(P)} svg path[data-edge][data-on="1"]`, (p) => p.map((x) => x.dataset.edge).sort());
check('fast matmul: only the arrows on the way to 15 are lit', (await onEdges()).join() === ['1>2', '2>3', '2>4', '3>5', '4>5', '5>6', '6>15'].sort().join(), (await onEdges()).join());
if (out) await page.screenshot({ path: out + '-path-matmul.png' });

await clickText(P, 'Attention');
st = await states();
check('attention: seven chapters before chapter 20', (await stat('before')) === 7 && (await listed()).join() === '2,3,4,5,6,7,19,20', (await listed()).join());
check('attention: 7 and 19 are kept by the arrows', st[7] === 'pre' && st[19] === 'pre' && st[20] === 'target' && st[10] === 'dim');

await clickText(P, 'Sampling');
const sampling = bm.goal('sampling');
check('sampling: eleven chapters with the goal, the longest road, then module X1', (await stat('total')) === 11 && (await stat('before')) === 7 && (await listed()).join() === items([...bm.readingOrder(sampling.chapters).filter((n) => n !== 1), 'x1']).join(), (await listed()).join());
check('sampling: module X1 is marked in the strip of extra pages', (await page.$eval(`${bench(P)} svg g[data-extra=x1]`, (g) => g.dataset.state)) === 'target');

await clickText(P, 'Quantised kernels');
check('quantised kernels: chapter 15, then Appendix A and module X2', (await listed()).join() === '2,3,4,5,6,15,appA,x2', (await listed()).join());

// Click chapter boxes.
const clickChapter = async (n) => {
  await page.click(`${bench(P)} svg g[data-ch="${n}"]`);
  await pause(80);
};
await clickChapter(14);
st = await states();
check('clicking chapter 14: ten chapters before it, five of them in Part II', (await stat('before')) === 10 && chaptersIn(st, 'pre').filter((n) => bm.chapter(n).part === 'II').join() === '9,10,11,12,13' && (await text(`${bench(P)} [data-role=selection]`)).includes('Chapter 14: Sorting'), JSON.stringify(st));
check('chapter 14 leads nowhere further only under our reading of the junction, and says so', (await text(`${bench(P)} [data-role=leads]`)).includes('reading the junction under it as starting at chapter 9'), await text(`${bench(P)} [data-role=leads]`));
await clickChapter(22);
check('clicking chapter 22: twelve before it', (await stat('before')) === 12 && bm.chaptersBefore([22]).length === 12);
await clickChapter(10);
check('clicking chapter 10: chapter 9 is offered as optional, by the dashed arrow', (await stat('before')) === 5 && (await text(`${bench(P)} [data-role=optional]`)).includes('chapter 9 before chapter 10'));
await clickChapter(6);
check('clicking chapter 6: everything in Parts II and III except 24 builds on it', (await text(`${bench(P)} [data-role=leads]`)).includes(and(bm.leadsTo([6]))), await text(`${bench(P)} [data-role=leads]`));
// Keyboard: a box is a button.
await page.$eval(`${bench(P)} svg g[data-ch="23"]`, (g) => g.focus());
await page.keyboard.press('Enter');
await pause(80);
check('a chapter box can be chosen from the keyboard', (await text(`${bench(P)} [data-role=selection]`)).includes('Chapter 23') && (await text(`${bench(P)} [data-role=leads]`)).includes('leads nowhere further'));

// This course's order.
await clickText(P, 'Attention');
await clickText(P, "This course's order");
const att = bm.coursePath([20]);
st = await states();
check("course order, attention: the list is the model's course path", (await listed()).join() === items(att.order.filter((i) => i !== 1)).join() && (await listed()).join() === '2,3,4,5,6,10,20', (await listed()).join());
check('course order, attention: reduction is marked as added, 7 and 19 as read later', st[10] === 'added' && st[7] === 'later' && st[19] === 'later' && (await laterListed()).join() === att.later.join() && att.later.join() === '7,19', JSON.stringify(st));
check('course order, attention: the added chapter is labelled in the list', (await text(`${bench(P)} [data-role=order] li[data-item="10"]`)).includes('added by this course'));
const ahead = await page.$$eval(`${bench(P)} svg path[data-ahead="1"]`, (p) => p.map((x) => x.dataset.edge));
check('course order: the five arrows it runs ahead of are drawn heavy, with a legend', ahead.sort().join() === bm.courseDepartures().map((x) => `${x.from}>${x.to}`).sort().join() && ahead.length === 5 && !!(await page.$(`${bench(P)} [data-legend=ahead]`)), ahead.join());
const ordinals = await page.$$eval(`${bench(P)} svg g[data-ch], ${bench(P)} svg g[data-extra]`, (g) => g.map((x) => [x.dataset.ch ?? x.dataset.extra, Number(x.querySelector('[data-ordinal]')?.dataset.ordinal)]));
const order = bm.courseOrder();
check("course order: every box carries its position in the model's order", ordinals.length === 29 && ordinals.every(([id, o]) => order[o - 1] === (/^\d+$/.test(id) ? Number(id) : id)), JSON.stringify(ordinals));
if (out) await page.screenshot({ path: out + '-path-course.png' });
await clickText(P, 'Sampling');
check('course order, sampling: merge is read later', (await laterListed()).join() === '13' && (await listed()).join() === '2,3,4,5,6,9,10,11,12,14,x1' && (await text(`${bench(P)} [data-role=later]`)).includes('radix sort'), (await listed()).join());
await clickText(P, 'The book\'s arrows');
await clickText(P, 'Reading real backends');
const x3 = bm.coursePath([], ['x3']);
check("reading real backends switches to this course's order and lists everything before module X3", (await page.$eval(`${bench(P)} [data-role=mode] button[aria-pressed=true]`, (b) => b.textContent)) === "This course's order" && (await listed()).join() === items(x3.order.filter((i) => i !== 1)).join() && (await listed()).at(-1) === 'x3', (await listed()).join());
await clickText(P, 'The book\'s arrows');
check('by the arrows alone, module X3 is off the chart', (await text(`${bench(P)} [data-role=off-chart]`)).includes('not on the book\'s chart'));

// Shading by depth tier.
await clickText(P, 'Fast matmul');
await page.click(`${bench(P)} label.check input`);
await pause(80);
const tiers = await page.$$eval(`${bench(P)} svg g[data-ch]`, (g) => Object.fromEntries(g.map((x) => [x.dataset.ch, [x.dataset.tier, x.querySelector('rect:not([fill=none])').getAttribute('fill')]])));
check('shading: each box carries the model\'s tier, and deep chapters are filled', bm.CHAPTERS.every((c) => tiers[c.n][0] === c.tier && (tiers[c.n][1] === 'var(--ink)') === (c.tier === 'Deep')) && !!(await page.$(`${bench(P)} [data-legend=tier]`)), JSON.stringify(tiers));
if (out) await page.screenshot({ path: out + '-path-tiers.png' });

// ---- edition map (static) ----
const mapRows = await page.$$eval('table[data-role=edition-map] tbody tr', (trs) => trs.map((tr) => ({ n: tr.dataset.n, fate: tr.dataset.fate, text: tr.textContent.replace(/\s+/g, ' '), links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
const changed = bm.FOURTH_TO_FIFTH.filter((c) => c.fate.kind !== 'same');
check('the edition map has one row for chapters 1 to 11 and one for each chapter after', mapRows.length === 1 + changed.length && mapRows[0].n === `1-${bm.firstRenumbered() - 1}` && mapRows[0].text.includes('the same numbers'), JSON.stringify(mapRows.map((r) => r.n)));
check("every row carries the model's verdict", mapRows.slice(1).every((r, i) => Number(r.n) === changed[i].n && r.fate === changed[i].fate.kind), JSON.stringify(mapRows.map((r) => r.fate)));
const mapRow = (n) => mapRows.find((r) => r.n === String(n));
check('4th-edition chapter 16 is Deep learning, now chapter 19, linked to this course', bm.fifthFor(16) === 19 && mapRow(16).text.includes('16. Deep learning') && mapRow(16).links.join() === '#/ch19', JSON.stringify(mapRow(16)));
check('4th-edition chapter 14 is Sparse matrix computation, now chapter 17', mapRow(14).text.includes('Sparse matrix computation') && mapRow(14).links.join() === '#/ch17');
check('removed chapters: 17 points to chapter 21, 21 to nothing, 22 to Appendix C', mapRow(17).text.includes('removed') && mapRow(17).links.join() === '#/ch21' && mapRow(21).text.includes('removed') && mapRow(21).links.length === 0 && mapRow(22).links.join() === '#/appC' && bm.fifthFor(21) === null);
check('the note under the map says the pairing is our inference', (await page.$eval('table[data-role=edition-map] + p', (e) => e.textContent)).includes('our inference'));
if (out) await (await page.$('table[data-role=edition-map]')).screenshot({ path: out + '-edition.png' });

// ---- questions ----
q = await question('Which chapter of the 5th edition do you open');
await (await q.$('input[type=number]')).type('16');
await (await q.$('button.primary')).click();
await pause(150);
check('edition question: the old number is marked wrong and the answer is 19', (await q.evaluate((e) => e.textContent)).includes('Not quite. The answer is 19'));
q = await question('which of these can wait until afterwards');
await (await q.$$('input[type=radio]'))[3].click();
await (await q.$('button.primary')).click();
await pause(150);
check('postpone question: chapter 7 is correct', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
q = await question('how many Part II chapters come before chapter 14');
await (await q.$('input[type=number]')).type('5');
await (await q.$('button.primary')).click();
await pause(150);
check('sort question: 5 is correct and the explanation names the chapters', (await q.evaluate((e) => e.textContent)).includes('Correct.') && (await q.evaluate((e) => e.textContent)).includes('9, 10, 11, 12 and 13'));
q = await question('own run time is three times longer');
const portable = async (i) => {
  await (await q.$$('input[type=radio]'))[i].click();
  await (await q.$('button.primary')).click();
  await pause(150);
  return (await q.evaluate((e) => e.textContent)).replace(/\s+/g, ' ');
};
let pt = await portable(0);
check('portability question: "slower hardware" is ruled out by the equal peaks', pt.includes('Not quite.') && pt.includes('the peaks are the same'), pt.slice(-300));
pt = await portable(1);
check("portability question: Amdahl is marked wrong, because the figure is the kernel's own time", pt.includes('Not quite.') && pt.includes("kernel's own run time"), pt.slice(-300));
pt = await portable(2);
await pause(600);
check("portability question: retuning is right, with AMD's documented example", pt.includes('Correct.') && pt.includes('half of the compute resources') && pt.includes('B5'), pt.slice(-400));
const last = puts.at(-1) ?? {};
check(
  'answers and the prediction are written to the progress file under their ids',
  last.answers?.['ch01/1.7/p-before20']?.value === 18 && last.answers?.['ch01/1.7/q-edition16']?.correct === false && last.answers?.['ch01/1.7/q-postpone']?.correct === true &&
    last.answers?.['ch01/1.7/q-sortneeds']?.correct === true && last.answers?.['ch01/1.7/q-portable']?.correct === true,
  JSON.stringify(last.answers),
);
check('first answers enter the review schedule', ['q-edition16', 'q-postpone', 'q-sortneeds', 'q-portable'].every((k) => last.review?.['ch01/1.7/' + k]));

// ---- the page as a whole ----
const body = (await page.$eval('.prose', (e) => e.textContent)).replace(/\s+/g, ' ');
check('what the 5th edition changed is a paragraph naming the four new chapters', (await text('p[data-role=changes]')).replace(/\s+/g, ' ').includes(`Four chapters are new: ${and(bm.NEW_IN_FIFTH)}.`) && !(await page.$('table[data-role=changes]')));
check('one explorer on the page, and the backends table has three rows', (await page.$$('section.bench')).length === 1 && (await page.$$('table[data-role=backends] tbody tr')).length === 3);
const pytorch = (await page.$eval('table[data-role=backends] tbody tr', (tr) => tr.textContent)).replace(/\s+/g, ' ');
check("PyTorch row: mostly HIPified CUDA plus AMD-only files; MPS on Apple's frameworks and its own kernels", pytorch.includes('HIPify') && pytorch.includes('AMD-only files') && pytorch.includes('MPSGraph') && pytorch.includes('.metal'), pytorch);
check("the arrows are the book's prerequisites; treating them as advice is marked as ours", body.includes('The Preface presents the arrows as prerequisites') && body.includes('Treating them as advice, which this course may overrule, is our reading'));
check('MPI is left out of this course, whose chapter 23 keeps streams, copies and pinned memory', body.includes('which this course leaves out (its chapter 23 keeps only streams'));
check('the emulator habit separates emu (fixed orders) from race (missing atomics)', body.includes('three fixed thread orders') && body.includes('a missing atomic, need the race backend') && body.includes('only for schedules that actually occur'), body.slice(body.indexOf('Correct functionality'), body.indexOf('Correct functionality') + 500));
check("Apple's sequence is the same ground without the two copies, linked to the summary", body.includes('without the two copies') && !!(await page.$('.prose a[href="#/ch01/wrap"]')));
check('the page itself does not scroll sideways', await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
