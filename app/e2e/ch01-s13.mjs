// Browser check for section 1.3 ("Speed-up and what limits it"): the two predictions,
// the Amdahl explorer, the "what caps the kernel" explorer, the challenge classification
// and the questions. Every number asserted is computed from the tested model
// (src/sim/amdahl.ts), and the book's three worked cases are also pinned as literals.
// Usage: node e2e/ch01-s13.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

// The model the page runs on. Node strips the types itself; on a Node too old for that,
// take the same module as the dev server compiles it for the browser.
let m;
try {
  m = await import('../src/sim/amdahl.ts');
} catch {
  const code = await (await fetch(BASE + '/src/sim/amdahl.ts')).text();
  m = await import('data:text/javascript;charset=utf-8,' + encodeURIComponent(code));
}

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file, and start from an empty one so that the
// predictions are not already committed.
const puts = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
  if (r.method() === 'PUT') {
    puts.push(JSON.parse(r.postData()));
    return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  }
  return r.respond({ status: 200, contentType: 'application/json', body: '{"version":1}' });
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const question = (needle) =>
  page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const qText = (q) => q.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
const setRange = async (sel, value) => {
  await page.$eval(
    sel,
    (el, v) => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
      el.dispatchEvent(new Event('input', { bubbles: true }));
    },
    value,
  );
  await pause(40);
};
const clickText = async (scope, label) => {
  const done = await page.evaluate(
    (scope, label) => {
      const btn = [...document.querySelectorAll(scope + ' button')].find((b) => b.textContent.trim() === label);
      if (!btn || btn.disabled) return false;
      btn.click();
      return true;
    },
    scope,
    label,
  );
  await pause(60);
  return done;
};

// The page's own number formats (src/lib/reports.ts fmt, and the helpers at the top of S13.tsx).
const fmt = (v, d = 3) =>
  !isFinite(v) ? 'n/a' : v === 0 ? '0' : Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : Number(v.toPrecision(d)).toString();
const times = (v) => (isFinite(v) ? fmt(Number(v.toPrecision(6))) + '×' : 'unbounded');
const share = (v) => fmt(v * 100, 4) + '%';
const frac = (p) => Number((p * 100).toFixed(4)) + '%';

await goto(page, 'ch01/1.3');
check('section heading is shown', (await text('.page h1')).includes('Speed-up and what limits it'));
const A = bench("Amdahl's law");
const C = bench('What caps the kernel');
const W = bench('Which challenge is it?');
for (const b of [A, C, W]) await page.waitForSelector(b);
check('the three interactives are on the page', (await page.$$('section.bench')).length === 3);

// ------------------------------------------------------------ predictions
let q = await question('spends 30% of its run time');
check('first prediction hides its answer until committed', !(await qText(q)).includes(times(m.speedup(0.3, 100))));
await (await q.$('input[type=number]')).type('30');
await (await q.$('button.primary')).click();
await pause(150);
let t = await qText(q);
check(
  "committing 30 reveals the book's 1.42 and 1.43 and reacts to the guess",
  t.includes('You predicted 30') && t.includes('= 1.42×') && t.includes('did not move') && t.includes('no more than 1.43×'),
  t,
);
check('the revealed figures are the model\'s', t.includes(times(m.speedup(0.3, 100))) && t.includes(times(m.ceiling(0.3))) && t.includes(share(m.remaining(0.3, 100).total)), t);

q = await question('Now the part is 99% of the run time');
check('second prediction hides its answer until committed', !(await qText(q)).includes('50.3×'));
await (await q.$('input[type=number]')).type('99');
await (await q.$('button.primary')).click();
await pause(150);
t = await qText(q);
check(
  'committing 99 reveals 50.3 (the book: about 50) and reacts to the guess',
  t.includes('You predicted 99') && t.includes(times(m.speedup(0.99, 100))) && t.includes('50.3×') && t.includes('about 50') && t.includes('natural guess'),
  t,
);
await pause(600);
check(
  'both predictions are written to the progress file under their ids',
  puts.some((p) => p.answers?.['ch01/1.3/p-thirty']?.value === 30 && p.answers?.['ch01/1.3/p-ninetynine']?.value === 99),
);

// ------------------------------------------------------------ explorer 1: Amdahl's law
const aOut = (name) => text(`${A} [data-out="${name}"]`);
const P = `${A} input[aria-label="parallel fraction"]`;
const S = `${A} input[aria-label="speed-up of the accelerated part"]`;
const U = `${A} input[aria-label="unbounded speed-up"]`;
const pIndex = (p) => m.FRACTIONS.indexOf(p);
const sIndex = (s) => m.SPEEDUPS.indexOf(s);
const setUnbounded = async (on) => {
  if ((await page.$eval(U, (e) => e.checked)) !== on) await page.click(U);
  await pause(60);
};
const pressed = (scope) => page.$$eval(scope + ' button[aria-pressed="true"]', (b) => b.map((x) => x.textContent.trim()));

check('the explorer starts on the case just predicted: 99% and 100×', (await aOut('p')) === '99%' && (await aOut('s')) === '100×' && (await aOut('overall')) === '50.3×');
check('and marks that preset as the current one', (await pressed(`${A} [data-presets]`)).join() === '99%, 100×');

const BOOK = ['1.42×', '1.43×', '50.3×']; // p. 9: about 1.42, no more than 1.43, about 50
for (const [i, c] of m.BOOK_CASES.entries()) {
  await clickText(A, c.label);
  const got = await aOut('overall');
  check(`preset "${c.label}" gives ${BOOK[i]}, as the model does`, got === BOOK[i] && got === times(m.speedup(c.p, c.s)), got);
  check(
    `preset "${c.label}" sets both controls`,
    (await aOut('p')) === frac(c.p) && (await aOut('s')) === times(c.s) && (await page.$eval(U, (e) => e.checked)) === !isFinite(c.s),
  );
}

await clickText(A, '30%, 100×');
check('30% and 100×: ceiling 1.43×, 70.3% of the time left, 99.57% of it untouched', (await aOut('ceiling')) === '1.43×' && (await aOut('left')) === '70.3%' && (await aOut('share')) === '99.57%');
check('30%: no s reaches 100× overall', (await aOut('needed')) === 'none is enough');
const widths = () =>
  page.$$eval(`${A} [data-bar]`, (bars) =>
    Object.fromEntries(bars.map((b) => [b.dataset.bar, Object.fromEntries([...b.children].map((c) => [c.dataset.part, parseFloat(c.style.width)]))])),
  );
let w = await widths();
const near = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol;
check('time bars: the untouched part has the same length before and after', near(w.before.untouched, 70, 1e-3) && w.after.untouched === w.before.untouched, JSON.stringify(w));
check('time bars: the accelerated part shrinks from 30% to 0.3%', near(w.before.accelerated, 30, 1e-3) && near(w.after.accelerated, 0.3, 1e-3), JSON.stringify(w));
check('time bars: the magnified bar shows the untouched share of what is left', near(w.zoom.untouched, 100 * m.untouchedShare(0.3, 100), 1e-3), JSON.stringify(w.zoom));
if (out) await page.screenshot({ path: out + '-amdahl-30.png' });

// The ceiling belongs to p alone; s only moves the point towards it.
await setRange(P, pIndex(0.99));
let last = 0;
let rising = true;
let agree = true;
let ceilings = new Set();
for (const s of [1, 2, 10, 50, 100, 500, 1000]) {
  await setRange(S, sIndex(s));
  const got = await aOut('overall');
  if (got !== times(m.speedup(0.99, s))) agree = false;
  const v = m.speedup(0.99, s);
  if (v <= last) rising = false;
  last = v;
  ceilings.add(await aOut('ceiling'));
}
check('at 99% the overall figure follows the model for every s tried and keeps rising', agree && rising);
check('while the ceiling stays at 100× whatever s is', ceilings.size === 1 && ceilings.has('100×'), [...ceilings].join());
check('at 99%, s = 1,000 only reaches about 91×', (await aOut('overall')) === '91×' && times(m.speedup(0.99, 1000)) === '91×');

// A grid of other settings against the model.
let bad = [];
for (let i = 0; i < m.FRACTIONS.length; i += 6)
  for (let j = 0; j < m.SPEEDUPS.length; j += 5) {
    await setRange(P, i);
    await setRange(S, j);
    const want = times(m.speedup(m.FRACTIONS[i], m.SPEEDUPS[j]));
    const got = await aOut('overall');
    if (got !== want) bad.push(`p=${m.FRACTIONS[i]} s=${m.SPEEDUPS[j]}: ${got} != ${want}`);
  }
check('a grid of 77 settings of p and s all show the model\'s overall speed-up', bad.length === 0, bad.slice(0, 3).join('; '));

// The task: the smallest fraction at which 100x is reachable at all.
await setUnbounded(true);
check('with s unbounded the slider for s is disabled', await page.$eval(S, (e) => e.disabled));
let first = null;
for (let i = 0; i < m.FRACTIONS.length && first === null; i++) {
  await setRange(P, i);
  if ((await aOut('needed')) !== 'none is enough') first = m.FRACTIONS[i];
}
check('the task: 99% is the smallest fraction at which 100× can be reached', first === 0.99 && (await aOut('overall')) === '100×' && (await aOut('needed')) === 'unbounded', String(first));
const cy = (sel) => page.$eval(`${A} svg ${sel}`, (e) => e.getAttribute('cy'));
check('on the plot the point sits on the ceiling when s is unbounded', (await cy('[data-point]')) === (await cy('[data-limit]')));
await setRange(P, pIndex(0.99) - 1);
check('one stop lower (98.8%) the ceiling is 83.3× and nothing reaches 100×', (await aOut('p')) === '98.8%' && (await aOut('overall')) === times(m.ceiling(0.988)) && (await aOut('ceiling')) === '83.3×' && (await aOut('needed')) === 'none is enough');
await setRange(P, pIndex(0.999));
check('at 99.9% the ceiling is 1,000×', (await aOut('overall')) === '1,000×');
await setUnbounded(false);
await setRange(S, sIndex(1000));
check('at 99.9% a finite s for 100× exists and the readout gives the model\'s value', (await aOut('needed')) === times(m.kernelNeeded(100, 0.999)), await aOut('needed'));
check('at 99.9% and 1,000× the explorer shows 500×, the case of q-fivehundred', (await aOut('overall')) === '500×' && times(m.speedup(0.999, 1000)) === '500×', await aOut('overall'));
if (out) await page.screenshot({ path: out + '-amdahl-999.png' });

// ------------------------------------------------------------ explorer 2: what caps the kernel
const cOut = (name) => text(`${C} [data-out="${name}"]`);
const N = `${C} input[aria-label="threads"]`;
const M = `${C} input[aria-label="memory cap"]`;
const CP = `${C} input[aria-label="parallel fraction"]`;
const nIndex = (n) => m.THREADS.indexOf(n);
const lever = (id) => text(`${C} tr[data-lever="${id}"]`);
const goals = () => page.$$eval(`${C} [data-goal]`, (g) => g.map((x) => x.textContent.trim()));

check('the parallel fraction is shared: this explorer shows the 99.9% set in the one above', (await cOut('p')) === '99.9%');
check('"Back to the defaults" is offered', await clickText(C, 'Back to the defaults'));
let d = m.limits(0.99, 1000, 0.25, 10);
check('defaults: 1,000 threads, r = 1/4, M = 10×, p = 99%', (await cOut('n')) === '1,000' && (await cOut('M')) === '10×' && (await cOut('p')) === '99%' && (await pressed(`${C} [data-rates]`)).join() === '1/4');
check('and resetting moved the fraction in the first explorer too', (await aOut('p')) === '99%');
check('defaults: kernel 10×, set by memory', (await cOut('kernel')) === 'min(250, 10) = 10×, set by memory' && d.limitedBy === 'memory' && d.s === 10, await cOut('kernel'));
check('defaults: overall 9.17×', (await cOut('overall')) === '9.17×' && times(d.overall) === '9.17×');
check('the memory cap slider carries the book\'s rule-of-thumb tick', (await text(`${C} [data-tick="book"]`)).includes("the book's rule of thumb, p. 10"));
let detail = await cOut('detail');
check('defaults: break-even at 4 threads, memory saturates at 40, ceiling 100×', detail.includes('Break-even at 4 threads') && detail.includes('memory saturates at 40') && detail.includes('untouched part: 100×'), detail);
const corner = m.corners(0.25, 10);
check('those corners are the model\'s', corner.breakEven === 4 && corner.saturation === 40);

let same = true;
for (const n of [40, 100, 1000, 10000, 100000]) {
  await setRange(N, nIndex(n));
  if ((await cOut('overall')) !== '9.17×') same = false;
}
check('past saturation more threads buy nothing: 9.17× from 40 threads to 100,000', same);
await setRange(N, nIndex(20));
check('below saturation the thread count is the limit', (await cOut('kernel')).includes('set by the thread count') && (await cOut('overall')) === times(m.limits(0.99, 20, 0.25, 10).overall));
await setRange(N, nIndex(1000));

const lv = m.levers(0.99, 1000, 0.25, 10);
let rows = [await lever('threads'), await lever('memory'), await lever('untouched')];
check(
  'levers at the defaults: threads change nothing, the memory cap gives 50.3×, the untouched part 9.91×',
  rows[0].includes('9.17×') && rows[0].includes('no change') && rows[1].includes('50.3×') && rows[2].includes('9.91×') &&
    rows[1].includes(times(lv.memory)) && rows[2].includes(times(lv.untouched)) && lv.threads === lv.now,
  rows.join(' | '),
);
check('neither goal is reached at the defaults', (await goals()).join('|') === 'past 50×: not yet|past 100×: not yet', (await goals()).join('|'));
if (out) await page.screenshot({ path: out + '-caps-default.png' });

await setRange(M, sIndex(100));
check('raising the cap to 100 gives 50.3× and passes the first goal', (await cOut('overall')) === '50.3×' && times(m.limits(0.99, 1000, 0.25, 100).overall) === '50.3×' && (await goals())[0] === 'past 50×: reached' && (await goals())[1] === 'past 100×: not yet');
await setRange(M, sIndex(1000));
d = m.limits(0.99, 1000, 0.25, 1000);
check('raising it to 1,000 makes the threads the limit: 250× on the kernel, 71.6× overall', (await cOut('kernel')) === 'min(250, 1,000) = 250×, set by the thread count' && (await cOut('overall')) === '71.6×' && d.limitedBy === 'threads' && times(d.overall) === '71.6×', await cOut('kernel'));
await setRange(N, nIndex(100000));
await clickText(`${C} [data-rates]`, '1');
check('at 99% no number of threads and no cap passes 100×', (await cOut('overall')) === times(m.limits(0.99, 100000, 1, 1000).overall) && m.limits(0.99, 100000, 1, 1000).overall < 100 && (await goals())[1] === 'past 100×: not yet', await cOut('overall'));
await setRange(CP, pIndex(0.999));
check('only a larger parallel fraction does: 99.9% passes the second goal', (await cOut('overall')) === times(m.limits(0.999, 100000, 1, 1000).overall) && (await goals())[1] === 'past 100×: reached', await cOut('overall'));
check('and the first explorer follows the shared fraction', (await aOut('p')) === '99.9%');
if (out) await page.screenshot({ path: out + '-caps-past100.png' });

await clickText(`${C} [data-rates]`, '1/8');
await setRange(N, 0);
d = m.limits(0.999, 1, 0.125, 1000);
check('one slow thread is slower than the CPU core, and the page says so', (await cOut('overall')) === times(d.overall) && d.overall < 1 && (await cOut('slower')).includes('slower than the CPU core'));
check('the hatched stretch of the plot ends at break-even, 8 threads', (await cOut('detail')).includes('Break-even at 8 threads') && (await page.$(`${C} svg [data-region="slower"]`)) !== null);
const geom = await page.evaluate((sel) => {
  const g = (q) => document.querySelector(sel + ' svg ' + q);
  return { region: Number(g('[data-region="memory"]').getAttribute('x')), corner: Number(g('[data-corner="saturation"] circle').getAttribute('cx')) };
}, C);
check('the shaded stretch starts at the saturation corner', near(geom.region, geom.corner, 0.01), JSON.stringify(geom));

bad = [];
const rateLabel = { 0.125: '1/8', 0.25: '1/4', 0.5: '1/2', 1: '1' };
for (const [p, n, r, cap] of [[0.9, 8, 0.5, 2], [0.96, 300, 0.125, 50], [0.99, 5000, 0.25, 250], [0.9995, 60000, 1, 600], [0.3, 100, 1, 1000], [0.999, 2, 0.25, 1]]) {
  await setRange(CP, pIndex(p));
  await setRange(N, nIndex(n));
  await setRange(M, sIndex(cap));
  await clickText(`${C} [data-rates]`, rateLabel[r]);
  const want = m.limits(p, n, r, cap);
  const got = [await cOut('overall'), await cOut('kernel')];
  if (got[0] !== times(want.overall) || !got[1].includes(want.limitedBy === 'memory' ? 'set by memory' : 'set by the thread count'))
    bad.push(`${p},${n},${r},${cap}: ${got.join(' / ')} != ${times(want.overall)} by ${want.limitedBy}`);
}
check('six other settings of p, n, r and M all show the model\'s result and limiter', bad.length === 0, bad.join('; '));
await clickText(C, 'Back to the defaults');
check('back to the defaults restores 9.17×', (await cOut('overall')) === '9.17×');

// ------------------------------------------------------------ classification
const ANSWERS = { topp: 'work', matvec: 'memory', lengths: 'input', counters: 'sync', worksum: 'work', sparse: 'input', barrier: 'sync', bias: 'memory' };
const card = (id) => `${W} [data-card="${id}"]`;
const pick = async (id, choice) => {
  await page.click(`${card(id)} button[data-choice="${choice}"]`);
  await pause(50);
};
check('eight situations, none placed', (await page.$$(`${W} [data-card]`)).length === 8 && (await text(`${W} [data-which-score]`)) === '0 of 8 placed');
await pick('bias', 'sync');
let fb = await text(`${W} [data-feedback="bias"]`);
check('a wrong choice gets a reason at once and does not place the card', fb.includes('not this one') && fb.includes('embarrassingly parallel') && (await page.$eval(card('bias'), (e) => e.dataset.solved)) === 'false', fb);
await pick('bias', 'memory');
fb = await text(`${W} [data-feedback="bias"]`);
check('the right choice places it and names where the course deals with it', fb.includes('placed') && fb.includes('chapter 5, §5.1') && (await page.$eval(card('bias'), (e) => e.dataset.solved)) === 'true', fb);
check('a placed card locks its other choices', (await page.$$eval(`${card('bias')} button[data-choice]`, (b) => b.filter((x) => x.disabled).length)) === 3);
check('the feedback links to that chapter', (await page.$eval(`${W} [data-feedback="bias"] a`, (a) => a.getAttribute('href'))) === '#/ch05/5.1');
for (const [id, choice] of Object.entries(ANSWERS)) if (id !== 'bias') await pick(id, choice);
const solved = await page.$$eval(`${W} [data-card]`, (c) => c.filter((x) => x.dataset.solved === 'true').length);
check('all eight can be placed: two to each of the four challenges', solved === 8 && (await text(`${W} [data-which-score]`)) === '8 of 8 placed');
check('the closing remark appears once all are placed', (await text(`${W} [data-which-done]`)).includes('easy on three counts and hard on the fourth'));
if (out) await page.screenshot({ path: out + '-which-done.png' });
await clickText(W, 'Start again');
check('"Start again" clears the board', (await text(`${W} [data-which-score]`)) === '0 of 8 placed' && (await page.$(`${W} [data-which-done]`)) === null);

// ------------------------------------------------------------ the table and the notes
const table = await text('table[data-challenges]');
check('the four challenges are tabulated with the course\'s own modules', (await page.$$('table[data-challenges] tbody tr')).length === 4 && table.includes('X1 sampling') && table.includes('X2 quantised kernels') && table.includes('work efficiency'));
const notes = await page.$$eval('.prose > aside.note', (n) => n.map((x) => x.textContent.replace(/\s+/g, ' ')));
const prose = await text('.prose');
check("speed-up has the book's direction: A over B is B's time over A's, and below 1 means A is slower",
  prose.includes('The speed-up of system A over system B is the time B takes divided by the time A takes') && prose.includes('below 1 means A is the slower one'));
check("the remark about a model and a measurement disagreeing links to chapter 5's case", prose.includes('When a measurement lands above the roof') && !!(await page.$('.prose a[href="#/ch05/5.1"]')) && !prose.includes('On the machine this course was built on'));
check('the cap is the only change that moves the figure much, not the only one that moves it', prose.includes('Raising the cap is the only change that moves it much') && !prose.includes('Only raising the cap moves it'));
check('the formula is labelled as ours', notes.some((n) => n.includes('Our formula') && n.includes('The book prints no formula')));
check('both models state what they leave out', notes.filter((n) => n.includes('What this model leaves out')).length === 2);
check('the dated claims carry a "Today" note with the check date', notes.filter((n) => n.startsWith('Today') && n.includes('Checked 2026-11-20')).length === 2);

// ------------------------------------------------------------ questions
const typeAnswer = async (needle, value) => {
  const h = await question(needle);
  await (await h.$('input[type=number]')).type(String(value));
  await (await h.$('button.primary')).click();
  await pause(120);
  return qText(h);
};
t = await typeAnswer('gives about 500× overall', 99);
check("q-fivehundred: the explorer task's 99% is not the answer here; 99.9 is", t.includes('Not quite. The answer is 99.9 %'), t);
{
  const h = await question('gives about 500× overall');
  const input = await h.$('input[type=number]');
  await input.focus();
  await input.evaluate((e) => e.select());
  await input.type('99.9');
  await (await h.$('button.primary')).click();
  await pause(120);
  t = await qText(h);
}
check('q-fivehundred accepts 99.9 and explains it from the model', t.includes('Correct.') && t.includes(frac(m.fractionNeeded(500, 1000))) && t.includes('it shows 500×'), t.slice(-500));
check("q-fivehundred is the model's answer, and differs from the task's", Number((100 * m.fractionNeeded(500, 1000)).toFixed(1)) === 99.9 && m.fractionNeeded(100, Infinity) === 0.99);
t = await typeAnswer('kernels take 96% of the time', 20);
check('q-kernelneeded marks 20 wrong and gives 96', t.includes('Not quite. The answer is 96 ×') && Math.abs(m.kernelNeeded(20, 0.96) - 96) < 1e-6 && m.kernelNeeded(20, 0.95) === Infinity, t);
check('q-hostshare accepts 12.5', (await typeAnswer('take 8% of the time', 12.5)).includes('Correct.') && Math.abs(m.ceiling(0.92) - 12.5) < 1e-9);
q = await question('You have time for one of two jobs');
await (await q.$$('input[type=radio]'))[1].click();
await (await q.$('button.primary')).click();
await pause(120);
t = await qText(q);
check('q-bigslice: option B is marked wrong, with the reason', t.includes('Not quite.') && t.includes('caps the whole at 1.43×'), t);
await (await q.$$('input[type=radio]'))[0].click();
await (await q.$('button.primary')).click();
await pause(120);
t = await qText(q);
check('q-bigslice: option A is right, 2.17× against 1.43×', t.includes('Correct.') && t.includes(`= ${fmt(m.speedup(0.6, 10))}×`) && t.includes('= 2.17×') && t.includes('= 1.43×'), t);
q = await question('already keeps memory saturated');
await (await q.$$('input[type=radio]'))[1].click();
await (await q.$('button.primary')).click();
await pause(700);
check('q-saturated: "almost nothing" is right', (await qText(q)).includes('Correct.'));
const lastPut = puts[puts.length - 1] ?? {};
check(
  'answers are written to the progress file with their ids',
  lastPut.answers?.['ch01/1.3/q-fivehundred']?.correct === true &&
    lastPut.answers?.['ch01/1.3/q-kernelneeded']?.correct === false &&
    lastPut.answers?.['ch01/1.3/q-hostshare']?.correct === true &&
    lastPut.answers?.['ch01/1.3/q-bigslice']?.correct === true &&
    lastPut.answers?.['ch01/1.3/q-saturated']?.correct === true,
  JSON.stringify(Object.keys(lastPut.answers ?? {})),
);
check('first answers enter the review schedule', ['q-fivehundred', 'q-kernelneeded', 'q-hostshare', 'q-bigslice', 'q-saturated'].every((k) => lastPut.review?.['ch01/1.3/' + k]));

// ------------------------------------------------------------ narrow screen
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
const narrow = await page.evaluate(() => ({
  scroll: document.documentElement.scrollWidth,
  client: document.documentElement.clientWidth,
  plots: [...document.querySelectorAll('section.bench svg')].map((s) => s.getBoundingClientRect().width),
}));
check('at 390 px wide the page does not scroll sideways', narrow.scroll <= narrow.client, JSON.stringify(narrow));
check('and both plots are redrawn to fit', narrow.plots.length === 2 && narrow.plots.every((x) => x <= 360 && x >= 260), JSON.stringify(narrow.plots));
if (out) await page.screenshot({ path: out + '-narrow.png' });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
