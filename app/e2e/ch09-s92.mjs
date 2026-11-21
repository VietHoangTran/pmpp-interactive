// Browser check for section 9.2: the "Race on one bin" stepper reproduces the book's
// tables and the lockstep warp, the atomic switch never loses an update, the census of
// every interleaving appears after a prediction, the basic kernel is loaded from lab
// 9.1, and the questions answer. Expected values come from src/sim/histrace.ts.
// Usage: APP_URL=http://localhost:5173 node e2e/ch09-s92.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { PRESETS, enumerate, run } from '../src/sim/histrace.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
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
const S = 'section.bench[aria-label="Race on one bin"]';
const click = async (label, scope = S) => {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelector(scope).querySelectorAll('button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(50);
  return ok;
};
const thread = async (t) => {
  await page.$eval(`${S} button[data-thread="${t}"]`, (b) => b.click());
  await pause(50);
};
const setAtomic = async (on) => {
  await page.evaluate((S, on) => {
    const box = [...document.querySelectorAll(S + ' label.check')].find((l) => l.textContent.includes('atomic')).querySelector('input');
    if (box.checked !== on) box.click();
  }, S, on);
  await pause(80);
};
const table = () => page.$$eval(S + ' table[data-part="race-table"] tbody tr', (rs) => rs.map((r) => [...r.querySelectorAll('td')].map((c) => c.textContent.trim())));
const final = () => page.$eval(S + ' [data-final]', (e) => ({ v: Number(e.dataset.final), ok: e.dataset.correct, text: e.textContent.replace(/\s+/g, ' ') })).catch(() => null);
/** Choose a preset and run it to the end with Step. */
async function play(label) {
  await click(label);
  for (let k = 0; k < 20 && (await click('Step')); k++);
  await pause(80);
  return final();
}

await goto(page, 'ch09/9.2');
await page.waitForSelector(S);
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));

// ---- the book's tables and the lockstep warp, two threads
const want = PRESETS.map((p) => run(p.schedule(2), { n: 2 }).final);
check('the model gives 2, 1, 2, 1 and 1 for the presets', JSON.stringify(want) === '[2,1,2,1,1]');
const got = [];
for (const p of PRESETS) got.push((await play(p.label))?.v);
check('playing the presets ends at 2, 1, 2, 1 and 1', JSON.stringify(got) === JSON.stringify(want), JSON.stringify(got));

let f = await play('thread 1 first');
let rows = await table();
check('thread 1 first: (0) (1) (1) then (1) (2) (2), as in Fig. 9.4(a)', rows.map((r) => r.slice(1, 3).join('').match(/\((\d)\)/)[1]).join('') === '011122', JSON.stringify(rows));
check('and it reports every increment counted', f.ok === 'yes' && f.text.includes('all 2 increments counted'), f.text);
f = await play('overlapped, thread 1 ahead');
rows = await table();
check('overlapped, thread 1 ahead: (0) (1) (0) (1) (1) (1), as in Fig. 9.4(b)', rows.map((r) => r.slice(1, 3).join('').match(/\((\d)\)/)[1]).join('') === '010111', JSON.stringify(rows));
check('the step at time 6 is marked as overwriting thread 1\'s update', rows[5][3].includes('overwrites thread 1\'s update'), rows[5][3]);
check('the closing line names the missing increment and who erased it', f.ok === 'no' && f.text.includes("thread 1's increment is missing: thread 2's store at time 6 wrote 1 over it"), f.text);
if (out) await page.screenshot({ path: out + '-overlap.png' });

// Back and Restart replay a preset.
await click('Back');
check('Back removes the last step', (await table()).length === 5);
await click('Restart');
check('Restart empties the table', (await page.$eval(S + ' table[data-part="race-table"]', (e) => e.textContent)).includes('Nothing has run yet'));

// ---- four threads in lockstep
await click('4');
f = await play('lockstep warp');
check('4 threads in lockstep end at 1, three increments lost', f.v === 1 && f.text.includes("thread 1's increment is missing") && f.text.includes("thread 3's increment is missing"), f.text);
check('the 4-thread lockstep table has 12 steps, every load reading 0', (await table()).length === 12 && (await table()).slice(0, 4).every((r) => r.join(' ').includes('(0) Old ← bins[x]')));
if (out) await page.screenshot({ path: out + '-lockstep.png' });

// ---- building an order by clicking (3 threads)
await click('3');
for (const t of [0, 1, 0, 0, 2, 2, 2, 1, 1]) await thread(t);
f = await final();
const mine = run([0, 1, 0, 0, 2, 2, 2, 1, 1], { n: 3 });
check('a clicked order of three threads ends where the model says: thread 2\'s late store erases two updates', f.v === mine.final && f.v === 1 && f.text.includes("thread 3's increment is missing: thread 2's store at time 9 wrote 1 over it"), JSON.stringify(f));
check('a finished thread\'s button is disabled', await page.$eval(`${S} button[data-thread="0"]`, (b) => b.disabled));

// ---- atomic: exclusion, not order
await setAtomic(true);
await click('4');
for (const t of [2, 0, 3, 1]) await thread(t);
f = await final();
check('atomic, clicked order 3, 1, 4, 2: the bin ends at 4', f.v === 4 && f.ok === 'yes' && f.text.includes('The order was 3, 1, 4, 2'), JSON.stringify(f));
check('atomic: each thread has a single step', (await table()).length === 4);
f = await play('lockstep warp');
check('atomic lockstep preset: 4 of 4', f.v === 4, JSON.stringify(f));
f = await play('overlapped, thread 2 ahead');
check('atomic overlapped preset: 2 of 2, thread 2 first', f.v === 2 && f.text.includes('The order was 2, 1'), JSON.stringify(f));
if (out) await page.screenshot({ path: out + '-atomic.png' });
await setAtomic(false);

// ---- predict, then the census
const pq = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Of the 20 interleavings')));
await (await pq.$('input[type=number]')).type('6');
await (await pq.$('button.primary')).click();
await page.waitForSelector('table[data-part="census"]');
const census = await page.$$eval('table[data-part="census"] tbody tr', (rs) => rs.map((r) => ({ n: r.dataset.row, total: r.dataset.total, correct: r.dataset.correct, text: r.textContent })));
const m = [2, 3, 4].map((n) => enumerate(n));
check('the census rows match enumerate(2..4)', census.every((r, i) => Number(r.total) === m[i].interleavings && Number(r.correct) === m[i].correct), JSON.stringify(census));
check('2 of 20 and 6 of 1,680 are shown', census[0].text.includes('2 of 20') && census[1].text.includes('6 of 1,680') && census[2].text.includes('24 of 369,600'), census.map((c) => c.text).join(' | '));
check('with atomics every order is correct', census[2].text.includes('24 of 24'));
check('the reveal answers a wrong prediction', (await pq.evaluate((e) => e.textContent)).includes('You predicted 6') && (await pq.evaluate((e) => e.textContent)).includes('both store 1'));
if (out) await page.$eval('table[data-part="census"]', (e) => e.scrollIntoView({ block: 'center' }));
if (out) await page.screenshot({ path: out + '-census.png' });

// ---- the listing from lab 9.1
await page.waitForFunction(() => [...document.querySelectorAll('.code pre')].some((p) => p.textContent.includes('histogramBasic')) || document.querySelector('[data-part="listing-missing"]'), { timeout: 8000 });
const missing = await page.$('[data-part="listing-missing"]');
if (missing) check('lab 9.1 solution is loaded (missing: the page says so)', false, 'labs/ch09/lab1_racehunt/solution/kernel.cu region basic not found');
else {
  const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent));
  check('the basic kernel is shown with CUDA and HIP and Metal tabs', tabs.includes('CUDA and HIP') && tabs.includes('Metal'), tabs.join('|'));
  const code = await page.$$eval('.code pre', (ps) => ps.map((p) => p.textContent).join('\n'));
  check('the CUDA region is Fig. 9.6 with const and the atomic reference', code.includes('const unsigned char* image') && code.includes('cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);') && code.includes('fetch_add(1, cuda::memory_order_relaxed)') && !code.includes('// region'));
  const status = await page.$eval('.code .tabs .status', (e) => e.textContent);
  // The same words as the labs page's kernel.cu tab (LabKit solutionTabStatus).
  check('its status uses the project vocabulary, in the labs page\'s words', /^(emulator-checked|verified on (CUDA|HIP)|unverified)/.test(status) && (/verified on CUDA/.test(status) || status.includes('unverified')), status);
  await page.evaluate(() => [...document.querySelectorAll('.code .tabs button')].find((b) => b.textContent === 'Metal').click());
  await pause(80);
  const metal = await page.$$eval('.code pre', (ps) => ps.map((p) => p.textContent).join('\n'));
  check('the Metal tab declares atomic_uint bins and passes the order', metal.includes('device atomic_uint*') && metal.includes('atomic_fetch_add_explicit(&bins[b], 1u, memory_order_relaxed)'));
  const mstatus = await page.$eval('.code .tabs .status', (e) => e.textContent);
  check('Metal status is "verified on Metal" or says it is not', mstatus === 'verified on Metal' || mstatus.startsWith('unverified'), mstatus);
}

// ---- backends table and notes
const backends = await page.$$eval('table[data-part="backends"] tbody tr', (rs) => rs.map((r) => r.textContent));
check('the backends table has four rows, Metal without a scope', backends.length === 4 && backends[3].includes('none on read-modify-write'), backends.join(' | '));
const todays = await page.$$eval('.note.today', (n) => n.map((x) => x.textContent));
check('Today notes cite A1, A3, A10, D6', ['A1', 'A3', 'A10', 'D6'].every((k) => todays.some((t) => t.includes(k))), todays.join(' | ').slice(0, 300));
check('the stepper states what it simplifies', await page.$$eval('.note.caveat', (n) => n.some((x) => x.textContent.includes('undefined behaviour'))));

// ---- questions
const question = (text) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt').textContent.includes(t)), text);
async function choose(text, option) {
  const q = await question(text);
  await q.evaluate((e, o) => [...e.querySelectorAll('.opts label')].find((l) => l.textContent.includes(o)).querySelector('input').click(), option);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
}
let t = await choose('Which final values are possible', '1 or 4');
check('range: "1 or 4" is wrong', t.includes('Not quite.') && t.includes('partial'), t.slice(0, 200));
t = await choose('Which final values are possible', 'Any of 1, 2, 3 and 4');
check('range: accepted', t.includes('Correct.'));
t = await choose('Which index does A get', 'possibly the same as B');
check('order: "possibly the same" is wrong, it would be a lost update', t.includes('Not quite.') && t.includes('lost update'), t.slice(0, 300));
t = await choose('Which index does A get', 'B gets the other one');
check('order: accepted, pointing to chapter 12', t.includes('Correct.') && t.includes('Chapter 12'));
t = await choose('memory_order_relaxed enough', 'default');
check('relaxed: "it is the default" is wrong (seq_cst)', t.includes('Not quite.') && t.includes('seq_cst'), t.slice(0, 300));
t = await choose('memory_order_relaxed enough', 'one other memory access');
check('relaxed: accepted', t.includes('Correct.'));
t = await choose('narrower scope is faster', 'data race');
check('scope: accepted, says no test here catches it', t.includes('Correct.') && t.includes('no test'), t.slice(0, 300));
t = await choose('Which run is certain to pass', 'race, which runs');
check('emu blind spot: race is not certain to pass, a report fails the test', t.includes('Not quite.') && t.includes('a report fails the test'), t.slice(0, 300));
t = await choose('Which run is certain to pass', 'emu, under each of its three');
check('emu blind spot: emu is certain to pass, and that says nothing about atomicity', t.includes('Correct.') && t.includes('nothing about atomicity') && t.includes('race'), t.slice(0, 300));
// Questions the next paragraphs answer come before them (predict, then read).
const order = await page.evaluate(() => {
  const kids = [...document.querySelector('.prose').children];
  const at = (f) => kids.findIndex(f);
  return {
    scopeQ: at((k) => k.classList.contains('q') && k.textContent.includes('narrower scope is faster')),
    relaxedQ: at((k) => k.classList.contains('q') && k.textContent.includes('memory_order_relaxed enough')),
    choicesP: at((k) => k.tagName === 'P' && k.textContent.includes('Four choices hide in those two lines')),
    emuQ: at((k) => k.classList.contains('q') && k.textContent.includes('Which run is certain to pass')),
    emuP: at((k) => k.tagName === 'P' && k.textContent.includes('it passes, because a thread is never interrupted')),
  };
});
check('q-scope and q-relaxed come before "Four choices", q-emu-blind before the paragraph that answers it',
  order.scopeQ >= 0 && order.relaxedQ >= 0 && Math.max(order.scopeQ, order.relaxedQ) < order.choicesP && order.emuQ >= 0 && order.emuQ < order.emuP, JSON.stringify(order));
const caption = await page.$eval('figure.code figcaption, figure.code .caption', (e) => e.textContent).catch(() => '');
check('the listing caption names three differences, the name first', missing ? true : caption.includes('three differences') && caption.includes('histogramBasic'), caption);

// ---- narrow, dark
await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const wide = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
check('no sideways scrolling at 400 px', wide <= 1, String(wide));
// The backends table keeps its columns and scrolls inside its own wrapper; no identifier is broken mid-word.
const bt = await page.$eval('table[data-part="backends"]', (t) => {
  const wrap = t.parentElement;
  const codes = [...t.querySelectorAll('code')];
  const broken = codes.filter((c) => {
    const s = getComputedStyle(c);
    return s.overflowWrap === 'anywhere' || s.wordBreak === 'break-all';
  }).length;
  const firstCol = t.querySelector('tbody td').getBoundingClientRect().width;
  return { scrolls: getComputedStyle(wrap).overflowX === 'auto', broken, firstCol, tableW: t.getBoundingClientRect().width, wrapW: wrap.getBoundingClientRect().width };
});
check('at 400 px the backends table scrolls in its wrapper, with no identifier broken mid-word', bt.scrolls && bt.broken === 0 && bt.tableW > bt.wrapW, JSON.stringify(bt));
if (out) {
  await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-dark.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
