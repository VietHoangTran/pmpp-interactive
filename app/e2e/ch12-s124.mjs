// Browser check for section 12.4 (privatisation): the lab listing, the stepper "Step through
// the privatised filter", the outcome table "Every order at once", and two questions.
// Expected values are the ones src/sim/privfilter.test.ts pins (and checks against the
// emulator); that model imports another module without a file extension, so it is not
// imported here.
// Usage: APP_URL=http://localhost:5173 node e2e/ch12-s124.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';

const out = process.argv[2] && !process.argv[2].startsWith('--') ? process.argv[2] : null;
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000, dark });
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
const S = bench('Step through the privatised filter');
const T = bench('Every order at once');
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const narration = () => text(S + ' .narrate');
const clickIn = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const btn = [...document.querySelectorAll(scope + ' button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(60);
  return done;
};
const select = async (label, value) => {
  await page.evaluate((S, label, value) => {
    const s = document.querySelector(`${S} select[aria-label="${label}"]`);
    s.value = value;
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, S, label, value);
  await pause(120);
};
const setBox = async (word, on) => {
  await page.evaluate((S, word, on) => {
    const b = [...document.querySelectorAll(S + ' label.check')].find((l) => l.textContent.trim() === word).querySelector('input');
    if (b.checked !== on) b.click();
  }, S, word, on);
  await pause(120);
};
const toEnd = async () => {
  for (let i = 0; i < 40; i++) if (!(await clickIn(S, 'Run to next barrier'))) break;
};
const publicKeys = () => page.$$eval(S + ' [data-public]', (cs) => cs.map((c) => c.dataset.value).filter((v) => v !== ''));
const outcome = () => page.$eval(S + ' [data-outcome]', (e) => e.dataset.outcome).catch(() => null);

await goto(page, 'ch12/12.4');
await page.waitForSelector(S);
await page.waitForSelector('.code .tabs button');

// ---------------------------------------------------------------- the lab listing
const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent));
check('listing shows CUDA and Metal kernel tabs', tabs.includes('CUDA') && tabs.includes('Metal kernel'), tabs.join('|'));
const listing = await text('figure.code pre');
check('the CUDA tab is lab 12.1\'s privatised kernel', listing.includes('filterPrivatisedKernel') && listing.includes('thread_scope_block') && !listing.includes('filterAtomicKernel'), listing.slice(0, 120));

// ---------------------------------------------------------------- the stepper
check('opens on the book\'s keys and block order', (await narration()).includes('8 of the 16 keys pass cond()') && (await narration()).includes('order 1, 3, 2, 0'), await narration());
await clickIn(S, 'Step');
let n = await narration();
check('step 1: block 1, thread 0 zeroes the private counter', n.startsWith('Block 1, thread 0 sets the private counter outputSize_s to 0'), n);
check('the listing marks the zeroing line', (await text(S + ' .code .ln.hl')).includes('outputSize_s = 0'));
await clickIn(S, 'Run to next barrier');
n = await narration();
check('run to next barrier ends on barrier 1 opening', n.startsWith('Barrier 1 opens'), n);
await clickIn(S, 'Run to next barrier');
check('after barrier 2, block 1\'s list holds k4 and k6', JSON.stringify(await page.$$eval(S + ' [data-shared-block="1"] [data-slot]', (c) => c.map((x) => x.dataset.value))) === JSON.stringify(['k4', 'k6', 'CD', 'CD']));
await toEnd();
check('correct kernel: the outcome is correct', (await outcome()) === 'correct' && (await text(S)).includes('Correct: 8 keys'));
check('Fig. 12.5\'s block order gives chunks k4 k6 | k14 | k8 k10 k11 | k1 k3', JSON.stringify(await publicKeys()) === JSON.stringify(['k4', 'k6', 'k14', 'k8', 'k10', 'k11', 'k1', 'k3']), JSON.stringify(await publicKeys()));
check('no failure to jump to', !(await text(S)).includes('Go to the first failure'));
if (out) await page.screenshot({ path: out + '-stepper-end.png' });

await select('Block order', 'in-order');
await toEnd();
check('in the emulator\'s block order: k1 k3 k4 k6 k8 k10 k11 k14', JSON.stringify(await publicKeys()) === JSON.stringify(['k1', 'k3', 'k4', 'k6', 'k8', 'k10', 'k11', 'k14']), JSON.stringify(await publicKeys()));
await select('Block order', 'book');
await select('Thread order', 'reverse');
await toEnd();
check('reversed threads in the book\'s block order: k6 k4 | k14 | k11 k10 k8 | k3 k1', JSON.stringify(await publicKeys()) === JSON.stringify(['k6', 'k4', 'k14', 'k11', 'k10', 'k8', 'k3', 'k1']), JSON.stringify(await publicKeys()));

// Barrier 3 removed: reversed fails at once, in order passes.
await select('Block order', 'in-order');
await setBox('barrier 3', false);
check('the listing shows where barrier 3 was', (await text(S + ' figure.code pre')).includes('// barrier 3 removed'));
check('without barrier 3, reversed: a failure to jump to', await clickIn(S, 'Go to the first failure'));
n = await narration();
check('the failure: thread 1, the first with a key to copy, reads blockStart before it is written and stores out of range', n.startsWith('Block 0, thread 1 copies') && n.includes('blockStart has not been written yet') && n.includes('AddressSanitizer'), n);
await toEnd();
check('and the run stops there', (await outcome()) === 'crash' && (await text(S)).includes('Stopped:'));
if (out) await page.screenshot({ path: out + '-stepper-crash.png' });
await select('Thread order', 'forward');
await toEnd();
check('without barrier 3, in order: correct, nothing to jump to', (await outcome()) === 'correct' && !(await text(S)).includes('Go to the first failure'));

// Barrier 2 removed, in order: a short reservation, then a wrong result.
await setBox('barrier 3', true);
await setBox('barrier 2', false);
check('without barrier 2, in order: a failure to jump to', await clickIn(S, 'Go to the first failure'));
n = await narration();
check('the failure: block 0\'s thread 0 reserves 0 slots for 2 kept keys', n.includes('reserves outputSize_s = 0 public slots') && n.includes('block 0 keeps 2'), n);
await toEnd();
check('and the result is wrong: size 2, 4 keys lost', (await outcome()) === 'wrong' && (await text(S)).includes('size 2 (not 8), 4 keys lost'), await text(S + ' [data-outcome]'));
check('slots written past the public counter are marked', (await page.$$(S + ' [data-public].bad')).length === 2);
await setBox('barrier 2', true);

// The plain counter: right one thread at a time, flagged all the same.
await setBox('private counter is atomic', false);
await clickIn(S, 'Run to next barrier');
for (let i = 0; i < 8 && !(await narration()).includes('increments'); i++) await clickIn(S, 'Step');
n = await narration();
check('a plain increment is flagged as a race, not as a failure', n.includes('reads and increments') && n.includes('race check') && !n.includes('Failure'), n);
await toEnd();
check('and the run is correct one thread at a time', (await outcome()) === 'correct');
await setBox('private counter is atomic', true);

// Keys are editable.
await page.$eval(S + ' [data-key="0"]', (e) => e.click());
await pause(120);
check('clicking k0 keeps it: 9 keys pass', (await narration()).includes('9 of the 16 keys pass'), await narration());
await toEnd();
check('and the run is still correct with 9 keys', (await outcome()) === 'correct' && (await text(S)).includes('Correct: 9 keys'));
await clickIn(S, 'Reset');

// ---------------------------------------------------------------- the outcome table
const rows = await page.$$eval(T + ' tbody tr', (rs) => rs.map((r) => ({
  v: r.dataset.variant,
  cells: [...r.querySelectorAll('td[data-outcome]')].map((c) => c.dataset.outcome),
  race: r.querySelector('td[data-race]').dataset.race,
})));
const want = {
  correct: [['correct', 'correct', 'correct'], 'none'],
  'no-barrier-1': [['correct', 'crash', 'crash'], 'reported'],
  'no-barrier-2': [['wrong', 'correct', 'wrong'], 'reported'],
  'no-barrier-3': [['correct', 'crash', 'crash'], 'reported'],
  'plain-counter': [['correct', 'correct', 'correct'], 'reported'],
};
check('five rows, in order', JSON.stringify(rows.map((r) => r.v)) === JSON.stringify(Object.keys(want)));
for (const r of rows)
  check(`table row ${r.v}: ${want[r.v][0].join(', ')}; race ${want[r.v][1]}`, JSON.stringify(r.cells) === JSON.stringify(want[r.v][0]) && r.race === want[r.v][1], JSON.stringify(r));
check('the shuffled cell of barrier 2 says size 6, 1 key lost', (await text(T + ' tr[data-variant="no-barrier-2"] td:nth-child(4)')).includes('size 6 (not 8), 1 key lost'));
if (out) {
  await page.$eval(T, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-table.png' });
}

// ---------------------------------------------------------------- questions
const question = (needle) => page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), needle);
const qp = await question('How many atomics does');
await (await qp.$('input[type=number]')).type('4096');
await (await qp.$('button.primary')).click();
await pause(150);
check('q-public accepts 4096', (await qp.evaluate((e) => e.textContent)).includes('Correct.'));
const q3 = await question('Remove only the barrier after thread 0 writes');
const pick3 = async (label) => {
  await q3.evaluate((e, label) => [...e.querySelectorAll('label')].find((l) => l.textContent.trim().startsWith(label)).querySelector('input').click(), label);
  await (await q3.$('button.primary')).click();
  await pause(150);
  return q3.evaluate((e) => e.textContent);
};
const only = await pick3('Reversed only.');
check('q-barrier3: "Reversed only" is wrong, because the shuffled order exposes it too', only.includes('Not quite.') && only.includes('so does the shuffled order'));
// The table above shows the same: stopped under reversed and shuffled, correct in order.
const right3 = await pick3('Reversed and shuffled; in order passes.');
check('q-barrier3: "reversed and shuffled; in order passes" is right, and the explanation names the shuffled order', right3.includes('Correct.') && right3.includes('1, 3, 2, 0'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
