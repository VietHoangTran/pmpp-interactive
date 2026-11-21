// Browser check for section 12.3: the three listings, the prediction, the warp vote visualiser
// (every state compared with src/sim/warpvote.ts, imported directly), "Which line is which?"
// and three questions. The scene has its own check, ch12-scene-one-vote.mjs.
// Usage: node e2e/ch12-s123.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { checkLabStatuses } from './labstatus.mjs';
import { LINE_PAIRS, halves, hex, passFrom, preset, vote } from '../src/sim/warpvote.ts';

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
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const V = 'Warp vote visualiser';
const W = 'Which line is which?';
const press = async (scope, label, within = 'button') => {
  const done = await page.evaluate((sel, label, within) => {
    const btn = [...document.querySelectorAll(sel + ' ' + within)].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, bench(scope), label, within);
  await pause(90);
  return done;
};
const setForm = async (form) => {
  await page.$eval(`${bench(V)} select[aria-label="form"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, v);
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, form);
  await pause(90);
};
const setWidth = (w) => press(V, w === 8 ? '8 (toy)' : String(w), '[aria-label="warp width"] button');
const setCounter = async (v) => {
  await page.$eval(`${bench(V)} input[aria-label="counter before"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(90);
};
const box = async (word, on) => {
  await page.evaluate((sel, word, on) => {
    const i = [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes(word)).querySelector('input');
    if (i.checked !== on) i.click();
  }, bench(V), word, on);
  await pause(90);
};
const clickLanes = async (lanes) => {
  for (const l of lanes) await page.$eval(`${bench(V)} button[data-lane="${l}"]`, (b) => b.click());
  await pause(90);
};
const state = () => page.evaluate((sel) => {
  const root = document.querySelector(sel);
  const n = root.querySelector('[data-correct]');
  return {
    groups: [...root.querySelectorAll('[data-group]')].map((g) => [g.dataset.mask, Number(g.dataset.leader), Number(g.dataset.count), g.dataset.atomic, g.dataset.start]),
    slots: [...root.querySelectorAll('button[data-lane]')].map((b) => b.dataset.slot),
    pass: [...root.querySelectorAll('button[data-lane]')].map((b) => b.dataset.pass === '1'),
    correct: n.dataset.correct === '1',
    atomics: Number(n.dataset.atomics),
    flags: n.dataset.flags.split(' ').filter(Boolean).sort(),
    after: Number(n.dataset.after),
    compile: !!root.querySelector('[data-compile-error]'),
    text: n.textContent,
  };
}, bench(V));
/** What the page should show for this input, from the model. */
const expected = (input) => {
  const m = vote(input);
  return {
    groups: m.groups.map((g) => [hex(g.mask, g.maskBits), g.leader, g.count, g.atomic ? `${g.atomic.before}>${g.atomic.after}` : 'none', g.start == null ? 'undefined' : String(g.start)]),
    slots: m.lanes.map((L) => (L.writes ? (L.slot == null ? '?' : String(L.slot)) : '')),
    correct: m.correct,
    atomics: m.atomics,
    flags: [...m.flags].sort(),
    after: m.counterAfter,
    m,
  };
};
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const matches = async (name, input, extra = () => true) => {
  const s = await state();
  const e = expected(input);
  const ok = same(s.groups, e.groups) && same(s.slots, e.slots) && s.correct === e.correct && s.atomics === e.atomics && same(s.flags, e.flags) && s.after === e.after;
  check(name, ok && extra(s, e), JSON.stringify({ page: { ...s, text: undefined }, model: { ...e, m: undefined } }));
  return s;
};

await goto(page, 'ch12/12.3');
await page.waitForSelector(bench(V));

// ---- listings
await page.waitForSelector('.code .tabs button', { timeout: 8000 }).catch(() => null);
const listings = await page.$$eval('figure.code', (fs) => fs.map((f) => ({
  tabs: [...f.querySelectorAll('.tabs button')].map((b) => b.textContent),
  lines: [...f.querySelectorAll('pre .ln')].map((l) => l.textContent.replace(/\n$/, '')),
  status: f.querySelector('.tabs .status')?.textContent ?? '',
})));
const fig3 = listings.find((l) => l.tabs.includes('Fig. 12.3'));
const fig4 = listings.find((l) => l.tabs.includes('Fig. 12.4'));
const lab = listings.find((l) => l.tabs.includes('CUDA and HIP'));
check('Fig. 12.3 is shown with the book\'s 29 lines', fig3 && fig3.lines.length === 29 && fig3.lines[6].includes('__activemask()') && fig3.lines[19].includes('__shfl_sync(activeThreads, j, leader)'), JSON.stringify(fig3?.lines.slice(5, 8)));
check('Fig. 12.3\'s status is in the fixed vocabulary and says CUDA only', !!fig3 && fig3.status.startsWith('front-end-checked; emulator-checked') && fig3.status.includes('CUDA only'));
check('Fig. 12.4 is shown with the book\'s 26 lines', fig4 && fig4.lines.length === 26 && fig4.lines[6].includes('cg::coalesced_threads()') && fig4.lines[18].includes('shfl(j, 0)'));
check('lab 12.1\'s ballot form is shown, with a Metal tab', lab && lab.tabs.includes('Metal') && lab.lines.join('\n').includes('filterWarpAggregatedKernel') && lab.lines.join('\n').includes('lane_mask_t'), JSON.stringify(lab?.tabs));

// ---- predict
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Predict first') && e.textContent.includes('Which offset does lane 5')));
await (await pred.$('input[type=number]')).type('3');
await (await pred.$('button.primary')).click();
await pause(150);
const reveal = await pred.evaluate((e) => e.textContent);
check('the prediction reveals offset 3 for lane 5', reveal.includes('its offset is 3') && reveal.includes('That is your number'), reveal);

// ---- the visualiser
const book8 = { width: 8, pass: preset('book', 8), counter: 40, form: 'book' };
let s = await matches('opens on the book\'s lanes: mask 0x36, leader 1, count 4, one atomic 40 to 44, slots 40 to 43', book8,
  (s) => s.groups.length === 1 && s.groups[0][0] === '0x00000036' && s.groups[0][1] === 1 && s.slots.join(',') === ',40,41,,42,43,,' && s.correct);
check('the mask is printed in binary as 0011 0110', (await page.$eval(`${bench(V)} [data-mask-bits]`, (e) => e.textContent)) === '0011 0110');
const rows = await page.$$eval(`${bench(V)} tr[data-row-lane]`, (r) => r.map((x) => [...x.querySelectorAll('td')].map((t) => t.textContent)));
check('rank table: lane 5\'s lanes-below mask, the AND, rank 3, slot 43', same(rows[3], ['5', '0001 1111', '0001 0110', '3', '43']), JSON.stringify(rows));
check('the output strip fills slots 40 to 43 with lanes 1, 2, 4, 5', same(await page.$$eval(`${bench(V)} [data-out-slot]`, (x) => x.map((e) => e.dataset.outLanes)), ['1', '2', '4', '5']));
if (out) await page.$eval(bench(V), (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-visualiser.png' });

await box('split', true);
s = await matches('split into two snapshots: two atomics, the same slots', { ...book8, snapshots: halves(book8.pass) },
  (s) => s.atomics === 2 && s.slots.join(',') === ',40,41,,42,43,,' && s.text.includes('2 snapshots'));
if (out) await page.screenshot({ path: out + '-split.png' });
await box('split', false);

await setCounter(7);
await matches('the counter before moves every slot', { ...book8, counter: 7 }, (s) => s.slots.join(',') === ',7,8,,9,10,,');
await setCounter(40);

// planted bug: no - 1 after __ffs
await setForm('bug-ffs');
await matches('bug-ffs on the book\'s lanes: lane 2 leads and it passes, so the output is right by luck', { ...book8, form: 'bug-ffs' }, (s) => s.correct && s.groups[0][1] === 2);
await clickLanes([2]); // lanes 1, 4, 5 now
await matches('bug-ffs with lane 2 failing: leader 2 does not run, no atomic, undefined slots', { ...book8, form: 'bug-ffs', pass: passFrom(8, [1, 4, 5]) },
  (s) => !s.correct && s.flags.includes('leader-invalid') && s.flags.includes('shuffle-outside'));
await setWidth(32);
await press(V, 'only lane 31');
await matches('bug-ffs, only lane 31 passing: leader 32, a lane the warp does not have', { width: 32, pass: preset('lane31', 32), counter: 40, form: 'bug-ffs' },
  (s) => s.groups[0][1] === 32 && s.slots[31] === '?');
if (out) await page.screenshot({ path: out + '-bug-ffs.png' });

// planted bug: lanes-below includes the lane
await setForm('bug-below');
await press(V, 'the book\'s lanes 1, 2, 4, 5');
await matches('bug-below: every rank one too high, a hole and a key past the range', { width: 32, pass: preset('book', 32), counter: 40, form: 'bug-below' },
  (s) => s.flags.includes('gap') && s.flags.includes('out-of-range'));

// the naive HIP port on 64 lanes
await setWidth(64);
await setForm('book');
check('Fig. 12.3 as printed on 64 lanes does not compile (HIP\'s 64-bit mask)', (await state()).compile);
await setForm('hip-naive');
await matches('naive HIP port, book\'s lanes on 64 lanes: right, because no lane is above 31', { width: 64, pass: preset('book', 64), counter: 40, form: 'hip-naive' }, (s) => s.correct);
await press(V, 'only lanes 32 to 63');
await matches('naive HIP port, only lanes 32 to 63: leader -1, no atomic, undefined slots', { width: 64, pass: preset('upper', 64), counter: 40, form: 'hip-naive' },
  (s) => s.groups[0][1] === -1 && s.atomics === 0 && s.slots.slice(32).every((x) => x === '?'));
if (out) await page.$eval(bench(V), (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-hip.png' });
await setForm('ballot');
await matches('the ballot form on the same 64 lanes: right, one atomic, a 64-bit vote', { width: 64, pass: preset('upper', 64), counter: 40, form: 'ballot' },
  (s) => s.correct && s.groups[0][0] === '0xffffffff00000000');

// Metal
await setWidth(32);
await setForm('metal');
await press(V, 'none');
await clickLanes([3, 9]);
await matches('Metal, lanes 3 and 9 keep, lane 0 does not: lane 0 does the atomic, adding 2', { width: 32, pass: passFrom(32, [3, 9]), counter: 40, form: 'metal', guard: true },
  (s) => s.correct && s.groups[0][1] === 0 && s.groups[0][3] === '40>42');
await press(V, 'none');
await matches('Metal, nobody keeps, guarded as lab 12.1: no atomic', { width: 32, pass: preset('none', 32), counter: 40, form: 'metal', guard: true }, (s) => s.atomics === 0);
await box('skip the atomic', false);
await matches('Metal, nobody keeps, unguarded: lane 0 adds 0', { width: 32, pass: preset('none', 32), counter: 40, form: 'metal', guard: false },
  (s) => s.atomics === 1 && s.flags.includes('zero-add') && s.groups[0][3] === '40>40');
if (out) await page.screenshot({ path: out + '-metal.png' });

// every preset under every form at 32 lanes agrees with the model
const forms = ['book', 'ballot', 'metal', 'hip-naive', 'bug-ffs', 'bug-below'];
const presets = { book: 'the book\'s lanes 1, 2, 4, 5', none: 'none', all: 'all', lane31: 'only lane 31', alternate: 'alternating' };
await box('skip the atomic', true);
let all = true;
for (const f of forms) {
  await setForm(f);
  for (const [p, label] of Object.entries(presets)) {
    await press(V, label);
    const sNow = await state();
    const e = expected({ width: 32, pass: preset(p, 32), counter: 40, form: f, guard: true });
    if (!(same(sNow.groups, e.groups) && same(sNow.slots, e.slots) && sNow.correct === e.correct && same(sNow.flags, e.flags))) {
      all = false;
      console.log('  mismatch', f, p, JSON.stringify(sNow.groups), JSON.stringify(e.groups));
    }
  }
}
check('every form on every preset at 32 lanes matches the model', all);

// ---- which line is which?
const pick4 = async (n) => {
  await page.$eval(`${bench(W)} button[data-fig4-line="${n}"]`, (b) => b.click());
  await pause(90);
  return page.$eval(`${bench(W)} [data-verdict]`, (e) => e.dataset.verdict);
};
await page.$eval(`${bench(W)} button[data-pair="count"]`, (b) => b.click());
await pause(90);
check('a wrong match is marked', (await pick4(10)) === 'wrong');
for (const p of LINE_PAIRS) {
  await page.$eval(`${bench(W)} button[data-pair="${p.id}"]`, (b) => b.click());
  await pause(60);
  check(`"${p.fig3Text}" matches Fig. 12.4 line ${p.fig4}`, (await pick4(p.fig4)) === 'right');
}
check('all five pairs matched', (await page.$eval(`${bench(W)} [data-matched]`, (e) => e.dataset.matched)) === String(LINE_PAIRS.length));
if (out) await page.$eval(bench(W), (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-lines.png' });

// ---- questions
const q = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Which slot does lane 7 write')));
await (await q.$('input[type=number]')).type('43');
await (await q.$('button.primary')).click();
await pause(120);
check('numeric question: lane 7 writes slot 43', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
const choose = async (needle, start) => {
  const h = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await h.evaluate((e, s) => [...e.querySelectorAll('label')].find((l) => l.textContent.startsWith(s)).querySelector('input').click(), start);
  await (await h.$('button.primary')).click();
  await pause(120);
  return h.evaluate((e) => e.textContent);
};
check('choice question q-hip64: wrong output in lanes 32 to 63', (await choose('change only its mask argument', 'Wrong output whenever')).includes('Correct.'));
check('choice question q-metal: lane 0, adding 2', (await choose('Lane 0 fails the test; lanes 3 and 9 pass', 'Lane 0, adding 2')).includes('Correct.'));
check('choice question q-ffs: a wrong answer is explained', (await choose('Someone drops the', 'Never.')).includes('Not quite.'));

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await setWidth(8);
await setForm('book');
await press(V, 'the book\'s lanes 1, 2, 4, 5');
if (out) await page.$eval(bench(V), (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-light.png' });
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
check('at a narrow width the page does not scroll sideways', await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
if (out) await page.$eval(bench(V), (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-narrow.png' });

// The listings' statuses: LabKit's words from this machine's reports, and "failed on this machine" when a run failed.
await checkLabStatuses(check, 'ch12/12.3', [
  { label: 'CUDA and HIP', lab: 'ch12/lab1_unstable', file: 'kernel.cu' },
  { label: 'Metal', lab: 'ch12/lab1_unstable', file: 'kernel.metal' },
]);

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
