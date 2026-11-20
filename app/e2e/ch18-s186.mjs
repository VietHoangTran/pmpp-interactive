// Browser check for chapter 18's section 18.6: the stepper "One block's private frontier"
// (src/sim/bfspriv.ts), the stepper "Three grid barriers" with its "Does it fit?" tab
// (src/sim/gridbfs.ts), the levels-and-launches prediction, the scene "One launch for every
// level", the code listings and two questions. The models import another module without a
// file extension, so they are not imported here; the expected values are the ones their unit
// tests pin (bfspriv.test.ts, gridbfs.test.ts).
// Usage: APP_URL=http://localhost:5173 node e2e/ch18-s186.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';

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
const P = bench("One block's private frontier");
const G = bench('Three grid barriers');
const S = bench('One launch for every level');
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const attr = (sel, a) => page.$eval(sel, (e, a) => e.getAttribute(a), a).catch(() => null);
const pressIn = async (scope, label, exact = false) => {
  const ok = await page.evaluate((scope, label, exact) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => (exact ? x.textContent.trim() === label : x.textContent.trim().startsWith(label)));
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label, exact);
  await pause(120);
  return ok;
};
// Set a range, number input or select the way a user would, so React sees the change.
const setValue = async (sel, value) => {
  const ok = await page.evaluate((sel, value) => {
    const el = document.querySelector(sel);
    if (!el) return false;
    const proto = el.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, String(value));
    el.dispatchEvent(new Event(el.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  }, sel, value);
  await pause(150);
  return ok;
};
const setCheck = async (scope, label, on) => {
  await page.evaluate((scope, label, on) => {
    const box = [...document.querySelectorAll(scope + ' label.check')].find((l) => l.textContent.trim() === label).querySelector('input');
    if (box.checked !== on) box.click();
  }, scope, label, on);
  await pause(120);
};
const toEnd = (scope) => pressIn(scope, 'Run to the end');

await goto(page, 'ch18/18.6');
await page.waitForSelector(P);
await page.waitForSelector(G);

// ---- "One block's private frontier"
const finalTally = () => attr(`${P} [data-final-tally]`, 'data-final-tally');
const outcome = (scope) => attr(`${scope} [data-outcome]`, 'data-outcome');
check('the stepper starts at the corrected kernel: capacity 2, 4 threads, 2 blocks, fan-out',
  (await text(`${P} [data-value="capacity"]`)) === '2' && (await text(`${P} [data-value="threads"]`)) === '4' && (await text(`${P} [data-value="blocks"]`)) === '2');
await pressIn(P, 'Step', true);
let n = await text(`${P} [data-narrate]`);
check('one step: block 0, thread 0 zeroes the private counter', n.includes('Block 0, thread 0 sets the private counter numCurrFrontier_s to 0'), n);
await pressIn(P, 'Run to next barrier');
n = await text(`${P} [data-narrate]`);
check('run to the next barrier: barrier 1 opens for block 0', n.startsWith('Barrier 1 opens for block 0'), n);
await toEnd(P);
// bfspriv.test.ts: fan-out, capacity 2, 2 blocks of 4: 16 new, 12 overflows, 14 global atomics against 16.
check('capacity 2, fan-out: 14 global atomics against Fig. 18.12\'s 16, with 12 overflows', (await finalTally()) === '14/16/12', String(await finalTally()));
check('12 public slots were written by the overflow path', (await page.$$(`${P} [data-via="overflow"]`)).length === 12);
check('the public counter ends at 16', (await text(`${P} [data-public-count]`)) === '16');
check('the outcome is correct', (await outcome(P)) === 'correct');
const fit = await page.$eval(`${P} [data-outcome-chip]`, (c) => {
  const box = c.parentElement.parentElement.getBoundingClientRect();
  const r = c.getBoundingClientRect();
  return { sw: c.scrollWidth, cw: c.clientWidth, right: r.right, limit: box.right };
});
check('the outcome sentence wraps inside its panel instead of running past the edge', fit.sw <= fit.cw && fit.right <= fit.limit + 0.5, JSON.stringify(fit));
check('blocks reserve in the order they ran: block 0 then block 1, 2 slots each',
  (await attr(`${P} [data-reservations]`, 'data-reservations')) === '0:6+2 1:14+2', String(await attr(`${P} [data-reservations]`, 'data-reservations')));
const passes = await text(`${P} [data-commit-passes]`);
check('the commit addresses are consecutive within a pass', passes.includes('block 0: [6, 7]') && passes.includes('block 1: [14, 15]'), passes);
check('the corrected kernel races nowhere', (await attr(`${P} [data-races]`, 'data-races')) === '');
if (out) {
  await page.$eval(P, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-private.png' });
}

await setValue(`${P} input[aria-label="PRIVATE_FRONTIER_CAPACITY"]`, 8);
await toEnd(P);
check('capacity 8: no overflow, one global atomic per block (2 against 16)', (await finalTally()) === '2/16/0', String(await finalTally()));

await setValue(`${P} select[aria-label="Block order"]`, 'reversed');
await toEnd(P);
const rev = await attr(`${P} [data-reservations]`, 'data-reservations');
check('last block first: block 1 reserves slots 0 to 7, block 0 slots 8 to 15', rev === '1:0+8 0:8+8', String(rev));
check('another order, still correct', (await outcome(P)) === 'correct');
await setValue(`${P} select[aria-label="Block order"]`, 'in-order');

// The book's plain store: right numbers, flagged as a race.
await setValue(`${P} input[aria-label="PRIVATE_FRONTIER_CAPACITY"]`, 2);
await pressIn(P, 'plain store', true);
await toEnd(P);
check('the plain store gives the right result in order', (await outcome(P)) === 'correct' && (await finalTally()) === '14/16/12');
const races = (await attr(`${P} [data-races]`, 'data-races')) ?? '';
check('the race panel reports the fetch_add against the store', races.split(' ').includes('count|reset'), races);
const raceText = await text(`${P} [data-race="count|reset"]`);
const lineNums = [...raceText.matchAll(/line (\d+)/g)].map((m) => Number(m[1]));
const listingLine = (k) => page.$eval(`${P} .code pre .ln[data-n="${k}"]`, (e) => e.textContent.trim()).catch(() => '');
check('the race panel names two lines', lineNums.length === 2, raceText);
const l1 = await listingLine(lineNums[0]);
const l2 = await listingLine(lineNums[1]);
check('those lines are the fetch_add and the plain store',
  l1.includes('currFrontierIdx_s = numCurrFrontier_s_ref') && l2 === 'numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;', `${l1} | ${l2}`);
await pressIn(P, 'atomic store', true);
check('the atomic store: no race', (await attr(`${P} [data-races]`, 'data-races')) === '');

// Barrier 2 removed, in order: thread 0 reserves before the others count.
await setValue(`${P} input[aria-label="PRIVATE_FRONTIER_CAPACITY"]`, 8);
await setCheck(P, 'barrier 2', false);
await pressIn(P, 'Go to the first failure');
n = await text(`${P} [data-narrate]`);
check('barrier 2 removed: the first failure is a short reservation', n.includes('reserves') && n.includes('Failure') && n.includes('read the counter before every thread had counted'), n);
await toEnd(P);
check('barrier 2 removed: the outcome is not correct', (await outcome(P)) !== 'correct', String(await outcome(P)));
await setValue(`${P} select[aria-label="Thread order"]`, 'reverse');
await toEnd(P);
check('barrier 2 removed, reversed: correct (thread 0 reserves last)', (await outcome(P)) === 'correct');
await setCheck(P, 'barrier 2', true);
await setCheck(P, 'barrier 3', false);
await toEnd(P);
check('barrier 3 removed, reversed: stopped by an out-of-range store', (await outcome(P)) === 'crash');
await setCheck(P, 'barrier 3', true);
await setValue(`${P} select[aria-label="Thread order"]`, 'forward');

// The `if` commit with more entries than threads.
await setValue(`${P} input[aria-label="threads per block"]`, 2);
await setValue(`${P} input[aria-label="PRIVATE_FRONTIER_CAPACITY"]`, 4);
await pressIn(P, 'an if', true);
await toEnd(P);
const bad = await text(`${P} [data-outcome]`);
check('commit with an if, capacity 4, 2 threads: entries lost', (await outcome(P)) === 'wrong' && bad.includes('lost'), bad);
await pressIn(P, 'strided loop', true);
await toEnd(P);
check('the strided loop copies them all', (await outcome(P)) === 'correct');
await pressIn(P, 'Corrected kernel');

// ---- the code listings
const tabText = async (label) => {
  await page.evaluate((label) => [...document.querySelectorAll('.code .tabs button')].find((b) => b.textContent.trim() === label).click(), label);
  await pause(120);
  return page.evaluate((label) => {
    const fig = [...document.querySelectorAll('figure.code')].find((f) => [...f.querySelectorAll('.tabs button')].some((b) => b.textContent.trim() === label));
    return { code: fig.querySelector('pre').textContent, hl: [...fig.querySelectorAll('.ln.hl')].map((l) => l.textContent.trim()).join(' / ') };
  }, label);
};
let t = await tabText("the book's line 29");
check('the line-29 listing marks the plain store', t.hl === 'numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;', t.hl);
t = await tabText('corrected: atomic store');
check('the corrected listing marks the atomic store', t.hl.startsWith('numCurrFrontier_s_ref.store(PRIVATE_FRONTIER_CAPACITY,'), t.hl);
t = await tabText('corrected: clamp');
check('the clamp listing marks the clamp', t.hl.startsWith('if (numCurrFrontier_s > PRIVATE_FRONTIER_CAPACITY)'), t.hl);
t = await tabText('sizing and launch');
check('the launch passes six arguments', /kernelArgs, 0, 0\);/.test(t.code) && t.code.includes('cudaOccupancyMaxActiveBlocksPerMultiprocessor'), t.code.slice(0, 200));
t = await tabText('include, CUDA and HIP');
check('the include is chosen by vendor', t.code.includes('#include <hip/hip_cooperative_groups.h>') && t.code.includes('#include <cooperative_groups.h>'));

// ---- the prediction
const pq = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('1000 × 1000 grid')));
if (await pq.evaluate((e) => !!e.querySelector('input'))) {
  await (await pq.$('input')).type('1998');
  await pq.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.includes('Commit')).click());
  await pause(200);
}
const reveal = await pq.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
check('the reveal: 1,998 levels, 1,999 launches, frontiers never above 1,000', reveal.includes('1,998 levels') && reveal.includes('1,999 launches') && reveal.includes('more than 1,000'), reveal);

// ---- the scene
await page.$eval(S, (e) => e.scrollIntoView({ block: 'start' }));
check('the scene has six parts', (await page.$$(`${S} .seg button`)).length === 6);
const sceneAt = async (part, pos) => {
  await page.evaluate((S, part) => document.querySelectorAll(S + ' .seg button')[part - 1].click(), S, part);
  await pause(100);
  await setValue(`${S} input[type=range]`, pos);
  return page.$eval(`${S} svg`, (e) => e.textContent);
};
let svg = await sceneAt(1, 1000);
check('part 1: the 16 × 16 road grid takes 31 launches', svg.includes('launch 31 of 31'), svg.slice(0, 200));
svg = await sceneAt(4, 1000);
check('part 4: the too-large grid is refused with both error names', svg.includes('cudaErrorCooperativeLaunchTooLarge') && svg.includes('hipErrorCooperativeLaunchTooLarge'));
svg = await sceneAt(5, 1000);
check('part 5: three barriers and Exercise 3', svg.includes('grid.sync() 3') && svg.includes('Exercise 3'));
check('captions are off by default and the switch is there', !(await page.$(`${S} .caption`)) || (await page.$eval(S, (e) => e.textContent.includes('captions'))));
if (out) await page.screenshot({ path: out + '-scene.png' });

// ---- "Three grid barriers"
check('all three barriers, in order: correct', (await toEnd(G), (await outcome(G)) === 'correct'), String(await outcome(G)));
await setCheck(G, 'grid.sync() 2', false);
await toEnd(G);
check('barrier 2 removed, thread 0 first: early exit, deadlock', (await outcome(G)) === 'early exit, deadlock', String(await outcome(G)));
await pressIn(G, 'Go to the first failure');
n = await text(`${G} [data-narrate]`);
check('the first failure: a thread reads after grid thread 0 has reset the counter', n.includes('has already reset the counter'), n);
const hist = await text(`${G} [data-history]`);
check('the counter history shows the reset before that read', hist.includes('resets') && hist.includes('reads 0'), hist);
if (out) {
  await page.$eval(G, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-barriers.png' });
}
await pressIn(G, 'opens (the course', false);
await toEnd(G);
check('the same on the emulator\'s rule: early exit, no deadlock, wrong levels', (await outcome(G)) === 'early exit');
await pressIn(G, 'never opens', false);
await setValue(`${G} select[aria-label="Order"]`, 'reverse');
await toEnd(G);
check('barrier 2 removed, thread 0 last: correct', (await outcome(G)) === 'correct');
await setCheck(G, 'grid.sync() 2', true);
await setCheck(G, 'grid.sync() 3', false);
await toEnd(G);
check('barrier 3 removed, thread 0 last: appends lost to a late reset', (await outcome(G)) === 'appends lost to a late reset', String(await outcome(G)));
await setCheck(G, 'grid.sync() 3', true);
await setCheck(G, 'grid.sync() 1', false);
await toEnd(G);
check('barrier 1 removed, reversed: a thread read a partial count', (await outcome(G)) === 'a thread read a partial count', String(await outcome(G)));
await pressIn(G, 'Exercise 3', false);
await toEnd(G);
check('Exercise 3 with its one barrier: correct', (await outcome(G)) === 'correct');
await setCheck(G, 'the one grid.sync()', false);
await setValue(`${G} select[aria-label="Order"]`, 'forward');
await toEnd(G);
check('Exercise 3 without it: a partial count', (await outcome(G)) === 'a thread read a partial count', String(await outcome(G)));
await pressIn(G, "Fig. 18.17's swap", false);
await setValue(`${G} select[aria-label="Graph"]`, 'tree');
await toEnd(G);
check('the tree with three barriers: correct', (await outcome(G)) === 'correct');

// "Does it fit?"
await pressIn(G, 'Does it fit?', true);
await setValue(`${G} input[data-input="sms"]`, 2);
await setValue(`${G} input[data-input="per-sm"]`, 3);
await setValue(`${G} input[data-input="grid"]`, 7);
check('R = 2 × 3 = 6', (await text(`${G} [data-slots]`)) === '6');
check('G = 7 > R: the cooperative launch is refused', (await attr(`${G} [data-cooperative]`, 'data-cooperative')) === 'refused');
const coop = await text(`${G} [data-cooperative]`);
check('the refusal names both errors and Metal\'s lack of a counterpart', coop.includes('cudaErrorCooperativeLaunchTooLarge') && coop.includes('hipErrorCooperativeLaunchTooLarge') && coop.includes('Metal'), coop);
check('G = 7 > R under an ordinary launch: never completes', (await attr(`${G} [data-ordinary]`, 'data-ordinary')) === 'never completes');
if (out) await page.screenshot({ path: out + '-fit.png' });
await pressIn(G, 'Size it');
check('sized from the occupancy product: launches', (await attr(`${G} [data-cooperative]`, 'data-cooperative')) === 'launches' && (await page.$eval(`${G} input[data-input="grid"]`, (e) => e.value)) === '6');

// ---- questions
const q = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('One level appends 3,000 vertices')));
if (await q.evaluate((e) => !!e.querySelector('input[type=number]'))) {
  await (await q.$('input[type=number]')).type('24');
  await q.evaluate((e) => e.querySelector('button.primary').click());
  await pause(150);
}
check('q-atomics accepts 24', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
const qb = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Remove the second')));
if (await qb.evaluate((e) => !!e.querySelector('input[type=radio]'))) {
  await qb.evaluate((e) => e.querySelectorAll('input[type=radio]')[0].click());
  await qb.evaluate((e) => e.querySelector('button.primary').click());
  await pause(150);
}
check('q-barrier2 accepts the read-after-reset answer', (await qb.evaluate((e) => e.textContent)).includes('Correct.'));

// ---- light and dark themes, and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await page.$eval(P, (e) => e.scrollIntoView({ block: 'start' }));
if (out) await page.screenshot({ path: out + '-private-light.png' });
await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
check('at a narrow width the page does not scroll sideways', narrow);
if (out) {
  await page.$eval(P, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
  await page.$eval(G, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-grid.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
