// Browser check for section 6.6: the unrolling listing, the issue timeline, the
// "where does x live?" classification, the backend table and the questions.
// Expected values are computed by the model the page runs on (src/sim/unroll.ts),
// imported directly.
// Usage: node e2e/ch06-s66.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { DEFAULTS, WHERE_SNIPPETS, bodyText, schedule, whereLives } from '../src/sim/unroll.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
let puts = 0;
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.method() === 'PUT' && r.url().includes('/api/progress')) {
    puts++;
    return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  }
  return r.continue();
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (t) => `section.bench[aria-label="${t}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const clickText = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const btn = [...document.querySelector(scope).querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(80);
  return done;
};
const setRange = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
    set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(80);
};

await goto(page, 'ch06/6.6');
await page.waitForSelector(bench('Issue timeline'));

// ---- the listing
check('the unrolling listing is the snippet, with the reordered loop highlighted',
  (await text('figure.code pre')).includes('#pragma unroll 4') && (await page.$$eval('figure.code .ln.hl', (l) => l.length)) === 9);

// ---- issue timeline
const T = bench('Issue timeline');
const counters = async () => ({
  branches: Number(await text(T + ' [data-k="branches"]')),
  stallDep: Number(await text(T + ' [data-k="stallDep"]')),
  stallBranch: Number(await text(T + ' [data-k="stallBranch"]')),
  total: Number(await text(T + ' [data-k="total"]')),
});
const cells = () => page.$$eval(T + ' svg[aria-label="cycle-by-cycle issue of one thread"] [data-cell]', (g) => g.map((x) => x.dataset.cell).join(' '));
const expect = async (name, cfg) => {
  const m = schedule({ ...DEFAULTS, ...cfg });
  const c = await counters();
  const ok = c.branches === m.branches && c.stallDep === m.stallDep && c.stallBranch === m.stallBranch && c.total === m.total && (await cells()) === m.cells.map((x) => x.kind).join(' ')
    && (await text(T + ' [data-k="body"]')) === bodyText(cfg.unroll ?? 1, cfg.reorder ?? false);
  check(name, ok, JSON.stringify({ page: c, model: { branches: m.branches, stallDep: m.stallDep, stallBranch: m.stallBranch, total: m.total }, drawn: (await cells()).slice(0, 200) }));
  return c;
};
let c = await expect('no unrolling: 16 branches, 128 cycles with the default latencies', { unroll: 1 });
check('which are the numbers the lesson quotes', c.branches === 16 && c.total === 128 && c.stallDep === 48 && c.stallBranch === 32);
await clickText(T, 'unroll 4');
c = await expect('unrolled by 4: 4 branches, branch stalls 32 → 8, total 92', { unroll: 4 });
check('the dependence stalls stay at 48', c.stallDep === 48 && c.branches === 4 && c.total === 92);
await page.$eval(T + ' label.check input', (e) => e.click());
await pause(80);
c = await expect('unrolled by 4 and reordered: no dependence stalls, 44 cycles', { unroll: 4, reorder: true });
check('reordering lowered the stall cycles', c.stallDep === 0 && c.total === 44);
if (out) await page.screenshot({ path: out + '-timeline-reordered.png' });
await clickText(T, 'unroll 16 (full)');
c = await expect('fully unrolled: no branch at all', { unroll: 16, reorder: true });
check('and the body text says no loop is left', c.branches === 0 && (await text(T + ' p.ui')).startsWith('No loop left.'));
await clickText(T, 'unroll 2');
await setRange(T + ' input[aria-label="A-to-B latency"]', 6);
await setRange(T + ' input[aria-label="branch stall"]', 5);
await expect('the sliders feed the model: unroll 2, reordered, latency 6, branch stall 5', { unroll: 2, reorder: true, latencyAB: 6, branchStall: 5 });

// ---- where does x live?
const X = bench('Where does x live?');
for (const s of WHERE_SNIPPETS) {
  const want = whereLives(s);
  const wrong = want.place === 'registers' ? 'local memory (off-chip)' : 'registers';
  const right = want.place === 'registers' ? 'registers' : 'local memory (off-chip)';
  const card = `${X} [data-snippet="${s.id}"]`;
  check(`kernel "${s.id}" shows its code from the snippet file`, (await text(card + ' pre')).includes(s.code.split('\n')[0].trim()));
  await clickText(card, wrong);
  const v1 = await text(card + ' [data-k="verdict"]');
  await clickText(card, right);
  const v2 = await text(card + ' [data-k="verdict"]');
  check(`kernel "${s.id}": the wrong pick is marked, the right one (${want.place}) accepted, with the rule`,
    v1.startsWith('not this') && v2.startsWith('right') && v2.includes(want.place) && (await text(card)).includes(want.rule), `${v1} / ${v2}`);
}
check('the score counts the final picks', (await text(X + ' [data-k="score"]')) === `${WHERE_SNIPPETS.length} of ${WHERE_SNIPPETS.length} right so far.`, await text(X + ' [data-k="score"]'));
if (out) await page.screenshot({ path: out + '-where.png' });

// ---- backend table
const table = await text('table.data.wide');
check('the backend table names the pragma on CUDA and HIP and nothing documented on Metal',
  table.includes('#pragma unroll 1') && table.includes('AMD calls the numeric form a hint') && table.includes('nothing in the language specification') && table.includes('clang loop unroll(full)'));

// ---- questions
const q = async (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const qb = await q('unrolled by a factor of 4. How many loop branches');
await (await qb.$('input[type=number]')).type('4');
await (await qb.$('button.primary')).click();
await pause(120);
check('q-branches accepts 4', (await qb.evaluate((e) => e.textContent)).includes('Correct.'));
const choose = async (needle, label) => {
  const h = await q(needle);
  await h.evaluate((e, label) => [...e.querySelectorAll('label')].find((l) => l.textContent.startsWith(label)).querySelector('input').click(), label);
  await (await h.$('button.primary')).click();
  await pause(120);
  return h.evaluate((e) => e.textContent);
};
check('q-pragma1: the local-memory answer is correct', (await choose('#pragma unroll 1 above a coarsening loop', 'v likely goes to local memory')).includes('Correct.'));
check('q-metal: "Apple documents it" is wrong, with the reason', (await choose('You port a coarsened kernel to Metal', 'Yes: Apple documents this pragma')).includes('no loop pragma at all'));
check('q-metal: "the AIR records it" is wrong too: the driver compiles AIR again', (await choose('You port a coarsened kernel to Metal', 'Yes: the AIR records')).includes('compiled again by the driver'));
const qm = await choose('You port a coarsened kernel to Metal', 'You cannot tell');
check('q-metal: measure against constant indices, with Apple\'s advice quoted', qm.includes('Correct.') && qm.includes('randomly index into thread-scoped'), qm);
check('answers went to the (intercepted) progress API', puts >= 1, String(puts));

// ---- narrow screen
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(400);
const narrow = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
check('at a narrow width the page does not scroll sideways', narrow);
if (out) {
  await page.$eval(T, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
