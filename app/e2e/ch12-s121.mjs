// Browser check for section 12.1: the "Stable or unstable?" explorer (against src/sim/filter.ts,
// imported directly), the "Does it need order?" classification, and two questions.
// Usage: node e2e/ch12-s121.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { FIG12_1, arrivalOrder, crossings, inPlaceOverwrites, keepFrom, stableFilter, unstableFilter } from '../src/sim/filter.ts';

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
const names = (keys) => keys.map((k) => 'k' + k).join(' ');

const E = 'Stable or unstable?';
const panel = (name) => page.$eval(`${bench(E)} svg[data-panel="${name}"]`, (s) => ({ output: s.dataset.output, crossings: Number(s.dataset.crossings) }));
const line = (name) => page.$eval(`${bench(E)} [data-line="${name}"]`, (e) => e.textContent.replace(/\s+/g, ' ').trim());

await goto(page, 'ch12/12.1');
await page.waitForSelector(bench(E));

// ---- the book's keys
const keep = keepFrom(FIG12_1.n, FIG12_1.kept);
let st = await panel('stable');
check('stable panel: the book\'s eight keys in input order, no crossing arrows', st.output === names(stableFilter(keep)) && st.crossings === 0, JSON.stringify(st));
let un = await panel('unstable');
check('unstable panel, run 1: the order the book draws, with the model\'s crossing count',
  un.output === names(FIG12_1.bookUnstable) && un.crossings === crossings([...FIG12_1.bookUnstable]), JSON.stringify(un));
check('check line: same keys, same count', (await line('unstable')).includes('same keys, same count'));
if (out) await page.$eval(bench(E), (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-explorer.png' });

// ---- run again: a new completion order each time, the set never changes
const seen = new Set([un.output]);
for (let run = 1; run <= 3; run++) {
  await clickText(E, 'Run again');
  un = await panel('unstable');
  const want = unstableFilter(keep, arrivalOrder(keep, 1000 + run));
  check(`run ${run + 1}: the model's order, same keys, same count`, un.output === names(want) && (await line('unstable')).includes('same keys, same count'), un.output);
  seen.add(un.output);
}
check('the order changed between runs', seen.size >= 3, [...seen].join(' | '));
st = await panel('stable');
check('the stable panel did not change', st.output === names(stableFilter(keep)));

// ---- click a key: k2 is kept now
await page.$eval(`${bench(E)} svg[data-panel="stable"] g[data-key="2"]`, (g) => g.dispatchEvent(new MouseEvent('click', { bubbles: true })));
await pause(120);
const keep2 = keepFrom(16, [...FIG12_1.kept, 2]);
st = await panel('stable');
un = await panel('unstable');
check('clicking k2 keeps it: the stable output gains k2 in its place', st.output === names(stableFilter(keep2)), st.output);
check('and the unstable panel starts a new seeded run', un.output === names(unstableFilter(keep2, arrivalOrder(keep2, 1000))), un.output);
check('the key is drawn as kept in both panels', (await page.$$eval(`${bench(E)} g[data-key="2"][data-kept="1"]`, (g) => g.length)) === 2);

await clickText(E, 'Drop all');
st = await panel('stable');
check('drop all: empty output', st.output === '' && (await line('stable')).startsWith('0 kept'), JSON.stringify(st));
await clickText(E, 'Keep all');
st = await panel('stable');
check('keep all: sixteen keys, no crossings', st.output.split(' ').length === 16 && st.crossings === 0);
check('"The book\'s keys" brings back Fig. 12.1', await clickText(E, 'The book\'s keys'));
un = await panel('unstable');
check('and with it the book\'s unstable order', un.output === names(FIG12_1.bookUnstable), un.output);

// ---- in place
await page.evaluate((sel) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('in-place')).querySelector('input').click(), bench(E));
await pause(120);
const ov = inPlaceOverwrites(keep);
const held = await page.$$eval(`${bench(E)} svg[data-panel="stable"] g[data-held]`, (g) => g.map((x) => [Number(x.dataset.held), x.dataset.moves]));
check('in place: every slot says which key it held', JSON.stringify(held) === JSON.stringify(ov.map((o) => [o.held, o.heldMovesTo == null ? '' : String(o.heldMovesTo)])), JSON.stringify(held));
check('k3 lands where k1 lived, and k1 must move to 0 first', held[1][0] === 1 && held[1][1] === '0');
const kinds = await page.$$eval(`${bench(E)} svg[data-panel="stable"] g[data-held]`, (g) => g.map((x) => x.dataset.kind));
check('in place: each slot says whether its old key must be moved, tested, or is itself', JSON.stringify(kinds) === JSON.stringify(ov.map((o) => o.kind)), JSON.stringify(kinds));
const moved = ov.filter((o) => o.kind === 'moved').length;
const tested = ov.filter((o) => o.kind === 'tested').length;
check(`the line counts every write onto another key: ${moved} to move and ${tested} to test`,
  (await line('stable')).includes(`${moved + tested} of the 8 writes`) && (await line('stable')).includes(`${moved} kept keys that must be read and moved`) && (await line('stable')).includes(`${tested} dropped keys that must still be read`), await line('stable'));
check('dropped keys are marked too, with "test"', (await page.$$eval(`${bench(E)} svg[data-panel="stable"] g[data-kind="tested"] text`, (t) => t.map((x) => x.textContent))).includes('test'));
if (out) await page.screenshot({ path: out + '-inplace.png' });

// ---- does it need order?
const C = 'Does it need order?';
const place = async (job, label) => {
  await page.evaluate((sel, job, label) => {
    const row = document.querySelector(`${sel} [data-job="${job}"]`);
    [...row.querySelectorAll('button')].find((b) => b.textContent.trim() === label).click();
  }, bench(C), job, label);
  await pause(60);
  return page.$eval(`${bench(C)} [data-job="${job}"]`, (e) => ({ right: e.dataset.right, text: e.textContent }));
};
const answers = { 'sort-next': 'unstable is enough', dedup: 'stable needed', gather: 'stable needed', seeded: 'stable needed', count: 'no compaction needed', 'kv-free': 'unstable is enough' };
let r = await place('sort-next', 'stable needed');
check('a wrong placement is marked and gives the reason', r.right === '0' && r.text.includes('we would say: unstable is enough') && r.text.includes('thrown away'), r.text);
for (const [job, label] of Object.entries(answers)) {
  r = await place(job, label);
  check(`job "${job}" placed as "${label}" is right`, r.right === '1', r.text);
}
check('score: 6 of 6', (await page.$eval(`${bench(C)} [data-score]`, (e) => e.dataset.score)) === '6');
if (out) await page.$eval(bench(C), (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-classify.png' });

// ---- questions
const q = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Into which position does k6 move')));
await (await q.$('input[type=number]')).type('3');
await (await q.$('button.primary')).click();
await pause(120);
check('numeric question: k6 moves to position 3', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
const q2 = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Same logits, same seed, two runs')));
await q2.evaluate((e) => [...e.querySelectorAll('label')].find((l) => l.textContent.startsWith('The distribution of draws is right')).querySelector('input').click());
await (await q2.$('button.primary')).click();
await pause(120);
check('choice question: right distribution, seed not reproducible', (await q2.evaluate((e) => e.textContent)).includes('Correct.'));

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await pause(150);
if (out) await page.$eval(bench(E), (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-light.png' });
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
const narrow = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
check('at a narrow width the page does not scroll sideways', narrow);
if (out) await page.$eval(bench(E), (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-narrow.png' });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
