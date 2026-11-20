// Browser check for section 12.8 (related patterns): the column-removal prediction, the
// "which way does the dependency run?" explorer in all four patterns, and the questions.
// Every number asserted is computed from the model the page uses (src/sim/inplace.ts).
// Usage: APP_URL=http://localhost:5173 node e2e/ch12-s128.mjs [--dark] [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { blockDependencies, dedupFlags, dedupMoves, moveByIndex } from '../src/sim/inplace.ts';

const dark = process.argv.includes('--dark');
const out = process.argv.slice(2).find((a) => !a.startsWith('--'));
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
const E = 'section.bench[aria-label="Which way does the dependency run?"]';
const txt = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const press = async (label) => {
  const ok = await page.evaluate((E, label) => {
    const b = [...document.querySelector(E).querySelectorAll('button')].find((x) => x.textContent.trim() === label);
    if (!b) return false;
    b.click();
    return true;
  }, E, label);
  await pause(120);
  return ok;
};
const select = async (aria, value) => {
  await page.$eval(`${E} select[aria-label="${aria}"]`, (s, v) => {
    s.value = String(v);
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await pause(120);
};
const click = async (sel) => {
  await page.$eval(sel, (e) => e.click());
  await pause(100);
};
const shownDeps = async () => ({
  direction: await page.$eval(E + ' [data-part="direction"]', (e) => e.dataset.direction),
  edges: await page.$$eval(E + ' [data-edge]', (es) => es.map((e) => e.dataset.edge)),
});
const modelDeps = (d) => ({ direction: d.direction, edges: d.edges.map((e) => `${e.reader}>${e.writer}`) });
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

await goto(page, 'ch12/12.8');
await page.waitForSelector(E);

// ------------------------------------------------ predict first
const pq = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Predict first')));
await (await pq.$('input[type=number]')).type('13');
await (await pq.$('button.primary')).click();
await pause(150);
const pt = await pq.evaluate((e) => e.textContent);
check('prediction: committing 13 reveals 13 and the arithmetic', pt.includes('You predicted 13') && pt.includes('2 × 5 + 3 = 13') && pt.includes('You had it'), pt.slice(0, 200));

// ------------------------------------------------ remove columns (the default): 4 x 6, column 2
let mv = moveByIndex({ kind: 'removeCols', rows: 4, cols: 6, remove: [2] });
const dests = await page.$$eval(E + ' [data-part="source"] [data-src]', (s) => s.map((x) => x.dataset.dest));
check('remove columns: every destination as the model', same(dests, mv.moves.map((m) => (m.dest == null ? '' : String(m.dest)))), JSON.stringify(dests));
await click(E + ' [data-src="16"]');
check('p-col: (2, 4) goes to 2 × 5 + (4 − 1) = 13', (await txt(E + ' [data-part="formula"]')).includes('2 × 5 + (4 − 1) = 13'));
check('p-col: result slot 13 holds source element 16', (await txt(E + ' [data-part="result"] [data-slot="13"]')) === '16');
check('remove columns in place: forward, edges as the model', same(await shownDeps(), modelDeps(blockDependencies(mv.moves, 24, 6))), JSON.stringify(await shownDeps()));
check('remove columns: direction is forward', (await shownDeps()).direction === 'forward');
// Remove column 0 as well.
await click(E + ' [data-col-head="0"]');
mv = moveByIndex({ kind: 'removeCols', rows: 4, cols: 6, remove: [0, 2] });
check('removing column 0 too: result is 4 × 4, edges as the model', (await page.$$(E + ' [data-part="result"] [data-slot]')).length === 16 && same(await shownDeps(), modelDeps(blockDependencies(mv.moves, 24, 6))));
if (out) await page.screenshot({ path: out + '-cols.png' });

// Out of place: nothing to order.
await page.$eval(E + ' label.check input', (e) => e.click());
await pause(100);
check('out of place: no order between blocks', (await page.$eval(E + ' [data-part="direction"]', (e) => e.dataset.direction)) === 'out-of-place');
await page.$eval(E + ' label.check input', (e) => e.click());
await pause(100);

// ------------------------------------------------ remove rows: 6 x 4, rows 1 and 4, blocks of 2 rows
await press('remove rows');
await select('rows', 6);
await select('columns', 4);
await select('row blocks', 3);
await click(E + ' [data-row-head="4"]');
mv = moveByIndex({ kind: 'removeRows', rows: 6, cols: 4, remove: [1, 4] });
check('remove rows: destinations as the model', same(await page.$$eval(E + ' [data-part="source"] [data-src]', (s) => s.map((x) => x.dataset.dest)), mv.moves.map((m) => (m.dest == null ? '' : String(m.dest)))));
check('remove rows in place: forward, edges as the model', same(await shownDeps(), modelDeps(blockDependencies(mv.moves, 24, 8))) && (await shownDeps()).direction === 'forward', JSON.stringify(await shownDeps()));
await click(E + ' [data-src="23"]');
check('remove rows: (5, 3) goes to (5 − 2) × 4 + 3 = 15', (await txt(E + ' [data-part="formula"]')).includes('(5 − 2) × 4 + 3 = 15'));

// ------------------------------------------------ pad columns: 4 x 6 to width 8, one row per block
await press('pad columns');
await select('rows', 4);
await select('columns', 6);
await select('row blocks', 4);
await select('padding', 2);
mv = moveByIndex({ kind: 'padCols', rows: 4, cols: 6, pad: 2 });
const pd = blockDependencies(mv.moves, 24, 6);
check('pad: result is 4 × 8 with 8 pad slots', (await page.$$(E + ' [data-part="result"] [data-slot]')).length === 32 && (await txt(E + ' [data-part="result"]')).split('pad').length - 1 === 8);
check('q-pad: padding in place runs backward, edges as the model', same(await shownDeps(), modelDeps(pd)) && pd.direction === 'backward', JSON.stringify(await shownDeps()));
check('pad: every edge points from a later reader to an earlier writer', (await shownDeps()).edges.every((e) => Number(e.split('>')[0]) > Number(e.split('>')[1])));
check('pad: writes past the old end are counted as new memory', (await txt(E + ' [data-part="direction"]')).includes(`${pd.fresh} into new memory`));
if (out) await page.screenshot({ path: out + '-pad.png' });

// ------------------------------------------------ remove duplicates
await press('remove duplicates');
const typeKeys = async (s) => {
  await page.$eval(E + ' input[aria-label="sorted keys"]', (el, s) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, s);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, s);
  await pause(120);
};
const row = (name) => page.$$eval(`${E} [data-row="${name}"] span`, (s) => s.map((x) => x.textContent));
await typeKeys('3, 3, 5, 7, 7, 7, 9');
const d = dedupFlags([3, 3, 5, 7, 7, 7, 9]);
check('dedup: keep flags 1 0 1 1 0 0 1', same(await row('keep'), d.keep.map(String)), JSON.stringify(await row('keep')));
check('dedup: output 3 5 7 9, 4 kept', same(await row('out'), d.out.map(String)) && (await txt(E + ' [data-part="count"]')) === '4 kept');
const longKeys = [1, 1, 2, 3, 3, 3, 4, 5, 5, 6, 7, 7];
await typeKeys(longKeys.join(', '));
await select('keys per block', 3);
check('dedup in place: forward, edges as the model', same(await shownDeps(), modelDeps(blockDependencies(dedupMoves(longKeys), 12, 3))) && (await shownDeps()).direction === 'forward', JSON.stringify(await shownDeps()));
await click(`${E} [data-row="key"] span:nth-child(5)`);
const fx = await txt(E + ' [data-part="formula"]');
check('dedup: a dropped key says so, and that the value decides', fx.includes('same as key 3: dropped') && fx.includes('need a scan'), fx);
await typeKeys('3, 1, 2');
check('dedup: an unsorted list is refused', (await txt(E + ' [data-part="error"]')).includes('must be sorted'));

// ------------------------------------------------ questions
const answer = async (has, how) => {
  const h = await page.evaluateHandle((has) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt').textContent.includes(has)), has);
  if (how.value) await (await h.$('input[type=number]')).type(how.value);
  else await h.evaluate((e, pick) => [...e.querySelectorAll('label')].find((l) => l.textContent.trim().startsWith(pick)).querySelector('input').click(), how.pick);
  await (await h.$('button.primary')).click();
  await pause(120);
  return h.evaluate((e) => e.textContent);
};
check('q-dedup: 4 is accepted', (await answer('[3, 3, 5, 7, 7, 7, 9]', { value: '4' })).includes('Correct.'));
check('q-noscan: the index reason is accepted', (await answer('need no scan', { pick: 'Whether a key survives' })).includes('Correct.'));
check('q-order-only: a flag only is accepted', (await answer('What does each block pass on', { pick: 'Only a flag' })).includes('Correct.'));
check('q-pad: last to first is accepted', (await answer('from width 6 to width 8', { pick: 'Each block must finish reading' })).includes('Correct.'));
const w = await answer('from width 6 to width 8', { pick: 'None' });
check('q-pad: "none" is marked wrong with its reason', w.includes('Not quite.') && w.includes('slots 12 and 13'));

// Phone width: the page itself never scrolls sideways.
await page.setViewport({ width: 390, height: 800, deviceScaleFactor: 1 });
await pause(300);
const sw = await page.evaluate(() => document.documentElement.scrollWidth);
check('no sideways page scroll at 390 px', sw <= 390, String(sw));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
