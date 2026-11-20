// Browser check for section 12.7 (in-place stable filter): the overlap explorer and its
// dependence diagram, the ordering puzzle, the lab 12.5 listing, and the questions. Every
// number asserted is computed from the same model the page uses (src/sim/inplace.ts).
// The narrated animation has its own check, e2e/ch12-scene-scan-chain.mjs.
// Usage: APP_URL=http://localhost:5173 node e2e/ch12-s127.mjs [--dark] [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { BOOK_PUZZLE, FIG_KEEP, orderingTable, overlap } from '../src/sim/inplace.ts';

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
const bench = (t) => `section.bench[aria-label="${t}"]`;
const E = bench('In-place overlap explorer');
const P = bench('Ordering puzzle');
const txt = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const press = async (scope, label) => {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelector(scope).querySelectorAll('button')].find((x) => x.textContent.trim() === label);
    if (!b) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(100);
  return ok;
};
const select = async (scope, label, value) => {
  await page.evaluate((scope, label, value) => {
    const s = [...document.querySelector(scope).querySelectorAll('label.field')].find((l) => l.textContent.trim().startsWith(label)).querySelector('select');
    s.value = String(value);
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, scope, label, value);
  await pause(120);
};
const clickSvg = async (sel) => {
  await page.$eval(sel, (e) => e.dispatchEvent(new MouseEvent('click', { bubbles: true })));
  await pause(100);
};
const counts = () => page.$$eval(E + ' [data-count]', (cs) => Object.fromEntries(cs.map((c) => [c.dataset.count, Number(c.dataset.value)])));
const lit = () => page.$$eval(E + ' [data-part="chain"] [data-node]', (ns) => ns.filter((n) => n.dataset.lit === '1').map((n) => n.dataset.node));
const sameCounts = (shown, model) => ['same', 'earlier', 'later'].every((k) => (shown[k] ?? 0) === model[k]);

// Does the lab file exist yet? (It is written in parallel with this section.)
const labRes = await fetch(BASE + '/api/lab-file?path=' + encodeURIComponent('ch12/lab5_inplace/solution/kernel.cu'));
const labText = labRes.ok ? await labRes.text() : null;

await goto(page, 'ch12/12.7');
await page.waitForSelector(E);

// ------------------------------------------------ the explorer: Fig. 12.10, two blocks of eight
let m = overlap(FIG_KEEP, 8);
check('defaults: 16 keys, 2 blocks, counts as the model', sameCounts(await counts(), m.counts), JSON.stringify(await counts()));
check('one arrow per kept key', (await page.$$(E + ' [data-write]')).length === m.writes.length);
check('red arrows are exactly the cross-block writes', JSON.stringify(await page.$$eval(E + ' g[data-write][data-kind="earlier"]', (g) => g.map((x) => Number(x.dataset.write)))) === JSON.stringify(m.writes.filter((w) => w.kind === 'earlier').map((w) => w.key)));

await press(E, 'k3');
let sel = await txt(E + ' [data-part="selected"]');
check('k3: slot 3 to slot 1, over k1, read by block 0, a barrier', sel.includes('k3 moves from slot 3 to slot 1') && sel.includes('over k1') && sel.includes('block 0 reads') && sel.includes('Same block'), sel);
check('k3: the path is read(0) → barrier(0) → write(0)', (await txt(E + ' [data-part="path"]')).includes('read(0) → barrier(0) → write(0)'));

await clickSvg(E + ' g[data-write="8"]');
sel = await txt(E + ' [data-part="selected"]');
check('k8 (clicking its arrow): slot 4, over k4, an earlier block', sel.includes('k8 moves from slot 8 to slot 4') && sel.includes('over k4') && sel.includes('earlier block'), sel);
check('k8: the panel is marked as a hazard', await page.$eval(E + ' [data-part="selected"]', (e) => e.classList.contains('bad')));
check('k8: the path is read(0) → scan(0) → scan(1) → write(1)', (await txt(E + ' [data-part="path"]')).includes('Ordered: read(0) → scan(0) → scan(1) → write(1)'));
check('k8: the diagram lights exactly that path', JSON.stringify((await lit()).sort()) === JSON.stringify(['read0', 'scan0', 'scan1', 'write1']), JSON.stringify(await lit()));
const b1 = await page.$$eval(E + ' [data-part="writes"] tbody tr', (rs) => rs.map((r) => [...r.querySelectorAll('td')].map((d) => d.textContent.trim())));
check('q-crossblock: four of block 1\'s writes are read by block 0', b1.filter((r) => Number(r[1].split('→')[0]) >= 8 && r[3] === 'block 0').length === 4, JSON.stringify(b1));
if (out) await page.screenshot({ path: out + '-explorer.png' });

// Toggle a key: keep k0 as well.
await clickSvg(E + ' [data-cell="before-0"]');
const keep2 = [...FIG_KEEP];
keep2[0] = true;
m = overlap(keep2, 8);
check('keeping k0 too: counts follow the model', sameCounts(await counts(), m.counts), JSON.stringify([await counts(), m.counts]));
check('keeping k0 too: k0 stays in its own slot', (await page.$eval(E + ' g[data-write="0"]', (g) => g.dataset.kind)) === 'own');

// Four blocks of four, and 32 keys.
await press(E, "Book's keep pattern");
await select(E, 'Blocks', 4);
m = overlap(FIG_KEEP, 4);
check('four blocks of four: counts follow the model (2 same, 6 earlier)', sameCounts(await counts(), m.counts), JSON.stringify(await counts()));
await select(E, 'Keys', 32);
await select(E, 'Blocks', 3);
m = overlap([...FIG_KEEP, ...FIG_KEEP], 11);
check('32 keys in 3 blocks: counts follow the model, none into a later block', sameCounts(await counts(), m.counts) && m.counts.later === 0, JSON.stringify(await counts()));
check('32 keys: 32 cells per row', (await page.$$(E + ' [data-cell^="before-"]')).length === 32);

// Insertion: the hazards point the other way, and a forward chain does not order them.
await select(E, 'Keys', 16);
await select(E, 'Blocks', 4);
await press(E, 'insert (the reverse, §12.8)');
m = overlap(FIG_KEEP, 4, 'insert');
check('insert: counts follow the model, later-block writes only', sameCounts(await counts(), m.counts) && m.counts.earlier === 0 && m.counts.later > 0, JSON.stringify(await counts()));
const later = m.writes.find((w) => w.kind === 'later');
await clickSvg(E + ` g[data-write="${later.key}"]`);
check(`insert: k${later.key} must wait for a later block`, (await txt(E + ' [data-part="selected"]')).includes('A later block'));
check('insert with the chain first → last: not ordered', (await txt(E + ' [data-part="path"]')).startsWith('Not ordered: no path'), await txt(E + ' [data-part="path"]'));
await press(E, 'chain: last → first');
const want = ['read' + later.reader, ...Array.from({ length: later.reader - later.writer + 1 }, (_, i) => 'scan' + (later.reader - i)), 'write' + later.writer];
check('insert with the chain last → first: ordered, through every scan in between', JSON.stringify((await lit()).sort()) === JSON.stringify([...want].sort()), JSON.stringify(await lit()));
if (out) await page.screenshot({ path: out + '-insert.png' });

// ------------------------------------------------ the ordering puzzle
const table = () => page.$$eval(P + ' [data-part="table"] td[data-cell]', (c) => Object.fromEntries(c.map((x) => [x.dataset.cell, x.dataset.ordered === '1'])));
const modelTable = (cfg) => Object.fromEntries(orderingTable(cfg).map((x) => [`${x.reader}-${x.writer}`, x.verdict.ordered]));
const same = async (cfg) => JSON.stringify(await table()) === JSON.stringify(modelTable(cfg));
check('puzzle: the book\'s kernel orders every pair', (await same(BOOK_PUZZLE)) && Object.values(await table()).every(Boolean));
check('puzzle: the verdict says ordered, through synchronizes-with', (await txt(P + ' [data-part="verdict"]')).includes('Ordered') && (await txt(P + ' [data-part="links"]')).includes('synchronizes with'));
await page.evaluate((P) => [...document.querySelector(P).querySelectorAll('[aria-label="Store order"] button')].find((b) => b.textContent === 'relaxed').click(), P);
await pause(100);
check('puzzle: a relaxed store breaks every pair', (await same({ ...BOOK_PUZZLE, publish: 'relaxed' })) && !Object.values(await table()).some(Boolean));
check('puzzle: the verdict names the relaxed store', (await txt(P + ' [data-part="verdict"]')).includes('a relaxed store releases nothing'));
await press(P, "The book's kernel");
await page.evaluate((P) => [...document.querySelector(P).querySelectorAll('[aria-label="Load order"] button')].find((b) => b.textContent === 'relaxed').click(), P);
await pause(100);
check('puzzle: a relaxed load breaks every pair', (await same({ ...BOOK_PUZZLE, observe: 'relaxed' })) && !Object.values(await table()).some(Boolean));
await press(P, "The book's kernel");
await page.evaluate((P) => [...document.querySelector(P).querySelectorAll('[data-part="block-i"] input[type=checkbox]')][0].click(), P);
await pause(100);
check('puzzle: without block i\'s barrier only the leader\'s reads are ordered', (await same({ ...BOOK_PUZZLE, barrierI: false })) && (await table())['leader-other'] && !(await table())['other-other']);
check('puzzle: another thread\'s read is now a data race', (await page.$eval(P + ' [data-part="verdict"]', (e) => e.dataset.ordered)) === '0');
await press(P, "block i's leader");
check('puzzle: the leader\'s own read is still ordered', (await page.$eval(P + ' [data-part="verdict"]', (e) => e.dataset.ordered)) === '1');
await press(P, "The book's kernel");
await page.evaluate((P) => [...document.querySelector(P).querySelectorAll('[data-part="block-j"] input[type=checkbox]')][0].click(), P);
await pause(100);
check('puzzle: without block j\'s barrier only the leader\'s writes are ordered', await same({ ...BOOK_PUZZLE, barrierJ: false }));
if (out) await page.screenshot({ path: out + '-puzzle.png' });

// ------------------------------------------------ the listing, the scene's presence, the questions
const code = await page.$eval('figure.code', (e) => e.textContent);
if (labText) {
  check('lab 12.5 listing is the lab file, with its kernel', code.includes('stableFilterSinglePassKernel') && code.includes(labText.split('\n').find((l) => l.trim().length > 20).trim()));
  const hl = await page.$$eval('figure.code .ln.hl', (ls) => ls.map((l) => l.textContent));
  check('the listing marks the counter, both barriers, the acquire and the release', hl.length === 5 && hl.some((l) => l.includes('memory_order_acquire')) && hl.some((l) => l.includes('memory_order_release')) && hl.some((l) => l.includes('fetch_add(1')), JSON.stringify(hl));
} else {
  console.log('note lab 12.5 is not written yet: the listing shows its placeholder');
  check('lab 12.5 listing falls back to a placeholder naming the file', code.includes('labs/ch12/lab5_inplace/solution/kernel.cu'));
}
check('listing carries its verification status', (await page.$eval('figure.code .status', (e) => e.textContent)).includes('unverified on CUDA and HIP hardware'));
check('the scan-chain animation is on the page', !!(await page.$('section.scene[aria-label="The scan chain orders the blocks"]')));
check('the designs table covers CUDA, HIP and Metal', /CUDA.*HIP.*Metal/.test(await txt('table[data-part="designs"]')));
check('the HIP row gives the documented-safe one-block design and marks the single pass unverified', await page.$eval('table[data-part="designs"]', (t) => {
  const hip = [...t.querySelectorAll('tbody tr')].find((r) => r.children[0].textContent.trim() === 'HIP')?.textContent ?? '';
  return hip.includes('documented-safe') && hip.includes('one block walks its range') && hip.includes('unverified') && hip.includes('B6');
}));
check('the testing paragraph says race can report the missing order and emu cannot', (await txt('.prose')).includes('race can report it') && (await txt('.prose')).includes('emu cannot find this bug'));

const answer = async (has, how) => {
  const h = await page.evaluateHandle((has) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt').textContent.includes(has)), has);
  if (how.value) await (await h.$('input[type=number]')).type(how.value);
  else await h.evaluate((e, pick) => [...e.querySelectorAll('label')].find((l) => l.textContent.trim().startsWith(pick)).querySelector('input').click(), how.pick);
  await (await h.$('button.primary')).click();
  await pause(120);
  return h.evaluate((e) => e.textContent);
};
check('q-erratum: k1 is accepted', (await answer('Page 300 says', { pick: 'k1' })).includes('Correct.'));
check('q-crossblock: 4 is accepted', (await answer('How many of block 1', { value: '4' })).includes('Correct.'));
check('q-relaxed: the data race is accepted', (await answer('A colleague copies', { pick: 'Seeing a relaxed flag' })).includes('Correct.'));
check('q-blockidx: the progress reason is accepted', (await answer('take its number from an atomic counter', { pick: 'With blockIdx' })).includes('Correct.'));
check('q-metal: one threadgroup is accepted', (await answer('rests only on behaviour Apple documents', { pick: 'One threadgroup' })).includes('Correct.'));
check('q-ex2: "every backend" is accepted', (await answer('Exercise 2 builds', { pick: 'It runs on every backend' })).includes('Correct.'));
const w = await answer('Page 300 says', { pick: 'k0' });
check('q-erratum: k0 is marked wrong with its reason', w.includes('Not quite.') && w.includes('Slot 0 receives k1'));

// Phone width: the page itself never scrolls sideways.
await page.setViewport({ width: 390, height: 800, deviceScaleFactor: 1 });
await pause(300);
const sw = await page.evaluate(() => document.documentElement.scrollWidth);
check('no sideways page scroll at 390 px', sw <= 390, String(sw));
const ovf = await page.$eval('[data-part="overlap-frame"]', (f) => {
  const s = f.querySelector('svg');
  return { drawn: s.getBoundingClientRect().width, design: s.viewBox.baseVal.width, scrolls: f.scrollWidth > f.clientWidth };
});
check('at 390 px the overlap drawing keeps its design size and scrolls in its own frame', ovf.drawn >= ovf.design - 0.5 && ovf.scrolls, JSON.stringify(ovf));

const errs = labText ? errors : errors.filter((e) => !e.includes('status of 404'));
check('no console or page errors', errs.length === 0, errs.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
