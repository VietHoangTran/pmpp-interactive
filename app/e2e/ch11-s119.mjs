// Browser check for section 11.9: the bytes-per-element explorer (against
// src/sim/scantraffic.ts), the "who waits for whom" simulator and the flag classification
// (against src/sim/lookback.ts; both models imported directly), the lab 11.4 prediction, the
// lab 11.5 listing (or its placeholder until the lab exists), questions, margin notes and
// phone width. The narrated scene has its own check, ch11-scene-lookback.mjs.
// Usage: APP_URL=http://localhost:5173 node e2e/ch11-s119.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { BOOK_PRESET, designCeilings, designTraffic, labRows, rssOverSsa } from '../src/sim/scantraffic.ts';
import { MP_TABLE, messagePassing, simulate } from '../src/sim/lookback.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const fmt = (v, digits = 3) => {
  if (!isFinite(v)) return 'n/a';
  if (v === 0) return '0';
  if (Math.abs(v) >= 1000) return v.toLocaleString('en-US', { maximumFractionDigits: 0 });
  return Number(v.toPrecision(digits)).toString();
};
const bench = (t) => `section.bench[aria-label="${t}"]`;
const inBench = (b, sel) => page.$eval(`${bench(b)} ${sel}`, (e) => e.textContent.trim());
const press = async (b, group, label) => {
  const ok = await page.evaluate((sel, group, label) => {
    const root = document.querySelector(sel);
    const scope = group ? root.querySelector(`[aria-label="${group}"]`) : root;
    const btn = scope && [...scope.querySelectorAll('button')].find((x) => x.textContent.trim() === label);
    if (!btn) return false;
    btn.click();
    return true;
  }, bench(b), group, label);
  await pause(100);
  return ok;
};
const setInput = async (sel, value, event = 'input') => {
  await page.$eval(sel, (el, v, ev) => {
    const proto = el.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event(ev, { bubbles: true }));
  }, value, event);
  await pause(120);
};
const reports = await (await fetch(BASE + '/api/reports')).json();

await goto(page, 'ch11/11.9');
await page.waitForSelector(bench('Bytes per element'));

// ---------------------------------------------------------------- bytes per element
const BY = 'Bytes per element';
check('the book\'s numbers button', await press(BY, null, "The book's numbers"));
const bars = await page.$$eval(`${bench(BY)} [data-design]`, (g) => Object.fromEntries(g.map((x) => [x.dataset.design, Number(x.dataset.ceiling)])));
const want = designCeilings(BOOK_PRESET.bandwidthGBs, BOOK_PRESET.S, 4, false);
check('preset: 419, 209 and 279 × 10⁹ elements/s side by side',
  ['single', 'ssa', 'rss'].every((d) => Math.abs(bars[d] - want[d] / 1e9) < 1e-3) && Math.round(bars.single) === 419 && Math.round(bars.ssa) === 209 && Math.round(bars.rss) === 279, JSON.stringify(bars));
check('preset, scan-scan-add: 16.0039 bytes per element and 209', (await inBench(BY, '[data-testid="bpe"]')) === fmt(designTraffic('ssa', 2048).bytesPerElement, 6) && (await inBench(BY, '[data-testid="cap"]')) === '209');
const kernels = async () => page.$$eval(`${bench(BY)} tr[data-kernel]`, (r) => r.map((x) => [x.dataset.kernel, Number(x.querySelector('[data-bytes]').dataset.bytes)]));
let k = await kernels();
check('scan-scan-add lists three kernels: 8, 8/S, 8 bytes per element', JSON.stringify(k.map((x) => x[0])) === '["kernel 1","kernel 2","kernel 3"]' && Math.abs(k[0][1] - 8) < 1e-6 && Math.abs(k[1][1] - 8 / 2048) < 1e-6 && Math.abs(k[2][1] - 8) < 1e-6, JSON.stringify(k));
await press(BY, 'Design', 'reduce-scan-scan (three kernels)');
k = await kernels();
check('reduce-scan-scan: kernel 1 stores only N/S (4 + 4/S bytes), total 12 + 12/S', Math.abs(k[0][1] - (4 + 4 / 2048)) < 1e-6 && (await inBench(BY, '[data-testid="bpe"]')) === fmt(12 + 12 / 2048, 6) && (await inBench(BY, '[data-testid="cap"]')) === '279', JSON.stringify(k));
await setInput(`${bench(BY)} select[aria-label="segment size"]`, 4096, 'change');
await press(BY, 'Design', 'scan-scan-add (three kernels)');
check('S = 4096: scan-scan-add moves 16.002 bytes (question q-ssa)', (await inBench(BY, '[data-testid="bpe"]')) === '16.002');
await page.evaluate((sel) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('bookkeeping')).querySelector('input').click(), bench(BY));
await pause(120);
check('bookkeeping adds the block sums the book leaves out', (await inBench(BY, '[data-testid="bpe"]')) === fmt(designTraffic('ssa', 4096, 4, true).bytesPerElement, 6));
await press(BY, 'Design', 'single pass (one kernel)');
check('single pass with bookkeeping: 8 + 24/S', (await inBench(BY, '[data-testid="bpe"]')) === fmt(8 + 24 / 4096, 6));
await press(BY, 'Element size', 'half, 2 bytes');
check('half elements: data bytes halve, bookkeeping stays 4-byte', (await inBench(BY, '[data-testid="bpe"]')) === fmt(designTraffic('single', 4096, 2, true).bytesPerElement, 6));
if (out) {
  await page.$eval(bench(BY), (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-bytes.png' });
}

// ---------------------------------------------------------------- the lab 11.4 prediction
const P = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('How many times faster than scan-scan-add')));
await (await P.$('input[type=number]')).type('1.3');
await (await P.$('button.primary')).click();
await pause(200);
const reveal = await P.evaluate((e) => e.textContent);
const measured = Object.keys(reports).filter((b) => b !== 'emu').map((b) => rssOverSsa(labRows(reports[b], 'ch11/lab4_'))).filter(Boolean);
if (measured.length) check('the reveal shows the measured ratio', measured.every((m) => reveal.includes(`${fmt(m.ratio)}×`)), reveal);
else check('with no lab 11.4 run, the reveal gives the byte ratio and says how to measure', reveal.includes('16/12 ≈ 1.33') && reveal.includes('pmpp.py run 11.4'), reveal);

// ---------------------------------------------------------------- who waits for whom
const WW = 'Who waits for whom';
const outcome = () => page.$eval(`${bench(WW)} [data-outcome]`, (e) => e.dataset.outcome);
const table = () => page.$$eval(`${bench(WW)} tr[data-scheme]`, (r) => r.map((x) => ({ ...x.dataset })));
const sameAsModel = async (name, cfg) => {
  const got = await table();
  const wantRows = ['barrier', 'single', 'decoupled'].map((s) => simulate({ ...cfg, scheme: s }));
  const ok = got.length === 3 && got.every((g, i) => g.outcome === wantRows[i].outcome && Number(g.time) === wantRows[i].time && Number(g.chain) === wantRows[i].criticalPath && Number(g.repeated) === wantRows[i].repeated);
  check(name, ok, JSON.stringify(got));
  const sel = simulate(cfg);
  const rows = await page.$$eval(`${bench(WW)} [data-row]`, (g) => g.map((x) => [Number(x.dataset.row), x.dataset.start, x.dataset.end]));
  if (sel.outcome !== 'refused')
    check(name + ': one timeline row per block, starts and ends as the model', rows.length === cfg.G && rows.every(([hw, s, e]) => String(sel.blocks[hw].start ?? '') === s && String(sel.blocks[hw].end ?? '') === e), JSON.stringify(rows.slice(0, 4)));
  return sel;
};
let cfg = { G: 8, R: 4, order: 'reversed', seed: 1, index: 'hardware', scheme: 'single', progress: 'started', even: false };
check('default is the first preset: blockIdx.x, later blocks first: deadlock', (await outcome()) === 'deadlock');
await sameAsModel('first preset: every scheme as the model', cfg);
check('the deadlock names a block waiting on one that has not started', (await inBench(WW, '[data-outcome]')).includes('Index 4 waits on 3, which has not started.'));
check('a stuck marker is drawn', !!(await page.$(`${bench(WW)} [data-stuck]`)));
if (out) {
  await page.$eval(bench(WW), (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-wait-deadlock.png' });
}
await press(WW, 'Block index', 'claimed from a counter');
cfg = { ...cfg, index: 'counter' };
check('the counter: finished', (await outcome()) === 'done');
await sameAsModel('with the counter: every scheme as the model', cfg);
await press(WW, 'Progress rule', 'no guarantee (Metal)');
cfg = { ...cfg, progress: 'none' };
check('no guarantee: outcome as the model', (await outcome()) === simulate(cfg).outcome, await outcome());
await sameAsModel('no guarantee: every scheme as the model, the barrier refused', cfg);
await press(WW, null, 'no progress guarantee');
cfg = { ...cfg, G: 6, R: 3, order: 'index', seed: 6, index: 'counter', progress: 'none' };
check('preset "no progress guarantee": the claimed index stalls', (await outcome()) === 'stall');
check('a block is drawn resident but not run', (await page.$$(`${bench(WW)} rect[stroke="var(--fail)"]`)).length > 0);
await press(WW, null, 'a long chain');
cfg = { ...cfg, G: 32, R: 32, order: 'index', index: 'counter', progress: 'started' };
const long = await sameAsModel('preset "a long chain": every scheme as the model', cfg);
const t = await table();
check('a long chain: single lookback 31 links, decoupled far fewer, for repeated additions', long.criticalPath === 31 && Number(t[1].chain) === 31 && Number(t[2].chain) < 31 && Number(t[2].repeated) > 0 && Number(t[2].time) < Number(t[1].time), JSON.stringify(t));
check('a long chain: the grid barrier runs with every block resident (2 barriers)', t[0].outcome === 'done' && Number(t[0].chain) === 2);
if (out) {
  await page.$eval(bench(WW), (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-wait-long.png' });
}
await setInput(`${bench(WW)} input[aria-label="slots"]`, 16);
cfg = { ...cfg, R: 16 };
check('R = 16 < G: the grid barrier is refused', (await table())[0].outcome === 'refused' && (await page.$eval(`${bench(WW)} [data-testid="R"]`, (e) => e.textContent)) === '16');
await press(WW, 'Scheme', 'grid-wide barrier');
check('selecting the refused barrier explains why', (await outcome()) === 'refused' && (await inBench(WW, '[data-outcome]')).includes('32 blocks do not fit in 16 slots'));
await press(WW, 'Scheme', 'decoupled lookback');
await press(WW, 'Start order', 'seeded shuffle');
const shuffleLabel = await page.$eval(bench(WW), (e) => [...e.querySelectorAll('button')].map((x) => x.textContent.trim()).find((t) => t.startsWith('New shuffle')) ?? '');
const seedBefore = Number(shuffleLabel.match(/seed (\d+)/)?.[1]);
check('a seeded shuffle offers a new seed', Number.isFinite(seedBefore) && (await press(WW, null, shuffleLabel)), shuffleLabel);
cfg = { ...cfg, order: 'shuffle', seed: seedBefore + 1, scheme: 'decoupled' };
await sameAsModel('a new seeded shuffle, decoupled: every scheme as the model', cfg);
await page.evaluate((sel) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('equal segments')).querySelector('input').click(), bench(WW));
await pause(120);
cfg = { ...cfg, even: true };
await sameAsModel('equal segments: every scheme as the model', cfg);

// ---------------------------------------------------------------- make the flag safe
const FS = 'Make the flag safe';
const verdict = () => page.$eval(`${bench(FS)} [data-verdict]`, (e) => e.dataset.verdict);
const code = () => page.$eval(`${bench(FS)} .code pre`, (e) => e.textContent);
check('relaxed on both sides: data race', (await verdict()) === messagePassing('relaxed', 'relaxed', 'device').verdict && (await verdict()) === 'race');
let kept = await page.$$eval(`${bench(FS)} [data-kept]`, (g) => g.map((x) => x.dataset.kept));
check('both reorderings are shown as possible', JSON.stringify(kept) === '["0","0"]');
await press(FS, 'Store order', 'release');
check('release + relaxed: still a race', (await verdict()) === 'race');
await press(FS, 'Load order', 'acquire');
check('release + acquire at device scope: guaranteed', (await verdict()) === 'guaranteed');
kept = await page.$$eval(`${bench(FS)} [data-kept]`, (g) => g.map((x) => x.dataset.kept));
check('both orders kept in the picture', JSON.stringify(kept) === '["1","1"]');
const c = await code();
check('the code is NVIDIA\'s spelling with the chosen orders', c.includes('f.store(1, cuda::memory_order_release);') && c.includes('while (f.load(cuda::memory_order_acquire) == 0) {}') && c.includes('thread_scope_device'));
// The first code figure in the bench is the CUDA listing; its status is in the fixed vocabulary.
const cudaStatus = () => page.$eval(`${bench(FS)} .code .tabs .status`, (e) => e.textContent.trim());
check('the guaranteed combination: the CUDA tab says front-end-checked', (await cudaStatus()) === 'front-end-checked', await cudaStatus());
await press(FS, 'Scope', 'thread_scope_block');
check('block scope: undefined whatever the orders', (await verdict()) === 'scope' && (await inBench(FS, '[data-verdict]')).includes('UB: data race'));
const all = await page.$$eval(`${bench(FS)} table tr[data-verdict]`, (r) => r.map((x) => x.dataset.verdict));
check('after three tries, the table of all eight combinations matches the model', JSON.stringify(all) === JSON.stringify(MP_TABLE.map((r) => r.verdict)), JSON.stringify(all));
const statusOf = (sel) => page.$$eval(`${bench(FS)} ${sel} .tabs .status`, (s) => s.map((x) => x.textContent.trim()));
check('block scope: the CUDA tab says front-end-checked, undefined on purpose', (await cudaStatus()) === 'front-end-checked; undefined on purpose', await cudaStatus());
// The Metal tabs load labs/ch11/snippets/s119_flag_msl41.metal and ..._msl32.metal.
await page.waitForFunction((sel) => document.querySelector(sel + ' [data-testid="metal"] pre')?.textContent.includes('memory_order_release'), { timeout: 10000 }, bench(FS)).catch(() => {});
const metal = await inBench(FS, '[data-testid="metal"]');
const msl41 = await (await fetch(BASE + '/api/lab-file?path=' + encodeURIComponent('ch11/snippets/s119_flag_msl41.metal'))).text();
check('the Metal panel shows the checked MSL 4.1 file: mem_flags spelling, coherent(device), still unsafe', metal.includes('memory_order_release, mem_flags::mem_device') && metal.includes('coherent(device)') && metal.includes('unsafe on Metal') && metal.includes(msl41.split('\n').find((l) => l.includes('atomic_store_explicit')).trim()));
check('the Metal tabs are marked unverified (compiled, never run)', (await statusOf('[data-testid="metal"]')).every((s) => s.startsWith('unverified') && s.includes('never run')), JSON.stringify(await statusOf('[data-testid="metal"]')));
check('the Metal panel no longer claims more than the test shows', !metal.includes('Both compile;'));
await page.evaluate((sel) => [...document.querySelectorAll(sel + ' [data-testid="metal"] .tabs button')].find((b) => b.textContent.includes('3.2')).click(), bench(FS));
await pause(100);
check('the Metal panel\'s second tab: relaxed atomics with device fences', (await inBench(FS, '[data-testid="metal"] pre')).includes('atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);'));
if (out) {
  await page.$eval(bench(FS), (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-flag.png' });
}

// ---------------------------------------------------------------- lab 11.5's listing
const lab5 = await fetch(BASE + '/api/lab-file?path=' + encodeURIComponent('ch11/lab5_singlepass/solution/kernel.cu'));
if (lab5.ok) {
  const listing = await page.evaluate(() => [...document.querySelectorAll('figure.code')].map((f) => f.textContent).find((t) => t.includes('lab 11.5')) ?? '');
  check('lab 11.5\'s solution is shown with its flag orders marked', listing.includes('memory_order_acquire') && listing.includes('memory_order_release'));
} else check('lab 11.5 not written yet: its placeholder names the planned file', (await page.$eval('[data-testid="lab5-missing"]', (e) => e.textContent)).includes('ch11/lab5_singlepass/solution/kernel.cu'));

// ---------------------------------------------------------------- questions
const answer = async (needle, pick) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(n)), needle);
  if (typeof pick === 'number') await (await q.$('input[type=number]')).type(String(pick));
  else await q.evaluate((e, p) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes(p)).querySelector('input').click(), pick);
  await (await q.$('button.primary')).click();
  await pause(120);
  return (await q.evaluate((e) => e.textContent)).includes('Correct.');
};
check('q-ssa: 16.002 accepted', await answer('with S = 4096 elements per block', 16.002));
const qs = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes('with S = 4096 elements per block')));
await qs.evaluate((e) => {
  const el = e.querySelector('input[type=number]');
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '16');
  el.dispatchEvent(new Event('input', { bubbles: true }));
});
await pause(80);
await (await qs.$('button.primary')).click();
await pause(120);
check('q-ssa: 16 alone is not accepted (the 8/S term counts)', !(await qs.evaluate((e) => e.textContent)).includes('Correct.'));
check('q-deadlock: the claimed index is the right answer', await answer('waits on the block with', 'an index claimed from an atomic counter'));
check('q-metal: several dispatches', await answer('10⁸ floats on Metal', 'A hierarchical scan in several dispatches'));
check('q-scope: undefined behaviour', await answer('thread_scope_block', 'Undefined behaviour'));
check('q-ex5: three states', await answer('Exercise 5 asks for decoupled lookback', 'Three: nothing yet'));

// ---------------------------------------------------------------- batched rows (facts D1)
const prose = await page.$eval('.prose', (e) => e.textContent);
check('batched rows: llama.cpp\'s CUDA build takes CUB\'s device scan (lookback) when a row is long for its batch',
  prose.includes('above 4,096') && prose.includes('up to 37') && prose.includes('decoupled lookback') && !prose.includes('none of llama.cpp') && !prose.includes('need none of the machinery'));

// ---------------------------------------------------------------- layout
const groups = await page.$$eval('.prose > [data-note-group]', (g) => g.length);
check('stacked margin notes are grouped (two places)', groups === 2, String(groups));
const afterBench = await page.$$eval('.prose > section.bench + aside.note, .prose > figure.code + aside.note', (n) => n.length);
check('no margin note directly after a bench or a code block', afterBench === 0, String(afterBench));
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
const wide = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
check('no sideways page scroll at 390 px', wide <= 0, String(wide));
// The simulator at phone width: the timeline switches to its narrow layout and fits, so the
// deadlock line, its label and the scheme table are on screen without scrolling the bench.
const narrowFit = async () => page.$eval(bench(WW), (b) => {
  const body = b.querySelector('.body');
  const svg = b.querySelector('[data-testid="timeline"] svg');
  const wrap = b.querySelector('[data-testid="schemes"]');
  const table = wrap.querySelector('table');
  const stuck = svg.querySelector('[data-stuck]');
  const vb = svg.viewBox.baseVal;
  const t = stuck?.querySelector('text').getBBox();
  return {
    layout: svg.dataset.layout,
    tableLayout: table.dataset.layout,
    body: body.scrollWidth - body.clientWidth,
    svg: Math.round(svg.getBoundingClientRect().width - body.clientWidth),
    // On a phone the app's CSS makes a data table its own scroller, so measure the table too.
    table: Math.max(wrap.scrollWidth - wrap.clientWidth, table.scrollWidth - table.clientWidth, Math.round(table.getBoundingClientRect().right - body.getBoundingClientRect().right)),
    schemes: [...table.querySelectorAll('th[data-scheme]')].map((x) => x.dataset.scheme + ':' + x.dataset.outcome).join(' '),
    stuck: stuck ? t.x >= 0 && t.x + t.width <= vb.width && Number(stuck.dataset.x) <= vb.width : null,
    label: svg.querySelector('[data-row] text').textContent,
  };
});
await press(WW, null, 'blockIdx.x, later blocks first');
let nf = await narrowFit();
check('390 px, deadlock preset: narrow timeline, nothing in the bench scrolls sideways, the stuck line and its label inside the drawing',
  nf.layout === 'narrow' && nf.body <= 1 && nf.svg <= 1 && nf.table <= 1 && nf.stuck === true && nf.label === 'b0', JSON.stringify(nf));
const deadlockCfg = { G: 8, R: 4, order: 'reversed', seed: 1, index: 'hardware', progress: 'started', even: false };
check('390 px: the table turns to one column per scheme, with the model\'s outcomes',
  nf.tableLayout === 'narrow' && nf.schemes === ['barrier', 'single', 'decoupled'].map((s) => `${s}:${simulate({ ...deadlockCfg, scheme: s }).outcome}`).join(' '), JSON.stringify(nf));
await press(WW, null, 'a long chain');
nf = await narrowFit();
check('390 px, a long chain: short labels (i0 b0), still no sideways scroll', nf.layout === 'narrow' && nf.body <= 1 && nf.table <= 1 && nf.label === 'i0 b0', JSON.stringify(nf));
await press(WW, null, 'no progress guarantee');
nf = await narrowFit();
check('390 px, no progress guarantee: the stall marker inside the drawing', nf.stuck === true && nf.body <= 1, JSON.stringify(nf));
if (out) {
  await page.$eval(bench(WW), (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-wait.png' });
}

const expected404 = (u) => /\/api\/lab-file\?path=ch11%2Flab5_/.test(u) || /\/content\/ch11\/narration\/(manifest\.json|lookback-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a lab file not yet written and missing narration audio are allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
