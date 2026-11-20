// Browser check for section 23.4, "Events, priorities, and who waits": the dependency puzzle
// (every ordering of Fig. 23.14, the feedback, the NCCL and NVSHMEM comparison), the two code
// listings with the book's line numbers, the "Who waits" explorer and its bug hunt (every
// read-out and every drawn exchange checked against src/sim/haloloop.ts, imported directly),
// the prediction, the questions, a narrow screen. The narrated scene has its own check
// (ch23-scene-loop.mjs). Passes with or without narration audio.
// Usage: node e2e/ch23-s234.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { DEFAULTS, changedEdges, edges, simulate } from '../src/sim/haloloop.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file, and keep what would have been written.
const puts = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.method() === 'PUT' && r.url().includes('/api/progress')) {
    try { puts.push(JSON.parse(r.postData() ?? '{}')); } catch { /* not JSON */ }
    r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  } else r.continue();
});
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const r2 = (x) => (Math.round(x * 100) / 100).toFixed(2);
const P = '[data-puzzle]';
const W = '[data-who-waits]';

await goto(page, 'ch23/23.4');
await page.waitForSelector(P);
await page.waitForSelector(W);

// ------------------------------------------------------------ the code
const loopLines = await page.$$eval(P + ' .code pre .ln', (ls) => ls.map((l) => [Number(l.dataset.n), l.textContent.replace(/\n$/, '')]));
check('Fig. 23.14 is shown with 43 numbered lines', loopLines.length === 43 && loopLines[0][0] === 1 && loopLines[42][0] === 43, String(loopLines.length));
check('line 19 is the corrected wait on computeTop', loopLines[18][1].trim() === 'cudaStreamWaitEvent(internalStream, computeTop, 0);', loopLines[18][1]);
check('line 27 is the host sync on topStream, line 37 the one on internalStream',
  loopLines[26][1].includes('cudaStreamSynchronize(topStream)') && loopLines[36][1].includes('cudaStreamSynchronize(internalStream)'));
check('the listing says front-end-checked, with its file', (await page.$eval(P + ' .code .status', (e) => e.textContent)) === 'front-end-checked (labs/ch23/snippets/overlap_loop.cu)');
const ncclFirst = await page.evaluate(() => {
  const f = [...document.querySelectorAll('figure.code')].find((x) => x.textContent.includes('ncclGroupStart'));
  const ls = [...f.querySelectorAll('.ln')];
  return { first: Number(ls[0].dataset.n), last: Number(ls[ls.length - 1].dataset.n), text: ls[0].textContent, l43: ls.find((l) => l.dataset.n === '43')?.textContent ?? '' };
});
check('Fig. 23.20 lines 27–44 keep the book\'s numbers', ncclFirst.first === 27 && ncclFirst.last === 44 && ncclFirst.text.startsWith('ncclGroupStart') && ncclFirst.l43.includes('exchangeTop'), JSON.stringify(ncclFirst));

// ------------------------------------------------------------ the puzzle
const book = edges('streams');
const rowIds = await page.$$eval(P + ' [data-edge]', (rs) => rs.map((r) => r.dataset.edge));
check('ten orderings, in the model\'s order', JSON.stringify(rowIds) === JSON.stringify(book.map((e) => e.id)), rowIds.join(' '));
const compareDisabled = await page.$$eval(P + ' [aria-label="Compare with"] button', (bs) => bs.map((b) => b.disabled));
check('the comparison is locked until all ten are placed', JSON.stringify(compareDisabled) === '[false,true,true]', JSON.stringify(compareDisabled));
const pickAndCheck = async (id, mech) => {
  await page.select(`${P} select[aria-label="Mechanism for ${id}"]`, mech);
  await pause(40);
  await page.click(`${P} button[aria-label="Check ${id}"]`);
  await pause(80);
  return page.$eval(`${P} [data-feedback="${id}"]`, (e) => e.textContent);
};
let fb = await pickAndCheck('reset-top', 'stream');
check('a wrong pick explains itself: stream order across two streams', fb.includes('not that') && fb.includes('different streams') && fb.includes('line 04'), fb);
fb = await pickAndCheck('copy-allreduce', 'blocking');
check('a wrong pick for the reduction says the host has to wait first', fb.includes('host has to wait first'), fb);
fb = await pickAndCheck('top-copy', 'host');
check('picking a host sync where Fig. 23.14 uses an event points at the lines', fb.includes('stop the host') && fb.includes('11 and 19'), fb);
for (const e of book) {
  fb = await pickAndCheck(e.id, e.mechanism);
  const solved = await page.$eval(`${P} [data-edge="${e.id}"]`, (r) => r.dataset.solved);
  const cite = /[,–]/.test(e.lines) ? `lines ${e.lines}` : `line ${e.lines}`;
  check(`${e.id}: ${e.mechanism} is accepted and ${cite} cited`, solved === 'true' && fb.includes(cite), fb);
  if (e.id === 'top-copy') {
    const hl = await page.$$eval(P + ' .code .ln.hl', (ls) => ls.map((l) => Number(l.dataset.n)));
    check('checking a row highlights the lines it cites (11 and 19)', JSON.stringify(hl) === '[11,19]', JSON.stringify(hl));
  }
}
check('the score reaches 10 of 10', (await page.$eval(P + ' [data-puzzle-score]', (e) => e.textContent.trim())) === '10 of 10 placed');
check('with all ten placed, a summary appears', !!(await page.$(P + ' [data-puzzle-done]')));
for (const v of ['nccl', 'nvshmem']) {
  await page.evaluate((P, v) => [...document.querySelectorAll(P + ' [aria-label="Compare with"] button')][v === 'nccl' ? 1 : 2].click(), P, v);
  await pause(120);
  const rows = await page.$$eval(P + ' [data-compare-row]', (rs) => rs.map((r) => ({ id: r.dataset.compareRow, m: r.dataset.mechanism, changed: r.dataset.changed === 'true' })));
  const want = edges(v);
  const ch = changedEdges('streams', v);
  check(`compare with ${v}: every row shows the model's mechanism`, rows.length === 10 && rows.every((r, k) => r.id === want[k].id && r.m === want[k].mechanism), JSON.stringify(rows));
  check(`compare with ${v}: exactly the changed rows are marked (${ch.length})`, JSON.stringify(rows.filter((r) => r.changed).map((r) => r.id)) === JSON.stringify(ch));
  const sum = await page.$eval(P + ' [data-compare-summary]', (e) => e.textContent);
  check(`compare with ${v}: the summary counts ${ch.length} changes`, sum.includes(`${ch.length} answers changed`), sum);
}
if (out) {
  await page.$eval(P, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-puzzle.png' });
}

// ------------------------------------------------------------ who waits
const readouts = () => page.$$eval(W + ' [data-readout]', (rs) => Object.fromEntries(rs.map((r) => [r.dataset.readout, r.dataset.value])));
const want = (r) => ({ iter: r2(r.iterTime), blocked: r2(r.hostBlocked), gap: r2(r.bottomGap), hidden: r.hidden ? 'yes' : 'no' });
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const clickSeg = async (group, nth) => {
  await page.evaluate((W, group, nth) => [...document.querySelectorAll(`${W} [aria-label="${group}"] button`)][nth].click(), W, group, nth);
  await pause(120);
};
const VERS = ['mpi', 'streams', 'nccl', 'nvshmem'];
const version = (v) => clickSeg('Version', VERS.indexOf(v));
const slide = async (label, v) => {
  await page.$eval(`${W} input[aria-label="${label}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(100);
};
const defect = async (d) => { await page.select(`${W} select[aria-label="Remove one ordering"]`, d); await pause(120); };
const drawn = (lane, kind) => page.$$eval(`${W} svg [data-lane="${lane}"][data-seg="${kind}"]`, (gs) => gs.map((g) => [g.dataset.iter, g.dataset.start, g.dataset.end]));
const problems = () => page.$$eval(W + ' [data-violation]', (ls) => ls.map((l) => [l.dataset.violation, l.textContent]));
const defaults = async () => { await page.evaluate((W) => [...document.querySelectorAll(W + ' button')].find((b) => b.textContent.trim() === 'Back to the defaults').click(), W); await pause(120); };

check('the explorer starts on Fig. 23.14 at the defaults, with the model\'s read-outs',
  (await page.$eval(W, (e) => e.dataset.version)) === 'streams' && same(await readouts(), want(simulate({ ...DEFAULTS, version: 'streams' }))), JSON.stringify(await readouts()));
for (const v of VERS) {
  await version(v);
  const m = simulate({ ...DEFAULTS, version: v });
  check(`${v}: read-outs match the model (${r2(m.iterTime)} per iteration, ${m.hidden ? 'hidden' : 'not hidden'})`, same(await readouts(), want(m)), JSON.stringify(await readouts()));
  const ex = await drawn('linkBottom', 'exchange');
  check(`${v}: the bottom exchanges are drawn where the model puts them`,
    same(ex, m.iters.map((it) => [String(it.iter), r2(it.exBottom.start), r2(it.exBottom.end)])), JSON.stringify(ex));
  const times = await page.$$eval(W + ' [data-version-time]', (ss) => ss.map((s) => s.dataset.value));
  check(`${v}: the comparison line lists all four versions' times`, same(times, VERS.map((x) => r2(simulate({ ...DEFAULTS, version: x }).iterTime))), JSON.stringify(times));
}
await version('streams');
const gapMark = await page.$$eval(W + ' svg [data-gap]', (gs) => gs.map((g) => g.dataset.gap));
check('Fig. 23.14: the bottom rows wait 4.02 for the top exchange, in both iterations (p. 562)', same(gapMark, ['4.02', '4.02']), JSON.stringify(gapMark));
await version('nccl');
check('NCCL: no gap is drawn', (await page.$$(W + ' svg [data-gap]')).length === 0);
const ncclTopStream = await drawn('top', 'exchange');
check('NCCL: the exchanges sit in the boundary streams', ncclTopStream.length === 2);
if (out) {
  await page.$eval(W, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-nccl.png' });
}

// Fig. 23.13: hidden as long as the exchange fits behind the internal kernel.
await slide('top exchange', 6);
await slide('bottom exchange', 6);
let m = simulate({ ...DEFAULTS, exTop: 6, exBottom: 6, version: 'nccl' });
check('NCCL, exchanges of 6: still hidden', (await readouts()).hidden === 'yes' && m.hidden && same(await readouts(), want(m)));
await slide('top exchange', 9);
await slide('bottom exchange', 9);
m = simulate({ ...DEFAULTS, exTop: 9, exBottom: 9, version: 'nccl' });
check('NCCL, exchanges of 9: not hidden, and the iteration grows', (await readouts()).hidden === 'no' && same(await readouts(), want(m)) && m.iterTime > 9, JSON.stringify(await readouts()));
await version('streams');
await slide('top exchange', 3);
await slide('bottom exchange', 3);
m = simulate({ ...DEFAULTS, exTop: 3, exBottom: 3, version: 'streams' });
check('Fig. 23.14, exchanges of 3: one after the other they no longer fit', (await readouts()).hidden === 'no' && same(await readouts(), want(m)));
await defaults();

// Priorities: internal kernel first.
await clickSeg('Launch order', 1);
await clickSeg("Boundary streams' priority", 0);
m = simulate({ ...DEFAULTS, version: 'streams', issue: 'internal-first', priority: 'equal' });
let pend = await drawn('top', 'pending');
check('internal first, equal priority: the top kernel waits for slots from 0.12 to 6.10 (after the last internal blocks)',
  same(await readouts(), want(m)) && pend.length >= 1 && pend[0][1] === '0.12' && pend[0][2] === r2(m.iters[0].top.start) && r2(m.iters[0].top.start) === '6.10', JSON.stringify(pend));
await clickSeg("Boundary streams' priority", 1);
m = simulate({ ...DEFAULTS, version: 'streams', issue: 'internal-first', priority: 'higher' });
pend = await drawn('top', 'pending');
check('internal first, higher priority: it starts at the first block retirement, 2.10, not at once',
  same(await readouts(), want(m)) && pend[0][2] === '2.10', JSON.stringify(pend));
if (out) {
  await page.$eval(W, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-priority.png' });
}
await defaults();

// The bug hunt.
await defect('sync-top');
m = simulate({ ...DEFAULTS, version: 'streams', defect: 'sync-top' });
let probs = await problems();
check('line 27 removed: the rows are sent before they are computed, as the model says', same(probs, m.violations.map((v) => [v.kind, v.text])) && probs.some(([k]) => k === 'send'), JSON.stringify(probs));
check('and it is marked on the timeline', (await page.$$(W + ' svg [data-fail="send"]')).length >= 1);
await defect('reset-waits');
check('lines 06–07 removed in Fig. 23.14: nothing fires, and the explorer says why', (await page.$(W + ' [data-no-problems]')) !== null && (await page.$eval(W + ' [data-quiet]', (e) => e.textContent)).includes('the reset always wins'));
await defect('compute-waits');
await clickSeg('Launch order', 1);
await clickSeg("Boundary streams' priority", 0);
m = simulate({ ...DEFAULTS, version: 'streams', defect: 'compute-waits', issue: 'internal-first', priority: 'equal' });
probs = await problems();
check('lines 19–20 removed, internal first, equal priority: the norm is copied before the boundary kernels added their parts',
  same(probs, m.violations.map((v) => [v.kind, v.text])) && probs.filter(([k]) => k === 'copy').length === 4, JSON.stringify(probs.map((p) => p[0])));
await defaults();
await defect('pageable');
m = simulate({ ...DEFAULTS, version: 'streams', defect: 'pageable' });
check('pageable l2norm_h: the host stalls at line 21 and the iteration grows', (await page.$eval(W + ' [data-problems]', (e) => e.textContent)).includes('No wrong result, but the host stalls')
  && same(await readouts(), want(m)) && m.iterTime > simulate({ ...DEFAULTS, version: 'streams' }).iterTime + 5);
await version('nccl');
check('switching to NCCL keeps an ordering that exists there (pageable)', (await page.$eval(`${W} select[aria-label="Remove one ordering"]`, (s) => s.value)) === 'pageable');
await defect('reset-waits');
await slide('top exchange', 2);
await slide('bottom exchange', 10);
m = simulate({ ...DEFAULTS, exTop: 2, exBottom: 10, version: 'nccl', defect: 'reset-waits' });
probs = await problems();
check('NCCL without lines 06–07, exchanges 2 and 10: a stale halo and a lost part of the norm', same(probs, m.violations.map((v) => [v.kind, v.text])) && probs.some(([k]) => k === 'reset') && probs.some(([k]) => k === 'halo'), JSON.stringify(probs.map((p) => p[0])));
await defect('none');
check('the same durations with the book\'s code: nothing broken', (await page.$(W + ' [data-no-problems]')) !== null);
await defect('exchange-waits');
probs = await problems();
check('NCCL without lines 43–44: the next top kernel reads its halo early', probs.length === 1 && probs[0][0] === 'halo' && probs[0][1].includes('top halo'), JSON.stringify(probs));
if (out) {
  await page.$eval(W, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-bug.png' });
}
await version('streams');
check('switching to Fig. 23.14 drops an ordering it does not have', (await page.$eval(`${W} select[aria-label="Remove one ordering"]`, (s) => s.value)) === 'none');
await defaults();
await version('nvshmem');
await defect('barrier');
check('NVSHMEM without the barrier, at the defaults: nothing fires yet', (await page.$(W + ' [data-no-problems]')) !== null);
await slide('bottom exchange', 10);
probs = await problems();
m = simulate({ ...DEFAULTS, exBottom: 10, version: 'nvshmem', defect: 'barrier' });
check('NVSHMEM without the barrier, a long exchange: the next kernel reads a halo still in flight', same(probs, m.violations.map((v) => [v.kind, v.text])) && probs[0][0] === 'halo');
await defaults();
await version('mpi');
const mpiState = await page.evaluate((W) => ({
  prio: [...document.querySelectorAll(`${W} [aria-label="Boundary streams' priority"] button`)].every((b) => b.disabled),
  opts: [...document.querySelector(`${W} select[aria-label="Remove one ordering"]`).options].map((o) => o.value),
}), W);
check('MPI without overlap: one stream, so no priority, and nothing to remove', mpiState.prio && same(mpiState.opts, ['none']), JSON.stringify(mpiState));
await version('streams');

// ------------------------------------------------------------ the prediction
const pq = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Predict first') && e.textContent.includes('still hide behind the internal kernel')));
await (await pq.$('input[type=number]')).type('5');
await (await pq.$('button.primary')).click();
await pause(200);
const nccl0 = simulate({ ...DEFAULTS, exTop: 0.5, exBottom: 0.5, version: 'nccl' }).iters[0];
const win = nccl0.internal.end - Math.max(nccl0.top.end, nccl0.bottom.end);
const shown = await pq.evaluate((e) => e.querySelector('[data-window]')?.dataset.window ?? '');
check(`the prediction reveals the model's window, ${r2(win)}`, shown === r2(win) && r2(win) === '6.00', shown);

// ------------------------------------------------------------ questions
const question = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const qText = (h) => h.evaluate((e) => e.textContent);
const answerNum = async (needle, v) => {
  const qq = await question(needle);
  await (await qq.$('input[type=number]')).type(String(v));
  await (await qq.$('button.primary')).click();
  await pause(120);
  return qText(qq);
};
const answerChoice = async (needle, idx) => {
  const qq = await question(needle);
  const radios = await qq.$$('input[type=radio]');
  await radios[idx].click();
  await (await qq.$('button.primary')).click();
  await pause(120);
  return qText(qq);
};
check('q-threerows: row 8', (await answerNum('which row of the local', 8)).includes('Correct.'));
check('q-prio: a wrong pick (preemption) is marked wrong', (await answerChoice('When do they start?', 1)).includes('Not quite.'));
check('q-prio: first claim on freed slots', (await answerChoice('When do they start?', 0)).includes('Correct.'));
check('q-memset: the legacy default stream', (await answerChoice('actually do at line 04', 0)).includes('Correct.'));
check('q-mechanism: an event and a stream wait', (await answerChoice('why is the order of the host', 0)).includes('Correct.'));
check('q-nvshmem-barrier: puts may land later', (await answerChoice('still needed at line 18', 0)).includes('Correct.'));
await pause(600);
check('answers are written to the progress file', puts.some((p) => p.answers?.['ch23/23.4/q-threerows']?.correct === true && p.answers?.['ch23/23.4/p-window']));

// ------------------------------------------------------------ light theme, narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
if (out) {
  await page.$eval(W, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-light.png' });
}
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
const narrow = await page.evaluate((W) => ({
  page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1,
  bench: document.querySelector(W).getBoundingClientRect().right <= document.documentElement.clientWidth + 1,
}), W);
check('at 390 px nothing makes the page scroll sideways', narrow.page && narrow.bench, JSON.stringify(narrow));
if (out) {
  await page.$eval(W, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (u) => /\/content\/ch23\/narration\/(manifest\.json|[a-z-]+-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
