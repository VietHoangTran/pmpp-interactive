// Browser check for section 4.8, querying device properties: the corrected query loop
// (a lab snippet), the erratum hunt on the field list of p. 89, the table "Your device,
// field by field" against this machine's reports, the launch explorer against the
// model (src/sim/launchconfig.ts, imported here directly), the Metal excerpt and
// recipe, five questions, and the page at a narrow width.
// Usage: node e2e/ch04-s48.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { BOOK_LAUNCH, RULES, appleRecipe, checkLaunch, launchLimitsFromReport, suggest } from '../src/sim/launchconfig.ts';
import { occupancy, pct } from '../src/sim/occupancy.ts';

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
const H = 'section.bench[aria-label="The field list on p. 89"]';
const T = 'section.bench[aria-label="Your device, field by field"]';
const L = 'section.bench[aria-label="Derive a launch configuration"]';
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const attr = (sel, a) => page.$eval(sel, (e, a) => e.getAttribute(a), a);
const click = async (sel) => {
  await page.$eval(sel, (e) => e.click());
  await pause(100);
};
const setInput = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(80);
};
const setSelect = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, v);
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(100);
};
const q = (words) => page.evaluateHandle((w) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(w)), words);
const answerChoice = async (words, optionWords) => {
  const h = await q(words);
  const picked = await h.evaluate((e, w) => {
    const l = [...e.querySelectorAll('.opts label')].find((x) => x.textContent.trim().includes(w));
    if (!l) return false;
    l.querySelector('input').click();
    return true;
  }, optionWords);
  await pause(60);
  await page.evaluate((e) => e.querySelector('button.primary').click(), h);
  await pause(120);
  return { picked, text: await h.evaluate((e) => e.textContent) };
};

const reports = await (await fetch(BASE + '/api/reports')).json();
const synthetic = (b) => b === 'emu' || b === 'race' || String(reports[b].device.note ?? '').includes('synthetic');
const gpus = Object.keys(reports).filter((b) => !synthetic(b)).sort();
const queried = (b) => {
  const run = reports[b];
  if (!run) return null;
  const rows = (variant) => run.results.filter((x) => x.lab === 'ch04/lab1_launchconfig' && x.variant === variant).flatMap((x) => x.bench);
  const workPass = run.work_state?.['ch04/lab1_launchconfig'] !== 'untouched' && run.results.some((x) => x.lab === 'ch04/lab1_launchconfig' && x.variant === 'work' && x.tests.length && x.built && x.test_rc === 0 && x.tests.every((t) => t.pass));
  return (workPass ? rows('work').find((r) => r.name === 'queried') : null) ?? rows('solution').find((r) => r.name === 'queried') ?? null;
};
const solutionPasses = (b) => {
  const rs = reports[b]?.results.filter((x) => x.lab === 'ch04/lab1_launchconfig' && x.variant === 'solution' && x.tests.length) ?? [];
  return rs.length > 0 && rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass));
};

await goto(page, 'ch04/4.8');
await page.waitForSelector(H);
await page.waitForSelector(L);
await page.waitForFunction(() => [...document.querySelectorAll('figure.code pre')].some((p) => p.textContent.includes('describeDevices')));

// ---- margin notes and the corrected loop
const notes = await page.$$eval('aside.note', (a) => a.map((n) => n.textContent.replace(/\s+/g, ' ').trim()));
check('book claims carry their pages (p. 87, p. 88, p. 89)', ['§4.8, p. 87', '§4.8, p. 88', '§4.8, p. 89'].every((p) => notes.some((n) => n.startsWith('Book ' + p))), notes.filter((n) => n.startsWith('Book')).join(' | '));
check('vendor facts are dated with their items (A12, A15, A16, A17, A18; B9; C1, C5, C6; D1, D2, D4)', ['A15', 'A17', 'A18', 'B9', 'C6', 'D4'].every((k) => notes.some((n) => n.includes(k))), notes.join(' | '));
check('the integrated-GPU reason is flagged unverified, not repeated', notes.some((n) => n.includes('integrated GPUs') && n.includes('unverified')));
const codes = await page.$$eval('figure.code', (fs) => fs.map((f) => ({ code: f.querySelector('pre').textContent, status: f.querySelector('.status')?.textContent ?? '' })));
const loop = codes.find((c) => c.code.includes('describeDevices'));
check('the corrected loop is the snippet file: int index, both return codes checked, no clock rate', !!loop && loop.code.includes('for (int i = 0; i < devCount; i++)') && loop.code.includes('!= cudaSuccess) return 0') && !loop.code.includes('clockRate') && loop.status === 'front-end-checked; unverified on CUDA hardware', JSON.stringify(loop?.status));

// ---- the erratum hunt
const hunt = () => page.evaluate((H) => ({
  found: Number(document.querySelector(H + ' [data-out=hunt]').dataset.found),
  panel: document.querySelector(H + ' [data-out=hunt]').textContent.replace(/\s+/g, ' ').trim(),
  states: Object.fromEntries([...document.querySelectorAll(H + ' [data-line]')].map((l) => [l.dataset.line, l.dataset.state])),
}), H);
const lines = await page.$$eval(`${H} [data-line]`, (ls) => ls.map((l) => l.textContent));
check('the listing has the book\'s loop and the seven fields of p. 89', lines.length === 13 && lines[3].includes('for(unsigned int i = 0; i < devCount; i++) {') && ['maxThreadsPerBlock', 'multiProcessorCount', 'clockRate', 'maxThreadsDim[0]', 'maxGridSize[0]', 'regsPerBlock', 'warpSize'].every((f, i) => lines[5 + i].includes('devProp.' + f)), lines.join('\n'));
await click(`${H} [data-line="4"]`);
let hs = await hunt();
check('question A, line 4: not it, and the reason names the signed/unsigned comparison', hs.found === 0 && hs.states['4'] === 'wrong' && hs.panel.includes('not line 4') && hs.panel.includes('unsigned index'), hs.panel);
await click(`${H} [data-line="8"]`);
hs = await hunt();
check('question A, line 8: found; the reveal gives cudaDeviceGetAttribute with cudaDevAttrClockRate', hs.found === 1 && hs.states['8'] === 'right' && hs.panel.includes('removed from cudaDeviceProp in CUDA 13.0') && hs.panel.includes('cudaDevAttrClockRate'), hs.panel);
await click(`${H} button[data-hunt="describe"]`);
await click(`${H} [data-line="11"]`);
hs = await hunt();
check('question B, line 11: found; the reveal points at regsPerMultiprocessor', hs.found === 2 && hs.states['11'] === 'right' && hs.panel.includes('regsPerMultiprocessor'), hs.panel);
await click(`${H} [data-line="6"]`);
hs = await hunt();
check('question B, a wrong line afterwards is marked wrong and explained', hs.states['6'] === 'wrong' && hs.panel.includes('not line 6') && hs.panel.includes('The comment is what the field means.'), hs.panel);
await page.$eval(`${H} [data-line="12"]`, (e) => e.focus());
await page.keyboard.press('Enter');
await pause(100);
hs = await hunt();
check('a line can be chosen from the keyboard', hs.states['12'] === 'wrong', JSON.stringify(hs.states));
if (out) await page.$eval(H, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-hunt.png' });

// ---- your device, field by field
const table = () => page.evaluate((T) => {
  const t = document.querySelector(T + ' table');
  return {
    family: t.dataset.family,
    device: t.dataset.device,
    none: Number(document.querySelector(T + ' [data-out=field-summary]').dataset.none),
    rows: Object.fromEntries([...t.querySelectorAll('tr[data-field]')].map((r) => [r.dataset.field, {
      kind: r.dataset.kind,
      value: r.querySelector('[data-value]')?.dataset.value ?? null,
      lab: r.querySelector('[data-lab]')?.dataset.lab ?? null,
    }])),
  };
}, T);
const buttons = await page.$$eval(`${T} button[data-device]`, (bs) => bs.map((b) => b.dataset.device));
check('the table offers this machine\'s GPU backends and the book, never an emulator', JSON.stringify(buttons) === JSON.stringify([...gpus, 'book']), JSON.stringify(buttons));
let tb = await table();
if (gpus.includes('metal')) {
  await click(`${T} button[data-device="metal"]`);
  tb = await table();
  const d = reports.metal.device;
  const qr = queried('metal');
  check('Metal: 9 of 14 rows have no counterpart; width and threadgroup limit are pipeline rows', tb.family === 'metal' && tb.none === 9 && tb.rows.width.kind === 'pipeline' && tb.rows.perblock.kind === 'pipeline' && tb.rows.threadslots.kind === 'none' && tb.rows.clock.kind === 'none', JSON.stringify(tb));
  check('Metal: device values are the report\'s, never filled in', tb.rows.name.value === String(d.name) && tb.rows.dims.value === `${d['maxThreadsPerThreadgroup.width']} × ${d['maxThreadsPerThreadgroup.height']} × ${d['maxThreadsPerThreadgroup.depth']}` && tb.rows.sharedblock.value === String(d.maxThreadgroupMemoryLength) && tb.rows.width.value === 'none', JSON.stringify(tb.rows));
  if (qr) check('Metal: lab 4.1\'s query column shows the pipeline width and limit it read', tb.rows.width.lab === String(qr.threadExecutionWidth) && tb.rows.perblock.lab === String(qr.maxTotalThreadsPerThreadgroup), JSON.stringify([tb.rows.width, qr.threadExecutionWidth]));
  else check('Metal: without lab 4.1\'s run there is no query column', tb.rows.width.lab === null);
}
for (const b of gpus.filter((x) => x !== 'metal')) {
  await click(`${T} button[data-device="${b}"]`);
  tb = await table();
  const d = reports[b].device;
  check(`${b}: warp width and thread slots are the report's values`, tb.rows.width.value === String(d.warpSize ?? 'none') && tb.rows.threadslots.value === String(d.maxThreadsPerMultiProcessor ?? 'none'), JSON.stringify(tb.rows));
}
await click(`${T} button[data-device="book"]`);
tb = await table();
check("the book's example: its figures with pages, and none where the book gives none", tb.family === 'cuda' && tb.rows.width.value === '32 (p. 74)' && tb.rows.threadslots.value === '2048 (p. 85)' && tb.rows.regssm.value === '65,536 (p. 86)' && tb.rows.regsblock.value === 'none' && tb.rows.grid.value === 'none', JSON.stringify(tb.rows));
if (out) await page.$eval(T, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-fields.png' });

// ---- the Metal excerpt and recipe
const mm = codes.find((c) => c.code.includes('part 1: ask'));
check("the Metal excerpt is lab 4.1's part 1 only, with a status read from this machine's run", !!mm && !mm.code.includes('part 2') && mm.status === (solutionPasses('metal') ? 'verified on Metal' : 'unverified: no passing Metal run on this machine'), JSON.stringify(mm?.status));
const qm = queried('metal');
const shape = qm ? appleRecipe(qm.threadExecutionWidth, qm.maxTotalThreadsPerThreadgroup) : null;
check("Apple's recipe is applied to the pipeline's own numbers, or not at all", (await attr('[data-out=metal-recipe]', 'data-shape')) === (shape ? `${shape.x}x${shape.y}` : 'none'), await attr('[data-out=metal-recipe]', 'data-shape'));

// ---- the launch explorer
const launch = () => page.evaluate((L) => {
  const r = document.querySelector(L + ' [data-out=rules]');
  const n = (s) => Number(document.querySelector(L + ' [data-out=' + s + ']').textContent.replace(/,/g, ''));
  return {
    ok: r.dataset.ok, failed: r.dataset.failed, skipped: r.dataset.skipped,
    names: [...r.querySelectorAll('tr[data-rule]')].map((t) => t.children[1].textContent.trim()),
    tpb: n('threads-per-block'), launched: n('launched'), working: n('working'), surplus: n('surplus'),
    grid: document.querySelector(L + ' [data-out=grid]').textContent.trim(),
    occ: document.querySelector(L + ' [data-out=launch-occupancy]').dataset.value,
    suggestion: document.querySelector(L + ' [data-out=suggestion]')?.textContent.replace(/\s+/g, ' ').trim() ?? null,
  };
}, L);
const wantLaunch = (limits, rows, cols, x, y) => {
  const c = checkLaunch(limits, rows, cols, { x, y });
  const o = occupancy({ threadSlots: limits.threadSlots, blockSlots: limits.blockSlots, blockSlotsDerived: limits.blockSlotsDerived, regs: null, shared: null, width: limits.width, maxThreadsPerBlock: limits.maxThreadsPerBlock }, { threads: c.threadsPerBlock, regs: 0, shared: 0 });
  return { ok: c.ok ? 'yes' : 'no', failed: c.failed.join(','), skipped: c.skipped.join(','), tpb: c.threadsPerBlock, launched: c.launched, working: c.working, surplus: c.surplus, occ: o.usable ? String(pct(o.byWarps)) : 'n/a' };
};
const agrees = (s, m) => Object.keys(m).every((k) => String(s[k]) === String(m[k]));
const shape2 = async (x, y) => {
  await setInput(`${L} input[aria-label="x, exactly"]`, x);
  await setInput(`${L} input[aria-label="y, exactly"]`, y);
};
let ls = await launch();
let lm = wantLaunch(BOOK_LAUNCH, 1, 4096, 16, 16);
check('the explorer starts at the habit: batch 1, 16 × 16 breaks only "no taller than the rows"', agrees(ls, lm) && ls.failed === 'rows' && ls.surplus === 61440, JSON.stringify({ ls, lm }));
check('it lists the rules by the names lab 4.1 prints', JSON.stringify(ls.names) === JSON.stringify(RULES.map((r) => r.name)), JSON.stringify(ls.names));
await click(`${L} button[data-act="suggest"]`);
ls = await launch();
let sg = suggest(BOOK_LAUNCH, 1, 4096);
lm = wantLaunch(BOOK_LAUNCH, 1, 4096, sg.x, sg.y);
check('"Suggest one" at batch 1 gives the model\'s 64 × 1: every rule holds, no surplus thread', agrees(ls, lm) && sg.x === 64 && sg.y === 1 && ls.ok === 'yes' && ls.surplus === 0 && ls.suggestion.startsWith('Set to 64 × 1'), JSON.stringify({ ls, lm }));
if (out) await page.$eval(L, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-launch-suggested.png' });
await click(`${L} button[data-problem="64x4096"]`);
check('choosing another problem clears the old suggestion', (await launch()).suggestion === null);
await click(`${L} button[data-act="suggest"]`);
ls = await launch();
sg = suggest(BOOK_LAUNCH, 64, 4096);
check('at batch 64 the suggestion is the model\'s 32 × 32', sg.x === 32 && sg.y === 32 && ls.suggestion.startsWith('Set to 32 × 32') && agrees(ls, wantLaunch(BOOK_LAUNCH, 64, 4096, 32, 32)), JSON.stringify(ls));
await shape2(16, 16);
ls = await launch();
check('and at batch 64, 16 × 16 is fine too', agrees(ls, wantLaunch(BOOK_LAUNCH, 64, 4096, 16, 16)) && ls.ok === 'yes', JSON.stringify(ls));
await shape2(32, 24);
ls = await launch();
check('768 threads per block fail "divides the thread slots" on the book\'s limits, 75% by slots', agrees(ls, wantLaunch(BOOK_LAUNCH, 64, 4096, 32, 24)) && ls.failed === 'divides' && ls.occ === '75', JSON.stringify(ls));
await shape2(64, 32);
ls = await launch();
check('2048 threads per block fail "within maxThreadsPerBlock"', agrees(ls, wantLaunch(BOOK_LAUNCH, 64, 4096, 64, 32)) && ls.failed.split(',').includes('per-block'), JSON.stringify(ls));

const T1536 = { ...BOOK_LAUNCH, threadSlots: 1536, blockSlots: 16 };
await setSelect(`${L} select[aria-label="Launch limits"]`, 't1536');
await click(`${L} button[data-problem="1x4096"]`);
await shape2(64, 1);
ls = await launch();
check('on 1536 / 16 limits, 64 × 1 at batch 1 runs out of block slots', agrees(ls, wantLaunch(T1536, 1, 4096, 64, 1)) && ls.failed === 'block-slots', JSON.stringify(ls));
await click(`${L} button[data-act="suggest"]`);
ls = await launch();
sg = suggest(T1536, 1, 4096);
check('and the suggestion there is 96 × 1', sg.x === 96 && sg.y === 1 && ls.suggestion.startsWith('Set to 96 × 1') && ls.ok === 'yes', JSON.stringify(ls));

const W64 = { ...BOOK_LAUNCH, width: 64, maxThreadsPerBlock: 256, maxThreadsDim: [256, 256, 256] };
await setSelect(`${L} select[aria-label="Launch limits"]`, 'w64');
await shape2(32, 1);
ls = await launch();
check('on the 64-wide test device, 32 × 1 is half a warp and too small to fill the slots', agrees(ls, wantLaunch(W64, 1, 4096, 32, 1)) && ls.failed === 'warps,block-slots', JSON.stringify(ls));
await click(`${L} button[data-act="suggest"]`);
ls = await launch();
sg = suggest(W64, 1, 4096);
check('and the suggestion there is the model\'s, in whole 64-wide warps', ls.suggestion.startsWith(`Set to ${sg.x} × ${sg.y}`) && sg.x % 64 === 0 && ls.ok === 'yes', JSON.stringify(ls));

await setSelect(`${L} select[aria-label="Launch limits"]`, 'custom');
await setInput(`${L} input[aria-label="Thread slots per SM"]`, 0);
await shape2(64, 1);
ls = await launch();
check('custom limits without thread slots: the slot rules are skipped and no occupancy is stated', ls.skipped === 'divides,block-slots' && ls.occ === 'n/a' && ls.ok === 'yes', JSON.stringify(ls));
for (const b of Object.keys(reports).sort()) {
  const lim = launchLimitsFromReport(reports[b].device, b);
  const offered = await page.$$eval(`${L} select[aria-label="Launch limits"] option`, (os) => os.map((o) => o.value));
  check(`the ${b} report is ${lim ? 'offered' : 'not offered'} as launch limits`, offered.includes(b) === !!lim, JSON.stringify(offered));
}

// ---- questions
let a = await answerChoice('A student says they multiplied two 1024', 'It cannot have run as described');
check('q-ex9: the launch is invalid', a.picked && a.text.includes('Correct.'));
a = await answerChoice('registers per thread that still allows full occupancy', 'regsPerBlock, as the book');
check('q-regsfield: the book\'s regsPerBlock is marked wrong, with the reason', a.picked && a.text.includes('Not quite.') && a.text.includes('most one block may hold'), a.text.slice(0, 300));
a = await answerChoice('registers per thread that still allows full occupancy', 'regsPerMultiprocessor');
check('q-regsfield: regsPerMultiprocessor is right', a.picked && a.text.includes('Correct.'));
a = await answerChoice('Why can that report not', 'on a compiled pipeline');
check('q-metalwidth: the width belongs to a compiled pipeline', a.picked && a.text.includes('Correct.'));
a = await answerChoice('reports compute capability 12.0', 'Nothing, without querying');
check('q-cc: nothing without querying', a.picked && a.text.includes('Correct.'));
a = await answerChoice('maxThreadsPerMultiProcessor = 0', 'Treat the field as not reported');
check('q-zero: treat 0 as not reported', a.picked && a.text.includes('Correct.'));

// ---- narrow screen
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((L) => {
  const b = document.querySelector(L).getBoundingClientRect();
  return { page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, select: [...document.querySelectorAll(L + ' select')].every((s) => s.getBoundingClientRect().right <= b.right + 1) };
}, L);
check('at a narrow width nothing overflows, and the limits list fits its bench', narrow.page && narrow.select, JSON.stringify(narrow));
if (out) await page.$eval(L, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-narrow.png' });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
