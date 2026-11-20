// Browser check for section C.6 (Unified memory): the "Fig. C.1, line by line" stepper in
// every system and with both switches, the totals table, the prediction, two questions,
// light and dark themes and a narrow screen. Every number is compared with the model
// (src/sim/managedmem.ts, imported here directly).
// Usage: APP_URL=http://localhost:5702 node e2e/appC-sc6.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { FIG_HEAD, figLines, moved, runFig, steppable } from '../src/sim/managedmem.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1100 });
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
const B = 'section.bench[aria-label="Fig. C.1, line by line"]';
const MODES = { limited: 'bulk', full: 'migrate', apple: 'one' };

const press = async (label) => {
  const ok = await page.evaluate((B, label) => {
    const btn = [...document.querySelectorAll(B + ' button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, B, label);
  await pause(50);
  return ok;
};
const setMode = async (mode) => {
  await page.select(B + ' select[aria-label="System"]', mode);
  await pause(80);
};
const setBox = async (word, on) => {
  await page.evaluate((B, word, on) => {
    const box = [...document.querySelectorAll(B + ' label.check')].find((l) => l.textContent.includes(word)).querySelector('input');
    if (box.checked !== on) box.click();
  }, B, word, on);
  await pause(80);
};
const program = async (name) => {
  await page.evaluate((B, name) => [...document.querySelectorAll(B + ' .seg button')].find((b) => b.textContent.trim().startsWith(name)).click(), B, name);
  await pause(80);
};
const state = () =>
  page.evaluate((B) => {
    const root = document.querySelector(B);
    const cells = {};
    for (const c of root.querySelectorAll('[data-box][data-page]')) {
      const k = `${c.dataset.box}:${c.dataset.buf}`;
      (cells[k] ??= []).push(c.dataset.present === 'yes' ? 1 : 0);
    }
    const counter = Object.fromEntries([...root.querySelectorAll('[data-counter]')].map((c) => [c.dataset.counter, Number(c.querySelector('[data-value]').textContent)]));
    const totals = Object.fromEntries([...root.querySelectorAll('tr[data-program]')].map((r) => [r.dataset.program, {
      cells: [...r.querySelectorAll('td.num')].map((t) => Number(t.textContent)),
      outcome: r.querySelector('[data-outcome]').dataset.outcome,
    }]));
    return {
      narrate: root.querySelector('.narrate').textContent.replace(/\s+/g, ' ').trim(),
      current: root.querySelector('[data-current="yes"]')?.textContent.trim() ?? null,
      memories: [...root.querySelectorAll('[data-memory]')].map((m) => m.dataset.memory),
      cells,
      counter,
      totals,
      status: root.querySelector('[data-status]')?.textContent.trim() ?? '',
    };
  }, B);
/** The model's expected cells for a view: per memory box and buffer, which pages are present. */
const expectCells = (view, mode) => {
  const res = {};
  for (const [b, v] of Object.entries(view)) {
    const boxes = mode === 'one' ? ['one'] : v.kind === 'managed' ? ['cpu', 'gpu'] : [v.kind === 'host' ? 'cpu' : 'gpu'];
    for (const box of boxes) res[`${box}:${b}`] = v.pages.map((p) => (box === 'one' ? p.loc === 'one' : p.loc === box || p.loc === 'both') ? 1 : 0);
  }
  return res;
};
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

await goto(page, 'appC/C.6');
await page.waitForSelector(B);
await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));

// ---- the listings are the model's lines (which the unit test ties to the snippet files)
const listing = await page.$$eval(B + ' [data-listing]', (cols) =>
  Object.fromEntries(cols.map((c) => [c.dataset.listing, [...c.querySelectorAll('[data-line]')].map((l) => ({ i: Number(l.dataset.line), text: l.textContent.trim(), changed: !!l.dataset.changed }))])),
);
check('three listings, in Fig. C.1\'s order then ours', same(Object.keys(listing), ['cpu', 'managed', 'explicit']), Object.keys(listing).join(','));
for (const p of ['cpu', 'managed', 'explicit']) {
  const model = figLines(p).map((l, i) => ({ i, text: l.text.trim(), changed: l.changed })).filter((l) => l.text !== '' || p === 'cpu');
  const shown = listing[p].map((l) => ({ ...l, text: l.text.trim() }));
  check(`the ${p} listing shows the model's lines in order, with Fig. C.1's changes marked`, same(shown.map((l) => [l.i, l.text, l.changed]), model.map((l) => [l.i, l.text, l.changed])), JSON.stringify(shown));
}
check('each listing opens with sortfile\'s signature', (await page.$eval(B, (e) => e.textContent)).split(FIG_HEAD).length === 4);

// ---- full mode, managed, step by step against the model
await setMode('migrate');
await program('managed');
let fr = runFig('managed', 'migrate');
let order = steppable(fr.lines);
let s = await state();
check('before the first step: nothing has run, both memories empty', s.narrate.startsWith('Nothing has run') && same(s.memories, ['cpu', 'gpu']));
let allMatch = true;
for (let k = 0; k < order.length; k++) {
  await press('Step');
  s = await state();
  const line = fr.lines[order[k]];
  let last = -1;
  for (let j = 0; j <= k; j++) if (fr.stepOf[order[j]] >= 0) last = fr.stepOf[order[j]];
  const view = last >= 0 ? fr.run.steps[last].view : fr.run.initial;
  const c = last >= 0 ? fr.run.steps[last].counters : { h2d: 0, d2h: 0, gpuFaults: 0, cpuFaults: 0 };
  const okCells = same(s.cells, expectCells(view, 'migrate'));
  const okCount = s.counter['pages to GPU'] === c.h2d && s.counter['pages to CPU'] === c.d2h && s.counter['GPU faults'] === c.gpuFaults && s.counter['CPU faults'] === c.cpuFaults;
  const okLine = s.current === line.text.trim() && s.narrate.startsWith(line.text.trim());
  if (!(okCells && okCount && okLine)) {
    allMatch = false;
    check(`full mode, line ${k + 1} (${line.text.trim()})`, false, JSON.stringify({ s, want: { cells: expectCells(view, 'migrate'), c } }));
  }
  if (line.role === 'read-file') check('after fread every page is in CPU memory', same(s.cells['cpu:data'], [1, 1, 1, 1, 1, 1, 1, 1]) && s.narrate.includes('8 pages placed in CPU memory'), s.narrate);
  if (line.role === 'sort') {
    check('after the launch: 8 GPU faults, 8 pages moved, all pages in GPU memory, kernel running',
      s.narrate.includes('8 GPU faults, 8 pages moved') && same(s.cells['gpu:data'], [1, 1, 1, 1, 1, 1, 1, 1]) && s.status === 'kernel may be running', JSON.stringify(s));
    if (out) await page.screenshot({ path: out + '-full-launch.png' });
  }
  if (line.role === 'use') check('use_data: 8 CPU faults bring every page back', s.narrate.includes('8 CPU faults, 8 pages moved') && same(s.cells['cpu:data'], [1, 1, 1, 1, 1, 1, 1, 1]), s.narrate);
}
check('every line of the managed port in full mode matches the model (cells, counters, highlighted line)', allMatch);
check('at the end: 8 to GPU, 8 to CPU, 8 GPU and 8 CPU faults', s.counter['pages to GPU'] === 8 && s.counter['pages to CPU'] === 8 && s.counter['GPU faults'] === 8 && s.counter['CPU faults'] === 8, JSON.stringify(s.counter));
check('Step is disabled at the end', !(await press('Step')));

// ---- the totals table, every mode and switch, against the model
const totalsMatch = async (label, mode, opts) => {
  const t = (await state()).totals;
  const bad = ['cpu', 'managed', 'explicit'].filter((p) => {
    const r = runFig(p, mode, opts).run;
    const want = [r.counters.h2d, r.counters.d2h, r.counters.gpuFaults, r.counters.cpuFaults, ...(mode === 'one' ? [r.counters.copiedInMemory] : [])];
    return !same(t[p].cells, want) || t[p].outcome !== (r.errors[0]?.kind ?? 'ok');
  });
  check(`totals table, ${label}`, bad.length === 0, JSON.stringify({ shown: t, bad }));
  return t;
};
let t = await totalsMatch('full', 'migrate', {});
check('full, kernel touches everything: managed moves as many pages as the explicit copies (16)', moved({ h2d: t.managed.cells[0], d2h: t.managed.cells[1] }) === 16 && t.explicit.cells[0] + t.explicit.cells[1] === 16);
await setBox('first half', true);
t = await totalsMatch('full, half touched', 'migrate', { half: true });
check('full, half touched: managed moves 4 and 4, the copies still 8 and 8', same(t.managed.cells.slice(0, 2), [4, 4]) && same(t.explicit.cells.slice(0, 2), [8, 8]));
await setMode('bulk');
t = await totalsMatch('limited, half touched', 'bulk', { half: true });
check('limited, half touched: managed still moves all 8 each way', same(t.managed.cells.slice(0, 2), [8, 8]));
// Step to the launch in limited mode: a bulk move of all 8 pages though the kernel touches 4.
await program('managed');
for (let k = 0; k < 4; k++) await press('Step');
s = await state();
check('limited launch: every page moves before the kernel runs, whatever it touches', s.narrate.includes('8 pages, whatever the kernel touches') && same(s.cells['gpu:data'], [1, 1, 1, 1, 1, 1, 1, 1]), s.narrate);
await setBox('first half', false);
await totalsMatch('limited', 'bulk', {});

// ---- the bug hunt: delete the synchronise
await setBox('delete', true);
t = await totalsMatch('limited, synchronise deleted', 'bulk', { noSync: true });
check('limited without the synchronise: a segmentation fault', t.managed.outcome === 'segfault');
const struck = await page.$eval(B + ' [data-listing="managed"] [data-deleted="yes"]', (e) => e.textContent.trim()).catch(() => null);
check('the deleted line is shown struck out', struck === 'cudaDeviceSynchronize();', String(struck));
fr = runFig('managed', 'bulk', { noSync: true });
order = steppable(fr.lines);
check('the deleted line is not stepped', order.length === 6);
await program('managed');
await press('Run to the end');
await press('Back');
s = await state();
check('stepping to use_data: the narration reports the segmentation fault', s.current === 'use_data(data);' && s.narrate.includes('Segmentation fault'), s.narrate);
check('the narration box is marked as a failure', await page.$eval(B + ' .narrate', (e) => e.classList.contains('bad')));
await press('Step');
s = await state();
check('after the crash, cudaFree never runs and the program is stopped', s.narrate.includes('never runs') && s.status === 'program stopped', JSON.stringify([s.narrate, s.status]));
if (out) await page.screenshot({ path: out + '-limited-nosync.png' });

await setMode('migrate');
t = await totalsMatch('full, synchronise deleted', 'migrate', { noSync: true });
await press('Run to the end');
await press('Back');
s = await state();
check('full without the synchronise: no crash, a race on pages 0–7', t.managed.outcome === 'race' && s.narrate.includes('A race.') && s.narrate.includes('pages 0–7'), s.narrate);

await setMode('one');
t = await totalsMatch('Apple silicon, synchronise deleted', 'one', { noSync: true });
await press('Run to the end');
await press('Back');
s = await state();
check('Apple silicon without the wait: an ordering error, not a crash', t.managed.outcome === 'ordering' && s.narrate.includes('An ordering error.'), s.narrate);
await setBox('delete', false);

// ---- Apple silicon: one memory, nothing moves
t = await totalsMatch('Apple silicon', 'one', {});
s = await state();
check('Apple silicon draws one memory', same(s.memories, ['one']), JSON.stringify(s.memories));
check('Apple silicon: the managed port moves nothing; the explicit copies copy 16 pages inside memory', t.managed.cells[0] + t.managed.cells[1] === 0 && t.explicit.cells[4] === 16, JSON.stringify(t));
await press('Restart');
await press('Step');
await press('Step');
s = await state();
check('Apple silicon: the allocation line names the shared-storage buffer', s.narrate.includes('MTLResourceStorageModeShared'), s.narrate);
await press('Run to the end');
s = await state();
check('Apple silicon at the end: zero pages moved, every page in the one memory until the free', s.counter['pages to GPU'] === 0 && s.counter['pages to CPU'] === 0, JSON.stringify(s.counter));
if (out) await page.screenshot({ path: out + '-apple.png' });

// ---- the explicit port in full mode
await setMode('migrate');
await program('explicit');
await press('Run to the end');
s = await state();
check('explicit copies: two copies of 8 pages, no faults', s.counter['pages to GPU'] === 8 && s.counter['pages to CPU'] === 8 && s.counter['GPU faults'] === 0 && s.counter['CPU faults'] === 0, JSON.stringify(s.counter));
check('explicit copies: host buffer in CPU memory, device buffer in GPU memory', 'cpu:data' in s.cells && 'gpu:d_data' in s.cells && !('gpu:data' in s.cells));

// ---- the prediction and a question
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Predict first') && e.textContent.includes('page crossings')));
await (await pred.$('input[type=number]')).type('8');
await (await pred.$('button.primary')).click();
await pause(150);
const reveal = await pred.evaluate((e) => e.textContent);
const m = moved(runFig('managed', 'migrate').run.counters);
check('the prediction is committed and the reveal gives the model\'s 16 and the copies\' 16', reveal.includes('You predicted 8') && reveal.includes(`The model says ${m}`) && reveal.includes(`moves ${m}`), reveal);

const q = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('how many bytes cross the link')));
await q.evaluate((e) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes('About the same')).querySelector('input').click());
await (await q.$('button.primary')).click();
await pause(120);
check('q-same-bytes: the right answer is accepted', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
const q2 = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Delete cudaDeviceSynchronize()')));
await q2.evaluate((e) => [...e.querySelectorAll('label')].find((l) => l.textContent.startsWith('Both work')).querySelector('input').click());
await (await q2.$('button.primary')).click();
await pause(120);
const v2 = await q2.evaluate((e) => e.textContent);
check('q-delete-sync: a wrong answer is marked and explained', v2.includes('Not quite.') && v2.includes('Coherent pages do not order anything'), v2.slice(0, 200));

// ---- themes and a narrow screen
await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));
await setMode('migrate');
await program('managed');
for (let k = 0; k < 4; k++) await press('Step');
if (out) await page.screenshot({ path: out + '-dark.png' });
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await pause(150);
if (out) await page.screenshot({ path: out + '-light.png' });
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((B) => ({
  page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1,
  bench: document.querySelector(B).getBoundingClientRect().width <= document.documentElement.clientWidth,
}), B);
check('at a narrow width nothing makes the page scroll sideways', narrow.page && narrow.bench, JSON.stringify(narrow));
const sysSel = await page.evaluate((B) => {
  const s = document.querySelector(B + ' select[aria-label="System"]').getBoundingClientRect();
  const body = document.querySelector(B + ' .body');
  const b = body.getBoundingClientRect();
  return { right: Math.round(s.right), bodyRight: Math.round(b.right), scrolls: body.scrollWidth > body.clientWidth + 1 };
}, B);
check('  the System select fits inside the bench body, which does not scroll sideways', sysSel.right <= sysSel.bodyRight && !sysSel.scrolls, JSON.stringify(sysSel));
if (out) {
  await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
