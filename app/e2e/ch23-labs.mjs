// Browser check for the chapter 23 labs page: status table, the three predictions against this
// machine's measurements, the parts and first-touch read-outs, the results table with its
// serial ratio, the depth-against-breadth table, and the reference-solution toggle. Everything
// expected is computed here from /api/reports and /api/lab-file, so the check holds on any
// machine, with or without runs.
// Usage: APP_URL=http://localhost:<port> node e2e/ch23-labs.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file, and start from an empty one so that every
// prediction is uncommitted whatever the reader has already done.
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
const saved = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
  if (r.method() === 'PUT') saved.push(r.postData() ?? '');
  return r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : JSON.stringify(EMPTY) });
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const shot = async (name, sel) => {
  if (!out) return;
  await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};

// ---- the same reading of the reports that the page makes (app/src/lib/reports.ts, LabKit.tsx, Labs.tsx)
const LAB = { num: '23.1', id: 'ch23/lab1_overlap', dir: 'lab1_overlap' };
const BACKENDS = ['metal', 'emu', 'cuda', 'hip'];
const NAME = { metal: 'Metal', emu: 'emulator', cuda: 'CUDA', hip: 'HIP' };
const fmt = (v, digits = 3) =>
  !isFinite(v) ? 'n/a' : v === 0 ? '0' : Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : Number(v.toPrecision(digits)).toString();
const results = (run, id, variant) => (run ? run.results.filter((x) => x.lab === id && x.variant === variant) : []);
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = results(run, id, variant).filter((x) => x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const benchRows = (run, id, variant) => results(run, id, variant).flatMap((x) => x.bench);
const solutionText = (reports, b, id) => {
  const s = labStatus(reports[b], id, 'solution');
  return s === 'pass' ? (b === 'emu' ? 'emulator-checked' : `verified on ${NAME[b]}`) : s === 'fail' ? 'unverified: failed on this machine' : 'no run on this machine';
};
const real = (reports) => Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const preferred = (reports, id) =>
  real(reports).flatMap((b) => {
    const mine = labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : [];
    const rows = mine.length ? mine : benchRows(reports[b], id, 'solution');
    return rows.length ? [{ backend: b, variant: mine.length ? 'work' : 'solution', rows }] : [];
  });
const compare = (predicted, measured) => {
  const r = predicted / measured;
  if (Math.abs(r - 1) <= 0.1) return 'Your prediction was within 10% of the measurement.';
  return r < 1 ? `The measurement is ${fmt(1 / r)} times your prediction.` : `Your prediction is ${fmt(r)} times the measurement.`;
};
const row = (set, name, where = {}) => set.rows.find((r) => r.name === name && Object.entries(where).every(([k, v]) => r[k] === v));

await goto(page, 'ch23/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = async () => (await page.$eval('.prose', (e) => e.textContent)).replace(/\s+/g, ' ');

// ---- structure, commands and what the page says
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is titled "Lab"', h1.trim().endsWith('Lab'), h1);
const text0 = await body();
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run 23.1', 'python3 labs/pmpp.py run 23.1 --bench', 'python3 labs/pmpp.py run 23.1 --backend emu',
  'python3 labs/pmpp.py run 23.1 --solution', 'python3 labs/pmpp.py reset 23.1 --yes'])
  check(`command shown: ${c}`, text0.includes(c));
check('the page says its numbers are measurements, not expected values', text0.includes('None is an expected value'));
check('the lab is called optional and adds no milestone', text0.includes('It is optional') && text0.includes('nothing to the through-line project') && text0.includes('not a milestone'));
check('the emulator is described as synchronous: it never shows overlap', text0.includes('every stream call is synchronous') && text0.includes('never whether anything overlapped'));
check('no race flavour, and why', text0.includes('not built for race') && text0.includes('ThreadSanitizer'));
check('the differences from the book: non-blocking streams, no priorities or events, "requires" pinned',
  text0.includes('cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)') && text0.includes('No priorities, no events.') && text0.includes('“Requires” pinned memory.') &&
  text0.includes('will revert to a synchronous behavior which will not overlap with other work'));
check('the chunk rule and the j-th non-empty chunk in stream j mod streams', text0.includes('[k*n/chunks, (k+1)*n/chunks)') && text0.includes('j mod streams'));
check('what the tests cannot see is listed (emulator time, copy streams, depth against breadth, the semaphore)',
  text0.includes('Anything about time on the emulator.') && text0.includes('The copies’ streams.') && text0.includes('Depth against breadth.') && text0.includes('The semaphore.'));
check('Metal: slotBuffer, a dispatch semaphore that starts at K, the drain on your thread',
  text0.includes('slotBuffer') && text0.includes('dispatch semaphore that starts at K') && text0.includes('the drain stays on yours'));
check('the size cap is called a parameter of the lab, not a device fact', text0.includes('a parameter of this lab') && text0.includes('not a fact about any device'));
check('nothing on the page says it is being written', !/being written/i.test(text0));

// ---- status table of the reference solution, cell by cell against the reports
const cells = await page.$$eval('table[data-table="solution-status"] td[data-lab]', (tds) =>
  tds.map((td) => ({ lab: td.dataset.lab, backend: td.dataset.backend, status: td.dataset.status, text: td.textContent.trim(), chip: td.querySelector('.chip')?.className ?? '' })));
check('status table has 1 lab x 4 backends', cells.length === 4, String(cells.length));
for (const b of BACKENDS) {
  const c = cells.find((x) => x.backend === b);
  const want = solutionText(reports, b, LAB.id);
  check(`status ${b}: "${want}"`, c && c.status === want && c.text === want, JSON.stringify(c));
  check(`  ${b}: a pass is green, a failure red, no run is neither`,
    c && (want.startsWith('verified') || want === 'emulator-checked' ? c.chip === 'chip ok' : want.startsWith('unverified') ? c.chip === 'chip fail' : c.chip === ''), JSON.stringify(c));
}
await shot('status', 'table[data-table="solution-status"]');

// ---- status of the reader's own work
{
  const shown = await page.$$eval(`[data-work="${LAB.id}"] [data-backend]`, (s) => Object.fromEntries(s.map((x) => [x.dataset.backend, x.textContent])));
  const ok = BACKENDS.every((b) => {
    const run = reports[b];
    if (!run) return shown[b].includes('no run on this machine');
    const rs = results(run, LAB.id, 'work');
    const total = rs.reduce((n, x) => n + x.tests.length, 0);
    const passed = rs.reduce((n, x) => n + x.tests.filter((t) => t.pass).length, 0);
    const s = labStatus(run, LAB.id, 'work');
    if (rs.some((x) => !x.built)) return shown[b].includes('does not build');
    if (s === 'none') return shown[b].includes('not started');
    return shown[b].includes(`${s === 'pass' ? 'passes,' : 'not passing:'} ${passed} of ${total} tests`);
  });
  check('status of your work matches the reports on every backend', ok, JSON.stringify(shown));
}

// ---- the parts table and the first-touch line
const sets = preferred(reports, LAB.id);
{
  const withParts = sets.filter((s) => row(s, 'parts'));
  const shown = await page.$$eval('table[data-table="parts"] tr[data-parts]', (trs) =>
    trs.map((tr) => ({ b: tr.dataset.parts, in: tr.querySelector('[data-col="in"]').textContent, kernel: tr.querySelector('[data-col="kernel"]').textContent, out: tr.querySelector('[data-col="out"]').textContent, engines: tr.querySelector('[data-col="engines"]').textContent })));
  if (withParts.length) {
    const ok = withParts.every((s) => {
      const p = row(s, 'parts');
      const metal = p.fill_s !== undefined;
      const r = shown.find((x) => x.b === s.backend);
      return r && r.in === fmt((metal ? p.fill_s : p.h2d_s) * 1000) && r.kernel === fmt(p.kernel_s * 1000) && r.out === fmt((metal ? p.drain_s : p.d2h_s) * 1000) &&
        (metal ? r.engines === 'no copy engine' : r.engines === String(p.async_engines ?? ''));
    });
    check(`parts: ${withParts.map((s) => s.backend).join(', ')} read-outs match the parts row (Metal: no copy engine)`, ok && shown.length === withParts.length, JSON.stringify(shown));
  } else {
    const empty = await page.$eval('[data-parts-empty]', (e) => e.textContent).catch(() => '');
    check('no parts rows, and the page says what to run', empty.includes('python3 labs/pmpp.py run 23.1 --bench'), empty);
  }
  for (const s of sets.filter((s) => row(s, 'firsttouch'))) {
    const f = row(s, 'firsttouch');
    const t = await page.$eval(`[data-firsttouch="${s.backend}"]`, (e) => e.textContent.replace(/\s+/g, ' ')).catch(() => '');
    check(`first touch on ${s.backend}: ${fmt(f.fresh_fill_s * 1000)} ms against ${fmt(f.fill_s * 1000)} ms, ${fmt(f.fresh_fill_s / f.fill_s)} times`,
      t.includes(`${fmt(f.fresh_fill_s * 1000)} ms`) && t.includes(`${fmt(f.fill_s * 1000)} ms`) && t.includes(`${fmt(f.fresh_fill_s / f.fill_s)} times as long`), t);
  }
}
await shot('parts', 'table[data-table="parts"], [data-parts-empty]');

// ---- predictions: commit, then the measured value from the reports appears with a comparison
const secs = (r) => (typeof r?.seconds === 'number' ? r.seconds : undefined);
const stagedRatio = (host) => sets.flatMap((s) => {
  const a = secs(row(s, 'serial', { host }));
  const b = secs(row(s, 'staged', { streams: 4, chunks: 16, order: 'depth', host }));
  return a && b ? [{ backend: s.backend, value: a / b }] : [];
});
const expected = {
  staged: stagedRatio('pinned'),
  pageable: stagedRatio('pageable'),
  metal: sets.flatMap((s) => {
    const a = secs(row(s, 'inflight', { buffers: 1, chunks: 16 }));
    const b = secs(row(s, 'inflight', { buffers: 3, chunks: 16 }));
    return a && b ? [{ backend: s.backend, value: a / b }] : [];
  }),
};
const predictions = [
  { key: 'staged', id: 'ch23/labs/p-staged', find: 'at 4 streams, 16 chunks, depth first, pinned host memory', guess: 1.5, missing: 'This ratio exists only on CUDA and HIP' },
  { key: 'pageable', id: 'ch23/labs/p-pageable', find: 'The same ratio with pageable host buffers', guess: 1.1, missing: 'This ratio exists only on CUDA and HIP' },
  { key: 'metal', id: 'ch23/labs/p-metal', find: 'in-flight time with 1 buffer divided by the in-flight time with 3 buffers', guess: 2, missing: 'This ratio exists only on Metal' },
];
for (const p of predictions) {
  const el = (await page.evaluateHandle((find) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.kind')?.textContent === 'Predict first' && e.textContent.replace(/\s+/g, ' ').includes(find)), p.find)).asElement();
  if (!el) { check(`${p.key}: prediction block present`, false); continue; }
  check(`${p.key}: nothing is revealed before a prediction is committed`, (await el.$('[data-measured]')) === null);
  const input = await el.$('input[aria-label="your prediction"]');
  await input.focus();
  await input.type(String(p.guess));
  await el.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Commit prediction').click());
  await pause(150);
  const shown = await el.$$eval('[data-measured]', (ps) => ps.map((x) => ({ measured: x.dataset.measured, backend: x.dataset.backend ?? '', bold: x.querySelector('b')?.textContent ?? '', text: x.textContent.replace(/\s+/g, ' ') })));
  const want = expected[p.key];
  if (want.length) {
    const ok = want.every((w) => {
      const s = shown.find((x) => x.backend === w.backend);
      return s && s.measured === fmt(w.value) && s.bold === fmt(w.value) && s.text.includes(compare(p.guess, w.value));
    });
    check(`${p.key}: measured value from /api/reports is shown (${want.map((w) => `${w.backend} ${fmt(w.value)}`).join(', ')}) and compared with ${p.guess}`,
      ok && shown.length === want.length, JSON.stringify(shown));
    check(`  ${p.key}: the reveal says whose code was measured and the times behind it`,
      shown.every((s) => /with (your kernel|the reference solution)/.test(s.text) && / ms/.test(s.text)), JSON.stringify(shown));
  } else {
    check(`${p.key}: no measurement in the reports, and the reveal says why and what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes(p.missing) && shown[0].text.includes('python3 labs/pmpp.py run 23.1 --bench'), JSON.stringify(shown));
  }
  check(`  ${p.key}: the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${p.guess}`));
}
await pause(600);
const ans = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
check('the three predictions were sent to /api/progress under their ids, and none reached the real file',
  predictions.every((p) => ans[p.id]?.value === p.guess), Object.keys(ans).join(', '));
await shot('predict', '.q');

// ---- results table: every serial, staged and inflight row in the reports, and nothing invented
const PIPE = new Set(['serial', 'staged', 'inflight']);
const all = real(reports).flatMap((b) => {
  const mine = labStatus(reports[b], LAB.id, 'work') === 'pass' ? benchRows(reports[b], LAB.id, 'work') : [];
  return [...mine, ...benchRows(reports[b], LAB.id, 'solution')];
}).filter((r) => PIPE.has(r.name));
if (all.length) {
  const shown = await page.$$eval(`table[data-results="${LAB.num}"] tbody tr`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
  const head = await page.$$eval(`table[data-results="${LAB.num}"] thead th`, (ths) => ths.map((t) => t.textContent.trim()));
  const col = (h) => head.indexOf(h);
  const base = (r) => all.find((x) => x.name === 'serial' && x.backend === r.backend && x.variant === r.variant && String(x.host) === (r.name === 'inflight' ? 'shared' : String(r.host)));
  const ok = all.every((r, i) => {
    const s = shown[i];
    const b = base(r);
    return s && s[col('Row')] === r.name && s[col('Whole call, ms')] === fmt(r.seconds * 1000) &&
      s[col('Streams or buffers')] === String(r.streams ?? r.buffers ?? '') && s[col('Chunks')] === String(r.chunks ?? '') &&
      s[col('Serial / this')] === (b ? fmt(b.seconds / r.seconds) : '') &&
      (r.overlap_observed === undefined || s[col('Overlap seen')] === (r.overlap_observed === true ? 'yes' : 'no'));
  });
  check(`results table has the ${all.length} serial, staged and in-flight rows, with time, serial ratio and overlap read from the reports`,
    ok && shown.length === all.length, JSON.stringify({ head, first: shown.slice(0, 2) }));
} else {
  const empty = await page.$eval(`[data-results-empty="${LAB.num}"]`, (e) => e.textContent).catch(() => '');
  check('no benchmark rows in the reports, and the empty state says what to run', empty.includes('./verify.sh'), empty);
}
await shot('results', `section.bench[aria-label="Lab ${LAB.num} on this machine"]`);

// ---- the fourth guess: depth against breadth
{
  const orderSets = sets.filter((s) => [1, 2, 3, 4, 8].some((k) => row(s, 'staged', { streams: k, chunks: 16, order: 'depth', host: 'pinned' }) && row(s, 'staged', { streams: k, chunks: 16, order: 'breadth', host: 'pinned' })));
  if (orderSets.length) {
    const shown = await page.$$eval('table[data-order] tbody tr', (trs) => trs.map((tr) => ({ b: tr.dataset.backend, k: tr.dataset.streams, r: tr.querySelector('[data-col="ratio"]').textContent })));
    const ok = shown.length > 0 && shown.every((x) => {
      const s = orderSets.find((o) => o.backend === x.b);
      const d = row(s, 'staged', { streams: Number(x.k), chunks: 16, order: 'depth', host: 'pinned' }).seconds;
      const br = row(s, 'staged', { streams: Number(x.k), chunks: 16, order: 'breadth', host: 'pinned' }).seconds;
      return x.r === fmt(br / d);
    });
    check('the depth-against-breadth table matches the staged rows', ok, JSON.stringify(shown));
  } else {
    const t = await page.$eval('[data-order="none"]', (e) => e.textContent.replace(/\s+/g, ' ')).catch(() => '');
    check('no staged rows from CUDA or HIP: the page says Metal has no issue order to compare and what to run', t.includes('Metal has no issue order to compare') && t.includes('python3 labs/pmpp.py run 23.1 --bench'), t);
  }
}

// ---- reference solution: hidden by default, then the three real files, with computed status
check('no reference solution is shown before it is asked for', (await page.$$('figure.code')).length === 0);
const same = (a, b) => a.replace(/\s+$/g, '') === b.replace(/\s+$/g, '');
const P = `p[data-solution="${LAB.num}"] button`;
check('toggle starts collapsed', (await page.$eval(P, (b) => b.getAttribute('aria-expanded') + '|' + b.textContent.trim())) === 'false|Show the reference solution');
await page.click(P);
await page.waitForSelector('figure.code .tabs button');
const files = {};
for (const f of ['kernel.cu', 'kernel.metal', 'launch.mm'])
  files[f] = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => (r.ok ? r.text() : 'MISSING')), `ch23/${LAB.dir}/solution/${f}`);
check('the three solution files exist in the repository', Object.values(files).every((t) => t !== 'MISSING' && t.length > 100));
const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
check('tabs are CUDA, Metal kernel, Metal host', tabs.join('|') === 'CUDA|Metal kernel|Metal host', tabs.join('|'));
const st = (b) => labStatus(reports[b], LAB.id, 'solution');
const got = [st('cuda') === 'pass' ? 'verified on CUDA' : '', st('hip') === 'pass' ? 'verified on HIP' : '', st('emu') === 'pass' ? 'emulator-checked' : ''].filter(Boolean);
const lacking = ['cuda', 'hip'].filter((b) => st(b) !== 'pass').map((b) => NAME[b]);
const cuStatus = got.length ? got.join(', ') + (lacking.length ? `; unverified on ${lacking.join(' and ')}` : '') : 'unverified (no run on this machine)';
const metalStatus = st('metal') === 'pass' ? 'verified on Metal' : st('metal') === 'fail' ? 'unverified (failed on this machine)' : 'unverified (no run on this machine)';
const order = [['kernel.cu', cuStatus], ['kernel.metal', metalStatus], ['launch.mm', metalStatus]];
for (let i = 0; i < 3; i++) {
  await page.evaluate((i) => [...document.querySelectorAll('figure.code .tabs button')][i].click(), i);
  const [f, status] = order[i];
  await page.waitForFunction(() => !document.querySelector('figure.code pre').textContent.startsWith('// loading'), { timeout: 10000 });
  await pause(60);
  const shown = await page.$eval('figure.code pre', (e) => e.textContent);
  const tabStatus = await page.$eval('figure.code .tabs .status', (e) => e.textContent);
  check(`${tabs[i]} tab is solution/${f} verbatim (${files[f].split('\n').length - 1} lines), status "${status}"`, same(shown, files[f]) && tabStatus === status, tabStatus);
}
await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')][0].click());
await pause(80);
check('tier colours are applied: device arrays global, per-thread values registers',
  (await page.$$('figure.code .v-global')).length > 0 && (await page.$$('figure.code .v-reg')).length > 0);
await shot('solution', `p[data-solution="${LAB.num}"]`);
await page.click(P);
await pause(80);
check('the toggle hides it again', (await page.$$('figure.code')).length === 0);

// ---- links and margin notes
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
check('the page links to page 23.3 and to the summary', links.includes('#/ch23/23.3') && links.includes('#/ch23/wrap'), links.join(' '));
for (const l of links) {
  await goto(page, l.slice(2));
  const ok = await page.evaluate(() => !!document.querySelector('.page h1') && !document.querySelector('.page').textContent.includes('being written'));
  check(`link ${l} opens a written page`, ok);
}
await goto(page, 'ch23/labs');
const misplaced = await page.$$eval('.prose > aside.note', (as) => as.filter((a) => {
  const prev = a.previousElementSibling;
  return prev && !['P', 'UL', 'OL', 'ASIDE'].includes(prev.tagName);
}).map((a) => `${a.previousElementSibling.tagName}.${a.previousElementSibling.className}`));
check('every margin note follows a paragraph, a list or another note', misplaced.length === 0, misplaced.join(', '));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
