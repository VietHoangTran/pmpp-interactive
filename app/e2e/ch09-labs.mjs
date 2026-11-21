// Browser check for the chapter 9 labs page: status table, the race lines, seven predictions
// against this machine's measurements (three shared with sections 9.3, 9.4 and 9.5), the
// results tables, the milestone line and the reference-solution toggles. Everything expected
// is computed here from /api/reports and /api/lab-file, so the check holds on any machine,
// with or without runs. The progress file is replaced by an empty one and never written.
// Usage: APP_URL=http://localhost:<port> node e2e/ch09-labs.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
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

// ---- the same reading of the reports that the page makes (lib/reports.ts, components/LabKit.tsx)
const LABS = [
  { num: '9.1', id: 'ch09/lab1_racehunt', dir: 'lab1_racehunt' },
  { num: '9.2', id: 'ch09/lab2_privatised', dir: 'lab2_privatised' },
  { num: '9.3', id: 'ch09/lab3_tokencounts', dir: 'lab3_tokencounts' },
];
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
const preferred = (reports, id) =>
  Object.keys(reports).filter((b) => b !== 'emu').flatMap((b) => {
    const mine = labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : [];
    const rows = mine.length ? mine : benchRows(reports[b], id, 'solution');
    return rows.length ? [{ backend: b, rows }] : [];
  });
const compare = (predicted, measured) => {
  const r = predicted / measured;
  if (Math.abs(r - 1) <= 0.1) return 'Your prediction was within 10% of the measurement.';
  return r < 1 ? `The measurement is ${fmt(1 / r)} times your prediction.` : `Your prediction is ${fmt(r)} times the measurement.`;
};
const find = (rows, name, data, coarse) => rows.find((r) => r.name === name && r.data === data && (coarse == null || r.coarse === coarse));

await goto(page, 'ch09/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure, commands, backends
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Labs"', h1.includes('Labs'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((x) => x.textContent.trim()));
const labH2 = h2.filter((t) => t.startsWith('Lab '));
check('three lab headings, in order', JSON.stringify(labH2) === JSON.stringify([
  'Lab 9.1 (optional): race hunt, two histogram kernels',
  'Lab 9.2: privatised in shared memory, coarsened and run-merged',
  'Lab 9.3: token counts for repetition penalties (M5)',
]), JSON.stringify(labH2));
const text0 = await body();
check('nothing on the page says it is being written', !/being written/i.test(text0));
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run 9.1', 'python3 labs/pmpp.py run 9.1 --bench', 'python3 labs/pmpp.py run 9.1 --backend emu',
  'python3 labs/pmpp.py run 9.1 --solution', 'python3 labs/pmpp.py reset 9.1 --yes', 'python3 labs/pmpp.py run 9.1 --backend race'])
  check(`command shown: ${c}`, text0.includes(c));
check('the race flavour is explained as ThreadSanitizer, the only backend that sees a missing atomic', text0.includes('ThreadSanitizer') && text0.includes('the only backend'));
check('scopes are said to be invisible to every backend here', text0.includes('Neither emulator gives scopes or memory orders any effect') && text0.includes('Check every scope by reading it'));
check('the HIP build\'s libhipcxx dependency is stated, without a claim of having tried it', text0.includes('links libhipcxx::libhipcxx when CMake finds it') && text0.includes('Nobody has tried'));
check('the CUDA/HIP caveat and the vocabulary are stated', text0.includes('front-end-checked') && text0.includes('emulator-checked means'));
const misplaced = await page.$$eval('.prose > .note', (ns) => ns.filter((n) => n.previousElementSibling?.tagName !== 'P').map((n) => n.textContent.slice(0, 60)));
check('every margin note follows a paragraph', misplaced.length === 0, misplaced.join(' | '));

// ---- status table of the reference solutions
const cells = await page.$$eval('table[data-table="solution-status"] td[data-lab]', (tds) =>
  tds.map((td) => ({ lab: td.dataset.lab, backend: td.dataset.backend, status: td.dataset.status, text: td.textContent.trim(), chip: td.querySelector('.chip')?.className ?? '' })));
check('status table has 3 labs x 4 backends', cells.length === 12, String(cells.length));
for (const b of BACKENDS) {
  const mine = cells.filter((c) => c.backend === b);
  const want = LABS.map((l) => solutionText(reports, b, l.id));
  check(`status column ${b}: ${LABS.map((l, i) => `${l.num} ${want[i]}`).join('; ')}`,
    mine.length === 3 && mine.every((c, i) => c.status === want[i] && c.text === want[i]), JSON.stringify(mine));
}
await shot('status', 'table[data-table="solution-status"]');

// ---- the race lines, which the status table does not show
for (const l of LABS) {
  const line = await page.$eval(`[data-race-lab="${l.num}"]`, (e) => ({ s: e.dataset.raceStatus, text: e.textContent })).catch(() => null);
  const run = reports.race;
  const want = labStatus(run, l.id, 'solution');
  const work = results(run, l.id, 'work');
  const races = work.reduce((k, x) => k + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0);
  check(`lab ${l.num}: race line reads the race reports (${run ? want : 'no run'}${races ? `, ${races} race(s) in your work` : ''})`,
    !!line && (!run ? line.text.includes('No run of lab') :
      line.s === want && (want !== 'pass' || line.text.includes('emulator-checked on the race backend')) && (!races || line.text.includes(`ThreadSanitizer reported ${races} race`))),
    JSON.stringify(line));
}

// ---- status of the reader's own work
for (const l of LABS) {
  const row = await page.$$eval(`[data-work="${l.id}"] [data-backend]`, (s) => Object.fromEntries(s.map((x) => [x.dataset.backend, x.textContent])));
  const ok = BACKENDS.every((b) => {
    const run = reports[b];
    if (!run) return row[b].includes('no run on this machine');
    const rs = results(run, l.id, 'work');
    const total = rs.reduce((n, x) => n + x.tests.length, 0);
    const passed = rs.reduce((n, x) => n + x.tests.filter((t) => t.pass).length, 0);
    const s = labStatus(run, l.id, 'work');
    if (!rs.length) return row[b].includes('not started');
    if (rs.some((x) => !x.built)) return row[b].includes('does not build');
    if (s === 'none') return row[b].includes('not started');
    return row[b].includes(`${s === 'pass' ? 'passes,' : 'not passing:'} ${passed} of ${total} tests`);
  });
  check(`lab ${l.num}: status of your work matches the reports on every backend`, ok, JSON.stringify(row));
}

// ---- predictions: commit, then the measured value from the reports, compared
const g = (r) => r?.gpixels_per_s;
const expected = {
  skew: preferred(reports, LABS[0].id).flatMap((s) => {
    const u = g(find(s.rows, 'basic', 'uniform')); const c = g(find(s.rows, 'basic', 'constant'));
    return u && c ? [{ backend: s.backend, value: u / c }] : [];
  }),
  pool: preferred(reports, LABS[0].id).flatMap((s) => {
    const b = g(find(s.rows, 'basic', 'constant')); const p = g(find(s.rows, 'pool', 'constant'));
    return b && p ? [{ backend: s.backend, value: p / b }] : [];
  }),
  shared: preferred(reports, LABS[1].id).flatMap((s) => {
    const b = g(find(s.rows, 'basic', 'constant', 1)); const sh = g(find(s.rows, 'shared', 'constant', 16));
    return b && sh ? [{ backend: s.backend, value: sh / b }] : [];
  }),
  inter: preferred(reports, LABS[1].id).flatMap((s) => {
    const i = find(s.rows, 'shared', 'uniform', 16)?.seconds; const c = find(s.rows, 'contiguous', 'uniform', 16)?.seconds;
    return i && c ? [{ backend: s.backend, value: c / i }] : [];
  }),
  runs: preferred(reports, LABS[1].id).flatMap((s) => {
    const r = g(find(s.rows, 'runs', 'sky', 16)); const sh = g(find(s.rows, 'shared', 'sky', 16));
    return r && sh ? [{ backend: s.backend, value: r / sh }] : [];
  }),
  tpool: preferred(reports, LABS[2].id).flatMap((s) => {
    const c = find(s.rows, 'count', 'uniform')?.seconds; const p = find(s.rows, 'count_pool', 'uniform')?.seconds;
    return c && p ? [{ backend: s.backend, value: p / c }] : [];
  }),
  append: preferred(reports, LABS[2].id).flatMap((s) => {
    const c = find(s.rows, 'count', 'uniform')?.seconds; const a = find(s.rows, 'append', 'uniform')?.seconds;
    return c && a ? [{ backend: s.backend, value: c / a }] : [];
  }),
};
const predictions = [
  { key: 'skew', id: 'ch09/9.3/p-skew-slowdown', find: 'slower will the basic kernel run on a constant image', guess: 20 },
  { key: 'pool', id: 'ch09/labs/p-lab1-pool-gap', find: 'will the pool kernel run on your GPU', guess: 4 },
  { key: 'shared', id: 'ch09/9.4/p-shared-speedup', find: 'will the shared-memory kernel run on your GPU', guess: 30 },
  { key: 'inter', id: 'ch09/9.5/p-interleaved-gain', find: 'faster than the contiguous kernel will the interleaved', guess: 2 },
  { key: 'runs', id: 'ch09/labs/p-lab2-runs-sky', find: 'On the sky image at factor 16', guess: 3 },
  { key: 'tpool', id: 'ch09/labs/p-lab3-pool', find: 'Uniform ids: how many times as long', guess: 10 },
  { key: 'append', id: 'ch09/labs/p-lab3-append', find: 'recounting the whole context', guess: 4096 },
];
check('seven predictions on the page', (await page.$$eval('.q', (qs) => qs.filter((q) => q.querySelector('.kind')?.textContent === 'Predict first').length)) === 7);
for (const p of predictions) {
  const el = (await page.evaluateHandle((f) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.kind')?.textContent === 'Predict first' && e.textContent.includes(f)), p.find)).asElement();
  if (!el) { check(`${p.id}: prediction block present`, false); continue; }
  check(`${p.id}: nothing is revealed before a prediction is committed`, (await el.$('[data-measured]')) === null);
  const input = await el.$('input[aria-label="your prediction"]');
  await input.focus();
  await input.type(String(p.guess));
  await el.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Commit prediction').click());
  await pause(150);
  const shown = await el.$$eval('[data-measured]', (ps) => ps.map((x) => ({ measured: x.dataset.measured, backend: x.dataset.backend ?? '', bold: x.querySelector('b')?.textContent ?? '', text: x.textContent })));
  const want = expected[p.key];
  if (want.length) {
    const ok = want.every((w) => {
      const s = shown.find((x) => x.backend === w.backend);
      return s && s.measured === fmt(w.value) && s.bold === fmt(w.value) && s.text.includes(compare(p.guess, w.value)) && /with (your kernel|the reference solution)/.test(s.text);
    });
    check(`${p.id}: measured ${want.map((w) => `${w.backend} ${fmt(w.value)}`).join(', ')} shown and compared with ${p.guess}: "${compare(p.guess, want[0].value)}"`,
      ok && shown.length === want.length, JSON.stringify(shown));
  } else {
    check(`${p.id}: no measurement in the reports, and the reveal says what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes('--bench'), JSON.stringify(shown));
  }
  check(`  ${p.id}: the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${p.guess}`));
}
await pause(700);
const answers = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
check('every prediction was sent under its id (three of them the sections\' ids), and none reached the real file',
  predictions.every((p) => answers[p.id]?.value === p.guess), Object.keys(answers).join(', '));

// ---- results tables: every benchmark row in the reports, and nothing invented
const sets = (id) => Object.keys(reports).filter((b) => b !== 'emu').flatMap((b) => {
  const mine = labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : [];
  return [...mine, ...benchRows(reports[b], id, 'solution')];
});
const perSecond = (v, w) => (v >= 1e9 ? `${fmt(v / 1e9)} G ${w}/s` : v >= 1e6 ? `${fmt(v / 1e6)} M ${w}/s` : v >= 1e3 ? `${fmt(v / 1e3)} k ${w}/s` : `${fmt(v)} ${w}/s`);
const tables = [
  { lab: LABS[0], cell: (r) => [fmt(r.seconds * 1000), fmt(r.gpixels_per_s)] },
  { lab: LABS[1], cell: (r) => [String(r.coarse), fmt(r.seconds * 1000), fmt(r.gpixels_per_s)] },
  { lab: LABS[2], cell: (r) => [fmt(r.seconds * 1000), r.tokens_per_s != null ? perSecond(r.tokens_per_s, 'tokens') : perSecond(r.elements_per_s, 'logits')] },
];
for (const t of tables) {
  const rows = sets(t.lab.id);
  if (rows.length) {
    const shown = await page.$$eval(`table[data-results="${t.lab.num}"] tbody tr`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    check(`lab ${t.lab.num}: results table has the ${rows.length} benchmark row(s) from the reports, with their figures`,
      shown.length === rows.length && rows.every((r, i) => t.cell(r).every((c) => shown[i].includes(c))), JSON.stringify(shown.slice(0, 3)));
  } else {
    const empty = await page.$eval(`[data-results-empty="${t.lab.num}"]`, (e) => e.textContent).catch(() => '');
    check(`lab ${t.lab.num}: no benchmark rows, and the empty state says what to run`, empty.includes('./verify.sh'), empty);
  }
}
await shot('lab2-results', 'section.bench[aria-label="Lab 9.2 on this machine"]');

// ---- milestone
const m5 = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b] && labStatus(reports[b], LABS[2].id, 'work') === 'pass');
const ms = await page.$eval('[data-milestone]', (e) => ({ v: e.dataset.milestone, text: e.textContent }));
check(`milestone line is read from the reports (${m5.join(',') || 'none'})`, ms.v === (m5.join(',') || 'none'), JSON.stringify(ms));
check('the milestone paragraph names M5, lab 9.3, the later chapters and module X1', text0.includes('Milestone M5') && text0.includes('Chapters 11, 12 and 14') && text0.includes('module X1'));

// ---- reference solutions: hidden by default, then the three real files with computed status
check('no reference solution is shown before it is asked for', (await page.$$('figure.code')).length === 0);
const same = (a, b) => a.replace(/\s+$/g, '') === b.replace(/\s+$/g, '');
for (const l of LABS) {
  const P = `p[data-solution="${l.num}"] button`;
  check(`lab ${l.num}: toggle starts collapsed`, (await page.$eval(P, (b) => b.getAttribute('aria-expanded') + '|' + b.textContent.trim())) === 'false|Show the reference solution');
  await page.click(P);
  await page.waitForSelector('figure.code .tabs button');
  const files = {};
  for (const f of ['kernel.cu', 'kernel.metal', 'launch.mm'])
    files[f] = await page.evaluate((p) => fetch('/api/lab-file?path=' + encodeURIComponent(p)).then((r) => (r.ok ? r.text() : 'MISSING')), `ch09/${l.dir}/solution/${f}`);
  check(`lab ${l.num}: the three solution files exist`, Object.values(files).every((t) => t !== 'MISSING' && t.length > 100));
  const st = (b) => labStatus(reports[b], l.id, 'solution');
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
    const shownCode = await page.$eval('figure.code pre', (e) => e.textContent);
    const tabStatus = await page.$eval('figure.code .tabs .status', (e) => e.textContent);
    check(`lab ${l.num}: tab ${i + 1} is solution/${f} verbatim, status "${status}"`, same(shownCode, files[f]) && tabStatus === status, tabStatus);
  }
  if (l.num === '9.2') {
    await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')][0].click());
    await pause(80);
    check('  lab 9.2: bins_s is coloured as shared memory, image and bins as global', (await page.$$('figure.code .v-shared')).length > 0 && (await page.$$('figure.code .v-global')).length > 0);
    await shot('solution-lab2', `p[data-solution="${l.num}"]`);
  }
  await page.click(P);
  await pause(80);
  check(`lab ${l.num}: the toggle hides it again`, (await page.$$('figure.code')).length === 0);
}

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch09/9.3', '#/ch09/9.4', '#/ch09/9.5', '#/ch09/9.6', '#/x1/X1.3']);
check(`links go to this chapter's sections and to X1.3 (${links.join(', ')})`, links.length >= 4 && links.every((h) => known.has(h)), links.join(', '));

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch09/labs');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'table[data-table="solution-status"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
