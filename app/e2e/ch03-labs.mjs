// Browser check for the chapter 3 labs page: commands, status table, the five lab blocks,
// predictions against this machine's measurements, results tables and the
// reference-solution toggles. Everything expected is computed here from /api/reports and
// /api/lab-file, so the check holds on any machine, with or without runs.
// Usage: node e2e/ch03-labs.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file, and start from an empty one so every
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

// ---- the same reading of the reports that the page makes (app/src/lib/reports.ts, components/LabKit.tsx)
const LABS = [
  { num: '3.1', id: 'ch03/lab1_grayscale', dir: 'lab1_grayscale', name: 'grayscale' },
  { num: '3.2', id: 'ch03/lab2_blur', dir: 'lab2_blur', name: 'blur' },
  { num: '3.3', id: 'ch03/lab3_naive_gemm', dir: 'lab3_naive_gemm', name: 'naive GEMM (M1)' },
  { num: '3.4', id: 'ch03/lab4_granularity', dir: 'lab4_granularity', name: 'granularity' },
  { num: '3.5', id: 'ch03/lab5_maphunt', dir: 'lab5_maphunt', name: 'map hunt' },
];
const BACKENDS = ['metal', 'emu', 'cuda', 'hip'];
const NAME = { metal: 'Metal', emu: 'emulator', cuda: 'CUDA', hip: 'HIP' };
const fmt = (v, digits = 3) =>
  !isFinite(v) ? 'n/a' : v === 0 ? '0' : Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : Number(v.toPrecision(digits)).toString();
const results = (run, id, variant) => (run ? run.results.filter((x) => x.lab === id && x.variant === variant) : []);
const labStatus = (run, id, variant) => {
  // An untouched starter fails by design: the page reports it as "not started".
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
// One set of benchmark rows per backend other than emu: the reader's own if it passed and was timed, else the reference.
const preferred = (reports, id) =>
  Object.keys(reports).filter((b) => b !== 'emu').flatMap((b) => {
    const mine = labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : [];
    const rows = mine.length ? mine : benchRows(reports[b], id, 'solution');
    return rows.length ? [{ backend: b, variant: mine.length ? 'work' : 'solution', rows }] : [];
  });
const compare = (predicted, measured) => {
  if (!(predicted > 0)) return 'not a positive ratio';
  const r = predicted / measured;
  if (Math.abs(r - 1) <= 0.1) return 'Your prediction was within 10% of the measurement.';
  return r < 1 ? `The measurement is ${fmt(1 / r)} times your prediction.` : `Your prediction is ${fmt(r)} times the measurement.`;
};

await goto(page, 'ch03/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- page structure
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['Running them', 'The four backends', 'Verification status of the reference solutions',
  'Lab 3.1: colour to grayscale', 'Lab 3.2: image blur', 'Lab 3.3: naive GEMM for a linear layer',
  'Lab 3.4: one thread per element, per row or per column', 'Lab 3.5: map hunt', 'What these labs leave for later'];
check('the page has its standard sections and one block per lab, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));

// ---- commands and backends
const text0 = await body();
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run 3.3', 'python3 labs/pmpp.py run 3.3 --bench', 'python3 labs/pmpp.py run 3.3 --backend emu',
  'python3 labs/pmpp.py run 3.3 --solution', 'python3 labs/pmpp.py reset 3.3 --yes'])
  check(`command shown: ${c}`, text0.includes(c));
check('the emulator is described with AddressSanitizer and three thread orders',
  text0.includes('AddressSanitizer') && text0.includes('three fixed thread orders (in order, reversed, shuffled)'));
check('the race flavour of the emulator is described, with the labs built for it',
  text0.includes('--backend race') && text0.includes('ThreadSanitizer') && text0.includes('labs 3.3, 3.4 and 3.5'));
check('the CUDA/HIP caveat and how to help are stated',
  text0.includes('front-end-checked') && text0.includes('has never been compiled') && text0.includes('bring back the report'));
check('the page says its numbers are measurements, not expected values', text0.includes('None is an expected value'));
check('every lab names the files the reader edits', LABS.every((l) => ['kernel.cu', 'kernel.metal', 'launch.mm'].every((f) => text0.includes(`labs/${l.id}/work/${f}`))));
check('lab 3.1 warns against copying Fig. 3.4 and names the four differences',
  text0.includes('Do not copy Fig. 3.4') && text0.includes('the cast, the constants, the argument order and the name'));
check('lab 3.3 is named as the first half of milestone M1, with its storage convention',
  text0.includes('first half of milestone M1') && text0.includes('B holds the weight as (in, out)') && text0.includes('transpose of what PyTorch stores'));
check('what is ours is labelled (Metal files, the rectangular kernel, the map-hunt kernel)',
  (await page.$$eval('.prose aside.note.beyond', (a) => a.length)) >= 3);
const hunt = await page.$$eval('table[data-table="maphunt-tests"] tbody tr', (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim()).join(' ')));
check('lab 3.5 lists its five named tests with their sizes',
  JSON.stringify(hunt) === JSON.stringify(['cube 4 16 16 0', 'wide 4 6 20 0', 'tall 4 20 6 0', 'odd batch 5 16 16 0', 'window 4 16 16 1']), JSON.stringify(hunt));
// The table above is the driver's own list.
const mainCpp = await page.evaluate(() => fetch('/api/lab-file?path=' + encodeURIComponent('ch03/lab5_maphunt/main.cpp')).then((r) => (r.ok ? r.text() : '')));
check('  those sizes are the ones in the lab driver',
  ['{"cube", 4, 16, 16, 0', '{"wide", 4, 6, 20, 0', '{"tall", 4, 20, 6, 0', '{"odd batch", 5, 16, 16, 0', '{"window", 4, 16, 16, 1'].every((s) => mainCpp.includes(s)));

// ---- links to other pages lead to sections that exist
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch03/start', '#/ch03/3.1', '#/ch03/3.2', '#/ch03/3.3', '#/ch03/3.4', '#/ch03/labs', '#/ch03/wrap', '#/ch05/labs']);
check(`links go to sections that exist (${links.join(', ')})`, links.length >= 5 && links.every((l) => known.has(l)), links.join(', '));

// ---- status table of the reference solutions, cell by cell against the reports
const cells = await page.$$eval('table[data-table="solution-status"] td[data-lab]', (tds) =>
  tds.map((td) => ({ lab: td.dataset.lab, backend: td.dataset.backend, status: td.dataset.status, text: td.textContent.trim(), chip: td.querySelector('.chip')?.className ?? '' })));
check('status table has 5 labs x 4 backends', cells.length === 20, String(cells.length));
for (const b of BACKENDS) {
  const mine = cells.filter((c) => c.backend === b);
  const want = LABS.map((l) => solutionText(reports, b, l.id));
  const ok = mine.length === LABS.length && mine.every((c, i) => c.status === want[i] && c.text === want[i]);
  check(`status column ${b}: ${LABS.map((l, i) => `${l.num} ${want[i]}`).join('; ')}`, ok, JSON.stringify(mine));
  check(`  ${b}: a pass is green, a failure red, no run is neither`,
    mine.every((c, i) => (want[i].startsWith('verified') || want[i] === 'emulator-checked' ? c.chip === 'chip ok' : want[i].startsWith('unverified') ? c.chip === 'chip fail' : c.chip === '')), JSON.stringify(mine));
}
if (reports.metal) {
  const verified = LABS.filter((l) => labStatus(reports.metal, l.id, 'solution') === 'pass').map((l) => l.num);
  check(`Metal results from this machine's reports are in the table (verified on Metal: labs ${verified.join(', ') || 'none'})`,
    verified.every((n) => cells.find((c) => c.lab === n && c.backend === 'metal').text === 'verified on Metal'));
  check('the run line names the Metal device and the time of the run',
    text0.includes(`metal (${reports.metal.device.name}), ${reports.metal.generated.replace('T', ' ')}`));
} else console.log('note  no Metal run on this machine; the Metal-specific assertions were skipped');
await shot('status', 'table[data-table="solution-status"]');

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
    if (rs.some((x) => !x.built)) return row[b].includes('does not build');
    if (s === 'none') return row[b].includes('not started');
    return row[b].includes(`${s === 'pass' ? 'passes,' : 'not passing:'} ${passed} of ${total} tests`);
  });
  check(`lab ${l.num}: status of your work matches the reports on every backend`, ok, JSON.stringify(row));
}

// ---- predictions: commit, then the measured value from the reports appears with a comparison
const secs = (set, name, where) => set.rows.find((r) => r.name === name && where(r))?.seconds;
const ratio = (id, name, top, bottom) =>
  preferred(reports, id).flatMap((s) => {
    const a = secs(s, name, top);
    const b = secs(s, name, bottom);
    return a && b ? [{ backend: s.backend, value: a / b }] : [];
  });
const expected = {
  tall: ratio(LABS[0].id, 'grayscale', (r) => r.block_w === 1 && r.block_h === 256, (r) => r.block_w === 256 && r.block_h === 1),
  radius: ratio(LABS[1].id, 'blur', (r) => r.radius === 4, (r) => r.radius === 1),
  doubling: ratio(LABS[2].id, 'square', (r) => r.width === 1024, (r) => r.width === 512),
  row: ratio(LABS[3].id, 'granularity', (r) => r.mode === 'row', (r) => r.mode === 'element'),
};
const predictions = [
  { key: 'tall', lab: '3.1', id: 'ch03/labs/p-lab1-tall-wide', find: 'tall blocks divided by', guess: 1, cmd: 'run 3.1 --bench' },
  { key: 'radius', lab: '3.2', id: 'ch03/labs/p-lab2-radius', find: 'reads 81 values', guess: 9, cmd: 'run 3.2 --bench' },
  { key: 'doubling', lab: '3.3', id: 'ch03/labs/p-lab3-doubling', find: 'multiplies the arithmetic by 8', guess: 8, cmd: 'run 3.3 --bench' },
  { key: 'row', lab: '3.4', id: 'ch03/labs/p-lab4-row', find: 'one thread per row divided by', guess: 30, cmd: 'run 3.4 --bench' },
];
const findPredict = (find) =>
  page.evaluateHandle((find) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.kind')?.textContent === 'Predict first' && e.textContent.includes(find)), find);
const commit = async (el, guess) => {
  const input = await el.$('input[aria-label="your prediction"]');
  await input.focus();
  await input.type(String(guess));
  await el.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Commit prediction').click());
  await pause(150);
};
for (const p of predictions) {
  const el = (await findPredict(p.find)).asElement();
  if (!el) { check(`lab ${p.lab}: prediction block present`, false); continue; }
  check(`lab ${p.lab}: nothing is revealed before a prediction is committed`, (await el.$('[data-measured]')) === null);
  await commit(el, p.guess);
  const shown = await el.$$eval('[data-measured]', (ps) => ps.map((x) => ({ measured: x.dataset.measured, backend: x.dataset.backend ?? '', bold: x.querySelector('b')?.textContent ?? '', text: x.textContent })));
  const want = expected[p.key];
  if (want.length) {
    const ok = want.every((w) => {
      const s = shown.find((x) => x.backend === w.backend);
      return s && s.measured === fmt(w.value) && s.bold === fmt(w.value) && s.text.includes(compare(p.guess, w.value));
    });
    check(`lab ${p.lab}: measured value from /api/reports is shown (${want.map((w) => `${w.backend} ${fmt(w.value)}`).join(', ')}) and compared with the prediction of ${p.guess}: "${compare(p.guess, want[0].value)}"`,
      ok && shown.length === want.length, JSON.stringify(shown));
    check(`  lab ${p.lab}: the reveal says whose kernel was measured and gives the two times`,
      shown.every((s) => /with (your kernel|the reference solution)/.test(s.text) && (s.text.match(/ ms/g) ?? []).length === 2), JSON.stringify(shown));
  } else {
    check(`lab ${p.lab}: no measurement in the reports, and the reveal says what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes('No measurement on this machine yet') && shown[0].text.includes(p.cmd), JSON.stringify(shown));
  }
  check(`  lab ${p.lab}: the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${p.guess}`));
  if (p.key === 'tall') await shot('predict-lab1', 'section.bench[aria-label="Lab 3.1 on this machine"]');
}

// ---- lab 3.5: the prediction is a count of failing named tests, read from the reader's last run on a device
{
  const NAMED = ['cube', 'wide', 'tall', 'odd batch', 'window'];
  const want = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race').flatMap((b) => {
    const tests = results(reports[b], LABS[4].id, 'work').flatMap((x) => x.tests);
    const of = (n) => tests.find((t) => t.case.startsWith(n + ':'));
    if (!NAMED.every(of)) return [];
    return [{ backend: b, failed: NAMED.filter((n) => !of(n).pass), untouched: reports[b].work_state?.[LABS[4].id] === 'untouched' }];
  });
  const el = (await findPredict('of the five named tests you expect to fail')).asElement();
  check('lab 3.5: prediction block present, nothing revealed before committing', !!el && (await el.$('[data-measured]')) === null);
  await commit(el, 2);
  const shown = await el.$$eval('[data-measured]', (ps) => ps.map((x) => ({ measured: x.dataset.measured, backend: x.dataset.backend ?? '', bold: x.querySelector('b')?.textContent ?? '', text: x.textContent })));
  if (want.length) {
    const ok = want.every((w) => {
      const s = shown.find((x) => x.backend === w.backend);
      return s && s.measured === String(w.failed.length) && s.bold === String(w.failed.length) && w.failed.every((n) => s.text.includes(n)) &&
        s.text.includes(w.untouched ? 'with work/ still as shipped' : 'after work/ had been edited') &&
        s.text.includes(w.failed.length === 2 ? 'That is the number you predicted.' : 'That is not the number you predicted.');
    });
    check(`lab 3.5: the reveal counts the named tests that failed in the reports (${want.map((w) => `${w.backend}: ${w.failed.length} of 5, ${w.failed.join(', ') || 'none'}`).join('; ')})`,
      ok && shown.length === want.length, JSON.stringify(shown));
  } else {
    check('lab 3.5: no device run of the work in the reports, and the reveal says what to run',
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes('python3 labs/pmpp.py run 3.5'), JSON.stringify(shown));
  }
  await pause(700); // the app saves 400 ms after the last change
  const stored = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
  check('the five predictions were sent to /api/progress under their ids, with the values committed, and none reached the real file',
    predictions.every((p) => stored[p.id]?.value === p.guess) && stored['ch03/labs/p-lab5-failing']?.value === 2 &&
    Object.values(stored).every((a) => a.correct === undefined), JSON.stringify(stored).slice(0, 300));
  await shot('predict-lab5', 'table[data-table="maphunt-tests"]');
}

// ---- results tables: every benchmark row in the reports, and nothing invented
const sets = (id) => Object.keys(reports).filter((b) => b !== 'emu').flatMap((b) => {
  const mine = labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : [];
  return [...mine, ...benchRows(reports[b], id, 'solution')];
});
const tables = [
  { lab: LABS[0], keep: (r) => r.name === 'grayscale', cells: (r) => [`${r.block_w} × ${r.block_h}`, fmt(r.seconds * 1000), fmt(r.mpix_per_s)] },
  { lab: LABS[1], keep: (r) => r.name === 'blur', cells: (r) => [String(r.radius), String((2 * r.radius + 1) ** 2), fmt(r.seconds * 1000), fmt(r.mpix_per_s)] },
  { lab: LABS[2], keep: (r) => r.name === 'square' || r.name === 'linear_layer', cells: (r) => [fmt(r.seconds * 1000), fmt(r.gflops), ...(r.name === 'linear_layer' ? [fmt(r.rows_per_s)] : [])] },
  { lab: LABS[3], keep: (r) => r.name === 'granularity', cells: (r) => [String(r.width), fmt(r.seconds * 1000), fmt(r.gflops), fmt(r.time_vs_element)] },
];
for (const t of tables) {
  const rows = sets(t.lab.id).filter(t.keep);
  if (rows.length) {
    const shown = await page.$$eval(`table[data-results="${t.lab.num}"] tbody tr`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    check(`lab ${t.lab.num}: results table has the ${rows.length} benchmark row(s) from the reports, with their figures`,
      shown.length === rows.length && rows.every((r, i) => t.cells(r).every((c) => shown[i].includes(c))), JSON.stringify(shown));
  } else {
    const empty = await page.$eval(`[data-results-empty="${t.lab.num}"]`, (e) => e.textContent).catch(() => '');
    check(`lab ${t.lab.num}: no benchmark rows in the reports, and the empty state says what to run`, empty.includes('./verify.sh') && (await page.$(`table[data-results="${t.lab.num}"]`)) === null, empty);
  }
}
check('lab 3.5 says it has no benchmark and that the kernel is meant to fail as shipped',
  (await page.$eval('section.bench[aria-label="Lab 3.5 on this machine"]', (e) => e.textContent)).includes('This lab has no benchmark') &&
  (await body()).includes('As shipped, the kernel is meant to fail'));
await shot('lab3', 'section.bench[aria-label="Lab 3.3 on this machine"]');

// ---- reference solutions: hidden by default, then the three real files, with computed status
check('no reference solution is shown before it is asked for', (await page.$$('figure.code')).length === 0);
const same = (a, b) => a.replace(/\s+$/g, '') === b.replace(/\s+$/g, '');
for (const l of LABS) {
  const P = `p[data-solution="${l.num}"] button`;
  check(`lab ${l.num}: toggle starts collapsed`, (await page.$eval(P, (b) => b.getAttribute('aria-expanded') + '|' + b.textContent.trim())) === 'false|Show the reference solution');
  await page.click(P);
  await page.waitForSelector('figure.code .tabs button');
  const files = {};
  for (const f of ['kernel.cu', 'kernel.metal', 'launch.mm'])
    files[f] = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => (r.ok ? r.text() : 'MISSING')), `ch03/${l.dir}/solution/${f}`);
  check(`lab ${l.num}: the three solution files exist in the repository`, Object.values(files).every((t) => t !== 'MISSING' && t.length > 100));
  const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
  check(`lab ${l.num}: tabs are CUDA, Metal kernel, Metal host`, tabs.join('|') === 'CUDA|Metal kernel|Metal host', tabs.join('|'));
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
    const shown = await page.$eval('figure.code pre', (e) => e.textContent);
    const tabStatus = await page.$eval('figure.code .tabs .status', (e) => e.textContent);
    check(`lab ${l.num}: ${tabs[i]} tab is solution/${f} verbatim (${files[f].split('\n').length - 1} lines), status "${status}"`, same(shown, files[f]) && tabStatus === status, tabStatus);
  }
  if (l.num === '3.3') {
    await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')][0].click());
    await pause(80);
    check('  lab 3.3: tier colours are applied to the real file', (await page.$$('figure.code .v-global')).length > 0 && (await page.$$('figure.code .v-reg')).length > 0);
    await shot('solution-lab3', `p[data-solution="${l.num}"]`);
  }
  await page.click(P);
  await pause(80);
  check(`lab ${l.num}: the toggle hides it again`, (await page.$$('figure.code')).length === 0);
}

// ---- phone width: tables scroll inside their own boxes, the page does not scroll sideways
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(200);
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'section.bench[aria-label="Lab 3.4 on this machine"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
