// Browser check for the appendix C labs page: commands (with the lab named by its
// directory; the runner also resolves "C.1"), the status table, the one lab
// block, its three predictions and one question, the table of what the benchmark
// reports first, the results table and the reference-solution toggle. Everything
// expected is computed here from /api/reports and /api/lab-file, so the check holds on
// any machine, with or without runs.
// Usage: node e2e/appC-labs.mjs [screenshot-prefix] [--dark]
// Environment: APP_URL (default http://localhost:5173), E2E_WIDTH (default 1440).
import { open, goto } from './browser.mjs';
import { paradigm, parseParadigmRow } from '../src/sim/umparadigm.ts';
import { ratioFromProbe } from '../src/sim/zerocopy.ts';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: Number(process.env.E2E_WIDTH || 1440), height: 1000 });
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

// ---- the same reading of the reports that the page makes (lib/reports.ts, components/LabKit.tsx)
const LAB = { num: 'appC/lab1', id: 'appC/lab1_feeding', dir: 'lab1_feeding', name: 'three ways to feed a kernel' };
const RUN = 'python3 labs/pmpp.py run appC/lab1';
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
// One set of benchmark rows per backend other than emu: the reader's own if it passed and was timed, else the reference.
const preferred = (reports, id) =>
  Object.keys(reports).filter((b) => b !== 'emu').flatMap((b) => {
    const mine = labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : [];
    const rows = mine.length ? mine : benchRows(reports[b], id, 'solution');
    return rows.length ? [{ backend: b, rows }] : [];
  });
const compare = (predicted, measured) => {
  if (!(predicted > 0)) return 'not a positive ratio';
  const r = predicted / measured;
  if (Math.abs(r - 1) <= 0.1) return 'Your prediction was within 10% of the measurement.';
  return r < 1 ? `The measurement is ${fmt(1 / r)} times your prediction.` : `Your prediction is ${fmt(r)} times the measurement.`;
};
const feed = (rows, workload, path) => rows.find((r) => r.name === 'feed' && r.workload === workload && r.path === path);
const ms = (s) => `${fmt(s * 1000)} ms`;

await goto(page, 'appC/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- page structure
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['Running them', 'The four backends', 'Verification status of the reference solutions', 'Lab C.1: three ways to feed a kernel', 'What this lab leaves for later'];
check('the page has its standard sections and the lab block, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
const h3 = await page.$$eval('.prose h3', (hs) => hs.map((h) => h.textContent.trim()));
check('the lab block has its three parts: testing, predicting, measuring', JSON.stringify(h3) === JSON.stringify(['Testing', 'Before you benchmark, predict', 'What your machine measured']), JSON.stringify(h3));
check('the page title is "Labs"', (await page.$eval('.page h1', (e) => e.textContent)).includes('Labs'));

// ---- commands, backends, what is ours
const text0 = await body();
for (const c of ['./verify.sh', RUN, `${RUN} --bench`, `${RUN} --backend emu`, `${RUN} --solution`, 'python3 labs/pmpp.py reset appC/lab1 --yes'])
  check(`command shown: ${c}`, text0.includes(c));
check('the page says the runner also accepts C.1', text0.includes('accepts the appendix number') && text0.includes('pmpp.py run C.1'));
check('the emulator is described with AddressSanitizer and three thread orders', text0.includes('AddressSanitizer') && text0.includes('three fixed thread orders (in order, reversed, shuffled)'));
check('the page says the lab is not built for the race flavour', text0.includes('This lab is not built for race'));
check('the CUDA/HIP caveat and how to help are stated', text0.includes('front-end-checked') && text0.includes('has never been compiled') && text0.includes('bring back the report'));
check('the page says its numbers are measurements, not expected values', text0.includes('None is an expected value'));
check('the two files the reader edits are named', ['kernel.cu', 'launch.mm'].every((f) => text0.includes(`labs/${LAB.id}/work/${f}`)) && !text0.includes(`labs/${LAB.id}/work/kernel.metal`));
const paths = await page.$$eval('table[data-table="paths"] tbody tr', (trs) => trs.map((tr) => tr.children[0].textContent.trim()));
check('the paths table has the six feed functions, CUDA family first', paths.join(',') === 'feedExplicit,feedMapped,feedManaged,feedPrivateCopy,feedNoCopy,feedShared', paths.join(','));
check('the book\'s second-pointer procedure is named as out of date, with the test that checks it',
  text0.includes('is the 2009 one') && text0.includes('the kernel received the very pointer produce wrote through'));
check('what the tests cannot see is listed: the synchronise, the zero start, leaks, speed',
  text0.includes('A missing synchronise before consume') && text0.includes('A result that does not start at zero') && text0.includes('Whether a path is fast'));
check('the 256 MiB cap is called the lab\'s parameter, not a device fact', text0.includes('the cap is the lab\'s, not a fact about any device'));
const notes = await page.$$eval('.prose aside.note', (as) => as.map((a) => ({ cls: a.className, text: a.textContent })));
check('what is ours is labelled (the passes and Metal paths, the break-even rule)', notes.filter((n) => n.cls.includes('beyond')).length >= 2);
check('vendor facts are Today notes citing the facts files (A3, A10, A14)',
  ['A3', 'A10', 'A14'].every((id) => notes.some((n) => n.text.includes(id))), JSON.stringify(notes.map((n) => n.text.slice(0, 60))));
check('no "could not load", no "being written" and no em dash anywhere on the page', !/could not load|being written|—/i.test(text0));

// ---- links lead to sections that exist
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/appC/start', ...['C.1', 'C.2', 'C.3', 'C.4', 'C.5', 'C.6', 'C.7', 'C.8', 'C.9'].map((s) => '#/appC/' + s), '#/appC/labs', '#/appC/wrap']);
check(`links go to sections that exist (${links.join(', ')})`, links.length >= 4 && links.every((l) => known.has(l)), links.join(', '));

// ---- status table of the reference solution, cell by cell against the reports
const cells = await page.$$eval('table[data-table="solution-status"] td[data-lab]', (tds) =>
  tds.map((td) => ({ lab: td.dataset.lab, backend: td.dataset.backend, status: td.dataset.status, text: td.textContent.trim(), chip: td.querySelector('.chip')?.className ?? '' })));
check('status table has 1 lab x 4 backends', cells.length === 4, String(cells.length));
for (const b of BACKENDS) {
  const c = cells.find((x) => x.backend === b);
  const want = solutionText(reports, b, LAB.id);
  check(`status ${b}: ${want}`, c && c.status === want && c.text === want, JSON.stringify(c));
  check(`  ${b}: a pass is green, a failure red, no run is neither`,
    c && (want.startsWith('verified') || want === 'emulator-checked' ? c.chip === 'chip ok' : want.startsWith('unverified') ? c.chip === 'chip fail' : c.chip === ''), JSON.stringify(c));
}
check('the status table names the lab by its directory', (await page.$eval('table[data-table="solution-status"] tbody td', (e) => e.textContent)) === `Lab ${LAB.num}, ${LAB.name}`);

// ---- status of the reader's own work
{
  const row = await page.$$eval(`[data-work="${LAB.id}"] [data-backend]`, (s) => Object.fromEntries(s.map((x) => [x.dataset.backend, x.textContent])));
  const ok = BACKENDS.every((b) => {
    const run = reports[b];
    if (!run) return row[b].includes('no run on this machine');
    const rs = results(run, LAB.id, 'work');
    const total = rs.reduce((n, x) => n + x.tests.length, 0);
    const passed = rs.reduce((n, x) => n + x.tests.filter((t) => t.pass).length, 0);
    const s = labStatus(run, LAB.id, 'work');
    if (!rs.length) return row[b].includes('not started');
    if (rs.some((x) => !x.built)) return row[b].includes('does not build');
    if (s === 'none') return row[b].includes('not started');
    return row[b].includes(`${s === 'pass' ? 'passes,' : 'not passing:'} ${passed} of ${total} tests`);
  });
  check('status of your work matches the reports on every backend', ok, JSON.stringify(row));
}

// ---- predictions: commit, then the measured value from the reports appears with a comparison
const sets = preferred(reports, LAB.id);
const ratio = (workload, a, b) =>
  sets.flatMap((s) => {
    const ra = feed(s.rows, workload, a);
    const rb = feed(s.rows, workload, b);
    if (!ra || !ra.seconds || !rb?.seconds) return [];
    return [{ backend: s.backend, value: ra.seconds / rb.seconds, words: [ms(ra.seconds), ms(rb.seconds), workload] }];
  });
const predictions = [
  { id: 'appC/labs/p-sparse', find: 'the explicit path\'s time divided by the mapped path\'s time', guess: 4, want: ratio('sparse1', 'explicit', 'mapped'), cmd: 'machine with CUDA or ROCm' },
  { id: 'appC/labs/p-repeat', find: 'the mapped path\'s time divided by the explicit path\'s time', guess: 3, want: ratio('dense8', 'mapped', 'explicit'), cmd: 'machine with CUDA or ROCm' },
  { id: 'appC/labs/p-metal', find: 'the private-copy path\'s time divided by the shared path\'s time', guess: 1.5, want: ratio('dense8', 'private', 'shared'), cmd: 'on a Mac' },
];
const findPredict = (find) =>
  page.evaluateHandle((find) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.kind')?.textContent === 'Predict first' && e.textContent.replace(/\s+/g, ' ').includes(find)), find);
const commit = async (el, guess) => {
  const input = await el.$('input[aria-label="your prediction"]');
  await input.focus();
  await input.type(String(guess));
  await el.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Commit prediction').click());
  await pause(150);
};
check('the page has three predictions', (await page.$$eval('.q .kind', (ks) => ks.filter((k) => k.textContent === 'Predict first').length)) === 3);
for (const p of predictions) {
  const el = (await findPredict(p.find)).asElement();
  if (!el) { check(`${p.id}: prediction block present`, false); continue; }
  check(`${p.id}: nothing is revealed before a prediction is committed`, (await el.$('[data-measured]')) === null);
  await commit(el, p.guess);
  const shown = await el.$$eval('[data-measured]', (ps) => ps.map((x) => ({ measured: x.dataset.measured, backend: x.dataset.backend ?? '', bold: x.querySelector('b')?.textContent ?? '', text: x.textContent })));
  if (p.want.length) {
    const ok = p.want.every((w) => {
      const s = shown.find((x) => x.backend === w.backend);
      return s && s.measured === fmt(w.value) && s.bold === fmt(w.value) && s.text.includes(compare(p.guess, w.value)) && w.words.every((x) => s.text.includes(x));
    });
    check(`${p.id}: measured value from /api/reports is shown (${p.want.map((w) => `${w.backend} ${fmt(w.value)}`).join(', ')}) with its figures and compared with ${p.guess}`,
      ok && shown.length === p.want.length, JSON.stringify(shown));
    check(`  ${p.id}: the reveal says whose kernel was measured`, shown.every((s) => /with (your kernel|the reference solution)/.test(s.text)), JSON.stringify(shown));
    if (p.id === 'appC/labs/p-metal') {
      const s = sets.find((x) => x.backend === 'metal');
      const nc = s && feed(s.rows, 'dense8', 'nocopy');
      const sh = s && feed(s.rows, 'dense8', 'shared');
      if (nc && sh) check('  p-metal: the reveal also gives no-copy against shared', shown[0].text.includes(`no-copy took ${ms(nc.seconds)}, ${fmt(nc.seconds / sh.seconds)} times shared`), shown[0].text);
    }
  } else {
    check(`${p.id}: no measurement in the reports, and the reveal says what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes('No measurement on this machine yet') && shown[0].text.includes(p.cmd), JSON.stringify(shown));
  }
  check(`  ${p.id}: the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${p.guess}`));
}
await shot('predict', 'section.bench[aria-label="What the benchmark reports before it times anything"]');

// ---- the managed-paradigm question
{
  const q = (await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Which path does feedManaged behave like on each?')))).asElement();
  check('the paradigm question is present and unanswered', !!q && (await q.$('.verdict')) === null);
  const pick = async (i) => {
    await (await q.$$('input[type=radio]'))[i].click();
    await (await q.$('button.primary')).click();
    await pause(120);
  };
  await pick(3);
  check('paradigm question: "only touched rows cross" is wrong, because machine A is limited', (await q.evaluate((e) => e.textContent)).includes('Not quite.') && (await q.evaluate((e) => e.textContent)).includes('the full paradigm'));
  await pick(0);
  check('paradigm question: explicit on A, mapped on B is right, with the facts items', (await q.evaluate((e) => e.textContent)).includes('Correct.') && (await q.evaluate((e) => e.textContent)).includes('A10') && (await q.evaluate((e) => e.textContent)).includes('B5'));
  await pause(700); // the app saves 400 ms after the last change
  const stored = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
  check('the three predictions and the answer were sent to /api/progress under their ids, and none reached the real file',
    predictions.every((p) => stored[p.id]?.value === p.guess) && stored['appC/labs/q-managed-like'] !== undefined &&
    predictions.every((p) => stored[p.id]?.correct === undefined), JSON.stringify(stored).slice(0, 400));
}

// ---- what the benchmark reports first: paradigm, probe, addresses
{
  const want = sets.flatMap((s) => {
    const outRows = [];
    const par = s.rows.find((r) => r.name === 'paradigm');
    if (par) {
      const a = parseParadigmRow(par);
      outRows.push({ key: s.backend + 'paradigm', says: a ? paradigm(a).row.name : `reported as ${par.paradigm}` });
    }
    const probe = s.rows.find((r) => r.name === 'probe');
    const rho = ratioFromProbe(probe);
    if (probe && rho != null) outRows.push({ key: s.backend + 'probe', says: `the link is ${fmt(100 * rho, 2)}% of device bandwidth here` });
    const addr = s.rows.find((r) => r.name === 'addresses');
    if (addr) outRows.push({ key: s.backend + 'addresses', says: `shared buffer: ${addr.shared_same ? 'the same value' : 'different values'}; no-copy buffer: ${addr.nocopy_same ? 'the same value' : 'different values'}` });
    return outRows;
  });
  const shown = await page.$$eval('table[data-table="instruments"] tbody tr', (trs) => trs.map((tr) => ({ key: tr.dataset.row, text: tr.textContent })));
  if (want.length)
    check(`the first table has the instrument rows from the reports (${want.map((w) => w.key).join(', ')})`,
      shown.length === want.length && want.every((w, i) => shown[i].key === w.key && shown[i].text.includes(w.says)), JSON.stringify(shown));
  else
    check('no instrument rows in the reports, and the empty state says what to run',
      shown.length === 0 && (await page.$eval('[data-instruments-empty]', (e) => e.textContent)).includes(`${RUN} --bench`));
  if (want.some((w) => w.key.endsWith('addresses')))
    check('  the addresses row is labelled observed, not documented', shown.some((s) => s.text.includes('observed, not documented')));
}

// ---- results table: every feed row in the reports, nothing invented
{
  const all = Object.keys(reports).filter((b) => b !== 'emu').flatMap((b) => {
    const mine = labStatus(reports[b], LAB.id, 'work') === 'pass' ? benchRows(reports[b], LAB.id, 'work') : [];
    return [...mine, ...benchRows(reports[b], LAB.id, 'solution')];
  }).filter((r) => r.name === 'feed');
  const best = new Map();
  for (const r of all) {
    const k = `${r.backend}/${r.variant}/${r.workload}`;
    if (!best.has(k) || r.seconds < best.get(k)) best.set(k, r.seconds);
  }
  if (all.length) {
    const shown = await page.$$eval(`table[data-results="${LAB.num}"] tbody tr`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    const heads = await page.$$eval(`table[data-results="${LAB.num}"] thead th`, (ths) => ths.map((t) => t.textContent.trim()));
    const ok = shown.length === all.length && all.every((r, i) => {
      const cells = [String(r.workload), String(r.passes), fmt(r.table_bytes / 2 ** 20), fmt(r.touched_bytes / r.table_bytes), fmt(r.seconds * 1000), fmt(r.seconds / best.get(`${r.backend}/${r.variant}/${r.workload}`))];
      return cells.every((c) => shown[i].includes(c));
    });
    check(`the results table has the ${all.length} feed rows from the reports, with their figures and the ratio to the fastest path`, ok, JSON.stringify(shown.slice(0, 2)));
    const metal = all.some((r) => r.gpu_kernel_s !== undefined);
    check(`  its GPU columns appear exactly when a row has them (${metal ? 'Metal rows present' : 'none'})`,
      heads.includes('GPU kernels, ms') === metal && heads.includes('GPU copy, ms') === metal && heads.at(-1) === '× fastest path', heads.join(' | '));
    check('  every row has a ratio of at least 1, and each workload has one row at exactly 1',
      shown.every((r) => Number(r.at(-1)) >= 1) && [...best.keys()].every((k) => shown.some((r) => r.at(-1) === '1' && r.includes(k.split('/')[2]))), JSON.stringify(shown.map((r) => r.at(-1))));
  } else {
    const empty = await page.$eval(`[data-results-empty="${LAB.num}"]`, (e) => e.textContent).catch(() => '');
    check('no feed rows in the reports, and the empty state says what to run', empty.includes('./verify.sh') && (await page.$(`table[data-results="${LAB.num}"]`)) === null, empty);
  }
  await shot('results', `section.bench[aria-label="Lab ${LAB.num} on this machine"]`);
}

// ---- reference solution: hidden by default, then the three real files, with computed status
check('no reference solution is shown before it is asked for', (await page.$$('figure.code')).length === 0);
{
  const same = (a, b) => a.replace(/\s+$/g, '') === b.replace(/\s+$/g, '');
  const P = `p[data-solution="${LAB.num}"] button`;
  check('the toggle starts collapsed', (await page.$eval(P, (b) => b.getAttribute('aria-expanded') + '|' + b.textContent.trim())) === 'false|Show the reference solution');
  await page.click(P);
  await page.waitForSelector('figure.code .tabs button');
  const files = {};
  for (const f of ['kernel.cu', 'kernel.metal', 'launch.mm'])
    files[f] = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => (r.ok ? r.text() : 'MISSING')), `appC/${LAB.dir}/solution/${f}`);
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
  check('  tier colours are applied to the real file', (await page.$$('figure.code .v-global')).length > 0 && (await page.$$('figure.code .v-reg')).length > 0);
  await shot('solution', `p[data-solution="${LAB.num}"]`);
  await page.click(P);
  await pause(80);
  check('the toggle hides it again', (await page.$$('figure.code')).length === 0);
}

// ---- margin notes follow text, never a bench, a table or a code block
const after = await page.$$eval('.prose > aside.note', (as) => as.map((a) => a.previousElementSibling?.tagName ?? ''));
check('every margin note directly follows a paragraph or a list', after.every((t) => ['P', 'UL', 'OL'].includes(t)), after.join(','));

// ---- a link to another section works
await page.evaluate(() => [...document.querySelectorAll('.prose a')].find((a) => a.getAttribute('href') === '#/appC/C.2').click());
await page.waitForFunction(() => document.querySelector('.page h1')?.textContent.includes('Zero-copy'), { timeout: 10000 });
check('the link to section C.2 opens it', (await page.$eval('.page h1', (e) => e.textContent)).includes('Zero-copy'));

// ---- phone width: tables scroll inside their own boxes, the page does not scroll sideways
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'appC/labs');
await pause(300);
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'table[data-table="paths"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
