// Browser check for the chapter 19 labs page: standard sections and commands, the status
// table, the race and GEMM-source lines, the computed layer table, the three predictions
// against this machine's measurements, the results and against-direct tables, llama.cpp's
// rule applied to the five layers, the project line and the reference-solution toggle.
// Everything expected is computed here from /api/reports, /api/lab-file and the models,
// so the check holds on any machine, with or without runs.
// Usage: APP_URL=http://localhost:<port> node e2e/ch19-labs.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';
import { expansion, gemmShape } from '../src/sim/unfold.ts';
import { BOOK_DEFAULTS, llamaCuda } from '../src/sim/convlib.ts';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file, and start from an empty one so that no
// prediction is already committed whatever the reader has done.
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
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

// ---- the same reading of the reports that the page makes (app/src/lib/reports.ts, LabKit)
const ID = 'ch19/lab1_convgemm';
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
const solutionText = (reports, b) => {
  const s = labStatus(reports[b], ID, 'solution');
  return s === 'pass' ? (b === 'emu' ? 'emulator-checked' : `verified on ${NAME[b]}`) : s === 'fail' ? 'unverified: failed on this machine' : 'no run on this machine';
};
const real = (reports) => Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const preferred = (reports) =>
  real(reports).flatMap((b) => {
    const mine = labStatus(reports[b], ID, 'work') === 'pass' ? benchRows(reports[b], ID, 'work') : [];
    const rows = mine.length ? mine : benchRows(reports[b], ID, 'solution');
    return rows.length ? [{ backend: b, rows, variant: mine.length ? 'work' : 'solution' }] : [];
  });
const allRows = (reports) =>
  real(reports).flatMap((b) => [...(labStatus(reports[b], ID, 'work') === 'pass' ? benchRows(reports[b], ID, 'work') : []), ...benchRows(reports[b], ID, 'solution')]);
const compare = (predicted, measured) => {
  const r = predicted / measured;
  if (Math.abs(r - 1) <= 0.1) return 'Your prediction was within 10% of the measurement.';
  return r < 1 ? `The measurement is ${fmt(1 / r)} times your prediction.` : `Your prediction is ${fmt(r)} times the measurement.`;
};
const row = (rows, name, shape) => rows.find((r) => r.name === name && r.shape === shape);

// The five layers of main.cpp (Labs.test.ts ties the page's copy to the file).
const LAYERS = {
  early: { N: 8, C: 3, H: 224, W: 224, M: 64, K: 3 },
  middle: { N: 8, C: 64, H: 56, W: 56, M: 64, K: 3 },
  late: { N: 8, C: 256, H: 14, W: 14, M: 256, K: 3 },
  pointwise: { N: 8, C: 256, H: 14, W: 14, M: 256, K: 1 },
  small: { N: 64, C: 6, H: 14, W: 14, M: 16, K: 5 },
};

await goto(page, 'ch19/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const text0 = await page.$eval('.prose', (e) => e.textContent);

// ---- structure and standard sections
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Labs" of chapter 19', h1.includes('Labs'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['Running them', 'The four backends', 'Verification status of the reference solutions', 'Lab 19.1: a convolutional layer two ways'];
check('the page has its sections in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
const h3 = await page.$$eval('.prose h3', (hs) => hs.map((h) => h.textContent.trim()));
check('headings are in sentence case', [...h2, ...h3].every((h) => !/ [A-Z][a-z]+ [A-Z][a-z]/.test(h.replace(/Lab \d+\.\d+/, ''))), JSON.stringify(h3));
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run 19.1', 'python3 labs/pmpp.py run 19.1 --bench', 'python3 labs/pmpp.py run 19.1 --backend emu',
  'python3 labs/pmpp.py run 19.1 --backend race', 'python3 labs/pmpp.py reset 19.1 --yes', 'PMPP_CH19_GEMM=work'])
  check(`command shown: ${c}`, text0.includes(c));
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('no em dash used as an aside', !/ — /.test(text0));
const misplaced = await page.$$eval('.prose > .note', (ns) => ns.filter((n) => n.previousElementSibling?.tagName !== 'P').map((n) => n.textContent.slice(0, 60)));
check('every margin note follows a paragraph', misplaced.length === 0, misplaced.join(' | '));

// ---- status table, cell by cell against the reports
const cells = await page.$$eval('table[data-table="solution-status"] td[data-lab]', (tds) =>
  tds.map((td) => ({ lab: td.dataset.lab, backend: td.dataset.backend, status: td.dataset.status, text: td.textContent.trim() })));
check('status table has 1 lab x 4 backends', cells.length === 4, String(cells.length));
for (const b of BACKENDS) {
  const c = cells.find((x) => x.backend === b);
  const want = solutionText(reports, b);
  check(`status ${b}: ${want}`, c && c.status === want && c.text === want, JSON.stringify(c));
}
await shot('status', 'table[data-table="solution-status"]');

// ---- race line
const race = await page.$eval('[data-race-status]', (e) => ({ s: e.dataset.raceStatus, w: e.dataset.raceWork, text: e.textContent }));
const rs = labStatus(reports.race, ID, 'solution');
const rw = labStatus(reports.race, ID, 'work');
check(`race line: solution ${rs}, work ${rw}, from the reports`, race.s === rs && race.w === rw, JSON.stringify(race));
if (rs === 'pass') {
  const n = results(reports.race, ID, 'solution').flatMap((x) => x.tests).length;
  check(`  race line counts ${n} tests and says emulator-checked`, race.text.includes(`all ${n} tests`) && race.text.includes('emulator-checked'), race.text);
}

// ---- which GEMM
const gemm = await page.$eval('[data-gemm-source]', (e) => ({ src: e.dataset.gemmSource, lab64: e.dataset.lab64 }));
const sources = [...new Set(real(reports).flatMap((b) => ['work', 'solution'].flatMap((v) => benchRows(reports[b], ID, v).map((r) => r.gemm_source).filter((g) => typeof g === 'string'))))];
const lab64 = [...real(reports), 'emu'].filter((b) => labStatus(reports[b], 'ch06/lab4_linear', 'work') === 'pass');
check(`GEMM source read from the bench rows (${sources.join(',') || 'none'})`, gemm.src === (sources.join(',') || 'none'), JSON.stringify(gemm));
check(`lab 6.4 status read from the reports (${lab64.join(',') || 'none'})`, gemm.lab64 === (lab64.join(',') || 'none'), JSON.stringify(gemm));

// ---- the layer table: computed from the models, never measured
const layers = await page.$$eval('table[data-table="layers"] tbody tr', (trs) =>
  trs.map((tr) => ({ id: tr.dataset.layer, exp: tr.querySelector('[data-col="expansion"]').textContent, tiles: tr.querySelector('[data-col="tiles"]').textContent, phases: tr.querySelector('[data-col="phases"]').textContent, text: tr.textContent })));
check('the layer table has the five layers in order', layers.map((l) => l.id).join() === Object.keys(LAYERS).join(), JSON.stringify(layers.map((l) => l.id)));
for (const l of layers) {
  const s = LAYERS[l.id];
  const g = gemmShape(s, { tile: 16 });
  const tiles = `${Math.ceil(g.rows / 16)} × ${Math.ceil(g.cols / 16)}`;
  check(`  ${l.id}: expansion ${expansion(s).toFixed(2)}, ${tiles} tiles, ${g.phases} phases, GEMM ${g.rows} × ${g.inner} × ${g.cols}`,
    l.exp === expansion(s).toFixed(2) && l.tiles === tiles && l.phases === String(g.phases) && l.text.includes(`${g.rows} × ${g.inner} × ${g.cols}`), JSON.stringify(l));
}
await shot('layers', 'table[data-table="layers"]');

// ---- predictions: commit, then the measured value from the reports appears with a comparison
const secs = (r) => (r && typeof r.seconds === 'number' ? r.seconds : undefined);
const speedup = (layer) => preferred(reports).flatMap((s) => {
  const a = secs(row(s.rows, 'direct', layer));
  const b = secs(row(s.rows, 'unroll_gemm', layer));
  return a && b ? [{ backend: s.backend, value: a / b }] : [];
});
const share = preferred(reports).flatMap((s) => {
  const g = row(s.rows, 'unroll_gemm', 'early');
  return g && g.seconds && g.unroll_seconds ? [{ backend: s.backend, value: (100 * g.unroll_seconds) / g.seconds }] : [];
});
const predictions = [
  { name: 'late', find: 'The late layer. Direct time divided by', guess: 1.5, want: speedup('late') },
  { name: 'early', find: 'The same ratio on the early layer', guess: 0.8, want: speedup('early') },
  { name: 'unroll share', find: 'what percentage of unroll + GEMM', guess: 20, want: share },
];
const blocks = await page.$$eval('.q', (qs) => qs.filter((e) => e.querySelector('.kind')?.textContent === 'Predict first').length);
check('the page has three predictions, none committed at the start', blocks === 3 && (await page.$$('[data-measured]')).length === 0, String(blocks));
for (const p of predictions) {
  const el = (await page.evaluateHandle((find) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.kind')?.textContent === 'Predict first' && e.textContent.includes(find)), p.find)).asElement();
  if (!el) { check(`${p.name}: prediction block present`, false); continue; }
  const input = await el.$('input[aria-label="your prediction"]');
  check(`${p.name}: nothing is revealed before a prediction is committed`, !!input && (await el.$('[data-measured]')) === null);
  if (!input) continue;
  await input.type(String(p.guess));
  await el.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Commit prediction').click());
  await pause(150);
  const shown = await el.$$eval('[data-measured]', (ps) => ps.map((x) => ({ measured: x.dataset.measured, backend: x.dataset.backend ?? '', text: x.textContent })));
  if (p.want.length) {
    const ok = p.want.every((w) => {
      const s = shown.find((x) => x.backend === w.backend);
      return s && s.measured === fmt(w.value) && s.text.includes(compare(p.guess, w.value)) && /with (your kernel|the reference solution)/.test(s.text);
    });
    check(`${p.name}: measured value from /api/reports (${p.want.map((w) => `${w.backend} ${fmt(w.value)}`).join(', ')}) compared with ${p.guess}`, ok && shown.length === p.want.length, JSON.stringify(shown));
  } else {
    check(`${p.name}: no measurement in the reports, and the reveal says what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes('python3 labs/pmpp.py run 19.1 --bench'), JSON.stringify(shown));
  }
  check(`  ${p.name}: the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${p.guess}`));
}
await shot('predict', 'section.bench[aria-label="Lab 19.1 on this machine"]');

// ---- results and against-direct tables
const rows = allRows(reports);
if (rows.length) {
  const shown = await page.$$eval('table[data-results="19.1"] tbody tr', (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
  check(`results table has the ${rows.length} benchmark row(s) from the reports, with their times`,
    shown.length === rows.length && rows.every((r, j) => shown[j][2] === r.shape && shown[j][4] === fmt(r.seconds * 1000)), JSON.stringify(shown.slice(0, 2)));
  check('  unroll + GEMM rows carry the unroll share and the GEMM source', rows.every((r, j) =>
    r.name !== 'unroll_gemm' || (shown[j][8] === `${fmt((100 * r.unroll_seconds) / r.seconds)}%` && shown[j][11] === String(r.gemm_source))), JSON.stringify(shown.find((s) => s[3] === 'unroll + GEMM')));
  const against = await page.$$eval('table[data-against] tbody tr', (trs) => trs.map((tr) => ({ b: tr.dataset.backend, layer: tr.dataset.layer, g: tr.querySelector('[data-col="unroll_gemm"]').textContent, i: tr.querySelector('[data-col="implicit"]').textContent })));
  const want = preferred(reports).flatMap((s) => Object.keys(LAYERS).filter((L) => row(s.rows, 'direct', L)).map((L) => {
    const d = row(s.rows, 'direct', L).seconds;
    const r = (n) => (row(s.rows, n, L) ? fmt(d / row(s.rows, n, L).seconds) : '');
    return { b: s.backend, layer: L, g: r('unroll_gemm'), i: r('implicit') };
  }));
  check(`against-direct table: ${want.length} row(s), each ratio from the same run`, JSON.stringify(against) === JSON.stringify(want), JSON.stringify(against));
} else {
  const empty = await page.$eval('[data-results-empty="19.1"]', (e) => e.textContent).catch(() => '');
  check('no benchmark rows, and both tables say so', empty.includes('./verify.sh') && (await page.$('[data-against="none"]')) !== null, empty);
}

// ---- llama.cpp's rule on the five layers, from the model
const llama = await page.$$eval('[data-llama-layer]', (s) => s.map((x) => [x.dataset.llamaLayer, x.dataset.llamaPath]));
const wantLlama = Object.entries(LAYERS).map(([id, s]) => [id, llamaCuda({ ...s, ...BOOK_DEFAULTS }).path]);
check(`llama.cpp's choice per layer matches the model (${wantLlama.map((x) => x.join(' ')).join(', ')})`, JSON.stringify(llama) === JSON.stringify(wantLlama), JSON.stringify(llama));
check('  the answer says none goes to the direct kernel, with C·K·K = 150 for the small layer',
  !wantLlama.some((x) => x[1] === 'direct') && text0.includes('None of the five goes to its direct kernel') && text0.includes('C·K·K = 150'));

// ---- the project
check('the project section says no milestone and that M2 is reused unchanged', text0.includes('This chapter adds no milestone') && text0.includes('launchLinear'));

// ---- the reference solution: hidden by default, then the three real files, with computed status
check('no reference solution is shown before it is asked for', (await page.$$('figure.code')).length === 0);
const same = (a, b) => a.replace(/\s+$/g, '') === b.replace(/\s+$/g, '');
const P = 'p[data-solution="19.1"] button';
check('the toggle starts collapsed', (await page.$eval(P, (b) => b.getAttribute('aria-expanded') + '|' + b.textContent.trim())) === 'false|Show the reference solution');
await page.click(P);
await page.waitForSelector('figure.code .tabs button');
const files = {};
for (const f of ['kernel.cu', 'kernel.metal', 'launch.mm'])
  files[f] = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => (r.ok ? r.text() : 'MISSING')), `ch19/lab1_convgemm/solution/${f}`);
check('the three solution files exist in the repository', Object.values(files).every((t) => t !== 'MISSING' && t.length > 100));
const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
check('tabs are CUDA, Metal kernel, Metal host', tabs.join('|') === 'CUDA|Metal kernel|Metal host', tabs.join('|'));
const st = (b) => labStatus(reports[b], ID, 'solution');
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
  check(`${tabs[i]} tab is solution/${f} verbatim, status "${status}"`, same(shown, files[f]) && tabStatus === status, tabStatus);
}
await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')][0].click());
await pause(80);
check('  the tensors are coloured as global memory, indices and sums as registers', (await page.$$('figure.code .v-global')).length > 0 && (await page.$$('figure.code .v-reg')).length > 0);
await shot('solution', 'p[data-solution="19.1"]');
await page.click(P);
await pause(80);
check('the toggle hides it again', (await page.$$('figure.code')).length === 0);

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch19/19.2', '#/ch19/19.3', '#/ch19/19.4', '#/ch19/wrap', '#/ch05/5.1', '#/ch07/7.5']);
check(`links go to known routes (${links.join(', ')})`, links.length >= 4 && links.every((l) => known.has(l)), links.join(', '));

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch19/labs');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'table[data-table="layers"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
