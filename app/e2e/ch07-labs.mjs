// Browser check for the chapter 7 labs page: status table, the four predictions against
// this machine's measurements, the results tables (with lab 7.1's model-ceiling column) and
// the reference-solution toggles. Everything expected is computed here from /api/reports and
// /api/lab-file, so the check holds on any machine, with or without runs.
// Usage: APP_URL=http://localhost:<port> node e2e/ch07-labs.mjs [screenshot-prefix] [--dark]
import { open, goto, BASE } from './browser.mjs';
import { tabStatus } from './labstatus.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file, and start from an empty one so every prediction is open.
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
  { num: '7.1', id: 'ch07/lab1_constconv', dir: 'lab1_constconv' },
  { num: '7.2', id: 'ch07/lab2_tiledconv', dir: 'lab2_tiledconv' },
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
const real = (reports) => Object.keys(reports).filter((b) => b !== 'emu');
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
const row = (set, name, where = {}) => set?.rows.find((r) => r.name === name && Object.entries(where).every(([k, v]) => r[k] === v));

await goto(page, 'ch07/labs');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const frontend = await (await fetch(BASE + '/api/frontend-check')).json();
await page.waitForFunction(() => !document.querySelector('td[data-status="loading"]'), { timeout: 10000 });
console.log(`note  backends with runs on this machine: ${Object.keys(reports).join(', ') || 'none'}`);
const body = () => page.$eval('.prose', (e) => e.textContent);

// ---- structure, commands, backends
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Labs"', h1.includes('Labs'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['Running them', 'The four backends', 'Verification status of the reference solutions',
  'Lab 7.1: the filter in constant memory, three ways', 'Lab 7.2: tiled convolution with halos', 'No project milestone in this chapter'];
check('the page has its parts, in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
const text0 = await body();
check('nothing on the page says it is being written', !/being written/i.test(text0));
for (const c of ['./verify.sh', 'python3 labs/pmpp.py run 7.1', 'python3 labs/pmpp.py run 7.1 --bench', 'python3 labs/pmpp.py run 7.1 --backend emu',
  'python3 labs/pmpp.py run 7.1 --solution', 'python3 labs/pmpp.py reset 7.1 --yes', 'python3 labs/pmpp.py run 7.2'])
  check(`command shown: ${c}`, text0.includes(c));
check('the emulator is described with AddressSanitizer and three thread orders',
  text0.includes('AddressSanitizer') && text0.includes('three fixed thread orders (in order, reversed, shuffled)'));
check('the CUDA/HIP caveat and how to help are stated',
  text0.includes('front-end-checked') && text0.includes('has never been compiled') && text0.includes('bring back the report'));
check('the page says its numbers are measurements, not expected values', text0.includes('None is an expected value'));
check('the Metal mapping of the three ways is given', ['device const float*', 'constant float*', 'constant Filter&', 'setBytes'].every((s) => text0.includes(s)));
check('the book\'s printed defects of Fig. 7.15 are left for part B, and the unsigned test is named', text0.includes('One of them is printed in the book') && text0.includes('threadIdx.x - FILTER_RADIUS + fCol >= 0'));
check('the extern rule is stated with its fact', text0.includes('extern __constant__') && text0.includes("nvcc’s default mode"));
check('every margin note follows a paragraph, never a bench or code block',
  await page.$$eval('.prose > aside.note', (as) => as.every((a) => { const p = a.previousElementSibling; return p && !p.matches('section.bench, figure.code, .wide'); })));

// ---- status table of the reference solutions, cell by cell against the reports
const cells = await page.$$eval('table[data-table="solution-status"] td[data-lab]', (tds) =>
  tds.map((td) => ({ lab: td.dataset.lab, backend: td.dataset.backend, status: td.dataset.status, text: td.textContent.trim(), chip: td.querySelector('.chip')?.className ?? '' })));
check('status table has 2 labs x 4 backends', cells.length === 8, String(cells.length));
for (const b of BACKENDS) {
  const mine = cells.filter((c) => c.backend === b);
  const want = LABS.map((l) => solutionText(reports, b, l.id));
  check(`status column ${b}: ${LABS.map((l, i) => `${l.num} ${want[i]}`).join('; ')}`,
    mine.length === LABS.length && mine.every((c, i) => c.status === want[i] && c.text === want[i]), JSON.stringify(mine));
}
await shot('status', 'table[data-table="solution-status"]');

// ---- the reader's own work
for (const l of LABS) {
  const r = await page.$$eval(`[data-work="${l.id}"] [data-backend]`, (s) => Object.fromEntries(s.map((x) => [x.dataset.backend, x.textContent])));
  const ok = BACKENDS.every((b) => {
    const run = reports[b];
    if (!run) return r[b].includes('no run on this machine');
    const rs = results(run, l.id, 'work');
    const total = rs.reduce((n, x) => n + x.tests.length, 0);
    const passed = rs.reduce((n, x) => n + x.tests.filter((t) => t.pass).length, 0);
    const s = labStatus(run, l.id, 'work');
    if (rs.some((x) => !x.built)) return r[b].includes('does not build');
    if (s === 'none') return r[b].includes('not started');
    return r[b].includes(`${s === 'pass' ? 'passes,' : 'not passing:'} ${passed} of ${total} tests`);
  });
  check(`lab ${l.num}: status of your work matches the reports on every backend`, ok, JSON.stringify(r));
}

// ---- predictions: commit, then the measured value from the reports appears with a comparison
const expected = {
  const: preferred(reports, LABS[0].id).flatMap((s) => {
    const r = row(s, 'const', { radius: 2 });
    return r?.speedup_vs_basic ? [{ backend: s.backend, value: r.speedup_vs_basic }] : [];
  }),
  byvalue: preferred(reports, LABS[0].id).flatMap((s) => {
    const a = row(s, 'byvalue', { radius: 2 })?.gflops;
    const b = row(s, 'const', { radius: 2 })?.gflops;
    return a && b ? [{ backend: s.backend, value: a / b }] : [];
  }),
  halo: preferred(reports, LABS[1].id).flatMap((s) => {
    const r = row(s, 'tiled_halo', { tile: 32, radius: 2 });
    return r?.speedup_vs_const ? [{ backend: s.backend, value: r.speedup_vs_const }] : [];
  }),
  cached: preferred(reports, LABS[1].id).flatMap((s) => {
    const c = row(s, 'tiled_cached', { tile: 32, radius: 2 })?.gflops;
    const h = row(s, 'tiled_halo', { tile: 32, radius: 2 })?.gflops;
    return c && h ? [{ backend: s.backend, value: c / h }] : [];
  }),
};
const predictions = [
  { key: 'const', id: 'ch07/labs/p-lab1-const-speedup', find: 'The model’s answer is 2', guess: 2 },
  { key: 'byvalue', id: 'ch07/labs/p-lab1-byvalue-vs-const', find: 'the by-value kernel’s GFLOP/s', guess: 1.1 },
  { key: 'halo', id: 'ch07/labs/p-lab2-halo-speedup', find: 'about 10.8 times', guess: 10.8 },
  { key: 'cached', id: 'ch07/labs/p-lab2-cached-vs-halo', find: 'the cached-halo kernel’s GFLOP/s', guess: 0.8 },
];
for (const p of predictions) {
  const q = await page.evaluateHandle((find) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.kind')?.textContent === 'Predict first' && e.textContent.includes(find)), p.find);
  const el = q.asElement();
  if (!el) { check(`${p.key}: prediction block present`, false); continue; }
  check(`${p.key}: nothing is revealed before a prediction is committed`, (await el.$('[data-measured]')) === null);
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
      return s && s.measured === fmt(w.value) && s.bold === fmt(w.value) && s.text.includes(compare(p.guess, w.value));
    });
    check(`${p.key}: measured value from /api/reports is shown (${want.map((w) => `${w.backend} ${fmt(w.value)}`).join(', ')}) and compared with ${p.guess}: "${compare(p.guess, want[0].value)}"`,
      ok && shown.length === want.length, JSON.stringify(shown));
    check(`  ${p.key}: the reveal says whose kernel was measured`, shown.every((s) => /with (your kernel|the reference solution)/.test(s.text)), JSON.stringify(shown));
  } else {
    check(`${p.key}: no measurement in the reports, and the reveal says what to run`,
      shown.length === 1 && shown[0].measured === 'none' && shown[0].text.includes('No measurement on this machine yet'), JSON.stringify(shown));
  }
  check(`  ${p.key}: the committed prediction is echoed`, (await el.evaluate((e) => e.textContent)).includes(`You predicted ${p.guess}`));
  if (p.key === 'const') await shot('predict-lab1', '.q');
}
await pause(700);
const last = saved.length ? JSON.parse(saved.at(-1)).answers ?? {} : {};
check('all four predictions were sent to /api/progress under their ids', predictions.every((p) => last[p.id] !== undefined), Object.keys(last).join(', '));

// ---- results tables: every benchmark row in the reports, and nothing invented
const sets = (id) => real(reports).flatMap((b) => {
  const mine = labStatus(reports[b], id, 'work') === 'pass' ? benchRows(reports[b], id, 'work') : [];
  return [...mine, ...benchRows(reports[b], id, 'solution')];
});
// Lab 7.1's last column: GFLOP/s over (model FLOP/B x lab 5.1's bandwidth on the same backend).
const bandwidth = Object.fromEntries(preferred(reports, 'ch05/lab1_roofline').flatMap((s) => {
  const v = row(s, 'roofline')?.bandwidth_gb_per_s;
  return v ? [[s.backend, v]] : [];
}));
for (const l of LABS) {
  const rows = sets(l.id);
  if (rows.length) {
    const shown = await page.$$eval(`table[data-results="${l.num}"] tbody tr`, (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent.trim())));
    check(`lab ${l.num}: results table has the ${rows.length} benchmark row(s) from the reports, with their GFLOP/s`,
      shown.length === rows.length && rows.every((r, i) => r.name === 'pipeline' || shown[i].includes(fmt(r.gflops))), JSON.stringify(shown.slice(0, 3)));
    if (l.num === '7.1') {
      const heads = await page.$$eval('table[data-results="7.1"] thead th', (ths) => ths.map((t) => t.textContent));
      const col = heads.indexOf('× model ceiling');
      const ratios = rows.map((r, i) => {
        const bw = bandwidth[r.backend];
        const want = bw && r.model_intensity && r.gflops ? fmt(r.gflops / (r.model_intensity * bw)) : '';
        return shown[i][col] === want;
      });
      check(`lab 7.1: the "× model ceiling" column is GFLOP/s over (model FLOP/B × lab 5.1's bandwidth) for every row (bandwidths: ${JSON.stringify(bandwidth)})`,
        col > 0 && ratios.every(Boolean), JSON.stringify(shown.map((s) => s[col])));
      const pipe = rows.findIndex((r) => r.name === 'pipeline');
      if (pipe >= 0) check('lab 7.1: the Metal pipeline row names its execution width and thread limit',
        shown[pipe][2].includes(`execution width ${rows[pipe].threadExecutionWidth}`) && shown[pipe][2].includes(`at most ${rows[pipe].maxTotalThreadsPerThreadgroup} threads`), shown[pipe][2]);
    } else {
      const constRows = rows.map((r, i) => [r, shown[i]]).filter(([r]) => r.name === 'const');
      check('lab 7.2: the provided constant-memory kernel\'s rows have no tile', constRows.every(([, s]) => s[2] === 'constant memory (provided)' && s[3] === '' && s[4] === ''), JSON.stringify(constRows.slice(0, 1)));
      const halo = rows.map((r, i) => [r, shown[i]]).filter(([r]) => r.name === 'tiled_halo');
      check('lab 7.2: every halo row carries the formula\'s FLOP/B and traffic ratio', halo.every(([r, s]) => s.includes(fmt(r.model_intensity)) && s.includes(fmt(r.model_traffic_ratio))));
    }
  } else {
    const empty = await page.$eval(`[data-results-empty="${l.num}"]`, (e) => e.textContent).catch(() => '');
    check(`lab ${l.num}: no benchmark rows in the reports, and the empty state says what to run`, empty.includes('./verify.sh'), empty);
  }
}
await shot('lab1', 'section.bench[aria-label="Lab 7.1 on this machine"]');

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
    files[f] = await page.evaluate((path) => fetch('/api/lab-file?path=' + encodeURIComponent(path)).then((r) => (r.ok ? r.text() : 'MISSING')), `ch07/${l.dir}/solution/${f}`);
  check(`lab ${l.num}: the three solution files exist in the repository`, Object.values(files).every((t) => t !== 'MISSING' && t.length > 100));
  const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
  check(`lab ${l.num}: tabs are CUDA, Metal kernel, Metal host`, tabs.join('|') === 'CUDA|Metal kernel|Metal host', tabs.join('|'));
  // LabKit's solutionTabStatus, from the shared helper (it adds this machine's CUDA front-end check).
  const cuStatus = tabStatus(reports, l.id, 'kernel.cu', frontend);
  const metalStatus = tabStatus(reports, l.id, 'kernel.metal', frontend);
  const order = [['kernel.cu', cuStatus], ['kernel.metal', metalStatus], ['launch.mm', metalStatus]];
  for (let i = 0; i < 3; i++) {
    await page.evaluate((i) => [...document.querySelectorAll('figure.code .tabs button')][i].click(), i);
    const [f, status] = order[i];
    await page.waitForFunction(() => !document.querySelector('figure.code pre').textContent.startsWith('// loading'), { timeout: 10000 });
    await pause(60);
    const shownCode = await page.$eval('figure.code pre', (e) => e.textContent);
    const tabStatus = await page.$eval('figure.code .tabs .status', (e) => e.textContent);
    check(`lab ${l.num}: ${tabs[i]} tab is solution/${f} verbatim, status "${status}"`, same(shownCode, files[f]) && tabStatus === status, tabStatus);
  }
  await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')][0].click());
  await pause(80);
  const tiers = l.num === '7.1' ? ['v-const', 'v-global', 'v-reg'] : ['v-const', 'v-global', 'v-shared', 'v-reg'];
  check(`lab ${l.num}: tier colours are applied to the real file (${tiers.join(', ')})`, (await Promise.all(tiers.map((t) => page.$$(`figure.code .${t}`)))).every((x) => x.length > 0));
  await shot(`solution-lab${l.num.slice(2)}`, `p[data-solution="${l.num}"]`);
  await page.click(P);
  await pause(80);
  check(`lab ${l.num}: the toggle hides it again`, (await page.$$('figure.code')).length === 0);
}

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch07/7.3', '#/ch07/7.4', '#/ch07/7.5', '#/ch07/7.6', '#/ch07/wrap']);
check(`links go to routes of this chapter (${links.join(', ')})`, links.length >= 5 && links.every((l) => known.has(l)), links.join(', '));

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch07/labs');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'section.bench[aria-label="Lab 7.2 on this machine"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
