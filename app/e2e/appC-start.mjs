// Browser check for appendix C, "Before you start": the map of who can dereference
// what (every era, every platform, every cell against the model), the p-weights
// prediction with the reader's own device memory, the warning on the two unified
// memories and the four prerequisite questions.
// Usage: node e2e/appC-start.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { ERAS, ROWS, SITES, VERDICT_LABEL, cell, changed, grid } from '../src/sim/derefmap.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
let puts = 0;
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.method() === 'PUT' && r.url().includes('/api/progress')) {
    puts++;
    return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  }
  return r.continue();
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const MAP = 'section.bench[aria-label="Who can dereference what, era by era"]';
const click = async (sel) => {
  const ok = await page.evaluate((sel) => {
    const e = document.querySelector(sel);
    if (!e || e.disabled) return false;
    e.click();
    return true;
  }, sel);
  await pause(60);
  return ok;
};
const clickText = async (scope, label) => {
  const ok = await page.evaluate(
    (scope, label) => {
      const b = [...document.querySelector(scope).querySelectorAll('button')].find((x) => x.textContent.trim() === label);
      if (!b || b.disabled) return false;
      b.click();
      return true;
    },
    scope,
    label,
  );
  await pause(60);
  return ok;
};
const cells = () =>
  page.$$eval(MAP + ' td[data-row]', (tds) => tds.map((t) => ({ row: t.dataset.row, site: t.dataset.site, v: t.dataset.verdict, changed: t.dataset.changed === '1', chip: t.querySelector('[data-chip]').textContent })));
const sameAsModel = (shown, g) =>
  shown.length === ROWS.length * SITES.length && shown.every((c) => g[c.row][c.site].verdict === c.v && VERDICT_LABEL[c.v] === c.chip);
const question = (part) => page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(part)), part);
const qText = (q) => q.evaluate((e) => e.textContent);
const pick = async (q, label) => {
  await q.evaluate((e, label) => {
    [...e.querySelectorAll('.opts label')].find((l) => l.textContent.trim().startsWith(label)).querySelector('input').click();
  }, label);
  await (await q.$('button.primary')).click();
  await pause(120);
};
const typeAnswer = async (q, v) => {
  const input = await q.$('input[type=number]');
  await input.focus();
  await input.evaluate((e) => e.select());
  await input.type(String(v));
  await (await q.$('button.primary')).click();
  await pause(120);
};

await goto(page, 'appC/start');
await page.waitForSelector(MAP);

// ---- the map: every era against the model
const strip = await page.$$eval(MAP + ' [aria-label="Eras"] button', (b) => b.map((x) => x.textContent.trim()));
check('the timeline strip has the nine eras, labelled by section', JSON.stringify(strip) === JSON.stringify(ERAS.map((e) => e.strip)), strip.join('|'));
let allEras = true;
const badEras = [];
for (const e of ERAS) {
  await click(`${MAP} button[data-era="${e.id}"]`);
  const shown = await cells();
  if (!sameAsModel(shown, grid('cuda', e.id))) {
    allEras = false;
    badEras.push(e.id);
  }
  const marked = shown.filter((c) => c.changed).map((c) => c.row + '/' + c.site).sort();
  const want = changed(e.id).map((c) => c.row + '/' + c.site).sort();
  if (JSON.stringify(marked) !== JSON.stringify(want)) {
    allEras = false;
    badEras.push(e.id + ' (changed marks)');
  }
  const head = await page.$eval(MAP + ' [data-era-head]', (h) => h.textContent);
  if (!head.includes(e.title) || (e.corrected && !head.includes(`(${e.corrected.item})`))) {
    allEras = false;
    badEras.push(e.id + ' (header)');
  }
}
check('in every CUDA era every cell, chip and "changed" mark matches derefmap.ts, and the header names the facts item', allEras, badEras.join(', '));

// Corrected dates as the reader sees them.
await click(`${MAP} button[data-era="kepler"]`);
let head = await page.$eval(MAP + ' [data-era-head]', (h) => h.textContent);
check('Kepler: the book says 2013, our sources March 2012 (A6)', head.includes('The book: 2013.') && head.includes('March 2012') && head.includes('(A6)'), head);
await click(`${MAP} button[data-era="vmm"]`);
head = await page.$eval(MAP + ' [data-era-head]', (h) => h.textContent);
check('C.9: the book says CUDA 11, our sources CUDA 10.2 (A15)', head.includes('CUDA 11') && head.includes('CUDA 10.2, November 2019 (A15)'), head);
check('the era header links to the section page', (await page.$eval(MAP + ' [data-era-head] a', (a) => a.getAttribute('href'))) === '#/appC/C.9');

// Previous and Next walk the strip.
await click(`${MAP} button[data-era="copies"]`);
check('Previous is disabled on the first era', !(await clickText(MAP, 'Previous')));
await clickText(MAP, 'Next');
await clickText(MAP, 'Next');
check('Next twice from the start lands on C.2', await page.$eval(MAP + ' button[data-era="zerocopy"]', (b) => b.getAttribute('aria-pressed') === 'true'));

// The three discoveries, through the cells and the detail panel.
await click(`${MAP} button[data-era="pascal"]`);
await click(`${MAP} td[data-row="malloc"][data-site="arg"] button`);
let detail = await page.$eval(MAP + ' [data-detail]', (d) => d.textContent);
check('Pascal: plain malloc memory is still "no" for a kernel, with the reason and A12', detail.includes(cell('cuda', 'pascal', 'malloc', 'arg').cost) && detail.includes('appC-facts A12') && detail.includes(': no'), detail);
await click(`${MAP} button[data-era="hmm"]`);
detail = await page.$eval(MAP + ' [data-detail]', (d) => d.textContent);
check('beyond the book (HMM or ATS): malloc memory faults in', detail.includes(VERDICT_LABEL.fault), detail);
await click(`${MAP} td[data-row="device"][data-site="host"] button`);
detail = await page.$eval(MAP + ' [data-detail]', (d) => d.textContent);
check('cudaMalloc memory, host code: still no, even with hardware coherence', detail.includes(cell('cuda', 'hmm', 'device', 'host').cost), detail);
if (out) await page.screenshot({ path: out + '-map-cuda.png' });

// HIP, with and without XNACK, and Metal.
await click(`${MAP} button[data-platform="hip"]`);
check('HIP hides the era strip', (await page.$$(MAP + ' [aria-label="Eras"]')).length === 0);
check('HIP cells match the model without XNACK', sameAsModel(await cells(), grid('hip', 'hmm', { xnack: false })));
await click(`${MAP} td[data-row="device"][data-site="host"] button`);
detail = await page.$eval(MAP + ' [data-detail]', (d) => d.textContent);
check('HIP: host code may read hipMalloc memory, over the interconnect (B3)', detail.includes('hipMalloc on this GPU') && detail.includes(VERDICT_LABEL.remote) && detail.includes('B3'), detail);
await page.evaluate((m) => document.querySelector(m + ' input[type=checkbox]').click(), MAP);
await pause(80);
check('HIP cells match the model with XNACK', sameAsModel(await cells(), grid('hip', 'hmm', { xnack: true })));
await click(`${MAP} button[data-platform="metal"]`);
check('Metal cells match the model', sameAsModel(await cells(), grid('metal', 'hmm')));
const rowHeads = await page.$$eval(MAP + ' tbody th', (t) => t.map((x) => x.textContent));
check('Metal rows are named by their Metal counterparts', rowHeads.includes('a private buffer') && rowHeads.includes('a threadgroup variable'), rowHeads.join('|'));
if (out) await page.screenshot({ path: out + '-map-metal.png' });
await click(`${MAP} button[data-platform="cuda"]`);

// Colour: ok and fail chips only.
const toneOk = await page.$$eval(MAP + ' td[data-row] [data-chip]', (cs) =>
  cs.every((c) => (c.dataset.chip === 'local' ? c.classList.contains('ok') : c.dataset.chip === 'no' ? c.classList.contains('fail') : !c.classList.contains('ok') && !c.classList.contains('fail'))),
);
check('green only for "yes, local", red only for "no"', toneOk);

// ---- the prediction and your device memory
const reports = await (await fetch(BASE + '/api/reports')).json();
const real = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const P = await question('8-billion-parameter');
await (await P.$('input[type=number]')).type('16');
await (await P.$('button.primary')).click();
await page.waitForFunction(() => document.querySelector('[data-yours]'), { timeout: 10000 });
const reveal = await qText(P);
check('the reveal confirms 16 GB and shows the arithmetic', reveal.includes('Right:') && reveal.includes('2 bytes'), reveal);
check('the reveal names the three answers and links C.2 and C.8', reveal.includes('Cut it into pieces') && (await P.$$eval('a', (a) => a.map((x) => x.getAttribute('href')))).join() === '#/appC/C.2,#/appC/C.8');
const yours = await P.$eval('[data-yours]', (e) => e.dataset.yours);
if (real.length) {
  const shown = await P.$$eval('[data-backend]', (ps) => ps.map((p) => ({ b: p.dataset.backend, fits: p.dataset.fits, text: p.textContent })));
  const want = real
    .map((b) => {
      const d = reports[b].device;
      const budget = Number(b === 'metal' ? d.recommendedMaxWorkingSetSize : d.totalGlobalMem);
      return budget > 0 ? { b, fits: 16e9 <= budget ? 'yes' : 'no', gb: (budget / 1e9).toFixed(1) + ' GB' } : null;
    })
    .filter(Boolean);
  check(
    'your devices: each one recorded is shown, with its own figure and the right fit verdict',
    yours === 'some' && want.every((w) => shown.some((s) => s.b === w.b && s.fits === w.fits && s.text.includes(w.gb))),
    JSON.stringify({ shown, want }),
  );
} else check('no device recorded: the page says so', yours === 'none');
if (out) await page.screenshot({ path: out + '-predict.png' });

// ---- the warning and the notes
const prose = await page.$eval('.prose', (e) => e.textContent);
check('the warning keeps the two unified memories apart', prose.includes('One name, two things') && prose.includes("CUDA's Unified Memory") && prose.includes("Apple's unified memory"));
check('no device figure is printed in the prose outside the reader\'s own report', !/\b\d+(\.\d+)?\s?(GB|TB)\/s\b/.test(prose.replace(/Bench|data-yours/g, '')));
check('every margin note directly follows text, never a bench or code', await page.evaluate(() => [...document.querySelectorAll('.prose > .note')].every((n) => ['P', 'UL', 'OL'].includes(n.previousElementSibling?.tagName) || n.previousElementSibling?.classList.contains('note'))));
check('a Simplification note sits beside the map', (await page.$$eval('.note.caveat', (n) => n.map((x) => x.textContent))).some((t) => t.includes('One cell is one documented rule')));
check('no em dash in the page', !prose.includes('—'));

// ---- the four prerequisite questions
let q = await question('printf("%f", A_d[0])');
await pick(q, 'It prints the value, slowly');
check('q-hostderef: the HIP distractor explains itself', (await qText(q)).includes('hipMalloc memory under HIP'));
await pick(q, 'It is not allowed');
check('q-hostderef: the right answer is accepted', (await qText(q)).includes('Correct.'));
q = await question('reads 1 GiB once');
await typeAnswer(q, 10);
check('q-membound: 10 ms', (await qText(q)).includes('Correct.'));
q = await question('__shared__ float tile[256]');
await pick(q, 'The threads of block 2');
check('q-sharedscope: block 2 after a barrier', (await qText(q)).includes('Correct.'));
q = await question('commits a command buffer');
await pick(q, 'Nothing: unified memory is coherent');
check('q-ordering: the coherence distractor is wrong and says why', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('removes the copy, not the ordering'));
check('answers went to the progress API (intercepted)', puts > 0, String(puts));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
