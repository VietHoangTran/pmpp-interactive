// Browser check for section 6.4: the exercise 5 table (fill in, reveal, gcd rule),
// the bank explorer (strides, tiles, padding, one word, 2-byte elements, AMD and
// Apple presets, per-lane arithmetic), questions, and the "your device" panel against
// this machine's /api/reports, a fixture report and no reports. Expected numbers come
// from the model the page runs on (src/sim/banks.ts).
// Usage: node e2e/ch06-s64.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { EX5_STRIDES, halfCase, sameWordCase, strideCase, tileAccess } from '../src/sim/banks.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
let fakeReports = null;
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.method() === 'PUT' && r.url().includes('/api/progress')) return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  if (fakeReports && r.method() === 'GET' && r.url().includes('/api/reports')) return r.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(fakeReports) });
  return r.continue();
});
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const clickText = async (scope, label) => {
  const done = await page.evaluate((sel, label) => {
    const btn = [...document.querySelector(sel).querySelectorAll('button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, bench(scope), label);
  await pause(80);
  return done;
};
const select = async (scope, field, value) => {
  const ok = await page.evaluate((sel, field, value) => {
    const l = [...document.querySelectorAll(sel + ' label.field')].find((x) => x.textContent.trim().startsWith(field));
    const s = l?.querySelector('select');
    if (!s) return false;
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(s, String(value));
    s.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  }, bench(scope), field, value);
  await pause(80);
  return ok;
};
const slide = async (scope, aria, value) => {
  await page.$eval(`${bench(scope)} input[aria-label="${aria}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(80);
};
const cells = (sel) => page.$eval(sel, (e) => [...e.querySelectorAll('td')].map((td) => td.textContent.trim()).join('|'));
const readout = (scope) => page.$eval(`${bench(scope)} [data-readout]`, (e) => ({ ...e.dataset }));

await goto(page, 'ch06/6.4');
const X = 'Exercise 5';
const B = 'Bank explorer';
await page.waitForSelector(bench(X));

// ---- exercise 5: fill in, reveal, rule
check('reveal is disabled until all seven are filled in', !(await clickText(X, 'Reveal')));
for (const s of EX5_STRIDES) await page.type(`${bench(X)} input[aria-label="stride ${s}"]`, String(strideCase(s).passes));
await clickText(X, 'Reveal');
const ex5 = await text(bench(X));
check('the model\'s own answers score 7 of 7', ex5.includes('7 of 7 right.'), ex5);
check('the reveal lists the banks: stride 24 hits 0, 8, 16, 24', (await text(`${bench(X)} tr[data-stride="24"]`)).includes('0, 8, 16, 24'));
check('the gcd rule appears only after the reveal, with gcd(24, 32) = 8', (await page.$(`${bench(X)} [data-gcd-rule]`)) != null && ex5.includes('gcd(24, 32) = 8'));
await page.$eval(`${bench(X)} input[aria-label="stride 12"]`, (el) => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '8');
  el.dispatchEvent(new Event('input', { bubbles: true }));
});
await pause(80);
check('changing an answer hides the reveal again', (await page.$(`${bench(X)} [data-gcd-rule]`)) == null);
await clickText(X, 'Reveal');
check('a wrong entry for stride 12 is marked wrong (the answer is 4-way)', (await page.$eval(`${bench(X)} tr[data-stride="12"]`, (e) => e.dataset.result)) === 'wrong' && (await text(`${bench(X)} tr[data-stride="12"]`)).includes('4-way'));
if (out) await page.$eval(bench(X), (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-ex5.png' });

// ---- explorer: strides
await page.$eval(bench(B), (e) => e.scrollIntoView());
for (const [s, want, used] of [[32, 32, 1], [31, 1, 32], [24, 8, 4], [16, 16, 2], [7, 1, 32]]) {
  await page.evaluate((sel, s) => [...document.querySelector(`${sel} [aria-label="Exercise 5 strides"]`).querySelectorAll('button')].find((b) => b.textContent === String(s)).click(), bench(B), s);
  await pause(80);
  const r = await readout(B);
  check(`stride ${s}: ${want} pass${want === 1 ? '' : 'es'} on ${used} bank${used === 1 ? '' : 's'}`, Number(r.passes) === want && Number(r.banksUsed) === used && strideCase(s).passes === want, JSON.stringify(r));
}
await page.evaluate((sel) => [...document.querySelector(`${sel} [aria-label="Exercise 5 strides"]`).querySelectorAll('button')].find((b) => b.textContent === '24').click(), bench(B));
await slide(B, 'lane', 5);
check('lane arithmetic: lane 5 at stride 24 is word 120, bank 24', (await text(`${bench(B)} [data-lane-arith]`)).includes('a[5·24] is word 120, bank 120 mod 32 = 24'), await text(`${bench(B)} [data-lane-arith]`));
const drawn = await page.$$eval(`${bench(B)} g[data-bank]`, (gs) => gs.filter((g) => Number(g.dataset.lanes) > 0).map((g) => [Number(g.dataset.bank), Number(g.dataset.words)]));
check('the picture puts 8 words in each of banks 0, 8, 16, 24', JSON.stringify(drawn) === JSON.stringify([[0, 8], [8, 8], [16, 8], [24, 8]]), JSON.stringify(drawn));
await slide(B, 'stride', 33);
check('stride 33 (odd): one pass', Number((await readout(B)).passes) === 1);

// ---- explorer: tiles and padding
await clickText(B, '32 × 32, turned store');
let r = await readout(B);
check('32 × 32 turned store: 32 passes, one bank', Number(r.passes) === 32 && Number(r.banksUsed) === 1 && tileAccess({ T: 32, pitch: 32, orientation: 'x-first' }).passes === 32, JSON.stringify(r));
if (out) await page.$eval(bench(B), (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-tile32.png' });
await clickText(B, 'padded to 33');
r = await readout(B);
check('padded to 33: one pass, 32 banks', Number(r.passes) === 1 && Number(r.banksUsed) === 32, JSON.stringify(r));
await slide(B, 'padding columns', 2);
check('padding by two columns (pitch 34): 2 passes', Number((await readout(B)).passes) === tileAccess({ T: 32, pitch: 34, orientation: 'x-first' }).passes && Number((await readout(B)).passes) === 2);
await slide(B, 'padding columns', 0);
await select(B, 'Index order', 'y-first');
check('a[threadIdx.y][threadIdx.x] never conflicts', Number((await readout(B)).passes) === 1);
await select(B, 'Index order', 'x-first');
await select(B, 'Warp of the block', 7);
await slide(B, 'lane', 2);
check('another warp of the block: lane 2 is thread (2, 7), word 2·32 + 7 = 71, bank 7', (await text(`${bench(B)} [data-lane-arith]`)).includes('thread (tx 2, ty 7): a[2][7] is word 2·32 + 7 = 71, bank 71 mod 32 = 7'), await text(`${bench(B)} [data-lane-arith]`));
await clickText(B, '16 × 16, turned store');
r = await readout(B);
check('16 × 16 tile, 16 × 16 block: two rows of threads per warp, 8 passes on 4 banks', Number(r.passes) === tileAccess({ T: 16, pitch: 16, orientation: 'x-first' }).passes && Number(r.passes) === 8 && Number(r.banksUsed) === 4, JSON.stringify(r));

// ---- one word, 2-byte elements, other vendors
await select(B, 'Access', 'same');
r = await readout(B);
check('every lane on one word: one pass on the NVIDIA preset (broadcast)', Number(r.passes) === sameWordCase().passes && Number(r.passes) === 1 && (await text(bench(B))).includes('broadcast'), JSON.stringify(r));
await select(B, 'Access', 'half');
r = await readout(B);
check('2-byte elements: 16 banks, one pass (same word, no conflict)', Number(r.passes) === halfCase().passes && Number(r.banksUsed) === 16, JSON.stringify(r));
await select(B, 'Access', 'stride');
await slide(B, 'stride', 1);
await select(B, 'GPU', 'amd-cdna');
r = await readout(B);
check('AMD CDNA 1–3 preset, unit stride: 64 lanes on 32 banks, NVIDIA\'s arithmetic says 2', Number(r.passes) === strideCase(1, 64, 32).passes && Number(r.passes) === 2, JSON.stringify(r));
check('and the caveat says AMD serves 16 lanes per cycle', (await text(`${bench(B)} [data-caveat]`)).includes('16 lanes per cycle'));
await select(B, 'GPU', 'amd-cdna4');
check('AMD CDNA 4 preset (64 banks): unit stride is one pass', Number((await readout(B)).passes) === 1 && (await page.$$(`${bench(B)} g[data-bank]`)).length === 64);
await select(B, 'Access', 'same');
await select(B, 'GPU', 'apple');
const ap = await text(bench(B));
check('Apple preset: not documented, no count, and the advice against many-thread reads of one location', (await page.$(`${bench(B)} [data-not-documented]`)) != null && (await page.$$(`${bench(B)} g[data-bank]`)).length === 0 && ap.includes('Apple advises removing accesses to the same location'), ap.slice(0, 300));
if (out) await page.$eval(bench(B), (e) => e.scrollIntoView()), await page.screenshot({ path: out + '-apple.png' });
await select(B, 'GPU', 'nvidia');

// ---- questions
const answerNumeric = async (needle, v) => {
  const q = await page.evaluateHandle((needle) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(needle)), needle);
  await (await q.$('input[type=number]')).type(String(v));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
check('q-padcost accepts 128 bytes', (await answerNumeric('How many extra bytes of shared memory', 128)).includes('Correct.'));
check('q-ex5-24 accepts 8', (await answerNumeric('writes a[threadIdx.x*24]', 8)).includes('Correct.'));
const shift = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('What does that C expression compute')));
await shift.evaluate((q) => [...q.querySelectorAll('label')].find((l) => l.textContent.includes('a right shift by 5 + i bits')).querySelector('input').click());
await (await shift.$('button.primary')).click();
await pause(120);
check('q-shift: the parse i >> (5 + i) is accepted', (await shift.evaluate((e) => e.textContent)).includes('Correct.'));

// ---- your device, against this machine's reports
const reports = await (await fetch(BASE + '/api/reports')).json();
const gpus = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race' && !/emulator/i.test(String(reports[b].device?.name ?? '')));
const rowsOf = (b) => {
  const res = reports[b].results.filter((x) => x.lab === 'ch06/lab3_banks');
  const s = res.filter((x) => x.variant === 'solution').flatMap((x) => x.bench);
  return s.length ? s : res.filter((x) => x.variant === 'work').flatMap((x) => x.bench);
};
const probed = gpus.filter((b) => rowsOf(b).some((x) => x.name === 'bank_probe'));
const Y = 'Your device: banks';
await page.$eval(bench(Y), (e) => e.scrollIntoView());
if (probed.length) {
  const b = probed[0];
  const row32 = rowsOf(b).find((x) => x.name === 'bank_probe' && x.stride === 32);
  const cell = await cells(`${bench(Y)} [data-probe] tr[data-stride="32"]`);
  const ratio = row32?.relative_to_stride1;
  check(`your device (${b}): stride 32 shows the model's 32 passes beside the measured ratio`, cell.startsWith('32|32|') && (ratio == null || cell.includes(`${Number(ratio.toPrecision(3))}×`)), cell);
  const onlyMeasured = probed.filter((x) => x !== 'cuda');
  if (onlyMeasured.length)
    check(`non-NVIDIA devices (${onlyMeasured.join(',')}) are labelled measurements only`, (await page.$eval(`${bench(Y)} [data-measurement-only]`, (e) => e.dataset.measurementOnly)) === onlyMeasured.join(',') && (await text(bench(Y))).includes('those are measurements only'));
} else {
  check('with no lab 6.3 measurement the panel says what to run', (await text(bench(Y))).includes('python3 labs/pmpp.py run 6.3 --solution --bench'));
}
if (out) await page.screenshot({ path: out + '-device.png' });

// ---- fixture report
fakeReports = {
  fixturegpu: {
    generated: '2026-11-20', device: { name: 'Fixture GPU' },
    results: [{
      lab: 'ch06/lab3_banks', variant: 'solution', backend: 'fixturegpu', tile: 16, built: true, test_rc: 0, tests: [],
      bench: [
        ...[1, 2, 32, 33].map((s) => ({ name: 'bank_probe', stride: s, seconds: 1, ns_per_access: s === 32 ? 40 : 10, relative_to_stride1: s === 32 ? 4 : 1 })),
        { name: 'transpose', tile: 32, pitch: 32, rows: 4096, cols: 4096, seconds: 2, gb_per_s: 100 },
        { name: 'transpose', tile: 32, pitch: 33, rows: 4096, cols: 4096, seconds: 1, gb_per_s: 150 },
      ],
    }],
  },
};
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector('.page[data-ready]');
await page.waitForSelector(bench(Y));
await pause(400);
check('fixture: stride 32 row reads model 32, measured 4×', (await cells(`${bench(Y)} [data-probe] tr[data-stride="32"]`)) === '32|32|4×', await cells(`${bench(Y)} [data-probe] tr[data-stride="32"]`));
check('fixture: stride 33 row reads model 1, measured 1×', (await cells(`${bench(Y)} [data-probe] tr[data-stride="33"]`)) === '33|1|1×', await cells(`${bench(Y)} [data-probe] tr[data-stride="33"]`));
const tr = await text(`${bench(Y)} [data-transpose]`);
check('fixture: the transpose table shows 100 and 150 GB/s, padded 1.5× the unpadded', tr.includes('100') && tr.includes('150') && tr.includes('1.5×'), tr);

fakeReports = {};
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector('.page[data-ready]');
await page.waitForSelector(bench(Y));
await pause(400);
check('no reports: no measurement yet, and the command to run', (await text(bench(Y))).includes('No measurement yet') && (await page.$(`${bench(Y)} [data-probe]`)) == null);
fakeReports = null;

// ---- light theme and a narrow screen
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
check('at a narrow width nothing scrolls the page sideways', await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
if (out) {
  await page.$eval(bench(B), (e) => e.scrollIntoView());
  await page.screenshot({ path: out + '-narrow.png' });
}

const expected404 = (u) => /\/content\/ch06\/narration\/(manifest\.json|[\w-]+\.mp3)$/.test(u);
const realErrors = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', realErrors.length === 0 && missing.every(expected404), [...realErrors, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
