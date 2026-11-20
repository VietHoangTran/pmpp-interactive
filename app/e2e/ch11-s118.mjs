// Browser check for section 11.8: the scan-ceiling explorer (every option against
// src/sim/scantraffic.ts, imported directly), the measured device and lab 11.2 chart when
// this machine has them, the copy prediction and its reveal, the four questions, and no
// sideways scroll at phone width.
// Usage: APP_URL=http://localhost:5173 node e2e/ch11-s118.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import {
  BOOK_DEVICE,
  bestRegisterTiled,
  blockSizes,
  ceiling,
  copyGBs,
  intensityLimit,
  kernelRates,
  labRows,
  measuredBandwidth,
  scanBytes,
  threshold,
} from '../src/sim/scantraffic.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
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
// The app's number format (src/lib/reports.ts fmt), repeated here so the check stays in Node.
const fmt = (v, digits = 3) => {
  if (!isFinite(v)) return 'n/a';
  if (v === 0) return '0';
  if (Math.abs(v) >= 1000) return v.toLocaleString('en-US', { maximumFractionDigits: 0 });
  return Number(v.toPrecision(digits)).toString();
};
const B = 'section.bench[aria-label="The scan ceiling"]';
const tid = (id) => page.$eval(`${B} [data-testid="${id}"]`, (e) => e.textContent.trim());
const press = async (group, label) => {
  const ok = await page.evaluate((B, group, label) => {
    const g = document.querySelector(`${B} [aria-label="${group}"]`);
    const b = g && [...g.querySelectorAll('button')].find((x) => x.textContent.trim() === label);
    if (!b) return false;
    b.click();
    return true;
  }, B, group, label);
  await pause(80);
  return ok;
};
const selectDevice = async (value) => {
  await page.$eval(`${B} select[aria-label="device"]`, (s, v) => {
    s.value = v;
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await pause(120);
};
const expectCeiling = async (name, bw, elem, written) => {
  const b = scanBytes(elem, written);
  const want = { bytes: String(b.total), opb: fmt(intensityLimit(b.total)), cap: fmt(ceiling(bw, b.total) / 1e9) };
  const got = { bytes: await tid('bytes'), opb: await tid('opb'), cap: await tid('cap') };
  check(name, JSON.stringify(got) === JSON.stringify(want), `${JSON.stringify(got)} vs ${JSON.stringify(want)}`);
};

const reports = await (await fetch(BASE + '/api/reports')).json();

await goto(page, 'ch11/11.8');
await page.waitForSelector(B);

// ---------------------------------------------------------------- the explorer, the book's device
await expectCeiling('book, float, scan only: 8 bytes, 0.125 OP/B, 419 × 10⁹', BOOK_DEVICE.bandwidthGBs, 'float', 'scan');
check('the book\'s ceiling reads 419', (await tid('cap')) === '419');
const regime = await tid('regime');
check('memory-bound, threshold 20, about 160 times less', regime.includes('memory-bound') && regime.includes(`threshold is ${fmt(threshold(BOOK_DEVICE.tflopsNvidia, BOOK_DEVICE.bandwidthGBs))} FLOP/B`) && regime.includes('160 times'), regime);
check('half button', await press('Element type', 'half (2 bytes)'));
await expectCeiling('book, half: 4 bytes, 0.25 OP/B', BOOK_DEVICE.bandwidthGBs, 'half', 'scan');
check('flag button', await press('What is written', 'the scan and a one-byte flag'));
await expectCeiling('book, half plus flag: 5 bytes', BOOK_DEVICE.bandwidthGBs, 'half', 'scan+mask');
await press('Element type', 'float (4 bytes)');
await expectCeiling('book, float plus flag: 9 bytes (lab 11.6\'s row)', BOOK_DEVICE.bandwidthGBs, 'float', 'scan+mask');
await press('What is written', 'the scan only');
if (out) await page.screenshot({ path: out + '-book.png' });

// ---------------------------------------------------------------- custom bandwidth
await selectDevice('custom');
await page.$eval(`${B} input[aria-label="bandwidth"]`, (el) => {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '800');
  el.dispatchEvent(new Event('input', { bubbles: true }));
});
await pause(120);
await expectCeiling('custom 800 GB/s, float: 100 × 10⁹', 800, 'float', 'scan');

// ---------------------------------------------------------------- this machine
const options = await page.$$eval(`${B} select[aria-label="device"] option`, (o) => o.map((x) => x.value));
const real = Object.keys(reports).filter((b) => b !== 'emu');
const withBw = real.filter((b) => measuredBandwidth(reports[b]));
check('one device option per backend with a measured bandwidth', JSON.stringify(options.filter((o) => o !== 'book' && o !== 'custom')) === JSON.stringify(withBw), `${options} vs ${withBw}`);
for (const b of withBw) {
  await selectDevice(b);
  const m = measuredBandwidth(reports[b]);
  await expectCeiling(`${b}: ceiling from the measured ${fmt(m.gbs)} GB/s (${m.source})`, m.gbs, 'float', 'scan');
  const rows = labRows(reports[b], 'ch11/lab2_');
  const copy = copyGBs(rows);
  const sizes = blockSizes(rows);
  if (copy && kernelRates(rows, copy).length) {
    // The largest block size is shown first; every size has its own button.
    for (const size of [...sizes].reverse()) {
      if (sizes.length > 1) check(`${b}: block size ${size} button`, await press('Block size', `${size} threads`));
      const rates = kernelRates(rows.filter((r) => r.block === size), copy);
      const bars = await page.$$eval(`${B} [data-share]`, (g) => g.map((x) => Number(x.dataset.share)));
      check(`${b}, ${size} threads: the chart shows each kernel's share of the copy rate`, bars.length === rates.length && bars.every((s, i) => Math.abs(s - rates[i].share) < 1e-3), JSON.stringify(bars));
      const best = bestRegisterTiled(rates);
      check(`${b}, ${size} threads: the panel names the best register-tiled kernel`, (await tid('measured')).includes(`${Math.round(best.rate.share * 100)}%`));
    }
    const over = await page.$$eval(`${B} [data-share]`, (g) => g.filter((x) => Number(x.dataset.share) >= 0.8).map((x) => x.querySelector('rect').getAttribute('fill')));
    check(`${b}: bars at or past 80% are drawn in the success colour`, over.every((f) => f === 'var(--ok)'));
  } else {
    const bars = await page.$$eval(`${B} [data-share]`, (g) => g.length);
    check(`${b}: no lab 11.2 rows yet, and the panel says how to get them`, bars === 0 && (await tid('measured')).includes('pmpp.py run 11.2'));
  }
  if (out) {
    await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));
    await page.screenshot({ path: `${out}-${b}.png` });
  }
}

// ---------------------------------------------------------------- the prediction and its reveal
const P = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('share of your own copy kernel')));
await (await P.$('input[type=number]')).type('75');
await (await P.$('button.primary')).click();
await pause(200);
const reveal = await P.evaluate((e) => e.textContent);
const best = real.map((b) => {
  const rows = labRows(reports[b], 'ch11/lab2_');
  const copy = copyGBs(rows);
  return copy ? bestRegisterTiled(kernelRates(rows, copy)) : undefined;
}).filter(Boolean);
check('the prediction is committed', reveal.includes('You predicted 75 %'), reveal.slice(0, 120));
if (best.length) check('the reveal names the best measured share', best.every((x) => reveal.includes(`${Math.round(x.rate.share * 100)}%`)), reveal);
else check('with no lab 11.2 run, the reveal says how to measure', reveal.includes('No measurement on this machine yet') && reveal.includes('pmpp.py run 11.2'));

// ---------------------------------------------------------------- questions
const answerNumeric = async (needle, value) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return (await q.evaluate((e) => e.textContent)).includes('Correct.');
};
check('q-half: 0.25 OP/B is accepted', await answerNumeric('reads half-precision values', 0.25));
check('q-ceiling: 100 is accepted', await answerNumeric('a plain copy kernel measures 800 GB/s', 100));
check('q-topp-bytes: 9 is accepted', await answerNumeric('fused top-p kernel of lab 11.6', 9));
const q80 = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('reads its whole input twice')));
await q80.evaluate((e) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes('no redundant global accesses. It moves 12 bytes')).querySelector('input').click());
await (await q80.$('button.primary')).click();
await pause(120);
check('q-80: the two-part rule is the right answer', (await q80.evaluate((e) => e.textContent)).includes('Correct.'));

// ---------------------------------------------------------------- margin notes and narrow screens
const groups = await page.$$eval('.prose > [data-note-group]', (g) => g.length);
check('the two notes beside the 80% paragraph share one margin slot', groups === 1, String(groups));
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(300);
const wide = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
check('no sideways page scroll at 390 px', wide <= 0, String(wide));
if (out) {
  await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
