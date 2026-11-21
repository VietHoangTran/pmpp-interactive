// Browser check for chapter 6, "Before you start": the measurements panel (lab 3.4's
// three granularities and lab 5.2's tiled speed-up, compared with /api/reports), the
// chapter map and labs table, all five prerequisite questions, and the page at a
// narrow width.
// Usage: node e2e/ch06-start.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
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
const M = 'section.bench[aria-label="Your measurements so far"]';
const norm = (s) => s.replace(/\s+/g, ' ').trim();

await goto(page, 'ch06/start');
await page.waitForSelector(M);
await page.waitForFunction((M) => !document.querySelector(M + ' [data-measured="loading"]'), {}, M);

// ---- measurements, against the reports the page reads
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));
const gpus = Object.keys(reports).filter((b) => b !== 'emu' && b !== 'race');
const rowsOf = (b, lab) => {
  const res = reports[b].results.filter((x) => x.lab === lab);
  const work = res.filter((x) => x.variant === 'work').flatMap((x) => x.bench);
  return work.length ? work : res.filter((x) => x.variant === 'solution').flatMap((x) => x.bench);
};
const expected = gpus
  .map((b) => {
    const g = rowsOf(b, 'ch03/lab4_granularity').filter((r) => r.name === 'granularity');
    const t = rowsOf(b, 'ch05/lab2_tiled').find((r) => r.name === 'tiled' && r.tile === 16);
    return { b, g, t };
  })
  .filter((x) => x.g.length || x.t);
const shown = await page.$$eval(M + ' [data-measured]', (els) => els.map((e) => e.getAttribute('data-measured')));
if (!expected.length) {
  check('no lab 3.4 or 5.2 measurement: the panel says so and names the commands', shown.includes('none') &&
    norm(await page.$eval(M, (e) => e.textContent)).includes('python3 labs/pmpp.py run 3.4'));
} else {
  check('one block per GPU backend with measurements, and not the race emulator', JSON.stringify(shown) === JSON.stringify(expected.map((x) => x.b)), JSON.stringify(shown));
  for (const x of expected) {
    const sel = `${M} [data-measured="${x.b}"]`;
    for (const mode of ['element', 'row', 'col']) {
      const r = x.g.find((y) => y.mode === mode);
      if (!r) continue;
      const cells = await page.$$eval(`${sel} tr[data-mode="${mode}"] td`, (t) => t.map((c) => c.textContent));
      const ms = Number(cells[1].replace(/,/g, ''));
      check(`${x.b}: lab 3.4 ${mode} time shown matches the report (${(r.seconds * 1000).toPrecision(3)} ms)`, Math.abs(ms - r.seconds * 1000) / (r.seconds * 1000) < 0.01, cells.join('|'));
    }
    const row = x.g.find((y) => y.mode === 'row');
    const col = x.g.find((y) => y.mode === 'col');
    if (row && col) {
      const want = row.seconds > col.seconds ? 'row-slower' : 'row-not-slower';
      const got = await page.$eval(`${sel} [data-row-vs-col]`, (e) => e.getAttribute('data-row-vs-col'));
      check(`${x.b}: the row-versus-column sentence follows the measurement (${want})`, got === want, got);
      const txt = norm(await page.$eval(`${sel} [data-row-vs-col]`, (e) => e.textContent));
      check(`${x.b}: and points to section 6.1`, txt.includes('6.1'), txt);
    }
    if (x.t) {
      const got = await page.$eval(`${sel} [data-tiled]`, (e) => e.getAttribute('data-tiled'));
      check(`${x.b}: lab 5.2 tile-16 speed-up is the report's speedup_vs_naive (${x.t.speedup_vs_naive})`, Number(got) === x.t.speedup_vs_naive, got);
    } else {
      check(`${x.b}: no lab 5.2 row: the panel says so`, (await page.$(`${sel} [data-tiled="none"]`)) !== null);
    }
  }
}
if (out) await page.$eval(M, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-measured.png' });

// ---- the chapter map and the labs
const mapLinks = await page.$$eval('table[data-map] a', (a) => a.map((x) => x.getAttribute('href')));
check('the map links all nine sections 6.1 to 6.9', JSON.stringify(mapLinks) === JSON.stringify(['6.1', '6.2', '6.3', '6.4', '6.5', '6.6', '6.7', '6.8', '6.9'].map((s) => '#/ch06/' + s)), mapLinks.join(' '));
const labs = await page.$$eval('table[data-labs] tr[data-lab]', (t) => t.map((r) => r.textContent));
check('five labs listed, lab 6.4 as milestone M2', labs.length === 5 && labs[3].includes('milestone M2'), labs.join(' | '));

// ---- the five questions
const answerChoice = async (promptPart, optionPart) => {
  const q = await page.evaluateHandle((p) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(p)), promptPart);
  const ok = await q.evaluate((e, o) => {
    if (!e) return false;
    const l = [...e.querySelectorAll('.opts label')].find((x) => x.textContent.includes(o));
    if (!l) return false;
    l.querySelector('input').click();
    return true;
  }, optionPart);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(100);
  return ok && (await q.evaluate((e) => e.querySelector('.verdict')?.textContent ?? ''));
};
const answerNumber = async (promptPart, value) => {
  const q = await page.evaluateHandle((p) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(p)), promptPart);
  await q.evaluate((e, v) => {
    const el = e.querySelector('input[type=number]');
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(100);
  return q.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
};
let v = await answerNumber('How many array elements apart', 4);
check('q-neighbours: a wrong answer is marked and the answer given', v.includes('Not quite. The answer is 1 elements.'), v);
v = await answerNumber('How many array elements apart', 1);
check('q-neighbours: 1 element apart is correct, and the explanation links ch. 3 §3.4', v.includes('Correct.') && v.includes('ch. 3 §3.4'), v);
v = await answerChoice('form warp 0', 'threadIdx.x 0 to 15 in rows threadIdx.y = 0 and 1');
check('q-warp2d: two 16-wide rows is correct', v && v.includes('Correct.') && v.includes('ch. 4 §4.4'), v);
v = await answerNumber('waits 396 cycles', 1);
check('q-hide: 1% is correct, and the explanation points to 6.2', v.includes('Correct.') && v.includes('6.2'), v);
v = await answerChoice('false', 'The one after the dot-product loop');
check('q-false: barrier 2 is correct', v && v.includes('Correct.') && v.includes('ch. 5 §5.4'), v);
v = await answerChoice('60% of the bandwidth', 'Tiling it through shared memory');
check('q-vecadd: tiling is marked wrong with its reason', v && v.includes('Not quite.'), v);
v = await answerChoice('60% of the bandwidth', 'wider, well-formed requests');
check('q-vecadd: wider, well-formed requests is correct', v && v.includes('Correct.') && v.includes('ch. 5 §5.1'), v);

// ---- narrow screen
await page.setViewport({ width: 390, height: 900 });
await pause(300);
const fits = await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1);
check('at 390 px nothing overflows the page', fits);
if (out) await page.$eval(M, (e) => e.scrollIntoView({ block: 'start' })), await page.screenshot({ path: out + '-narrow.png' });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
