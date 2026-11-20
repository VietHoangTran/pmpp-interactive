// Browser check for appendix C, section C.2: the "What a mapped read costs"
// explorer (every link preset, both plots, the break-even), the embedding
// prediction and the questions.
// Usage: node e2e/appC-sc2.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { BOOK_RULE, H100_RATIO, better, breakEven, copyFirstCost, mappedCost, mixedTime, ratioFromProbe } from '../src/sim/zerocopy.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
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
const B = 'section.bench[aria-label="What a mapped read costs"]';
const fmt = (v, digits = 3) => (Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : Number(v.toPrecision(digits)).toString());
const setInput = async (sel, v) => {
  await page.evaluate(
    (sel, v) => {
      const e = document.querySelector(sel);
      const proto = e.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(proto, 'value').set.call(e, String(v));
      e.dispatchEvent(new Event(e.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true }));
    },
    sel,
    v,
  );
  await pause(60);
};
const txt = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' '));
const rho = () => page.$eval(B + ' [data-rho]', (e) => Number(e.dataset.rho));
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

await goto(page, 'appC/C.2');
await page.waitForSelector(B);

// ---- link presets
const opts = await page.$$eval(B + ' select[aria-label="link"] option', (o) => o.map((x) => x.value));
check("presets: the book's rule, the book's example device, custom", opts[0] === 'book' && opts[1] === 'h100' && opts[opts.length - 1] === 'custom', opts.join('|'));
check("the book's rule is 10%", (await rho()) === BOOK_RULE);
await setInput(B + ' select[aria-label="link"]', 'h100');
check("the book's example device: about 3.8%, with its two figures and A4", Math.abs((await rho()) - H100_RATIO) < 1e-12 && (await txt(B + ' [data-rho-note]')).includes('128 GB/s') && (await txt(B + ' [data-rho-note]')).includes('3.35 TB/s') && (await txt(B + ' [data-rho-note]')).includes('A4'));
await setInput(B + ' select[aria-label="link"]', 'custom');
await setInput(B + ' input[aria-label="custom ratio"]', 0.25);
check('custom: the slider sets rho', (await rho()) === 0.25);
// "Yours" comes from lab C.1's probe rows, if any.
const reports = await (await fetch(BASE + '/api/reports')).json();
const probes = Object.keys(reports).filter((b) => b !== 'metal' && b !== 'race' && b !== 'emu').flatMap((b) => {
  const row = reports[b].results.filter((x) => x.lab === 'appC/lab1_feeding' && x.variant === 'solution').flatMap((x) => x.bench).find((r) => r.name === 'probe');
  const r = ratioFromProbe(row);
  return r == null ? [] : [{ b, r }];
});
if (probes.length) {
  for (const p of probes) {
    await setInput(B + ' select[aria-label="link"]', 'yours-' + p.b);
    check(`yours (${p.b}) uses the probe row's ratio`, Math.abs((await rho()) - p.r) < 1e-12);
  }
} else check('no probe row yet: the page says how to get one', opts.every((o) => !o.startsWith('yours')) && (await txt(B + ' [data-rho-note]')).includes('No "yours" yet') && (await txt(B + ' [data-rho-note]')).includes('appC/lab1 --bench'));

// ---- first plot: the mixed kernel
await setInput(B + ' select[aria-label="link"]', 'book');
await setInput(B + ' input[aria-label="mapped fraction"]', 0.1);
let mixed = await txt(B + ' [data-mixed]');
check('f = 10% at the book\'s 10%: 1.9× without overlap, 1× with perfect overlap (q-mixed)', mixed.includes(`${fmt(mixedTime(0.1, 0.1, 'none'))}×`) && mixed.includes(`perfectly overlapped, ${fmt(mixedTime(0.1, 0.1, 'perfect'))}×`), mixed);
check('one tenth of the bytes takes more than half the time', mixed.includes('52.6%'), mixed);
await setInput(B + ' input[aria-label="mapped fraction"]', 0.05);
mixed = await txt(B + ' [data-mixed]');
check('f = 5%: 1.45× (the page\'s "almost half again")', mixed.includes(`${fmt(mixedTime(0.05, 0.1, 'none'))}×`), mixed);

// ---- second plot: in place against copy first
await setInput(B + ' input[aria-label="share touched per pass"]', 6);
let passes = await txt(B + ' [data-passes]');
check('s = 1/64 at 10%: break-even 71.1 passes (q-breakeven)', passes.includes(`Break-even at ${fmt(breakEven(1 / 64, 0.1))} passes`) && fmt(breakEven(1 / 64, 0.1)) === '71.1', passes);
for (const e of [1, 1.85, 1.9, 3]) {
  await setInput(B + ' input[aria-label="passes"]', e);
  const k = Math.max(1, Math.round(10 ** e));
  const win = await page.$eval(B + ' [data-passes]', (p) => p.dataset.winner);
  passes = await txt(B + ' [data-passes]');
  check(
    `k = ${k}: costs ${fmt(mappedCost(k, 1 / 64, 0.1))} in place and ${fmt(copyFirstCost(k, 1 / 64, 0.1))} copied, ${better(k, 1 / 64, 0.1)} wins`,
    win === better(k, 1 / 64, 0.1) && passes.includes(`= ${fmt(mappedCost(k, 1 / 64, 0.1))}.`) && passes.includes(`= ${fmt(copyFirstCost(k, 1 / 64, 0.1))}.`),
    passes,
  );
}
await setInput(B + ' input[aria-label="share touched per pass"]', 0);
await setInput(B + ' input[aria-label="passes"]', 0);
check('a dense table read once: reading in place still wins, barely', (await page.$eval(B + ' [data-passes]', (p) => p.dataset.winner)) === 'mapped');
check('both plots are drawn', (await page.$$(B + ' svg polyline')).length === 4);
if (out) {
  await page.$eval(B, (b) => b.scrollIntoView());
  await page.screenshot({ path: out + '-explorer.png' });
}

// ---- prediction and questions
const P = await question('An embedding table of 1 GiB');
await (await P.$('input[type=number]')).type('1024');
await (await P.$('button.primary')).click();
await pause(150);
check('p-embedding: 1024, and the reveal points at the break-even question', (await qText(P)).includes('Right:') && (await qText(P)).includes('next question'));
let q = await question('90% of its bytes');
await typeAnswer(q, 1.9);
check('q-mixed: 1.9', (await qText(q)).includes('Correct.'));
q = await question('Above how many passes');
await typeAnswer(q, 71.1);
check('q-breakeven: 71.1', (await qText(q)).includes('Correct.'));
q = await question('What do you pass to the kernel?');
await pick(q, 'The value from cudaHostGetDevicePointer');
check('q-today-pointer: the 2009 answer is marked wrong', (await qText(q)).includes('Not quite.'));
await pick(q, 'The same pointer.');
check('q-today-pointer: the same pointer', (await qText(q)).includes('Correct.'));
q = await question('the host stalling at every copy');
await pick(q, 'Pin the model');
check('q-pinmemory: pinning the weights is wrong, and the batch is named as what crosses', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('the batch is what needs page-locked memory'));
await pick(q, 'The batches are in pageable memory');
check('q-pinmemory: pageable batches, fixed by pin_memory=True', (await qText(q)).includes('Correct.') && (await qText(q)).includes('may block the host'));

// ---- notes and voice
const notes = await page.$$eval('.prose > .note', (n) => n.map((x) => x.textContent).join(' || '));
check('the dangling "we mentioned" is flagged as the PDF\'s', notes.includes('In the PDF') && notes.includes('we mentioned'));
check('Today notes cite A3 and A4; HIP B2, Metal C2 and D4, PyTorch D1', ['A3', 'A4', 'B2', 'C2', 'D4', 'D1'].every((s) => notes.includes(s)));
check('every margin note directly follows text', await page.evaluate(() => [...document.querySelectorAll('.prose > .note')].every((n) => ['P', 'UL', 'OL'].includes(n.previousElementSibling?.tagName) || n.previousElementSibling?.classList.contains('note'))));
const prose = await page.$eval('.prose', (e) => e.textContent);
check('no em dash in the page', !prose.includes('—'));
check('links to the labs page and back to the map', (await page.$$eval('.prose a', (a) => a.map((x) => x.getAttribute('href')))).filter((h) => h === '#/appC/labs' || h === '#/appC/start').length >= 2);
check('answers went to the progress API (intercepted)', puts > 0);
check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
