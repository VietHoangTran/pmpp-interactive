// Browser check for section 22.4: the prediction p-crossover and its reveal, the explorer
// "Reformulate or accelerate?" in both problems (size, speed-up and batch sliders; the tie
// moves with the speed-up while the slopes do not; the KV-cache memory of Eq. 20.8), the three
// questions, notes, links and the page at 390 px. Expected values come from
// src/sim/formulation.ts, imported here directly.
// Usage: APP_URL=http://localhost:<port> node e2e/ch22-s224.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';
import { GiB, advantage, crossover, curves, kvBytes, slope, time, tieFactor } from '../src/sim/formulation.ts';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
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
const norm = (s) => s.replace(/\s+/g, ' ').trim();
const shot = async (name, sel) => {
  if (!out) return;
  await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};
const R = 'section.bench[aria-label="Reformulate or accelerate?"]';
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const attr = (sel, a) => page.$eval(sel, (e, a) => e.getAttribute(a), a);
const press = async (label) => {
  const done = await page.evaluate((R, label) => {
    const btn = [...document.querySelectorAll(R + ' button')].find((b) => b.textContent.trim() === label);
    if (!btn) return false;
    btn.click();
    return true;
  }, R, label);
  await pause(90);
  return done;
};
const setInput = async (sel, value) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(90);
};
const SPEEDS = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000];
const readout = async () => ({
  cross: Number(await attr(R + ' [data-readout="formulation"]', 'data-crossover')),
  sd: Number(await attr(R + ' [data-readout="formulation"]', 'data-slope-direct')),
  sr: Number(await attr(R + ' [data-readout="formulation"]', 'data-slope-reformed')),
  adv: await text(R + ' [data-value="advantage"]'),
  size: await text(R + ' [data-value="size"]'),
  speed: await text(R + ' [data-value="speedup"]'),
});
// The page's number format: whole numbers with separators from 1000 up, three significant digits below.
const sig = (v) => (v >= 1000 ? Math.round(v).toLocaleString('en-US') : Number(v.toPrecision(3)).toLocaleString('en-US'));
const close = (a, b, rel = 1e-3) => Math.abs(a - b) <= rel * Math.abs(b);
const q = (part) => page.evaluateHandle((p) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(p)), part);
const answerChoice = async (part, option) => {
  const h = await q(part);
  const ok = await h.evaluate((e, o) => {
    const l = [...e.querySelectorAll('.opts label')].find((x) => x.textContent.includes(o));
    if (!l) return false;
    l.querySelector('input').click();
    return true;
  }, option);
  await pause(60);
  await (await h.$('button.primary')).click();
  await pause(100);
  return ok && norm(await h.evaluate((e) => (e.querySelector('.verdict')?.textContent ?? '') + ' | ' + (e.querySelector('.opts label.wrong .muted')?.textContent ?? '')));
};

await goto(page, 'ch22/22.4');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
await page.waitForSelector(R);
const C = curves('coulomb');
const K = curves('kv');

// ---- the prediction
const pred = await q('By what factor must the volume grow');
check('nothing is revealed before the prediction is committed', !(await pred.evaluate((e) => e.textContent.includes('You predicted'))));
await (await pred.$('input[type=number]')).type('3.16');
await page.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Commit prediction').click(), pred);
await pause(150);
const reveal = norm(await pred.evaluate((e) => e.textContent));
check(`the reveal gives ${tieFactor(10)} from the model and answers the √10 guess`, reveal.includes('You predicted 3.16 ×') && reveal.includes('they meet where V = 10') && reveal.includes('A factor of √10 would be right if the cutoff cost stayed fixed'), reveal);
check(`  and ties it to the explorer: 1000× gives ${advantage(C, 1, 1000)} times faster at size 1 and a tie at ${crossover(C, 1000).toFixed(3)}`, reveal.includes('a speed-up of 1000× gives exactly this case') && reveal.includes('10 times faster at size 1') && reveal.includes('tie at 10× the volume'), reveal);

// ---- the explorer: the book's problem, default size 1 and 1000×
let r = await readout();
check('default: electrostatic map, size 1, speed-up 1000×', (await attr(R + ' button[aria-pressed="true"]', 'aria-pressed')) === 'true' && (await text(R + ' button[aria-pressed="true"]')).startsWith('Electrostatic map') && r.size === '1× the volume' && r.speed === '1000×', JSON.stringify(r));
check('at size 1 direct on the GPU is 10 times faster, tie at 10× the volume', r.adv === '10 times faster' && close(r.cross, crossover(C, 1000)) && close(r.cross, 10) && (await text(R + ' [data-value="crossover"]')) === '10× the volume', JSON.stringify(r));
check('slopes 2 and 1', close(r.sd, 2, 2e-3) && close(r.sr, 1, 2e-3), JSON.stringify(r));
const series = await page.$$eval(R + ' polyline[data-series]', (p) => p.map((x) => x.dataset.series));
check('four curves: direct and cutoff, on the CPU and on the GPU', JSON.stringify(series) === JSON.stringify(['direct-slow', 'direct-fast', 'reformed-slow', 'reformed-fast']), JSON.stringify(series));
check('the tie is marked on the plot', (await page.$(R + ' [data-mark="crossover"]')) !== null);
const rowTimes = await page.$$eval(R + ' tr[data-series-row]', (rs) => rs.map((x) => x.dataset.seriesRow));
check('the readout table lists the four times', rowTimes.length === 4);
await shot('coulomb-1000', R);

// speed-up down to 100: the tie moves, the slopes do not
await setInput(R + ' input[aria-label="GPU speed-up"]', SPEEDS.indexOf(100));
const r100 = await readout();
check(`speed-up 100×: the tie moves to ${crossover(C, 100).toFixed(3)} (from 10)`, r100.speed === '100×' && close(r100.cross, crossover(C, 100)) && r100.cross < r.cross && r100.adv === 'the same time', JSON.stringify(r100));
check('  and neither slope changes', r100.sd === r.sd && r100.sr === r.sr, JSON.stringify([r.sd, r.sr, r100.sd, r100.sr]));
await setInput(R + ' input[aria-label="GPU speed-up"]', SPEEDS.indexOf(10));
const r10 = await readout();
check(`speed-up 10×: the tie moves down again, to ${crossover(C, 10).toFixed(3)}; slopes unchanged`, close(r10.cross, crossover(C, 10)) && r10.cross < r100.cross && r10.sr === r.sr && r10.sd === r.sd, JSON.stringify(r10));

// size: past the tie the reformulation wins even on the CPU
await setInput(R + ' input[aria-label="Problem size"]', 40); // 10^(40/10 - 2) = 100
const rBig = await readout();
const advBig = advantage(C, 100, 10);
check(`size 100× at 10×: direct on the GPU is ${(1 / advBig).toFixed(3)} times slower than cutoff on the CPU`, rBig.size === '100× the volume' && rBig.adv === `${sig(1 / advBig)} times slower`, JSON.stringify(rBig));
check('  and the slopes are still 2 and 1', close(rBig.sd, 2, 2e-3) && close(rBig.sr, 1, 2e-3), JSON.stringify(rBig));
await setInput(R + ' input[aria-label="Problem size"]', 0);
const rTiny = await readout();
check('at the smallest size (0.01×, one neighbourhood) the slopes agree with the model', rTiny.size === '0.01× the volume' && close(rTiny.sr, slope((x) => time(C, 'reformed-fast', x, 10), 0.01), 2e-3) && close(rTiny.sd, 2, 2e-3), JSON.stringify(rTiny));

// ---- the second problem: decoding with and without a KV cache
check('switch to the decoding problem', await press('Decoding a sequence (ours, from chapter 20)'));
let k = await readout();
check('decoding: token 1,024, the tie at token S (10)', k.size === 'token 1,024' && close(k.cross, crossover(K, 10)) && close(k.cross, 10, 1e-3), JSON.stringify(k));
check('  and the cache lowers the slope by exactly one', close(k.sd - k.sr, 1, 5e-3) && close(k.sr, slope((x) => time(K, 'reformed-fast', x, 10), 1024), 2e-3), JSON.stringify(k));
await setInput(R + ' input[aria-label="GPU speed-up"]', SPEEDS.indexOf(1000));
const k1000 = await readout();
check('speed-up 1000×: the tie moves to token 1000, the slopes stay', close(k1000.cross, 1000, 1e-3) && k1000.sr === k.sr && k1000.sd === k.sd, JSON.stringify(k1000));
check('  and at token 1,024 recomputing on the GPU is just slower than the cache on the CPU', k1000.adv === `${sig(1 / advantage(K, 1024, 1000))} times slower`, JSON.stringify(k1000));
// memory against length and batch
const mem = async () => ({ bytes: Number(await attr(R + ' [data-readout="kv-memory"]', 'data-bytes')), gib: await text(R + ' [data-value="kv-gib"]'), batch: await text(R + ' [data-value="batch"]') });
let m = await mem();
check('KV memory at 8 × 1,024 tokens matches Eq. 20.8 (4 GiB)', m.batch === '8' && m.bytes === kvBytes(1024, 8) && m.gib === `${kvBytes(1024, 8) / GiB} GiB`, JSON.stringify(m));
await setInput(R + ' input[aria-label="Batch"]', 4);
m = await mem();
check('batch 16 doubles it', m.batch === '16' && m.bytes === 2 * kvBytes(1024, 8) && m.gib === '8 GiB', JSON.stringify(m));
await setInput(R + ' input[aria-label="Problem size"]', 48); // token 4096
m = await mem();
check("batch 16 × 4,096 tokens is 16 times the book's 2 GiB", (await text(R + ' [data-value="size"]')) === 'token 4,096' && m.bytes === 16 * 2 * GiB && m.gib === '32 GiB', JSON.stringify(m));
await shot('kv', R);
check('back to the book’s problem: the memory panel goes away', (await press("Electrostatic map (the book's)")) && (await page.$(R + ' [data-readout="kv-memory"]')) === null);

// ---- questions
let v = await answerChoice('gives up exactness to cut the complexity', 'The KV cache');
check('q-what-relaxed: the KV cache is wrong, it spends memory', v && v.includes('Not quite.') && v.includes('spends memory'), v);
v = await answerChoice('gives up exactness to cut the complexity', 'Cutoff summation');
check('q-what-relaxed: cutoff summation is right', v && v.includes('Correct.') && v.includes('square of the volume'), v);
v = await answerChoice('What does it spend instead', 'Accuracy, because the cached keys');
check('q-kv-cost: stale keys is wrong, with the causal-mask reason', v && v.includes('Not quite.') && v.includes('causal mask'), v);
v = await answerChoice('What does it spend instead', 'Memory that grows');
check('q-kv-cost: memory is right; the explanation gives Eq. 20.8, 2 GiB and points to 22.5', v && v.includes('Correct.') && v.includes('Eq. 20.8') && v.includes('2 GiB') && v.includes('22.5'), v);
v = await answerChoice('Which one is a reformulation', 'Split each attention row');
check('q-why-domain: split softmax is a decomposition, not a reformulation', v && v.includes('Not quite.') && v.includes('decomposition'), v);
v = await answerChoice('Which one is a reformulation', 'Attend only to the most recent');
check('q-why-domain: the key window is right, physics plus computing', v && v.includes('Correct.') && v.includes('physics'), v);

// ---- notes and links
const misplaced = await page.$$eval('.prose > .note', (ns) => ns.filter((n) => n.previousElementSibling?.tagName !== 'P').map((n) => n.textContent.slice(0, 60)));
check('every margin note follows a paragraph', misplaced.length === 0, misplaced.join(' | '));
const beyond = norm(await page.$eval('.prose .note.beyond', (e) => e.textContent));
check('the KV-cache bridge is labelled as ours, with chapter 20 pages', beyond.includes('This bridge is ours') && beyond.includes('pp. 488–492'), beyond);
check('a "what this model leaves out" note', (await page.$$('.prose .note.caveat')).length === 1);
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch22/22.5', '#/ch20/20.4', '#/ch20/20.6']);
check(`every link goes to a known route (${links.length})`, links.every((l) => known.has(l)) && links.includes('#/ch20/20.4'), links.join(', '));

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch22/22.4');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', R);

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
