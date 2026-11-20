// Browser check for section 7.4: lab 7.1's code split into parts, the "One address or many"
// explorer, the narrated animation "One address, one warp", the "Where would you put it?"
// classification, the predictions and the questions. Expected counts come from
// src/sim/constcache.ts. It passes with or without content/ch07/narration/manifest.json:
// with no audio the player runs on estimated timings and shows each script regardless.
// Usage: node e2e/ch07-s74.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { DEMO_SEED, distinctCount, expectedDistinct, replaySchedule, warpAddresses } from '../src/sim/constcache.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
const puts = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.method() === 'PUT' && r.url().includes('/api/progress')) {
    puts.push(JSON.parse(r.postData()));
    return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  }
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
const text = (sel) => page.$eval(sel, (e) => e.textContent);
const question = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const VOCAB = /^(reading reports|emulator-checked|unverified|verified on Metal)/;

await goto(page, 'ch07/7.4');
await page.waitForSelector('.code .tabs button');

// ---- lab 7.1's code, one part at a time ----
const figures = await page.$$('figure.code');
check('two listings from lab 7.1', figures.length === 2, String(figures.length));
const tabText = async (fig, label) => {
  await fig.evaluate((f, label) => [...f.querySelectorAll('.tabs button')].find((b) => b.textContent === label).click(), label);
  await pause(60);
  return fig.evaluate((f) => ({ code: f.querySelector('pre').textContent, status: f.querySelector('.status')?.textContent ?? '' }));
};
let c = await tabText(figures[0], 'CUDA');
check('constant part, CUDA: the __constant__ array and its copy, nothing else', c.code.includes('__constant__ float F_c[') && c.code.includes('cudaMemcpyToSymbol(F_c') && !c.code.includes('convBasicKernel') && !c.code.includes('struct Filter'), c.code.slice(0, 200));
check('CUDA status uses the fixed vocabulary', VOCAB.test(c.status), c.status);
check('F_c is coloured as constant memory', (await figures[0].$$eval('.v-const', (e) => e.map((x) => x.textContent))).includes('F_c'));
c = await tabText(figures[0], 'Metal kernel');
check('constant part, Metal kernel: conv_const takes constant float*', c.code.includes('kernel void conv_const') && c.code.includes('constant float*') && !c.code.includes('conv_basic('), c.code.slice(0, 200));
check('Metal status uses the fixed vocabulary', VOCAB.test(c.status), c.status);
c = await tabText(figures[0], 'Metal host');
check('constant part, Metal host: encodeConvConst binds a buffer', c.code.includes('encodeConvConst') && c.code.includes('setBuffer:F') && !c.code.includes('encodeConvByValue'));
c = await tabText(figures[1], 'CUDA');
check('by-value part, CUDA: struct Filter passed by value', c.code.includes('struct Filter') && c.code.includes('const Filter f') && !c.code.includes('__constant__ float F_c'));
c = await tabText(figures[1], 'Metal kernel');
check('by-value part, Metal kernel: constant Filter&', c.code.includes('constant Filter&'));
c = await tabText(figures[1], 'Metal host');
check('by-value part, Metal host: setBytes of the struct', c.code.includes('setBytes:&f length:sizeof(f)'));

// ---- what the reader's devices report ----
const yours = await text('[data-yours]');
const reports = await (await fetch(BASE + '/api/reports')).json();
check('the device line leaves out the emulators', !yours.includes('(race)') && !yours.includes('(emu)'), yours);
if (reports.metal) check('Metal: says Apple documents neither size', yours.includes('Apple documents neither'), yours);

// ---- predictions ----
let q = await question('16-entry table in constant memory');
await (await q.$('input[type=number]')).type('14');
await (await q.$('button.primary')).click();
await pause(150);
let t = await q.evaluate((e) => e.textContent);
check('codebook prediction reveals 13.97 and calls 14 close', t.includes('13.97') && t.includes('Close.'), t);
q = await question("What is the kernel's FLOP/B now");
await (await q.$('input[type=number]')).type('0.25');
await (await q.$('button.primary')).click();
await pause(150);
t = await q.evaluate((e) => e.textContent);
check('intensity prediction reveals 0.5 and explains a guess of 0.25', t.includes('0.5 FLOP/B') && t.includes('still charged the filter load'), t);

// ---- explorer: one address or many ----
const X = 'One address or many';
const pick = async (control, label) => {
  const ok = await page.evaluate((sel, control, label) => {
    const b = [...document.querySelectorAll(`${sel} [data-control="${control}"] button`)].find((x) => x.textContent.trim().startsWith(label));
    if (!b) return false;
    b.click();
    return true;
  }, bench(X), control, label);
  await pause(80);
  return ok;
};
const cell = (k) => page.$eval(`${bench(X)} [data-cell="${k}"]`, (e) => e.textContent.trim());
const state = async () => ({ distinct: Number(await cell('distinct')), nvidia: Number(await cell('nvidia')), amd: await cell('amd') });
const asked = () => page.$$eval(`${bench(X)} svg [data-entry]`, (es) => es.filter((e) => e.dataset.state !== 'idle').map((e) => Number(e.dataset.entry)));
const lanes = () => page.$$eval(`${bench(X)} svg [data-lane]`, (es) => es.length);

let s = await state();
check('filter tap: 1 address, 1 read, AMD scalar cache serves it', s.distinct === 1 && s.nvidia === 1 && s.amd.includes('yes'), JSON.stringify(s));
check('filter tap: the asked entry is F[0][0]', JSON.stringify(await asked()) === '[0]');
await page.evaluate((sel) => document.querySelector(sel + ' [data-next-tap]').click(), bench(X));
await pause(80);
check('next tap: still one address, now F[0][1]', (await state()).distinct === 1 && JSON.stringify(await asked()) === '[1]' && (await text(bench(X) + ' svg')).includes('F[0][1]'));
await pick('pattern', 'one entry per lane');
s = await state();
check('T[lane] at 32 lanes: 32 addresses, 32 reads, not uniform', s.distinct === 32 && s.nvidia === 32 && s.amd.includes('no'), JSON.stringify(s));
await pick('pattern', 'one per 8 lanes');
s = await state();
check('T[lane/8] at 32 lanes: 4 addresses', s.distinct === 4 && s.nvidia === 4, JSON.stringify(s));
await pick('warp', '64');
check('64-lane wavefront draws 64 lanes', (await lanes()) === 64);
s = await state();
check('T[lane/8] at 64 lanes: 8 addresses', s.distinct === 8, JSON.stringify(s));
await pick('pattern', 'one entry per lane');
check('T[lane] at 64 lanes: 64 addresses', (await state()).distinct === 64);
await pick('warp', '32');
await pick('pattern', 'codebook');
let want = distinctCount(warpAddresses('codebook', { warpSize: 32, tableSize: 16, seed: DEMO_SEED }));
s = await state();
check(`codebook, seed ${DEMO_SEED}: the model's ${want} distinct entries`, s.distinct === want && s.nvidia === want && s.amd.includes('no'), JSON.stringify(s));
check('codebook: the entries drawn as asked are the model\'s', JSON.stringify((await asked()).sort((a, b) => a - b)) === JSON.stringify([...new Set(warpAddresses('codebook', { warpSize: 32, tableSize: 16, seed: DEMO_SEED }))].sort((a, b) => a - b)));
check('codebook: expected value 13.97 is shown', (await cell('expected')) === '13.97', await cell('expected'));
await page.evaluate((sel) => document.querySelector(sel + ' [data-reseed]').click(), bench(X));
await pause(80);
const seed = Number(await page.$eval(bench(X) + ' [data-seed]', (e) => e.dataset.seed));
want = distinctCount(warpAddresses('codebook', { warpSize: 32, tableSize: 16, seed }));
check(`reseeded (seed ${seed}): the model's count, ${want}`, seed === DEMO_SEED + 1 && (await state()).distinct === want);
await pick('entries', '256');
want = distinctCount(warpAddresses('codebook', { warpSize: 32, tableSize: 256, seed }));
check(`256 entries: the model's count ${want}, expected ${expectedDistinct(256, 32).toFixed(1)}`, (await state()).distinct === want && (await cell('expected')).startsWith('30.1'), await cell('expected'));
await pick('warp', '64');
await pick('entries', '16');
want = distinctCount(warpAddresses('codebook', { warpSize: 64, tableSize: 16, seed }));
check(`64 lanes, 16 entries: the model's count ${want}, expected 15.74`, (await state()).distinct === want && (await cell('expected')) === '15.74');
if (out) {
  await page.$eval(bench(X), (e) => e.scrollIntoView());
  await page.screenshot({ path: out + '-explorer.png' });
}

// ---- the narrated animation ----
const S = 'section.scene[aria-label="One address, one warp"]';
const N = 6;
const LABELS = ['One address', 'Every tap', 'A codebook', 'AMD', 'Apple', 'Side by side'];
const CODES = warpAddresses('codebook', { warpSize: 32, tableSize: 16, seed: DEMO_SEED });
const ROUNDS = replaySchedule(CODES);
const res = await fetch(BASE + '/content/ch07/narration/manifest.json');
const manifest = res.ok ? await res.json() : null;
const scripts = [];
for (let n = 1; n <= N; n++) scripts.push(strip(await (await fetch(`${BASE}/content/ch07/narration/constant-broadcast-${n}.txt`)).text()));
check('every part has a script of 35 to 60 words', scripts.every((x) => { const w = x.split(' ').length; return w >= 35 && w <= 60; }), scripts.map((x) => x.split(' ').length).join(','));
const parts = await page.$$eval(S + ' .seg button', (bs) => bs.map((b) => b.textContent.trim()));
check('six parts with their labels', parts.length === N && parts.every((p, i) => p === `${i + 1}. ${LABELS[i]}`), parts.join('|'));
check('a captions switch is offered', !!(await page.$(S + ' label.check input')) && (await text(S)).includes('captions'));
const go = async (n) => {
  await page.evaluate((sel, label) => [...document.querySelectorAll(sel + ' .seg button')].find((b) => b.textContent.trim() === label).click(), S, `${n}. ${LABELS[n - 1]}`);
  await pause(120);
};
const seek = async (v) => {
  await page.$eval(S + ' input[type=range]', (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(120);
};
const drawn = () => page.$eval(S + ' .stage svg', (svg) => ({
  ...Object.fromEntries(Object.entries(svg.dataset)),
  served: svg.querySelectorAll('[data-lane][data-state="served"]').length,
  hot: [...svg.querySelectorAll('[data-entry][data-state="hot"]')].map((e) => Number(e.dataset.entry)),
  asked: svg.querySelectorAll('[data-entry]:not([data-state="idle"])').length,
  headline: svg.querySelector('[data-headline]')?.textContent ?? '',
}));
if (!manifest) {
  check('with no manifest the player says it runs on estimated timings', (await text(S + ' footer')).includes('No narration manifest found'));
  check('with no audio the caption shows the part\'s script', strip(await text(S + ' .caption')) === scripts[0]);
}
await go(1);
await seek(1000);
let d = await drawn();
check('part 1 ends with one address, one answer, all 32 lanes served', d.beat === '1' && d.distinct === '1' && d.answers === '1' && d.served === 32 && JSON.stringify(d.hot) === '[0]', JSON.stringify(d));
await go(2);
await seek(500);
d = await drawn();
const tap = Number(d.tap);
check('part 2 halfway: one hot entry, the current tap, answers = tap + 1', d.distinct === '1' && JSON.stringify(d.hot) === `[${tap}]` && Number(d.answers) === tap + 1, JSON.stringify(d));
await seek(1000);
d = await drawn();
check('part 2 ends at 25 answers for the 5×5 filter', d.answers === '25' && d.headline.startsWith('25 taps'), JSON.stringify(d));
await go(3);
d = await drawn();
check(`part 3: the model's ${ROUNDS.length} distinct entries are asked from the start`, Number(d.distinct) === ROUNDS.length && d.asked === ROUNDS.length && d.rounds === '0', JSON.stringify(d));
await seek(600);
d = await drawn();
const done = Number(d.rounds);
const servedWant = new Set(ROUNDS.slice(0, done).flatMap((r) => r.lanes)).size;
check(`part 3 at 60%: ${done} rounds answered, exactly their lanes served`, done > 0 && done < ROUNDS.length && d.served === servedWant, JSON.stringify(d));
check('part 3: the hot entry is the next round\'s address', JSON.stringify(d.hot) === `[${ROUNDS[done].address}]`);
await seek(1000);
d = await drawn();
check(`part 3 ends with ${ROUNDS.length} rounds and every lane served`, Number(d.rounds) === ROUNDS.length && d.served === 32, JSON.stringify(d));
check('part 3 headline says the count the narration says (14)', d.headline.startsWith(`${ROUNDS.length} different entries`) && scripts[2].includes(String(ROUNDS.length)));
await go(4);
await seek(1000);
d = await drawn();
check('part 4: AMD serves the filter tap, not the codebook', d['amdFilter'] === 'yes' && d['amdCodebook'] === 'no', JSON.stringify(d));
check('part 4 says AMD documents no size', (await text(S + ' .stage')).includes('not documented'));
await go(5);
await seek(1000);
check('part 5 holds only Apple\'s own descriptions and the lab', (await text(S + ' .stage')).includes('WWDC 2016') && (await text(S + ' .stage')).includes('Lab 7.1'));
await go(6);
await seek(1000);
d = await drawn();
check('part 6: 1 round for the filter against the model\'s count for the codebook', d['filterRounds'] === '1' && Number(d['codebookRounds']) === ROUNDS.length, JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-scene6.png' });

// captions on: remembered, and the caption follows the part
const capBox = await page.evaluateHandle((sel) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('captions')).querySelector('input'), S);
if (!(await capBox.evaluate((e) => e.checked))) await capBox.click();
await pause(100);
check('captions on: the caption is part 6\'s script', strip(await text(S + ' .caption')) === scripts[5]);
check('the captions choice is remembered in this browser', (await page.evaluate(() => localStorage.getItem('captions'))) === 'on');
// play with the sound off: the clock runs and the part advances by itself at the end
const soundBox = await page.evaluateHandle((sel) => [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('sound')).querySelector('input'), S);
if (await soundBox.evaluate((e) => e.checked)) await soundBox.click();
await go(1);
await seek(985);
await page.evaluate((sel) => document.querySelector(sel + ' button.primary').click(), S);
await pause(1500);
const current = await page.$$eval(S + ' .seg button', (bs) => bs.findIndex((b) => b.getAttribute('aria-pressed') === 'true') + 1);
check('playing moves on to part 2 by itself', current === 2, String(current));
await page.evaluate((sel) => document.querySelector(sel + ' button.primary').click(), S);
await capBox.click();
await page.evaluate(() => localStorage.removeItem('captions'));

// ---- classification ----
const W = 'Where would you put it?';
const place = async (item, bin) => {
  await page.evaluate((sel, item, bin) => document.querySelector(`${sel} [data-item="${item}"] [data-bin="${bin}"]`).click(), bench(W), item, bin);
  await pause(80);
  return page.$eval(`${bench(W)} [data-item="${item}"]`, (e) => ({ verdict: e.dataset.verdict, why: e.querySelector('[data-why]')?.textContent ?? '' }));
};
let v = await place('codebook', 'const');
check('a codebook in __constant__ is not the best home, and the card says why', v.verdict === 'wrong' && v.why.includes('14 of the 16') && v.why.includes('Better:'), JSON.stringify(v));
v = await place('gamma', 'const');
check('LayerNorm weights read per lane: __constant__ is not the best home', v.verdict === 'wrong' && v.why.includes('worst case'), JSON.stringify(v));
v = await place('rope', 'const');
check('a 2 MiB table does not fit 64 KB', v.verdict === 'wrong' && v.why.includes('does not fit'), JSON.stringify(v));
v = await place('filter', 'arg');
check('the filter by value is a good home', v.verdict === 'right');
v = await place('filter', 'const');
check('the filter in __constant__ is a good home too', v.verdict === 'right');
v = await place('codebook', 'shared');
check('a codebook staged in shared memory is a good home', v.verdict === 'right');
const GOOD = { sampling: 'arg', filter: 'const', codebook: 'global', rope: 'global', gamma: 'global', shapes: 'arg' };
for (const [item, bin] of Object.entries(GOOD)) {
  v = await place(item, bin);
  check(`${item} → ${bin}: a good home`, v.verdict === 'right', JSON.stringify(v));
}
check('the tally reads six of six', (await text(bench(W) + ' [data-where-result]')).includes('All six placed, 6 of 6'));
await pause(600);
let last = puts[puts.length - 1] ?? {};
const where = last.answers?.['ch07/7.4/c-where'];
check('placements are kept in the progress file under ch07/7.4/c-where', where?.correct === true && Object.entries(GOOD).every(([k, b]) => where.value?.[k] === b) && Object.keys(where.value).length === 6, JSON.stringify(where));
if (out) {
  await page.$eval(bench(W), (e) => e.scrollIntoView());
  await page.screenshot({ path: out + '-where.png' });
}

// ---- questions ----
const choose = async (needle, i) => {
  const h = await question(needle);
  await (await h.$$('input[type=radio]'))[i].click();
  await (await h.$('button.primary')).click();
  await pause(100);
  return h.evaluate((e) => e.textContent);
};
check('extern question: whole-program mode rejects it', (await choose('conv.cu declares', 1)).includes('Correct.'));
t = await choose('change nothing else', 3);
check('__device__ question: a wrong answer explains where read-only comes from', t.includes('Not quite.') && t.includes('qualifier'), t);
check('__device__ question: the right answer', (await choose('change nothing else', 0)).includes('Correct.'));
check('codebook question', (await choose('What is a plausible reason', 0)).includes('Correct.'));
check('Metal question', (await choose('plays the part of', 0)).includes('Correct.'));
check('setBytes question', (await choose('32 × 32 block of float weights', 0)).includes('Correct.'));
q = await question('8 KB per SM');
await (await q.$('input[type=number]')).type('45');
await (await q.$('button.primary')).click();
await pause(100);
check('cache-fit question accepts 45', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
await pause(600);
last = puts[puts.length - 1] ?? {};
check('answers are written under their ids', ['q-extern', 'q-device-vs-constant', 'q-codebook', 'q-metal', 'q-setbytes', 'q-cache-fit'].every((k) => last.answers?.[`ch07/7.4/${k}`]?.correct === true) && last.answers?.['ch07/7.4/p-codebook']?.value === 14, JSON.stringify(Object.keys(last.answers ?? {})));

// ---- narrow screen ----
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch07/7.4');
const wide = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
check('no sideways page scroll at 390 px', wide <= 0, `${wide}px`);
if (out) {
  await page.$eval(bench(X), (e) => e.scrollIntoView());
  await page.screenshot({ path: out + '-390.png' });
}

const expected404 = (u) => /\/content\/ch07\/narration\/(manifest\.json|[\w-]+\.mp3)$/.test(u);
const realErrors = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', realErrors.length === 0 && missing.every(expected404), [...realErrors, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
