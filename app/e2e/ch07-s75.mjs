// Browser check for section 7.5: the lab listing, the narrated scene, the
// "load, wait, compute" stepper and its outcome table, the tile/intensity
// explorer with Fig. 7.14, the one-warp row view, the measured-against-formula
// panel, and the questions. Expected numbers are the ones sim/convtile.test.ts pins.
// Usage: node e2e/ch07-s75.mjs [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto, BASE } from './browser.mjs';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const labFile = (p) => fs.existsSync(path.join(repo, 'labs', p));
const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);
const missing = [];
page.on('response', (r) => { if (r.status() === 404) missing.push(r.url()); });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (title) => `section.bench[aria-label="${title}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent);
const clickText = async (scope, label) => {
  const done = await page.evaluate((sel, label) => {
    const btn = [...document.querySelectorAll(sel + ' button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(80);
  return done;
};
/** Set a range, number or select control, found by its aria-label inside a scope, the way React sees it. */
const setControl = async (scope, label, value) => {
  const ok = await page.evaluate((sel, label, value) => {
    const el = document.querySelector(`${sel} [aria-label="${label}"]`);
    if (!el) return false;
    if (el.tagName === 'SELECT') {
      el.value = String(value);
      el.dispatchEvent(new Event('change', { bubbles: true }));
    } else {
      const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
      set.call(el, String(value));
      el.dispatchEvent(new Event('input', { bubbles: true }));
    }
    return true;
  }, scope, label, value);
  await pause(100);
  return ok;
};
const toggle = async (scope, labelText, on) => {
  await page.evaluate((sel, t, on) => {
    const box = [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes(t)).querySelector('input');
    if (box.checked !== on) box.click();
  }, scope, labelText, on);
  await pause(120);
};
const question = (needle) =>
  page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
const answerNumeric = async (needle, value) => {
  const q = await question(needle);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent.includes('Correct.'));
};
const answerChoice = async (needle, optionText) => {
  const q = await question(needle);
  await q.evaluate((e, t) => [...e.querySelectorAll('.opts label')].find((l) => l.textContent.includes(t)).querySelector('input').click(), optionText);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent.includes('Correct.'));
};

await goto(page, 'ch07/7.5');
await page.waitForSelector(bench('Step through one block'));

// ------------------------------------------------------------- the listing
if (labFile('ch07/lab2_tiledconv/solution/kernel.cu')) {
  await page.waitForFunction(() => [...document.querySelectorAll('figure.code .tabs button')].some((b) => b.textContent === 'Metal kernel'), { timeout: 10000 });
  const listing = await page.evaluateHandle(() =>
    [...document.querySelectorAll('figure.code')].find((f) => [...f.querySelectorAll('.tabs button')].some((b) => b.textContent === 'Metal kernel')),
  );
  const cu = await listing.evaluate((f) => f.querySelector('pre').textContent);
  check('listing shows the halo part of lab 7.2 (CUDA)', cu.includes('convTiledHaloKernel') && cu.includes('__syncthreads()'), cu.slice(0, 120));
  check('listing stops before the cached part and drops the part markers', !cu.includes('convCachedKernel') && !cu.includes('@part'));
  await listing.evaluate((f) => [...f.querySelectorAll('.tabs button')].find((b) => b.textContent === 'Metal kernel').click());
  await pause(100);
  const mtl = await listing.evaluate((f) => f.querySelector('pre').textContent);
  check('Metal tab shows the threadgroup version', mtl.includes('threadgroup float N_s') && mtl.includes('threadgroup_barrier'), mtl.slice(0, 120));
} else {
  console.log('skip listing: labs/ch07/lab2_tiledconv/solution/kernel.cu is not written yet');
}

// ------------------------------------------------------------- prediction of the input/output ratio
check('q-ratio accepts 1.5625', await answerNumeric('16 × 16 output tile with a 5 × 5 filter', 1.5625));

// ------------------------------------------------------------- the narrated scene
const SC = 'section.scene';
const script = async (n) => (await (await fetch(`${BASE}/content/ch07/narration/load-wait-compute-${n}.txt`)).text()).replace(/\[[^\]]*\]\s*/g, '').trim();
check('scene offers six parts', (await page.$$(SC + ' .seg button')).length === 6);
check('scene has a captions option', (await text(SC)).includes('captions'));
const caption = async () => ((await page.$(SC + ' .caption')) ? (await text(SC + ' .caption')).trim() : '');
check('without audio, part 1 shows its script as the caption', (await caption()) === (await script(1)), await caption());
const scenePart = async (k, pos) => {
  await page.evaluate((sel, k) => document.querySelectorAll(sel + ' .seg button')[k - 1].click(), SC, k);
  await pause(100);
  await setControl(SC, 'Position in this part', pos);
  await pause(120);
  return text(SC + ' svg');
};
let svg = await scenePart(2, 1000);
check('scene part 2 ends with all 64 slots filled, 15 of them ghost zeros', svg.includes('64 of 64 slots filled') && svg.includes('15 of them ghost cells'), svg.slice(-200));
svg = await scenePart(3, 900);
check('scene part 3 opens the barrier', svg.includes('the barrier opens'));
svg = await scenePart(4, 1000);
check('scene part 4 ends with 36 outputs from 324 multiply-adds, 28 threads idle', svg.includes('36 of 36 outputs stored') && svg.includes('324 multiply-adds') && svg.includes('ring of 28 threads'), svg.slice(-200));
svg = await scenePart(6, 1000);
check('scene part 6 tallies 400 B against 1296 B: 1.62 and 0.50, then 5.42', ['400 B', '1296 B', '1.62', '0.50', '5.42'].every((s) => svg.includes(s)), svg.slice(-240));
check('scene part 6 caption is its script', (await caption()) === (await script(6)));
if (out) await (await page.$(SC)).screenshot({ path: out + '-scene.png' });

// ------------------------------------------------------------- stepper
const S = bench('Step through one block');
const narr = () => text(S + ' .narrate');
await clickText(S, 'Step');
let n = await narr();
check('step 1: thread (0,0) finds a ghost cell and stores 0', n.includes('Thread (tx 0, ty 0)') && n.includes('ghost cell') && n.includes('stores 0 into N_s[0][0]'), n);
check('the ghost-store line of the kernel is highlighted', (await text(S + ' .code .ln.hl')).includes('0.0f'));
await clickText(S, 'Step');
check('step 2: it waits at the barrier', (await narr()).includes('reaches __syncthreads()'));
await clickText(S, 'Run to barrier');
check('run to barrier ends on the barrier opening', (await narr()).startsWith('The barrier opens'));
const slots = await page.$$eval(S + ' [data-k="ns"] span', (s) => s.map((x) => x.textContent));
check('after the barrier every slot is filled, the top row and left column with zeros', slots.every((v) => v !== '·') && slots.slice(0, 8).every((v) => v === '0') && [8, 16, 24].every((i) => slots[i] === '0'), slots.join(','));
await clickText(S, 'Run to end');
let all = await text(S);
check('correct kernel: block (0,0) outputs match the reference', all.includes("This block's 36 outputs match the reference."), all.slice(-300));
check('correct kernel: no failure to jump to', !all.includes('Go to the first failure'));
await page.evaluate((sel) => document.querySelector(sel + ' [aria-label="thread (tx 6, ty 6)"]').click(), S);
await pause(100);
const th = await text(S + ' [data-k="thread"]');
check('thread (6,6) computes P[5][5] from the patch at N_s[5][5]', th.includes('computes P[5][5]') && th.includes('corner is N_s[5][5]'), th);
await page.evaluate((sel) => document.querySelector(sel + ' [aria-label="thread (tx 7, ty 3)"]').click(), S);
await pause(100);
check('thread (7,3) is on the outer ring', (await text(S + ' [data-k="thread"]')).includes('outer ring'));
if (out) await (await page.$(S)).screenshot({ path: out + '-stepper.png' });

await toggle(S, 'barrier', false);
check('without the barrier, in order: a failure to jump to', await clickText(S, 'Go to the first failure'));
n = await narr();
check('the first failure: thread (1,1) reads N_s[1][2] before thread (2,1) loaded it', n.includes('thread (tx 1, ty 1) read N_s[1][2] before thread (tx 2, ty 1) had loaded it'), n);
check('the failing slot is marked', (await page.$$(S + ' [data-k="ns"] span.bad')).length === 1);
await clickText(S, 'Run to end');
check('without the barrier the block result is wrong', (await text(S)).includes("This block's output tile is wrong."));
await setControl(S, 'thread order', 'lockstep');
await clickText(S, 'Run to end');
check('in lockstep the same kernel gives the right answer', (await text(S)).includes("This block's 36 outputs match the reference."));
await toggle(S, 'barrier', true);
await setControl(S, 'thread order', 'in-order');
await toggle(S, 'ghost slots set to 0', false);
await clickText(S, 'Go to the first failure');
n = await narr();
check('without ghost zeros: a read of a slot nobody set to 0', n.includes('nobody set to 0'), n);
await clickText(S, 'Run to end');
all = await text(S);
check('without ghost zeros: wrong values in the tile', all.includes("This block's output tile is wrong.") && all.includes('stored values differ'), all.slice(-300));
await toggle(S, 'ghost slots set to 0', true);
await toggle(S, '− FILTER_RADIUS in row and col', false);
await clickText(S, 'Run to end');
all = await text(S);
check('without the offset: 11 outputs of its tile never written, 11 stores outside it', all.includes('11 outputs of its own tile were never written') && all.includes('11 stores landed outside its tile'), all.slice(-300));
await toggle(S, '− FILTER_RADIUS in row and col', true);
await setControl(S, 'image size', 14);
await setControl(S, 'block', '1,1');
await clickText(S, 'Run to end');
all = await text(S);
check('14 × 14, block (1,1): an interior block, correct', all.includes("This block's 36 outputs match the reference."), all.slice(-200));

// ------------------------------------------------------------- outcome table
const O = bench('Every block, every thread order');
const cells = await page.$$eval(O + ' tbody tr', (rows) => rows.map((r) => [...r.querySelectorAll('td[data-outcome]')].map((c) => c.dataset.outcome)));
check('as written: correct under every order', cells[0].every((c) => c === 'correct'), JSON.stringify(cells[0]));
check('no barrier: lockstep correct, the three other orders wrong', cells[1][0] === 'correct' && cells[1].slice(1).every((c) => c === 'wrong'), JSON.stringify(cells[1]));
check('no ghost zeros and no offset: wrong under every order', cells[2].every((c) => c === 'wrong') && cells[3].every((c) => c === 'wrong'), JSON.stringify(cells.slice(2)));
check('no ghost zeros: 36 wrong on a 10 × 10 image', (await text(O + ' tbody tr:nth-child(3)')).includes('36 wrong'));
check('no offset: 19 never written', (await text(O + ' tbody tr:nth-child(4)')).includes('19 never written'));

// ------------------------------------------------------------- explorer
const E = bench('Tile size, filter size, intensity');
const k = (key) => text(`${E} [data-k="${key}"]`);
check('default: input 32, r = 2 gives 5.42 FLOP/B', (await k('ai')) === '5.42', await k('ai'));
check('default: ideal 6.25, factor 0.87', (await k('ideal')) === '6.25' && (await k('att')) === '0.87');
check('default: 1024 threads fit the book limit; 240 idle', (await k('threads')).includes('1,024') && (await k('threads')).includes('fits') && (await k('idle')).startsWith('240 of 1,024'));
check('default: 28 × 28 outputs, 1.306 inputs per output', (await k('out')).startsWith('28 × 28') && (await k('ratio')) === '1.306');
await setControl(E, 'filter radius', 4);
check('9 × 9 with a 32 × 32 input tile: 14.58', (await k('ai')) === '14.58', await k('ai'));
await clickText(E, 'Block = output tile (Ex. 11)');
await setControl(E, 'output tile edge', 32);
check('organisation 2, 32 × 32 outputs, 9 × 9: 15.80, 6,400 B, 1024 threads', (await k('ai')) === '15.80' && (await k('shared')).startsWith('6,400 B') && (await k('threads')).startsWith('1,024'), [await k('ai'), await k('shared')].join(' | '));
check('organisation 2: loads per thread 2 (flat), 4 (strided)', (await k('loads')).includes('at most 2 with a flat loop') && (await k('loads')).includes('4 with a loop'));
await clickText(E, 'Block = input tile (Fig. 7.12)');
check('back to organisation 1: the same output tile needs 1600 threads, over the limit', (await k('threads')).includes('1,600') && (await k('threads')).includes('over 1,024'), await k('threads'));
const cell = (m, t) => page.$eval(`${E} td[data-m="${m}"][data-t="${t}"]`, (c) => ({ v: c.textContent.trim(), b: c.dataset.bound }));
const published = { '5,28': '5.42', '13,16': '20.80', '9,64': '17.88', '3,8': '1.76', '11,28': '21.29', '9,ideal': '20.25' };
const got = {};
for (const key of Object.keys(published)) {
  const [m, t] = key.split(',');
  got[key] = (await cell(m, t)).v;
}
check('heat map shows the published Fig. 7.14 values', Object.keys(published).every((key) => got[key] === published[key]), JSON.stringify(got));
check('at 20 FLOP/B: 17.88 memory-bound, 20.80 and 20.25 compute-bound', (await cell(9, 64)).b === 'memory' && (await cell(13, 16)).b === 'compute' && (await cell(9, 'ideal')).b === 'compute');
await setControl(E, 'threshold', 'custom');
await setControl(E, 'custom threshold', 5);
check('a 5 FLOP/B threshold turns 5.42 compute-bound', (await cell(5, 28)).b === 'compute' && (await cell(5, 16)).b === 'memory');
await page.evaluate((sel) => document.querySelector(`${sel} td[data-m="7"][data-t="16"] button`).click(), E);
await pause(120);
check('clicking the 7 × 7, t = 16 cell loads it: 8.48', (await k('ai')) === '8.48' && (await k('out')).startsWith('16 × 16'), await k('ai'));
await setControl(E, 'bytes per element', 2);
check('2-byte elements double the intensity: 16.95, and the table to 10.84 at 5 × 5, t = 28', (await k('ai')) === '16.95' && (await cell(5, 28)).v === '10.84', [await k('ai'), (await cell(5, 28)).v].join(' | '));
await setControl(E, 'bytes per element', 4);
await setControl(E, 'filter radius', 2);
await setControl(E, 'input tile edge', 4);
check('an input tile of 4 with r = 2 has no output tile', (await text(E)).includes('No output tile'));
// The reader's own numbers, when lab runs have recorded them: lab 5.1's threshold and the device's limits.
const options = (label) => page.$$eval(`${E} [aria-label="${label}"] option`, (o) => o.map((x) => ({ v: x.value, t: x.textContent })));
const mine = (await options('threshold')).find((o) => o.t.startsWith('Yours, from lab 5.1'));
if (mine) {
  const th = Number(/: ([\d.]+) FLOP\/B$/.exec(mine.t)[1]);
  await setControl(E, 'threshold', mine.v);
  const c = await cell(5, 28);
  check(`your lab 5.1 threshold (${th}) recolours the table`, c.b === (5.42 >= th ? 'compute' : 'memory') && (await text(E)).includes(`coloured at ${th} FLOP/B`), JSON.stringify(c));
} else console.log('skip threshold: no lab 5.1 roofline in reports/runs');
const dev = (await options('device limits')).find((o) => o.t.startsWith('Yours:'));
if (dev) {
  await setControl(E, 'device limits', dev.v);
  await setControl(E, 'input tile edge', 32);
  const sh = await k('shared');
  check(`your device's limits are applied (${dev.t})`, /fits|over/.test(sh) && !sh.includes('no limit'), sh);
  await setControl(E, 'device limits', 'book');
} else console.log('skip device limits: no real device in reports/runs');
if (out) await (await page.$(E)).screenshot({ path: out + '-explorer.png' });

// ------------------------------------------------------------- one warp, one row
const R = bench('One warp, one tile row');
const strip = (label) => page.evaluate((sel, l) => [...document.querySelectorAll(sel + ' [data-k]')].find((e) => e.dataset.k.startsWith(l))?.textContent ?? '', R, label);
let load = await strip('Fig. 7.12, load');
let store = await strip('Fig. 7.12, store');
check('Fig. 7.12 load, bx = 1: 5 sectors for 128 B', load.startsWith('5 segments (160 B) for 128 B'), load);
check('Fig. 7.12 store: 4 sectors for 112 B', store.startsWith('4 segments (128 B) for 112 B'), store);
check('32-wide tile: 4 sectors, all used, both ways', (await strip('32-wide tile, load')).startsWith('4 segments (128 B) for 128 B requested: 100%') && (await strip('32-wide tile, store')).startsWith('4 segments (128 B) for 128 B requested: 100%'));
await setControl(R, 'image width', 1003);
const counts = new Set();
for (let bx = 0; bx < 8; bx++) {
  await setControl(R, 'block column', bx);
  counts.add((await strip('Fig. 7.12, store')).slice(0, 1));
}
check('image width 1003: a 28-float store takes 4 or 5 sectors depending on the block', counts.has('4') && counts.has('5'), [...counts].join(','));
await setControl(R, 'image width', 1024);
await setControl(R, 'block column', 1);
await setControl(R, 'warp width', 64);
load = await strip('Fig. 7.12, load');
check('a 64-wide warp covers two block rows: 10 sectors for 256 B', load.startsWith('10 segments (320 B) for 256 B'), load);
await setControl(R, 'segment rule', 'amd128');
store = await strip('32-wide tile, store');
check('AMD 128-byte lines: a 32-wide aligned store is one line per row', store.startsWith('2 segments (256 B) for 256 B'), store);
await setControl(R, 'segment rule', 'metal');
check('Apple: no documented rule, nothing counted', (await text(R)).includes('Apple documents no coalescing rule'));

// ------------------------------------------------------------- questions
check('q-p175: thread (6,6)', await answerChoice('the text on p. 175 says thread (5,5)', 'Thread (6,6).'));
check('q-one-barrier: no write-after-read dependence', await answerChoice('needs two barriers per phase', 'never overwritten, so no write has to wait for earlier reads'));
check('q-ai9: 14.58', await answerNumeric('Input tile 32 × 32, filter 9 × 9', 14.58));
check('q-org2: organisation 2', await answerChoice('Which change raises the kernel', 'Make the block the output tile'));
check('q-ex11: 2', await answerNumeric('28 × 28 output tile (and block)', 2));

// ------------------------------------------------------------- your device against the formula
const P = await question('How many times faster do you expect it to run');
const committed = await P.evaluate((e) => !!e.querySelector('.verdict'));
if (!committed) {
  await (await P.$('input[type=number]')).type('3');
  await (await P.$('button.primary')).click();
  await pause(300);
}
const verdict = await P.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
const hasRuns = verdict.includes('measurements sit');
check('prediction reveals either the measurements or how to get them', hasRuns || verdict.includes('python3 labs/pmpp.py run 7.2 --solution --bench'), verdict.slice(0, 200));
if (hasRuns) check('the scatter plots at least one point', (await P.evaluate((e) => e.querySelectorAll('svg circle').length)) > 0);
else check('the empty state asks for the lab run and a reload, not ./verify.sh', verdict.includes('reload this page') && !verdict.includes('verify.sh'), verdict.slice(0, 200));
const pageText = (await page.$eval('.prose', (e) => e.textContent)).replace(/\s+/g, ' ');
check('the HIP path is called unverified, never compiled', pageText.includes('it has not been compiled or run on AMD hardware (unverified)') && !pageText.includes('compiles the CUDA file unchanged'));
check('one bridge: the MLX note has moved to section 7.6', !pageText.includes('depthwise 2D convolution for Metal') && pageText.includes('ssm-conv.cu'));

// ------------------------------------------------------------- errors, narrow screen
const expected404 = (u) =>
  /\/content\/ch07\/narration\/(manifest\.json|[\w-]+\.mp3)$/.test(u) ||
  (/\/api\/lab-file\?path=ch07%2Flab2_tiledconv/.test(u) && !labFile('ch07/lab2_tiledconv/solution/kernel.cu'));
const realErrors = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', realErrors.length === 0 && missing.every(expected404), [...realErrors, ...missing.filter((u) => !expected404(u))].join(' | '));

await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await pause(400);
check('no sideways page scroll at 390 px', await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
if (out) await page.screenshot({ path: out + '-narrow.png' });

await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
