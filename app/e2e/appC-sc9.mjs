// Browser check for section C.9: the quoted declarations, the "Reserve, create, map,
// grant" stepper (the book's three calls crash, the fix, the book's two-GPU example,
// misaligned sizes, teardown order), the "Growing a KV cache" explorer, the prediction
// and the questions. Expected values come from src/sim/vmmgrow.ts, imported here
// directly. The narrated scene in this section has its own check,
// appC-scene-grow-buffer.mjs.
// Usage: APP_URL=http://localhost:5173 node e2e/appC-sc9.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { LLAMA2_7B_BYTES_PER_TOKEN, MiB, STRATEGIES, bookExample, bookThreeCalls, grow, run } from '../src/sim/vmmgrow.ts';
import { binary } from '../src/sim/addrwidth.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);
const missing = [];
page.on('response', (r) => {
  if (r.status() === 404) missing.push(r.url());
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const S = 'section.bench[aria-label="Reserve, create, map, grant"]';
const K = 'section.bench[aria-label="Growing a KV cache"]';
const G = 2 * MiB;
const text = (sel) => page.evaluate((s) => document.querySelector(s)?.textContent.replace(/\s+/g, ' ').trim() ?? '', sel);
const attr = (sel, a) => page.evaluate((s, a) => document.querySelector(s)?.getAttribute(a), sel, a);
const press = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(80);
  return done;
};
const act = async (name) => {
  await page.click(`${S} button[data-act="${name}"]`);
  await pause(80);
};
const sel = async (label, v) => {
  await page.select(`${S} select[aria-label="${label}"]`, String(v));
  await pause(40);
};
const last = async () => ({ kind: await attr(S + ' [data-out="last"]', 'data-kind'), code: await attr(S + ' [data-out="last"]', 'data-code'), text: await text(S + ' [data-out="last"]') });
const slotsDrawn = () => page.$$eval(S + ' svg g[data-slot]', (g) => g.map((x) => ({ device: x.dataset.device, read: x.dataset.read, a0: x.querySelector('[data-access="0"]').dataset.on, a1: x.querySelector('[data-access="1"]').dataset.on })));
const logKinds = () => page.$$eval(S + ' [data-part="log"] li', (l) => l.map((x) => x.dataset.kind));
const setRange = async (s, v) => {
  await page.$eval(s, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(120);
};
const setNumber = async (s, v) => {
  await page.$eval(s, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(120);
};
const answer = async (match, value) => {
  const q = await page.evaluateHandle((m) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(m)), match);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
const choose = async (match, option) => {
  const q = await page.evaluateHandle((m) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(m)), match);
  await q.evaluate((e, o) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes(o)).querySelector('input').click(), option);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};

await goto(page, 'appC/C.9');
await page.waitForSelector(S);
check('every margin note (or note group) directly follows text, never another note', await page.evaluate(() => [...document.querySelectorAll('.prose > .note')].every((n) => ['P', 'UL', 'OL'].includes(n.previousElementSibling?.tagName))));

// ---- the quoted declarations
const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent));
check('the declarations come in two tabs, CUDA Driver API and HIP', tabs.includes('CUDA Driver API') && tabs.includes('HIP'), tabs.join('|'));
const code = await text('figure.code pre');
check('the CUDA tab shows the four calls, cuMemSetAccess included', ['cuMemAddressReserve', 'cuMemCreate', 'cuMemMap', 'cuMemSetAccess'].every((n) => code.includes(n + '(')));
check('and is labelled unverified, not compiled', (await text('figure.code .status')).includes('unverified'));

// ---- the book's three calls: the kernel crashes
const three = run(G, bookThreeCalls(G)).log;
await page.click(`${S} button[data-preset="three"]`);
await pause(100);
let l = await last();
check("the book's three calls end in a crash: mapped, but no access", l.kind === 'crash' && l.code === 'no-access' && l.kind === three[3].kind && l.code === three[3].code, JSON.stringify(l));
check('the message names cuMemSetAccess', l.text.includes('cuMemSetAccess'));
check('the log is the model\'s: ok, ok, ok, crash', JSON.stringify(await logKinds()) === JSON.stringify(three.map((o) => o.kind)));
let d = await slotsDrawn();
check('granules 0 to 3 are GPU 0\'s and marked "no access"; 4 to 7 are empty', d.slice(0, 4).every((x) => x.device === '0' && x.read === 'no-access') && d.slice(4).every((x) => x.device === ''), JSON.stringify(d));
if (out) await page.screenshot({ path: out + '-c9-crash.png' });
check('q-setaccess: cuMemSetAccess is missing', (await choose('crashes. What is missing?', 'cuMemSetAccess, granting GPU 0')).includes('Correct.'));

// The fix: grant GPU 0 access to granules 0 to 3, then read again.
await press(S + ' [aria-label="Grant to"]', 'GPU 0');
await sel('Grant from granule', 0);
await sel('Grant to granule', 3);
await act('access');
check('granting GPU 0 access to granules 0 to 3 succeeds', (await last()).kind === 'ok');
await press(S + ' [aria-label="Kernel on"]', 'GPU 0');
await sel('Read from granule', 0);
await sel('Read to granule', 3);
await act('read');
l = await last();
d = await slotsDrawn();
check('now the kernel reads all four granules from its own memory', l.kind === 'ok' && d.slice(0, 4).every((x) => x.read === 'local' && x.a0 === '1'), JSON.stringify(l));

// A size that is not a whole number of granules.
await press(S + ' [aria-label="Create on"]', 'GPU 1');
await sel('Chunk size', 1.5);
await act('create');
l = await last();
check('1.5 granules is refused, with HIP\'s documented out-of-memory error named', l.kind === 'error' && l.code === 'misaligned' && l.text.includes('out-of-memory'), JSON.stringify(l));
await sel('Chunk size', 4);
await act('create');
check('4 granules on GPU 1 is created as h2', (await last()).kind === 'ok' && (await text(S + ' [data-chunk="2"]')).includes('on GPU 1'));
await sel('Chunk to map', 2);
await sel('Map at granule', 2);
await act('map');
check('mapping h2 over h1 is refused', (await last()).code === 'overlap');
await sel('Map at granule', 5);
await act('map');
check('mapping h2 past the end of the range is refused', (await last()).code === 'outside');
await sel('Map at granule', 4);
await act('map');
check('mapping h2 at granule 4 succeeds', (await last()).kind === 'ok');

// ---- the book's example: one pointer over two GPUs
const ex = run(G, bookExample(G));
await page.click(`${S} button[data-preset="book"]`);
await pause(100);
d = await slotsDrawn();
const pieces = ex.log[ex.log.length - 1].pieces;
check("the book's example: GPU 0 backs granules 0 to 3, GPU 1 backs 4 to 7", d.map((x) => x.device).join('') === '00001111', JSON.stringify(d.map((x) => x.device)));
check('a kernel on GPU 0 reads the first half locally and the second half remotely, as the model says', d.slice(0, 4).every((x) => x.read === 'local') && d.slice(4).every((x) => x.read === 'remote') && pieces.map((p) => p.status).join(',') === 'local,remote');
check('and the message says so', (await text(S + ' [data-out="last"]')).includes('through one pointer'));
await press(S + ' [aria-label="Kernel on"]', 'GPU 1');
await sel('Read from granule', 0);
await sel('Read to granule', 7);
await act('read');
check('a kernel on GPU 1 over the same range crashes: no access for GPU 1', (await last()).code === 'no-access');
await press(S + ' [aria-label="Grant to"]', 'GPU 1');
await sel('Grant from granule', 0);
await sel('Grant to granule', 7);
await act('access');
await act('read');
d = await slotsDrawn();
check('after its own grant GPU 1 reads: remote first half, local second half', (await last()).kind === 'ok' && d.slice(0, 4).every((x) => x.read === 'remote') && d.slice(4).every((x) => x.read === 'local'));
if (out) await page.screenshot({ path: out + '-c9-book.png' });

// ---- teardown
await sel('Chunk to tear down', 1);
await act('release');
check('releasing a mapped chunk is flagged as outside the documented order', (await last()).kind === 'order');
await act('free');
check('freeing the range with chunks mapped is flagged too', (await last()).kind === 'order');
await act('unmap');
await sel('Chunk to tear down', 2);
await act('unmap');
await sel('Chunk to tear down', 1);
await act('release');
await sel('Chunk to tear down', 2);
await act('release');
await act('free');
check('unmap, release, free in order leaves everything clean', (await page.$(S + ' [data-out="clean"]')) !== null && (await last()).kind === 'ok');

// ---- the prediction
const dbl = grow('double', { tokens: 4096, bytesPerToken: LLAMA2_7B_BYTES_PER_TOKEN, chunk: 2 * MiB });
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('How many GiB does it copy over the whole run')));
await (await pred.$('input[aria-label="your prediction"]')).type('6');
await pred.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Commit prediction').click());
await pause(150);
const pt = await pred.evaluate((e) => e.textContent);
check('the prediction reveals the model\'s 1.998 GiB', pt.includes(`${Number((dbl.copiedTotal / 1024 ** 3).toFixed(3))} GiB`) && pt.includes('halve going back in time'), pt.slice(0, 200));

// ---- growing a KV cache
const rows = () => page.$$eval(K + ' tr[data-strategy]', (r) => Object.fromEntries(r.map((x) => [x.dataset.strategy, { copied: Number(x.dataset.copied), peak: Number(x.dataset.peak), unused: Number(x.dataset.unused), moves: Number(x.dataset.moves) }])));
const expectRows = (tokens, kib, chunkMiB) => {
  const b = kib * 1024;
  const chunk = chunkMiB * MiB;
  const reserve = Math.ceil((4096 * b) / chunk) * chunk;
  return Object.fromEntries(STRATEGIES.map((s) => {
    const x = grow(s.key, { tokens, bytesPerToken: b, chunk, reserve });
    return [s.key, { copied: x.copiedTotal, peak: x.peak, unused: x.unused, moves: x.pointerChanges }];
  }));
};
let got = await rows();
check('defaults (1500 tokens, 512 KiB, 2 MiB): all three rows match the model', JSON.stringify(got) === JSON.stringify(expectRows(1500, 512, 2)), JSON.stringify(got));
await setRange(K + ' input[type=range]', 4096);
got = await rows();
check('at 4096 tokens all three rows match the model', JSON.stringify(got) === JSON.stringify(expectRows(4096, 512, 2)), JSON.stringify(got));
check('doubling copies 2046 MiB and moves the pointer 10 times; peak 3 GiB', got.double.copied === 2046 * MiB && got.double.moves === 10 && got.double.peak === 3 * 1024 * MiB, JSON.stringify(got.double));
check('exact fit copies about 4 TiB', Math.abs(got.exact.copied / 2 ** 40 - 4) < 0.01);
check('reserve-and-map: nothing copied, no pointer change, nothing unused', got.vmm.copied === 0 && got.vmm.moves === 0 && got.vmm.unused === 0);
check('the data held is 2 GiB', (await text(K + ' [data-out="live"]')) === '2 GiB');
await press(K + ' [aria-label="Strategy"]', 'Reserve once, map chunks');
check('choosing reserve-and-map redraws the chart for it', (await attr(K + ' [data-part="chart"]', 'data-strategy')) === 'vmm' && (await text(K + ' [data-out="note"]')).startsWith('Reserved addresses'));
await press(K + ' [aria-label="Strategy"]', 'Allocate twice as much and copy');
check('the doubling note gives the model\'s ratio of 3', (await text(K + ' [data-out="note"]')).includes('up to 3 times'));
if (out) await page.screenshot({ path: out + '-c9-grow.png' });
// The question's case: one chunk per token, 16 tokens.
await setNumber(K + ' input[aria-label="KV cache per token in KiB"]', 2048);
await setRange(K + ' input[type=range]', 16);
got = await rows();
check('2048 KiB per token, 16 tokens: doubling copies 30 MiB, 15 chunks, as the question says', got.double.copied === 30 * MiB && got.double.copied === expectRows(16, 2048, 2).double.copied && got.vmm.copied === 0, JSON.stringify(got));
await page.select(K + ' select[aria-label="Chunk size"]', '8');
await pause(100);
got = await rows();
check('with an 8 MiB chunk the rows still match the model', JSON.stringify(got) === JSON.stringify(expectRows(16, 2048, 8)), JSON.stringify(got));

// ---- the other questions
check('q-copied: 15 chunks', (await answer('How many chunks are copied in all', 15)).includes('Correct.'));
check('q-pointer: no new pointer', (await choose('need a new pointer', 'No: the base address stays')).includes('Correct.'));
const wrongPeer = await choose('What does PyTorch', 'expandable_segments turns peer access on by itself');
check('q-peer: a wrong pick explains why', wrongPeer.includes('Not quite.'));
check('q-peer: the runtime peer call does not reach cuMemMap memory', (await choose('What does PyTorch', 'cudaDeviceEnablePeerAccess does not reach memory allocated with cuMemMap')).includes('Correct.'));
check('the section links back to §C.5 and on to chapter 20 §20.7', (await page.$('a[href="#/appC/C.5"]')) !== null && (await page.$('a[href="#/ch20/20.7"]')) !== null);

// ---- narrow screen
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((S, K) => ({ page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, s: document.querySelector(S).getBoundingClientRect().width <= document.documentElement.clientWidth, k: document.querySelector(K).getBoundingClientRect().width <= document.documentElement.clientWidth }), S, K);
check('at a narrow width nothing scrolls the page sideways', narrow.page && narrow.s && narrow.k, JSON.stringify(narrow));
if (out) {
  await page.$eval(S, (el) => el.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-c9-narrow-stepper.png' });
  await page.$eval(K, (el) => el.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-c9-narrow-grow.png' });
}

// The scene on this page fetches a narration manifest and audio that may not exist yet.
const expected404 = (u) => /\/content\/appC\/narration\/(manifest\.json|grow-buffer-\d\.mp3)$/.test(u);
const real = errors.filter((e) => !/Failed to load resource: the server responded with a status of 404/.test(e));
check('no console or page errors (a missing narration manifest is allowed)', real.length === 0 && missing.every(expected404), [...real, ...missing.filter((u) => !expected404(u))].join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
