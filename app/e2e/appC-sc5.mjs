// Browser check for section C.5: the "Peer access matrix" explorer (one-way grants, the
// eight-peer limit, NVSwitch, IOMMU, a kernel following a pointer, one halo exchange
// both ways) and the questions. Expected values come from src/sim/peeraccess.ts,
// imported here directly.
// Usage: APP_URL=http://localhost:5173 node e2e/appC-sc5.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { MAX_PEERS_WITHOUT_NVSWITCH, enableAll, haloExchange } from '../src/sim/peeraccess.ts';

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
const B = 'section.bench[aria-label="Peer access matrix"]';
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
const cell = async (i, j) => {
  await page.click(`${B} button[data-cell="${i}-${j}"]`);
  await pause(60);
};
const on = (i, j) => attr(`${B} button[data-cell="${i}-${j}"]`, 'data-on').then(Number);
const peers = () => page.$$eval(B + ' td[data-peers]', (t) => t.map((x) => Number(x.dataset.peers)));
const status = () => text(B + ' [data-out="status"]');
const setBox = async (label, v) => {
  await page.evaluate((B, label, v) => {
    const box = [...document.querySelectorAll(B + ' label.check')].find((l) => l.textContent.includes(label)).querySelector('input');
    if (box.checked !== v) box.click();
  }, B, label, v);
  await pause(80);
};
const choose = async (match, option) => {
  const q = await page.evaluateHandle((m) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(m)), match);
  await q.evaluate((e, o) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes(o)).querySelector('input').click(), option);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};

await goto(page, 'appC/C.5');
await page.waitForSelector(B);

// ---- one grant, one direction
check('starts with 4 GPUs and an empty 4 × 4 matrix', Number(await attr(B + ' [data-part="matrix"]', 'data-n')) === 4 && (await page.$$(B + ' button[data-cell]')).length === 16 && (await peers()).every((p) => p === 0));
await cell(0, 1);
check('clicking (0, 1) grants GPU 0 access to GPU 1', (await on(0, 1)) === 1 && (await on(1, 0)) === 0);
const s01 = await status();
check('the status names the call on GPU 0 and says the reverse is not granted', s01.includes('on GPU 0: cudaDeviceEnablePeerAccess(1, 0)') && s01.includes("GPU 1 may not load from GPU 0's"), s01);
await page.select(B + ' select[aria-label="Kernel on GPU"]', '0');
await page.select(B + ' select[aria-label="Memory of GPU"]', '1');
await pause(60);
check('a kernel on GPU 0 loads from GPU 1: peer load', (await attr(B + ' [data-out="deref"]', 'data-how')) === 'peer');
await page.select(B + ' select[aria-label="Kernel on GPU"]', '1');
await page.select(B + ' select[aria-label="Memory of GPU"]', '0');
await pause(60);
check('a kernel on GPU 1 loading from GPU 0: an error, with cudaMemcpyPeer as the fallback', (await attr(B + ' [data-out="deref"]', 'data-how')) === 'refused' && (await text(B + ' [data-out="deref"]')).includes('cudaMemcpyPeer'));
await page.select(B + ' select[aria-label="Memory of GPU"]', '1');
await pause(60);
check('a kernel on GPU 1 loading from its own memory: local', (await attr(B + ' [data-out="deref"]', 'data-how')) === 'local');
await cell(0, 1);
check('clicking again takes the grant back', (await on(0, 1)) === 0 && (await status()).startsWith('Taken back'));

// ---- the halo exchange
const copy = haloExchange('copy');
await press(B + ' [aria-label="Halo exchange"]', 'host-driven copies');
const h1 = await page.$eval(B + ' [data-out="halo"]', (e) => ({ s: Number(e.dataset.staging), w: Number(e.dataset.waits), c: Number(e.dataset.copies) }));
check('host-driven copies: 2 staging buffers, 2 host waits, 2 host copies, as the model counts', h1.s === copy.staging && h1.w === copy.hostWaits && h1.c === copy.hostCopies && h1.s === 2, JSON.stringify(h1));
check('and the model\'s six steps are listed', (await page.$$(B + ' [data-part="halo-steps"] li')).length === copy.steps.length);
await press(B + ' [aria-label="Halo exchange"]', 'peer loads');
const h2 = await page.$eval(B + ' [data-out="halo"]', (e) => ({ s: Number(e.dataset.staging), w: Number(e.dataset.waits), c: Number(e.dataset.copies) }));
check('peer loads: no staging buffer, no host wait, no host copy', h2.s === 0 && h2.w === 0 && h2.c === 0, JSON.stringify(h2));
check('the ordering step is marked as ours', (await text(B + ' [data-part="halo-steps"] li[data-kind="order"]')).includes('ours'));
check('with no grants the peer exchange is not ready, both directions missing', (await attr(B + ' [data-out="halo-ready"]', 'data-ready')) === '0' && (await text(B + ' [data-out="halo-ready"]')).includes('GPU 0 → GPU 1 and GPU 1 → GPU 0'));
await cell(0, 1);
check('with only GPU 0 → GPU 1, GPU 1 → GPU 0 is still missing', (await text(B + ' [data-out="halo-ready"]')).includes('GPU 1 → GPU 0') && !(await text(B + ' [data-out="halo-ready"]')).includes('GPU 0 → GPU 1'));
await cell(1, 0);
check('with both directions it is ready', (await attr(B + ' [data-out="halo-ready"]', 'data-ready')) === '1');
if (out) await page.screenshot({ path: out + '-c5-two-way.png' });

// ---- the eight-peer limit
const changeN = async (n) => {
  await page.select(B + ' select[aria-label="Number of GPUs"]', String(n));
  await pause(100);
};
await changeN(9);
check('changing the system clears every grant', (await peers()).length === 9 && (await peers()).every((p) => p === 0) && (await status()).includes('cleared'));
check('nine GPUs: all to all is possible within the limit', (await attr(B + ' [data-out="all-to-all"]', 'data-possible')) === '1');
await press(B, 'Enable every pair');
const e9 = enableAll({ n: 9, nvswitch: false, iommu: false });
check(`nine GPUs: all ${e9.granted} grants succeed, eight peers each`, (await status()).includes(`all ${e9.granted} grants succeeded`) && (await peers()).every((p) => p === 8), await status());
await changeN(16);
check('sixteen GPUs without NVSwitch: all to all is impossible', (await attr(B + ' [data-out="all-to-all"]', 'data-possible')) === '0' && (await text(B + ' [data-out="all-to-all"]')).includes('needs 15 peers'));
await press(B, 'Enable every pair');
const e16 = enableAll({ n: 16, nvswitch: false, iommu: false });
check(`sixteen GPUs: ${e16.granted} succeed and ${e16.refused} are refused`, (await status()).includes(`${e16.granted} grants succeeded and ${e16.refused} were refused`), await status());
check(`every row stops at ${MAX_PEERS_WITHOUT_NVSWITCH} peers`, (await peers()).every((p) => p === MAX_PEERS_WITHOUT_NVSWITCH));
await cell(0, 15);
check('a ninth grant is refused with the documented reason', (await on(0, 15)) === 0 && (await status()).includes('the most a device supports without NVSwitch'), await status());
check('refused cells are marked', (await page.$eval(`${B} button[data-cell="0-15"]`, (b) => b.textContent.trim())) === '×');
if (out) await page.screenshot({ path: out + '-c5-sixteen.png' });
await setBox('NVSwitch', true);
check('NVSwitch makes all to all possible', (await attr(B + ' [data-out="all-to-all"]', 'data-possible')) === '1');
await press(B, 'Enable every pair');
check('with NVSwitch all 240 grants succeed', (await status()).includes(`all ${enableAll({ n: 16, nvswitch: true, iommu: false }).granted} grants succeeded`) && (await peers()).every((p) => p === 15));
await setBox('NVSwitch', false);
await changeN(2);

// ---- IOMMU
await setBox('IOMMU', true);
await cell(0, 1);
check('bare-metal Linux with the IOMMU on refuses every grant', (await on(0, 1)) === 0 && (await status()).includes('IOMMU'));
check('and all to all is impossible there', (await attr(B + ' [data-out="all-to-all"]', 'data-possible')) === '0' && (await attr(B + ' [data-out="all-to-all"]', 'data-all')) === 'impossible');
// With NVSwitch too, A8 says nothing: the explorer says "not documented" and grants nothing.
await setBox('NVSwitch', true);
await cell(0, 1);
check('NVSwitch with the IOMMU on: no grant, and the status says not documented rather than refused', (await on(0, 1)) === 0 && (await status()).includes('Not documented') && !(await status()).includes('Refused'), await status());
check('  the cell shows "?" and the status is not marked as a failure', (await page.$eval(`${B} button[data-cell="0-1"]`, (b) => b.textContent.trim())) === '?' && !(await page.$eval(B + ' [data-out="status"]', (e) => e.classList.contains('bad'))));
check('  all to all is "not documented", not impossible', (await attr(B + ' [data-out="all-to-all"]', 'data-all')) === 'unknown' && (await text(B + ' [data-out="all-to-all"]')).includes('not documented'));
await setBox('NVSwitch', false);
await setBox('IOMMU', false);

// ---- questions
check('q-removes: the staging copy and the host set-up', (await choose('what does loading the neighbour', 'The staging copy of the halo')).includes('Correct.'));
check('q-oneway: one direction only', (await choose('Which loads now work', "Kernels on GPU 0 may load from GPU 1's memory. The reverse")).includes('Correct.'));
check('q-eight: no, eight peers without NVSwitch', (await choose('sixteen GPUs and no NVSwitch', 'No. Each GPU would need 15 peers')).includes('Correct.'));
const wrong = await choose('What becomes of the peer-access path', 'Bind the other GPU');
check('q-metal: a wrong pick explains why', wrong.includes('Not quite.') && wrong.includes('blit copies'));
check('q-metal: no Metal counterpart on Apple silicon', (await choose('What becomes of the peer-access path', 'It has no Metal counterpart to port to')).includes('Correct.'));

// ---- narrow screen
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((B) => ({ page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, bench: document.querySelector(B).getBoundingClientRect().width <= document.documentElement.clientWidth }), B);
check('at a narrow width nothing scrolls the page sideways', narrow.page && narrow.bench, JSON.stringify(narrow));
if (out) {
  await page.$eval(B, (el) => el.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-c5-narrow.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
