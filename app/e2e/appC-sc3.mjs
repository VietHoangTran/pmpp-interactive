// Browser check for appendix C, section C.3: the copy-direction explorer (both
// address-space modes, the registered buffer), the "Follow the list" stepper on
// every system, the chase kernel listing, the reader's Metal address observation
// and the questions.
// Usage: node e2e/appC-sc3.mjs [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto, BASE } from './browser.mjs';
import { inferKind, REGISTERED } from '../src/sim/uvacopy.ts';
import { cell } from '../src/sim/derefmap.ts';

const out = process.argv[2];
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
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
const C = 'section.bench[aria-label="Which way does the copy go?"]';
const L = 'section.bench[aria-label="Follow the list"]';
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
const result = () => page.$eval(C + ' [data-kind]', (e) => ({ kind: e.dataset.kind, detail: e.querySelector('[data-detail]').textContent }));
const place = async (src, dst) => {
  await setInput(C + ' input[aria-label="src address"]', src);
  await setInput(C + ' input[aria-label="dst address"]', dst);
};

await goto(page, 'appC/C.3');
await page.waitForSelector(C);

// ---- copy direction, one shared address space
const cases = [
  [0x3000, 0xc400, 'the q-direction case: host to GPU 1'],
  [0x8100, 0x1100, 'GPU 0 to host'],
  [0x8100, 0xc100, 'GPU 0 to GPU 1'],
  [0x2100, 0x1100, 'host to host'],
  [0x1100, 0x7800, 'a value in a gap'],
];
for (const [src, dst, name] of cases) {
  await place(src, dst);
  const r = await result();
  const m = inferKind(dst, src, 'uva', false);
  check(`UVA, ${name}: ${m.kind}`, r.kind === m.kind && r.detail === m.detail, JSON.stringify(r));
}
await place(0x3000, 0xc400);
check('the q-direction case names GPU 1', (await result()).detail.includes('to GPU 1'));
// Click on the map to place dst in GPU 0's range.
await clickText(C, 'dst');
await page.$eval(C + ' svg', (s) => s.scrollIntoView({ block: 'center' }));
await pause(80);
const box = await page.$eval(C + ' svg', (s) => {
  const b = s.getBoundingClientRect();
  return { x: b.left, y: b.top, w: b.width, h: b.height };
});
await page.mouse.click(box.x + (box.w * (14 + (0x9000 / 0x10000) * (640 - 28))) / 640, box.y + box.h * 0.6);
await pause(80);
check('clicking the map places the chosen marker', (await result()).kind === inferKind(0x9000, 0x3000, 'uva').kind && (await page.$eval(C + ' input[aria-label="dst address"]', (e) => e.value)) === String(0x9000));

// ---- registered buffer
await page.evaluate((c) => document.querySelector(c + ' input[type=checkbox]').click(), C);
await pause(60);
await place(REGISTERED.host[0] + 0x100, 0x9000);
check('registered buffer, host-side address: still host to device', (await result()).kind === inferKind(0x9000, REGISTERED.host[0] + 0x100, 'uva', true).kind);
await place(REGISTERED.device[0] + 0x100, 0x9000);
let r = await result();
check('registered buffer, device-side address: one buffer, two addresses, not guessed', r.kind === 'alias' && r.detail === inferKind(0x9000, REGISTERED.device[0] + 0x100, 'uva', true).detail, JSON.stringify(r));
if (out) {
  await page.$eval(C, (b) => b.scrollIntoView());
  await page.screenshot({ path: out + '-registered.png' });
}

// ---- separate spaces
await click(C + ' button[data-mode="separate"]');
await place(0x3000, 0x2000);
r = await result();
check('separate spaces: cannot tell', r.kind === 'ambiguous' && r.detail === inferKind(0x2000, 0x3000, 'separate').detail, JSON.stringify(r));
check('separate spaces: three lanes, and the registered option is off', (await page.$$eval(C + ' svg > g > rect[fill="none"]', (x) => x.length)) === 3 && (await page.$eval(C + ' input[type=checkbox]', (e) => e.disabled)));
check('the result box is marked as a failure', await page.$eval(C + ' [data-kind]', (e) => e.classList.contains('bad')));
if (out) await page.screenshot({ path: out + '-separate.png' });
await click(C + ' button[data-mode="uva"]');

// ---- follow the list
const ROW = { malloc: 'malloc', cudaHostAlloc: 'pinned', cudaMalloc: 'device', cudaMallocManaged: 'managed' };
const ERA = { uva: 'uva', pascal: 'pascal', hmm: 'hmm', 'um-limited': 'um-limited' };
const VALUES = [3, 1, 4, 1, 5];
/** What the hops should be, from the map's cells (ptrchase.ts's rule, restated for node). */
const expected = (nodes, setting) => {
  const hops = [];
  let sum = 0;
  for (let i = 0; i < nodes.length; i++) {
    const v = cell('cuda', ERA[setting], ROW[nodes[i]], i ? 'chased' : 'arg').verdict;
    const h = v === 'local' || v === 'bulk' ? 'local' : v === 'remote' ? 'remote' : v === 'fault' ? 'fault' : 'invalid';
    if (h !== 'invalid') sum += VALUES[i];
    hops.push({ verdict: h, sum });
    if (h === 'invalid') break;
  }
  return hops;
};
const shownHops = () => page.$$eval(L + ' tr[data-hop]', (rs) => rs.map((r) => ({ verdict: r.dataset.verdict, sum: Number(r.lastElementChild.textContent) })));
const nodes = () => page.$$eval(L + ' [data-node] select', (s) => s.map((x) => x.value));
const narrate = () => page.$eval(L + ' [data-narrate]', (e) => e.textContent.replace(/\s+/g, ' '));

let list = await nodes();
check('the stepper starts on CUDA 4 with five nodes and no hop taken', list.length === 5 && (await page.$eval(L + ' select[aria-label="system"]', (s) => s.value)) === 'uva' && (await shownHops()).length === 0);
const want = expected(list, 'uva');
let stepOk = true;
for (let i = 0; i < want.length; i++) {
  await clickText(L, 'Step');
  const shown = await shownHops();
  if (JSON.stringify(shown) !== JSON.stringify(want.slice(0, i + 1))) stepOk = false;
}
check('stepping shows each hop as the model gives it', stepOk, JSON.stringify(await shownHops()));
check('the default list is fine for two hops and fails on the third', JSON.stringify(want.map((h) => h.verdict)) === JSON.stringify(['local', 'remote', 'invalid']));
let n = await narrate();
check('the failing hop names the range, the rule and the stop', n.includes('host heap (malloc)') && n.includes(cell('cuda', 'uva', 'malloc', 'chased').cost) && n.includes('The kernel read 2 values correctly; the sum so far is 4.'), n);
check('Step is disabled after the failure', !(await clickText(L, 'Step')));
check('the failing node is marked', (await page.$eval(L + ' [data-node="2"]', (e) => e.dataset.verdict)) === 'invalid');
if (out) {
  await page.$eval(L, (b) => b.scrollIntoView());
  await page.screenshot({ path: out + '-list-fail.png' });
}

// Every system, run to failure (or to the end).
for (const s of ['pascal', 'hmm', 'um-limited', 'uva']) {
  await setInput(L + ' select[aria-label="system"]', s);
  check(`${s}: changing the system resets the walk`, (await shownHops()).length === 0);
  await clickText(L, 'Run to failure');
  const w = expected(list, s);
  const shown = await shownHops();
  check(`${s}: run to failure matches the model`, JSON.stringify(shown) === JSON.stringify(w), JSON.stringify({ shown, w }));
}
await setInput(L + ' select[aria-label="system"]', 'hmm');
await clickText(L, 'Run to failure');
n = await narrate();
check('HMM or ATS: the malloc node faults in and the kernel sums all five, 14', n.includes('the sum is 14'), n);

// Change one node's allocator: managed under Pascal.
await setInput(L + ' select[aria-label="system"]', 'pascal');
await setInput(L + ' select[aria-label="node 3 allocator"]', 'cudaMallocManaged');
list = await nodes();
await clickText(L, 'Run to failure');
check('Pascal with the third node managed: the walk completes, the managed hop faults', JSON.stringify(await shownHops()) === JSON.stringify(expected(list, 'pascal')) && (await shownHops())[2].verdict === 'fault');
await setInput(L + ' select[aria-label="node 1 allocator"]', 'malloc');
check('changing a node resets the walk', (await shownHops()).length === 0);
await clickText(L, 'Step');
check('a malloc head fails at the very first hop under Pascal', JSON.stringify(await shownHops()) === JSON.stringify(expected(await nodes(), 'pascal').slice(0, 1)) && (await shownHops())[0].verdict === 'invalid');
await clickText(L, 'Reset');
check('Reset clears the walk', (await shownHops()).length === 0);

// ---- listing
const k = fs.readFileSync(path.join(repo, 'labs/appC/snippets/c3_chase.cu'), 'utf8').trimEnd().split('\n');
const shown = await page.$$eval('.code pre .ln', (ls) => ls.map((l) => l.textContent.replace(/\n$/, '')));
check('the listing is labs/appC/snippets/c3_chase.cu, line for line, front-end-checked', JSON.stringify(shown) === JSON.stringify(k) && (await page.$eval('.code .tabs .status', (s) => s.textContent)) === 'front-end-checked');

// ---- your Mac's addresses
const reports = await (await fetch(BASE + '/api/reports')).json();
const row = reports.metal?.results.filter((x) => x.lab === 'appC/lab1_feeding' && x.variant === 'solution').flatMap((x) => x.bench).find((x) => x.name === 'addresses');
const addr = await page.$eval('[data-addresses]', (e) => ({ kind: e.dataset.addresses, text: e.textContent }));
if (row) check('your Mac: the observation is shown and labelled as not documented', addr.kind === 'some' && addr.text.includes('not documented'), JSON.stringify(addr));
else check('no Metal observation yet: the page says how to get one', addr.kind === 'none' && addr.text.includes('appC/lab1 --bench'), JSON.stringify(addr));

// ---- questions
let q = await question('dst came from');
await pick(q, 'A host-to-device copy into GPU 1');
check('q-direction', (await qText(q)).includes('Correct.'));
q = await question('every pointer works everywhere');
await pick(q, 'A kernel reading cudaMallocHost memory.');
check('q-address-not-access: the mapped-memory distractor says it is allowed', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('Allowed'));
await pick(q, 'Host code dereferencing a cudaMalloc pointer.');
check('q-address-not-access', (await qText(q)).includes('Correct.'));
q = await question('four-node list');
await typeAnswer(q, 2);
check('q-chase: 2', (await qText(q)).includes('Correct.'));
q = await question('To debug a sampler');
await pick(q, 'It is fine on both, only slow');
check('q-hip-host: "fine on both, only slow" is wrong, because CUDA does not allow the read', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('On CUDA the read is not allowed at all'));
await pick(q, 'It works on HIP because AMD documents');
check('q-hip-host: works on HIP, wrong in the CUDA build', (await qText(q)).includes('Correct.') && (await qText(q)).includes('CPUs can access memory allocated via hipMalloc()'));

// ---- notes and voice
const notes = await page.$$eval('.prose > .note', (x) => x.map((e) => e.textContent).join(' || '));
check('Today notes cite A5, ch02-facts A4 and A12; HIP B3; Metal C3; A14', ['A5', 'A4', 'A12', 'B3', 'C3', 'A14'].every((s) => notes.includes(s)));
const prose = await page.$eval('.prose', (e) => e.textContent);
check('GMAC is presented as the book\'s account', prose.includes("that is the book's account"));
check('every margin note directly follows text', await page.evaluate(() => [...document.querySelectorAll('.prose > .note')].every((n) => ['P', 'UL', 'OL'].includes(n.previousElementSibling?.tagName) || n.previousElementSibling?.classList.contains('note'))));
check('no em dash in the page', !prose.includes('—'));
check('answers went to the progress API (intercepted)', puts > 0);
check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
