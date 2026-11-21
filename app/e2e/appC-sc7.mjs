// Browser check for section C.7: the "Does it all fit?" explorer, the "Which paradigm is
// this machine?" explorer (switches, table rows, the "Yours" button against the app's
// own /api/reports) and the questions. Expected values come from src/sim/addrwidth.ts
// and src/sim/umparadigm.ts, imported here directly.
// Usage: APP_URL=http://localhost:5173 node e2e/appC-sc7.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { GiB, binary, coverage, isEmulator, memoryFigures, smallestWidth } from '../src/sim/addrwidth.ts';
import { ROWS, checklist, paradigm, paradigmFromRun } from '../src/sim/umparadigm.ts';

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
const F = 'section.bench[aria-label="Does it all fit?"]';
const P = 'section.bench[aria-label="Which paradigm is this machine?"]';
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
const setRange = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(80);
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

const reports = await (await fetch(BASE + '/api/reports')).json();

await goto(page, 'appC/C.7');
await page.waitForSelector(F);

check('q-49: 49 bits is correct', (await answer('two separate ranges of', 49)).includes('Correct.'));

// ---- does it all fit
const fit = async () => ({ fits: await attr(F + ' [data-part="strip"]', 'data-fits'), smallest: Number(await attr(F + ' [data-out="smallest"]', 'data-bits')), verdict: await text(F + ' [data-part="verdict"]') });
let f = await fit();
let c = coverage(49, 48, 64 * GiB);
check('49 bits, 48-bit CPU, 64 GiB: fits, smallest width 49', f.fits === '1' && c.fits && f.smallest === smallestWidth(48, 64 * GiB) && f.smallest === 49 && f.verdict === `Fits, with ${binary(c.spare).text} to spare.`, JSON.stringify(f));
await press(F + ' [aria-label="GPU virtual address width"]', '48');
f = await fit();
check('48 bits: does not fit, the GPU memory falls past the end', f.fits === '0' && f.verdict === `Does not fit: ${binary(coverage(48, 48, 64 * GiB).over).text} of GPU memory falls past the end.`, JSON.stringify(f));
await page.select(F + ' select[aria-label="CPU user space"]', '47');
await pause(80);
f = await fit();
check("Linux's default 47 bits: fits in 48, smallest width 48", f.fits === '1' && f.smallest === 48 && coverage(48, 47, 64 * GiB).fits, JSON.stringify(f));
await page.select(F + ' select[aria-label="CPU user space"]', '56');
await press(F + ' [aria-label="GPU virtual address width"]', '49');
f = await fit();
check('56-bit user space in 49 bits: the CPU range alone is 128 times too big; smallest width 57', f.fits === '0' && f.verdict.includes('128 times') && f.smallest === smallestWidth(56, 64 * GiB) && f.smallest === 57, JSON.stringify(f));
await page.select(F + ' select[aria-label="CPU user space"]', '48');
await press(F + ' [aria-label="GPU virtual address width"]', '32');
f = await fit();
check('32 bits against a 48-bit CPU: 65,536 times too big', f.fits === '0' && f.verdict.includes('65,536 times'), JSON.stringify(f));
await press(F + ' [aria-label="GPU virtual address width"]', '49');
await setRange(F + ' input[type=range]', 19);
f = await fit();
check('512 TiB of GPU memory beside a 48-bit CPU needs 50 bits', f.smallest === smallestWidth(48, 2 ** 19 * GiB) && f.smallest === 50 && f.fits === '0', JSON.stringify(f));
await setRange(F + ' input[type=range]', 18);
f = await fit();
check('256 TiB, exactly 2^48 bytes, still fits in 49', f.smallest === 49 && f.fits === '1', JSON.stringify(f));
const yoursGpu = memoryFigures(reports).filter((x) => x.field === 'totalGlobalMem');
const presets = await page.$$(F + ' button[data-yours]');
check('one "yours" preset per CUDA or HIP device in the reports', presets.length === yoursGpu.length, `${presets.length} vs ${yoursGpu.length}`);
if (yoursGpu.length) {
  await presets[0].click();
  await pause(80);
  check('the preset places your device memory', (await text(F + ' [data-out="smallest"]')).includes(binary(yoursGpu[0].bytes).text));
} else check('with no CUDA or HIP device the explorer says so', (await text(F)).includes('No CUDA or HIP device is recorded'));
if (out) await page.screenshot({ path: out + '-c7-fit.png' });

// ---- which paradigm
const state = async () => ({
  key: await attr(P + ' [data-out="paradigm"]', 'data-key'),
  listed: await attr(P + ' [data-out="paradigm"]', 'data-listed'),
  checks: await page.$$eval(P + ' li[data-check][data-answer]', (l) => Object.fromEntries(l.map((x) => [x.dataset.check, x.dataset.answer]))),
  on: await page.$$eval(P + ' tr[data-row]', (r) => r.filter((x) => x.dataset.on === '1').map((x) => x.dataset.row)),
});
const expect = (a) => ({ key: paradigm(a).row.key, listed: paradigm(a).listed ? '1' : '0', checks: Object.fromEntries(checklist(a).map((k) => [k.key, k.answer])) });
const box = async (k, v) => {
  await page.evaluate((P, k, v) => {
    const b = document.querySelector(`${P} input[data-attr="${k}"]`);
    if (b.checked !== v) b.click();
  }, P, k, v);
  await pause(80);
};
const same = (s, a) => {
  const e = expect(a);
  return s.key === e.key && s.listed === e.listed && JSON.stringify(s.checks) === JSON.stringify(e.checks) && s.on.length === 1 && s.on[0] === e.key;
};
const A = (c, p, h) => ({ concurrentManagedAccess: c, pageableMemoryAccess: p, usesHostPageTables: h });
let s = await state();
check('starts on "managed allocations only", its row highlighted, the checklist from the model', same(s, A(true, false, false)), JSON.stringify(s));
await box('pageableMemoryAccess', true);
s = await state();
check('PageableMemoryAccess on: HMM, malloc pointers work', same(s, A(true, true, false)) && s.checks.malloc === 'yes', JSON.stringify(s));
await box('usesHostPageTables', true);
s = await state();
check('all three on: ATS, coherence by cache lines', same(s, A(true, true, true)) && (await text(P + ' li[data-check="granularity"]')).includes('cache lines'), JSON.stringify(s));
await box('concurrentManagedAccess', false);
s = await state();
check('ConcurrentManagedAccess off with the others on: limited, flagged as not in the table', same(s, A(false, true, true)) && s.listed === '0' && (await text(P)).includes('not in the table'), JSON.stringify(s));
for (const r of ROWS) {
  await page.click(`${P} tr[data-row="${r.key}"]`);
  await pause(80);
  s = await state();
  check(`clicking the "${r.name}" row selects it`, s.key === r.key && s.listed === '1' && s.on[0] === r.key, JSON.stringify(s));
}
await page.click(`${P} tr[data-row="limited"]`);
await pause(80);
s = await state();
check('limited: every capability is "no"', Object.values(s.checks).every((v) => v === 'no'), JSON.stringify(s.checks));
if (out) await page.screenshot({ path: out + '-c7-paradigm.png' });

// "Yours", against what the reports actually hold.
const found = Object.keys(reports).filter((b) => !isEmulator(b) && b !== 'metal').map((b) => ({ b, a: paradigmFromRun(reports[b]) })).filter((x) => x.a);
await press(P, 'Yours');
const yn = await text(P + ' [data-out="yours"]');
if (found.length) {
  s = await state();
  check('"Yours" loads lab C.1\'s attributes from the reports', yn.includes('as lab C.1 reported it') && s.key === paradigm(found[0].a).row.key, yn);
} else {
  check('with no CUDA or HIP lab C.1 result, "Yours" says how to get one', yn.includes('python3 labs/pmpp.py run appC/lab1 --bench'), yn);
  if ('metal' in reports) check('and explains that Metal has no such attributes', yn.includes('Your Metal device has no such attributes'), yn);
}

// ---- questions
check('q-pascal-malloc: it fails', (await choose('linked list that the host built', "It fails: Pascal's faulting covers managed memory")).includes('Correct.'));
const wrong = await choose('A device reports', 'Everything works, malloc and mmap pointers included');
check('q-paradigm: a wrong pick explains why', wrong.includes('Not quite.') && wrong.includes('PageableMemoryAccess = 1'));
check('q-paradigm: managed allocations only', (await choose('A device reports', 'Full Unified Memory for managed allocations only')).includes('Correct.'));
check('q-wsl: the limited one', (await choose('runs under WSL', 'The limited one')).includes('Correct.'));

// ---- narrow screen
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((F, P) => ({ page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, f: document.querySelector(F).getBoundingClientRect().width <= document.documentElement.clientWidth, p: document.querySelector(P).getBoundingClientRect().width <= document.documentElement.clientWidth }), F, P);
check('at a narrow width nothing scrolls the page sideways', narrow.page && narrow.f && narrow.p, JSON.stringify(narrow));
const cpuSel = await page.evaluate((F) => {
  const s = document.querySelector(F + ' select[aria-label="CPU user space"]').getBoundingClientRect();
  const body = document.querySelector(F + ' .body');
  const b = body.getBoundingClientRect();
  return { right: Math.round(s.right), bodyRight: Math.round(b.right), scrolls: body.scrollWidth > body.clientWidth + 1 };
}, F);
check('  the CPU user space select fits inside the bench body, which does not scroll sideways', cpuSel.right <= cpuSel.bodyRight && !cpuSel.scrolls, JSON.stringify(cpuSel));
if (out) {
  await page.$eval(P, (el) => el.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-c7-narrow.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
