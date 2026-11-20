// Browser check for section 5.6: the shared-memory budget calculator and the erratum hunt.
// Usage: node e2e/s56.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
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
const B = 'section.bench[aria-label="Shared-memory budget"]';
const H = 'section.bench[aria-label="Erratum hunt: Fig. 5.14"]';
const shot = async (name, sel) => {
  if (!out) return;
  await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};

await goto(page, 'ch05/5.6');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
await page.waitForSelector(B);
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));

// What the calculator shows right now.
const state = () =>
  page.$eval(B, (root) => {
    const o = root.querySelector('output[data-out="occupancy"]');
    const fit = {};
    for (const tr of root.querySelectorAll('tr[data-budget]')) fit[tr.dataset.budget] = tr.dataset.fit;
    return {
      occupancy: o.dataset.occupancy,
      binding: o.dataset.binding,
      shown: o.textContent.trim(),
      resident: root.querySelector('[data-out="resident"]')?.textContent ?? '',
      verdict: root.querySelector('[data-out="verdict"]').textContent,
      device: root.querySelector('select').value,
      fit,
      marked: [...root.querySelectorAll('tr[data-budget]')].filter((tr) => tr.textContent.includes('binds')).map((tr) => tr.dataset.budget),
    };
  });
const preset = async (key) => {
  await page.click(`${B} button[data-preset="${key}"]`);
  await pause(80);
};
const pressed = (key) => page.$eval(`${B} button[data-preset="${key}"]`, (b) => b.getAttribute('aria-pressed'));
const setField = async (label, value) => {
  const handle = await page.evaluateHandle(
    (B, label) => [...document.querySelectorAll(B + ' label.field')].find((l) => l.textContent.trim().startsWith(label))?.querySelector('input'),
    B,
    label,
  );
  const el = handle.asElement();
  if (!el) throw new Error('no field labelled ' + label);
  // Select what is there and type over it, as a reader would.
  await el.focus();
  await el.evaluate((e) => e.select());
  await el.type(String(value));
  await pause(80);
};

// ---- the device list: the book's example, every CUDA/HIP device that reports per-SM fields, custom; never emu.
const options = await page.$$eval(`${B} select option`, (o) => o.map((x) => ({ value: x.value, text: x.textContent })));
const perSm = ['sharedMemPerMultiprocessor', 'maxThreadsPerMultiProcessor', 'maxBlocksPerMultiProcessor', 'regsPerMultiprocessor'];
const expectDevices = Object.keys(reports).filter((b) => b !== 'emu' && perSm.every((f) => f in reports[b].device));
check('device list starts with the book\'s example and ends with custom numbers',
  options[0].value === 'book' && options[0].text.includes("book's example") && options.at(-1).value === 'custom', JSON.stringify(options));
check('the emulator is not offered as a device', !options.some((o) => o.value === 'emu'), JSON.stringify(options));
check(`devices from this machine's reports are offered (${expectDevices.join(', ') || 'none report per-SM fields here'})`,
  expectDevices.every((b) => options.some((o) => o.value === b)) && options.length === expectDevices.length + 2, JSON.stringify(options));
if (!expectDevices.length)
  check('with no CUDA/HIP device, the empty state says what to run', (await page.$eval(B, (e) => e.textContent)).includes('./verify.sh'));

// ---- the book's example device, each preset
await page.select(`${B} select`, 'book');
await pause(80);
let s = await state();
check('default is the tiled matmul with T = 16 on the book\'s device', s.device === 'book' && (await pressed('t16')) === 'true');

await preset('t16');
s = await state();
check('book device, tiled matmul T = 16: 100% occupancy', s.occupancy === '100' && s.shown === '100%', JSON.stringify(s));
check('  8 blocks by thread slots, 32 by block slots, 114 by shared memory; registers not counted',
  s.fit.threads === '8' && s.fit.blocks === '32' && s.fit.shared === '114' && s.fit.regs === 'none', JSON.stringify(s.fit));
check('  thread slots are the binding budget and the row is marked', s.binding === 'threads' && s.marked.join() === 'threads', JSON.stringify(s));
check('  says "Full occupancy"', s.verdict.includes('Full occupancy'), s.verdict);
await shot('t16', B);

await preset('t32');
s = await state();
check('book device, tiled matmul T = 32: still 100%, with 2 blocks of 1024 threads',
  s.occupancy === '100' && s.fit.threads === '2' && s.fit.shared === '28' && s.resident === '2,048', JSON.stringify(s));

await preset('book38');
s = await state();
check('book device, 38 KB with 256 threads: 75% occupancy', s.occupancy === '75' && s.shown === '75%', JSON.stringify(s));
check('  bound by shared memory (6 blocks), row marked, 1,536 resident threads',
  s.binding === 'shared' && s.marked.join() === 'shared' && s.fit.shared === '6' && s.resident === '1,536', JSON.stringify(s));
check('  verdict names shared memory', /Shared memory binds/.test(s.verdict), s.verdict);
check('  152 B per thread against an allowance of 114', (await page.$eval(B, (e) => e.textContent)).includes('152 B of shared memory per thread') &&
  (await page.$eval(B, (e) => e.textContent)).includes('at most 114 B per thread'));
await shot('book38', B);

// ---- exercise 12 brings its own limits
await preset('ex12a');
s = await state();
check('exercise 12 (a) switches to custom limits', s.device === 'custom' && (await pressed('ex12a')) === 'true', JSON.stringify(s));
check('exercise 12 (a): 75%, bound by shared memory', s.occupancy === '75' && s.binding === 'shared', JSON.stringify(s));
check('  32 by threads, 32 by block slots, 24 by shared memory, 37 by registers',
  s.fit.threads === '32' && s.fit.blocks === '32' && s.fit.shared === '24' && s.fit.regs === '37', JSON.stringify(s.fit));
await preset('ex12b');
s = await state();
check('exercise 12 (b): 100%, thread slots and registers run out together',
  s.occupancy === '100' && s.binding === 'threads+regs' && s.fit.threads === '8' && s.fit.regs === '8' && s.fit.shared === '12', JSON.stringify(s));

// ---- typed values: the register cliff of exercise 12 (b), then fully custom limits
await setField('Registers per thread', 33);
s = await state();
check('typing 33 registers per thread: 7 blocks, 87.5%, bound by registers',
  s.occupancy === '87.5' && s.binding === 'regs' && s.fit.regs === '7' && s.resident === '1,792', JSON.stringify(s));
check('  the preset is no longer marked as active', (await pressed('ex12b')) === 'false');
await setField('Thread slots per SM', 1536);
await setField('Block slots per SM', 16);
await setField('Shared memory per SM (bytes)', 65536);
await setField('Registers per SM', 32768);
await setField('Shared memory per block (bytes)', 30000);
await setField('Threads per block', 128);
await setField('Registers per thread', 20);
s = await state();
// threads 1536/128 = 12; blocks 16; shared 65536/30000 = 2; regs 32768/2560 = 12 -> 2 blocks, 256 threads, 16.7%
check('custom limits and kernel: 2 blocks by shared memory, 16.7%',
  s.device === 'custom' && s.fit.threads === '12' && s.fit.blocks === '16' && s.fit.shared === '2' && s.fit.regs === '12' &&
  s.binding === 'shared' && s.occupancy === '16.7' && s.resident === '256', JSON.stringify(s));
await setField('Shared memory per block (bytes)', 70000);
s = await state();
check('a block larger than the SM: 0% and a "does not fit" failure', s.occupancy === '0' && s.verdict.includes('does not fit'), JSON.stringify(s));
await setField('Shared memory per block (bytes)', 0);
await setField('Threads per block', 768);
await setField('Registers per thread', 0);
s = await state();
check('768-thread blocks on 1536 slots, nothing else counted: 100%', s.occupancy === '100' && s.fit.shared === 'none' && s.fit.regs === 'none', JSON.stringify(s));

// ---- back on the book's device: a block size that does not divide the SM
await page.select(`${B} select`, 'book');
await pause(80);
s = await state();
check('768-thread blocks on the book\'s device: 2 blocks, 75%, thread slots bind "not cleanly"',
  s.occupancy === '75' && s.binding === 'threads' && s.verdict.includes('not cleanly'), JSON.stringify(s));
await setField('Threads per block', 2000);
check('more threads than a block may have is flagged', (await page.$eval(B, (e) => e.textContent)).includes('block too large'));

// ---- Metal: what it does expose, and no occupancy
const metal = Object.entries(reports).filter(([, r]) => 'maxThreadgroupMemoryLength' in r.device);
for (const [name, run] of metal) {
  const t = await page.$eval(`${B} [data-metal="${name}"]`, (e) => e.textContent);
  const cap = Number(run.device.maxThreadgroupMemoryLength);
  check(`Metal panel for ${name} shows the reported maxThreadgroupMemoryLength (${cap.toLocaleString('en-US')} B)`,
    t.includes('maxThreadgroupMemoryLength') && t.includes(cap.toLocaleString('en-US') + ' B') && t.includes(String(run.device.name)), t);
  check('  names maxTotalThreadsPerThreadgroup as a per-pipeline property', t.includes('maxTotalThreadsPerThreadgroup') && t.includes('pipeline'));
  check('  says these are per-threadgroup limits and that we found no per-SM equivalent',
    t.includes('limits on one threadgroup') && t.includes('we found no documented Metal equivalent') && t.includes('no occupancy is computed'));
  await preset('t16');
  const fits = await page.$eval(`${B} [data-metal="${name}"]`, (e) => e.textContent);
  check('  checks the entered bytes per block against it', 2048 <= cap ? fits.includes('fits in one threadgroup') : fits.includes('does not fit'), fits);
}
if (!metal.length) console.log('note  no Metal device in the reports on this machine; Metal panel not exercised');

// ---- erratum hunt
await page.waitForSelector(H);
check('the corrected kernel is hidden before the hunt is checked', (await page.$$('figure.code')).length === 0);
check('the quote is the three printed lines', await page.$eval(`${H} pre`, (e) =>
  e.textContent.includes('extern __shared__ char float Mds_Nds[];') && e.textContent.includes('(float *) Mds_Nds + Mds_sz;') && e.textContent.includes('size/2, size/2);')));
const claims = await page.$$eval(`${H} label[data-claim]`, (ls) => ls.map((l) => l.dataset.claim));
check('six candidate statements', claims.length === 6, claims.join());
// Tick three real problems and one distractor; leave one real problem and one distractor alone.
for (const c of ['two-types', 'spelling', 'bytes-vs-elements', 'unsized']) await page.click(`${H} label[data-claim="${c}"] input`);
await page.evaluate((H) => [...document.querySelectorAll(H + ' button')].find((b) => b.textContent.trim() === 'Check').click(), H);
await pause(200);
const hunt = await page.$eval(`${H} [data-out="hunt"]`, (e) => ({ found: e.dataset.found, wrong: e.dataset.false, text: e.textContent }));
check('hunt reports 3 of 4 real problems found and 1 false alarm', hunt.found === '3' && hunt.wrong === '1' && hunt.text.includes('3 of the 4'), JSON.stringify(hunt));
const marks = await page.$$eval(`${H} label[data-claim]`, (ls) => Object.fromEntries(ls.map((l) => [l.dataset.claim, l.className + '|' + l.textContent])));
check('each statement is revealed as real or not, with the reader\'s call marked',
  marks['two-types'].startsWith('right|') && marks['two-types'].includes('A real problem. You caught it.') &&
  marks['unsized'].startsWith('wrong|') && marks['unsized'].includes('Not a problem. You flagged it.') &&
  marks['half-size'].startsWith('wrong|') && marks['half-size'].includes('A real problem. You missed it.') &&
  marks['too-small'].startsWith('right|') && marks['too-small'].includes('Not a problem. You left it alone, rightly.'), JSON.stringify(marks));
check('the explanation for line 07 does the arithmetic', marks['bytes-vs-elements'].includes('16,384 bytes'));
check('the fifth erratum (Figs. 5.9 and 5.11) is mentioned', hunt.text.includes('5.11') && hunt.text.includes('5.13'));
await shot('hunt', H);

// ---- the corrected kernel appears, from the tested lab files
await page.waitForSelector('figure.code .tabs button');
const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
check('corrected kernel has CUDA, Metal kernel and Metal host tabs', tabs.join('|') === 'CUDA|Metal kernel|Metal host', tabs.join('|'));
const file = (f) => page.evaluate((f) => fetch('/api/lab-file?path=ch05/lab5_dynamic/solution/' + f).then((r) => r.text()), f);
const shown = () => page.$eval('figure.code pre', (e) => e.textContent.replace(/\n$/, ''));
const same = (a, b) => a.replace(/\s+$/g, '') === b.replace(/\s+$/g, '');
await page.waitForFunction(() => document.querySelector('figure.code pre').textContent.includes('extern __shared__ float tiles[];'), { timeout: 10000 });
check('CUDA tab is labs/ch05/lab5_dynamic/solution/kernel.cu, verbatim', same(await shown(), await file('kernel.cu')));
check('CUDA tab status', (await page.$eval('figure.code .tabs .status', (e) => e.textContent)) === 'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware');
await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')][1].click());
await pause(100);
check('Metal kernel tab is solution/kernel.metal, verbatim', same(await shown(), await file('kernel.metal')));
check('Metal kernel tab status', (await page.$eval('figure.code .tabs .status', (e) => e.textContent)) === 'verified on Metal');
await page.evaluate(() => [...document.querySelectorAll('figure.code .tabs button')][2].click());
await pause(100);
check('Metal host tab is solution/launch.mm, verbatim', same(await shown(), await file('launch.mm')) && (await shown()).includes('setThreadgroupMemoryLength'));
await shot('kernel', 'figure.code');

// ---- the three questions are registered with the chapter and rendered
const prompts = await page.$$eval('.q .prompt', (p) => p.map((x) => x.textContent));
check('exercise 12 questions say "per block"', prompts.filter((t) => t.includes('per block') && t.includes('96 KB')).length === 2);
check('the tile-doubling question is present', prompts.some((t) => t.includes('TILE_WIDTH 32')));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
