// Browser check for section 4.1: the prediction about resident threads, the explorer
// "The device, level by level" in each backend's vocabulary and on each device this
// machine has a report for, and the three questions. Every expected name, count and
// query comes from the model (src/sim/blocksched.ts), imported here directly, applied
// to the same reports the page reads (/api/reports).
// Usage: node e2e/ch04-s41.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import {
  BOOK_DEVICE,
  BOOK_DEVICE_TODAY,
  TERMS,
  deviceFromReport,
  isGpuBackend,
  levels,
  noReport,
  residentThreads,
  threadsPerCore,
  totalCores,
} from '../src/sim/blocksched.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file.
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
const clean = (s) => s.replace(/\s+/g, ' ').trim();
// The page's number format (src/lib/reports.ts, fmt).
const fmt = (v, digits = 3) => (Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : v === 0 ? '0' : Number(v.toPrecision(digits)).toString());
const B = 'section.bench[aria-label="The device, level by level"]';
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const question = (part) => page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(part)), part);
const qText = (q) => q.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
const pick = async (q, label) => {
  await q.evaluate((e, label) => {
    [...e.querySelectorAll('.opts label')].find((l) => l.textContent.trim().startsWith(label)).querySelector('input').click();
  }, label);
  await (await q.$('button.primary')).click();
  await pause(120);
};
const clickText = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const btn = [...document.querySelectorAll(scope + ' button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(120);
  return done;
};
const setSelect = async (aria, value) => {
  await page.evaluate((B, aria, value) => {
    const s = document.querySelector(`${B} select[aria-label="${aria}"]`);
    s.value = String(value);
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, B, aria, value);
  await pause(120);
};
const rows = () => page.$$eval(B + ' table.levels tbody tr', (trs) => trs.map((tr) => {
  const td = tr.querySelectorAll('td');
  const says = td[1].querySelector('.says');
  return {
    id: tr.dataset.level,
    count: tr.dataset.count,
    name: td[0].textContent.trim(),
    how: [...td[1].childNodes].filter((n) => n.nodeType === 3).map((n) => n.textContent).join('').trim(),
    says: says ? says.textContent.trim() : null,
    ask: td[2].textContent.trim(),
  };
}));
const SAYS = { book: 'the book states it', reported: 'your device reported it', 'vendor-doc': "AMD's documentation, CDNA family", none: null };
// What the table must show for a device, row by row, from the model.
const expected = (d) => levels(d).map((l) => ({
  id: l.id,
  count: l.count == null ? '' : String(l.count),
  name: l.name ?? `no counterpart of the ${l.bookName} documented`,
  how: l.count != null ? (l.id === 'device' ? '1' : `${fmt(l.count, 6)} ${l.per}`) : l.id === 'onchip' || l.id === 'global' ? l.per : l.name == null ? 'not documented' : 'not reported',
  says: l.count != null ? SAYS[l.provenance] : null,
  ask: l.ask ?? 'no query',
}));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const askable = (d) => levels(d).filter((l) => ['cluster', 'unit', 'group', 'core'].includes(l.id) && l.ask != null).length;
const shot = async (name) => {
  if (!out) return;
  await page.$eval(B, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};

// The devices the page can offer: the book's example, and one per GPU report of this machine.
const reports = await (await fetch(BASE + '/api/reports')).json();
const mine = Object.keys(reports).filter(isGpuBackend).map((b) => deviceFromReport(b, reports[b].device));
const devicesFor = (v) => {
  const own = mine.filter((d) => d.vocabulary === v);
  return v === 'cuda' ? [BOOK_DEVICE, ...own] : own.length ? own : [noReport(v)];
};
const optionLabel = (d) => (d.source === 'book' ? "the book's example" : d.source === 'report' ? `your device: ${d.name}` : `no ${TERMS[d.vocabulary].api} device report on this machine`);

await goto(page, 'ch04/4.1');
await page.waitForSelector(B);

// ---------------------------------------------------------------- the lesson's numbers
const prose = clean(await page.$eval('.prose', (e) => e.textContent));
check("the book's example is stated from the model: 132 SMs of 128, 16,896 in all, 8 GPCs", prose.includes(`${BOOK_DEVICE.units} SMs of ${BOOK_DEVICE.coresPerUnit} streaming processors each, ${fmt(totalCores(BOOK_DEVICE), 6)} in all`) && prose.includes(`(GPCs), ${BOOK_DEVICE.clusters} on this device`) && totalCores(BOOK_DEVICE) === 16896);
const todays = (await page.$$eval('.note.today', (es) => es.map((e) => e.textContent))).map(clean);
const today = todays.find((t) => t.includes('TPC')) ?? '';
check('the Today note on the example carries the other variant, the TPC level and what is unverified', today.includes(`the PCIe variant has ${BOOK_DEVICE_TODAY.unitsOtherVariant}`) && today.includes(`${BOOK_DEVICE_TODAY.tpcs} TPCs make the 132`) && today.includes('(unverified)') && today.includes('A1, A2'), today);
check('the memory paragraph has its own Today note: HBM3 on the SXM5 only', todays.some((t) => t.includes('HBM3 is on the SXM5 variant only') && t.includes('A4')), JSON.stringify(todays));
// Margin notes must not open gaps in the text column: none is taller than the paragraph it follows by more than a few lines.
const tall = await page.evaluate(() => [...document.querySelectorAll('.prose > .note')].map((n) => {
  let p = n.previousElementSibling;
  while (p && p.classList.contains('note')) p = p.previousElementSibling;
  return { note: n.textContent.slice(0, 50), over: Math.round(n.getBoundingClientRect().height - (p ? p.getBoundingClientRect().height : 0)) };
}).filter((x) => x.over > 60));
check('at 1440 px no margin note is much taller than its paragraph', tall.length === 0, JSON.stringify(tall));
check('the explorer states what it simplifies', clean(await page.$eval('.note.caveat', (e) => e.textContent)).includes('not a model of hardware'));

// ---------------------------------------------------------------- prediction
let q = await question('How many threads do you expect one SM can have in progress at once');
check('before committing, the reveal is hidden', !(await qText(q)).includes('for every core'));
await (await q.$('input[type=number]')).type(String(BOOK_DEVICE.coresPerUnit));
await (await q.$('button.primary')).click();
await pause(150);
let t = await qText(q);
check('prediction reveals 2,048 threads per SM, 16 for every core, from p. 85', t.includes(`${fmt(BOOK_DEVICE.threadsPerUnit, 6)} threads per SM, ${threadsPerCore(BOOK_DEVICE)} for every core (p. 85)`) && threadsPerCore(BOOK_DEVICE) === 16, t);
check('it answers the one-per-core guess and gives the device totals', t.includes('One thread per core is the natural guess') && t.includes(`${fmt(residentThreads(BOOK_DEVICE), 6)} threads in progress on ${fmt(totalCores(BOOK_DEVICE), 6)} cores`), t);
check('and points to section 4.6 for the reason', await q.evaluate((e) => [...e.querySelectorAll('a')].some((a) => a.getAttribute('href') === '#/ch04/4.6')));

// ---------------------------------------------------------------- explorer: CUDA, the book's example
let r = await rows();
check("CUDA, the book's example: every row is the model's (names, counts, where they come from, queries)", same(r, expected(BOOK_DEVICE)), JSON.stringify(r));
check('132 SMs, 8 GPCs, 4 processing blocks and 128 streaming processors are shown', r.find((x) => x.id === 'unit').how === '132 on the device' && r.find((x) => x.id === 'cluster').how === '8 on the device' && r.find((x) => x.id === 'group').how === '4 in each SM' && r.find((x) => x.id === 'core').how === '128 in each SM', JSON.stringify(r.map((x) => x.how)));
check('only the SM count has a query: 1 of the 4 hardware levels', (await text(B + ' .summary')).includes('a CUDA program can ask for the size of 1.') && askable(BOOK_DEVICE) === 1, await text(B + ' .summary'));
const options = async () => page.$$eval(B + ' select[aria-label="Device"] option', (o) => o.map((x) => x.textContent));
check('the device list is the book\'s example plus this machine\'s CUDA reports', same(await options(), devicesFor('cuda').map(optionLabel)), (await options()).join(' | '));
const placed = levels(BOOK_DEVICE).find((l) => l.id === 'unit').holds.map((h) => `${h.what} (${h.where.startsWith('4.') ? 'section ' + h.where : h.where})`);
check('the SM is selected at first, with everything the model says later sections place there', (await page.$eval(B + ' .panel[data-selected]', (e) => e.dataset.selected)) === 'unit'
  && placed.length === 4 && placed[0] === 'whole blocks, a few at a time (section 4.2)' && same(await page.$$eval(B + ' .panel[data-selected] li', (li) => li.map((x) => x.textContent.replace(/\s+/g, ' ').trim())), placed), await text(B + ' .panel[data-selected]'));
const picture = clean(await page.$eval(B + ' svg', (e) => e.textContent));
check('the redrawn Fig. 4.1 labels the device, its clusters, the controllers and global memory', picture.startsWith('GPU') && picture.includes('GPC') && picture.includes('controller') && picture.includes('global memory') && !picture.includes('nothing inside'), picture);
await shot('cuda-book');

// Select levels: from the table, and from the picture.
await clickText(B + ' table.levels', 'streaming processor');
let panel = await text(B + ' .panel[data-selected]');
check('selecting the streaming processor says no property reports its count', (await page.$eval(B + ' .panel[data-selected]', (e) => e.dataset.selected)) === 'core' && panel.includes('No property of a CUDA device reports how many an SM has'), panel);
const g = await page.evaluateHandle((B) => [...document.querySelectorAll(B + ' svg text')].find((x) => x.textContent === 'global memory'), B);
await g.click();
await pause(150);
panel = await text(B + ' .panel[data-selected]');
check('clicking global memory in the picture selects it and points at section 4.6', (await page.$eval(B + ' .panel[data-selected]', (e) => e.dataset.selected)) === 'global' && panel.includes("the arrays behind a kernel's pointer arguments (section 4.6)"), panel);
check('the selected row is marked in the table', (await page.$eval(B + ' table.levels button[aria-pressed="true"]', (e) => e.textContent.trim())) === 'global memory');
const tile = await page.$(B + ' svg rect[width="8"]');
await tile.click();
await pause(150);
check('clicking a core tile in the picture selects the streaming processor', (await page.$eval(B + ' .panel[data-selected]', (e) => e.dataset.selected)) === 'core');

// A CUDA report of this machine, if there is one.
const cudaMine = devicesFor('cuda').slice(1);
if (cudaMine.length) {
  await setSelect('Device', 1);
  r = await rows();
  check('your CUDA device: the SM count is reported, the inside of an SM is not', same(r, expected(cudaMine[0])) && r.find((x) => x.id === 'core').how === 'not reported', JSON.stringify(r));
  await setSelect('Device', 0);
} else {
  check('no CUDA report on this machine: only the book\'s example is offered under CUDA', (await options()).length === 1);
}

// ---------------------------------------------------------------- explorer: HIP
await clickText(B + ' .seg', 'HIP on AMD');
const hip = devicesFor('hip')[0];
r = await rows();
check('HIP: AMD\'s names and the model\'s rows', same(r, expected(hip)) && r.find((x) => x.id === 'unit').name === 'compute unit (CU)' && r.find((x) => x.id === 'cluster').name === 'shader engine and shader array', JSON.stringify(r));
check('HIP: the inside of a CU comes from AMD\'s documentation, not from a device', r.find((x) => x.id === 'group').how === '4 in each CU' && r.find((x) => x.id === 'core').how === '64 in each CU' && r.find((x) => x.id === 'group').says === SAYS['vendor-doc'] && r.find((x) => x.id === 'core').ask === 'no query', JSON.stringify(r));
check('HIP: the device list is this machine\'s HIP reports, or says there is none', same(await options(), devicesFor('hip').map(optionLabel)), (await options()).join(' | '));
if (hip.source === 'none') check('HIP with no report: the CU count is "not reported", never a number', r.find((x) => x.id === 'unit').how === 'not reported' && r.find((x) => x.id === 'unit').count === '', JSON.stringify(r.find((x) => x.id === 'unit')));
else check('HIP with a report: the CU count is the reported multiProcessorCount', r.find((x) => x.id === 'unit').count === String(hip.units ?? ''), JSON.stringify(r.find((x) => x.id === 'unit')));
check('HIP: still 1 of the 4 hardware levels can be asked for', (await text(B + ' .summary')).includes('a HIP on AMD program can ask for the size of 1.') && askable(hip) === 1, await text(B + ' .summary'));
await clickText(B + ' table.levels', 'compute unit (CU)');
check('HIP: the CU row warns that the field may count half the compute units', (await text(B + ' .panel[data-selected]')).includes('half of them when the device runs in workgroup-processor mode'), await text(B + ' .panel[data-selected]'));
await shot('hip');

// ---------------------------------------------------------------- explorer: Metal
await clickText(B + ' .seg', 'Metal');
const metal = devicesFor('metal')[0];
r = await rows();
check('Metal: the model\'s rows', same(r, expected(metal)), JSON.stringify(r));
const under = r.filter((x) => ['cluster', 'unit', 'group', 'core'].includes(x.id));
check('Metal: no level under the device has a name, a count or a query, and each row says which level is missing', under.every((x) => x.how === 'not documented' && x.ask === 'no query' && x.count === '')
  && under.map((x) => x.name).join('|') === 'no counterpart of the GPC documented|no counterpart of the SM documented|no counterpart of the processing block documented|no counterpart of the streaming processor documented', JSON.stringify(under));
check('Metal: 0 of the 4 hardware levels can be asked for', (await text(B + ' .summary')).includes('a Metal program can ask for the size of 0.') && askable(metal) === 0, await text(B + ' .summary'));
check('Metal: the picture says nothing inside the device is documented', clean(await page.$eval(B + ' svg', (e) => e.textContent)).includes('nothing inside the device is documented'));
check('Metal: the device list is this machine\'s Metal report, or says there is none', same(await options(), devicesFor('metal').map(optionLabel)), (await options()).join(' | '));
check('Metal: no SM count, core count or thread limit is invented for the device', !/\d+ (on the device|in each)/.test(r.map((x) => x.how).join(' ')) && !(await text(B + ' .summary')).includes('threads in progress per'), r.map((x) => x.how).join(' | '));
await clickText(B + ' table.levels', 'no counterpart of the SM documented');
check('Metal: a level without a counterpart says what Apple does document', (await text(B + ' .panel[data-selected]')).includes('Apple documents threadgroups and SIMD-groups'), await text(B + ' .panel[data-selected]'));
await shot('metal');
await clickText(B + ' .seg', 'CUDA');
check('switching back to CUDA restores the book\'s example', same(await rows(), expected(BOOK_DEVICE)));

// ---------------------------------------------------------------- other backends and the bridge
check('the AMD paragraph is the documented description, and the mapping is labelled ours', prose.includes('four SIMD processors of 16 lanes each') && prose.includes('Setting SM beside CU, and processing block beside SIMD, is our comparison'));
check('the Python bridge is labelled unverified on CUDA', prose.includes('torch.cuda.get_device_properties(0).multi_processor_count') && prose.includes('Not run on a CUDA machine by this course: unverified.'));

// ---------------------------------------------------------------- questions
q = await question('at most 16,896 threads of my kernel are in progress');
await pick(q, 'No correction: one thread runs per core');
check('cores question: "one thread per core" is marked wrong with its reason', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('sixteen times more threads than it has cores'), await qText(q));
await pick(q, 'Too low: cores bound what executes in one cycle');
t = await qText(q);
check('cores question: the right answer is accepted, with both products from the model', t.includes('Correct.') && t.includes('132 × 128 = 16,896') && t.includes(`132 × 2048 = ${fmt(residentThreads(BOOK_DEVICE), 6)}`) && residentThreads(BOOK_DEVICE) === 270336, t);
q = await question('where does the data behind');
await pick(q, 'In host memory');
check('global memory question: host memory is marked wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('The kernel never touches host memory'), await qText(q));
await pick(q, 'In global memory, off the chip');
check('global memory question: global memory through the controllers is accepted', (await qText(q)).includes('Correct.') && (await qText(q)).includes('Source: ch. 4 §4.1, p. 68 and §4.6, p. 83'), await qText(q));
q = await question('What do the two numbers count');
await pick(q, 'SMs on both');
check('multiProcessorCount question: "SMs on both" is marked wrong', (await qText(q)).includes('Not quite.') && (await qText(q)).includes('It fills it with AMD\'s own unit'), await qText(q));
await pick(q, 'SMs on one; compute units, or half of them, on the other');
check('multiProcessorCount question: SMs against compute units is accepted', (await qText(q)).includes('Correct.') && (await qText(q)).includes('The name carries over; the unit does not.'), await qText(q));

// ---------------------------------------------------------------- narrow screen
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((B) => ({ page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1, bench: document.querySelector(B).getBoundingClientRect().width <= document.documentElement.clientWidth }), B);
check('at a narrow width nothing makes the page scroll sideways', narrow.page && narrow.bench, JSON.stringify(narrow));
await shot('narrow');

await pause(600);
check('answers were saved through PUT /api/progress, which this check intercepts', puts > 0, String(puts));
check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
