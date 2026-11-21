// Browser check for section C.8 (Page fault handling): the "Coherence by invalidation"
// explorer (every preset, the reader's prices, a script built by hand with a missing
// synchronise) and the "When the model does not fit" explorer (the cliff, both
// placements, the reader's device figures), four questions, light and dark themes and a
// narrow screen. Every number is compared with the model (src/sim/managedmem.ts).
// Usage: APP_URL=http://localhost:5702 node e2e/appC-sc8.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';
import { EXPLORE_POLICIES, PRESETS, cost, explore, moved, warmTraffic } from '../src/sim/managedmem.ts';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1100 });
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
const C = 'section.bench[aria-label="Coherence by invalidation"]';
const O = 'section.bench[aria-label="When the model does not fit"]';
const press = async (root, label) => {
  const ok = await page.evaluate((root, label) => {
    const btn = [...document.querySelectorAll(root + ' button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, root, label);
  await pause(60);
  return ok;
};
const setValue = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    const proto = el.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event(el.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true }));
  }, v);
  await pause(100);
};
const preset = async (id) => {
  await page.select(C + ' select[aria-label="Preset"]', id);
  await pause(100);
};
const panels = () =>
  page.evaluate((C) => Object.fromEntries([...document.querySelectorAll(C + ' [data-policy]')].map((p) => [p.dataset.policy, {
    totals: Object.fromEntries([...p.querySelectorAll('[data-total]')].map((t) => [t.dataset.total, Number(t.textContent)])),
    best: !!p.querySelector('[data-best]'),
    crashed: p.textContent.includes('segmentation fault'),
    errors: [...p.querySelectorAll('[data-error]')].map((e) => e.dataset.error),
    summary: p.querySelector('[data-step-summary]').textContent.trim(),
    cpu: [...p.querySelectorAll('[data-side="cpu"]')].map((g) => g.dataset.present),
    gpu: [...p.querySelectorAll('[data-side="gpu"]')].map((g) => g.dataset.present),
  }])), C);
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const presetScript = (id) => PRESETS.find((p) => p.id === id).script;
/** Do the panels show the model's totals and costs after `at` steps of `script`? */
const matches = (ps, script, at, m = 3, r = 1) => {
  const costs = EXPLORE_POLICIES.map((p) => cost(explore(p, script).steps[at].counters, m, r));
  const best = Math.min(...costs);
  const bad = EXPLORE_POLICIES.filter((p, k) => {
    const c = explore(p, script).steps[at].counters;
    const t = ps[p].totals;
    return !(t.h2d === c.h2d && t.d2h === c.d2h && t.gpuFaults === c.gpuFaults && t.cpuFaults === c.cpuFaults && t.remote === c.remote
      && t.invalidations === c.invalidations && Math.abs(t.cost - costs[k]) < 1e-9 && ps[p].best === (costs[k] === best && !explore(p, script).steps[at].crashed));
  });
  return { ok: bad.length === 0, bad, costs };
};

await goto(page, 'appC/C.8');
await page.waitForSelector(C);
await page.waitForSelector(O);
check('every margin note (or note group) directly follows text, never another note', await page.evaluate(() => [...document.querySelectorAll('.prose > .note')].every((n) => ['P', 'UL', 'OL'].includes(n.previousElementSibling?.tagName))));

// ---- the numeric question before the explorer
const q1 = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Sixty-four managed pages')));
await (await q1.$('input[type=number]')).type(String(explore('bulk', [{ kind: 'cpu-write', from: 0, to: 63 }, { kind: 'kernel-read', from: 0, to: 3 }], 64).counters.h2d - 4));
await (await q1.$('button.primary')).click();
await pause(120);
check('q-pages-moved: 60 is accepted', (await q1.evaluate((e) => e.textContent)).includes('Correct.'));

// ---- Coherence by invalidation: Fig. C.1, step by step
await page.$eval(C, (e) => e.scrollIntoView({ block: 'start' }));
let ps = await panels();
check('four policies side by side: bulk, migrate, map, read-mostly', same(Object.keys(ps), EXPLORE_POLICIES), Object.keys(ps).join(','));
check('before any step: sixteen pages per memory row, none present', EXPLORE_POLICIES.every((p) => ps[p].cpu.length === 16 && ps[p].gpu.length === 16 && ps[p].cpu.every((x) => x === 'no')));
const fig = presetScript('figC1');
let allSteps = true;
for (let k = 1; k <= fig.length; k++) {
  await press(C, 'Step');
  ps = await panels();
  const m = matches(ps, fig, k);
  if (!m.ok) { allSteps = false; check(`Fig. C.1 step ${k}`, false, JSON.stringify({ bad: m.bad, ps })); }
  if (k === 2) {
    check('Fig. C.1, after the kernel: fault and migrate holds every page on the GPU only, with 16 GPU faults',
      ps.migrate.gpu.every((x) => x === 'yes') && ps.migrate.cpu.every((x) => x === 'no') && ps.migrate.summary === 'this step: 16 to GPU, 16 GPU faults', JSON.stringify(ps.migrate));
    check('Fig. C.1, after the kernel: bulk moved all 16 at the launch, with no GPU fault', ps.bulk.summary === 'this step: 16 to GPU', ps.bulk.summary);
    check('Fig. C.1, after the kernel: map on access kept every page in CPU memory, mapped for the GPU',
      ps.map.cpu.every((x) => x === 'yes') && ps.map.gpu.every((x) => x === 'mapped') && ps.map.summary === 'this step: 16 remote', JSON.stringify(ps.map));
    check('Fig. C.1, after the kernel: read-mostly wrote, so it invalidated the CPU copies', ps.replicate.summary.includes('16 invalidations'), ps.replicate.summary);
    if (out) await page.screenshot({ path: out + '-coherence-fig.png' });
  }
}
check('Fig. C.1: every step of every policy matches the model (totals, costs, cheapest)', allSteps);

// ---- presets
await preset('one-page');
await press(C, 'Run to the end');
ps = await panels();
let m = matches(ps, presetScript('one-page'), presetScript('one-page').length);
check('one page of 16: totals match the model', m.ok, JSON.stringify(m.bad));
check('one page of 16: bulk moves all 16, fault and migrate 1, map 1 remote access, read-mostly 1 replica',
  ps.bulk.totals.h2d === 16 && ps.migrate.totals.h2d === 1 && ps.map.totals.remote === 1 && ps.map.totals.h2d === 0 && ps.replicate.totals.h2d === 1, JSON.stringify(ps));
check('read-mostly holds the read page in both memories', ps.replicate.cpu[5] === 'yes' && ps.replicate.gpu[5] === 'yes' && ps.replicate.gpu.filter((x) => x === 'yes').length === 1);

await preset('ping-pong');
await press(C, 'Run to the end');
ps = await panels();
m = matches(ps, presetScript('ping-pong'), presetScript('ping-pong').length);
check('ping-pong: totals match the model, and mapping is the cheapest at the default prices', m.ok && ps.map.best && !ps.migrate.best, JSON.stringify({ bad: m.bad, costs: m.costs }));
check('ping-pong: fault and migrate moves the counter page 8 times, mapping never', moved({ h2d: ps.migrate.totals.h2d, d2h: ps.migrate.totals.d2h }) === 8 && ps.map.totals.h2d + ps.map.totals.d2h === 0);
if (out) await page.screenshot({ path: out + '-coherence-pingpong.png' });
// The reader's prices: make moving almost free and remote access dear.
await setValue(C + ' input[aria-label="cost per page moved"]', 0.1);
await setValue(C + ' input[aria-label="cost per remote page access"]', 5);
ps = await panels();
m = matches(ps, presetScript('ping-pong'), presetScript('ping-pong').length, 0.1, 5);
check('with cheap moves and dear remote accesses the verdict flips away from mapping', m.ok && !ps.map.best && ps.migrate.best, JSON.stringify({ bad: m.bad, costs: m.costs }));
await setValue(C + ' input[aria-label="cost per page moved"]', 3);
await setValue(C + ' input[aria-label="cost per remote page access"]', 1);

await preset('decode');
await press(C, 'Run to the end');
ps = await panels();
m = matches(ps, presetScript('decode'), presetScript('decode').length);
check('decode loop: totals match the model; read-mostly is the cheapest at the default prices, mapping the dearest',
  m.ok && ps.replicate.best && Math.max(...m.costs) === ps.map.totals.cost, JSON.stringify({ bad: m.bad, costs: m.costs }));
check('decode loop: costs are 81, 78, 90 and 63', same(EXPLORE_POLICIES.map((p) => ps[p].totals.cost), [81, 78, 90, 63]), JSON.stringify(EXPLORE_POLICIES.map((p) => ps[p].totals.cost)));

// ---- a script built by hand, with the synchronise forgotten
await press(C, 'Clear');
await setValue(C + ' select[aria-label="First page"]', 0);
await setValue(C + ' select[aria-label="Last page"]', 3);
await press(C, '+ CPU writes');
await setValue(C + ' select[aria-label="Last page"]', 0);
await press(C, '+ kernel writes');
await press(C, '+ CPU reads');
const own = [{ kind: 'cpu-write', from: 0, to: 3 }, { kind: 'kernel-write', from: 0, to: 0 }, { kind: 'cpu-read', from: 0, to: 0 }];
const scriptShown = await page.$$eval(C + ' [data-op]', (bs) => bs.map((b) => b.textContent.trim()));
check('the script shows what was added', same(scriptShown, ['1. CPU writes pages 0–3', '2. kernel writes page 0', '3. CPU reads page 0']), scriptShown.join(' | '));
check('the preset selector says it is your script', (await page.$eval(C + ' select[aria-label="Preset"]', (e) => e.value)) === 'own');
await press(C, 'Run to the end');
ps = await panels();
check('a CPU read while the kernel may run: a segmentation fault when limited, a race everywhere else',
  ps.bulk.crashed && same(ps.bulk.errors, ['segfault']) && ['migrate', 'map', 'replicate'].every((p) => same(ps[p].errors, ['race']))
  && same(explore('bulk', own).errors.map((e) => e.kind), ['segfault']), JSON.stringify(EXPLORE_POLICIES.map((p) => ps[p].errors)));
m = matches(ps, own, own.length);
check('the hand-built script\'s totals match the model', m.ok, JSON.stringify(m.bad));
if (out) await page.screenshot({ path: out + '-coherence-own.png' });
await press(C, 'Back');
await press(C, 'Back');
ps = await panels();
check('Back steps the script backwards', ps.migrate.totals.h2d === 0 && ps.migrate.cpu.slice(0, 4).every((x) => x === 'yes'), JSON.stringify(ps.migrate));

// ---- When the model does not fit
await page.$eval(O, (e) => e.scrollIntoView({ block: 'start' }));
const readout = () => page.$eval(O + ' [data-readout]', (r) => ({
  weights: Number(r.dataset.weights),
  crossing: Number(r.dataset.crossing),
  policy: r.dataset.policy,
  text: r.textContent.replace(/\s+/g, ' ').trim(),
  devices: [...r.querySelectorAll('[data-device]')].map((d) => d.dataset.device),
}));
const placement = async (word) => {
  await page.evaluate((O, word) => [...document.querySelectorAll(O + ' .seg button')].find((b) => b.textContent.includes(word)).click(), O, word);
  await pause(100);
};
let r = await readout();
check('starts at 80%: 80 weight pages, nothing crossing once warm', r.weights === 80 && r.crossing === 0 && r.text.includes('Nothing: it all fits'), JSON.stringify(r));
check('the plot has both curves over 151 sizes', (await page.$$eval(O + ' polyline', (ls) => ls.map((l) => l.getAttribute('points').split(' ').length))).every((n) => n === 151));
await setValue(O + ' input[type=range]', 1.05);
r = await readout();
const lru105 = warmTraffic(105, 100, 'lru');
check('105% with LRU: every page migrates every token (105 in, 105 out)', r.weights === 105 && r.crossing === lru105.h2d && lru105.h2d === 105 && r.text.includes('105 pages migrated in and 105 evicted out') && r.text.includes('Every page, every token.'), JSON.stringify(r));
if (out) await page.screenshot({ path: out + '-cliff-lru.png' });
await placement('prefix');
r = await readout();
const pre105 = warmTraffic(105, 100, 'prefix');
check('105% with a resident prefix: only the excess crosses, 5 pages, about 5% of the weights', r.policy === 'prefix' && r.crossing === pre105.remote && pre105.remote === 5 && r.text.includes('5 pages read over the link') && r.text.includes('5% of the weights'), JSON.stringify(r));
if (out) await page.screenshot({ path: out + '-cliff-prefix.png' });
await setValue(O + ' input[type=range]', 2);
r = await readout();
check('200% with a resident prefix: 100 pages over the link per token', r.crossing === warmTraffic(200, 100, 'prefix').remote && r.crossing === 100, JSON.stringify(r));
await placement('least-recently-used');
r = await readout();
check('200% with LRU: 200 pages per token', r.crossing === 200, JSON.stringify(r));
await setValue(O + ' input[type=range]', 1);
r = await readout();
check('exactly 100%: nothing moves once warm', r.weights === 100 && r.crossing === 0, JSON.stringify(r));
// The reader's own device figures, from the reports.
const reports = await (await fetch(BASE + '/api/reports')).json();
const gpus = Object.entries(reports).filter(([b, x]) => b !== 'emu' && b !== 'race' && Number(x.device?.totalGlobalMem) > 0);
const metal = Number(reports.metal?.device?.recommendedMaxWorkingSetSize);
check('device memory comes from the reports: a CUDA or HIP device when one is recorded, else no gigabytes',
  gpus.length ? r.devices.includes(gpus[0][0]) : r.devices.includes('none'), JSON.stringify(r.devices));
if (metal > 0) {
  const gbText = (metal / 1e9).toLocaleString('en-US', { maximumFractionDigits: 1 });
  check('the Mac\'s recommendedMaxWorkingSetSize is the reader\'s own figure, labelled as advice', r.devices.includes('metal') && r.text.includes(`recommendedMaxWorkingSetSize of ${gbText} GB`) && r.text.includes('does not say what happens above'), r.text);
}

// ---- questions
const answer = async (needle, option) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await q.evaluate((e, o) => [...e.querySelectorAll('label')].find((l) => l.textContent.startsWith(o)).querySelector('input').click(), option);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
check('q-cliff: "All of them" is right', (await answer('Weights are 105% of device memory', 'All of them')).includes('Correct.'));
const sig = await answer('How do you write the prefetch?', 'Leave it to the rename');
check('q-signature: relying on the rename is wrong, and the int device is named', sig.includes('Not quite.') && sig.includes('takes an int device'), sig.slice(0, 200));
check('q-signature: branching with #if is right', (await answer('How do you write the prefetch?', 'Branch with #if')).includes('Correct.'));
const hints = await answer('Delete that one call', 'The hint kept the pages coherent');
check('q-hints: a wrong answer is marked and explained with NVIDIA\'s words', hints.includes('Not quite.') && hints.includes('Performance hints never impact the correctness'), hints.slice(0, 200));
check('q-hints: the missing synchronise is right', (await answer('Delete that one call', 'A missing synchronise')).includes('Correct.'));
check('q-pingpong: mapping is right', (await answer('take turns updating the same small counter page', 'Keep it in one memory')).includes('Correct.'));

// ---- themes and a narrow screen
await page.$eval(C, (e) => e.scrollIntoView({ block: 'start' }));
await preset('figC1');
await press(C, 'Step');
await press(C, 'Step');
if (out) await page.screenshot({ path: out + '-dark.png' });
await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
await pause(150);
if (out) await page.screenshot({ path: out + '-light.png' });
await page.$eval(O, (e) => e.scrollIntoView({ block: 'start' }));
if (out) await page.screenshot({ path: out + '-cliff-light.png' });
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const narrow = await page.evaluate((C, O) => ({
  page: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1,
  c: document.querySelector(C).getBoundingClientRect().width <= document.documentElement.clientWidth,
  o: document.querySelector(O).getBoundingClientRect().width <= document.documentElement.clientWidth,
}), C, O);
check('at a narrow width nothing makes the page scroll sideways', narrow.page && narrow.c && narrow.o, JSON.stringify(narrow));
if (out) {
  await page.$eval(C, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-coherence.png' });
  await page.$eval(O, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow-cliff.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
