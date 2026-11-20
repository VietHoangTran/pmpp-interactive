// Browser check for section X1.1, "The sampler chain": the "Reorder the chain" explorer
// (every order preset, both rows, every slider, float32, the stage chips with their arrows
// and drag-and-drop, a custom order, pinning, the prediction's case), the p-order Predict
// and the section's questions. Every kept set and drawn token is compared with
// src/sim/chain.ts, run inside the page through the app's module server.
// Usage: APP_URL=http://localhost:5173 node e2e/x1-sx11.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';

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
const B = 'section.bench[aria-label="Reorder the chain"]';

const clickIn = async (group, label) => {
  const ok = await page.evaluate((B, group, label) => {
    const g = document.querySelector(`${B} [aria-label="${group}"]`);
    const b = g && [...g.querySelectorAll('button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, B, group, label);
  await pause(80);
  return ok;
};
const clickButton = async (label) => {
  const ok = await page.evaluate((B, label) => {
    const b = [...document.querySelectorAll(`${B} button`)].find((x) => x.textContent.trim() === label || x.getAttribute('aria-label') === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, B, label);
  await pause(80);
  return ok;
};
const setRange = async (aria, value) => {
  await page.$eval(`${B} input[type=range][aria-label="${aria}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(80);
};
const panels = (part = 'panels') =>
  page.$$eval(`${B} [data-part="${part}"] [data-stage]`, (ps) => ps.map((p) => ({ stage: p.dataset.stage, ids: p.dataset.ids })));
const result = () => page.$eval(`${B} [data-readout="result"]`, (e) => ({ final: e.dataset.final, token: Number(e.dataset.token), text: e.textContent.replace(/\s+/g, ' ') }));

/** What the model says for this row, settings and order (an order preset id, or an explicit stage list with M5 rules). */
const model = (rowId, settings, order) =>
  page.evaluate(async (rowId, settings, order) => {
    const m = await import('/src/sim/chain.ts');
    const preset = typeof order === 'string' ? m.ORDERS[order] : null;
    const r = m.runChain(m.chainRow(rowId), settings, preset ? preset.order : order, preset ? m.optionsOf(preset) : m.M5_OPTIONS);
    return {
      panels: r.snapshots.map((s) => ({ stage: s.stage, ids: s.alive.flatMap((a, i) => (a ? [i] : [])).join(' ') })),
      final: m.finalSet(r).join(' '),
      token: r.token,
    };
  }, rowId, settings, order);
const D = await (async () => {
  await goto(page, 'x1/X1.1');
  await page.waitForSelector(B);
  return page.evaluate(async () => {
    const m = await import('/src/sim/chain.ts');
    return { def: m.DEFAULT_SETTINGS, porder: m.P_ORDER, counts: m.pOrderCounts(), llama: m.ORDERS.llama.order, hf: m.ORDERS.hf.order };
  });
})();
const same = async (name, rowId, settings, order, part = 'panels') => {
  const want = await model(rowId, settings, order);
  const got = await panels(part);
  check(name + ': every stage keeps what the model keeps', JSON.stringify(got) === JSON.stringify(want.panels), JSON.stringify(got) + ' vs ' + JSON.stringify(want.panels));
  if (part === 'panels') {
    const r = await result();
    check(name + ': the draw set and token match the model', r.final === want.final && r.token === want.token, `${r.final}/${r.token} vs ${want.final}/${want.token}`);
  }
  return want;
};
const tFromSlider = (v) => Number((0.1 * Math.pow(40, v / 1000)).toPrecision(2));
const sliderFromT = (T) => Math.round((Math.log(T / 0.1) / Math.log(40)) * 1000);

// ---------------------------------------------------------------- defaults: llama.cpp order, peaked row
const chips = await page.$$eval(`${B} [data-part="chips"] [data-chip]`, (cs) => cs.map((c) => [c.dataset.chip, c.dataset.modelled]));
check('llama.cpp preset shows its nine stages and the draw', JSON.stringify(chips.map((c) => c[0])) === JSON.stringify([...D.llama, 'draw']), JSON.stringify(chips));
check('DRY, top-n-sigma, typical and XTC are marked not modelled',
  JSON.stringify(chips.filter((c) => c[1] === '0').map((c) => c[0])) === JSON.stringify(['dry', 'top-n-sigma', 'typical', 'xtc']));
check('not-modelled chips have no move buttons', (await page.$$eval(`${B} [data-chip="dry"] button`, (b) => b.length)) === 0);
let s = { ...D.def };
await same('llama.cpp default, peaked row', 'peaked', s, 'llama');
if (out) await page.screenshot({ path: out + '-default.png' });

// ---------------------------------------------------------------- the sliders, one at a time
await setRange('k', 5); s.k = 5;
await same('k = 5', 'peaked', s, 'llama');
await setRange('p', 70); s.p = 0.7;
await same('p = 0.70', 'peaked', s, 'llama');
await setRange('min-p', 30); s.minP = 0.3;
await same('min-p = 0.30', 'peaked', s, 'llama');
check('the min-p panel states its logit threshold', (await page.$(`${B} [data-part="panels"] [data-stage="min-p"] [data-threshold]`)) !== null);
await setRange('repetition', 160); s.repeat = 1.6;
await setRange('frequency', 30); s.freq = 0.3;
await setRange('presence', 40); s.presence = 0.4;
await same('penalties 1.6 / 0.3 / 0.4', 'peaked', s, 'llama');
await setRange('u', 950); s.u = 0.95;
await same('u = 0.950', 'peaked', s, 'llama');
await page.evaluate((B) => [...document.querySelectorAll(`${B} label.check`)].find((l) => l.textContent.includes('float32')).querySelector('input').click(), B);
await pause(100);
s.float32 = true;
await same('float32 on', 'peaked', s, 'llama');

// T moves: under llama.cpp's order nothing but the draw weights change; top-k's set never does
const topkAt = async (T) => {
  await setRange('T', sliderFromT(T));
  return (await panels()).find((p) => p.stage === 'top-k').ids;
};
s = { ...D.def };
await clickIn('Row', 'three favourites, long tail');
for (const [aria, v] of [['k', s.k], ['p', s.p * 100], ['min-p', s.minP * 100], ['repetition', s.repeat * 100], ['frequency', 0], ['presence', 0], ['u', s.u * 1000]]) await setRange(aria, Math.round(v));
await page.evaluate((B) => [...document.querySelectorAll(`${B} label.check`)].find((l) => l.textContent.includes('float32')).querySelector('input').click(), B);
await pause(80);
const kA = await topkAt(0.2);
const kB = await topkAt(3);
check("llama.cpp order: top-k's set is the same at T = 0.2 and T = 3", kA === kB, kA + ' | ' + kB);
s.T = tFromSlider(sliderFromT(3));
await same('tail row, T = 3', 'tail', s, 'llama');

// ---------------------------------------------------------------- transformers: temperature first
check('transformers preset button', await clickIn('Order preset', 'transformers'));
const hfChips = await page.$$eval(`${B} [data-part="chips"] [data-chip]`, (cs) => cs.map((c) => c.dataset.chip));
check("transformers' chips put temperature before the truncations", JSON.stringify(hfChips) === JSON.stringify([...D.hf, 'draw']), JSON.stringify(hfChips));
check('transformers disables frequency and presence, and says why',
  (await page.$eval(`${B} input[aria-label="frequency"]`, (e) => e.disabled)) && (await page.$(`${B} [data-readout="no-freq"]`)) !== null);
await same('transformers, tail row, T = 3', 'tail', s, 'hf');
const pAt = async (T) => {
  await setRange('T', sliderFromT(T));
  const ps = await panels();
  return { k: ps.find((p) => p.stage === 'top-k').ids, p: ps.find((p) => p.stage === 'top-p').ids };
};
const lo = await pAt(0.3);
const hi = await pAt(3);
check("transformers: top-k's set still does not move with T", lo.k === hi.k, lo.k + ' | ' + hi.k);
check("transformers: top-p keeps more tokens at T = 3 than at T = 0.3", hi.p.split(' ').length > lo.p.split(' ').length, lo.p + ' | ' + hi.p);
s.T = tFromSlider(sliderFromT(3));
await same('transformers at T = 3 again', 'tail', s, 'hf');

// ---------------------------------------------------------------- mlx-lm and M5
check('mlx-lm preset button', await clickIn('Order preset', 'mlx-lm'));
await same('mlx-lm, tail row', 'tail', s, 'mlx');
check('M5 preset button', await clickIn('Order preset', 'M5 (this course)'));
await same('M5, tail row', 'tail', s, 'm5');
const m5chips = await page.$$eval(`${B} [data-part="chips"] [data-chip]`, (cs) => cs.map((c) => c.dataset.chip));
check("M5's chips are the five built stages and the draw", JSON.stringify(m5chips) === JSON.stringify(['penalties', 'top-k', 'top-p', 'min-p', 'temperature', 'draw']), JSON.stringify(m5chips));

// ---------------------------------------------------------------- a custom order
check('moving top-p earlier works', await clickButton('move top-p earlier'));
let order = (await page.$$eval(`${B} [data-part="chips"] [data-chip]`, (cs) => cs.map((c) => c.dataset.chip))).filter((c) => c !== 'draw');
check('the order is now custom: top-p before top-k', JSON.stringify(order) === JSON.stringify(['penalties', 'top-p', 'top-k', 'min-p', 'temperature']), JSON.stringify(order));
check('no preset button is pressed for a custom order', (await page.$$eval(`${B} [aria-label="Order preset"] button[aria-pressed="true"]`, (b) => b.length)) === 0);
await same('custom: top-p before top-k', 'tail', s, order);
for (let i = 0; i < 4; i++) await clickButton('move temperature earlier');
order = (await page.$$eval(`${B} [data-part="chips"] [data-chip]`, (cs) => cs.map((c) => c.dataset.chip))).filter((c) => c !== 'draw');
check('temperature moved to the front with its arrow', order[0] === 'temperature', JSON.stringify(order));
check('the first chip cannot move earlier', !(await clickButton('move temperature earlier')));
await same('custom: temperature first', 'tail', s, order);
// drag and drop: min-p onto the penalties slot, with the HTML drag events a mouse drag fires
// (headless Chrome does not turn puppeteer's mouse drag into them)
await page.evaluate(async (B) => {
  const src = document.querySelector(`${B} [data-chip="min-p"]`);
  const dst = document.querySelector(`${B} [data-chip="penalties"]`);
  const dt = new DataTransfer();
  const fire = (el, type) => el.dispatchEvent(new DragEvent(type, { bubbles: true, cancelable: true, dataTransfer: dt }));
  fire(src, 'dragstart');
  await new Promise((r) => setTimeout(r, 50));
  fire(dst, 'dragenter');
  fire(dst, 'dragover');
  fire(dst, 'drop');
  fire(src, 'dragend');
}, B);
await pause(150);
const afterDrag = (await page.$$eval(`${B} [data-part="chips"] [data-chip]`, (cs) => cs.map((c) => c.dataset.chip))).filter((c) => c !== 'draw');
check('dragging min-p onto penalties puts it in that slot', afterDrag.indexOf('min-p') === order.indexOf('penalties'), JSON.stringify(order) + ' -> ' + JSON.stringify(afterDrag));
order = afterDrag;
await same('custom order after the drag', 'tail', s, order);
if (out) await page.screenshot({ path: out + '-custom.png' });

// ---------------------------------------------------------------- pinning
await clickButton('Pin this order for comparison');
check('pinning shows a second set of panels', (await page.$(`${B} [data-part="pinned"]`)) !== null);
await clickIn('Order preset', 'llama.cpp default');
await same('pinned custom order, as computed by the model', 'tail', s, order, 'pinned-panels');
await same('llama.cpp beside the pinned order', 'tail', s, 'llama');
await clickButton('Unpin');
check('unpin removes the comparison', (await page.$(`${B} [data-part="pinned"]`)) === null);

// ---------------------------------------------------------------- the prediction's case
await clickButton("Load the prediction's case");
const P = D.porder.settings;
await same("the prediction's case under transformers", 'tail', P, 'hf');
await same("the pinned llama.cpp order in the prediction's case", 'tail', P, 'llama', 'pinned-panels');
const cmp = await page.$eval(`${B} [data-readout="compare"]`, (e) => ({ set: e.dataset.sameSet, text: e.textContent.replace(/\s+/g, ' ') }));
check(`the badge says the sets differ: transformers ${D.counts.hf}, llama.cpp ${D.counts.llama}`,
  cmp.set === '0' && cmp.text.includes('different sets') && (await result()).final.split(' ').length === D.counts.hf, cmp.text);
check('the prediction case sets T = 2, k = 8, p = 0.90, min-p off', (await page.$eval(B, (e) => e.textContent)).includes('T = 2.0') && (await page.$eval(B, (e) => e.textContent)).includes('k = 8') && (await page.$eval(B, (e) => e.textContent)).includes('min-p = 0.00 (off)'));
if (out) await page.screenshot({ path: out + '-predict-case.png' });

// ---------------------------------------------------------------- the Predict and the questions
const qHandle = (text) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(t)), text);
const pq = await qHandle("transformers' order still draw from");
if (await pq.evaluate((e) => !!e.querySelector('input[type=number]'))) {
  await (await pq.$('input[type=number]')).type(String(D.counts.hf));
  await pq.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.includes('Commit')).click());
  await pause(150);
}
const rev = await page.$eval('[data-reveal="order"]', (e) => e.textContent.replace(/\s+/g, ' '));
check(`the p-order reveal gives the model's counts (${D.counts.hf} and ${D.counts.llama})`, rev.includes(`keeps ${D.counts.hf}`) && rev.includes(`keeps ${D.counts.llama}`), rev.slice(0, 120));
async function answer(text, how) {
  const q = await qHandle(text);
  if (typeof how === 'number' && how < 10 && !(await q.evaluate((e) => !!e.querySelector('input[type=number]')))) {
    await q.evaluate((e, i) => e.querySelectorAll('input[type=radio]')[i].click(), how);
  } else {
    const input = await q.$('input[type=number]');
    await input.click({ clickCount: 3 });
    await input.type(String(how));
  }
  await pause(50);
  await q.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Check').click());
  await pause(120);
  return q.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
}
check('q-temp-first: top-p and min-p, not top-k', (await answer('Temperature 0.5 is moved', 0)).startsWith('Correct.'));
check('q-temp-first: "all three" is wrong', (await answer('Temperature 0.5 is moved', 1)).startsWith('Not quite.'));
check('q-renorm: 2 tokens', (await answer('Top-k with k = 3 runs first', 2)).startsWith('Correct.'));
check('q-renorm: 3 is the full-row answer and is rejected', (await answer('Top-k with k = 3 runs first', 3)).startsWith('Not quite.'));
check("q-hf-more: transformers' order", (await answer('T = 1.5 and p = 0.9', 0)).startsWith('Correct.'));
check('q-greedy-switch: greedy replaces dist', (await answer("common sampler, with", 0)).startsWith('Correct.'));

// ---------------------------------------------------------------- the rest of the page
const contract = await page.$eval('table[data-part="contract"]', (t) => t.textContent);
check('the M5 contract names the order, greedy, the tie rule and the inputs',
  contract.includes('penalties → top-k → top-p → min-p → temperature → draw') && contract.includes('lowest token id') && contract.includes('T ≤ 0') && contract.includes('uint32'));
check('the scene is on the page', (await page.$('section.scene[aria-label="One row through the chain"]')) !== null);
const todays = await page.$$eval('aside.note.today', (a) => a.map((x) => x.textContent));
check('every Today note was checked in the facts pass', todays.length >= 3 && todays.every((t) => t.includes('Checked 2026-11-20 in the facts pass')));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
