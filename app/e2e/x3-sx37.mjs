// Browser check for section X3.7: the decode-step stepper with its three stops, the two-pairings
// explorer, both narrated scenes ("One token through the step", "Two ways to pair a head"), the
// RoPE listing and two questions. Expected values come from src/sim/llamastep.ts and
// src/sim/ropepair.ts, run here in Node.
// Usage: APP_URL=http://localhost:5173 node e2e/x3-sx37.mjs [--dark] [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

const L = await import('../src/sim/llamastep.ts');
const P = await import('../src/sim/ropepair.ts');
const dark = process.argv.includes('--dark');
const out = process.argv.slice(2).find((a) => !a.startsWith('--'));
const { browser, page, errors } = await open({ width: 1440, height: 1000, dark });
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
const strip = (s) => s.replace(/\[[^\]]*\]\s*/g, '').replace(/\s+/g, ' ').trim();
const bench = (t) => `section.bench[aria-label="${t}"]`;
const S = bench('Step through one token');
const E = bench('Two pairings');
const press = async (scope, label) => {
  const ok = await page.evaluate((scope, label) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(80);
  return ok;
};
const setRange = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(100);
};
const setSelect = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, v);
  await pause(100);
};
// A number field: clear it, then type, so React sees each keystroke as a reader's would.
const typeInto = async (sel, v) => {
  await page.$eval(sel, (el) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '');
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await page.type(sel, String(v));
  await pause(80);
};
const at = () => page.$eval(S + ' [data-at]', (e) => Number(e.dataset.at));
const total = (k) => page.$eval(`${S} [data-total="${k}"]`, (e) => Number(e.dataset.value));

await goto(page, 'x3/X3.7');
await page.waitForSelector(S);

// ---------------------------------------------------------------- the stepper
const cfg0 = { ...L.SYNTHETIC };
let model = L.stepLaunches(cfg0, 5);
let tot = L.totals(model);
check('opens on the synthetic model at pos 5', (await total('launches')) === model.length && (await total('weights')) === tot.weightBytes,
  `${await total('launches')} ${await total('weights')}`);
check('launch count is 1 + 16 L + 4', model.length === L.launchCount(cfg0.layers) && (await page.$eval(S + ' [data-total="launches"]', (e) => e.textContent)).includes('1 + 16 × 4 + 4'));
const jobs = await page.$$eval(S + ' tr[data-job]', (r) => r.map((x) => x.dataset.job));
check('the table lists embed, layer 0 and the tail in the model\'s order',
  jobs.join() === model.filter((l) => l.layer === null || l.layer === 0).map((l) => l.job).join(), jobs.join());

await press(S, 'Step');
let n = await page.$eval(S + ' [data-part="narrate"]', (e) => e.textContent);
check('step 1 is the embedding gather from lab 22.2', n.includes('launchEmbed') && n.includes('lab 22.2') && n.includes('2.0 KiB'), n);

check('Run to the next question stops before RoPE', await press(S, 'Run to the next question'));
check('the RoPE stop is open', !!(await page.$(S + ' [data-gate="rope"]')) && (await at()) === model.findIndex((l) => l.job === 'rope-q'));
check('Step is blocked while the question is open', !(await press(S, 'Step')));
await page.evaluate((S) => [...document.querySelectorAll(S + ' [data-gate="rope"] label')].find((l) => l.textContent.trim() === 'q and k').querySelector('input').click(), S);
await pause(60);
await press(S, 'Commit');
check('q and k is right', (await page.$eval(S + ' [data-answer="rope"]', (e) => e.dataset.ok)) === '1');
await press(S, 'Step');
n = await page.$eval(S + ' [data-part="narrate"]', (e) => e.textContent);
check('the next launch rotates q as 4 heads with the row-5 table', n.includes('launchRope') && n.includes('as 4 heads') && n.includes('cos and sin row 5'), n);

await press(S, 'Run to the next question');
check('the append stop is open', !!(await page.$(S + ' [data-gate="append"]')));
await typeInto(S + ' input[aria-label="cache row"]', 4);
await press(S, 'Commit');
check('row 4 is marked wrong (pos is 5)', (await page.$eval(S + ' [data-answer="append"]', (e) => e.dataset.ok)) === '0');
await press(S, 'Step');
n = await page.$eval(S + ' [data-part="narrate"]', (e) => e.textContent);
check('the append writes cache row 5', n.includes('K, V cache row 5') && n.includes('lab 20.4'), n);
await press(S, 'Step');
n = await page.$eval(S + ' [data-part="narrate"]', (e) => e.textContent);
const att = model.find((l) => l.job === 'attention');
check('attention is lab 20.5\'s grouped kernel over 6 rows', n.includes('launchDecodeAttentionGrouped') && n.includes('6 cached rows') && n.includes(`${(att.cacheBytes / 1024).toFixed(1)} KiB`), n);

await press(S, 'Run to the next question');
check('the LM head stop is open', !!(await page.$(S + ' [data-gate="share"]')) && (await at()) === model.findIndex((l) => l.job === 'lm-head'));
await typeInto(S + ' input[aria-label="LM head share"]', (100 * tot.lmHeadShare).toFixed(1));
await press(S, 'Commit');
const share = await page.$eval(S + ' [data-answer="share"]', (e) => ({ ok: e.dataset.ok, t: e.textContent }));
check('the computed share is accepted and shown', share.ok === '1' && share.t.includes((100 * tot.lmHeadShare).toFixed(1) + ' %'), share.t);
await press(S, 'Run to the next question');
check('the step runs to its last launch', (await at()) === model.length);
n = await page.$eval(S + ' [data-part="narrate"]', (e) => e.textContent);
check('the last launch is the advance of lab X3.2', n.includes('launchAdvancePositions'), n);
if (out) await page.screenshot({ path: out + '-step.png' });

// Change the model: 2 layers, 4 KV heads, pos 100.
await typeInto(S + ' input[aria-label="Layers"]', 2);
await setSelect(S + ' select[aria-label="KV heads"]', 4);
await setRange(S + ' input[aria-label="pos"]', 100);
const cfg1 = { ...cfg0, layers: 2, kvHeads: 4 };
model = L.stepLaunches(cfg1, 100);
tot = L.totals(model);
check('2 layers: 37 launches', (await total('launches')) === 37 && model.length === 37);
check('cache bytes at pos 100 match the model', (await total('cache')) === tot.cacheBytes, String(await total('cache')));
check('weights per token match the model', (await total('weights')) === tot.weightBytes);
const attnFn = await page.$eval(S + ' tr[data-job="attention"] code', (e) => e.textContent);
check('with equal head counts the attention is lab 20.4\'s kernel', attnFn === 'launchDecodeAttention', attnFn);
check('the stepper restarted', (await at()) === 0);

// ---------------------------------------------------------------- the pairing explorer
const verdict = () => page.$eval(E + ' [data-part="verdict"]', (e) => ({ same: e.dataset.same, diff: Number(e.dataset.diff) }));
const T8 = P.ropeTables(64, 8);
const C8 = P.makeCase(8);
let v = await verdict();
const want = P.compare(C8, 15, 'norm', false, T8, 15);
check('consecutive kernel on half-split weights differs, by the model\'s amount', v.same === '0' && Math.abs(v.diff - want.maxScoreDiff) < 1e-9, JSON.stringify(v));
const pairs = await page.$$eval(E + ' [data-el]', (d) => d.map((x) => Number(x.dataset.pair)));
check('NORM pairs neighbours', pairs.join() === '0,0,1,1,2,2,3,3', pairs.join());
await page.evaluate((E) => [...document.querySelectorAll(E + ' label.check')].find((l) => l.textContent.includes('permute')).querySelector('input').click(), E);
await pause(100);
v = await verdict();
check('with the permutation the scores equal the reference', v.same === '1' && v.diff < 1e-9, JSON.stringify(v));
check('the match is drawn in the success colour', (await page.$eval(E + ' path[data-curve="kernel"]', (e) => e.getAttribute('stroke'))) === 'var(--ok)');
await press(E, 'half split (NEOX)');
v = await verdict();
check('permuted weights through a half-split kernel differ again', v.same === '0' && Math.abs(v.diff - P.compare(C8, 15, 'neox', true, T8, 15).maxScoreDiff) < 1e-9, JSON.stringify(v));
await page.evaluate((E) => [...document.querySelectorAll(E + ' label.check')].find((l) => l.textContent.includes('permute')).querySelector('input').click(), E);
await pause(100);
check('half split on unpermuted weights is the reference', (await verdict()).same === '1');
await setSelect(E + ' select[aria-label="Head size"]', 12);
const p12 = await page.$$eval(E + ' [data-el]', (d) => d.map((x) => Number(x.dataset.pair)));
check('head 12, NEOX: element j pairs with j + 6', p12.join() === [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5].join(), p12.join());
await press(E, 'consecutive (NORM)');
await setRange(E + ' input[aria-label="position"]', 30);
v = await verdict();
check('head 12 at pos 30: the model\'s difference', Math.abs(v.diff - P.compare(P.makeCase(12), 30, 'norm', false, P.ropeTables(64, 12), 15).maxScoreDiff) < 1e-9, JSON.stringify(v));
if (out) await page.screenshot({ path: out + '-pairs.png' });

// ---------------------------------------------------------------- the scenes
async function scene(title, clip, n, verify) {
  const sc = `section.scene[aria-label="${title}"]`;
  await page.$eval(sc, (e) => e.scrollIntoView({ block: 'start' }));
  const words = [];
  for (let k = 1; k <= n; k++) words.push(strip(await (await fetch(`${BASE}/content/x3/narration/${clip}-${k}.txt`)).text()).split(' ').length);
  check(`${title}: every script is 35 to 60 words`, words.every((w) => w >= 35 && w <= 60), words.join(' '));
  const parts = await page.$$eval(sc + ' .seg button', (b) => b.length);
  check(`${title}: ${n} parts`, parts === n, String(parts));
  const box = await page.$eval(sc, (e) => [...e.querySelectorAll('label.check')].find((l) => l.textContent.includes('captions')).querySelector('input').checked);
  check(`${title}: captions switch exists and is off by default`, box === false);
  const manifest = await (await fetch(`${BASE}/content/x3/narration/manifest.json`)).json();
  const hasAudio = !!manifest?.clips?.[`${clip}-1`];
  if (!hasAudio) {
    const cap = await page.$eval(sc + ' .caption', (e) => e.textContent.replace(/\s+/g, ' ').trim()).catch(() => '');
    const s1 = strip(await (await fetch(`${BASE}/content/x3/narration/${clip}-1.txt`)).text());
    check(`${title}: with no audio part 1 shows its script`, cap === s1, cap.slice(0, 60));
  }
  for (let k = 0; k < n; k++) {
    await page.evaluate((sc, k) => document.querySelectorAll(sc + ' .seg button')[k].click(), sc, k);
    await pause(120);
    await page.$eval(sc + ' input[type=range]', (el) => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, '1000');
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await pause(120);
    const beat = await page.$eval(sc + ' .stage svg', (e) => Number(e.dataset.beat));
    check(`${title}: part ${k + 1} draws beat ${k + 1}`, beat === k + 1, String(beat));
    await verify(sc, k + 1);
  }
}

const MIS = P.compare(C8, 15, 'norm', false, T8, 15);
await scene('Two ways to pair a head', 'rope-pairs', 6, async (sc, b) => {
  if (b === 1) {
    const pr = await page.$$eval(sc + ' [data-elements="norm"] [data-el]', (d) => d.map((x) => Number(x.dataset.pair)));
    check('rope-pairs 1: consecutive pairs', pr.join() === '0,0,1,1,2,2,3,3', pr.join());
  }
  if (b === 2) {
    const ang = await page.$$eval(sc + ' [data-circle]', (c) => c.map((x) => Number(x.dataset.angle)));
    const q = P.matvec(C8.Wq, C8.xq, 8, C8.din);
    const r = P.rope(q, 3, 'norm', T8);
    check('rope-pairs 2: angles at position 3 match the model', ang.every((a, j) => Math.abs(a - Math.atan2(r[2 * j + 1], r[2 * j])) < 1e-3), ang.join());
  }
  if (b === 3) {
    const pr = await page.$$eval(sc + ' [data-elements="neox"] [data-el]', (d) => d.map((x) => Number(x.dataset.pair)));
    check('rope-pairs 3: half-split pairs', pr.join() === '0,1,2,3,0,1,2,3', pr.join());
  }
  if (b === 4) check('rope-pairs 4: the mismatch is the model\'s', (await page.$eval(sc + ' [data-diff]', (e) => e.dataset.diff)) === MIS.maxScoreDiff.toFixed(4));
  if (b === 5) check('rope-pairs 5: the permuted kernel matches', (await page.$eval(sc + ' [data-same]', (e) => e.dataset.same)) === '1');
  if (b === 6) {
    const ok = await page.$$eval(sc + ' [data-rowmap]', (r) => r.map((x) => x.dataset.ok));
    check('rope-pairs 6: n_head_kv works, the query head count does not', ok.join() === '1,0', ok.join());
  }
});

await scene('One token through the step', 'llama-step', 7, async (sc, b) => {
  if (b === 3) {
    const t = await page.$eval(sc + ' [data-shape="append"]', (e) => e.textContent);
    check('llama-step 3: the append writes row 5', t.includes('K, V cache row 5'), t);
  }
  if (b === 7) {
    const pos = await page.$eval(sc + ' [data-pos]', (e) => Number(e.dataset.pos));
    const t = L.totals(L.stepLaunches(L.SYNTHETIC, pos));
    const mm = await page.$eval(sc + ' [data-part="matrix products"]', (e) => Number(e.dataset.bytes));
    check('llama-step 7: totals at the end of the beat match the model', pos === L.SYNTHETIC.seqLen - 1 && mm === t.byKind.matmul, `${pos} ${mm}`);
    const head = await page.$eval(sc + ' [data-headline]', (e) => e.textContent);
    check('llama-step 7: launches per token in the headline', head.startsWith(`${L.launchCount(L.SYNTHETIC.layers)} launches`), head);
  }
});

// ---------------------------------------------------------------- listing and questions
await page.waitForSelector('figure.code .tabs button');
const code = await page.$eval('figure.code pre', (e) => e.textContent);
check('the RoPE listing is lab X3.1\'s kernel', code.includes('ropeKernel') && code.includes('cosT'), code.slice(0, 80));

const q1 = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('SwiGLU for one token of SmolLM2')));
await (await q1.$('input[type=number]')).type('18432');
await (await q1.$('button.primary')).click();
await pause(120);
check('q-swiglu-bytes accepts 18432', (await q1.evaluate((e) => e.textContent)).includes('Correct.'));
const q2 = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('does RoPE rotate, and what does the cache hold')));
await (await q2.$$('input[type=radio]'))[0].click();
await (await q2.$('button.primary')).click();
await pause(120);
check('q-rope-v accepts q and k', (await q2.evaluate((e) => e.textContent)).includes('Correct.'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
