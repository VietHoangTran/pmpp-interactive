// Browser check for section X1.8: the "Find u" stepper (rows, n, T, u typed and from Philox,
// survivors per thread, warp width, every phase, and the barrier hunt under each order), the
// "Near a boundary" explorer, the "Same bits, three samplers" explorer, the prediction, the
// questions and the code. Every number asserted is recomputed here from the model
// (src/sim/draw.ts and src/sim/philox.ts).
// Usage: APP_URL=http://localhost:5173 node e2e/x1-sx18.mjs [--dark] [screenshot-prefix]
import { registerHooks } from 'node:module';
import { open, goto } from './browser.mjs';

registerHooks({
  resolve(spec, ctx, next) {
    try {
      return next(spec, ctx);
    } catch (e) {
      if (spec.startsWith('.') && !/\.\w+$/.test(spec)) return next(spec + '.ts', ctx);
      throw e;
    }
  },
});
const D = await import('../src/sim/draw.ts');
const P = await import('../src/sim/philox.ts');

const dark = process.argv.includes('--dark');
const out = process.argv.slice(2).find((a) => !a.startsWith('--'));
const { browser, page, errors } = await open({ width: 1440, height: 1000, dark });
await page.setRequestInterception(true);
const puts = [];
page.on('request', (r) => {
  if (r.method() === 'PUT' && r.url().includes('/api/progress')) {
    puts.push(r.postData());
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
const bench = (t) => `section.bench[aria-label="${t}"]`;
const S = bench('Find u');
const B = bench('Near a boundary');
const M = bench('Same bits, three samplers');
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const press = async (scope, label) => {
  const done = await page.evaluate((scope, label) => {
    const b = [...document.querySelectorAll(scope + ' button')].find((x) => x.textContent.trim() === label);
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, scope, label);
  await pause(100);
  return done;
};
const setInput = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(150);
};
const result = () => page.$eval(S + ' [data-result]', (e) => ({ index: Number(e.dataset.result), correct: Number(e.dataset.correct), fallback: e.dataset.fallback }));
const f32 = Math.fround;
const rowOf = (id) => D.STEP_ROWS.find((r) => r.id === id);
const wOf = (r) => D.weightsFrom(D.sampleLogits(r.n), r.T);

await goto(page, 'x1/X1.8');
await page.waitForSelector(S);

// ---------------------------------------------------------------- the stepper, 200 survivors
const long = rowOf('long');
const kl = D.drawKernel(wOf(long), f32(long.u), D.STEPPER_CFG);
check('stepper opens on the 200-survivor row, nothing run', (await text(S + ' .narrate')).startsWith('One block of 16 threads owns this row of 200'));
await press(S, 'Step');
check("step 1: phase 1's total and target are the model's",
  Number(await page.$eval(S + ' [data-total]', (e) => e.dataset.total)) === kl.total && Number(await page.$eval(S + ' [data-target]', (e) => e.dataset.target)) === kl.target);
await press(S, 'Step');
await press(S, 'Step');
const chunksSeen = await page.$$eval(S + ' tr[data-chunk]', (e) => e.map((x) => x.dataset.crosses));
check('steps 2 and 3: chunks 0 and 1 are skipped', JSON.stringify(chunksSeen) === JSON.stringify(['0', '0', '']), JSON.stringify(chunksSeen));
await press(S, 'Step');
check('step 4: chunk 2 crosses', (await page.$eval(S + ' tr[data-chunk="2"]', (e) => e.dataset.crosses)) === '1');
await press(S, 'Step');
const ch = kl.chunks[2];
const hits = await page.$$eval(S + ' tr[data-thread]', (e) => e.map((x) => x.dataset.hit));
check("step 5: each thread's first flagged survivor is the model's", JSON.stringify(hits) === JSON.stringify(ch.lanes.map((l) => (l.hit >= 0 ? String(l.first + l.hit) : ''))), JSON.stringify(hits));
await press(S, 'Step');
const ballots = await page.$$eval(S + ' [data-ballot]', (e) => e.map((x) => Number(x.dataset.bits)));
check('step 6: one ballot per warp, as the model', JSON.stringify(ballots) === JSON.stringify(ch.ballots), JSON.stringify(ballots));
await press(S, 'Step');
let r = await result();
check(`done: survivor ${kl.index} (133), equal to the CPU reference`, r.index === kl.index && r.index === 133 && r.index === D.inverseCdf(wOf(long), f32(long.u)).index, JSON.stringify(r));
if (out) await page.screenshot({ path: out + '-stepper.png' });

// the chunks table stays inside its panel and off the lanes column, at two widths
const chunksFit = () => page.evaluate((S) => {
  const panel = document.querySelector(S + ' [data-part="chunks"]');
  const scroll = panel.querySelector('[data-part="chunks-scroll"]');
  const table = scroll.querySelector('table');
  const right = panel.parentElement.nextElementSibling;
  const p = panel.getBoundingClientRect(), sc = scroll.getBoundingClientRect(), t = table.getBoundingClientRect();
  const clipped = getComputedStyle(scroll).overflowX === 'auto';
  return {
    scrollInside: sc.right <= p.right + 1,
    tableVisible: t.right <= p.right + 1 || clipped,
    clearOfLanes: Math.min(sc.right, t.right) <= right.getBoundingClientRect().left + 1,
    overflowPx: Math.round(t.right - p.right),
  };
}, S);
for (const width of [1440, 1024]) {
  await page.setViewport({ width, height: 1000 });
  await pause(200);
  const fit = await chunksFit();
  check(`${width} px: the chunks table does not spill over the lanes column`, fit.scrollInside && fit.tableVisible && fit.clearOfLanes, JSON.stringify(fit));
}
await page.setViewport({ width: 1440, height: 1000 });
await pause(200);

// widths and survivors per thread
for (const [vec, warp] of [[1, 4], [8, 16]]) {
  await press(S, String(vec));
  await page.evaluate((S, w) => [...document.querySelectorAll(S + ' [aria-label="Warp width"] button')].find((b) => b.textContent.trim() === w).click(), S, String(warp));
  await pause(100);
  await press(S, 'Run to the end');
  const k = D.drawKernel(wOf(long), f32(long.u), { threads: 16, warp, vec });
  r = await result();
  const nCh = await page.$$eval(S + ' tr[data-chunk]', (e) => e.length);
  check(`${vec} per thread, warp ${warp}: ${k.chunks.length} chunks walked, survivor ${k.index}`, r.index === k.index && nCh === k.chunks.length, `${r.index} ${nCh}`);
}
await press(S, '4');
await page.evaluate((S) => [...document.querySelectorAll(S + ' [aria-label="Warp width"] button')].find((b) => b.textContent.trim() === '8').click(), S);
await pause(100);

// n, T and u typed
await setInput(S + ' input[aria-label=n]', 57);
await setInput(S + ' input[aria-label=T]', 1.5);
await setInput(S + ' input[aria-label=u]', 0.33);
await press(S, 'Run to the end');
const wt = D.weightsFrom(D.sampleLogits(57), 1.5);
r = await result();
check('n = 57, T = 1.5, u = 0.33: the kernel model\'s token', r.index === D.drawKernel(wt, f32(0.33)).index, JSON.stringify(r));
// u from Philox
await page.$eval(S + ' input[aria-label="u from Philox"]', (e) => e.click());
await pause(150);
await setInput(S + ' input[aria-label="philox seed"]', 11);
await setInput(S + ' input[aria-label="philox stream"]', 2);
await setInput(S + ' input[aria-label="philox step"]', 9);
const pu = P.m5U(11n, 2n, 9n);
check("u from Philox is M5's u for (11, 2, 9)", Number(await page.$eval(S + ' [data-u]', (e) => e.dataset.u)) === pu);
await press(S, 'Run to the end');
r = await result();
check('and the token is the model\'s for that u', r.index === D.drawKernel(wt, pu).index);
await page.$eval(S + ' input[aria-label="u from Philox"]', (e) => e.click());
await pause(100);

// ---------------------------------------------------------------- the barrier hunt
await press(S, '120 survivors, T = 2');
const hunt = rowOf('hunt');
const hw = wOf(hunt);
const cases = [
  ['remove A', 'written', 'emulator, in order', 'in-order'],
  ['remove A', 'written', 'emulator, reversed', 'reversed'],
  ['remove A', 'written', 'warp 0 runs ahead (a GPU may)', 'warp-ahead'],
  ['remove B', 'read', 'emulator, in order', 'in-order'],
  ['remove B', 'read', 'emulator, reversed', 'reversed'],
  ['remove B', 'read', 'warp 0 runs ahead (a GPU may)', 'warp-ahead'],
];
for (const [btn, m, ob, o] of cases) {
  await press(S, btn);
  await press(S, ob);
  await press(S, 'Run to the end');
  const h = D.huntDraw(hw, f32(hunt.u), D.STEPPER_CFG, m, o);
  r = await result();
  const wrong = Number(await page.$eval(S + ' [data-part=hunt]', (e) => e.dataset.wrong));
  check(`${btn}, ${o}: ${h.hunt.wrong} threads read the wrong value, token ${h.index}${h.index === h.correct ? '' : ` instead of ${h.correct}`}`,
    wrong === h.hunt.wrong && r.index === h.index && r.correct === h.correct, `${wrong} ${JSON.stringify(r)}`);
  if (out && m === 'written' && o === 'in-order') await page.screenshot({ path: out + '-hunt.png' });
}
check('the narrator turns red when the token is wrong', !!(await page.$(S + ' .narrate.bad')));
await press(S, 'both in place');

// ---------------------------------------------------------------- near a boundary
const row = D.weightsFrom(D.sampleLogits(40), 1);
const tokensOf = () => page.$$eval(B + ' tr[data-order]', (e) => Object.fromEntries(e.map((x) => [x.dataset.order, x.dataset.tokens])));
const expectGrid = (b, pert) => {
  const c = Math.round(D.flipPoints(row, 'double')[b] * 2 ** 24);
  const ks = Array.from({ length: 25 }, (_, i) => i - 12);
  const g = Object.fromEntries(['double', 'f32-loop', 'f32-kernel'].map((o) => [o, ks.map((d) => D.chooseIn(row, (c + d) / 2 ** 24, o)).join(',')]));
  if (pert) g.perturbed = ks.map((d) => D.chooseIn(D.perturb(row, pert), (c + d) / 2 ** 24, 'f32-kernel')).join(',');
  return g;
};
let g = await tokensOf();
check('boundary 20: the three orders\' tokens over 25 steps of u are the model\'s', JSON.stringify(g) === JSON.stringify(expectGrid(20, 0)), JSON.stringify(g).slice(0, 200));
check('boundary 20: the float32 orders disagree with double somewhere in the window', g['f32-loop'] !== g.double || g['f32-kernel'] !== g.double);
await press(B, '2⁻²⁰');
g = await tokensOf();
check('perturbation 2^-20 adds a fourth row, as the model', JSON.stringify(g) === JSON.stringify(expectGrid(20, 2 ** -20)));
await setInput(B + ' input[aria-label=boundary]', 6);
await setInput(B + ' input[aria-label="u offset"]', -1);
g = await tokensOf();
check('boundary 6 with the perturbation: the model\'s grid', JSON.stringify(g) === JSON.stringify(expectGrid(6, 2 ** -20)));
const c6 = Math.round(D.flipPoints(row, 'double')[6] * 2 ** 24);
const u6 = (c6 - 1) / 2 ** 24;
check('the readout gives the margin of u = boundary − 1 step', Number(await page.$eval(B + ' [data-part=readout]', (e) => e.dataset.margin)) === D.marginOf(row, u6));
const measure = Number(await page.$eval(B + ' [data-part=risk]', (e) => e.dataset.measure));
const want = D.disagreement([{ weights: row, order: 'double' }, { weights: row, order: 'f32-loop' }, { weights: row, order: 'f32-kernel' }, { weights: D.perturb(row, 2 ** -20), order: 'f32-kernel' }]).measure;
check('the share of u at risk is the model\'s', measure === want, `${measure} ${want}`);
if (out) await page.screenshot({ path: out + '-boundary.png' });

// ---------------------------------------------------------------- three samplers
const freq = () => page.$$eval(M + ' td[data-method]', (e) => e.map((x) => Number(x.dataset.count)));
const expectFreq = (n, seed, q = 'one-minus') => {
  const c = D.compareSamplers(D.SAMPLER_PROBS, n, seed, 0, q);
  return { counts: D.SAMPLER_PROBS.flatMap((_, i) => ['inverse-cdf', 'race', 'gumbel'].map((m) => c.counts[m][i])), c };
};
await press(M, '10');
check('10 draws: counts are the model\'s', JSON.stringify(await freq()) === JSON.stringify(expectFreq(10, 5).counts));
await press(M, '100,000');
const e100 = expectFreq(100000, 5);
check('100,000 draws: counts are the model\'s', JSON.stringify(await freq()) === JSON.stringify(e100.counts));
const agree = await page.$eval(M + ' [data-part=agree]', (e) => [Number(e.dataset.invRace), Number(e.dataset.invGumbel), Number(e.dataset.raceGumbel)]);
check('the per-draw agreement is the model\'s and well below 100%', JSON.stringify(agree) === JSON.stringify([e100.c.agree.invRace, e100.c.agree.invGumbel, e100.c.agree.raceGumbel]) && agree.every((a) => a < 0.9));
await setInput(M + ' input[aria-label="sampler seed"]', 9);
const firsts = await page.$$eval(M + ' tr[data-draw]', (e) => e.map((x) => x.dataset.picks));
const wantFirst = Array.from({ length: 8 }, (_, d) => {
  const us = D.uniformsFor(9, 0, d, 6);
  return ['inverse-cdf', 'race', 'gumbel'].map((m) => D.sampleWith(D.SAMPLER_PROBS, us, m)).join(',');
});
check('seed 9: the first eight draws by each method are the model\'s', JSON.stringify(firsts) === JSON.stringify(wantFirst), JSON.stringify(firsts));
await page.$eval(M + ' input[aria-label="race with q = -ln u"]', (e) => e.click());
await pause(400);
const agree2 = Number(await page.$eval(M + ' [data-part=agree]', (e) => e.dataset.raceGumbel));
check('with q = −ln u the race and Gumbel-max agree on every draw', agree2 === 1, String(agree2));
if (out) await page.screenshot({ path: out + '-samplers.png' });

// ---------------------------------------------------------------- prediction and questions
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((q) => q.textContent.includes('equal weights 1, 1, 1, 1')));
await (await pred.$('input')).type('3');
await (await pred.$('button.primary')).click();
await pause(300);
check('the p-draw prediction reveals 2', (await page.$eval('[data-reveal]', (e) => e.dataset.reveal).catch(() => null)) === '2');
const answer = async (fragment, how) => {
  const q = await page.evaluateHandle((f) => [...document.querySelectorAll('.q')].find((x) => x.querySelector('.prompt')?.textContent.includes(f)), fragment);
  if (typeof how === 'number') (await q.$$('input[type=radio]'))[how].click();
  else await (await q.$('input[type=number]')).type(how);
  await pause(80);
  await (await q.$('button.primary')).click();
  await pause(150);
  return q.evaluate((x) => x.querySelector('.verdict')?.className ?? '');
};
check('q-risk: 74 is right', (await answer('draws per million are at risk', '74')).includes('right'));
check('q-ge-gt: the first option is right', (await answer('[0, 0.5, 0.5]', 0)).includes('right'));
check('q-barrier: "emu, in order" is marked wrong', (await answer('Remove barrier B', 1)).includes('wrong'));
check('q-multinomial: the exponential race is right', (await answer('torch.multinomial(probs, 1)', 0)).includes('right'));
check('answers were saved through PUT /api/progress (intercepted)', puts.length > 0);

// ---------------------------------------------------------------- code
await page.waitForFunction(() => [...document.querySelectorAll('figure.code')].some((f) => f.textContent.includes('drawKernel')), { timeout: 10000 }).catch(() => null);
const hl = await page.$$eval('figure.code .ln.hl', (e) => e.map((x) => x.textContent));
check('the snippet is shown with barriers A and B highlighted', hl.length === 2 && hl[0].includes('barrier A') && hl[1].includes('barrier B'), JSON.stringify(hl));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
