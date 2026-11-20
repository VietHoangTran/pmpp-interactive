// Browser check for section X1.6: the threshold prediction, the "Min-p before or after
// temperature" explorer (min-p, T, placement, top-p, the planted logit) and the "Filter before
// you sort" explorer (vocabulary, shape, another row, min-p, p), and the questions. Every
// number asserted is recomputed here from the model (src/sim/nucleus.ts).
// Usage: APP_URL=http://localhost:5173 node e2e/x1-sx16.mjs [--dark] [screenshot-prefix]
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
const M = await import('../src/sim/nucleus.ts');

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
const TE = 'section.bench[aria-label="Min-p before or after temperature"]';
const FE = 'section.bench[aria-label="Filter before you sort"]';
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const seg = async (scope, group, label) => {
  await page.evaluate((scope, group, label) => {
    const g = document.querySelector(`${scope} [role=group][aria-label="${group}"]`);
    [...g.querySelectorAll('button')].find((b) => b.textContent.trim().startsWith(label)).click();
  }, scope, group, label);
  await pause(250);
};
const setInput = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(250);
};
const q = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);

await goto(page, 'x1/X1.6');
await page.waitForSelector(TE);

// ---------------------------------------------------------------- the threshold prediction
const thr = M.minPThreshold(M.THRESHOLD_Q.lmax, M.THRESHOLD_Q.minP);
const pred = await q('The largest logit is 5.0 and min-p is 0.05');
await (await pred.$('input[type=number]')).type('2.004');
await (await pred.$('button.primary')).click();
await pause(250);
const rev = await text('[data-reveal=threshold]');
check('p-threshold: 5 + ln 0.05 = 2.004, as predicted', Math.abs(thr - 2.004) < 5e-4 && rev.includes(`= ${thr.toFixed(3)}, as you said`), rev);

// ---------------------------------------------------------------- before or after temperature
const tempState = async () => ({
  rows: await page.$$eval(TE + ' tr[data-row]', (rs) => Object.fromEntries(rs.map((r) => [r.dataset.row, Number(r.dataset.kept)]))),
  bars: await page.$$eval(TE + ' g[data-token]', (gs) => gs.map((g) => [g.dataset.keep, g.dataset.topp])),
  readout: await text(TE + ' [data-readout=placement]'),
});
const tempAgree = async (name, row, minP, T, p, placement) => {
  const v = M.minPView(row, minP, T, p);
  const st = await tempState();
  const bars = v.tokens.map((t) => [(placement === 'before' ? t.before : t.after) ? '1' : '0', t.topP ? '1' : '0']);
  check(name, st.rows.before === v.keptBefore && st.rows.after === v.keptAfter && st.rows.topp === v.keptTopP && JSON.stringify(st.bars) === JSON.stringify(bars),
    JSON.stringify({ st: st.rows, v: [v.keptBefore, v.keptAfter, v.keptTopP] }));
  return { v, st };
};
await page.$eval(TE, (e) => e.scrollIntoView({ block: 'start' }));
let r = await tempAgree('opens on the 16-token row, min-p 0.05, T = 1: 8 kept either way', M.MINP_ROW, 0.05, 1, 0.95, 'before');
check('at T = 1 both placements keep 8 and the readout says so', r.v.keptBefore === 8 && r.v.keptAfter === 8 && r.st.readout.startsWith('At T = 1.00 both placements keep 8'));
await setInput(TE + ' input[aria-label=temperature]', 70);
r = await tempAgree('T = 0.7, before temperature: the threshold does not move', M.MINP_ROW, 0.05, 0.7, 0.95, 'before');
check('T = 0.7: after-temperature min-p is stricter', r.v.keptAfter < r.v.keptBefore && r.st.readout.includes('stricter') && r.v.keptBefore === 8, r.st.readout);
await seg(TE, 'Placement', 'after');
r = await tempAgree('T = 0.7, after temperature: the bars follow the stricter threshold', M.MINP_ROW, 0.05, 0.7, 0.95, 'after');
if (out) await (await page.$(TE)).screenshot({ path: out + '-temp-after-07.png' });
await setInput(TE + ' input[aria-label=temperature]', 150);
r = await tempAgree('T = 1.5, after temperature: looser', M.MINP_ROW, 0.05, 1.5, 0.95, 'after');
check('T = 1.5: after-temperature min-p keeps more', r.v.keptAfter > r.v.keptBefore && r.st.readout.includes('looser'), r.st.readout);
await seg(TE, 'min-p', '0.2');
await setInput(TE + ' input[aria-label=top-p]', 60);
await tempAgree('min-p 0.2, top-p 0.60, T = 1.5: the model\'s counts', M.MINP_ROW, 0.2, 1.5, 0.6, 'after');
// The planted logit at T = 1, min-p 0.05.
await seg(TE, 'min-p', '0.05');
await setInput(TE + ' input[aria-label=temperature]', 100);
await setInput(TE + ' input[aria-label=top-p]', 95);
await seg(TE, 'Placement', 'before');
await page.evaluate((TE) => document.querySelector(TE + ' label.check input').click(), TE);
await pause(250);
const pl = M.plantedRow(0.05);
const t32 = M.minPTestsF32(pl.logits, 0.05);
r = await tempAgree('planted: 17 tokens, the model\'s counts', pl.logits, 0.05, 1, 0.95, 'before');
const planted = await page.$eval(TE + ' [data-readout=planted]', (e) => [e.dataset.logit, e.dataset.prob, e.textContent]);
check('planted: the logit test keeps it, the float32 probability test drops it', r.st.bars.length === 17 && planted[0] === '1' && planted[1] === '0'
  && t32.logit[pl.planted] && !t32.prob[pl.planted] && planted[2].includes('the two tests disagree by one rounding'), JSON.stringify(planted));
if (out) await (await page.$(TE)).screenshot({ path: out + '-temp-planted.png' });

// ---------------------------------------------------------------- filter before you sort
const filterAgree = async (name, V, s, seed, p, minP) => {
  const f = M.filterBeforeSort(M.rowView(V, s, seed), p, minP);
  const sv = await page.$eval(FE + ' [data-readout=survivors]', (e) => Number(e.dataset.survivors));
  const routes = await page.$$eval(FE + ' tr[data-route]', (rs) => rs.map((x) => [x.dataset.route, Number(x.dataset.kept), x.dataset.ids]));
  const verdict = await page.$eval(FE + ' [data-readout=verdict]', (e) => [e.dataset.identical, e.dataset.cDiffers]);
  const band = await page.$eval(FE + ' [data-readout=band]', (e) => Number(e.dataset.band));
  const ids = (a) => Array.from(a.slice(0, 50)).join(',');
  const want = [['A', f.routeA.length, ids(f.routeA)], ['B', f.routeB.length, ids(f.routeB)], ['C', f.routeC.length, ids(f.routeC)]];
  check(name, sv === f.survivors && JSON.stringify(routes) === JSON.stringify(want) && verdict[0] === (f.identical ? '1' : '0')
    && verdict[1] === (f.cDiffers ? '1' : '0') && band === f.band, JSON.stringify({ sv, f: f.survivors, routes: routes.map((x) => x.slice(0, 2)), verdict, band, fb: f.band }));
  return f;
};
await page.$eval(FE, (e) => e.scrollIntoView({ block: 'start' }));
let V = 151936, s = 1, seed = 1, p = 0.95, minP = 0.05;
let f = await filterAgree('opens on 151,936 tokens, s = 1, min-p 0.05, p = 0.95: the model\'s routes', V, s, seed, p, minP);
check('min-p 0.05 leaves tens of survivors out of 151,936, and A and B agree', f.survivors >= 10 && f.survivors < 100 && f.identical
  && (await text(FE + ' [data-readout=verdict]')).includes('A and B keep identical sets'));
if (out) await (await page.$(FE)).screenshot({ path: out + '-filter.png' });
s = 0.5;
await setInput(FE + ' input[aria-label=shape]', s);
f = await filterAgree('a flat row (s = 0.5): the model\'s routes', V, s, seed, p, minP);
await seg(FE, 'Vocabulary', '50,257');
V = 50257;
await setInput(FE + ' input[aria-label=top-p]', 70);
p = 0.7;
await seg(FE, 'min-p', '0.1');
minP = 0.1;
f = await filterAgree('50,257 tokens, p = 0.70, min-p 0.1: the model\'s routes', V, s, seed, p, minP);
await page.evaluate((FE) => [...document.querySelectorAll(FE + ' button')].find((b) => b.textContent.trim() === 'another row').click(), FE);
await pause(250);
seed = 2;
f = await filterAgree('another row: the model\'s routes', V, s, seed, p, minP);
check('A and B identical on every row tried', f.identical);
await seg(FE, 'min-p', 'off');
minP = 0;
f = await filterAgree('min-p off: every token survives', V, s, seed, p, minP);
check('min-p off removes the saving', f.survivors === V && (await text(FE + ' [data-readout=survivors]')).includes('no saving at all'));

// ---------------------------------------------------------------- questions
const numeric = async (needle, value) => {
  const h = await q(needle);
  await (await h.$('input[type=number]')).type(String(value));
  await (await h.$('button.primary')).click();
  await pause(150);
  return (await h.evaluate((e) => e.textContent)).includes('Correct.');
};
const choice = async (needle, index) => {
  const h = await q(needle);
  await (await h.$$('input[type=radio]'))[index].click();
  await (await h.$('button.primary')).click();
  await pause(150);
  return (await h.evaluate((e) => e.textContent)).includes('Correct.');
};
const ba = M.minPView(M.BEFORE_AFTER.logits, M.BEFORE_AFTER.minP, M.BEFORE_AFTER.T, 1);
check('q-before-after accepts 2 (threshold 3.70 on the scaled logits)', ba.keptAfter === 2 && Math.abs(ba.thrAfter / M.BEFORE_AFTER.T - 3.70) < 5e-3 && (await numeric('min-p after temperature) keep', 2)));
check('q-space: one compare, no softmax', await choice('the one a kernel should use', 0));
check('q-prefix: min(a, b)', await choice('how many tokens survive both', 0));
check('q-equivalent: yes in exact arithmetic', await choice('Does that keep the same tokens', 0));
check('q-mask: static shapes', await choice('instead of compacting them into a list', 0));

// Phone width: nothing makes the page scroll sideways.
await page.setViewport({ width: 390, height: 800, deviceScaleFactor: 1 });
await pause(300);
const sw = await page.evaluate(() => document.documentElement.scrollWidth);
check('no sideways page scroll at 390 px', sw <= 390, String(sw));
if (out) {
  await page.$eval(TE, (e) => e.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: out + '-narrow.png' });
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
