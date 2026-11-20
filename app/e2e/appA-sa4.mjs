// Browser check for appendix A section A.4: the alignment-and-rounding stepper, the "Your
// device" panel (lab A.1's probe rows), the FP32-to-BF16 explorer and its code listing, the
// prediction and the questions. Results are compared with the models (src/sim/fpadd.ts and
// src/sim/minifloat.ts, imported in the page) and with /api/reports.
// The narrated scene on this page has its own check: e2e/appA-scene-disappearing-addend.mjs.
// Usage: node e2e/appA-sa4.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file. When fakeReports is set, /api/reports is answered
// with it, so the "Your device" panel's filled-in state can be checked without lab results.
let fakeReports = null;
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : fakeReports && r.method() === 'GET' && r.url().includes('/api/reports')
      ? r.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(fakeReports) })
      : r.continue(),
);

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const bench = (t) => `section.bench[aria-label="${t}"]`;
const A = bench('Alignment and rounding');
const D = bench('Your device');
const B = bench('FP32 to BF16 on the bits');
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
const pressPreset = async (scope, id) => {
  await page.$eval(`${scope} button[data-preset="${id}"]`, (b) => b.click());
  await pause(100);
};
const setText = async (sel, v) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, v);
  await pause(100);
};
const narrate = () => page.$eval(A + ' .narrate', (e) => e.textContent.trim());
const result = () => page.$eval(A + ' [data-result]', (e) => ({ result: Number(e.dataset.result), err: Number(e.dataset.err), reason: e.dataset.reason }));
const modelAdd = (fmt, a, b, mode) => page.evaluate(async (fmt, a, b, mode) => {
  const fp = await import('/src/sim/fpadd.ts');
  const mf = await import('/src/sim/minifloat.ts');
  const f = mf.PRESETS.find((x) => x.id === fmt);
  const t = fp.alignTrace(f, a, b, mode);
  const r = t.steps.find((s) => s.kind === 'round');
  return { result: t.result, err: t.errorUlps, reason: r?.reason, steps: t.steps.length, viaAdd: mf.add(f, a, b, mode) };
}, fmt, a, b, mode);

await goto(page, 'appA/A.4');
await page.waitForSelector(A);

// ---- the stepper, on the book's example
check('nothing has happened before the first step', (await narrate()).startsWith('Nothing has happened yet'));
await press(A, 'Step');
let n = await narrate();
check('step 1 compares the exponents 1 and -2, 3 apart', n.includes('1.00 × 2^1') && n.includes('1.00 × 2^-2') && n.includes('differ by 3') && n.includes('denormal'), n);
await press(A, 'Step');
check('step 2 is the first shift, into the guard bit', (await narrate()).startsWith('Shift 1'));
await press(A, 'Finish the shifts');
const small = await page.$eval(A + ' tr[data-row="small"]', (r) => ({ window: r.querySelector('[data-window]').dataset.window, tail: r.querySelector('[data-tail]')?.dataset.tail, grs: r.querySelector('[data-grs]')?.dataset.grs }));
check('after 3 shifts the window is 000 and the 1 sits in the guard bit (G R S = 1 0 0)', small.window === '000' && small.tail === '100' && small.grs === '100', JSON.stringify(small));
await press(A, 'Run to the end');
let r = await result();
let m = await modelAdd('toy5', 2, 0.25, 'rne');
check('the book\'s 2 + 0.25 gives 2.0, a tie at 0.5 ULP, as the model says', r.result === 2 && r.err === 0.5 && r.reason === 'tie-even' && m.result === 2 && m.viaAdd === 2, JSON.stringify([r, m]));
check('the narration explains the tie', (await narrate()).includes('a tie') && (await narrate()).includes('nearest-even keeps it'));
if (out) await (await page.$(A)).screenshot({ path: out + '-stepper-book.png' });
await press(A, 'up (toward +∞)');
r = await result();
check('switching to rounding up keeps the position and gives 2.5', r.result === 2.5 && (await page.$eval(A + ' [aria-live]', (e) => e.textContent)).startsWith('step 7'), JSON.stringify(r));
await press(A, 'to nearest, ties to even');

// Presets, each run to the end in every mode and compared with the model.
const presets = await page.evaluate(async () => (await import('/src/sim/fpadd.ts')).ALIGN_PRESETS.map((p) => ({ id: p.id, fmt: p.fmt.id, a: p.a, b: p.b })));
const MODES = [['rne', 'to nearest, ties to even'], ['rtz', 'toward zero'], ['up', 'up (toward +∞)'], ['down', 'down (toward −∞)']];
let presetsOk = true;
for (const p of presets) {
  await pressPreset(A, p.id);
  await press(A, 'Run to the end');
  for (const [mode, label] of MODES) {
    await press(A, label);
    const got = await result();
    const want = await modelAdd(p.fmt, p.a, p.b, mode);
    if (got.result !== want.result || got.reason !== want.reason || Math.abs(got.err - want.err) > 1e-12) {
      presetsOk = false;
      check(`preset ${p.id} in ${mode}`, false, JSON.stringify([got, want]));
    }
  }
  await press(A, 'to nearest, ties to even');
}
check('every preset in all four modes shows the model\'s result, rule and error', presetsOk);
await pressPreset(A, 'stall');
await press(A, 'Run to the end');
check('2048 + 1 in FP16 stays 2048 to nearest', (await result()).result === 2048);
await press(A, 'up (toward +∞)');
check('and becomes 2050 rounding up', (await result()).result === 2050);
await press(A, 'to nearest, ties to even');
await pressPreset(A, 'probe');
await press(A, 'Run to the end');
check('1 + 0.75 ULP in FP32 rounds up to 1 + 2^-23', (await result()).result === 1 + 2 ** -23);
await press(A, 'toward zero');
r = await result();
check('and toward zero gives 1, 0.75 ULP off (Exercise 3\'s adder)', r.result === 1 && r.err === 0.75, JSON.stringify(r));
if (out) await (await page.$(A)).screenshot({ path: out + '-stepper-probe.png' });
await press(A, 'to nearest, ties to even');
// Typed operands.
await setText(A + ' input[aria-label="operand a"]', '1000');
await setText(A + ' input[aria-label="operand b"]', '0.3');
await press(A, 'FP16');
await press(A, 'Run to the end');
r = await result();
m = await modelAdd('fp16', 1000, 0.3, 'rne');
check('typed 1000 + 0.3 in FP16 gives 1000.5, about 0.4 ULP off', r.result === 1000.5 && Math.abs(r.err - m.err) < 1e-12 && Math.abs(r.err - 0.4) < 0.001, JSON.stringify([r, m]));
check('and says the input 0.3 was rounded to the format', (await page.$eval(A, (e) => e.textContent)).includes('your input was rounded'));
await setText(A + ' input[aria-label="operand b"]', 'abc');
check('a non-number is refused with a hint', (await page.$eval(A, (e) => e.textContent)).includes('Type two numbers'));

// ---- the prediction
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Which will the adder return')));
await (await pred.$('input[type=number]')).type('2');
await (await pred.$('button.primary')).click();
await pause(150);
const pt = await pred.evaluate((e) => e.textContent);
check('committing 2.0 reveals the tie rule', pt.includes('You predicted 2') && pt.includes('Right, but for a reason the book does not give'), pt.slice(0, 160));

// ---- your device
const reports = await (await fetch(BASE + '/api/reports')).json();
const withProbe = Object.keys(reports).filter((b) =>
  (reports[b].results ?? []).some((x) => x.lab === 'appA/lab1_sumerror' && (x.bench ?? []).some((row) => row.name === 'probe')));
const shownBackends = await page.$$eval(D + ' tr[data-backend]', (rs) => rs.map((r) => r.dataset.backend));
if (withProbe.length) {
  check('the panel lists every backend with a probe row', JSON.stringify(shownBackends.sort()) === JSON.stringify(withProbe.sort()), JSON.stringify([shownBackends, withProbe]));
  if (withProbe.includes('emu')) check('the emulator row is labelled as the CPU\'s arithmetic', (await page.$eval(D + ' tr[data-backend="emu"]', (e) => e.textContent)).includes('the CPU'));
} else {
  check('with no probe results the panel says so and gives the command', (await page.$(D + ' [data-empty]')) !== null && (await page.$eval(D, (e) => e.textContent)).includes('python3 labs/pmpp.py run appA/lab1 --bench'));
}

// ---- FP32 to BF16
const bf = () => page.$eval(B + ' [data-rounded]', (e) => ({ rounded: Number(e.dataset.rounded), truncated: Number(e.dataset.truncated), carry: e.dataset.carry, tie: e.dataset.tie, text: e.textContent }));
const modelBf = (bits) => page.evaluate(async (bits) => {
  const fp = await import('/src/sim/fpadd.ts');
  const c = fp.bf16FromBits(bits);
  return { rounded: c.rounded, truncated: c.truncated };
}, bits);
const bfPresets = await page.evaluate(async () => (await import('/src/sim/fpadd.ts')).BF16_PRESETS);
let bfOk = true;
for (const p of bfPresets) {
  await pressPreset(B, p.id);
  const got = await bf();
  const want = await modelBf(p.bits);
  if (got.rounded !== want.rounded || got.truncated !== want.truncated) { bfOk = false; check(`BF16 preset ${p.id}`, false, JSON.stringify([got, want])); }
}
check('every BF16 preset shows the model\'s rounded and truncated bits', bfOk);
await pressPreset(B, 'tie-even');
let c = await bf();
check('tie with an even kept bit: stays 0x3f80 (1.0)', c.tie === '1' && c.rounded === 0x3f80, JSON.stringify(c));
await pressPreset(B, 'tie-odd');
c = await bf();
check('tie with an odd kept bit: rounds up to 0x3f82', c.tie === '1' && c.rounded === 0x3f82);
await pressPreset(B, 'carry');
c = await bf();
check('just under 2: the rounding carries into the exponent and gives 2', c.carry === '1' && c.rounded === 0x4000 && c.text.includes('on into the exponent'));
await pressPreset(B, 'max');
c = await bf();
check('the largest FP32 rounds to BF16 infinity, and the page says why', c.rounded === 0x7f80 && (await page.$eval(B, (e) => e.textContent)).includes('it becomes infinity'));
await pressPreset(B, 'snan');
c = await bf();
check('a NaN with a low payload: rounding keeps it a quiet NaN, truncation makes it infinity', c.rounded === 0x7fc0 && c.truncated === 0x7f80 && c.text.includes('the NaN became infinity'));
await setText(B + ' input[aria-label="FP32 value"]', '0.1');
c = await bf();
const tenth = await page.evaluate(async () => { const fp = await import('/src/sim/fpadd.ts'); return fp.bf16FromBits(fp.bitsFromF32(0.1)); });
check('a typed 0.1 converts as the model says', c.rounded === tenth.rounded && c.truncated === tenth.truncated, JSON.stringify([c.rounded, tenth.rounded]));
if (out) await (await page.$(B)).screenshot({ path: out + '-bf16.png' });
const listing = await page.$eval('.code pre', (e) => e.textContent);
check('the code shown is labs/appA/snippets/bf16_round.h, with llama.cpp\'s rounding line', listing.includes('bf16_round(float f)') && listing.includes('(u + (0x7fffu + ((u >> 16) & 1u))) >> 16'), listing.slice(0, 120));
check('the highlighted line is that rounding line', (await page.$eval('.code .ln.hl', (e) => e.textContent)).includes('0x7fffu'));

// ---- the scene is on the page
check('the narrated scene "The disappearing addend" is on the page', (await page.$('section.scene[aria-label="The disappearing addend"]')) !== null);

// ---- questions
const q = (needle) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(t)), needle);
const qe5 = await q('Exercise 5');
await (await qe5.$('input[type=number]')).type('63');
await (await qe5.$('button.primary')).click();
await pause(120);
check('Exercise 5: 63 ULP is accepted', (await qe5.evaluate((e) => e.textContent)).includes('Correct.'));
const qt = await q('accumulator holds 2048');
await (await qt.$$('input[type=radio]'))[0].click();
await (await qt.$('button.primary')).click();
await pause(120);
check('q-tie: 2048, a tie to even, is correct', (await qt.evaluate((e) => e.textContent)).includes('Correct.'));
const qx = await q('Exercise 3');
await (await qx.$$('input[type=radio]'))[1].click();
await (await qx.$('button.primary')).click();
await pause(120);
check('Exercise 3: 0.5 ULP is marked wrong, with the reason', (await qx.evaluate((e) => e.textContent)).includes('Not quite.'));

// ---- your device, filled in: synthetic probe rows (test data, not a measurement)
const probeRun = (rounding, keeps, fuses) => ({
  generated: 'test', device: { name: 'TEST DEVICE' },
  results: [{ lab: 'appA/lab1_sumerror', variant: 'solution', backend: 'x', tile: 0, built: true, test_rc: 0, tests: [],
    bench: [{ name: 'probe', rounding, keeps_denormals: keeps, fuses_mul_add: fuses }] }],
});
fakeReports = { metal: probeRun('toward-zero', false, true), emu: probeRun('nearest', true, false) };
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector(D + ' tr[data-backend]', { timeout: 15000 });
const metalRow = await page.$eval(D + ' tr[data-backend="metal"]', (e) => e.textContent);
const emuRow = await page.$eval(D + ' tr[data-backend="emu"]', (e) => e.textContent);
check('with probe rows, each backend gets a line: toward zero, flushed, fused', metalRow.includes('TEST DEVICE') && metalRow.includes('toward zero') && metalRow.includes('flushed to zero') && metalRow.includes('fused (one rounding)'), metalRow);
check('the emulator line is the CPU\'s arithmetic: rounds up, kept, two roundings', emuRow.includes('the CPU') && emuRow.includes('rounds up') && emuRow.includes('kept') && emuRow.includes('two roundings'), emuRow);
fakeReports = null;

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
