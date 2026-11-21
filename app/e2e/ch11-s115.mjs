// Browser check for section 11.5: the prediction, the "Work and steps as N grows" explorer
// (every preset, every charge, the warp width, the N and P controls, the break-even readout)
// and the questions. Expected values come from src/sim/prefixnet.ts, imported here directly.
// Usage: node e2e/ch11-s115.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { breakEven, closedForm, span, stepsWithUnits } from '../src/sim/prefixnet.ts';

const out = process.argv[2];
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
const B = 'section.bench[aria-label="Work and steps as N grows"]';
const press = async (label) => {
  const done = await page.evaluate((B, label) => {
    const btn = [...document.querySelectorAll(B + ' button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, B, label);
  await pause(80);
  return done;
};
const setInput = async (sel, value) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(80);
};
const setSelect = async (sel, value) => {
  await page.$eval(sel, (el, v) => {
    el.value = v;
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  await pause(80);
};
/** The table as the page shows it: per row, additions, steps and span from data attributes. */
const table = () => page.$$eval(B + ' tbody tr', (rows) => Object.fromEntries(rows.map((r) => [r.dataset.row, {
  adds: Number(r.dataset.adds), steps: Number(r.dataset.steps), span: Number(r.dataset.span),
  ratio: r.querySelector('[data-cell="work-ratio"]').textContent.trim(), vs: r.querySelector('[data-cell="vs-loop"]').textContent.trim(),
}])));
const model = (n, p, charge, w = 32) => ({
  sequential: { adds: closedForm('sequential', n), steps: stepsWithUnits('sequential', n, p, charge, w), span: span('sequential', n) },
  'kogge-stone': { adds: closedForm('kogge-stone', n), steps: stepsWithUnits('kogge-stone', n, p, charge, w), span: span('kogge-stone', n) },
  'brent-kung': { adds: closedForm('brent-kung', n), steps: stepsWithUnits('brent-kung', n, p, charge, w), span: span('brent-kung', n) },
});
const agrees = (got, want) => Object.keys(want).every((k) => got[k].adds === want[k].adds && got[k].steps === want[k].steps && got[k].span === want[k].span);
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const question = (needle) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(t)), needle);

await goto(page, 'ch11/11.5');
await page.waitForSelector(B);

// ---- the prediction, before the explorer
const pred = await question('how many times fewer steps does Kogge-Stone take');
await (await pred.$('input[type=number]')).type('3.2');
await (await pred.$('button.primary')).click();
await pause(150);
const reveal = await pred.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
check('prediction reveals 320 steps and 3.2 times fewer from the model', reveal.includes(`${stepsWithUnits('kogge-stone', 1024, 32, 'threads')} steps`) && reveal.includes('3.2 times fewer'), reveal);
check('a right prediction is acknowledged', reveal.includes('That is what you predicted.'), reveal);

// ---- the explorer's default: the book's 1,024 elements on 32 units, every thread charged
let t = await table();
check('default: N = 1,024, P = 32, the book\'s charge', (await text(B + ' [data-value="n"]')) === '1,024' && (await text(B + ' [data-value="p"]')) === '32');
check('default table agrees with the model (320 steps against 1,024)', agrees(t, model(1024, 32, 'threads')) && t['kogge-stone'].steps === 320 && t.sequential.steps === 1024, JSON.stringify(t));
check('Kogge-Stone shows 3.2 times fewer steps', t['kogge-stone'].vs.includes('3.2× fewer'), t['kogge-stone'].vs);
check('the book\'s estimate (N·log2N)/P reads 320', (await text(B + ' [data-value="book-ks"]')) === '320');
if (out) await page.screenshot({ path: out + '-default.png' });

// ---- presets
await press('Book: 512 elements');
t = await table();
check('512 preset: 4097 additions against 511, 8.02 times the loop\'s', t['kogge-stone'].adds === 4097 && t.sequential.adds === 511 && t['kogge-stone'].ratio === '8.02', JSON.stringify(t['kogge-stone']));
check('512 preset: unlimited units, 512/9 = 56.9 times fewer steps', (await text(B + ' [data-value="unlimited-ratio"]')) === '56.9' && t['kogge-stone'].steps === 9 && t['brent-kung'].steps === 17);
check('512 preset: the naive scan does 130,816 additions', (await text(B + ' [data-value="naive"]')) === closedForm('naive', 512).toLocaleString('en-US'));
await press('A vocabulary row: 2¹⁷ elements, 1,024 units');
t = await table();
check('vocabulary preset agrees with the model', agrees(t, model(1 << 17, 1024, 'threads')) && t['kogge-stone'].steps < t.sequential.steps, JSON.stringify(t));
await press('Book: 1,024 elements, 32 units');

// ---- charges
await press('its additions');
t = await table();
check('charging additions only: Kogge-Stone 289 steps, Brent-Kung 71', agrees(t, model(1024, 32, 'adds')) && t['kogge-stone'].steps === 289 && t['brent-kung'].steps === 71, JSON.stringify(t));
check('the warp width is disabled unless whole warps are charged', await page.$eval(B + ' select[aria-label="warp width"]', (e) => e.disabled));
check('the explanation of the charge changes with it', (await text(B + ' [data-readout="charge"]')).startsWith('Only the threads that add'));
await press('whole warps');
await setSelect(B + ' select[aria-label="warp width"]', 'w64');
t = await table();
check('whole warps of 64 agree with the model', agrees(t, model(1024, 32, 'warps', 64)), JSON.stringify(t));
await press('every thread (the book)');

// ---- N and P controls, and the break-even
await setInput(B + ' input[aria-label="Elements, as a power of two"]', 12);
check('the N slider sets N = 4,096', (await text(B + ' [data-value="n"]')) === '4,096');
const exact = B + ' input[aria-label="Execution units, exactly"]';
await setInput(exact, 12);
t = await table();
check('4,096 elements on 12 units: Kogge-Stone needs more steps than the loop', t['kogge-stone'].steps >= 4096 && t['kogge-stone'].vs.includes('more'), JSON.stringify(t['kogge-stone']));
await setInput(exact, 13);
t = await table();
check('on 13 units it needs fewer', t['kogge-stone'].steps === stepsWithUnits('kogge-stone', 4096, 13, 'threads') && t['kogge-stone'].vs.includes('fewer'), JSON.stringify(t['kogge-stone']));
const be = await page.$eval(B + ' [data-readout="breakeven"]', (e) => ({ ks: Number(e.dataset.ks), bk: Number(e.dataset.bk) }));
check('break-even readout: 13 units for Kogge-Stone, the model\'s value for Brent-Kung', be.ks === 13 && be.bk === breakEven('brent-kung', 4096, 'threads'), JSON.stringify(be));
await setInput(B + ' input[aria-label="Execution units"]', 50);
check('the P slider sets 32 units on its log scale', (await text(B + ' [data-value="p"]')) === '32');
await setInput(B + ' input[aria-label="Elements, as a power of two"]', 20);
t = await table();
check('a million elements still agree with the model', agrees(t, model(1 << 20, 32, 'threads')), JSON.stringify(t));
if (out) await page.screenshot({ path: out + '-explorer.png' });

// ---- questions
const answer = async (needle, value) => {
  const q = await question(needle);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent.includes('Correct.'));
};
check('q-work512 accepts 4097', await answer('How many additions does Kogge-Stone perform on a segment of 512', 4097));
check('q-ratio accepts 8.02', await answer('How many times the sequential loop', 8.02));
check('q-breakeven accepts 13', await answer('smallest number of execution units', 13));
check('q-steps rejects 320', !(await answer('2,048 elements on 64 execution units', 320)));
const choice = async (needle, optionText) => {
  const q = await question(needle);
  await q.evaluate((e, o) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes(o)).querySelector('input').click(), optionText);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent.includes('Correct.'));
};
check('q-slots: idle threads in executing warps', await choice('real resource use is closer', 'Idle threads still occupy lanes'));
check('q-energy: extra work still costs energy', await choice('Is its extra work then free', 'each extra addition still costs energy'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
