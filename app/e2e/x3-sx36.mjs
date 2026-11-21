// Browser check for section X3.6: the tie stepper (five implementations, the CUDA butterfly
// offset by offset, Metal's simd_max), the where-it-runs sort, the M4 listing and two questions.
// Every expected value comes from src/sim/argmaxties.ts, run here in Node.
// Usage: APP_URL=http://localhost:5173 node e2e/x3-sx36.mjs [--dark] [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { checkLabStatuses } from './labstatus.mjs';

const M = await import('../src/sim/argmaxties.ts');
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
const bench = (t) => `section.bench[aria-label="${t}"]`;
const T = bench('Ties in one warp');
const R = bench('Where each stage runs');
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
const results = () => page.$$eval(T + ' tr[data-impl]', (rs) => Object.fromEntries(rs.map((r) => [r.dataset.impl, r.dataset.result === undefined ? null : Number(r.dataset.result)])));
const lanes = () => page.$$eval(T + ' [data-lane]', (bs) => bs.map((b) => ({ idx: Number(b.dataset.idx), hot: b.dataset.hot === '1' })));
const expected = (cols) => Object.fromEntries(M.IMPLS.map((k) => [k.id, M.argmaxBy(k.id, M.tieRow(cols))]));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

await goto(page, 'x3/X3.6');
await page.waitForSelector(T);

// ---------------------------------------------------------------- the tie stepper
const planted = await page.$$eval(T + ' [data-col][data-planted="1"]', (b) => b.map((x) => Number(x.dataset.col)));
check('opens with columns 1 and 2 planted', same(planted, [1, 2]), JSON.stringify(planted));
const shown = await page.$$eval(T + ' [data-col]', (b) => b.map((x) => Number(x.dataset.value)));
check('the row shows the model row', same(shown, M.tieRow([1, 2])), shown.join(' '));
check('nothing revealed before Reveal', Object.values(await results()).every((v) => v === null));

const guesses = { cuda: '2', metal: '1', host: '1', torch: '1', m4: '1' };
for (const [id, v] of Object.entries(guesses)) {
  const sel = `${T} tr[data-impl="${id}"] input`;
  await page.click(sel, { clickCount: 3 });
  await page.type(sel, v);
}
check('Reveal is pressed', await press(T, 'Reveal'));
const r12 = await results();
check('columns 1 and 2: every implementation matches the model', same(r12, expected([1, 2])), JSON.stringify(r12));
check('CUDA returns 2 and Metal 2 for columns 1 and 2 (ch10 D2)', r12.cuda === 2 && r12.metal === 2 && r12.host === 1 && r12.m4 === 1);
const chips = await page.$$eval(T + ' tr[data-impl] .chip', (c) => c.map((x) => x.textContent));
check('the wrong guess is marked', chips.includes('you said 1') && chips.filter((c) => c === 'as you said').length === 4, chips.join('|'));

const bf = M.cudaButterfly(M.tieRow([1, 2]));
let st = await lanes();
check('stage 0: every lane holds its own column', st.every((l, k) => l.idx === k));
for (let s = 1; s <= M.OFFSETS.length; s++) {
  await press(T, 'Step');
  st = await lanes();
  const want = bf[s].lanes.map((p) => p.i);
  const hot = bf[s].changed;
  check(`offset ${bf[s].offset}: lane indices and changed lanes match the model`,
    same(st.map((l) => l.idx), want) && same(st.flatMap((l, k) => (l.hot ? [k] : [])), hot), JSON.stringify(st.map((l) => l.idx)));
  if (bf[s].offset === 2) check('at offset 2 lane 0 takes column 2', st[0].idx === 2 && st[0].hot);
}
check('after offset 1 lane 0 holds 2 and lane 1 holds 1', st[0].idx === 2 && st[1].idx === 1);
const narr = await page.$eval(T + ' [data-part="narrate"]', (e) => e.textContent);
check('the narration names lane 0\'s index', narr.includes('Lane 0 holds index 2'), narr);
await press(T, 'Step');
st = await lanes();
check('the Metal stage marks the lanes holding the maximum', same(st.flatMap((l, k) => (l.hot ? [k] : [])), M.metalMaxLanes(M.tieRow([1, 2]))));
check('Step stops at the last stage', !(await press(T, 'Step')));
if (out) await page.screenshot({ path: out + '-ties.png' });

for (const preset of [['0 and 1', [0, 1]], ['0 and 16', [0, 16]], ['3, 7 and 30', [3, 7, 30]]]) {
  await press(T, preset[0]);
  await press(T, 'Reveal');
  const r = await results();
  check(`preset ${preset[0]}: every implementation matches the model`, same(r, expected(preset[1])), JSON.stringify(r));
}
await press(T, '0 and 1');
await press(T, 'Reveal');
const r01 = await results();
check('columns 0 and 1: Metal 1, CUDA 0, M4 0', r01.metal === 1 && r01.cuda === 0 && r01.m4 === 0, JSON.stringify(r01));

// Plant by clicking columns.
await press(T, 'clear');
for (const c of [5, 9]) {
  await page.$eval(`${T} [data-col="${c}"]`, (b) => b.click());
  await pause(60);
}
await press(T, 'Reveal');
const r59 = await results();
check('columns planted by clicking: results match the model', same(r59, expected([5, 9])), JSON.stringify(r59));

// ---------------------------------------------------------------- where each stage runs
const pick = async (stage, bin) => {
  await page.$eval(`${R} tr[data-stage="${stage}"] button[data-bin="${bin}"]`, (b) => b.click());
  await pause(40);
};
check('Check is disabled until every stage is placed', !(await press(R, 'Check')));
for (const s of M.STAGE_LIST) await pick(s, 'host');
check('Check runs for llama.cpp defaults', await press(R, 'Check'));
let score = await page.$eval(R + ' [data-part="score"]', (e) => e.textContent);
check('llama.cpp defaults: all five on the host', score.startsWith('5 of 5'), score);
check('the order is shown after checking', (await page.$eval(R + ' [data-part="order"]', (e) => e.textContent)).includes('temperature → dist or greedy'));

await press(R, 'transformers generate');
for (const s of M.STAGE_LIST) await pick(s, s === 'draw' ? 'host' : 'framework');
await press(R, 'Check');
score = await page.$eval(R + ' [data-part="score"]', (e) => e.textContent);
const verdicts = await page.$$eval(R + ' tr[data-stage]', (rs) => Object.fromEntries(rs.map((r) => [r.dataset.stage, r.dataset.verdict])));
check('transformers: the draw on the host is marked wrong', score.startsWith('4 of 5') && verdicts.draw === 'wrong', score + JSON.stringify(verdicts));
check('every verdict agrees with the STAGES table', M.STAGE_LIST.every((s) => (verdicts[s] === 'right') === (M.cellOf('transformers', s).bin === (s === 'draw' ? 'host' : 'framework'))));
const cites = await page.$eval(R + ' tbody', (e) => e.textContent);
check('the source column cites the facts items', cites.includes('x1 D4, D5') && cites.includes('torch.multinomial'));

await press(R, 'llama.cpp with -bs');
for (const s of M.STAGE_LIST) await pick(s, 'kernels');
await press(R, 'Check');
score = await page.$eval(R + ' [data-part="score"]', (e) => e.textContent);
check('llama.cpp with -bs: all five in its graph', score.startsWith('5 of 5'), score);
if (out) await page.screenshot({ path: out + '-where.png' });

// ---------------------------------------------------------------- listing and questions
await page.waitForSelector('figure.code .tabs button');
const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
check('the M4 listing has CUDA and Metal tabs', tabs.includes('CUDA') && tabs.includes('Metal kernel'), tabs.join('|'));
check('the listing is lab 10.3\'s combine', (await page.$eval('figure.code pre', (e) => e.textContent)).includes('otherV == v && otherI < i'));

const q1 = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('Columns 1 and 2 hold the same')));
await (await q1.$('input[type=number]')).type('2');
await (await q1.$('button.primary')).click();
await pause(120);
check('q-tie-cuda accepts 2', (await q1.evaluate((e) => e.textContent)).includes('Correct.'));

const q2 = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('in that order')));
await (await q2.$$('input[type=radio]'))[0].click();
await (await q2.$('button.primary')).click();
await pause(120);
check('q-tie-metal accepts 1, 0, 0', (await q2.evaluate((e) => e.textContent)).includes('Correct.'));

const q3 = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('sampling chain of ordinary generation')));
await (await q3.$$('input[type=radio]'))[1].click();
await (await q3.$('button.primary')).click();
await pause(120);
check('q-default-place rejects the GPU answer', (await q3.evaluate((e) => e.textContent)).includes('Not quite.'));

// The listings' statuses: LabKit's words from this machine's reports, and "failed on this machine" when a run failed.
await checkLabStatuses(check, 'x3/X3.6', [
  { label: 'CUDA', lab: 'ch10/lab3_softmax_argmax', file: 'kernel.cu' },
  { label: 'Metal kernel', lab: 'ch10/lab3_softmax_argmax', file: 'kernel.metal' },
]);

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
