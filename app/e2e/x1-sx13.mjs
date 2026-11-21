// Browser check for section X1.3, "Penalties from token counts": the Penalties explorer
// (three sliders, three sign rules, both orders, clicking columns, appending the most
// likely token, resetting the context), the "Who reads which logit" explorer (every
// mapping, batch, warp width, segment model and Metal size), lab 9.3's listing, the
// Predict and every question. Expected values come from src/sim/penalties.ts, run inside
// the page through the app's module server.
// Usage: APP_URL=http://localhost:5173 node e2e/x1-sx13.mjs [screenshot-prefix]
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
const P = 'section.bench[aria-label="Penalties"]';
const A = 'section.bench[aria-label="Who reads which logit"]';

const clickIn = async (bench, group, match) => {
  const ok = await page.evaluate((bench, group, match) => {
    const b = [...document.querySelectorAll(`${bench} [aria-label="${group}"] button`)].find((x) => x.textContent.trim().startsWith(match));
    if (!b) return false;
    b.click();
    return true;
  }, bench, group, match);
  await pause(80);
  return ok;
};
const clickButton = async (bench, match) => {
  const ok = await page.evaluate((bench, match) => {
    const b = [...document.querySelectorAll(`${bench} button`)].find((x) => x.textContent.trim().startsWith(match));
    if (!b) return false;
    b.click();
    return true;
  }, bench, match);
  await pause(80);
  return ok;
};
const setSlider = async (label, raw) => {
  await page.$eval(`${P} input[aria-label="${label}"]`, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, raw);
  await pause(80);
};
// The explorer as shown: context, counts and logits after, per column.
const shown = () =>
  page.evaluate((P) => {
    const ctx = document.querySelector(`${P} [data-readout="context"]`);
    const cols = [...document.querySelectorAll(`${P} g[data-token]`)];
    const pen = document.querySelector(`${P} [data-readout="pen"]`);
    return {
      context: ctx.querySelector('.mono').textContent.trim().split(/\s+/).map(Number),
      length: Number(ctx.dataset.length),
      counts: cols.map((g) => Number(g.dataset.count)),
      after: cols.map((g) => Number(g.dataset.after)),
      raised: pen.dataset.raised ? pen.dataset.raised.split(' ').map(Number) : [],
      gained: pen.dataset.gained ? pen.dataset.gained.split(' ').map(Number) : [],
      text: pen.textContent.replace(/\s+/g, ' '),
    };
  }, P);
// The model's answer for the same context and settings.
const model = (context, s) =>
  page.evaluate(async (context, s) => {
    const m = await import('/src/sim/penalties.ts');
    const before = m.PEN_LOGITS.map(m.f32);
    const counts = m.countTokens(context, m.PEN_VOCAB);
    const after = m.applyPenalties(before, counts, { ...s, float32: true });
    const { raised, gained } = m.penaltyEffects(before, after, counts);
    return { counts, after, before, raised, gained, next: m.argmaxLowest(after), context0: [...m.PEN_CONTEXT] };
  }, context, s);
const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

await goto(page, 'x1/X1.3');
await page.waitForSelector(P);
await page.waitForSelector(A);

// ---------------------------------------------------------------- penalties: the opening state
let S = { repeat: 1.25, freq: 0, presence: 0, rule: 'llama', order: 'repeat-first' };
let v = await shown();
let M = await model(v.context, S);
check('the explorer opens on the twenty-token context', v.length === 20 && same(v.context, M.context0), v.context.join(' '));
check('the counts under the columns are the model\'s counts of that context', same(v.counts, M.counts), v.counts.join(' '));
check("repetition 1.25 under llama.cpp's rule: every logit after is the model's float32 value", same(v.after, M.after), JSON.stringify(v.after));
check("under llama.cpp's rule no repeated token's logit goes up", v.raised.length === 0 && M.raised.length === 0 && v.text.includes("No repeated token's logit went up"), v.text);
check('penalised tokens that gain probability by renormalisation are named apart (2, 4, 5, 7, 8)', same(v.gained, M.gained) && same(v.gained, [2, 4, 5, 7, 8]) && v.text.includes('renormalises'), v.text);
const llamaAfter = v.after;
const tableAfter = await page.$$eval(`${P} tr[data-row="after"] td`, (t) => t.map((x) => Number(x.textContent)));
check('the table\'s "logit after" row agrees with the bars to 3 decimals', tableAfter.every((x, i) => Math.abs(x - M.after[i]) < 0.0006), tableAfter.join(' '));
if (out) await page.screenshot({ path: out + '-penalties.png' });

// ---------------------------------------------------------------- the three sign rules
check("rule button (transformers, mlx-lm)", await clickIn(P, 'Repetition rule', 'transformers'));
S = { ...S, rule: 'hf' };
v = await shown();
M = await model(v.context, S);
check("transformers' rule matches the model", same(v.after, M.after));
check("on this row transformers' rule and llama.cpp's give the same logits (they differ only at exactly 0, where both give 0)", same(v.after, llamaAfter));
check('rule button (CTRL)', await clickIn(P, 'Repetition rule', 'CTRL'));
S = { ...S, rule: 'ctrl' };
v = await shown();
M = await model(v.context, S);
check("CTRL's rule matches the model", same(v.after, M.after));
check("under CTRL's rule token 1 (logit -2.0, used 4 times) becomes -1.6: raised", v.counts[1] === 4 && Math.abs(v.after[1] - -1.6) < 1e-6 && v.raised.includes(1) && same(v.raised, M.raised), `${v.after[1]} raised=${v.raised}`);
check('the readout names every raised token (1, 4, 6: the repeated negative logits)', same(v.raised, [1, 4, 6]) && v.text.includes('higher') && v.text.includes('1, 4, 6'), v.text);
const probs1 = await page.$$eval(`${P} tr[data-row="p-before"] td, ${P} tr[data-row="p-after"] td`, (t) => t.map((x) => Number(x.textContent)));
check("and token 1's probability rises under CTRL's rule", probs1[12 + 1] > probs1[1], `${probs1[1]} -> ${probs1[13]}`);
await setSlider('repetition', 200);
S = { ...S, repeat: 2 };
const before1 = (await model(v.context, { ...S, repeat: 1.25 })).after[1];
v = await shown();
M = await model(v.context, S);
check('raising θ to 2 under CTRL raises token 1 further (-2.0 / 2 = -1.0)', same(v.after, M.after) && v.after[1] === -1 && v.after[1] > before1);
check('rule button (llama.cpp)', await clickIn(P, 'Repetition rule', 'llama.cpp'));
S = { ...S, rule: 'llama' };

// ---------------------------------------------------------------- frequency and presence
await setSlider('repetition', 100);
await setSlider('presence', 50);
S = { ...S, repeat: 1, presence: 0.5 };
v = await shown();
M = await model(v.context, S);
const base = M.before;
check('presence 0.5 alone matches the model', same(v.after, M.after));
check('presence moves every repeated token by 0.5, once, whatever its count', v.counts.every((c, i) => (c > 0 ? Math.abs(base[i] - v.after[i] - 0.5) < 1e-6 : v.after[i] === base[i])));
await setSlider('presence', 0);
await setSlider('frequency', 20);
S = { ...S, presence: 0, freq: 0.2 };
v = await shown();
M = await model(v.context, S);
check('frequency 0.2 alone matches the model', same(v.after, M.after));
check('frequency moves each token by 0.2 per occurrence (token 3, used 5 times, by 1.0)', v.counts.every((c, i) => Math.abs(base[i] - v.after[i] - 0.2 * c) < 1e-6) && v.counts[3] === 5, v.counts.join(' '));

// ---------------------------------------------------------------- the order
await setSlider('repetition', 125);
await setSlider('presence', 30);
S = { ...S, repeat: 1.25, presence: 0.3 };
v = await shown();
M = await model(v.context, S);
check('all three penalties, repetition first, match the model', same(v.after, M.after));
const first = v.after;
check('order button (subtract, then repetition)', await clickIn(P, 'Penalty order', 'subtract'));
S = { ...S, order: 'subtract-first' };
v = await shown();
M = await model(v.context, S);
check('subtract first matches the model', same(v.after, M.after));
check('swapping the order changes every repeated logit and no other', v.counts.every((c, i) => (c > 0 ? v.after[i] !== first[i] : v.after[i] === first[i])));
await clickIn(P, 'Penalty order', 'repetition, then');
S = { ...S, order: 'repeat-first' };

// ---------------------------------------------------------------- the context: click, append, reset
await page.$eval(`${P} g[data-token="11"]`, (g) => g.dispatchEvent(new MouseEvent('click', { bubbles: true })));
await pause(80);
v = await shown();
M = await model(v.context, S);
check('clicking column 11 adds token 11 to the context', v.length === 21 && v.context[20] === 11 && v.counts[11] === 1 && same(v.after, M.after), v.context.join(' '));
const next = M.next;
const label = await page.evaluate((P) => [...document.querySelectorAll(`${P} button`)].find((b) => b.textContent.startsWith('Append')).textContent, P);
check("the append button names the model's most likely token", label.includes(`(${next})`), label);
check('append button', await clickButton(P, 'Append the most likely token'));
v = await shown();
M = await model(v.context, S);
check('appending adds that token, one decode step', v.length === 22 && v.context[21] === next && same(v.counts, M.counts) && same(v.after, M.after), v.context.join(' '));
check('reset button', await clickButton(P, 'Reset the context'));
v = await shown();
check('reset restores the twenty-token context', v.length === 20 && same(v.context, M.context0));

// ---------------------------------------------------------------- who reads which logit
const access = () => page.$eval(`${A} [data-part="access"]`, (t) => ({ tx: Number(t.dataset.transactions), active: Number(t.dataset.active), text: t.textContent.replace(/\s+/g, ' ') }));
const accessModel = (mapping, batch, width, model, metalSize) =>
  page.evaluate(async (mapping, batch, width, model, metalSize) => {
    const m = await import('/src/sim/penalties.ts');
    const seg = m.segmentFor(model, metalSize);
    const a = m.mappingAccess(mapping, { batch, vocab: m.QWEN3_VOCAB, warpWidth: width, blockDim: 256 }, seg);
    return { tx: a.transactions, active: a.lanes.length, seg, eff: a.efficiency };
  }, mapping, batch, width, model, metalSize);
const MAP_LABEL = { element: 'one thread per element', block: 'one block per row', row: 'one thread per row' };
const MODEL_LABEL = { nvidia: 'NVIDIA', amd64: 'AMD: 64', amd128: 'AMD: 128', metal: 'Apple' };
let allMatch = true;
for (const width of [32, 64]) {
  check(`warp width ${width} button`, await clickIn(A, 'Warp width', String(width)));
  for (const segModel of ['nvidia', 'amd64', 'amd128']) {
    await clickIn(A, 'Segment model', MODEL_LABEL[segModel]);
    for (const batch of [1, 8, 64]) {
      await clickIn(A, 'Batch', String(batch));
      for (const mapping of ['element', 'block', 'row']) {
        await clickIn(A, 'Mapping', MAP_LABEL[mapping]);
        const got = await access();
        const exp = await accessModel(mapping, batch, width, segModel, 64);
        if (got.tx !== exp.tx || got.active !== exp.active) {
          allMatch = false;
          console.log(`  mismatch: ${mapping} batch ${batch} width ${width} ${segModel}: shown ${got.tx}/${got.active}, model ${exp.tx}/${exp.active}`);
        }
      }
    }
  }
}
check('every mapping, batch, width and vendor segment model shows the model\'s segment count and active lanes', allMatch);

await clickIn(A, 'Warp width', '32');
await clickIn(A, 'Segment model', 'NVIDIA');
await clickIn(A, 'Batch', '64');
await clickIn(A, 'Mapping', MAP_LABEL.element);
let a = await access();
check('one thread per element, width 32, NVIDIA: 32 lanes, 4 sectors, all useful', a.active === 32 && a.tx === 4 && a.text.includes('100%'), a.text);
await clickIn(A, 'Mapping', MAP_LABEL.block);
a = await access();
check('one block per row is as coalesced: 4 sectors', a.tx === 4 && a.text.includes('100%'), a.text);
await clickIn(A, 'Mapping', MAP_LABEL.row);
a = await access();
check('one thread per row at batch 64, width 32: one sector per lane, 32 of them', a.active === 32 && a.tx === 32, a.text);
if (out) await page.screenshot({ path: out + '-access.png' });
await clickIn(A, 'Batch', '1');
a = await access();
check('one thread per row at batch 1: one active lane in the whole launch', a.active === 1 && a.tx === 1 && a.text.includes('threads with any work in the launch1'), a.text);

check('segment model button (Apple: not documented)', await clickIn(A, 'Segment model', 'Apple'));
const metalSizes = await page.$$eval(`${A} [aria-label="Metal segment size"] button`, (b) => b.map((x) => x.textContent.trim()));
check('Metal asks the reader for a segment size', metalSizes.join('|') === '32 B|64 B|128 B', metalSizes.join('|'));
await clickIn(A, 'Batch', '64');
await clickIn(A, 'Mapping', MAP_LABEL.element);
for (const size of [32, 64, 128]) {
  await clickIn(A, 'Metal segment size', `${size} B`);
  const got = await access();
  const exp = await accessModel('element', 64, 32, 'metal', size);
  check(`Metal at the reader's ${size} B: the model's count (${exp.tx})`, got.tx === exp.tx && exp.seg === size, got.text);
}
await clickIn(A, 'Segment model', 'NVIDIA');
check('the Metal size choice disappears for a vendor model', (await page.$(`${A} [aria-label="Metal segment size"]`)) === null);

// ---------------------------------------------------------------- the listing
const tabs = await page.$$eval('figure.code .tabs button', (b) => b.map((x) => x.textContent));
check('the listing has CUDA and Metal tabs', tabs.includes('CUDA') && tabs.includes('Metal kernel'), tabs.join('|'));
const code = await page.$eval('figure.code pre', (e) => e.textContent);
check("the CUDA tab is lab 9.3's applyPenalties and its launcher", code.includes('__global__ void applyPenalties') && code.includes('void launchApplyPenalties'));
check('the CUDA tab states its status in the course vocabulary', /emulator-checked|unverified|verified on/.test(await page.$eval('figure.code .status', (e) => e.textContent)));

// ---------------------------------------------------------------- the Predict and the questions
const qHandle = (text) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(t)), text);
const pq = await qHandle('What does llama.cpp\'s rule make it');
if (await pq.evaluate((e) => !!e.querySelector('input[type=number]'))) {
  await (await pq.$('input[type=number]')).type('-2.5');
  await pq.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.includes('Commit')).click());
  await pause(150);
}
const reveal = await page.$eval('[data-reveal="rep-neg"]', (e) => e.textContent);
check('the Predict reveals -2.5, and -1.6 under CTRL', reveal.includes('−2.5') && reveal.includes('−1.6') && reveal.includes('as you said'), reveal);

async function answer(text, how, numeric = false) {
  const q = await qHandle(text);
  if (!numeric) await q.evaluate((e, i) => e.querySelectorAll('input[type=radio]')[i].click(), how);
  else {
    const input = await q.$('input[type=number]');
    await input.click({ clickCount: 3 });
    await input.type(String(how));
  }
  await pause(50);
  await q.evaluate((e) => [...e.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Check').click());
  await pause(120);
  return q.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
}
check('q-append: one thread owns each row', (await answer('appendTokens', 0)).startsWith('Correct.'));
check('q-append: "the GPU serialises writes" is wrong', (await answer('appendTokens', 2)).startsWith('Not quite.'));
check('q-zero: no, both give 0', (await answer('logit is exactly 0.0', 0)).startsWith('Correct.'));
check('q-freq: 0.4', (await answer('Logit 1.2, count 3', 0.4, true)).startsWith('Correct.'));
check('q-order: 0.6', (await answer('Logit 2.0, count 2', 0.6, true)).startsWith('Correct.'));
check('q-order: 0.8 (the reverse order) is wrong', (await answer('Logit 2.0, count 2', 0.8, true)).startsWith('Not quite.'));
check('q-hf-counts: prompt included, no frequency or presence', (await answer("transformers' penalties", 0)).startsWith('Correct.'));
check('q-bytes: 24', (await answer('Minimum bytes moved per logit for the pair', 24, true)).startsWith('Correct.'));

const todays = await page.$$eval('aside.note.today', (n) => n.map((x) => x.textContent));
check('every Today note was checked in the facts pass', todays.length >= 3 && todays.every((x) => x.includes('Checked 2026-11-20 in the facts pass')), String(todays.length));
const sources = await page.$$eval('aside.note', (n) => n.map((x) => x.textContent).filter((t) => t.startsWith('Book')));
check('the page cites chapter 9 §9.2 and §9.4', sources.some((x) => x.includes('9.2')) && sources.some((x) => x.includes('9.4')), sources.join(' | '));
check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
