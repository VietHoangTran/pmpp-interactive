// Browser check for section 11.1: the listing of s111_sequential.h, the "Cut the sausage"
// explorer (cut points, exclusive starts, the conversion, the token-count preset, a typed
// row), the p-crossing prediction, the "Where the nucleus ends" explorer (four rules on
// every preset, p, float32 in both orders, the row on which they disagree, the band), and
// two questions. Expected values come from src/sim/scan.ts, imported directly.
// Usage: node e2e/ch11-s111.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';
import { ALL_PRESETS, BOOK_ORDERS, FLIP, RULES, band, exclusive, exclusiveIn, f32, inclusive, toExclusive, toInclusive, topP } from '../src/sim/scan.ts';

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
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const bench = (t) => `section.bench[aria-label="${t}"]`;
const S = bench('Cut the sausage');
const Nu = bench('Where the nucleus ends');
const click = async (scope, label) => {
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
  await pause(80);
};
const typeInto = async (sel, text) => {
  await page.$eval(sel, (el, text) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, text);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, text);
  await pause(80);
};
const rowNums = (scope, row) => page.$$eval(`${scope} tr[data-row="${row}"] td`, (tds) => tds.map((t) => Number(t.textContent)));
const cuts = () => page.$$eval(S + ' [data-part="cut"]', (gs) => gs.map((g) => Number(g.dataset.at)));
const kept = () => page.$$eval(Nu + ' tr[data-rule]', (rs) => Object.fromEntries(rs.map((r) => [r.dataset.rule, Number(r.dataset.kept)])));
const model = (probs, p, arith) => Object.fromEntries(RULES.map((r) => [r, topP(probs, p, r, arith).kept]));

await goto(page, 'ch11/11.1');
await page.waitForSelector(S);

// ---- the listing
await page.waitForFunction(() => [...document.querySelectorAll('.code pre')].some((p) => p.textContent.includes('sequential_scan')));
const listing = await page.$$eval('.code', (cs) => cs.map((c) => ({ text: c.textContent, hl: [...c.querySelectorAll('.ln.hl')].map((l) => l.textContent) })).find((c) => c.text.includes('sequential_scan')));
check('the listing is labs/ch11/snippets/s111_sequential.h, with the loop of Fig. 11.1', listing.text.includes('output[i] = output[i - 1] + input[i];') && listing.text.includes('inclusive_to_exclusive'));
check('the loop of Fig. 11.1 is the highlighted part', listing.hl.length === 3 && listing.hl[2].includes('output[i] = output[i - 1] + input[i];'), JSON.stringify(listing.hl));

// ---- cut the sausage
check('the book\'s orders give cuts at the inclusive scan, 3 4 11 11 15 16 22 25', same(await cuts(), inclusive(BOOK_ORDERS)), JSON.stringify(await cuts()));
check('the table shows both scans from the model', same(await rowNums(S, 'inclusive'), inclusive(BOOK_ORDERS)) && same(await rowNums(S, 'exclusive'), exclusive(BOOK_ORDERS)));
let r = await page.$eval(S + ' [data-readout="sausage"]', (e) => e.textContent);
check('25 inches served, 15 left, 7 additions', r.includes('25 inches of the 40-inch sausage are served and 15 are left') && r.includes('7 additions'), r);
check('the two cuts at 11 share one label, "11 ×2"', (await page.$$eval(S + ' [data-part="cut"] text', (t) => t.map((x) => x.textContent))).includes('11 ×2'));
await click(S, 'exclusive: where pieces start');
check('exclusive marks the starts, 0 3 4 11 11 15 16 22', same(await cuts(), exclusive(BOOK_ORDERS)), JSON.stringify(await cuts()));
await page.$eval(S + ' label.check input', (e) => e.click());
await pause(80);
const derived = await page.$$eval(S + ' [data-part="derived"]', (gs) => gs.map((g) => Number(g.dataset.value)));
check('the conversion from exclusive shifts left and adds the last size: the inclusive scan', same(derived, toInclusive(exclusive(BOOK_ORDERS), BOOK_ORDERS)) && same(derived, inclusive(BOOK_ORDERS)), JSON.stringify(derived));
await click(S, 'inclusive: where pieces end');
const derived2 = await page.$$eval(S + ' [data-part="derived"]', (gs) => gs.map((g) => Number(g.dataset.value)));
check('the conversion from inclusive shifts right and puts 0 first', same(derived2, toExclusive(inclusive(BOOK_ORDERS))), JSON.stringify(derived2));
if (out) await page.screenshot({ path: out + '-sausage.png' });
await click(S, 'token counts of six sequences');
r = await page.$eval(S + ' [data-readout="sausage"]', (e) => e.textContent);
check('token counts: sequence 4 starts at slot 15, the buffer needs 21 slots', r.includes('starts at slot 15') && r.includes('needs 21 slots'), r);
await typeInto(S + ' input[type=text]', '2 0 5');
check('a typed row is scanned: 2 2 7 and 0 2 2', same(await rowNums(S, 'inclusive'), [2, 2, 7]) && same(await rowNums(S, 'exclusive'), [0, 2, 2]));
await typeInto(S + ' input[type=text]', '2 x 5');
check('a row that does not parse is flagged and the last good one kept', (await page.$eval(S, (e) => e.textContent)).includes('keeps the last list that parsed') && same(await rowNums(S, 'inclusive'), [2, 2, 7]));

// ---- the prediction
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('How many tokens does llama.cpp keep')));
await (await pred.$('input[type=number]')).type('2');
await (await pred.$('button.primary')).click();
await pause(150);
const reveal = await pred.evaluate((e) => e.textContent);
const d = ALL_PRESETS.find((p) => p.id === 'dyadic');
check('the reveal gives the model\'s count and masses', reveal.includes(`It keeps ${topP(d.probs, d.p, 'exclusive').kept}`) && reveal.includes(topP(d.probs, d.p, 'exclusive').before.join(', ')) && reveal.includes('You had it.'), reveal);

// ---- the nucleus explorer
for (const pr of ALL_PRESETS.filter((x) => x.id !== 'flip')) {
  await click(Nu, pr.label);
  await click(Nu, 'exact');
  check(`${pr.label}: the four rules keep what the model keeps at p = ${pr.p}`, same(await kept(), model(pr.probs, pr.p)), JSON.stringify([await kept(), model(pr.probs, pr.p)]));
}
await click(Nu, '0.5 0.25 0.125 0.125');
let k = await kept();
check('dyadic row at 0.75: the projects keep 2, the off-by-one 1', k.exclusive === 2 && k.inclusive === 1 && k.mlx === 2 && k.ascending === 2);
await click(Nu, 'four tied at 0.25');
k = await kept();
check('four tied at p = 0.5: mlx-lm keeps all 4, the others cut at 2', k.mlx === 4 && k.exclusive === 2 && k.ascending === 2);
await click(Nu, 'peaked');
await setRange(Nu + ' input[type=range]', 30);
k = await kept();
const peaked = ALL_PRESETS.find((p) => p.id === 'peaked');
check('peaked at p = 0.30: the off-by-one keeps nothing, the projects keep the top token', k.inclusive === 0 && k.exclusive === 1 && same(k, model(peaked.probs, 0.3)), JSON.stringify(k));
check('the readout says so', (await page.$eval(Nu + ' [data-readout="nucleus"]', (e) => e.textContent)).includes('none at all'));
const crossLabel = await page.$eval(Nu + ' [data-part="mass"]', (e) => e.textContent);
check('the drawing names the crossing token', crossLabel.includes('token 0 crosses p'), crossLabel);

// float32: the row on which the orders disagree
await click(Nu, 'two orders disagree');
const loopArith = { kind: 'f32', order: 'loop' };
k = await kept();
check('the flip row switches to float32, loop order: the projects\' rule keeps 6', k.exclusive === 6 && same(k, model(FLIP.probs, FLIP.p, loopArith)), JSON.stringify(k));
const row5 = await page.$eval(Nu + ` tr[data-token="${FLIP.token}"][data-disagree]`, (e) => ({ dis: e.dataset.disagree, text: e.textContent }));
check('token 5 is the row where the orders disagree: 0x3f599999 keeps, 0x3f59999a drops', row5.dis === '1' && row5.text.includes('0x3f599999') && row5.text.includes('0x3f59999a') && row5.text.includes('keep') && row5.text.includes('drop'), JSON.stringify(row5));
const dis = await page.$$eval(Nu + ' tr[data-disagree="1"]', (rs) => rs.map((x) => Number(x.dataset.token)));
const loop = exclusiveIn(FLIP.probs, 'loop');
const ks = exclusiveIn(FLIP.probs, 'kogge-stone');
check('and the only one, as the model computes', same(dis, loop.flatMap((v, i) => ((v < f32(FLIP.p)) !== (ks[i] < f32(FLIP.p)) ? [i] : []))), JSON.stringify(dis));
const verdicts = await page.$$eval(Nu + ' td[data-verdict]', (t) => t.map((x) => x.dataset.verdict));
check('the band column is the model\'s: "either" only for token 5', same(verdicts, band(FLIP.probs, FLIP.p).verdict), JSON.stringify(verdicts));
r = await page.$eval(Nu + ' [data-readout="orders"]', (e) => e.textContent);
check('the readout: -1 step in loop order, 0 in Kogge-Stone order, exactly p', r.includes('-1 steps from p in loop order and 0 in Kogge-Stone order') && r.includes('0.85: precisely p'), r);
await click(Nu, 'Kogge-Stone order');
k = await kept();
check('in Kogge-Stone order the projects\' rule keeps 5', k.exclusive === 5 && same(k, model(FLIP.probs, FLIP.p, { kind: 'f32', order: 'kogge-stone' })), JSON.stringify(k));
await click(Nu, 'exact');
k = await kept();
check('exactly, the mass before token 5 is p, so it is dropped: 5', k.exclusive === 5);
if (out) await page.screenshot({ path: out + '-nucleus.png' });

// ---- questions
const ask = async (text, value) => {
  const q = await page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(t)), text);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
check('q-offsets accepts 15', (await ask('At which slot does sequence 4', 15)).includes('Correct.'));
check('q-rule accepts 1', (await ask('keep a token while the', 1)).includes('Correct.'));
const choice = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('rather than with')));
await choice.evaluate((e) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes('leaves std::inclusive_scan')).querySelector('input').click());
await (await choice.$('button.primary')).click();
await pause(120);
check('q-cpu-ref accepts the order answer and quotes cppreference', (await choice.evaluate((e) => e.textContent)).includes('non-deterministic if the add_op is not associative'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
