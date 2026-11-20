// Browser check for section X3.8: the header decoder, the size prediction, the C++ check,
// the checkpoint explorer (every preset and a typed header) and three questions. Expected
// values come from src/sim/ckptformat.ts, run here in Node.
// Usage: APP_URL=http://localhost:5173 node e2e/x3-sx38.mjs [--dark] [screenshot-prefix]
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
const M = await import('../src/sim/ckptformat.ts');

const dark = process.argv.includes('--dark');
const out = process.argv.slice(2).find((a) => !a.startsWith('--'));
const { browser, page, errors } = await open({ width: 1440, height: 1000, dark });
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
const bench = (title) => `section.bench[aria-label="${title}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' ').trim());
const attr = (sel, a) => page.$eval(sel, (e, a) => e.getAttribute(a), a);
const clickText = async (scope, label) => {
  const ok = await page.evaluate((scope, label) => {
    const btn = [...document.querySelectorAll(scope + ' button')].find((b) => b.textContent.trim() === label);
    if (!btn || btn.disabled) return false;
    btn.click();
    return true;
  }, scope, label);
  await pause(80);
  return ok;
};
// Set a React-controlled input through the native setter, as typing would.
const setInput = async (sel, value) => {
  await page.$eval(sel, (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(80);
};
const n0 = (x) => x.toLocaleString('en-US');

await goto(page, 'x3/X3.8');
const D = bench('Decode a header');
const E = bench('Checkpoint explorer');
await page.waitForSelector(D);
await page.waitForSelector(E);

// ---------------------------------------------------------------- the decoder
const H = M.STORIES_15M;
const dump = await page.$$eval(D + ' [data-field]', (els) => els.map((e) => e.textContent));
const bytes = M.hex(M.encodeHeader(H));
check('the dump shows the 28 bytes the model encodes', M.HEADER_FIELDS.every((f, i) => dump[i].includes(bytes.slice(4 * i, 4 * i + 4).join(' '))), dump.join(' | '));
await setInput(D + ' input[aria-label="your int32"]', 288);
await clickText(D, 'Reveal');
let says = await text(D + ' [data-part="decode-says"]');
check('field 1: a right guess is confirmed', says.includes('= 288') && says.includes('You had it'), says);
check('field 1: the dump shows the value', (await attr(D + ' [data-field="dim"]', 'data-shown')) === '288');
await clickText(D, 'Next field');
await setInput(D + ' input[aria-label="your int32"]', 3);
await clickText(D, 'Reveal');
says = await text(D + ' [data-part="decode-says"]');
check('field 2: a wrong guess is shown against 768', says.includes('= 768') && says.includes('You wrote 3'), says);
await clickText(D, 'Next field');
for (let i = 2; i < 7; i++) {
  await clickText(D, 'Reveal');
  await clickText(D, i < 6 ? 'Next field' : 'Finish');
}
const shown = await page.$$eval(D + ' [data-field]', (els) => els.map((e) => Number(e.dataset.shown)));
check('all seven fields decode to the model header', M.HEADER_FIELDS.every((f, i) => shown[i] === H[f]), shown.join(','));
says = await text(D + ' [data-part="decode-done"]');
check('the summary gives head size 48 and the score', says.includes('= 48') && says.includes('1 of 7 by you'), says);
if (out) await page.screenshot({ path: out + '-decoder.png' });

// ---------------------------------------------------------------- the size prediction
const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('how many bytes long is stories15M.bin')));
await (await pred.$('input[aria-label="your prediction"]')).type(String(M.fileBytes(H)));
await (await pred.$('button.primary')).click();
await pause(150);
const reveal = await pred.evaluate((e) => e.querySelector('[data-reveal]')?.dataset.reveal);
check('the size prediction reveals the E1 size', Number(reveal) === 60816028 && M.fileBytes(H) === 60816028, reveal);
check('an exact prediction is called exact', (await pred.evaluate((e) => e.textContent)).includes('Exactly right.'));

// ---------------------------------------------------------------- the C++ check
const code = await page.$$eval('.code', (els) => els.map((e) => e.textContent));
check('the C++ check is the snippet file', code.some((c) => c.includes('ckpt_check') && c.includes('expected %lld bytes, file has %lld')), code.length);
check('its status names the unit test', code.some((c) => c.includes('ckptformat.test.ts')));

// ---------------------------------------------------------------- the explorer
const pick = (label) => clickText(E + ' .seg', label);
const fit = async () => ({
  head: Number(await attr(E + ' [data-part="fit"]', 'data-head')),
  m7: await attr(E + ' [data-part="fit"]', 'data-m7'),
  lanes: await attr(E + ' [data-part="fit"]', 'data-lanes'),
});
const todos = () => page.$$eval(E + ' [data-part="checklist"] li[data-todo="yes"]', (els) => els.map((e) => e.dataset.item));
const want = (arch) => M.checklist(arch).filter((i) => i.todo).map((i) => i.id).join(',');

for (const c of M.CHECKPOINTS) {
  await pick(c.label);
  const bytesShown = Number(await attr(E + ' [data-part="size"]', 'data-bytes'));
  const expect = c.format === 'llama2c' ? M.fileBytes(c.header) : M.safetensorsBytes(c.st.headerLen, c.st.tensors);
  check(`${c.label}: size by formula ${n0(expect)}`, bytesShown === expect, String(bytesShown));
  if (c.hubBytes != null) check(`${c.label}: agrees with the Hub to the byte`, (await text(E + ' [data-part="size"]')).includes("the Hub's size, to the byte"));
  const f = await fit();
  const mf = M.fitOf(c.arch);
  check(`${c.label}: head ${mf.headSize}, M7 ${mf.m7 ? 'fits' : 'does not fit'}`, f.head === mf.headSize && f.m7 === (mf.m7 ? 'yes' : 'no') && f.lanes === String(mf.perLane ?? ''), JSON.stringify(f));
  const t = (await todos()).join(',');
  check(`${c.label}: checklist ${want(c.arch) || 'empty'}`, t === want(c.arch), t);
}
const st = await attr(E + ' [data-part="size"]', 'data-tensors'); // the last preset is SmolLM2
check('SmolLM2: 272 tensors, first bytes 40 77 00 ...', st === '272' && (await text(E + ' [data-part="first-bytes"]')).includes('40 77 00 00 00 00 00 00 7B'));
if (out) await page.screenshot({ path: out + '-smol.png' });

// A download check on stories260K.
await pick('stories260K');
await setInput(E + ' input[aria-label="download size"]', 1056540);
check('the right download size matches', (await attr(E + ' [data-part="download"]', 'data-ok')) === 'yes');
await setInput(E + ' input[aria-label="download size"]', 1056540 - 4096);
const dl = await text(E + ' [data-part="download"]');
check('a truncated download is caught with its shortfall', (await attr(E + ' [data-part="download"]', 'data-ok')) === 'no' && dl.includes('4,096 bytes short'), dl);

// A typed header.
await pick('your own header');
await setInput(E + ' input[aria-label="dim"]', 100);
check('a dim that does not split into heads is refused', (await text(E + ' [data-part="problem"]')).includes('not a multiple of n_heads'));
const llama = { dim: 4096, hidden_dim: 11008, n_layers: 2, n_heads: 32, n_kv_heads: 8, vocab_size: 32000, seq_len: 64 };
for (const [k, v] of Object.entries(llama)) await setInput(E + ` input[aria-label="${k}"]`, v);
check('a typed header gets the E1 size', Number(await attr(E + ' [data-part="size"]', 'data-bytes')) === M.fileBytes(llama));
const f = await fit();
check('head size 128 fits M7, 4 per lane', f.head === 128 && f.m7 === 'yes' && f.lanes === '4', JSON.stringify(f));
await setInput(E + ' input[aria-label="vocab_size"]', -32000);
check('a negative vocab_size adds the wcls block', !!(await page.$(E + ' tr[data-block="wcls"]')) && Number(await attr(E + ' [data-part="size"]', 'data-bytes')) === M.fileBytes({ ...llama, vocab_size: -32000 }));
if (out) await page.screenshot({ path: out + '-custom.png' });

// ---------------------------------------------------------------- questions
const answerNumeric = async (needle, v) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await (await q.$('input[type=number]')).type(String(v));
  await (await q.$('button.primary')).click();
  await pause(120);
  return (await q.evaluate((e) => e.textContent)).includes('Correct.');
};
const answerChoice = async (needle, idx) => {
  const q = await page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
  await q.evaluate((e, i) => e.querySelectorAll('input[type=radio]')[i].click(), idx);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return (await q.evaluate((e) => e.textContent)).includes('Correct.');
};
check('q-size-260k accepts 1,056,540', await answerNumeric('How many bytes should the file have', 1056540));
check('q-st-n accepts 30,536', await answerNumeric('At what byte offset in the file', 30536));
check('q-head64: SmolLM2 is the one', await answerChoice('by changing only HEAD_DIM', 0));
check('q-bf16: a wrong option is marked wrong', !(await answerChoice('stores the 16 bits 0x3F80', 1)));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
