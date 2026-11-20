// Browser check for section 5.4: the kernel stepper and the barrier outcome table.
// Usage: node e2e/s54.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';

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
const bench = (scope) => `section.bench[aria-label="${scope}"]`;
const text = (sel) => page.$eval(sel, (e) => e.textContent);
const clickText = async (scope, label) => {
  const done = await page.evaluate(
    (scope, label) => {
      const root = [...document.querySelectorAll('section.bench')].find((b) => b.getAttribute('aria-label') === scope);
      const btn = [...root.querySelectorAll('button')].find((b) => b.textContent.trim() === label);
      if (!btn || btn.disabled) return false;
      btn.click();
      return true;
    },
    scope,
    label,
  );
  await pause(60);
  return done;
};

await goto(page, 'ch05/5.4');
await page.waitForSelector(bench('Step through one block'));
await page.waitForSelector('.code .tabs button');

// The real lab solution is shown, with three tabs.
const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent));
check('listing shows CUDA, Metal kernel and Metal host tabs', ['CUDA', 'Metal kernel', 'Metal host'].every((t) => tabs.includes(t)), tabs.join('|'));
check('CUDA tab is the tested lab file', (await text('.code pre')).includes('tiledMatmulKernel<<<dimGrid, dimBlock>>>'));

// Stepper: thread (0,0) in the default 4x4 / 2x2 problem, "runs as far as it can" order.
const S = 'Step through one block';
await clickText(S, 'Step');
let n = await text(bench(S) + ' .narrate');
check('step 1 is the M load with its index arithmetic', n.includes('loads M[0][0] into Mds[0][0]') && n.includes('= 0.'), n);
await clickText(S, 'Step');
n = await text(bench(S) + ' .narrate');
check('step 2 is the N load', n.includes('loads N[0][0] into Nds[0][0]'), n);
await clickText(S, 'Step');
n = await text(bench(S) + ' .narrate');
check('step 3: the thread stops at barrier 1', n.includes('reaches barrier 1'), n);
check('highlighted source line follows the step', (await text(bench(S) + ' .code .ln.hl')).includes('barrier 1'));
await clickText(S, 'Run to next barrier');
n = await text(bench(S) + ' .narrate');
check('run to next barrier ends on the barrier opening', n.startsWith('Barrier 1 opens'), n);
for (let i = 0; i < 12; i++) await clickText(S, 'Run to next barrier');
check('correct kernel: block results match', (await text(bench(S))).includes("This block's results match M × N."));
check('correct kernel: no failure to jump to', !(await text(bench(S))).includes('Go to the first failure'));
if (out) await page.screenshot({ path: out + '-stepper-end.png' });

// Remove barrier 2: a write-after-read failure appears and can be jumped to.
await page.evaluate((sel) => {
  [...document.querySelectorAll(sel + ' label.check')].find((l) => l.textContent.includes('barrier 2')).querySelector('input').click();
}, bench(S));
await pause(120);
check('removing barrier 2 offers a failure to jump to', await clickText(S, 'Go to the first failure'));
n = await text(bench(S) + ' .narrate');
check('the failure is an overwrite before read', n.includes('had already overwritten it for phase 1'), n);
check('the failing slot is marked', (await page.$$(bench(S) + ' .mx.shared span.bad')).length >= 1);
if (out) await page.screenshot({ path: out + '-stepper-fail.png' });
for (let i = 0; i < 40; i++) await clickText(S, 'Run to next barrier');
check('without barrier 2 the block result is wrong', (await text(bench(S))).includes("This block's results are wrong."));

// Lockstep hides it.
await page.evaluate((sel) => {
  const s = [...document.querySelectorAll(sel + ' label.field')].find((l) => l.textContent.startsWith('Thread order')).querySelector('select');
  s.value = 'lockstep';
  s.dispatchEvent(new Event('change', { bubbles: true }));
}, bench(S));
await pause(120);
for (let i = 0; i < 60; i++) await clickText(S, 'Step');
check('in lockstep the same broken kernel gives the right answer', (await text(bench(S))).includes("This block's results match M × N."));

// Outcome table: 4 kernels x 4 orders.
const O = 'Every thread order at once';
const cells = await page.$$eval(bench(O) + ' tbody tr', (rows) =>
  rows.map((r) => [...r.querySelectorAll('td[data-outcome]')].map((c) => c.dataset.outcome)),
);
check('with both barriers every order is correct', cells[0].every((c) => c === 'correct'), JSON.stringify(cells[0]));
check('lockstep column is correct for every kernel', cells.every((r) => r[0] === 'correct'), JSON.stringify(cells.map((r) => r[0])));
check('each removed barrier fails under the three other orders', cells.slice(1).every((r) => r.slice(1).every((c) => c === 'wrong')), JSON.stringify(cells));

// A question.
const q = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('flat index into M')));
await (await q.$('input[type=number]')).type('11');
await (await q.$('button.primary')).click();
await pause(120);
check('numeric question accepts the right index', (await q.evaluate((e) => e.textContent)).includes('Correct.'));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
