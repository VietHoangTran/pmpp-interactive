// Browser check for appendix C, section C.1: the "Which window?" explorer, the
// eight compile-and-memory cards under both rule sets, the two snippets and the
// questions.
// Usage: node e2e/appC-sc1.mjs [screenshot-prefix]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { open, goto } from './browser.mjs';
import { CARDS, VARIABLES, classify, judge } from '../src/sim/addrspace.ts';

const out = process.argv[2];
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
let puts = 0;
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (r.method() === 'PUT' && r.url().includes('/api/progress')) {
    puts++;
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
const WIN = 'section.bench[aria-label="Which window?"]';
const CARDB = 'section.bench[aria-label="Does it compile, and which memory does it read?"]';
const click = async (sel) => {
  const ok = await page.evaluate((sel) => {
    const e = document.querySelector(sel);
    if (!e || e.disabled) return false;
    e.click();
    return true;
  }, sel);
  await pause(60);
  return ok;
};
const setRange = async (sel, v) => {
  await page.evaluate(
    (sel, v) => {
      const e = document.querySelector(sel);
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(e, String(v));
      e.dispatchEvent(new Event('input', { bubbles: true }));
    },
    sel,
    v,
  );
  await pause(60);
};
const question = (part) => page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(part)), part);
const qText = (q) => q.evaluate((e) => e.textContent);
const pick = async (q, label) => {
  await q.evaluate((e, label) => {
    [...e.querySelectorAll('.opts label')].find((l) => l.textContent.trim().startsWith(label)).querySelector('input').click();
  }, label);
  await (await q.$('button.primary')).click();
  await pause(120);
};
const typeAnswer = async (q, v) => {
  const input = await q.$('input[type=number]');
  await input.focus();
  await input.evaluate((e) => e.select());
  await input.type(String(v));
  await (await q.$('button.primary')).click();
  await pause(120);
};
const readout = () =>
  page.$eval(WIN + ' [data-window]', (n) => ({ window: n.dataset.window, state: n.querySelector('[data-state]').textContent, load: n.querySelector('[data-load]').textContent, text: n.textContent }));

await goto(page, 'appC/C.1');
await page.waitForSelector(WIN);

// ---- "Which window?"
let allVars = true;
for (const v of VARIABLES) {
  await click(`${WIN} button[data-var="${v.id}"]`);
  const r = await readout();
  const k = classify(v.addr);
  if (r.window !== k.window || r.state !== k.state || r.load !== k.load || !r.text.includes(v.code)) allVars = false;
}
check('each variable lands in its window, with the state space and load the model gives', allVars);
for (const a of [0x0400, 0xa100, 0xd800, 0xfff0]) {
  await setRange(WIN + ' input[type=range]', a);
  const r = await readout();
  const k = classify(a);
  check(`address 0x${a.toString(16)}: ${k.window}, ${k.load}`, r.window === k.window && r.load === k.load, JSON.stringify(r));
}
const chip = await page.$eval(WIN + ' [data-window] .chip', (c) => c.className);
check('the readout chip carries the memory-tier colour (global outside the windows)', chip.includes('global'), chip);
// Local memory takes the global colour, as in chapter 5, with its own outline and label.
await click(`${WIN} button[data-var="local"]`);
const localChip = await page.$eval(WIN + ' [data-window] .chip', (c) => c.className);
const localText = (await readout()).text;
check('a local address: the chip is global-coloured and the readout says local memory lives in global memory', localChip.includes('global') && !localChip.includes('reg') && localText.includes('lives in global memory'), localChip + ' | ' + localText);
const localRect = await page.$eval(WIN + ' [data-window-rect="local"]', (g) => {
  const r = g.querySelector('rect');
  return { stroke: r.getAttribute('stroke'), fill: r.getAttribute('fill'), dash: r.getAttribute('stroke-dasharray'), label: g.textContent };
});
check('the local window is drawn in the global colour, dashed, labelled "in global memory"', localRect.stroke === 'var(--global)' && localRect.fill === 'var(--global-wash)' && !!localRect.dash && localRect.label.includes('in global memory'), JSON.stringify(localRect));
check('nothing in the strip uses the register colour', (await page.$eval(WIN + ' svg', (s) => s.innerHTML)).includes('--reg') === false);
// Click on the strip: the address follows the click.
await page.$eval(WIN + ' svg', (s) => s.scrollIntoView({ block: 'center' }));
await pause(80);
const box = await page.$eval(WIN + ' svg', (s) => {
  const b = s.getBoundingClientRect();
  return { x: b.left, y: b.top, w: b.width, h: b.height };
});
await page.mouse.click(box.x + box.w * (12 + ((0xd400 / 0x10000) * (640 - 24))) / 640, box.y + box.h * 0.35);
await pause(80);
let r = await readout();
check('clicking the strip inside the shared window moves the address there', r.window === 'shared', JSON.stringify(r));
if (out) await page.screenshot({ path: out + '-window.png' });
let q = await question('What makes the first call');
await pick(q, 'The address itself');
check('q-who-decides', (await qText(q)).includes('Correct.'));

// ---- cards
const bins = async () => page.$$eval(CARDB + ' [data-card]', (cs) => cs.map((c) => ({ id: c.dataset.card, result: c.dataset.result })));
const score = () => page.$eval(CARDB + ' [data-score]', (e) => e.textContent.trim());
check('eight cards, none sorted yet', (await bins()).length === CARDS.length && (await score()) === '0 of 8 sorted correctly');
// One wrong pick first: the feedback names the rule.
await click(`${CARDB} [data-card="msl-none"] button[data-bin="shared"]`);
let fb = await page.$eval(`${CARDB} [data-card="msl-none"]`, (c) => ({ r: c.dataset.result, t: c.querySelector('[data-feedback]').textContent }));
check('a wrong bin is marked and the feedback gives the rule and its source', fb.r === 'wrong' && fb.t.includes('Every pointer must carry an address space') && fb.t.includes('C1'), JSON.stringify(fb));
for (const c of CARDS) await click(`${CARDB} [data-card="${c.id}"] button[data-bin="${judge(c, 'spec').bin}"]`);
check('sorted by the specification: all eight right', (await bins()).every((b) => b.result === 'right') && (await score()) === '8 of 8 sorted correctly', await score());
fb = await page.$eval(`${CARDB} [data-card="msl-device-only"] [data-feedback]`, (e) => e.textContent);
check('the device-only helper given a tile: "no matching function"', fb.includes('no matching function for call'), fb);
if (out) await page.screenshot({ path: out + '-cards.png' });
await click(`${CARDB} button[data-rules="observed-4.1"]`);
const moved = (await bins()).filter((b) => b.result === 'wrong').map((b) => b.id);
const want = CARDS.filter((c) => judge(c, 'observed-4.1').bin !== judge(c, 'spec').bin).map((c) => c.id);
check('switching to the compiler observed here moves exactly the cards the model says', JSON.stringify(moved) === JSON.stringify(want) && want.join() === 'msl-none', JSON.stringify({ moved, want }));
fb = await page.$eval(`${CARDB} [data-card="msl-none"] [data-feedback]`, (e) => e.textContent);
check('the moved card says it is observed, not documented', fb.includes('Observed, not documented') && fb.includes('MSL 4.1'), fb);
await click(`${CARDB} [data-card="msl-none"] button[data-bin="shared"]`);
check('re-sorted under the observed rules: 8 of 8', (await score()) === '8 of 8 sorted correctly', await score());
await click(`${CARDB} button[data-rules="spec"]`);
check('and back under the specification it is wrong again: 7 of 8', (await score()) === '7 of 8 sorted correctly', await score());

q = await question('smallest number of function definitions');
await typeAnswer(q, 2);
check('q-overloads: 2', (await qText(q)).includes('Correct.'));
q = await question('MSL, compiled with');
await pick(q, 'It compiles, and p becomes generic');
check('q-missing-space: the generic distractor explains the 4.1 observation', (await qText(q)).includes('observed, not documented'));
await pick(q, 'A compile error');
check('q-missing-space: compile error', (await qText(q)).includes('Correct.'));

// ---- code
const tabs = await page.$$eval('.code .tabs button', (b) => b.map((x) => x.textContent));
check('the listing has a CUDA-and-HIP tab and a Metal tab', tabs.join('|') === 'CUDA and HIP|Metal', tabs.join('|'));
const shownLines = () => page.$$eval('.code pre .ln', (ls) => ls.map((l) => l.textContent.replace(/\n$/, '')));
const cu = fs.readFileSync(path.join(repo, 'labs/appC/snippets/c1_helper_generic.cu'), 'utf8').trimEnd().split('\n');
check('the CUDA tab is labs/appC/snippets/c1_helper_generic.cu, line for line', JSON.stringify(await shownLines()) === JSON.stringify(cu));
check('its status is front-end-checked', (await page.$eval('.code .tabs .status', (s) => s.textContent)) === 'front-end-checked');
const pColoured = await page.$$eval('.code pre span[class^="v-"]', (s) => s.filter((x) => x.textContent === 'p').length);
check('p carries no memory colour; tile is shared', pColoured === 0 && (await page.$$eval('.code pre .v-shared', (s) => s.some((x) => x.textContent === 'tile'))));
await page.evaluate(() => [...document.querySelectorAll('.code .tabs button')].find((b) => b.textContent === 'Metal').click());
await pause(80);
const metal = fs.readFileSync(path.join(repo, 'labs/appC/snippets/c1_helper_overloads.metal'), 'utf8').trimEnd().split('\n');
check('the Metal tab is labs/appC/snippets/c1_helper_overloads.metal, line for line', JSON.stringify(await shownLines()) === JSON.stringify(metal));
check('its status is verified on Metal', (await page.$eval('.code .tabs .status', (s) => s.textContent)) === 'verified on Metal');

q = await question('Same time?');
await pick(q, 'No. It is one instruction stream');
check('q-speed', (await qText(q)).includes('Correct.'));

// ---- notes and voice
const today = await page.$$eval('.note.today', (n) => n.map((x) => x.textContent).join(' '));
check('the Today note gives PTX generic addressing, cvta, isspacep and March 2010', ['generic addressing', 'cvta', 'isspacep', 'March 2010'].every((s) => today.includes(s)));
const beyond = await page.$$eval('.note.beyond', (n) => n.map((x) => x.textContent).join(' '));
check('HIP is sourced to the LLVM AMDGPU documentation, and the MSL 4.1 observation is labelled', beyond.includes('LLVM AMDGPU') && beyond.includes('Observed on the build machine'));
check('every margin note directly follows text', await page.evaluate(() => [...document.querySelectorAll('.prose > .note')].every((n) => ['P', 'UL', 'OL'].includes(n.previousElementSibling?.tagName) || n.previousElementSibling?.classList.contains('note'))));
const prose = await page.$eval('.prose', (e) => e.textContent);
check('no em dash in the page', !prose.includes('—'));
check('answers went to the progress API (intercepted)', puts > 0);
check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
