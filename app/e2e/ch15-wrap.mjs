// Browser check for chapter 15, "Summary and exercises": what to carry forward, the five
// questions of the mixed check (numbers taken from the same models the page imports),
// the snippet shown verbatim from labs/ch15/snippets, the exercises table, the errata
// list, the project status read from /api/reports, links, and the page at 390 px.
// Usage: APP_URL=http://localhost:<port> node e2e/ch15-wrap.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';
import { intensity } from '../src/sim/gemmreuse.ts';
import { SCHEDULES, run } from '../src/sim/gemmpipe.ts';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
  return r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : JSON.stringify(EMPTY) });
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const norm = (s) => s.replace(/\s+/g, ' ').trim();
const shot = async (name, sel) => {
  if (!out) return;
  await page.$eval(sel, (e) => e.scrollIntoView({ block: 'start' }));
  await pause(150);
  await page.screenshot({ path: `${out}-${name}.png` });
};

await goto(page, 'ch15/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary and exercises" of chapter 15', h1.includes('Summary and exercises'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', 'The book’s exercises', 'Where the PDF looks wrong', 'What this course shortened', 'The project so far'];
check('sections in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
const text0 = norm(await page.$eval('.prose', (e) => e.textContent));
check('nothing on the page says it is being written', !/being written/i.test(text0));
const misplaced = await page.$$eval('.prose > .note', (ns) => ns.filter((n) => n.previousElementSibling?.tagName !== 'P').map((n) => n.textContent.slice(0, 60)));
check('every margin note follows a paragraph', misplaced.length === 0, misplaced.join(' | '));
const carry = await page.evaluate(() => {
  const h = [...document.querySelectorAll('.prose h2')].find((x) => x.textContent.includes('carry forward'));
  let e = h?.nextElementSibling;
  while (e && e.tagName !== 'UL') e = e.nextElementSibling;
  return e ? e.querySelectorAll('li').length : 0;
});
check('seven points to carry forward', carry === 7, String(carry));

// ---- the snippet, verbatim
const file = await page.evaluate(() => fetch('/api/lab-file?path=' + encodeURIComponent('ch15/snippets/wrap_load_first.cu')).then((r) => (r.ok ? r.text() : 'MISSING')));
const code = await page.$eval('figure.code pre', (e) => e.textContent);
check('the load-first kernel is labs/ch15/snippets/wrap_load_first.cu verbatim', file !== 'MISSING' && code.trim() === file.trim());
check('  its status is the front-end check only', (await page.$eval('figure.code .tabs .status', (e) => e.textContent)) === 'front-end-checked; never run');

// ---- the mixed check
const q = (part) => page.evaluateHandle((p) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt')?.textContent.includes(p)), part);
const answerNumber = async (part, value) => {
  const h = await q(part);
  await h.evaluate((e, v) => {
    const el = e.querySelector('input[type=number]');
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, value);
  await pause(60);
  await (await h.$('button.primary')).click();
  await pause(100);
  return h.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
};
const answerChoice = async (part, option) => {
  const h = await q(part);
  const ok = await h.evaluate((e, o) => {
    const l = [...e.querySelectorAll('.opts label')].find((x) => x.textContent.includes(o));
    if (!l) return false;
    l.querySelector('input').click();
    return true;
  }, option);
  await pause(60);
  await (await h.$('button.primary')).click();
  await pause(100);
  // The verdict, then the reason shown beside the picked option (Ask shows an option's `why` there).
  return ok && (await h.evaluate((e) => (e.querySelector('.verdict')?.textContent ?? '') + ' | ' + (e.querySelector('.opts label.wrong .muted')?.textContent ?? '')));
};
const fpb = Math.round(intensity(4, 128) * 100) / 100;
let v = await answerNumber('Speculative decoding checks 4 drafted tokens', 32);
check(`q-decode4: 32 is wrong, and the answer is the model's ${fpb}`, v.includes('Not quite.') && v.includes(String(fpb)), v);
v = await answerNumber('Speculative decoding checks 4 drafted tokens', fpb);
check('q-decode4: correct, with the 0.5·m bound and a link to 15.2', v.includes('Correct.') && v.includes('below 0.5·m, here 2') && v.includes('15.2'), v);
// Our own count, independent of src/sim/gemmbanks.ts: lane-row g of a 32-lane warp reads row
// 4g of its quadrant, word 4g·ld (+ a constant); lanes in one lane-row share a word.
const deg = (ld) => {
  const per = new Map();
  for (let g = 0; g < 8; g++) { const b = (4 * g * ld) % 32; per.set(b, (per.get(b) ?? 0) + 1); }
  return Math.max(...per.values());
};
v = await answerChoice('You double bK to 16', '24: half a row more');
check(`q-bk16: 24 is wrong, with the model's ${deg(24)}-way conflict`, v && v.includes('Not quite.') && v.includes(`${deg(24)}-way`), v);
v = await answerChoice('You double bK to 16', '20: four words of padding');
check(`q-bk16: 20 is wrong, with the model's ${deg(20)}-way conflict`, v && v.includes('Not quite.') && v.includes(`${deg(20)}-way`), v);
v = await answerChoice('You double bK to 16', '17.');
check(`q-bk16: 17 is correct (model: ${deg(17) === 1 ? 'conflict-free' : deg(17) + '-way'})`, v && v.includes('Correct.') && deg(17) === 1 && v.includes('conflict-free'), v);
const allOk = SCHEDULES.every((s) => [1, 2, 3, 4, 5, 6].every((n) => run({ threads: 4, tiles: n, variant: 'load-first' }, s).correct));
v = await answerChoice('two calls in the loop swapped', 'a fast thread\'s loads now come');
check('q-load-first: "overwrites the buffer mm reads" is wrong, with its reason', v && v.includes('Not quite.') && v.includes('different buffers'), v);
v = await answerChoice('two calls in the loop swapped', 'Yes: the loads fill the buffer');
check(`q-load-first: correct, and the model agrees (${allOk ? 'no wrong sum' : 'a wrong sum'})`, v && v.includes('Correct.') && allOk && v.includes('finds no wrong sum'), v);
v = await answerChoice('y = gelu(x Wᵀ + b)', 'Into writeTile');
check('q-epilogue: writeTile is correct, with a link to 15.9', v && v.includes('Correct.') && v.includes('15.9'), v);
v = await answerChoice('4 × 4 thread tiles', 'Faster: four blocks per SM');
check('q-triage: more warps is wrong, citing the busy pipe', v && v.includes('Not quite.') && v.includes('counter-productive'), v);
v = await answerChoice('4 × 4 thread tiles', 'No better, probably worse');
check(`q-triage: correct, with ${intensity(128, 128)} -> ${intensity(64, 64)} FLOP/B and 0.25 -> 0.5 reads`, v && v.includes('Correct.') && v.includes(`from ${intensity(128, 128)} to ${intensity(64, 64)} FLOP/B`) && v.includes('from 0.25 to 0.5'), v);
await shot('check', 'figure.code');

// ---- exercises, errata, shortened
const ex = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) => trs.map((tr) => ({ n: tr.children[0].textContent, links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('all three exercises are mapped, each to a section and the labs page',
  ex.length === 3 && ex.map((e) => e.n).join() === '1,2,3' && ex.every((e) => e.links.some((l) => /^#\/ch15\/15\.\d$/.test(l))) && ex.slice(0, 1).every((e) => e.links.includes('#/ch15/labs')), JSON.stringify(ex));
const errata = await page.$$eval('[data-list="errata"] li', (li) => li.map((x) => x.textContent));
check('the errata list names the missing start offset, the p. 365 slip, 16 × 16 vs 1D and Fig. 15.14',
  errata.length >= 10 && errata.some((e) => e.includes('&A_s[tRow*bK]')) && errata.some((e) => e.includes('p. 365')) && errata.some((e) => e.includes('16 × 16')) && errata.some((e) => e.includes('Fig. 15.14')), String(errata.length));
check('the dated statements are summarised with a Today note', text0.includes('WGMMA is Hopper-only') && text0.includes('CUDA Tile'));
check('what was shortened names warp specialisation and asynchronous copies', text0.includes('warp specialisation') && text0.includes('asynchronous copies'));

// ---- project status
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = (run?.results ?? []).filter((x) => x.lab === id && x.variant === variant && x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const m3 = await page.$eval('[data-m3]', (e) => e.dataset.m3);
const pass = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b] && labStatus(reports[b], 'ch15/lab6_linear', 'work') === 'pass');
check(`milestone M3 status is read from the reports (${pass.join(',') || 'none'})`, m3 === (pass.join(',') || 'none'), m3);

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch15/15.2', '#/ch15/15.3', '#/ch15/15.5', '#/ch15/15.6', '#/ch15/15.7', '#/ch15/15.8', '#/ch15/15.9', '#/ch15/labs', '#/ch16', '#/ch19/19.3', '#/ch19/19.4', '#/ch20', '#/x2/X2.5']);
check(`every link goes to a known route (${links.length})`, links.every((l) => known.has(l)), links.filter((l) => !known.has(l)).join(', '));
for (const target of ['ch16', 'ch20']) {
  await goto(page, target);
  const t = await page.$eval('.page', (e) => e.textContent.slice(0, 200));
  check(`the link to ${target} opens a page`, t.length > 20, t);
}

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch15/wrap');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'table[data-table="exercises"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
