// Browser check for chapter 19, "Summary and exercises": what to carry forward, the five
// questions of the mixed check (numbers computed here from the same models the page
// imports), the snippet shown verbatim from labs/ch19/snippets, the exercises table, the
// errata list, the project status read from /api/reports, links, and the page at 390 px.
// Usage: APP_URL=http://localhost:<port> node e2e/ch19-wrap.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';
import { BOOK_SHAPE, gemmShape, xToB } from '../src/sim/unfold.ts';

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

await goto(page, 'ch19/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary and exercises" of chapter 19', h1.includes('Summary and exercises'), h1);
const h2 = await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent.trim()));
const wantH2 = ['What to carry forward', 'A mixed check', 'The book’s exercises', 'Where the PDF looks wrong', 'What this course shortened', 'The project so far'];
check('sections in order', JSON.stringify(h2) === JSON.stringify(wantH2), JSON.stringify(h2));
const text0 = norm(await page.$eval('.prose', (e) => e.textContent));
check('nothing on the page says it is being written', !/being written/i.test(text0));
check('no em dash used as an aside', !/ — /.test(text0));
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
const file = await page.evaluate(() => fetch('/api/lab-file?path=' + encodeURIComponent('ch19/snippets/wrap_backward_scatter.cu')).then((r) => (r.ok ? r.text() : 'MISSING')));
const code = await page.$eval('figure.code pre', (e) => e.textContent);
check('the scatter kernel is labs/ch19/snippets/wrap_backward_scatter.cu verbatim', file !== 'MISSING' && code.trim() === file.trim());
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
  return ok && (await h.evaluate((e) => (e.querySelector('.verdict')?.textContent ?? '') + ' | ' + (e.querySelector('.opts label.wrong .muted')?.textContent ?? '')));
};
check('five questions in the mixed check', (await page.$$('.q')).length === 5, String((await page.$$('.q')).length));

// LeNet C5: 16 maps of 5 x 5 in, 120 maps of 1 x 1 out, at batch 1 on 16 x 16 tiles.
// Our own count: one 1 x 1 map needs ceil(1 / 16) = 1 tile, so M blocks of 16 x 16 threads.
const g = { threads: 120 * Math.ceil(1 / 16) ** 2 * 256 };
const pct = Math.round(((100 * 120) / g.threads) * 100) / 100;
let v = await answerNumber('LeNet’s C5 turns 16 maps', 50);
check(`q-c5-direct: 50 is wrong, and the answer is the model's ${pct}`, v.includes('Not quite.') && v.includes(String(pct)), v);
v = await answerNumber('LeNet’s C5 turns 16 maps', pct);
check(`q-c5-direct: correct, with ${g.threads.toLocaleString('en-US')} threads, 0 printed blocks and the launchLinear reading`,
  v.includes('Correct.') && v.includes(g.threads.toLocaleString('en-US')) && v.includes('launches 0 blocks') && v.includes('launchLinear(X, F, Y, N, 400, 120)'), v);

v = await answerChoice('subsampling layer', 'About a factor of 4');
check('q-subsample: "a factor of 4" is wrong, with its reason', v && v.includes('Not quite.') && v.includes('no input is in two windows'), v);
v = await answerChoice('subsampling layer', 'Nothing: every input value is read by exactly one thread');
check('q-subsample: "nothing" is correct, with a link to 19.3', v && v.includes('Correct.') && v.includes('expansion ratio is 1'), v);

const centre = BOOK_SHAPE.M * xToB(BOOK_SHAPE, 0, 1, 1).length;
const corner = BOOK_SHAPE.M * xToB(BOOK_SHAPE, 0, 0, 0).length;
v = await answerChoice('first attempt at the input gradient', 'Yes: each thread reads only its own element');
check('q-backward: "independent threads" is wrong, with its reason', v && v.includes('Not quite.') && v.includes('which elements of dX'), v);
v = await answerChoice('first attempt at the input gradient', 'No: it needs a __syncthreads()');
check('q-backward: "a barrier" is wrong, with its reason', v && v.includes('Not quite.') && v.includes('different blocks'), v);
v = await answerChoice('first attempt at the input gradient', 'No: threads with overlapping patches');
check(`q-backward: correct, with the model's ${centre} writers of the centre pixel and ${corner} of a corner`,
  v && v.includes('Correct.') && v.includes(`so ${centre} threads read`) && v.includes(`a corner pixel has ${corner}`) && v.includes('atomicAdd') && v.includes('Eq. B.22'), v);

const per = gemmShape({ N: 1, C: 1, H: 14, W: 14, M: 1, K: 3 });
const dense = gemmShape({ N: 1, C: 256, H: 14, W: 14, M: 256, K: 3 });
const sh = (x) => `${x.rows} × ${x.inner} × ${x.cols}`;
v = await answerChoice('depthwise convolution', sh(dense));
check(`q-depthwise: the dense shape ${sh(dense)} is wrong`, v && v.includes('Not quite.') && v.includes('reads one'), v);
v = await answerChoice('depthwise convolution', `${sh(per)}: one row`);
check(`q-depthwise: ${sh(per)} is correct, against ${sh(dense)} dense`, v && v.includes('Correct.') && v.includes(sh(per)) && v.includes(sh(dense)), v);

v = await answerChoice('int8 with one float scale per output map', 'In the F tile load');
check(`q-int8-scale: the F tile load is correct but not cheapest (${dense.phases} phases)`, v && v.includes('Not quite.') && v.includes(`${dense.phases} phases`), v);
v = await answerChoice('int8 with one float scale per output map', 'In the X gather');
check('q-int8-scale: the X gather is wrong', v && v.includes('Not quite.') && v.includes('own scale'), v);
v = await answerChoice('int8 with one float scale per output map', 'Once per output');
check('q-int8-scale: once per output is correct, and the scale factors out of the sum', v && v.includes('Correct.') && v.includes('s·Σ f·b'), v);
await shot('check', 'figure.code');

// ---- exercises, errata, shortened
const ex = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) => trs.map((tr) => ({ n: tr.children[0].textContent, links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('all four exercises are mapped, each to a section',
  ex.length === 4 && ex.map((e) => e.n).join() === '1,2,3,4' && ex.every((e) => e.links.some((l) => /^#\/ch19\/19\.\d$/.test(l)) || e.n === '3'), JSON.stringify(ex));
const errata = await page.$$eval('[data-list="errata"] li', (li) => li.map((x) => x.textContent.replace(/\s+/g, ' ')));
check('the errata list has the three code errata, p. 468 and p. 469',
  errata.length >= 10 && errata.some((e) => e.includes('Fig. 19.5') && e.includes('528 of its 784')) && errata.some((e) => e.includes('Fig. 19.7')) &&
  errata.some((e) => e.includes('Fig. 19.11') && e.includes('drops 9 terms')) && errata.some((e) => e.includes('p. 468') && e.includes('14, 20 / 12, 24')) &&
  errata.some((e) => e.includes('u = 10 and v = 2')), String(errata.length));
check('what was shortened names the backward path and tiling the direct kernel', text0.includes('the backward path') && text0.includes('shared-memory tiling of the direct kernel'));

// ---- project status
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = (run?.results ?? []).filter((x) => x.lab === id && x.variant === variant && x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const pass = (id) => ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b] && labStatus(reports[b], id, 'work') === 'pass').join(',') || 'none';
const lab19 = await page.$eval('[data-lab19]', (e) => e.dataset.lab19);
const m2 = await page.$eval('[data-m2]', (e) => e.dataset.m2);
check(`lab 19.1 status is read from the reports (${pass('ch19/lab1_convgemm')})`, lab19 === pass('ch19/lab1_convgemm'), lab19);
check(`M2 status is read from the reports (${pass('ch06/lab4_linear')})`, m2 === pass('ch06/lab4_linear'), m2);
check('the project section says no milestone and names launchLinear', text0.includes('This chapter adds no milestone') && text0.includes('launchLinear(X, W, Y, batch, in, out)'));

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch19/19.2', '#/ch19/19.3', '#/ch19/19.4', '#/ch19/labs', '#/ch07/7.5', '#/ch20']);
check(`every link goes to a known route (${links.length})`, links.every((l) => known.has(l)), links.filter((l) => !known.has(l)).join(', '));
await goto(page, 'ch20');
const t20 = await page.$eval('.page', (e) => e.textContent.slice(0, 200));
check('the link to ch20 opens a page', t20.length > 20, t20);

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch19/wrap');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'table[data-table="exercises"]');

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
