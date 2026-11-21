// Browser check for chapter 20, "Summary and exercises": what to carry forward, the five
// questions of the mixed check (expected numbers worked out here by hand, independently of
// the models the page imports), the exercises table, the errata list, the project status
// read from /api/reports, links, and the page at 390 px.
// Usage: APP_URL=http://localhost:<port> node e2e/ch20-wrap.mjs [screenshot-prefix] [--dark]
import { open, goto } from './browser.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const out = args[0];
const dark = process.argv.includes('--dark');
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
const EMPTY = { version: 1, sections: {}, answers: {}, review: {} };
const puts = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
  if (r.method() === 'PUT') puts.push(r.postData() ?? '');
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

await goto(page, 'ch20/wrap');
await page.evaluate((t) => { document.documentElement.dataset.theme = t; }, dark ? 'dark' : 'light');
const reports = await page.evaluate(() => fetch('/api/reports').then((r) => r.json()));

// ---- structure
const h1 = await page.$eval('.page h1', (e) => e.textContent);
check('the page is "Summary and exercises" of chapter 20', h1.includes('Summary and exercises'), h1);
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
check('the page shows no code of its own', (await page.$$('figure.code')).length === 0);

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
  return norm(await h.evaluate((e) => e.querySelector('.verdict')?.textContent ?? ''));
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
  return ok && norm(await h.evaluate((e) => (e.querySelector('.verdict')?.textContent ?? '') + ' | ' + (e.querySelector('.opts label.wrong .muted')?.textContent ?? '')));
};
const r3 = (v) => Math.round(v * 1000) / 1000;

// Eq. 20.4 by hand: block A (m 2, D 3) and block B (m 5, D 1.5).
const D = 3 * Math.exp(2 - 5) + 1.5;
let v = await answerNumber('Flash-Decoding splits one head', 4.5);
check(`q-split-merge: 4.5 (adding without rescaling) is wrong, and the answer ${r3(D)} is given`, v.includes('Not quite.') && v.includes(String(r3(D))), v);
v = await answerNumber('Flash-Decoding splits one head', r3(D));
check(`q-split-merge: ${r3(D)} is correct, with the factor ${r3(Math.exp(-3))} and logsumexp ${r3(5 + Math.log(r3(D)))}`,
  v.includes('Correct.') && v.includes(String(r3(Math.exp(-3)))) && v.includes(String(r3(5 + Math.log(D)))), v);

// Lower-right mask by hand: row r sees c <= r + (nk - nq).
const first = 0 + (1000 - 4) + 1;
v = await answerNumber('Speculative decoding verifies 4', 1);
check(`q-spec-mask: 1 (upper-left alignment) is wrong; ${first} is given`, v.includes('Not quite.') && v.includes(String(first)), v);
v = await answerNumber('Speculative decoding verifies 4', first);
check(`q-spec-mask: ${first} is correct, and the last query sees all 1000`, v.includes('Correct.') && v.includes('sees all 1000'), v);

v = await answerChoice('128,000 vocabulary logits', 'The lower-right mask hides');
check('q-vocab: the mask answer is wrong, because nq = 1 sees every column', v && v.includes('Not quite.') && v.includes('sees every column'), v);
v = await answerChoice('128,000 vocabulary logits', 'The grid is one block');
check('q-vocab: one block is correct, and the cure is the Eq. 20.4 merge', v && v.includes('Correct.') && v.includes('Eq. 20.4') && v.includes('500 columns each'), v);

// Tile bytes by hand: K^T = B_c·128 + B_c·128/32, V = B_c·128, S = 32·B_c floats.
const tiles = (bc) => ({ kt: bc * 128 + (bc * 128) / 32, v: bc * 128, s: 32 * bc });
const dbl = (bc) => { const t = tiles(bc); return 4 * (2 * t.kt + 2 * t.v + t.s); };
const sgl = (bc) => { const t = tiles(bc); return 4 * (t.kt + t.v + t.s); };
v = await answerChoice('double-buffers lab 20.2', 'Wrong results: a fast warp');
check('q-double-buffer: "a fast warp overwrites" is wrong, with the barrier argument', v && v.includes('Not quite.') && v.includes('iteration j − 1'), v);
v = await answerChoice('double-buffers lab 20.2', 'Correct, but the tiles outgrow');
check(`q-double-buffer: correct, with ${sgl(16)} -> ${dbl(16)} B at B_c = 16 and ${dbl(8)} B at 8`,
  v && v.includes('Correct.') && [sgl(16), dbl(16), 32768, dbl(8)].every((n) => v.includes(n.toLocaleString('en-US'))), v);

// Eq. 20.12 by hand: 2 · 32 layers · 8 K/V heads · 128 values, 34 bytes per 32.
const q8 = (2 * 32 * 8 * 128 * 34) / 32;
v = await answerNumber('llama.cpp stores its cache as q8_0', 64);
check(`q-q8-cache: 64 (forgetting the scales) is wrong; ${q8 / 1024} KiB is given`, v.includes('Not quite.') && v.includes(String(q8 / 1024)), v);
v = await answerNumber('llama.cpp stores its cache as q8_0', q8 / 1024);
check(`q-q8-cache: ${q8 / 1024} KiB is correct, ${q8.toLocaleString('en-US')} bytes`, v.includes('Correct.') && v.includes(q8.toLocaleString('en-US')), v);
await shot('check', '.q');

// ---- exercises, errata, shortened
const ex = await page.$$eval('table[data-table="exercises"] tbody tr', (trs) => trs.map((tr) => ({ n: tr.children[0].textContent, links: [...tr.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('all four exercises are mapped, each to a page of this chapter',
  ex.length === 4 && ex.map((e) => e.n).join() === '1,2,3,4' && ex.every((e) => e.links.some((l) => /^#\/ch20\//.test(l))) && ex[0].links.includes('#/ch20/labs'), JSON.stringify(ex));
const errata = (await page.$$eval('[data-list="errata"] li', (li) => li.map((x) => x.textContent))).map(norm);
const need = ['gridDim.y', 'NEG_INFINITY', '__syncwarp()', 'multiple of both', 'logsumexp', 'p. 495', 'p. 502', 'load_KTV()', 'Eq. 20.15', 'query heads per K/V head', 'S x V', 'all zero elements', 'marker 4'];
check(`the errata list covers every item of the notes' "Unclear in the PDF" (${need.length} phrases)`, need.every((n) => errata.some((e) => e.includes(n))), need.filter((n) => !errata.some((e) => e.includes(n))).join(', '));
check('what was shortened names the backward pass and mixture of experts', text0.includes('backward pass') && text0.includes('mixture of experts'));

// ---- project status
const labStatus = (run, id, variant) => {
  if (variant === 'work' && run?.work_state?.[id] === 'untouched') return 'none';
  const rs = (run?.results ?? []).filter((x) => x.lab === id && x.variant === variant && x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
};
const bs = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
const passes = (id) => bs.filter((b) => labStatus(reports[b], id, 'work') === 'pass');
const m6 = passes('ch20/lab1_causal_softmax').filter((b) => passes('ch20/lab2_flash').includes(b));
const m7 = passes('ch20/lab4_decode');
const ms = await page.$eval('[data-milestones]', (e) => e.dataset.milestones);
check(`milestones M6 (${m6.join(',') || 'none'}) and M7 (${m7.join(',') || 'none'}) are read from the reports`, ms === `m6:${m6.join(',') || 'none'};m7:${m7.join(',') || 'none'}`, ms);

// ---- links
const links = await page.$$eval('.prose a[href^="#/"]', (as) => [...new Set(as.map((a) => a.getAttribute('href')))]);
const known = new Set(['#/ch20/start', '#/ch20/20.1', '#/ch20/20.3', '#/ch20/20.4', '#/ch20/20.5', '#/ch20/20.6', '#/ch20/20.7', '#/ch20/labs', '#/ch21', '#/ch22', '#/x1', '#/ch18/18.6', '#/x2/X2.7', '#/x3/X3.7']);
check(`every link goes to a known route (${links.length})`, links.every((l) => known.has(l)), links.filter((l) => !known.has(l)).join(', '));
for (const target of ['ch21', 'ch22', 'x1']) {
  await goto(page, target);
  const t = await page.$eval('.page', (e) => e.textContent.slice(0, 200));
  check(`the link to ${target} opens a page`, t.length > 20, t);
}

// ---- phone width
await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
await goto(page, 'ch20/wrap');
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
check('at 390 px the page does not scroll sideways', overflow <= 1, String(overflow));
await shot('phone', 'table[data-table="exercises"]');

await pause(600);
check('answers were saved through PUT /api/progress', ['q-split-merge', 'q-spec-mask', 'q-vocab', 'q-double-buffer', 'q-q8-cache'].every((id) => puts.some((p) => p.includes('ch20/wrap/' + id))));
check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
