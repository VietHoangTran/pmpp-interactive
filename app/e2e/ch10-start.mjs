// Browser check for chapter 10's "Before you start" page: the map of the chapter's
// seven kernels links to their sections, the notes are there, and every prerequisite
// question can be answered and reveals its explanation.
// Usage: node e2e/ch10-start.mjs [screenshot-prefix]
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
const question = (text) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt').textContent.includes(t)), text);
const qText = (q) => q.evaluate((e) => e.textContent.replace(/\s+/g, ' '));
async function numeric(text, value) {
  const q = await question(text);
  // Set the value as a user's edit would arrive (React listens for 'input').
  await (await q.$('input[type=number]')).evaluate((el, v) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return qText(q);
}
async function choose(text, option) {
  const q = await question(text);
  await q.evaluate((e, o) => [...e.querySelectorAll('.opts label')].find((l) => l.textContent.includes(o)).querySelector('input').click(), option);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return qText(q);
}

await goto(page, 'ch10/start');
await page.waitForSelector('table[data-part="kernel-map"]');

// ---- the map of the chapter
const rows = await page.$$eval('table[data-part="kernel-map"] tbody tr', (rs) => rs.map((r) => ({ link: r.querySelector('a').getAttribute('href'), cells: [...r.querySelectorAll('td')].map((c) => c.textContent.trim()) })));
check('the map lists seven kernels', rows.length === 7, String(rows.length));
check('each kernel links to its section, 10.3 to 10.10 in order', rows.map((r) => r.link).join(' ') === ['10.3', '10.4', '10.6', '10.7', '10.8', '10.9', '10.10'].map((s) => '#/ch10/' + s).join(' '), rows.map((r) => r.link).join(' '));
check('and names its figure: Figs. 10.5, 10.8, 10.10, 10.14, 10.16, 10.18, 10.20', rows.map((r) => r.cells[1].split(',')[0]).join(' ') === 'Fig. 10.5 Fig. 10.8 Fig. 10.10 Fig. 10.14 Fig. 10.16 Fig. 10.18 Fig. 10.20', rows.map((r) => r.cells[1]).join(' | '));
const body = await page.$eval('.prose', (e) => e.textContent);
check('the bridge to softmax and argmax is labelled as not in the book', body.includes('torch.softmax(logits / T, -1)') && body.includes('torch.argmax(logits, -1)') && (await page.$$eval('.note.beyond', (n) => n.some((x) => x.textContent.includes('does not mention language models')))));
check('the HIP and Metal differences carry a Today note with its facts items', await page.$$eval('.note.today', (n) => n.some((x) => x.textContent.includes('64-bit') && x.textContent.includes('simd_sum') && x.textContent.includes('A4, B1, C1'))));
check('the colour reminder is there', await page.$$eval('.note', (n) => n.some((x) => x.textContent.includes('Colour in this course'))));
check('the labs and milestone M4 are linked', body.includes('milestone M4') && (await page.$$eval('a', (as) => as.some((a) => a.getAttribute('href') === '#/ch10/labs'))));

// ---- the five questions, each labelled with its chapter
const labels = await page.$$eval('.q .kind', (ks) => ks.map((k) => k.textContent.trim()));
check('five questions, labelled with the chapter they come from', labels.join('|') === 'From chapter 4|From chapter 4|From chapter 4|From chapter 6|From chapter 9', labels.join('|'));

let t = await numeric('Thread 75', 75);
check('a wrong lane is marked and the answer revealed', t.includes('Not quite. The answer is 11.') && t.includes('warp 2, lane 11'), t.slice(0, 200));
t = await numeric('Thread 75', 11);
check('lane 11 is accepted, and the 64-wide case is explained', t.includes('Correct.') && t.includes('lane 11 of warp 1') && t.includes('ch. 4 §4.4'), t.slice(0, 300));
t = await numeric('How many warps diverge', 1);
check('one divergent warp is accepted, with the warp-by-warp reasoning', t.includes('Correct.') && t.includes('splits at 48'), t.slice(0, 300));
t = await choose('wait for each other at a barrier', 'may never be released');
check('the grid-barrier question accepts the scheduling answer', t.includes('Correct.') && t.includes('Section 10.9'), t.slice(0, 300));
t = await choose('wait for each other at a barrier', 'too slow');
check('the cost answer is marked wrong with its reason', t.includes('Not quite.') && t.includes('cost argument'), t.slice(0, 300));
t = await numeric('32-byte pieces', 8);
check('eight 32-byte transactions is accepted, against four for in[t]', t.includes('Correct.') && t.includes('four pieces') && t.includes('ch06-facts.md A2'), t.slice(0, 300));
t = await choose('without an atomic', 'The sum of some of the values');
check('the lost-update question accepts "some of the values" and points to 10.9', t.includes('Correct.') && t.includes('section 10.9'), t.slice(0, 300));
if (out) await page.screenshot({ path: out + '-questions.png', fullPage: true });

// ---- narrow screen
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
check('at a narrow width nothing scrolls sideways', await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
if (out) await page.screenshot({ path: out + '-narrow.png' });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
