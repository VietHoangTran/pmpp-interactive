// Browser check for chapter 9's "Before you start" page: the MLX bridge question comes
// first, the roadmap links to 9.2 to 9.6, the notes are labelled, and every
// prerequisite question can be answered and reveals its explanation.
// Usage: APP_URL=http://localhost:5173 node e2e/ch09-start.mjs [screenshot-prefix]
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
async function choose(text, option) {
  const q = await question(text);
  await q.evaluate((e, o) => [...e.querySelectorAll('.opts label')].find((l) => l.textContent.includes(o)).querySelector('input').click(), option);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return qText(q);
}

await goto(page, 'ch09/start');
await page.waitForSelector('table[data-part="roadmap"]');

// ---- the opener comes before any explanation
const order = await page.evaluate(() => {
  const prose = document.querySelector('.prose');
  const kids = [...prose.children];
  const q = kids.findIndex((k) => k.classList.contains('q'));
  const firstH2 = kids.findIndex((k) => k.tagName === 'H2');
  return { q, firstH2, label: kids[q]?.querySelector('.kind')?.textContent, text: kids[q]?.textContent ?? '' };
});
check('the MLX question opens the page, before the first heading', order.q >= 0 && order.q < order.firstH2 && order.q <= 2, JSON.stringify(order));
check('it shows the four lines of MLX', order.text.includes('a = mx.array([0, 0])') && order.text.includes('idx = mx.array([0, 1, 0, 1])') && order.text.includes('a[idx] += 1'));

let t = await choose('What does a hold', '[2, 2]');
check('[2, 2] is marked wrong and pointed at .at[idx].add', t.includes('Not quite.') && t.includes('a.at[idx].add(1) gives'), t.slice(0, 300));
t = await choose('What does a hold', '[1, 1]');
check('[1, 1] is accepted with the gather, add, scatter explanation', t.includes('Correct.') && t.includes('a gather, an add and a scatter') && t.includes('index_put_') && t.includes('D5'), t.slice(0, 400));

// ---- roadmap and notes
const links = await page.$$eval('table[data-part="roadmap"] tbody a', (as) => as.map((a) => a.getAttribute('href')));
check('the roadmap links 9.2 to 9.6 in order', links.join(' ') === ['9.2', '9.3', '9.4', '9.5', '9.6'].map((s) => '#/ch09/' + s).join(' '), links.join(' '));
const body = await page.$eval('.prose', (e) => e.textContent.replace(/\s+/g, ' '));
check('owner-computes and output interference are named', body.includes('owner-computes') && body.includes('output interference'));
check('every lab measures uniform and skewed inputs', body.includes('uniform bytes and a skewed image'));
check('the LLM mapping names M5, lab 9.3, bucketed top-k and chapter 10; the llama.cpp details live on the summary page', body.includes('milestone M5') && body.includes('lab 9.3') && body.includes('Bucketed top-k') && body.includes('Fig. 10.18') && !body.includes('llama.cpp\'s Metal top-k'));
check('the prerequisite heading does not repeat the page title', (await page.$$eval('.prose h2', (hs) => hs.map((h) => h.textContent))).includes('Check what this chapter assumes'));
check('the bridge is labelled as not in the book', await page.$$eval('.note.beyond', (n) => n.some((x) => x.textContent.includes('makes no link to language models'))));
check('the MLX facts carry a Today note with D5', await page.$$eval('.note.today', (n) => n.some((x) => x.textContent.includes('D5') && x.textContent.includes('2026-11-20'))));
check('the race backend paragraph names the command', body.includes('python3 labs/pmpp.py run 9.1 --backend race') && body.includes('ThreadSanitizer'));
check('the labs page is linked', await page.$$eval('a', (as) => as.some((a) => a.getAttribute('href') === '#/ch09/labs')));

// ---- the five prerequisite questions
const labels = await page.$$eval('.q .kind', (ks) => ks.map((k) => k.textContent.trim()));
check('six questions: the opener, then five labelled by chapter', labels.join('|') === 'Predict|From chapter 4|From chapter 5|From chapters 4 and 6|From chapter 6|From chapter 6', labels.join('|'));
t = await choose('can block 1 rely', 'block 0 has the lower index');
check('barrier scope: "blocks run in order" is marked wrong', t.includes('Not quite.') && t.includes('No order between blocks'), t.slice(0, 300));
t = await choose('can block 1 rely', 'covers one block');
check('barrier scope: the right answer links to 4.3', t.includes('Correct.') && t.includes('atomics, which need no order') && (await (await question('can block 1 rely')).evaluate((e) => !!e.querySelector('a[href="#/ch04/4.3"]'))), t.slice(0, 300));
t = await choose('What did the counts start at', 'Zero');
check('shared init: "zero" is wrong and mentions the 0xCD fill', t.includes('Not quite.') && t.includes('0xCD'), t.slice(0, 300));
t = await choose('What did the counts start at', 'nobody initialised');
check('shared init: accepted, with the barrier after clearing', t.includes('Correct.') && t.includes('barrier after the clearing'), t.slice(0, 300));
t = await choose('peak bandwidth. Why?', 'independent requests in flight');
check('latency: accepted, links to 4.6 and 6.2', t.includes('Correct.') && (await (await question('peak bandwidth. Why?')).evaluate((e) => !!e.querySelector('a[href="#/ch04/4.6"]') && !!e.querySelector('a[href="#/ch06/6.2"]'))), t.slice(0, 300));
t = await choose('fall on consecutive bytes', 'k*blockDim.x*gridDim.x');
check('coalescing: the interleaved index is accepted', t.includes('Correct.') && t.includes('Section 9.5'), t.slice(0, 300));
t = await choose('thread coarsening) make a kernel faster', 'less scheduling work');
check('coarsening: "helps whatever the kernel does" is wrong', t.includes('Not quite.') && t.includes('too few blocks'), t.slice(0, 300));
t = await choose('thread coarsening) make a kernel faster', 'paid for blocks the hardware would have run');
check('coarsening: accepted and tied to 9.4\'s private copies', t.includes('Correct.') && t.includes('cleared and merged once per block'), t.slice(0, 300));
if (out) await page.screenshot({ path: out + '-start.png', fullPage: true });

// ---- narrow screen: nothing scrolls the page sideways
await page.setViewport({ width: 400, height: 900, deviceScaleFactor: 1 });
await pause(300);
const wide = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
check('no sideways scrolling at 400 px', wide <= 1, String(wide));
if (out) await page.screenshot({ path: out + '-start-narrow.png', fullPage: true });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
