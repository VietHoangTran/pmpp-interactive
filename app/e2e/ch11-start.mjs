// Browser check for chapter 11's start page: the cost ledger links every row to its
// section, the six prerequisite questions are there and answer correctly, the labs
// paragraph names all six labs and links the labs page, and nothing errs.
// Usage: node e2e/ch11-start.mjs [screenshot-prefix]
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
const question = (text) => page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(t)), text);
const answerChoice = async (text, option) => {
  const q = await question(text);
  await q.evaluate((e, o) => [...e.querySelectorAll('label')].find((l) => l.textContent.trim().startsWith(o)).querySelector('input').click(), option);
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};
const answerNumber = async (text, value) => {
  const q = await question(text);
  await (await q.$('input[type=number]')).type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.textContent);
};

await goto(page, 'ch11/start');
await page.waitForSelector('table[data-part="ledger"]');

// The ledger.
const rows = await page.$$eval('table[data-part="ledger"] tbody tr', (rs) =>
  rs.map((r) => ({ cells: [...r.querySelectorAll('td')].map((c) => c.textContent.trim()), links: [...r.querySelectorAll('a')].map((a) => a.getAttribute('href')) })),
);
check('the ledger has nine rows', rows.length === 9, String(rows.length));
check('every ledger row names book pages and links a chapter 11 section', rows.every((r) => /^pp?\. \d/.test(r.cells[2]) && r.links.length === 1 && r.links[0].startsWith('#/ch11/11.')), JSON.stringify(rows.map((r) => [r.cells[2], r.links])));
check('the barrier counts: 2 · log₂B, log₂B, 2', rows[0].cells[1] === '2 · log₂B' && rows[1].cells[1] === 'log₂B' && rows[2].cells[1] === '2');
check('Kogge-Stone work and traffic rows carry the book\'s formulas', rows[3].cells[1] === 'N · log₂N − (N − 1)' && rows[3].cells[2] === 'p. 266' && rows.slice(6).map((r) => parseInt(r.cells[1], 10)).join(' ') === '16 12 8');

// Today notes and the ours-label.
const notes = await page.$$eval('aside.note', (n) => n.map((e) => e.className + ': ' + e.textContent));
check('a Today note on torch.cumsum cites ch11-facts D4', notes.some((n) => n.includes('today') && n.includes('scan_innermost_dim') && n.includes('D4')));
check('the sampling bridge is labelled as not in the book', notes.some((n) => n.includes('beyond') && n.includes('Not in the book')));

// Prerequisite questions.
const labels = await page.$$eval('.q .kind', (k) => k.map((e) => e.textContent));
check('six prerequisite questions, labelled by chapter', labels.filter((l) => l.startsWith('From chapter')).length === 6, labels.join(' | '));
check('associativity question accepts "associative"', (await answerChoice('regroups the additions', 'associative')).includes('Correct.'));
check('the shuffle edge question accepts "Its own v"', (await answerChoice('lane 30 calls', 'Its own v')).includes('Correct.'));
check('the sector question accepts 16', (await answerNumber('4*lane', 16)).includes('Correct.'));
check('the block order question accepts "No"', (await answerChoice('block 7 spins', 'No:')).includes('Correct.'));
check('the false dependence question accepts "The one after the read"', (await answerChoice('A loop reads a neighbour', 'The one after the read')).includes('Correct.'));
check('the memory order question shows its explanation', (await answerChoice('memory_order_relaxed', 'No other access of the thread')).includes('release and acquire'));

// Labs.
const text = await page.$eval('.prose', (e) => e.textContent);
check('all six labs are named', [1, 2, 3, 4, 5, 6].every((n) => text.includes(`Lab 11.${n},`)));
check('the labs page is linked', await page.$$eval('a', (as) => as.some((a) => a.getAttribute('href') === '#/ch11/labs')));
check('chapter 10 is linked only by section id', await page.$$eval('a', (as) => ['10.2', '10.7', '10.8', '10.10'].every((id) => as.some((a) => a.getAttribute('href') === '#/ch10/' + id))));
if (out) await page.screenshot({ path: out + '-start.png', fullPage: true });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
