// Browser check for chapter 14's start page: the section map, the notes, and all six
// prerequisite questions (one answered wrong first, then right).
// Usage: APP_URL=http://localhost:5173 node e2e/ch14-start.mjs [--dark] [screenshot-prefix]
import { open, goto } from './browser.mjs';

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
const question = (text) =>
  page.evaluateHandle((t) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(t)), text);
const answerNumeric = async (text, value) => {
  const q = await question(text);
  const input = await q.$('input[type=number]');
  await input.click();
  for (let k = 0; k < 6; k++) await page.keyboard.press('Backspace');
  await input.type(String(value));
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
};
const answerChoice = async (text, optionStart) => {
  const q = await question(text);
  const ok = await q.evaluate((e, s) => {
    const l = [...e.querySelectorAll('.opts label')].find((x) => x.textContent.trim().startsWith(s));
    if (!l) return false;
    l.querySelector('input').click();
    return true;
  }, optionStart);
  if (!ok) return 'option not found: ' + optionStart;
  await pause(60);
  await (await q.$('button.primary')).click();
  await pause(120);
  return q.evaluate((e) => e.querySelector('.verdict')?.textContent ?? '');
};

await goto(page, 'ch14/start');
await page.waitForSelector('.q');

// ---------------------------------------------------------------- the page
const rows = await page.$$eval('table[data-part="sections"] tbody tr', (rs) => rs.map((r) => r.querySelector('a')?.getAttribute('href')));
check('the section map lists 14.1 to 14.9, each linked', JSON.stringify(rows) === JSON.stringify(['14.1', '14.2', '14.3', '14.4', '14.5', '14.6', '14.7', '14.8', '14.9'].map((s) => '#/ch14/' + s)), JSON.stringify(rows));
const body = await page.$eval('.prose', (e) => e.textContent.replace(/\s+/g, ' '));
check('the bridge names torch.sort, ggml_argsort and the scan', body.includes('torch.sort') && body.includes('ggml_argsort') && body.includes('ggml_cumsum'));
check('top-k is called a selection, with torch.topk and cub::DeviceTopK', body.includes('selection') && body.includes('torch.topk') && body.includes('cub::DeviceTopK'));
check('the OneSweep note is there', body.includes('OneSweep') && body.includes('compute capability 6.0'));
check('the labs page is linked', (await page.$$('a[href="#/ch14/labs"]')).length >= 1);
check('two Today notes and one Not-in-the-book note', (await page.$$('aside.note.today')).length === 2 && (await page.$$('aside.note.beyond')).length === 1);
check('six questions', (await page.$$('.q')).length === 6);
if (out) await page.screenshot({ path: out + '-top.png' });

// ---------------------------------------------------------------- the questions
let v = await answerNumeric('exclusive scan (with +) of the bits', 4);
check('q-exscan: 4 is marked wrong with the answer 3', v.includes('Not quite. The answer is 3.'), v);
v = await answerNumeric('exclusive scan (with +) of the bits', 3);
check('q-exscan: 3 is correct, and the explanation points at 14.5', v.includes('Correct.') && v.includes('ones before'), v);
v = await answerNumeric('stable filter keeps the elements whose flags', 2);
check('q-filterpos: position 2', v.includes('Correct.') && v.includes('[0, 0, 1, 2, 2]'), v);
v = await answerNumeric('How many 32-byte sectors', 8);
check('q-stride-sectors: 8 sectors', v.includes('Correct.') && v.includes('bytes 0 to 251'), v);
v = await answerChoice('write the value 1 to the same flag', 'Not a race');
check('q-flagrace: "not a race" is wrong, with its reason', v.includes('Not quite.') && (await (await question('write the value 1 to the same flag')).evaluate((e) => e.textContent)).includes('Idempotent is not the same as race-free'), v);
v = await answerChoice('write the value 1 to the same flag', 'A data race');
check('q-flagrace: a data race, undefined behaviour', v.includes('Correct.') && v.includes('p. 333'), v);
v = await answerChoice('run a scan over the whole grid', 'Blocks need not all be resident');
check('q-gridscan: blocks are not guaranteed to be resident', v.includes('Correct.') && v.includes('section 14.5'), v);
v = await answerChoice('must write outputs', 'The co-rank pair (i, j)');
check('q-corank: the co-rank pair', v.includes('Correct.') && v.includes('section 14.3'), v);
const links = await page.$$eval('.q .verdict a', (as) => as.map((a) => a.getAttribute('href')));
for (const s of ['14.5', '14.4', '14.6', '14.2', '14.3'])
  check(`an explanation links to section ${s}`, links.includes('#/ch14/' + s), JSON.stringify(links));
if (out) await page.screenshot({ path: out + '-questions.png', fullPage: true });

// ---------------------------------------------------------------- phone width
await page.setViewport({ width: 390, height: 800, deviceScaleFactor: 1 });
await pause(300);
const sw = await page.evaluate(() => document.documentElement.scrollWidth);
check('no sideways page scroll at 390 px', sw <= 390, String(sw));
if (out) await page.screenshot({ path: out + '-narrow.png' });

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
