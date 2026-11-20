// Browser check for chapter 12's "Before you start": the sampler pipeline, the map of the
// chapter, the six prerequisite questions (each answered, each explanation linking back to
// its chapter), and the table of where the labs run.
// Usage: APP_URL=http://localhost:5173 node e2e/ch12-start.mjs [--dark]
import { open, goto } from './browser.mjs';

const dark = process.argv.includes('--dark');
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
const text = (sel) => page.$eval(sel, (e) => e.textContent.replace(/\s+/g, ' '));

await goto(page, 'ch12/start');

// The pipeline and the two families.
const pipe = await text('table[data-part="pipeline"]');
check('pipeline names the four steps in order', /Row maximum.*Threshold.*Compaction.*Sampling/.test(pipe), pipe);
check('pipeline points to lab 10.3 and lab 12.4', pipe.includes('lab 10.3') && pipe.includes('lab 12.4'));
const famLinks = await page.$$eval('table[data-part="families"] a', (as) => as.map((a) => a.getAttribute('href')));
check('the map links every section from 12.2 to 12.8', ['12.2', '12.3', '12.4', '12.5', '12.6', '12.7', '12.8'].every((s) => famLinks.includes('#/ch12/' + s)), famLinks.join(' '));
const page_ = await text('.page');
check('a Today note says nonzero_static does not wait', page_.includes('nonzero_static') && page_.includes('does not wait'));
check('a Beyond note marks the sampler pipeline as ours', page_.includes('this pipeline is ours'));

// The six questions: answer each correctly, check the verdict and the link back.
const Q = [
  { has: 'Thread 100', kind: 'numeric', value: '36', link: '#/ch04/4.4' },
  { has: '100,000 threads', kind: 'choice', pick: 'Atomics on one address are serialised', link: '#/ch09/9.3' },
  { has: 'exclusive scan', kind: 'numeric', value: '3', link: '#/ch11/11.1' },
  { has: 'private copy', kind: 'choice', pick: 'Far fewer updates', link: '#/ch09/9.4' },
  { has: 'out[base + i]', kind: 'choice', pick: 'A: the 32 addresses', link: '#/ch06/6.1' },
  { has: '__shfl_sync(0xffffffff', kind: 'choice', pick: 'The mask names lanes', link: '#/ch10/10.7' },
];
const qs = await page.$$('.q');
check('six questions', qs.length === 6, String(qs.length));
for (const q of Q) {
  const h = await page.evaluateHandle((has) => [...document.querySelectorAll('.q')].find((e) => e.querySelector('.prompt').textContent.includes(has)), q.has);
  if (!(await h.evaluate((e) => !!e))) {
    check(`question "${q.has}" is on the page`, false);
    continue;
  }
  if (q.kind === 'numeric') {
    await (await h.$('input[type=number]')).type(q.value);
  } else {
    await h.evaluate((e, pick) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes(pick)).querySelector('input').click(), q.pick);
  }
  await (await h.$('button.primary')).click();
  await pause(120);
  const t = await h.evaluate((e) => e.textContent);
  check(`"${q.has}": the right answer is accepted`, t.includes('Correct.'), t.slice(0, 160));
  const links = await h.evaluate((e) => [...e.querySelectorAll('.verdict a')].map((a) => a.getAttribute('href')));
  check(`"${q.has}": the explanation links back to ${q.link}`, links.includes(q.link), links.join(' '));
}
// A wrong answer is marked wrong, with its reason.
const wrong = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('100,000 threads')));
await wrong.evaluate((e) => [...e.querySelectorAll('label')].find((l) => l.textContent.includes('Neither')).querySelector('input').click());
await (await wrong.$('button.primary')).click();
await pause(120);
const wt = await wrong.evaluate((e) => e.textContent);
check('a wrong choice is marked and explained', wt.includes('Not quite.') && wt.includes('queue behind it'), wt.slice(0, 200));

// Where the labs run.
const labs = await page.$$eval('table[data-part="labs"] tbody tr', (rs) => rs.map((r) => [...r.querySelectorAll('td')].map((d) => d.textContent.trim())));
check('five labs listed', labs.length === 5, JSON.stringify(labs));
check('labs 12.1 to 12.4 run on every backend', labs.slice(0, 4).every((r) => r[2] === 'emu, race, Metal, CUDA, HIP'), JSON.stringify(labs));
check('lab 12.5 is CUDA family only, with no Metal', labs[4][2].includes('CUDA family only') && !labs[4][2].includes('Metal'), labs[4]?.[2]);

// No horizontal page scroll at phone width.
await page.setViewport({ width: 390, height: 800, deviceScaleFactor: 1 });
await pause(300);
const sw = await page.evaluate(() => document.documentElement.scrollWidth);
check('no sideways page scroll at 390 px', sw <= 390, String(sw));

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
