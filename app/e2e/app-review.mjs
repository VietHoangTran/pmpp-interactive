// Browser check for spaced review, the summary page, theme switching and every route.
// The progress file is faked in both directions, so the real one is untouched.
// Usage: node e2e/review.mjs
import { open, goto } from './browser.mjs';

const { browser, page, errors } = await open({ width: 1440, height: 1000 });
const day = (offset) => {
  const d = new Date();
  d.setDate(d.getDate() + offset);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
};
// Two questions due (one overdue), one not yet due.
const seeded = {
  version: 1,
  sections: { 'ch05/5.1': { visited: '2026-01-01T00:00:00.000Z', done: true } },
  answers: {},
  review: {
    'ch05/start/q-rowmajor': { box: 1, due: day(-2), seen: 2, lapses: 0 },
    'ch05/start/q-barrier': { box: 0, due: day(0), seen: 1, lapses: 1 },
    'ch05/5.1/q-regime': { box: 2, due: day(5), seen: 3, lapses: 0 },
  },
};
const puts = [];
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
  if (r.method() === 'PUT') puts.push(JSON.parse(r.postData()));
  return r.respond({ status: 200, contentType: 'application/json', body: r.method() === 'PUT' ? '{"ok":true}' : JSON.stringify(seeded) });
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const body = () => page.$eval('.page', (e) => e.textContent);

// ---- rail reflects progress ----
await goto(page, 'ch05/5.1');
check('rail shows how many reviews are due', (await page.$eval('.rail', (e) => e.textContent)).includes('2 due'));
check('rail ticks a finished section', (await page.$eval('.rail nav a[aria-current="page"]', (e) => e.textContent)).includes('✓'));

// ---- review session ----
await goto(page, 'review');
check('review starts with the most overdue question', (await body()).includes('Question 1 of 2') && (await body()).includes('By how many array elements'));
check('review does not prefill or reveal earlier answers', !(await body()).includes('Correct.') && (await page.$eval('.q input[type=number]', (e) => e.value)) === '');
await page.type('.q input[type=number]', '6');
await page.click('.q button.primary');
await pause(700);
check('a right answer moves the item up a box and pushes it out a week', puts.at(-1)?.review['ch05/start/q-rowmajor']?.box === 2 && puts.at(-1).review['ch05/start/q-rowmajor'].due === day(7), JSON.stringify(puts.at(-1)?.review['ch05/start/q-rowmajor']));
await page.evaluate(() => [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Next question').click());
await pause(150);
check('second due question appears', (await body()).includes('Question 2 of 2') && (await body()).includes('__syncthreads()'));
await (await page.$$('.q input[type=radio]'))[2].click();
await page.click('.q button.primary');
await pause(700);
const r2 = puts.at(-1)?.review['ch05/start/q-barrier'];
check('a wrong answer sends the item back to box 0, due tomorrow, and counts a lapse', r2?.box === 0 && r2.due === day(1) && r2.lapses === 2, JSON.stringify(r2));
check('the item not yet due was left alone', puts.at(-1).review['ch05/5.1/q-regime'].due === day(5));
await page.evaluate(() => [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Finish').click());
await pause(150);
check('the session ends cleanly', (await body()).includes('That was the last one due today.'));

// ---- summary page ----
await goto(page, 'ch05/wrap');
const q = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes('guaranteed to produce the transposed tiles')));
await (await q.$$('input[type=radio]'))[2].click();
await (await q.$('button.primary')).click();
await pause(200);
check('summary: the transpose defect question accepts "only B = 1"', (await q.evaluate((e) => e.textContent)).includes('Correct.'));
check('summary maps all twelve book exercises', (await body()).includes('The book\'s exercises') && (await page.$$('.page table.data tbody tr')).length >= 10);

// ---- theme ----
const bg = () => page.evaluate(() => getComputedStyle(document.body).backgroundColor);
const before = await bg();
await page.evaluate(() => [...document.querySelectorAll('.rail button')].find((b) => / theme$/.test(b.textContent)).click());
await pause(100);
check('theme button switches between light and dark', (await bg()) !== before);

// ---- every route renders without errors, at desktop and phone widths ----
// The routes come from the app's own chapter registry: every chapter that has sections.
// CHAPTERS=ch02,ch05 limits the sweep (useful while other chapters are being built).
const only = (process.env.CHAPTERS || '').split(',').filter(Boolean);
const built = await page.evaluate(async () => {
  const reg = await import('/src/chapters/registry.ts');
  const out = [];
  for (const id of reg.CHAPTER_IDS) {
    const ch = await reg.loadChapter(id);
    if (ch.sections.length) out.push([id, ch.sections.map((s) => s.id)]);
  }
  return out;
});
const routes = ['', 'review', ...built.filter(([id]) => !only.length || only.includes(id)).flatMap(([id, ss]) => ss.map((s) => `${id}/${s}`))];
console.log(`sweeping ${routes.length} routes in ${built.length} built chapters${only.length ? ` (limited to ${only.join(', ')})` : ''}`);
for (const width of [1440, 390]) {
  await page.setViewport({ width, height: 900, deviceScaleFactor: 1 });
  for (const route of routes) {
    try {
      await goto(page, route);
    } catch {
      const text = await page.$eval('.page', (e) => e.textContent.slice(0, 160)).catch(() => '');
      check(`route "${route}" at ${width}px: loads`, false, text);
      continue;
    }
    const info = await page.evaluate(() => ({
      placeholder: document.querySelector('.page').textContent.includes('This section is being written'),
      overflow: document.documentElement.scrollWidth - window.innerWidth,
      h1: document.querySelector('.page h1')?.textContent,
    }));
    check(`route "${route || 'home'}" at ${width}px: has content, no sideways page scroll`, !info.placeholder && !!info.h1 && info.overflow <= 1, JSON.stringify(info));
  }
}

check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
